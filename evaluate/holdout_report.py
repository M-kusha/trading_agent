#!/usr/bin/env python3
"""Score a trained model on a chronological development window and baselines.

A profit factor means nothing on its own here. XAUUSD rose 149.8% across this
dataset, so "made money" is the null hypothesis rather than the finding - a
leveraged buy-and-hold would clear most bars a strategy is usually judged by.
The only useful question is whether the policy beats the alternatives on data
excluded from fitting.  Once this report influences engineering, however, the
window is development validation—not a sealed final test.

Four things are measured:

1. Does it beat doing nothing?      -> always-flat, random-valid
2. Does it beat the market?         -> buy-and-hold, MA-cross
3. Does it work short?              -> the same holdout, price-mirrored, where
                                       the drift is inverted
4. Would it pass the prop firm?     -> 10% target, 5% daily DD, 10% max DD

Usage:
    python evaluate/holdout_report.py --model models/curriculum/curriculum_ppo_final.zip
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.core.env_types import PropFirmConfig  # noqa: E402
from envs.prop_firm_env import PropFirmTradingEnv  # noqa: E402
from modules.meta.ppo_observation_builder import PPO_OBS_SIZE, PPO_OBS_VERSION  # noqa: E402
from train.train_prop_firm import (  # noqa: E402
    _primary_frame,
    build_dataset_manifest,
    load_market_data,
    split_data_by_time,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("holdout_report")

FTMO_PROFIT_TARGET = 0.10
FTMO_DAILY_DD = 0.05
FTMO_MAX_DD = 0.10


@dataclass
class Result:
    name: str
    episodes: int = 0
    trades: int = 0
    total_pnl: float = 0.0
    return_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    expectancy_r: float = 0.0
    reward_risk: float = 0.0
    max_drawdown_pct: float = 0.0
    worst_daily_dd_pct: float = 0.0
    trades_per_day: float = 0.0
    median_bars_held: float = 0.0
    long_pnl: float = 0.0
    short_pnl: float = 0.0
    return_ci_low_pct: Optional[float] = None
    requested_episodes: int = 0
    independent_windows: int = 0
    overlapping_windows: bool = False
    episode_start_indices: List[int] = field(default_factory=list)
    episode_records: List[Dict[str, Any]] = field(default_factory=list)
    r_multiples: List[float] = field(default_factory=list)
    # An aggregate number hides the thing that matters most: a strategy can
    # look profitable overall while losing badly in exactly the regime it now
    # has to trade.
    by_vol: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def challenge_thresholds_met(self) -> bool:
        return (
            self.return_pct >= FTMO_PROFIT_TARGET * 100.0
            and self.max_drawdown_pct <= FTMO_MAX_DD * 100.0
            and self.worst_daily_dd_pct <= FTMO_DAILY_DD * 100.0
        )

    @property
    def challenge_threshold_diagnostic(self) -> str:
        # Historical/development paths cannot pass a broker challenge.  This
        # label only states whether three numerical thresholds were crossed;
        # it deliberately cannot be consumed as promotion evidence.
        if self.challenge_thresholds_met:
            return "THRESHOLDS_MET_DIAGNOSTIC_ONLY"
        reasons = []
        if self.return_pct < FTMO_PROFIT_TARGET * 100.0:
            reasons.append(f"profit {self.return_pct:.1f}% < 10%")
        if self.max_drawdown_pct > FTMO_MAX_DD * 100.0:
            reasons.append(f"maxDD {self.max_drawdown_pct:.1f}% > 10%")
        if self.worst_daily_dd_pct > FTMO_DAILY_DD * 100.0:
            reasons.append(f"dailyDD {self.worst_daily_dd_pct:.1f}% > 5%")
        return "THRESHOLDS_NOT_MET: " + ", ".join(reasons)


# ── policies ────────────────────────────────────────────────────────────────

class Policy:
    name = "policy"

    def reset(self) -> None:
        pass

    def act(self, env: PropFirmTradingEnv, obs: np.ndarray, mask: np.ndarray) -> int:
        raise NotImplementedError

    @staticmethod
    def _first_valid(mask: np.ndarray, candidates: List[int]) -> Optional[int]:
        for a in candidates:
            if 0 <= a < len(mask) and mask[a]:
                return a
        return None


class AlwaysFlat(Policy):
    """The floor. A strategy that cannot beat this has negative value."""
    name = "always-flat"

    def act(self, env, obs, mask):
        return env._ACTION_HOLD


class RandomValid(Policy):
    """Uniform over legal actions - the no-skill control."""
    name = "random-valid"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, mask):
        valid = np.flatnonzero(mask)
        return int(self.rng.choice(valid)) if valid.size else env._ACTION_HOLD


class BuyAndHold(Policy):
    """Enter long once per episode and never deliberately re-enter."""
    name = "one-shot-long"

    def reset(self) -> None:
        self.entered = False

    def act(self, env, obs, mask):
        if not getattr(self, "entered", False) and env.position is None and env.pending_entry is None:
            a = self._first_valid(mask, [env._ACTION_LONG_START + env._K - 1, env._ACTION_LONG_START])
            if a is not None:
                self.entered = True
                return a
        return env._ACTION_HOLD


class MACross(Policy):
    """Long above the slow MA, short below. A real but naive strategy."""
    name = "ma-cross"

    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast, self.slow = fast, slow

    def act(self, env, obs, mask):
        o = env._get_ohlcv(env._episode_instrument, lookback=self.slow + 2)
        close = o.get("close") if o else None
        if close is None or len(close) < self.slow + 1:
            return env._ACTION_HOLD

        fast = float(np.mean(close[-self.fast:]))
        slow = float(np.mean(close[-self.slow:]))
        want_long = fast > slow

        if env.position is not None:
            aligned = (env.position.direction == "long") == want_long
            if not aligned and mask[env._ACTION_CLOSE]:
                return env._ACTION_CLOSE
            return env._ACTION_HOLD

        start = env._ACTION_LONG_START if want_long else env._ACTION_SHORT_START
        a = self._first_valid(mask, [start + 1, start])
        return a if a is not None else env._ACTION_HOLD


class TrainedModel(Policy):
    name = "trained-model"

    def __init__(self, model, deterministic: bool = True):
        self.model = model
        self.deterministic = deterministic

    def act(self, env, obs, mask):
        action, _ = self.model.predict(
            obs[None, :], action_masks=mask[None, :], deterministic=self.deterministic
        )
        a = int(np.asarray(action).ravel()[0])
        if a < 0 or a >= len(mask) or not bool(mask[a]):
            raise RuntimeError(f"masked model returned illegal action {a}")
        return a


# ── evaluation ──────────────────────────────────────────────────────────────

def run_policy(
    policy: Policy,
    data: Dict[str, Dict[str, Any]],
    cfg: PropFirmConfig,
    episodes: int,
    seed: int,
    max_steps: int,
    episode_starts: Optional[Sequence[int]] = None,
) -> Result:
    run_cfg = copy.deepcopy(cfg)
    run_cfg.max_steps_per_episode = int(max_steps)
    env = PropFirmTradingEnv(data, run_cfg)
    res = Result(name=policy.name, requested_episodes=int(episodes))

    if episode_starts is None:
        episode_starts = build_non_overlapping_episode_starts(
            env,
            requested_episodes=episodes,
            max_steps=max_steps,
        )
    starts = [int(value) for value in episode_starts]
    if not starts:
        raise ValueError("evaluation has no legal non-overlapping episode window")
    if len(starts) > int(episodes):
        starts = starts[: int(episodes)]
    if starts != sorted(starts) or any(
        later - earlier <= int(max_steps)
        for earlier, later in zip(starts, starts[1:])
    ):
        raise ValueError("evaluation episode starts must be chronological and non-overlapping")
    if len(starts) < int(episodes):
        logger.warning(
            "%s: requested %d episodes but only %d independent %d-bar windows exist; "
            "running the honest independent set without manufacturing a confidence interval",
            policy.name,
            episodes,
            len(starts),
            max_steps,
        )
    res.independent_windows = len(starts)
    res.episode_start_indices = list(starts)

    bars_held: List[float] = []
    daily_dds: List[float] = []
    trade_pnls: List[float] = []
    episode_pnls: List[float] = []
    episode_returns: List[float] = []
    episode_max_dds: List[float] = []
    total_bars = 0

    for ep, scheduled_start in enumerate(starts):
        def _scheduled_start(low: int, high: int, *, target: int = scheduled_start) -> int:
            if not int(low) <= target < int(high):
                raise ValueError(
                    f"scheduled evaluation start {target} is outside legal range [{low}, {high})"
                )
            return target

        env._sample_episode_start = _scheduled_start  # type: ignore[method-assign]
        obs, _ = env.reset(seed=seed + ep)
        if int(env.current_step) != scheduled_start:
            raise RuntimeError(
                f"evaluation start contract failed: requested {scheduled_start}, "
                f"environment selected {env.current_step}"
            )
        policy.reset()
        peak = float(env.equity)
        episode_equity = [float(env.equity)]
        start_time = env._get_bar_dt(env._episode_instrument)
        worst_daily = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            mask = env.action_masks()
            action = policy.act(env, obs, mask)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
            steps += 1

            peak = max(peak, float(env.equity))
            episode_equity.append(float(env.equity))
            # Daily DD is not reported in episode_stats, so sample the env's
            # own calculation - it is a hard prop-firm limit and must be
            # measured, not inferred from the equity curve.
            _cur, daily = env._calc_dds()
            worst_daily = max(worst_daily, float(daily))

        total_bars += steps
        res.episodes += 1
        end_time = env._get_bar_dt(env._episode_instrument)
        episode_pnl = float(env.equity) - float(cfg.initial_balance)
        episode_return = episode_pnl / max(float(cfg.initial_balance), 1.0) * 100.0
        eq = np.asarray(episode_equity, dtype=float)
        running_peak = np.maximum.accumulate(eq)
        episode_max_dd = float(np.max((running_peak - eq) / np.maximum(running_peak, 1.0)) * 100.0)
        episode_pnls.append(episode_pnl)
        episode_returns.append(episode_return)
        episode_max_dds.append(episode_max_dd)

        # Read the env directly rather than from info: episode_stats is only
        # attached to info on termination, and max_steps cuts most episodes
        # short, so relying on info reported zero trades for every policy while
        # drawdown showed positions had clearly been taken.
        stats: Dict[str, Any] = {}
        if hasattr(env, "get_episode_stats"):
            stats = env.get_episode_stats() or {}
        if not stats and isinstance(info, dict):
            stats = info.get("episode_stats", {}) or {}
        for tr in stats.get("trades_with_regime", []) or []:
            res.trades += 1
            trade_pnls.append(float(tr.get("pnl", 0.0)))
            res.r_multiples.append(float(tr.get("r_multiple", 0.0)))
            bars_held.append(float(tr.get("bars_held", 0)))
            res.by_vol.setdefault(str(tr.get("volatility_regime", "unknown")), []).append(
                float(tr.get("pnl", 0.0))
            )

        ds = stats.get("direction_stats", {}) or {}
        res.long_pnl += float(ds.get("long_pnl", 0.0))
        res.short_pnl += float(ds.get("short_pnl", 0.0))
        daily_dds.append(worst_daily)
        res.episode_records.append({
            "episode": ep,
            "seed": seed + ep,
            "start_time": start_time.isoformat() if start_time is not None else None,
            "end_time": end_time.isoformat() if end_time is not None else None,
            "steps": steps,
            "trades": int(stats.get("trade_count", len(stats.get("trades_with_regime", []) or []))),
            "pnl": episode_pnl,
            "return_pct": episode_return,
            "max_drawdown_pct": episode_max_dd,
            "worst_daily_dd_pct": worst_daily * 100.0,
        })

    res.total_pnl = float(np.mean(episode_pnls)) if episode_pnls else 0.0
    res.return_pct = float(np.mean(episode_returns)) if episode_returns else 0.0
    if len(episode_returns) >= 2:
        ep_returns = np.asarray(episode_returns, dtype=float)
        res.return_ci_low_pct = float(
            ep_returns.mean() - 1.96 * ep_returns.std(ddof=1) / np.sqrt(ep_returns.size)
        )

    arr = np.asarray(trade_pnls, dtype=float)
    if arr.size:
        wins, losses = arr[arr > 0], arr[arr <= 0]
        res.win_rate = float((arr > 0).mean() * 100.0)
        gross_win = float(wins.sum())
        gross_loss = float(abs(losses.sum()))
        res.profit_factor = gross_win / gross_loss if gross_loss > 1e-9 else float("inf")

    r = np.asarray(res.r_multiples, dtype=float)
    if r.size:
        res.expectancy_r = float(r.mean())
        w, losers = r[r > 0], r[r <= 0]
        if w.size and losers.size and abs(losers.mean()) > 1e-9:
            res.reward_risk = float(w.mean() / abs(losers.mean()))

    if bars_held:
        res.median_bars_held = float(np.median(bars_held))
    if episode_max_dds:
        res.max_drawdown_pct = float(max(episode_max_dds))
    if daily_dds:
        res.worst_daily_dd_pct = float(max(daily_dds) * 100.0)
    if total_bars:
        res.trades_per_day = res.trades / (total_bars / 96.0)

    env.close()
    return res


def build_non_overlapping_episode_starts(
    env: PropFirmTradingEnv,
    *,
    requested_episodes: int,
    max_steps: int,
) -> List[int]:
    """Return paired chronological blocks without pseudo-replication.

    Sampling a dozen 3,000-bar episodes from a 3,218-bar post-cut window creates
    a dozen highly overlapping paths and a meaningless confidence interval.
    This schedule uses disjoint blocks and reports fewer episodes when the data
    cannot support the request.
    """
    requested = int(requested_episodes)
    block = int(max_steps)
    if requested <= 0 or block <= 0:
        raise ValueError("requested_episodes and max_steps must be positive")
    low, high = env._episode_sampling_bounds(
        env._episode_start_buffer(),
        max(0, int(env._exec_latency())),
    )
    candidates = np.arange(int(low), int(high), block + 1, dtype=np.int64)
    if candidates.size == 0:
        return []
    if candidates.size <= requested:
        return [int(value) for value in candidates]
    selected = np.linspace(0, candidates.size - 1, num=requested, dtype=np.int64)
    return [int(candidates[index]) for index in selected]


def build_policies(model) -> List[Policy]:
    policies: List[Policy] = [AlwaysFlat(), RandomValid(seed=0), BuyAndHold(), MACross()]
    if model is not None:
        policies.insert(0, TrainedModel(model))
    return policies


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def load_bound_model_provenance(
    model_path: Path,
    *,
    allow_unbound: bool,
) -> Dict[str, Any]:
    provenance_path = model_path.with_suffix(".provenance.json")
    if not provenance_path.exists():
        if allow_unbound:
            return {
                "status": "UNBOUND_MODEL_ALLOWED_FOR_DIAGNOSTICS",
                "path": str(provenance_path),
            }
        raise FileNotFoundError(
            f"model provenance is required for evaluation: {provenance_path}. "
            "Use --allow-unbound-model only for a non-promotional diagnostic."
        )
    payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid model provenance object: {provenance_path}")
    expected_hash = ((payload.get("model") or {}).get("sha256"))
    actual_hash = _sha256_file(model_path)
    if not expected_hash or str(expected_hash).lower() != actual_hash.lower():
        raise ValueError(
            f"model hash does not match provenance: expected={expected_hash!r}, "
            f"actual={actual_hash}"
        )
    schema = payload.get("observation_schema") or {}
    if str(schema.get("version", "")) != str(PPO_OBS_VERSION):
        raise ValueError(
            f"provenance observation schema {schema.get('version')!r} is incompatible "
            f"with runtime {PPO_OBS_VERSION!r}"
        )
    return payload


def format_by_regime(results: List[Result]) -> str:
    """Break the model's P&L down by the volatility regime of each entry."""
    model = next((r for r in results if r.name == "trained-model"), None)
    if model is None or not model.by_vol:
        return ""

    lines = ["", "  by volatility regime at entry:",
             "    %-10s %10s %8s %8s %9s" % ("regime", "P&L EUR", "trades", "win%", "avg EUR")]
    for regime in ("low", "medium", "high", "unknown"):
        vals = model.by_vol.get(regime)
        if not vals:
            continue
        a = np.asarray(vals, dtype=float)
        lines.append("    %-10s %10.0f %8d %8.1f %9.2f" % (
            regime, a.sum(), a.size, (a > 0).mean() * 100.0, a.mean()))
    return chr(10).join(lines)


def format_table(results: List[Result], title: str) -> str:
    head = (
        f"{'strategy':<16}{'return%':>9}{'PF':>7}{'WR%':>7}{'expR':>8}"
        f"{'R:R':>7}{'maxDD%':>8}{'trades':>8}{'t/day':>7}{'hold':>6}  threshold diagnostic"
    )
    lines = [f"\n{title}", "=" * len(head), head, "-" * len(head)]
    for r in results:
        pf = "inf" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
        lines.append(
            f"{r.name:<16}{r.return_pct:>9.2f}{pf:>7}{r.win_rate:>7.1f}"
            f"{r.expectancy_r:>8.3f}{r.reward_risk:>7.2f}{r.max_drawdown_pct:>8.2f}"
            f"{r.trades:>8}{r.trades_per_day:>7.1f}{r.median_bars_held:>6.0f}  "
            f"{r.challenge_threshold_diagnostic}"
        )
    return "\n".join(lines)


def load_ftmo_data(ftmo_dir: str, instrument: str, after: Optional[Any]) -> Optional[Dict[str, Dict[str, Any]]]:
    """Broker bars strictly after the processed dataset ends."""
    import pandas as pd

    d = Path(ftmo_dir)
    if not d.exists():
        return None

    frames: Dict[str, Any] = {}
    for tf in ("M15", "H1", "H4", "D1"):
        f = d / f"{instrument}_{tf}.csv"
        if not f.exists():
            continue
        df: pd.DataFrame = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        if df["time"].isna().any():
            raise ValueError(f"{f} contains invalid timestamps")
        if after is not None:
            df = df.loc[df["time"] > after]
        df = df.sort_values(by="time", ignore_index=True)
        if len(df) < 60:
            logger.warning("%s only %d bars after %s - skipping", tf, len(df), after)
            continue
        frames[tf] = df

    if "M15" not in frames:
        return None
    return {instrument: frames}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/curriculum/curriculum_ppo_final.zip")
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--holdout-ratio", type=float, default=0.15)
    ap.add_argument("--holdout-split-at", default=None)
    ap.add_argument("--extra-data-dir", default=None)
    ap.add_argument("--data-cutoff", default=None)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--instrument", default="XAUUSD")
    ap.add_argument("--out", default="logs/holdout_report.json")
    ap.add_argument(
        "--allow-unbound-model",
        action="store_true",
        help="Allow a checkpoint without matching provenance for diagnostics only.",
    )
    ap.add_argument(
        "--ftmo-dir", default="data/ftmo_live",
        help="Broker bars after the loaded dataset ends. Once inspected or used "
             "for engineering they are development data, not a sealed final test.",
    )
    args = ap.parse_args()

    data = load_market_data(
        instruments=[args.instrument],
        min_bars=5000,
        extra_dir=args.extra_data_dir,
        data_cutoff=args.data_cutoff,
    )
    _train, holdout, split_ts = split_data_by_time(
        data,
        args.holdout_ratio,
        split_at=args.holdout_split_at,
        include_holdout_context=True,
    )
    logger.info("Development holdout begins %s", split_ts)

    model = None
    model_provenance: Dict[str, Any] = {"status": "NO_MODEL_BASELINES_ONLY"}
    path = Path(args.model)
    if path.exists():
        model_provenance = load_bound_model_provenance(
            path,
            allow_unbound=bool(args.allow_unbound_model),
        )
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(path))
        shape = getattr(getattr(model, "observation_space", None), "shape", None)
        if shape != (PPO_OBS_SIZE,):
            raise ValueError(
                f"checkpoint observation shape {shape!r} is incompatible with current "
                f"{PPO_OBS_SIZE}-feature schema; retrain"
            )
        logger.info("Loaded %s", path)
    else:
        logger.warning("No model at %s - baselines only", path)

    report: Dict[str, Any] = {
        "report_type": "DEVELOPMENT_DIAGNOSTIC_NOT_LIVE_ACCEPTANCE",
        "split_ts": str(split_ts),
        "model": {
            "path": str(path),
            "sha256": _sha256_file(path) if path.exists() else None,
            "provenance": model_provenance,
        },
        "command": [sys.executable, *sys.argv],
        "loaded_dataset": build_dataset_manifest(
            data,
            role="holdout_report_loaded",
            boundary=args.data_cutoff,
        ),
        "sections": {},
        "section_metadata": {},
    }
    sections = []

    datasets = [
        ("DEVELOPMENT HOLDOUT (original prices)", holdout, 0.0),
        ("DEVELOPMENT HOLDOUT MIRRORED (short-side diagnostic)", holdout, 1.0),
    ]

    m15 = _primary_frame(next(iter(data.values())))
    last_trained = m15["time"].max() if m15 is not None and "time" in m15.columns else None
    ftmo = load_ftmo_data(args.ftmo_dir, args.instrument, last_trained)
    if ftmo is not None:
        n = len(ftmo[args.instrument]["M15"])
        logger.info("FTMO live bars after %s: %d M15", last_trained, n)
        datasets.append((f"BROKER DEVELOPMENT DATA ({n} M15 bars, broker spreads)", ftmo, 0.0))
    else:
        logger.warning("No FTMO data in %s - skipping the live out-of-sample section", args.ftmo_dir)

    for label, dset, mirror in datasets:
        cfg = PropFirmConfig()
        cfg.max_steps_per_episode = int(args.max_steps)
        cfg.mirror_augmentation_prob = mirror
        cfg.domain_randomization_enabled = False
        cfg.high_vol_oversample_prob = 0.0
        cfg.execution.deterministic_costs = True
        cfg.execution.rejection_enabled = False
        cfg.execution.spread_shock_enabled = False
        if dset is holdout:
            cfg.episode_start_min_time = str(split_ts)
        probe = PropFirmTradingEnv(dset, copy.deepcopy(cfg))
        episode_starts = build_non_overlapping_episode_starts(
            probe,
            requested_episodes=args.episodes,
            max_steps=args.max_steps,
        )
        probe.close()
        results = [
            run_policy(
                p,
                dset,
                cfg,
                args.episodes,
                args.seed,
                args.max_steps,
                episode_starts=episode_starts,
            )
            for p in build_policies(model)
        ]
        sections.append(format_table(results, label) + format_by_regime(results))
        report["sections"][label] = [
            vars(r)
            | {
                "challenge_threshold_diagnostic": (
                    r.challenge_threshold_diagnostic
                )
            }
            for r in results
        ]
        report["section_metadata"][label] = {
            "dataset": build_dataset_manifest(
                dset,
                role="holdout_report_section",
                boundary=split_ts if dset is holdout else last_trained,
            ),
            "requested_episodes": int(args.episodes),
            "independent_windows": len(episode_starts),
            "episode_steps": int(args.max_steps),
            "episode_start_indices": episode_starts,
            "overlapping_windows": False,
            "confidence_interval_valid": len(episode_starts) >= 2,
        }

    out = "\n".join(sections)
    print(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(_json_safe(report), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("Report written to %s", args.out)

    # The decisive comparison, stated rather than left for the reader to infer.
    for label, rows in report["sections"].items():
        by = {r["name"]: r for r in rows}
        m, bh = by.get("trained-model"), by.get("one-shot-long")
        if m and bh:
            verdict = "BEATS" if m["return_pct"] > bh["return_pct"] else "LOSES TO"
            print(f"\n{label}: model {verdict} buy-and-hold "
                  f"({m['return_pct']:.2f}% vs {bh['return_pct']:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
