#!/usr/bin/env python3
"""Score a trained model on unseen data against baselines it must beat.

A profit factor means nothing on its own here. XAUUSD rose 149.8% across this
dataset, so "made money" is the null hypothesis rather than the finding - a
leveraged buy-and-hold would clear most bars a strategy is usually judged by.
The only useful question is whether the policy beats the alternatives on data
it never trained on.

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
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.core.env_types import PropFirmConfig  # noqa: E402
from envs.prop_firm_env import PropFirmTradingEnv  # noqa: E402
from train.train_prop_firm import _primary_frame, load_market_data, split_data_by_time  # noqa: E402

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
    r_multiples: List[float] = field(default_factory=list)
    # An aggregate number hides the thing that matters most: a strategy can
    # look profitable overall while losing badly in exactly the regime it now
    # has to trade.
    by_vol: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def ftmo_pass(self) -> bool:
        return (
            self.return_pct >= FTMO_PROFIT_TARGET * 100.0
            and self.max_drawdown_pct <= FTMO_MAX_DD * 100.0
            and self.worst_daily_dd_pct <= FTMO_DAILY_DD * 100.0
        )

    @property
    def ftmo_verdict(self) -> str:
        if self.ftmo_pass:
            return "PASS"
        reasons = []
        if self.return_pct < FTMO_PROFIT_TARGET * 100.0:
            reasons.append(f"profit {self.return_pct:.1f}% < 10%")
        if self.max_drawdown_pct > FTMO_MAX_DD * 100.0:
            reasons.append(f"maxDD {self.max_drawdown_pct:.1f}% > 10%")
        if self.worst_daily_dd_pct > FTMO_DAILY_DD * 100.0:
            reasons.append(f"dailyDD {self.worst_daily_dd_pct:.1f}% > 5%")
        return "FAIL: " + ", ".join(reasons)


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
    """Enter long once, never exit. The benchmark a 149.8% uptrend flatters."""
    name = "buy-and-hold"

    def act(self, env, obs, mask):
        if env.position is None and env.pending_entry is None:
            a = self._first_valid(mask, [env._ACTION_LONG_START + env._K - 1, env._ACTION_LONG_START])
            if a is not None:
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
        try:
            action, _ = self.model.predict(
                obs[None, :], action_masks=mask[None, :], deterministic=self.deterministic
            )
        except TypeError:
            action, _ = self.model.predict(obs[None, :], deterministic=self.deterministic)
        a = int(np.asarray(action).ravel()[0])
        return a if mask[a] else env._ACTION_HOLD


# ── evaluation ──────────────────────────────────────────────────────────────

def run_policy(
    policy: Policy,
    data: Dict[str, Dict[str, Any]],
    cfg: PropFirmConfig,
    episodes: int,
    seed: int,
    max_steps: int,
) -> Result:
    env = PropFirmTradingEnv(data, cfg)
    res = Result(name=policy.name)

    equity_curve: List[float] = []
    bars_held: List[float] = []
    daily_dds: List[float] = []
    pnls: List[float] = []
    total_bars = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy.reset()
        peak = float(env.equity)
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
            equity_curve.append(float(env.equity))
            # Daily DD is not reported in episode_stats, so sample the env's
            # own calculation - it is a hard prop-firm limit and must be
            # measured, not inferred from the equity curve.
            try:
                _cur, daily = env._calc_dds()
                worst_daily = max(worst_daily, float(daily))
            except Exception:
                pass

        total_bars += steps
        res.episodes += 1

        # Read the env directly rather than from info: episode_stats is only
        # attached to info on termination, and max_steps cuts most episodes
        # short, so relying on info reported zero trades for every policy while
        # drawdown showed positions had clearly been taken.
        stats: Dict[str, Any] = {}
        if hasattr(env, "get_episode_stats"):
            try:
                stats = env.get_episode_stats() or {}
            except Exception as e:  # noqa: BLE001 - diagnostics must not abort a run
                logger.warning("episode stats unavailable for %s: %s", policy.name, e)
        if not stats and isinstance(info, dict):
            stats = info.get("episode_stats", {}) or {}
        for tr in stats.get("trades_with_regime", []) or []:
            res.trades += 1
            pnls.append(float(tr.get("pnl", 0.0)))
            res.r_multiples.append(float(tr.get("r_multiple", 0.0)))
            bars_held.append(float(tr.get("bars_held", 0)))
            res.by_vol.setdefault(str(tr.get("volatility_regime", "unknown")), []).append(
                float(tr.get("pnl", 0.0))
            )

        ds = stats.get("direction_stats", {}) or {}
        res.long_pnl += float(ds.get("long_pnl", 0.0))
        res.short_pnl += float(ds.get("short_pnl", 0.0))
        daily_dds.append(worst_daily)

    initial = float(cfg.initial_balance)
    res.total_pnl = float(np.sum(pnls))
    res.return_pct = res.total_pnl / max(initial, 1.0) * 100.0

    arr = np.asarray(pnls, dtype=float)
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
    if equity_curve:
        eq = np.asarray(equity_curve, dtype=float)
        running_peak = np.maximum.accumulate(eq)
        res.max_drawdown_pct = float(np.max((running_peak - eq) / np.maximum(running_peak, 1.0)) * 100.0)
    if daily_dds:
        res.worst_daily_dd_pct = float(max(daily_dds) * 100.0)
    if total_bars:
        res.trades_per_day = res.trades / (total_bars / 96.0)

    return res


def build_policies(model) -> List[Policy]:
    policies: List[Policy] = [AlwaysFlat(), RandomValid(seed=0), BuyAndHold(), MACross()]
    if model is not None:
        policies.insert(0, TrainedModel(model))
    return policies


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
        f"{'R:R':>7}{'maxDD%':>8}{'trades':>8}{'t/day':>7}{'hold':>6}  verdict"
    )
    lines = [f"\n{title}", "=" * len(head), head, "-" * len(head)]
    for r in results:
        pf = "inf" if r.profit_factor == float("inf") else f"{r.profit_factor:.2f}"
        lines.append(
            f"{r.name:<16}{r.return_pct:>9.2f}{pf:>7}{r.win_rate:>7.1f}"
            f"{r.expectancy_r:>8.3f}{r.reward_risk:>7.2f}{r.max_drawdown_pct:>8.2f}"
            f"{r.trades:>8}{r.trades_per_day:>7.1f}{r.median_bars_held:>6.0f}  {r.ftmo_verdict}"
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
        df: pd.DataFrame = pd.read_csv(f, parse_dates=["time"])
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
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--instrument", default="XAUUSD")
    ap.add_argument("--out", default="logs/holdout_report.json")
    ap.add_argument(
        "--ftmo-dir", default="data/ftmo_live",
        help="Bars pulled live from the broker, after the processed dataset ends. "
             "This is the real out-of-sample set: not a split of data the model "
             "trained beside, but a different regime at the broker's own spreads.",
    )
    args = ap.parse_args()

    data = load_market_data(instruments=[args.instrument], min_bars=5000)
    _train, holdout, split_ts = split_data_by_time(data, args.holdout_ratio)
    logger.info("Holdout begins %s - never seen during training", split_ts)

    model = None
    path = Path(args.model)
    if path.exists():
        from sb3_contrib import MaskablePPO
        model = MaskablePPO.load(str(path))
        logger.info("Loaded %s", path)
    else:
        logger.warning("No model at %s - baselines only", path)

    report: Dict[str, Any] = {"split_ts": str(split_ts), "model": str(path), "sections": {}}
    sections = []

    datasets = [
        ("HOLDOUT (unseen split, original prices)", holdout, 0.0),
        ("HOLDOUT MIRRORED (drift inverted - the short-side test)", holdout, 1.0),
    ]

    m15 = _primary_frame(next(iter(data.values())))
    last_trained = m15["time"].max() if m15 is not None and "time" in m15.columns else None
    ftmo = load_ftmo_data(args.ftmo_dir, args.instrument, last_trained)
    if ftmo is not None:
        n = len(ftmo[args.instrument]["M15"])
        logger.info("FTMO live bars after %s: %d M15", last_trained, n)
        datasets.append((f"FTMO LIVE DATA (never seen, {n} M15 bars, broker spreads)", ftmo, 0.0))
    else:
        logger.warning("No FTMO data in %s - skipping the live out-of-sample section", args.ftmo_dir)

    for label, dset, mirror in datasets:
        cfg = PropFirmConfig()
        cfg.mirror_augmentation_prob = mirror
        results = [
            run_policy(p, dset, cfg, args.episodes, args.seed, args.max_steps)
            for p in build_policies(model)
        ]
        sections.append(format_table(results, label) + format_by_regime(results))
        report["sections"][label] = [vars(r) | {"ftmo": r.ftmo_verdict} for r in results]

    out = "\n".join(sections)
    print(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Report written to %s", args.out)

    # The decisive comparison, stated rather than left for the reader to infer.
    for label, rows in report["sections"].items():
        by = {r["name"]: r for r in rows}
        m, bh = by.get("trained-model"), by.get("buy-and-hold")
        if m and bh:
            verdict = "BEATS" if m["return_pct"] > bh["return_pct"] else "LOSES TO"
            print(f"\n{label}: model {verdict} buy-and-hold "
                  f"({m['return_pct']:.2f}% vs {bh['return_pct']:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
