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

# Preregistered. Below this many independent, non-overlapping windows no
# interval is published and no verdict is issued - the June-August development
# window yields one, which is why that section must print no conclusion at all.
MIN_INDEPENDENT_BLOCKS = 10
MIN_TRADES_FOR_VERDICT = 30
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_ALPHA = 0.05


@dataclass(frozen=True)
class ExecutionContract:
    """What a trade is declared to cost when the result is used as evidence.

    Evaluation was silently cheaper than anything the model would meet live:
    commission_spec resolved to NONE, latency collapsed to zero once domain
    randomization was disabled, no overnight financing existed at all, and the
    effective spread came out at 7 points against a measured FTMO median of 40.
    A result produced under those conditions flatters the strategy and cannot be
    compared to a broker.

    The contract is versioned so a stored report can be read back and audited
    against the costs it was actually produced under.
    """

    version: str = "acceptance-1"
    # Spread comes from the data's own spread column; this scales it.
    data_spread_scale: float = 1.0
    base_spread_points: float = 0.22
    slippage_points_sigma: float = 0.06
    max_slippage_points: float = 0.35
    # Per lot, per side - matching the terminal curriculum stage.
    commission_per_lot_per_side: float = 3.0
    latency_bars: int = 1
    # XAUUSD financing, points per lot per night. Both sides are charged on
    # gold at most brokers; these are conservative retail figures.
    swap_long_points_per_night: float = -1.2
    swap_short_points_per_night: float = -0.8
    # 22:00 UTC is the industry-standard rollover.
    rollover_hour_utc: int = 22

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "data_spread_scale": self.data_spread_scale,
            "base_spread_points": self.base_spread_points,
            "slippage_points_sigma": self.slippage_points_sigma,
            "max_slippage_points": self.max_slippage_points,
            "commission_per_lot_per_side": self.commission_per_lot_per_side,
            "latency_bars": self.latency_bars,
            "swap_long_points_per_night": self.swap_long_points_per_night,
            "swap_short_points_per_night": self.swap_short_points_per_night,
            "rollover_hour_utc": self.rollover_hour_utc,
        }


ACCEPTANCE_CONTRACT = ExecutionContract()


def apply_execution_contract(cfg: PropFirmConfig, contract: ExecutionContract) -> Dict[str, Any]:
    """Impose the contract on an evaluation config and report what took effect.

    Returns the settings actually resolved, so the report states its costs
    rather than leaving them to be inferred from defaults.
    """
    from envs.core.execution_model import CommissionMode, CommissionSpec

    ex = cfg.execution
    ex.data_spread_scale = float(contract.data_spread_scale)
    ex.base_spread_points = float(contract.base_spread_points)
    ex.slippage_points_sigma = float(contract.slippage_points_sigma)
    ex.max_slippage_points = float(contract.max_slippage_points)
    ex.latency_bars = int(contract.latency_bars)

    # deterministic_costs must not mean free: slippage stays at its declared
    # magnitude, it simply stops being random.
    ex.deterministic_costs = True
    ex.rejection_enabled = False
    ex.spread_shock_enabled = False

    rate = float(contract.commission_per_lot_per_side)
    ex.commission_spec = (
        CommissionSpec(mode=CommissionMode.PER_LOT_PER_SIDE, commission_rate=rate)
        if rate > 0.0
        else CommissionSpec(mode=CommissionMode.NONE, commission_rate=0.0)
    )

    return {
        "contract": contract.as_dict(),
        "resolved": {
            "data_spread_scale": ex.data_spread_scale,
            "base_spread_points": ex.base_spread_points,
            "slippage_points_sigma": ex.slippage_points_sigma,
            "max_slippage_points": ex.max_slippage_points,
            "latency_bars": ex.latency_bars,
            "commission_mode": str(getattr(ex.commission_spec, "mode", "none")),
            "commission_rate": float(getattr(ex.commission_spec, "commission_rate", 0.0)),
            "deterministic_costs": ex.deterministic_costs,
        },
    }


def swap_cost_eur(
    trades: Sequence[Dict[str, Any]],
    contract: ExecutionContract,
    *,
    bars_per_day: float = 96.0,
    point_value_per_lot: float = 100.0,
) -> float:
    """Overnight financing the environment does not model, priced from the trades.

    The env carries no swap at all, so a multi-day hold currently costs nothing
    to finance. Rather than leave that omission silent, it is computed from each
    trade's recorded holding time and direction and deducted explicitly.
    """
    total = 0.0
    for tr in trades:
        bars = float(tr.get("bars_held", 0.0) or 0.0)
        lots = float(tr.get("lot_size", 0.0) or 0.0)
        if bars <= 0.0 or lots <= 0.0:
            continue
        nights = int(bars // bars_per_day)
        if nights <= 0:
            continue
        direction = str(tr.get("direction", "")).lower()
        if direction not in ("long", "short"):
            # Defaulting to "long" would silently charge every short the wrong
            # financing rate, which is how a cost model quietly becomes fiction.
            raise ValueError(
                f"trade record has no usable direction ({tr.get('direction')!r}); "
                f"cannot price overnight financing"
            )
        rate = (
            contract.swap_long_points_per_night
            if direction == "long"
            else contract.swap_short_points_per_night
        )
        total += nights * rate * lots * point_value_per_lot / 100.0
    return float(total)


@dataclass
class Result:
    name: str
    episodes: int = 0
    trades: int = 0
    # Scope is part of the name. `total_pnl` held a MEAN across reset episodes
    # while reading as a total, which is how a +2.57% mean became a published
    # +25.70%. Every figure below states which population it summarises.
    mean_episode_pnl_eur: float = 0.0
    pooled_sample_pnl_eur: float = 0.0
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
    reconciliation_checked: int = 0
    ci_status: str = "not_computed"
    passive_return_pct: Optional[float] = None
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
        episode_trade_pnl = 0.0
        episode_long_pnl = 0.0
        episode_short_pnl = 0.0
        episode_trade_count = 0
        for tr in stats.get("trades_with_regime", []) or []:
            res.trades += 1
            episode_trade_count += 1
            episode_trade_pnl += float(tr.get("pnl", 0.0))
            trade_pnls.append(float(tr.get("pnl", 0.0)))
            res.r_multiples.append(float(tr.get("r_multiple", 0.0)))
            bars_held.append(float(tr.get("bars_held", 0)))
            res.by_vol.setdefault(str(tr.get("volatility_regime", "unknown")), []).append(
                float(tr.get("pnl", 0.0))
            )

        ds = stats.get("direction_stats", {}) or {}
        episode_long_pnl = float(ds.get("long_pnl", 0.0))
        episode_short_pnl = float(ds.get("short_pnl", 0.0))
        res.long_pnl += episode_long_pnl
        res.short_pnl += episode_short_pnl

        # Reconciliation. A scope error is silent by nature - a mean printed as
        # a total looks like a plausible number - so the accounting is checked
        # against itself rather than trusted. An open position at the horizon
        # leaves unrealised P&L outside the trade list, which is the one
        # legitimate gap; anything else is a defect.
        reconciliation = _reconcile_episode(
            equity_change=episode_pnl,
            trade_pnl=episode_trade_pnl,
            long_pnl=episode_long_pnl,
            short_pnl=episode_short_pnl,
            recorded_trades=episode_trade_count,
            reported_trades=int(stats.get("trade_count", episode_trade_count)),
            position_open=env.position is not None,
        )
        if reconciliation["failures"]:
            raise AssertionError(
                f"{policy.name} episode {ep} accounting does not reconcile: "
                + "; ".join(reconciliation["failures"])
            )
        res.reconciliation_checked += 1
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

    res.mean_episode_pnl_eur = float(np.mean(episode_pnls)) if episode_pnls else 0.0
    res.pooled_sample_pnl_eur = float(np.sum(episode_pnls)) if episode_pnls else 0.0
    res.return_pct = float(np.mean(episode_returns)) if episode_returns else 0.0
    # A 1.96-sigma bound computed from two blocks is arithmetic, not inference.
    # It requires a preregistered minimum number of independent windows, and
    # even then a normal approximation is wrong for a small sample of serially
    # dependent block returns - so the bound comes from a bootstrap over whole
    # blocks, which makes no distributional assumption.
    if len(episode_returns) >= MIN_INDEPENDENT_BLOCKS:
        res.return_ci_low_pct = _block_bootstrap_lower_bound(
            np.asarray(episode_returns, dtype=float), seed=seed
        )
        res.ci_status = "computed"
    else:
        res.return_ci_low_pct = None
        res.ci_status = (
            f"withheld: {len(episode_returns)} independent windows, "
            f"{MIN_INDEPENDENT_BLOCKS} required"
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


def evaluate_section_verdict(
    rows: List[Dict[str, Any]],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Decide whether a section supports any conclusion at all.

    INSUFFICIENT_EVIDENCE unless the sample can support inference;
    NOT_SUPPORTED unless the model clears every control. Only a result that
    survives all of it earns SUPPORTED - and even then it is development
    evidence, never live approval.
    """
    by = {r.get("name"): r for r in rows}
    model = by.get("trained-model")
    flat = by.get("always-flat")

    reasons: List[str] = []
    blocking: List[str] = []

    if model is None:
        return {
            "verdict": "NO_MODEL_EVALUATED",
            "reasons": ["no trained-model row in this section"],
            "blocking": ["model_absent"],
        }

    windows = int(meta.get("independent_windows", 0) or 0)
    if windows < MIN_INDEPENDENT_BLOCKS:
        blocking.append("independent_windows")
        reasons.append(
            f"{windows} independent windows, {MIN_INDEPENDENT_BLOCKS} required - "
            f"no interval and no verdict can come from this sample"
        )

    trades = int(model.get("trades", 0) or 0)
    if trades < MIN_TRADES_FOR_VERDICT:
        blocking.append("trades")
        reasons.append(f"{trades} trades, {MIN_TRADES_FOR_VERDICT} required")

    if meta.get("overlapping_windows"):
        blocking.append("overlapping_windows")
        reasons.append("windows overlap, so blocks are not independent")

    if int(model.get("reconciliation_checked", 0) or 0) < int(model.get("episodes", 0) or 0):
        blocking.append("reconciliation")
        reasons.append("not every episode passed accounting reconciliation")

    if blocking:
        return {"verdict": "INSUFFICIENT_EVIDENCE", "reasons": reasons, "blocking": blocking}

    ci_low = model.get("return_ci_low_pct")
    if ci_low is None:
        blocking.append("confidence_interval")
        reasons.append("no confidence bound available")
    elif float(ci_low) <= 0.0:
        blocking.append("positive_lower_bound")
        reasons.append(
            f"95% lower bound {float(ci_low):+.3f}% is not above zero - consistent "
            f"with having no edge"
        )
    else:
        reasons.append(f"95% lower bound {float(ci_low):+.3f}% is above zero")

    model_ret = float(model.get("return_pct", 0.0))
    if flat is None:
        blocking.append("flat_control_missing")
        reasons.append("always-flat control absent")
    elif model_ret <= float(flat.get("return_pct", 0.0)):
        blocking.append("beats_flat")
        reasons.append(
            f"model {model_ret:+.3f}% does not beat standing aside "
            f"({float(flat.get('return_pct', 0.0)):+.3f}%)"
        )
    else:
        reasons.append("beats standing aside")

    passive = model.get("passive_return_pct")
    if passive is None:
        reasons.append("no passive market benchmark available (not blocking)")
    elif model_ret <= float(passive):
        blocking.append("beats_market")
        reasons.append(
            f"model {model_ret:+.3f}% does not beat the passive market return "
            f"({float(passive):+.3f}%)"
        )
    else:
        reasons.append("beats the passive market return")

    if float(model.get("max_drawdown_pct", 0.0)) > FTMO_MAX_DD * 100.0:
        blocking.append("drawdown")
        reasons.append(
            f"worst window drawdown {float(model.get('max_drawdown_pct', 0.0)):.2f}% "
            f"breaches the {FTMO_MAX_DD * 100:.0f}% limit"
        )

    if blocking:
        return {"verdict": "NOT_SUPPORTED", "reasons": reasons, "blocking": blocking}

    reasons.append("development evidence only - not live approval")
    return {"verdict": "SUPPORTED_DEVELOPMENT_ONLY", "reasons": reasons, "blocking": []}


def _block_bootstrap_lower_bound(
    block_returns: np.ndarray,
    *,
    seed: int,
    alpha: float = BOOTSTRAP_ALPHA,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> float:
    """Lower confidence bound by resampling whole blocks with replacement.

    Blocks are the unit of independence here: bars inside a window are serially
    dependent, so resampling bars would badly understate the interval.
    Resampling whole windows makes no normality assumption, which matters at the
    sample sizes this evaluation can actually produce.
    """
    rng = np.random.default_rng(seed)
    n = block_returns.size
    idx = rng.integers(0, n, size=(resamples, n))
    means = block_returns[idx].mean(axis=1)
    return float(np.percentile(means, 100.0 * alpha))


def compute_passive_return_pct(
    data: Dict[str, Dict[str, Any]],
    instrument: str,
    episode_starts: Sequence[int],
    horizon: int,
) -> Optional[float]:
    """Mean buy-and-hold price return over the same windows, outside the env.

    The `one-shot-long` policy trades through the environment, so its result is
    shaped by stops, time decay, spread and the drawdown veto - it answers "what
    happens if the agent only ever buys", not "what did the market do". This is
    the market's own return over identical windows, which is the benchmark a
    strategy actually has to beat.
    """
    frames = data.get(instrument) or {}
    df = frames.get("M15")
    if df is None or getattr(df, "empty", True) or not len(episode_starts):
        return None

    close = df["close"].to_numpy(dtype=float)
    rets: List[float] = []
    for start in episode_starts:
        end = min(int(start) + int(horizon), close.size - 1)
        if end <= start or close[start] <= 0:
            continue
        rets.append((close[end] / close[start] - 1.0) * 100.0)
    return float(np.mean(rets)) if rets else None


def _reconcile_episode(
    *,
    equity_change: float,
    trade_pnl: float,
    long_pnl: float,
    short_pnl: float,
    recorded_trades: int,
    reported_trades: int,
    position_open: bool,
    tolerance_eur: float = 0.01,
) -> Dict[str, Any]:
    """Check an episode's books against themselves.

    Three identities must hold: closed-trade P&L explains the change in equity,
    the long and short decomposition sums to the same figure, and the trade
    count the env reports matches the trades actually recorded. A position still
    open at the horizon holds unrealised P&L that is outside the trade list -
    that is the only tolerated discrepancy, and it is reported rather than
    hidden.
    """
    failures: List[str] = []

    direction_sum = long_pnl + short_pnl
    if abs(direction_sum - trade_pnl) > tolerance_eur:
        failures.append(
            f"long+short {direction_sum:.4f} != trade P&L {trade_pnl:.4f}"
        )

    if recorded_trades != reported_trades:
        failures.append(
            f"recorded {recorded_trades} trades, env reported {reported_trades}"
        )

    residual = equity_change - trade_pnl
    if not position_open and abs(residual) > tolerance_eur:
        failures.append(
            f"equity moved {equity_change:.4f} but closed trades explain "
            f"{trade_pnl:.4f} (residual {residual:.4f}) with no open position"
        )

    return {
        "failures": failures,
        "unrealised_residual_eur": float(residual) if position_open else 0.0,
    }


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
        "--contract-version", default=ACCEPTANCE_CONTRACT.version,
        help="Execution contract the result is produced under. Recorded in the "
             "report so stored evidence can be audited against its own costs.",
    )
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

    contract = ACCEPTANCE_CONTRACT

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
        "section_verdicts": {},
        "execution_contract": ACCEPTANCE_CONTRACT.as_dict(),
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
        # Costs are declared, not inherited. Setting deterministic_costs alone
        # left commission at NONE and latency at zero, so evaluation ran far
        # cheaper than the terminal curriculum stage or the live book.
        execution_settings = apply_execution_contract(cfg, contract)
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

        # The market's own return over the identical windows, computed outside
        # the environment. one-shot-long trades through stops, time decay,
        # spread and the drawdown veto, so it answers "what if the agent only
        # ever bought", not "what did the market do". Only the latter is the
        # benchmark a strategy has to clear.
        passive = compute_passive_return_pct(
            dset, args.instrument, episode_starts, args.max_steps
        )
        for r in results:
            r.passive_return_pct = passive

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
            "confidence_interval_valid": len(episode_starts) >= MIN_INDEPENDENT_BLOCKS,
            "execution_settings": execution_settings,
        }

    out = "\n".join(sections)
    print(out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(_json_safe(report), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("Report written to %s", args.out)

    # Fail closed. The previous version compared the model to one-shot-long and
    # printed BEATS or LOSES TO, which could return a confident verdict from a
    # single window, from three trades, or against a benchmark that is not the
    # market. Every condition below must hold; missing evidence is a refusal to
    # conclude, never a pass.
    print()
    for label, rows in report["sections"].items():
        meta = report["section_metadata"].get(label, {})
        verdict = evaluate_section_verdict(rows, meta)
        report.setdefault("section_verdicts", {})[label] = verdict

        print(f"{label}")
        print(f"   VERDICT: {verdict['verdict']}")
        for line in verdict["reasons"]:
            print(f"     - {line}")
        print()

    Path(args.out).write_text(
        json.dumps(_json_safe(report), indent=2, default=str, allow_nan=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
