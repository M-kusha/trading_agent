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

# Populated by load_ftmo_data so the report can name the exact broker slice
# it consumed, without polluting the dataset the environment iterates.
BROKER_SLICE_FINGERPRINTS: Dict[str, Any] = {}


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
class AccountBreach:
    kind: str
    at: Optional[str]
    drawdown_pct: float
    limit_pct: float
    equity: float


class AccountLedger:
    """A single continuous prop-firm account, with explicit reset rules.

    Drawdown was previously inferred from a concatenated equity curve across
    independently reset episodes, so the jump from one episode's closing balance
    back to the next episode's opening balance registered as a loss that never
    happened. A prop-firm limit is a property of one account over calendar time;
    it cannot be recovered from a pile of resets.

    Rules are stated rather than assumed:
      * the daily anchor rolls at `rollover_hour_utc`, the broker's day boundary,
        not at midnight local time;
      * the maximum-loss anchor is the initial balance for a static rule, or the
        running peak for a trailing one, matching `trailing_drawdown`;
      * a breach is `>=` the limit, matching the environment exactly, so the
        ledger and the env cannot disagree about whether an account died.
    """

    def __init__(
        self,
        initial_balance: float,
        *,
        max_dd_limit: float = FTMO_MAX_DD,
        daily_dd_limit: float = FTMO_DAILY_DD,
        trailing: bool = False,
        rollover_hour_utc: int = 22,
    ) -> None:
        self.initial_balance = float(initial_balance)
        self.max_dd_limit = float(max_dd_limit)
        self.daily_dd_limit = float(daily_dd_limit)
        self.trailing = bool(trailing)
        self.rollover_hour_utc = int(rollover_hour_utc)

        self.equity = float(initial_balance)
        self.peak = float(initial_balance)
        self.day_start = float(initial_balance)
        self._day_key: Optional[Any] = None

        self.worst_total_dd = 0.0
        self.worst_daily_dd = 0.0
        self.breaches: List[AccountBreach] = []
        self.dead = False
        self.bars = 0

    def _trading_day(self, ts: Any) -> Any:
        """Day key rolling at the broker hour, so a session is one day."""
        import pandas as pd

        t = pd.Timestamp(ts)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return (t - pd.Timedelta(hours=self.rollover_hour_utc)).date()

    def mark(self, timestamp: Any, equity: float) -> Optional[AccountBreach]:
        """Record one bar. Returns a breach the first time the account dies."""
        self.bars += 1
        self.equity = float(equity)

        if timestamp is not None:
            day = self._trading_day(timestamp)
            if self._day_key is None:
                self._day_key = day
            elif day != self._day_key:
                self._day_key = day
                self.day_start = self.equity

        self.peak = max(self.peak, self.equity)

        anchor = self.peak if self.trailing else self.initial_balance
        total_dd = max(0.0, (anchor - self.equity) / max(anchor, 1.0))
        daily_dd = max(0.0, (self.day_start - self.equity) / max(self.day_start, 1.0))

        self.worst_total_dd = max(self.worst_total_dd, total_dd)
        self.worst_daily_dd = max(self.worst_daily_dd, daily_dd)

        if self.dead:
            return None

        # `>=`, matching the environment's own breach test.
        if total_dd >= self.max_dd_limit:
            breach = AccountBreach("max_drawdown", _iso(timestamp), total_dd * 100.0,
                                   self.max_dd_limit * 100.0, self.equity)
        elif daily_dd >= self.daily_dd_limit:
            breach = AccountBreach("daily_drawdown", _iso(timestamp), daily_dd * 100.0,
                                   self.daily_dd_limit * 100.0, self.equity)
        else:
            return None

        self.dead = True
        self.breaches.append(breach)
        return breach

    def summary(self) -> Dict[str, Any]:
        return {
            "initial_balance": self.initial_balance,
            "final_equity": self.equity,
            "return_pct": (self.equity - self.initial_balance) / max(self.initial_balance, 1.0) * 100.0,
            "worst_total_drawdown_pct": self.worst_total_dd * 100.0,
            "worst_daily_drawdown_pct": self.worst_daily_dd * 100.0,
            "max_dd_limit_pct": self.max_dd_limit * 100.0,
            "daily_dd_limit_pct": self.daily_dd_limit * 100.0,
            "anchor": "peak" if self.trailing else "initial_balance",
            "rollover_hour_utc": self.rollover_hour_utc,
            "bars": self.bars,
            "survived": not self.dead,
            "breaches": [vars(b) for b in self.breaches],
        }


def _iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    try:
        return ts.isoformat()
    except AttributeError:
        return str(ts)


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
        # Deliberately not a prop-firm verdict.  These figures are means and
        # worst-cases over independently reset windows; no single account ever
        # lived through them, so a challenge cannot be passed or failed here.
        # Survival belongs to the continuous replay, which walks one account.
        raise NotImplementedError(
            "prop-firm thresholds do not apply to reset-window statistics; "
            "read the continuous_replay section instead"
        )

    @property
    def window_statistics_summary(self) -> str:
        """What these numbers are: statistics over reset windows, nothing more."""
        pf = (
            "n/a"
            if not self.trades or not np.isfinite(self.profit_factor)
            else f"{self.profit_factor:.2f}"
        )
        return (
            f"mean {self.return_pct:+.3f}%/window over {self.episodes} windows, "
            f"PF {pf}, worst-window DD {self.max_drawdown_pct:.2f}%"
        )


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
        # An "infinite" profit factor is what one winning trade and no losers
        # looks like; printing `inf` dresses a non-result as a perfect one.
        res.profit_factor = (
            gross_win / gross_loss if gross_loss > 1e-9 else float("nan")
        )

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


def build_artifact_integrity(args: Any, contract: ExecutionContract) -> Dict[str, Any]:
    """Everything needed to reproduce or distrust this report later.

    A stored result that does not name the code, the costs and the working-tree
    state that produced it cannot be audited - and an invalid artifact is
    indistinguishable from a valid one once its context is gone.
    """
    import subprocess

    def _git(*cmd: str) -> Optional[str]:
        try:
            return subprocess.run(
                ["git", *cmd], capture_output=True, text=True, timeout=10,
                cwd=str(PROJECT_ROOT),
            ).stdout.strip() or None
        except Exception:
            return None

    dirty = _git("status", "--porcelain")

    return {
        "evaluator_sha256": _sha256_file(Path(__file__)),
        "git_head": _git("rev-parse", "HEAD"),
        # A dirty tree means the committed code is not what ran.
        "git_dirty": bool(dirty),
        "git_dirty_files": len(dirty.splitlines()) if dirty else 0,
        "command": [sys.executable, *sys.argv],
        "seed": int(getattr(args, "seed", 0)),
        "episodes_requested": int(getattr(args, "episodes", 0)),
        "episode_bars": int(getattr(args, "max_steps", 0)),
        "continuous_bars": int(getattr(args, "continuous_bars", 0)),
        "execution_contract": contract.as_dict(),
        "broker_slice_fingerprints": dict(BROKER_SLICE_FINGERPRINTS),
        "metric_definitions": {
            "return_pct": "mean of per-window account returns; NOT a compounded total",
            "mean_episode_pnl_eur": "mean P&L per reset window",
            "pooled_sample_pnl_eur": "sum across all windows; not an account balance",
            "max_drawdown_pct": "worst single-window drawdown, peak-to-trough within that window",
            "return_ci_low_pct": f"{int((1 - BOOTSTRAP_ALPHA) * 100)}% lower bound, block bootstrap over whole windows",
            "trades_per_day": "trades divided by trading days derived from bar timestamps",
            "continuous_replay": "one account walked chronologically; the only place prop-firm limits apply",
        },
    }


def trading_days_from_records(episode_records: Sequence[Dict[str, Any]]) -> float:
    """Trading days from actual timestamps, not bars/96.

    bars/96 assumes a 24h market with no gaps, so a window spanning a weekend is
    counted as though it traded through it and every per-day rate is understated.
    """
    import pandas as pd

    days = 0.0
    for rec in episode_records:
        start, end = rec.get("start_time"), rec.get("end_time")
        if not start or not end:
            continue
        span = (pd.Timestamp(end) - pd.Timestamp(start)).total_seconds() / 86400.0
        if span > 0:
            days += span
    return days


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


def run_continuous_replay(
    policy: Policy,
    data: Dict[str, Dict[str, Any]],
    cfg: PropFirmConfig,
    *,
    start_index: int,
    max_bars: int,
    instrument: str,
    contract: ExecutionContract,
) -> Dict[str, Any]:
    """Walk the window once, chronologically, on ONE account.

    This is the challenge-survival product and it answers a different question
    from the reset blocks: not "what does a typical month look like" but "does
    this account still exist at the end". Prop-firm limits belong here and only
    here - applying them to a mean over reset episodes is a category error,
    because no single account ever experienced that mean.

    The account is never reset. If it breaches, the replay stops, because a real
    account would be closed.
    """
    env = PropFirmTradingEnv(data, copy.deepcopy(cfg))
    env.reset(seed=0)
    env.current_step = int(start_index)

    ledger = AccountLedger(
        initial_balance=float(cfg.initial_balance),
        trailing=bool(getattr(cfg, "trailing_drawdown", False)),
        rollover_hour_utc=contract.rollover_hour_utc,
    )

    policy.reset()
    obs = env._get_observation()
    trades: List[Dict[str, Any]] = []
    breach: Optional[AccountBreach] = None
    bars = 0

    for _ in range(int(max_bars)):
        mask = env.action_masks()
        action = policy.act(env, obs, mask)
        obs, _reward, terminated, truncated, _info = env.step(int(action))
        bars += 1

        breach = ledger.mark(env._get_bar_dt(instrument), float(env.equity))
        if breach is not None:
            break
        if terminated or truncated:
            break

    stats = env.get_episode_stats() or {}
    trades = list(stats.get("trades_with_regime", []) or [])
    env.close()

    summary = ledger.summary()
    swap = swap_cost_eur(trades, contract)
    summary.update({
        "policy": policy.name,
        "bars_walked": bars,
        "trades": len(trades),
        "gross_return_pct": summary["return_pct"],
        "overnight_financing_eur": swap,
        "net_return_pct": summary["return_pct"] + swap / max(cfg.initial_balance, 1.0) * 100.0,
        "terminated_reason": (
            breach.kind if breach is not None else stats.get("termination_reason") or "horizon"
        ),
    })
    return summary


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


def bind_evaluation_to_provenance(
    provenance: Dict[str, Any],
    *,
    loaded_manifest: Dict[str, Any],
    train_manifest: Dict[str, Any],
    holdout_manifest: Dict[str, Any],
    resolved_split_at: Any,
    data_cutoff: Any,
    allow_unbound: bool,
) -> Dict[str, Any]:
    """Refuse to evaluate a model against data it was not trained under.

    A report is only evidence about the model it names. Nothing previously
    stopped the evaluator loading a different dataset, a different split or a
    later cutoff than the run that produced the checkpoint, and reporting the
    result as though it described that model.

    Fingerprints, the resolved split and the cutoff must all agree. Any
    disagreement is a hard stop rather than a warning, because a mismatched
    evaluation is not merely imprecise - it measures something else.
    """
    if provenance.get("status") == "UNBOUND_MODEL_ALLOWED_FOR_DIAGNOSTICS":
        return {"bound": False, "reason": "unbound diagnostic run", "mismatches": []}

    recorded = provenance.get("datasets") or {}
    mismatches: List[str] = []

    def _compare(role: str, actual: Dict[str, Any]) -> None:
        want = (recorded.get(role) or {}).get("fingerprint")
        got = actual.get("dataset_fingerprint")
        if want and got and str(want) != str(got):
            mismatches.append(
                f"{role} dataset fingerprint differs: model trained on {want}, "
                f"evaluation loaded {got}"
            )

    _compare("loaded", loaded_manifest)
    _compare("train", train_manifest)
    _compare("holdout", holdout_manifest)

    args = provenance.get("arguments") or {}
    recorded_split = args.get("resolved_holdout_split_at")
    if recorded_split and str(recorded_split) != str(resolved_split_at):
        mismatches.append(
            f"holdout split differs: model used {recorded_split}, "
            f"evaluation resolved {resolved_split_at}"
        )

    recorded_cutoff = args.get("data_cutoff")
    if recorded_cutoff and str(recorded_cutoff) != str(data_cutoff):
        mismatches.append(
            f"data cutoff differs: model used {recorded_cutoff}, "
            f"evaluation used {data_cutoff}"
        )

    if mismatches and not allow_unbound:
        bullet = chr(10) + "  - "
        raise ValueError(
            "evaluation is not bound to this model's provenance:"
            + bullet
            + bullet.join(mismatches)
        )

    return {
        "bound": not mismatches,
        "reason": "matched" if not mismatches else "mismatched (allowed)",
        "mismatches": mismatches,
    }


def training_end_from_provenance(provenance: Dict[str, Any]) -> Optional[str]:
    """The last bar the model actually trained on.

    Deriving this from the current CSV maximum is wrong the moment new broker
    bars are appended: `last_trained` jumps forward, the external section finds
    nothing after it, and a three-way comparison silently becomes two-way. The
    model's own record is the only correct source.
    """
    train = (provenance.get("datasets") or {}).get("train") or {}
    boundary = train.get("boundary")
    if boundary:
        return str(boundary)
    frames = train.get("frames") or []
    lasts = [
        str(f["last_time"])
        for f in frames
        if isinstance(f, dict) and f.get("last_time")
    ]
    return max(lasts) if lasts else None


def assert_no_training_overlap(
    dataset: Dict[str, Dict[str, Any]],
    instrument: str,
    training_end: Optional[str],
) -> None:
    """An evaluation bar at or before the training end is not out-of-sample."""
    if not training_end:
        return
    frames = dataset.get(instrument) or {}
    df = frames.get("M15")
    if df is None or getattr(df, "empty", True) or "time" not in getattr(df, "columns", []):
        return

    import pandas as pd

    boundary = pd.Timestamp(training_end)
    if boundary.tzinfo is None:
        boundary = boundary.tz_localize("UTC")
    times = pd.to_datetime(df["time"], utc=True)
    overlapping = int((times <= boundary).sum())
    if overlapping:
        raise ValueError(
            f"{overlapping} evaluation bars fall at or before the model's training "
            f"end ({training_end}); those bars are in-sample and cannot support an "
            f"out-of-sample claim"
        )


def format_continuous(replays: List[Dict[str, Any]]) -> str:
    """Render the single-account product, where prop-firm limits actually apply."""
    if not replays:
        return ""
    nl = chr(10)
    head = "    %-16s %9s %9s %10s %9s %8s  %s" % (
        "strategy", "return%", "net%", "worstDD%", "dailyDD%", "trades", "outcome")
    lines = ["", "  continuous single-account replay (challenge survival):", head,
             "    " + "-" * (len(head) - 4)]
    for r in replays:
        outcome = "SURVIVED" if r.get("survived") else f"BREACHED ({r.get('terminated_reason')})"
        lines.append("    %-16s %9.2f %9.2f %10.2f %9.2f %8d  %s" % (
            r.get("policy", "?"), r.get("gross_return_pct", 0.0), r.get("net_return_pct", 0.0),
            r.get("worst_total_drawdown_pct", 0.0), r.get("worst_daily_drawdown_pct", 0.0),
            int(r.get("trades", 0)), outcome))
    return nl.join(lines)


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
        f"{'R:R':>7}{'maxDD%':>8}{'trades':>8}{'t/day':>7}{'hold':>6}  window statistics  "
    )
    lines = [f"\n{title}", "=" * len(head), head, "-" * len(head)]
    for r in results:
        pf = "n/a" if not np.isfinite(r.profit_factor) else f"{r.profit_factor:.2f}"
        lines.append(
            f"{r.name:<16}{r.return_pct:>9.2f}{pf:>7}{r.win_rate:>7.1f}"
            f"{r.expectancy_r:>8.3f}{r.reward_risk:>7.2f}{r.max_drawdown_pct:>8.2f}"
            f"{r.trades:>8}{r.trades_per_day:>7.1f}{r.median_bars_held:>6.0f}  "
            f"{r.window_statistics_summary}"
        )
    return "\n".join(lines)


def load_ftmo_data(
    ftmo_dir: str,
    instrument: str,
    after: Optional[Any],
    cutoff: Optional[Any] = None,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Broker bars strictly after the processed dataset ends, validated and frozen."""
    import pandas as pd

    d = Path(ftmo_dir)
    if not d.exists():
        return None

    frames: Dict[str, Any] = {}
    fingerprints: Dict[str, Any] = {}
    for tf in ("M15", "H1", "H4", "D1"):
        f = d / f"{instrument}_{tf}.csv"
        if not f.exists():
            continue
        df: pd.DataFrame = pd.read_csv(f)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        if df["time"].isna().any():
            raise ValueError(f"{f} contains invalid timestamps")

        df = df.sort_values(by="time", ignore_index=True)
        _validate_broker_frame(df, source=f, timeframe=tf)

        # A frozen upper bound, so a report is reproducible after more bars are
        # pulled. Without it the same command produces a different dataset
        # tomorrow and the two results are quietly incomparable.
        if cutoff is not None:
            df = df.loc[df["time"] <= pd.Timestamp(cutoff)]
        if after is not None:
            df = df.loc[df["time"] > after]

        df = df.reset_index(drop=True)
        if len(df) < 60:
            logger.warning("%s only %d bars after %s - skipping", tf, len(df), after)
            continue
        frames[tf] = df
        fingerprints[tf] = _frame_fingerprint(df)

    if "M15" not in frames:
        return None

    # Fingerprints are returned alongside the data, never inside it: the env
    # iterates the top-level dict as instruments, so an extra key there would be
    # loaded as a tradeable symbol.
    BROKER_SLICE_FINGERPRINTS.clear()
    BROKER_SLICE_FINGERPRINTS.update(fingerprints)
    return {instrument: frames}


def _validate_broker_frame(df: Any, *, source: Any, timeframe: str) -> None:
    """The checks the main loader applies, applied here too.

    Broker bars were being trusted on arrival while historical bars were
    validated - so a duplicated, out-of-order or non-finite row from MT5 would
    have entered evaluation silently.
    """
    import pandas as pd

    if df.empty:
        return

    dupes = int(df["time"].duplicated().sum())
    if dupes:
        raise ValueError(f"{source}: {dupes} duplicate timestamps in {timeframe}")

    if not df["time"].is_monotonic_increasing:
        raise ValueError(f"{source}: {timeframe} timestamps are not chronological")

    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            raise ValueError(f"{source}: {timeframe} is missing {col}")
        values = pd.to_numeric(df[col], errors="coerce")
        if not values.notna().all():
            raise ValueError(f"{source}: {timeframe}.{col} contains non-numeric values")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{source}: {timeframe}.{col} contains non-finite values")
        if (values <= 0).any():
            raise ValueError(f"{source}: {timeframe}.{col} contains non-positive prices")

    if (df["high"] < df["low"]).any():
        raise ValueError(f"{source}: {timeframe} has bars where high < low")

    if "spread" in df.columns:
        spread = pd.to_numeric(df["spread"], errors="coerce")
        if (spread < 0).any():
            raise ValueError(f"{source}: {timeframe} has negative spread")


def _frame_fingerprint(df: Any) -> Dict[str, Any]:
    """Identify the exact slice used, so a report names its own inputs."""
    if df.empty:
        return {"rows": 0, "sha256": None}
    payload = f"{len(df)}|{df['time'].iloc[0]}|{df['time'].iloc[-1]}|{float(df['close'].iloc[-1])}"
    return {
        "rows": int(len(df)),
        "first_time": str(df["time"].iloc[0]),
        "last_time": str(df["time"].iloc[-1]),
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
    }


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
        "--continuous-bars", type=int, default=3000,
        help="Bars for the single-account chronological replay, the product "
             "prop-firm limits actually apply to.",
    )
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

    # Refuse to evaluate a model against data it was not trained under. A report
    # is only evidence about the model it names.
    binding = bind_evaluation_to_provenance(
        model_provenance,
        loaded_manifest=build_dataset_manifest(
            data, role="holdout_report_loaded", boundary=args.data_cutoff
        ),
        train_manifest=build_dataset_manifest(
            _train, role="holdout_report_train", boundary=split_ts
        ),
        holdout_manifest=build_dataset_manifest(
            holdout, role="holdout_report_holdout", boundary=split_ts
        ),
        resolved_split_at=split_ts,
        data_cutoff=args.data_cutoff,
        allow_unbound=bool(args.allow_unbound_model),
    )
    if binding["mismatches"]:
        for line in binding["mismatches"]:
            logger.warning("PROVENANCE MISMATCH (allowed): %s", line)

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
        "artifact_integrity": {},  # filled once the datasets are resolved
        "provenance_binding": binding,
    }
    sections = []

    datasets = [
        ("DEVELOPMENT HOLDOUT (original prices)", holdout, 0.0),
        ("DEVELOPMENT HOLDOUT MIRRORED (short-side diagnostic)", holdout, 1.0),
    ]

    # The model's own record of where its training data ended, not the current
    # CSV maximum. Deriving it from the loaded frame is wrong the moment broker
    # bars are appended: last_trained jumps to the newest bar, load_ftmo_data
    # finds nothing after it, and the external comparison silently disappears -
    # which is exactly what happened once the merge was added.
    provenance_end = training_end_from_provenance(model_provenance)
    m15 = _primary_frame(next(iter(data.values())))
    csv_end = m15["time"].max() if m15 is not None and "time" in m15.columns else None

    last_trained = provenance_end if provenance_end else csv_end
    report["training_end"] = {
        "value": str(last_trained) if last_trained is not None else None,
        "source": "model_provenance" if provenance_end else "loaded_csv_fallback",
        "loaded_csv_max": str(csv_end) if csv_end is not None else None,
    }
    if provenance_end is None and model_provenance.get("status") not in (
        "NO_MODEL_BASELINES_ONLY",
        "UNBOUND_MODEL_ALLOWED_FOR_DIAGNOSTICS",
    ):
        raise ValueError(
            "model provenance carries no training end; the external section "
            "cannot be shown to be out-of-sample"
        )

    ftmo = load_ftmo_data(args.ftmo_dir, args.instrument, last_trained)
    if ftmo is not None:
        n = len(ftmo[args.instrument]["M15"])
        logger.info("FTMO live bars after %s: %d M15", last_trained, n)
        # An external section that overlaps training is not external.
        assert_no_training_overlap(ftmo, args.instrument, last_trained)
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

        # The second, separate product: one account walked chronologically.
        # Survival is a property of a single account over calendar time and
        # cannot be read off a mean over reset windows.
        replays = []
        if episode_starts:
            for p_ in build_policies(model):
                replay_cfg = copy.deepcopy(cfg)
                replay_cfg.max_steps_per_episode = int(args.continuous_bars)
                replays.append(
                    run_continuous_replay(
                        p_, dset, replay_cfg,
                        start_index=int(episode_starts[0]),
                        max_bars=int(args.continuous_bars),
                        instrument=args.instrument,
                        contract=contract,
                    )
                )
        report.setdefault("continuous_replay", {})[label] = replays

        sections.append(
            format_table(results, label)
            + format_by_regime(results)
            + format_continuous(replays)
        )
        report["sections"][label] = [
            vars(r)
            | {
                "window_statistics_summary": r.window_statistics_summary
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

    # Built here, not at report construction: the broker slice fingerprints are
    # only known once load_ftmo_data has run.
    report["artifact_integrity"] = build_artifact_integrity(args, contract)

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
