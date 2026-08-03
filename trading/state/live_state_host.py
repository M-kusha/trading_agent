"""Live observation state, built by the training environment itself.

WHY NOT AN EXTRACTED LIBRARY
----------------------------
The plan originally called for lifting envs/prop_firm/{signals,observation} into
a shared package. Measuring the coupling first showed why that is the wrong
shape: those mixins reference **73 attributes they do not define** -
`self.config` (20 uses), `self.position` (8), `self.current_step` (7), plus
methods such as `_calc_dds`, `_get_ohlcv`, `_get_bar_dt`, `_atr_vol_proxy`,
`_in_prime_window` and `_tf_minutes`.

They are not a library. They are environment internals. Extracting them means
either converting 73 dependencies into explicit parameters, or reproducing most
of the environment inside the live agent - and the second is exactly the
duplication that produced the original train/live divergence.

THE APPROACH
------------
PropFirmTradingEnv already accepts a plain `Dict[instrument, Dict[timeframe,
DataFrame]]`. Nothing in it requires the frames to come from a CSV. So live
maintains a rolling window of MT5 bars in that same shape, hands it to the same
class, and points `current_step` at the newest bar.

Live and training then share one implementation by construction rather than by
convention: there is no second code path to drift.

    host = LiveStateHost(instrument="XAUUSD")
    host.update({"M15": m15_df, "H1": h1_df, "H4": h4_df, "D1": d1_df})
    inputs = host.observation_inputs()      # -> PPOObservationBuilder.build(**inputs)
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from envs.core.env_types import PropFirmConfig, PropPosition
from envs.core.execution_model import ExecutionModel
from envs.prop_firm_env import PropFirmTradingEnv

# Bars each timeframe must supply. The observation contract requires 60 M15 and
# 30 HTF; the environment itself requests M15 120, H1 60, H4 40, D1 40. These
# are those requests with margin, so a short window fails here - with a clear
# message - rather than deeper inside the builder.
REQUIRED_BARS: Dict[str, int] = {"M15": 150, "H1": 80, "H4": 60, "D1": 60}


class LiveStateHost:
    """Feeds live bars to the training environment to produce observation inputs."""

    def __init__(
        self,
        instrument: str = "XAUUSD",
        config: Optional[PropFirmConfig] = None,
    ) -> None:
        self.instrument = instrument
        self.config = config or PropFirmConfig(
            live_mode=True,
            raw_evaluation_mode=True,
            domain_randomization_enabled=False,
            mirror_augmentation_prob=0.0,
            high_vol_oversample_prob=0.0,
        )
        self._env: Optional[PropFirmTradingEnv] = None
        self._bars: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------ data

    def update(self, frames: Dict[str, pd.DataFrame]) -> None:
        """Replace the rolling window with the latest bars.

        `frames` maps timeframe -> DataFrame with lowercase OHLCV columns and a
        `time` column, exactly the shape train_prop_firm.load_market_data
        produces. Rebuilding the env is cheap relative to a live bar interval
        and keeps the data path identical to training.
        """
        missing = [tf for tf in REQUIRED_BARS if tf not in frames]
        if missing:
            raise ValueError(f"LiveStateHost missing timeframes: {', '.join(missing)}")

        short = {
            tf: len(frames[tf])
            for tf, need in REQUIRED_BARS.items()
            if len(frames[tf]) < need
        }
        if short:
            detail = ", ".join(f"{tf}={n} (need {REQUIRED_BARS[tf]})" for tf, n in short.items())
            raise ValueError(f"LiveStateHost has insufficient history: {detail}")

        prepared: Dict[str, pd.DataFrame] = {}
        for tf, frame in frames.items():
            df = frame.copy()
            if isinstance(df.index, pd.DatetimeIndex):
                idx = pd.to_datetime(df.index, utc=True, errors="coerce")
                # Index.isna() returns an ndarray at runtime; the type stub
                # claims Index[bool], which has no .any(). np.asarray states the
                # actual type without changing behaviour.
                if bool(np.asarray(idx.isna()).any()):
                    raise ValueError(f"LiveStateHost {tf} contains invalid timestamps")
                df.index = idx
                index_name = df.index.name or "time"
                df.index.name = index_name
                df = df.reset_index(drop=False)
                if index_name != "time" and "time" not in df.columns:
                    df = df.rename(columns={index_name: "time"})
            elif "time" in df.columns:
                parsed = pd.to_datetime(df["time"], utc=True, errors="coerce")
                if parsed.isna().any():
                    raise ValueError(f"LiveStateHost {tf} contains invalid timestamps")
                df["time"] = parsed
            else:
                raise ValueError(f"LiveStateHost {tf} has no UTC timestamp column/index")
            prepared[tf] = df.reset_index(drop=True)

        m15 = prepared["M15"]
        if "spread" not in m15.columns:
            raise ValueError("LiveStateHost M15 is missing executable spread")
        recent_spread = pd.to_numeric(m15["spread"].tail(20), errors="coerce").to_numpy(dtype=float)
        if recent_spread.size < 20 or not np.all(np.isfinite(recent_spread)) or np.any(recent_spread <= 0.0):
            raise ValueError("LiveStateHost M15 has missing/non-positive recent spread")

        self._bars = prepared
        self._env = PropFirmTradingEnv({self.instrument: self._bars}, self.config)
        self._env._episode_instrument = self.instrument
        # Point at the newest bar. reset() is deliberately not called: it would
        # sample a random episode start, which is a training concern.
        self._env.current_step = len(self._bars["M15"]) - 1
        # reset() is a training operation and cannot run on a short rolling
        # live window, but open-position account features still need the same
        # executable bid/ask model used in training. Initialize only that
        # deterministic component; no episode sampling or augmentation occurs.
        exec_cfg = self._env._build_episode_execution_config()
        self._env._episode_execution_cfg = exec_cfg
        self._env._exec = ExecutionModel(exec_cfg, np.random.default_rng(0))

    # ------------------------------------------------------------ observation

    def observation_inputs(self) -> Dict[str, Any]:
        """The exact keyword arguments PPOObservationBuilder.build() expects."""
        env = self._require_env()
        instrument = self.instrument
        return {
            "market_data": env._prepare_market_data(instrument),
            "expert_signals": env._prepare_expert_signals(instrument),
            "risk_state": env._prepare_risk_state(),
            "account_state": env._prepare_account_state(instrument),
            "trading_mode_state": env._prepare_trading_mode_state(instrument),
            "governor_state": env._get_governor_state(),
            # v8.0. Both paths call the same env producer, so live and training
            # cannot drift apart on session or spread - which is the whole point
            # of routing live through the env rather than reimplementing it.
            "session_state": env._prepare_session_state(instrument),
        }

    def sync_account(
        self,
        balance: float,
        equity: float,
        position: Any = None,
        daily_trades: int = 0,
        consecutive_losses: int = 0,
        initial_balance: Optional[float] = None,
        day_start_balance: Optional[float] = None,
        peak_balance: Optional[float] = None,
    ) -> None:
        """Mirror broker account state onto the host.

        Without this the observation's account and risk blocks describe a fresh
        simulated account rather than the live one.
        """
        env = self._require_env()
        env.balance = float(balance)
        env.equity = float(equity)
        if initial_balance is not None:
            env.config.initial_balance = float(initial_balance)
        if day_start_balance is not None:
            env.day_start_balance = float(day_start_balance)
        if peak_balance is not None:
            env.peak_balance = float(peak_balance)
        risk_values = {
            "balance": env.balance,
            "equity": env.equity,
            "initial_balance": env.config.initial_balance,
            "day_start_balance": env.day_start_balance,
            "peak_balance": env.peak_balance,
        }
        if any(not np.isfinite(float(value)) for value in risk_values.values()):
            raise ValueError(f"non-finite live account snapshot: {risk_values}")
        if any(float(risk_values[key]) <= 0.0 for key in risk_values):
            raise ValueError(f"non-positive live account anchor: {risk_values}")
        env.position = self._coerce_position(position) if position is not None else None
        env.daily_trades = int(daily_trades)
        env.consecutive_losses = int(consecutive_losses)

    def _coerce_position(self, raw: Any) -> PropPosition:
        if isinstance(raw, PropPosition):
            return raw
        if not isinstance(raw, dict):
            raise ValueError(f"unsupported live position payload: {type(raw).__name__}")

        instrument = str(raw.get("instrument", raw.get("symbol", self.instrument)) or self.instrument)
        side_value = raw.get("side")
        if side_value is None:
            label = str(raw.get("direction", raw.get("type", raw.get("action", "")))).lower()
            side = 1 if label in ("long", "buy") else (-1 if label in ("short", "sell") else 0)
        else:
            try:
                side = 1 if float(side_value) > 0.0 else (-1 if float(side_value) < 0.0 else 0)
            except (TypeError, ValueError):
                side = 0
        if side == 0:
            raise ValueError(f"live position has no valid direction: {raw!r}")

        def _number(*keys: str) -> float:
            for key in keys:
                if key in raw and raw[key] is not None:
                    try:
                        value = float(raw[key])
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        return value
            return 0.0

        entry = _number("entry_price", "open_price", "price_open")
        lots = _number("lots", "lot_size", "volume")
        contract = _number("contract_size")
        if contract <= 0.0:
            contract = 100.0 if "XAU" in instrument.upper() or "GOLD" in instrument.upper() else 1.0
        if lots <= 0.0:
            units = _number("units")
            lots = units / contract if units > 0.0 else 0.0
        stop = _number("sl", "stop_price", "stop_loss")
        if not (entry > 0.0 and lots > 0.0 and stop > 0.0):
            raise ValueError(
                f"live position lacks executable entry/size/stop: entry={entry}, lots={lots}, stop={stop}"
            )

        initial_risk = _number("initial_risk_eur")
        if initial_risk <= 0.0:
            initial_risk = abs(entry - stop) * contract * lots
        if not (np.isfinite(initial_risk) and initial_risk > 0.0):
            raise ValueError(f"live position has invalid stop risk: {initial_risk!r}")

        raw_time = raw.get("open_time", raw.get("entry_time"))
        entry_dt = None
        if raw_time is not None:
            try:
                if isinstance(raw_time, (int, float, np.integer, np.floating)):
                    stamp = pd.Timestamp(float(raw_time), unit="s", tz="UTC")
                else:
                    stamp = pd.Timestamp(raw_time)
                    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
                entry_dt = stamp.to_pydatetime()
            except Exception as exc:
                raise ValueError(f"live position has invalid entry time {raw_time!r}") from exc

        return PropPosition(
            instrument=instrument,
            direction="long" if side > 0 else "short",
            entry_price=entry,
            entry_dt=entry_dt,
            entry_bar=int(self._require_env().current_step),
            lot_size=lots,
            initial_risk_eur=initial_risk,
            stop_price=stop,
            stop_distance_price=abs(entry - stop),
        )

    @property
    def env(self) -> PropFirmTradingEnv:
        return self._require_env()

    def _require_env(self) -> PropFirmTradingEnv:
        if self._env is None:
            raise RuntimeError(
                "LiveStateHost has no data. Call update() with the current bar "
                "window before requesting observation inputs - an observation "
                "built from absent market data is not tradeable."
            )
        return self._env
