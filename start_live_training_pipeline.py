#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, cast

import numpy as np

logger = logging.getLogger("LiveTrainingPipeline")
running = True


def _signal_handler(signum, frame):
    global running
    logger.info("Received signal %s, shutting down...", signum)
    running = False


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _resolve_model_path(explicit_path: Optional[str]) -> Optional[str]:
    root = _project_root()

    def _fix_zip_suffix(p: Path) -> Path:
        name = p.name
        if name.lower().endswith(".zip.zip"):
            return p.with_name(name[:-4])
        return p

    if explicit_path:
        p = Path(os.path.expandvars(os.path.expanduser(str(explicit_path))))
        if not p.is_absolute():
            p = root / p
        p = _fix_zip_suffix(p)
        if p.exists():
            return str(p)
        if not p.name.lower().endswith(".zip"):
            p_zip = _fix_zip_suffix(p.with_name(p.name + ".zip"))
            if p_zip.exists():
                return str(p_zip)

    candidates = [
        "models/best/best_model.zip",
        "models/propfirm/best/best_model.zip",
        "models/curriculum/best/best_model.zip",
        "models/curriculum/propfirm/best/best_model.zip",
        "models/propfirm/propfirm_ppo_final.zip",
        "models/ppo_trading_model.zip",
        "models/ppo_final_model.zip",
        "models/modern_ppo_final.zip",
    ]
    for cand in candidates:
        p = _fix_zip_suffix(root / cand)
        if p.exists():
            return str(p)
    return None


def _parse_csv_list(raw: str) -> List[str]:
    parts = [p.strip() for p in (raw or "").split(",")]
    return [p for p in parts if p]


def _to_internal_symbol(symbol: str) -> str:
    s = str(symbol or "").strip()
    if len(s) == 6 and "/" not in s:
        return f"{s[:3]}/{s[3:]}"
    return s


def _safe_symbol_slug(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("\\", "").replace(" ", "_")


def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    try:
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()

    try:
        to_py = getattr(obj, "to_pydatetime", None)
        if callable(to_py):
            dt = to_py()
            if isinstance(dt, datetime):
                return dt.isoformat()
    except Exception:
        pass
    return str(obj)


def _hash_obj(obj: Any) -> str:
    try:
        payload = json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":")).encode("utf-8")
    except Exception:
        payload = str(obj).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _safe_close(df: Any, i: int) -> Optional[float]:
    try:
        cols = getattr(df, "columns", None)
        if cols is not None:
            for c in ("close", "Close", "CLOSE", "c"):
                try:
                    if c in cols:
                        return float(df[c].iloc[i])
                except Exception:
                    continue

        row = df.iloc[i]


        if isinstance(row, Mapping):
            for c in ("close", "Close", "CLOSE", "c"):
                if c in row:
                    try:
                        return float(row[c])  # type: ignore[index]
                    except Exception:
                        pass


        to_dict_fn = getattr(row, "to_dict", None)
        if callable(to_dict_fn):
            try:
                d = to_dict_fn()
                if isinstance(d, Mapping):
                    for c in ("close", "Close", "CLOSE", "c"):
                        if c in d:
                            return float(d[c])  # type: ignore[index]
            except Exception:
                pass


        try:
            return float(cast(Any, row)[-1])
        except Exception:
            return None

    except Exception:
        return None


@dataclass(frozen=True)
class BarSignature:
    tf: str
    closed_ts: str
    forming_ts: str
    closed_close: Optional[float]
    forming_close: Optional[float]


def _make_bar_signature(tf: str, df: Any) -> Optional[BarSignature]:
    try:
        if df is None or getattr(df, "empty", True):
            return None
        idx = getattr(df, "index", None)
        if idx is None or len(idx) < 2:
            return None
        closed_ts = str(idx[-2])
        forming_ts = str(idx[-1])
        closed_close = _safe_close(df, -2)
        forming_close = _safe_close(df, -1)
        return BarSignature(
            tf=tf,
            closed_ts=closed_ts,
            forming_ts=forming_ts,
            closed_close=closed_close,
            forming_close=forming_close,
        )
    except Exception:
        return None


def _summarize_live_data(market_data: Dict[str, Any], instrument: str, timeframes: List[str]) -> str:
    blocks = market_data.get(instrument) if isinstance(market_data, dict) else None
    if not isinstance(blocks, dict):
        return "no market_data"
    parts: List[str] = []
    for tf in timeframes:
        df = blocks.get(tf)
        if df is None or getattr(df, "empty", True):
            parts.append(f"{tf}:missing")
            continue
        try:
            last_ts = df.index[-1]
            parts.append(f"{tf}:bars={len(df)} last={last_ts}")
        except Exception:
            parts.append(f"{tf}:bars=?")
    return " | ".join(parts)


def _format_primary_experts(expert_signals: Dict[str, Any]) -> str:
    experts = expert_signals.get("experts", {}) if isinstance(expert_signals, dict) else {}
    if not isinstance(experts, dict) or not experts:
        return "Experts: (none)"

    def _fmt_one(name: str) -> str:
        block = experts.get(name, {})
        if not isinstance(block, dict):
            return f"{name}=?"
        direction = str(block.get("direction", "neutral"))
        score = float(block.get("score", 0.0) or 0.0)
        conf = float(block.get("confidence", 0.5) or 0.5)
        return f"{name}={direction} score={score:+.2f} conf={conf:.2f}"

    ordered = ["trend", "momentum", "theme", "seasonality"]
    items = [_fmt_one(k) for k in ordered if k in experts]
    if not items:

        items = [_fmt_one(k) for k in list(experts.keys())[:4]]
    return "Experts(M15): " + " | ".join(items)


def _format_htf_experts(expert_signals: Dict[str, Any]) -> str:
    htf = expert_signals.get("htf_experts", {}) if isinstance(expert_signals, dict) else {}
    if not isinstance(htf, dict) or not htf:
        return "HTF: (none)"
    parts: List[str] = []
    for tf in ["H1", "H4", "D1"]:
        b = htf.get(tf, {})
        if not isinstance(b, dict):
            continue
        td = str(b.get("trend_direction", "neutral"))
        ts = float(b.get("trend_strength", 0.0) or 0.0)
        md = str(b.get("momentum_direction", "neutral"))
        ms = float(b.get("momentum_strength", 0.0) or 0.0)
        rsi = float(b.get("rsi", 50.0) or 50.0)
        adx = float(b.get("adx", 0.0) or 0.0)
        struct = float(b.get("structure_bias", 0.0) or 0.0)
        parts.append(f"{tf}: trend={td}({ts:.2f}) mom={md}({ms:.2f}) rsi={rsi:.1f} adx={adx:.1f} struct={struct:+.1f}")
    return "HTF: " + " | ".join(parts) if parts else "HTF: (none)"


def _format_committee(committee_state: Dict[str, Any]) -> str:
    if not isinstance(committee_state, dict) or not committee_state:
        return "Committee: (none)"
    action = str(committee_state.get("action", "flat"))
    score = float(committee_state.get("score", 0.0) or 0.0)
    conf = float(committee_state.get("confidence", 0.5) or 0.5)
    agree = float(committee_state.get("agreement", 0.5) or 0.5)
    frag = float(committee_state.get("fragility", 0.5) or 0.5)
    return f"Committee: action={action} score={score:+.2f} conf={conf:.2f} agree={agree:.2f} frag={frag:.2f}"


def _format_trading_timing(trading_mode_state: Dict[str, Any]) -> str:
    if not isinstance(trading_mode_state, dict) or not trading_mode_state:
        return "Timing: (none)"
    entry = trading_mode_state.get("entry_timing", {})
    if not isinstance(entry, dict):
        entry = {}
    allowed = bool(entry.get("entry_allowed", True))
    ql = float(entry.get("entry_quality_long", 0.0) or 0.0)
    qs = float(entry.get("entry_quality_short", 0.0) or 0.0)
    vol_state = str(entry.get("vol_state", "unknown"))
    zone = str(entry.get("zone_type", "unknown"))
    prime = bool(float(entry.get("in_prime_window", 0.0) or 0.0) >= 0.5)
    hour = float(entry.get("hour_normalized", 0.5) or 0.5)
    return f"Timing: entry_allowed={allowed} q_long={ql:.2f} q_short={qs:.2f} vol={vol_state} zone={zone} prime={prime} hour={hour:.2f}"


def _format_risk_line(
    *,
    current_dd: float,
    daily_dd: float,
    daily_trades: int,
    session_trades: int,
    consecutive_losses: int,
    on_cooldown: bool,
    env_cfg: Any,
) -> str:
    try:
        max_dd_thr = float(env_cfg.max_drawdown_limit) - float(env_cfg.max_dd_safety_buffer)
        daily_dd_thr = float(env_cfg.daily_drawdown_limit) - float(env_cfg.daily_dd_safety_buffer)
    except Exception:
        max_dd_thr, daily_dd_thr = 0.0, 0.0
    return (
        f"Risk: dd={current_dd:.2%} (thr={max_dd_thr:.2%}) "
        f"daily_dd={daily_dd:.2%} (thr={daily_dd_thr:.2%}) "
        f"trades_today={daily_trades}/{int(getattr(env_cfg, 'max_trades_per_day', 0) or 0)} "
        f"session={session_trades}/{int(getattr(env_cfg, 'max_trades_per_session', 0) or 0)} "
        f"consec_losses={consecutive_losses}/{int(getattr(env_cfg, 'max_consecutive_losses', 0) or 0)} "
        f"cooldown={'ON' if on_cooldown else 'off'}"
    )


def _format_position_line(pos: Optional["_LivePosition"], now_dt: datetime) -> str:
    if pos is None:
        return "Position: FLAT"
    side = "LONG" if pos.side > 0 else "SHORT"
    age_s = None
    if pos.open_time:
        try:
            age_s = max(0.0, (now_dt - pos.open_time).total_seconds())
        except Exception:
            age_s = None
    if age_s is not None:
        return f"Position: {side} {pos.lots:.2f} lots pnl={pos.profit:+.2f} age={age_s/3600.0:.2f}h"
    return f"Position: {side} {pos.lots:.2f} lots pnl={pos.profit:+.2f}"


def _format_mask_line(
    *,
    mask_builder: Any,
    action_mask: Optional[np.ndarray],
    enforce_hard_rules: bool,
    hard_reasons: List[str],
) -> str:
    if action_mask is None:
        return "Mask: (none)"
    try:
        summary = mask_builder.get_mask_summary(action_mask)
    except Exception:
        summary = {}
    if isinstance(summary, dict) and summary:
        base = (
            f"Mask: hold={summary.get('hold_allowed')} close={summary.get('close_allowed')} "
            f"long_allowed={summary.get('long_allowed')} short_allowed={summary.get('short_allowed')} "
            f"allowed={summary.get('total_allowed')}/{summary.get('total_actions')}"
        )
    else:
        base = f"Mask: allowed={int(action_mask.sum())}/{len(action_mask)}"
    if enforce_hard_rules:
        if hard_reasons:
            return base + f" | hard_rules=ON blocked_by={','.join(hard_reasons[:4])}"
        return base + " | hard_rules=ON"
    return base + " | hard_rules=off"


@dataclass
class _LivePosition:
    symbol: str
    side: int
    lots: float
    profit: float
    open_time: Optional[datetime]


def _get_live_position(
    mt5: Any,
    symbol: str,
    *,
    tz: Optional[Any] = None,
) -> Optional[_LivePosition]:
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return None
        pos = positions[0]
        side = 1 if int(getattr(pos, "type", 0)) == 0 else -1
        lots = float(getattr(pos, "volume", 0.0) or 0.0)
        profit = float(getattr(pos, "profit", 0.0) or 0.0)
        open_ts = getattr(pos, "time", None)
        open_dt = None
        if open_ts:
            try:
                open_utc = datetime.fromtimestamp(int(open_ts), tz=timezone.utc)
                if tz is not None:
                    open_dt = open_utc.astimezone(tz).replace(tzinfo=None)
                else:
                    open_dt = open_utc.replace(tzinfo=None)
            except Exception:
                open_dt = None
        return _LivePosition(symbol=symbol, side=side, lots=lots, profit=profit, open_time=open_dt)
    except Exception:
        return None


def _build_account_state(
    *,
    balance: float,
    equity: float,
    initial_balance: float,
    current_dd: float,
    trades_today: int,
    win_rate: float,
    position: Optional[_LivePosition],
    time_in_position_bars: float,
    on_cooldown: bool,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "balance": float(balance),
        "equity": float(equity),
        "initial_balance": float(initial_balance),
        "current_drawdown": float(current_dd),
        "win_rate": float(win_rate),
        "trades_today": int(trades_today),
        "has_position": bool(position is not None),
        "position_direction": 0.0,
        "position_size": 0.0,
        "unrealized_pnl": 0.0,
        "time_in_position": float(time_in_position_bars),
        "on_cooldown": 1.0 if on_cooldown else 0.0,
    }
    if position is not None:
        state["position_direction"] = 1.0 if position.side > 0 else -1.0
        state["position_size"] = float(position.lots)
        state["unrealized_pnl"] = float(position.profit)
    return state


def _translate_decision_to_intent(
    *,
    intent: str,
    size_mult: float,
    position: Optional[_LivePosition],
) -> Optional[Dict[str, Any]]:
    intent = (intent or "").lower().strip()
    strength = float(np.clip(size_mult, 0.0, 1.0))

    if intent == "hold":
        return None

    if intent == "close":
        if position is None:
            return None
        return {
            "id": f"ppo-{uuid.uuid4().hex[:10]}",
            "instrument": position.symbol,
            "action": "close",
            "confidence": 1.0,
            "intensity": 1.0,
            "size_eur": 0.0,
        }

    if intent in ("long", "short"):
        if position is not None:
            same_side = (intent == "long" and position.side > 0) or (intent == "short" and position.side < 0)
            if same_side:
                return None
        return {
            "id": f"ppo-{uuid.uuid4().hex[:10]}",
            "instrument": position.symbol if position else "",
            "action": "open_long" if intent == "long" else "open_short",
            "confidence": 1.0,
            "intensity": max(0.1, strength),
            "size_eur": 0.0,
        }

    return None


async def _run_orchestrator_mode() -> int:
    cmd = [sys.executable, str(_project_root() / "start_live_trading.py")]
    logger.info("Delegating to orchestrator runner: %s", " ".join(cmd))
    try:
        proc = await asyncio.create_subprocess_exec(*cmd)
        return await proc.wait()
    except Exception as e:
        logger.error("Failed to start orchestrator runner: %s", e)
        return 1


async def _run_training_mode(args: argparse.Namespace) -> int:

    try:
        import MetaTrader5 as _MT5  # type: ignore
        mt5: Any = cast(Any, _MT5)
    except Exception as e:
        logger.error("MetaTrader5 import failed: %s", e)
        return 1

    from envs.core.env_types import PropFirmConfig
    from envs.prop_firm_env import PropFirmTradingEnv
    from live.live_connector import LiveDataConnector
    from modules.executor.executor import Executor
    from modules.meta.live_action_mask import LiveActionMaskBuilder, LiveMaskConfig
    from modules.meta.ppo_core import PPOCore, PPOCoreConfig
    from modules.meta.ppo_observation_builder import PPOObservationBuilder
    from modules.utils.info_bus import InfoBusManager

    Path("logs").mkdir(exist_ok=True)
    Path("state").mkdir(exist_ok=True)

    os.environ["EXECUTION_MODE"] = "live"
    os.environ["TRADING_MODE"] = "live"


    try:
        from modules.core.trading_mode import TradingModeManager

        TradingModeManager.set_mode("LIVE")
    except Exception:
        pass

    instruments = _parse_csv_list(args.instruments)
    if not instruments:
        instruments = ["XAUUSD"]

    mt5_instruments = list(instruments)
    primary_mt5_symbol = mt5_instruments[0]


    instruments = [_to_internal_symbol(s) for s in mt5_instruments]
    primary_instrument = instruments[0]

    timeframes = _parse_csv_list(args.timeframes)
    if not timeframes:
        timeframes = ["M15", "H1", "H4", "D1"]


    bus = InfoBusManager.get_instance()
    bus.set(
        "environment_config",
        {
            "instruments": instruments,
            "mt5_instruments": mt5_instruments,
            "initial_balance": float(args.initial_balance or 100_000.0),
            "mode": "live",
            "max_steps": 10_000_000,
            "bus_data_active": True,
        },
        module="LiveTrainingPipeline",
        thesis="startup",
    )
    bus.set("execution_mode", "live", module="LiveTrainingPipeline", thesis="live mode active")


    try:
        from live.mt5_credentials import MT5Credentials

        if not mt5.initialize(MT5Credentials.PATH):
            logger.error("MT5 initialize failed: %s", mt5.last_error())
            return 1
        if not mt5.login(MT5Credentials.ACCOUNT, password=MT5Credentials.PASSWORD, server=MT5Credentials.SERVER):
            logger.error("MT5 login failed: %s", mt5.last_error())
            mt5.shutdown()
            return 1
    except Exception as e:
        logger.error("MT5 connection failed: %s", e)
        try:
            mt5.shutdown()
        except Exception:
            pass
        return 1

    account = mt5.account_info()
    if not account:
        logger.error("MT5 account_info() unavailable")
        return 1

    initial_balance = float(args.initial_balance or float(getattr(account, "balance", 100_000.0)))
    bus.set("account_balance", float(getattr(account, "balance", initial_balance)), module="LiveTrainingPipeline", thesis="MT5 balance")
    bus.set("account_equity", float(getattr(account, "equity", initial_balance)), module="LiveTrainingPipeline", thesis="MT5 equity")


    env_cfg = PropFirmConfig()
    env_cfg.instruments = instruments
    env_cfg.primary_timeframe = str(args.primary_timeframe).upper().strip()
    env_cfg.initial_balance = float(initial_balance)
    env_cfg.live_mode = True


    connector = LiveDataConnector(instruments=mt5_instruments, timeframes=timeframes)
    try:
        connector.connect()
    except Exception as e:
        logger.error("LiveDataConnector.connect() failed: %s", e)
        return 1

    hist = connector.get_historical_data(n_bars=int(args.n_bars)) or {}
    if not hist:
        logger.error("No historical data returned from LiveDataConnector")
        return 1


    env = PropFirmTradingEnv(hist, config=env_cfg, apply_curriculum_overrides=False)

    obs_builder = PPOObservationBuilder()


    ppo_core = PPOCore(config=PPOCoreConfig())
    try:
        ppo_core.set_instruments(instruments)
    except Exception:
        pass

    model_path = _resolve_model_path(args.model_path)
    if model_path:
        try:
            ppo_core.load(model_path)
            logger.info("Loaded model: %s", model_path)
        except Exception as e:
            logger.error("Failed to load model '%s': %s", model_path, e)
    else:
        logger.warning("No model found; PPOCore will use untrained weights")


    mask_cfg = LiveMaskConfig(
        daily_drawdown_limit=float(env_cfg.daily_drawdown_limit),
        max_drawdown_limit=float(env_cfg.max_drawdown_limit),
        daily_dd_safety_buffer=float(env_cfg.daily_dd_safety_buffer),
        max_dd_safety_buffer=float(env_cfg.max_dd_safety_buffer),
        max_trades_per_day=int(env_cfg.max_trades_per_day),
        max_trades_per_session=int(env_cfg.max_trades_per_session),
        max_consecutive_losses=int(env_cfg.max_consecutive_losses),
        min_minutes_between_entries=int(env_cfg.min_minutes_between_entries),
        min_minutes_after_loss=int(env_cfg.min_minutes_after_loss),
        enforce_hard_rules=bool(args.enforce_hard_rules),
        no_new_trades_start=env_cfg.no_new_trades_start,
        no_new_trades_end=env_cfg.no_new_trades_end,
        hard_close_time=env_cfg.hard_close_time,
        size_buckets=ppo_core.config.size_buckets,
    )
    mask_builder = LiveActionMaskBuilder(mask_cfg)


    executor: Optional[Executor] = None
    if bool(args.execute):
        try:
            executor = Executor(
                config={
                    "hold_positions_without_signal": bool(args.executor_hold_without_signal),
                }
            )
            logger.info("Executor initialized (LIVE). Orders will be sent.")
        except Exception as e:
            logger.error("Executor initialization failed: %s", e)
            return 1
    else:
        logger.info("Dry-run mode: no orders will be sent (use --execute to trade).")

    last_decision_bar_ts: Optional[Any] = None
    last_status_log = 0.0
    last_wait_log = 0.0
    last_manage_ts = 0.0
    last_loss_time: Optional[datetime] = None
    last_entry_time: Optional[datetime] = None
    day_start_balance: float = float(getattr(account, "equity", initial_balance) or initial_balance)
    current_day = None
    daily_trades = 0
    session_trades = 0
    consecutive_losses = 0
    consecutive_wins = 0
    winning_trades = 0
    total_trades = 0


    last_tf_sigs: Dict[str, Optional[BarSignature]] = {}
    last_state_hashes: Dict[str, str] = {}
    same_hash_streak: Dict[str, int] = {}
    last_closed_ts_str: Optional[str] = None
    last_decision_idx: Optional[int] = None


    has_pending_entry = False
    has_pending_exit = False
    pending_since: Optional[float] = None
    pending_timeout_s = 20.0

    try:
        logger.info(
            "TRAINING-PIPELINE MODE | instruments=%s | primary_tf=%s | only_on_new_bar=%s | enforce_hard_rules=%s | execute=%s",
            instruments,
            env_cfg.primary_timeframe,
            bool(args.only_on_new_bar),
            bool(args.enforce_hard_rules),
            bool(args.execute),
        )
        while running:
            t0 = time.time()
            try:
                market_data = connector.get_historical_data(n_bars=int(args.n_bars)) or {}
                if not market_data or primary_instrument not in market_data:
                    now = time.time()
                    if now - last_wait_log > 15 * 60.0:
                        logger.warning(
                            "Waiting for market data... instrument=%s (mt5=%s) timeframes=%s keys=%s",
                            primary_instrument,
                            primary_mt5_symbol,
                            timeframes,
                            list(market_data.keys()) if isinstance(market_data, dict) else [],
                        )
                        last_wait_log = now
                    if executor is not None and bool(args.manage_between_bars):
                        if now - last_manage_ts >= max(0.25, float(args.manage_interval_s)):
                            await executor.process()
                            last_manage_ts = now
                    await asyncio.sleep(float(args.poll_s))
                    continue

                bus.set("market_data", market_data, module="LiveTrainingPipeline", thesis="market data update")

                primary_block = market_data[primary_instrument].get(env_cfg.primary_timeframe)
                if primary_block is None or getattr(primary_block, "empty", True):
                    if time.time() - last_status_log > 30.0:
                        logger.warning(
                            "No primary timeframe data yet: %s %s (available=%s)",
                            primary_instrument,
                            env_cfg.primary_timeframe,
                            list((market_data.get(primary_instrument) or {}).keys()),
                        )
                        last_status_log = time.time()
                    if executor is not None and bool(args.manage_between_bars):
                        now_ts = time.time()
                        if now_ts - last_manage_ts >= max(0.25, float(args.manage_interval_s)):
                            await executor.process()
                            last_manage_ts = now_ts
                    await asyncio.sleep(float(args.poll_s))
                    continue

                latest_bar_ts = primary_block.index[-1]
                if len(primary_block.index) < 2:
                    if time.time() - last_status_log > 30.0:
                        logger.warning(
                            "Not enough bars for closed-bar decision yet (need >=2). latest_bar=%s",
                            str(latest_bar_ts),
                        )
                        last_status_log = time.time()
                    if executor is not None and bool(args.manage_between_bars):
                        now_ts = time.time()
                        if now_ts - last_manage_ts >= max(0.25, float(args.manage_interval_s)):
                            await executor.process()
                            last_manage_ts = now_ts
                    await asyncio.sleep(float(args.poll_s))
                    continue

                closed_bar_ts = primary_block.index[-2]


                if bool(args.only_on_new_bar):
                    if last_decision_bar_ts is not None and closed_bar_ts == last_decision_bar_ts:

                        now = time.time()
                        if now - last_wait_log > 15 * 60.0:
                            logger.info(
                                "Waiting for new CLOSED %s bar... last_closed=%s latest(forming)=%s",
                                env_cfg.primary_timeframe,
                                str(closed_bar_ts),
                                str(latest_bar_ts),
                            )
                            last_wait_log = now


                        if executor is not None and bool(args.manage_between_bars):
                            now_ts = time.time()
                            if now_ts - last_manage_ts >= max(0.25, float(args.manage_interval_s)):
                                await executor.process()
                                last_manage_ts = now_ts
                        await asyncio.sleep(float(args.poll_s))
                        continue
                    last_decision_bar_ts = closed_bar_ts
                    decision_idx = int(len(primary_block) - 2)
                    decision_bar_ts = closed_bar_ts
                    forming_bar_ts = latest_bar_ts
                else:

                    decision_idx = int(len(primary_block) - 1)
                    decision_bar_ts = latest_bar_ts
                    forming_bar_ts = latest_bar_ts


                if last_decision_idx is not None and decision_idx == last_decision_idx and bool(args.only_on_new_bar):


                    pass
                last_decision_idx = decision_idx

                account = mt5.account_info()
                if account:
                    balance = float(getattr(account, "balance", initial_balance))
                    equity = float(getattr(account, "equity", balance))
                else:
                    balance = float(initial_balance)
                    equity = float(initial_balance)


                env.data = market_data
                env._episode_instrument = primary_instrument
                env.current_step = max(0, int(decision_idx))
                env.episode_bars = int(env.current_step)
                env.balance = float(balance)
                env.equity = float(equity)
                env.config.initial_balance = float(initial_balance)
                env.day_start_balance = float(day_start_balance)


                try:
                    env._update_peak_balance()
                except Exception:
                    env.peak_balance = max(float(getattr(env, "peak_balance", balance)), float(equity))

                bar_dt = None
                try:
                    bar_dt = env._get_bar_dt(primary_instrument)
                except Exception:
                    bar_dt = None
                now_dt = bar_dt if isinstance(bar_dt, datetime) else datetime.now()
                if getattr(now_dt, "tzinfo", None) is not None:
                    now_dt = now_dt.replace(tzinfo=None)


                try:
                    day = now_dt.date()
                except Exception:
                    day = None
                if day is not None and day != current_day:
                    current_day = day
                    day_start_balance = float(equity)
                    daily_trades = 0
                env.day_start_balance = float(day_start_balance)


                live_pos = _get_live_position(
                    mt5,
                    primary_mt5_symbol,
                    tz=getattr(env, "tz", None),
                )
                if live_pos and live_pos.open_time and (last_entry_time is None):
                    last_entry_time = live_pos.open_time


                if has_pending_entry and live_pos is not None:
                    has_pending_entry = False
                    pending_since = None
                if has_pending_exit and live_pos is None:
                    has_pending_exit = False
                    pending_since = None
                if pending_since is not None and (time.time() - pending_since) > pending_timeout_s:

                    has_pending_entry = False
                    has_pending_exit = False
                    pending_since = None


                try:
                    current_dd, daily_dd = env._calc_dds()
                except Exception:
                    current_dd, daily_dd = 0.0, 0.0


                time_in_pos_bars = 0.0
                if live_pos and live_pos.open_time:
                    mins = max(1, int(env._tf_minutes()) if hasattr(env, "_tf_minutes") else 15)
                    age_min = (now_dt - live_pos.open_time).total_seconds() / 60.0
                    time_in_pos_bars = float(max(0.0, age_min / float(mins)))


                on_cooldown = False
                if last_loss_time is not None:
                    mins_since_loss = (now_dt - last_loss_time).total_seconds() / 60.0
                    on_cooldown = mins_since_loss < float(env_cfg.min_minutes_after_loss)


                win_rate = float(winning_trades / max(total_trades, 1))
                account_state = _build_account_state(
                    balance=balance,
                    equity=equity,
                    initial_balance=initial_balance,
                    current_dd=float(current_dd),
                    trades_today=int(daily_trades),
                    win_rate=win_rate,
                    position=live_pos,
                    time_in_position_bars=time_in_pos_bars,
                    on_cooldown=on_cooldown,
                )


                env.total_trades = int(total_trades)
                env.winning_trades = int(winning_trades)
                env.total_pnl = float(equity - initial_balance)
                env.daily_trades = int(daily_trades)
                env.consecutive_losses = int(consecutive_losses)
                env.consecutive_wins = int(consecutive_wins)
                env._session_trades = int(session_trades or daily_trades)
                env.session_start_balance = float(day_start_balance)
                env.session_pnl = float(equity - day_start_balance)


                sig_lines: List[str] = []
                try:
                    blocks = market_data.get(primary_instrument, {})
                    if isinstance(blocks, dict):
                        for tf in timeframes:
                            df = blocks.get(tf)
                            sig = _make_bar_signature(tf, df)
                            last_sig = last_tf_sigs.get(tf)
                            last_tf_sigs[tf] = sig
                            if sig is None:
                                sig_lines.append(f"{tf}:sig=NA")
                            else:

                                cc = "NA" if sig.closed_close is None else f"{sig.closed_close:.4f}"
                                fc = "NA" if sig.forming_close is None else f"{sig.forming_close:.4f}"
                                changed = ""
                                if last_sig is not None and (sig.closed_ts != last_sig.closed_ts or sig.closed_close != last_sig.closed_close):
                                    changed = "Δ"
                                sig_lines.append(f"{tf}:closed={sig.closed_ts} c={cc} forming={sig.forming_ts} c={fc}{changed}")
                except Exception:
                    sig_lines = []


                closed_ts_str = str(closed_bar_ts)
                new_closed = (last_closed_ts_str is None) or (closed_ts_str != last_closed_ts_str)
                last_closed_ts_str = closed_ts_str


                market_state = env._prepare_market_data(primary_instrument)
                expert_signals = env._prepare_expert_signals(primary_instrument)
                committee_state = env._prepare_committee_state(expert_signals)
                risk_state = env._prepare_risk_state()
                memory_state = env._prepare_memory_state(primary_instrument)
                trading_mode_state = env._prepare_trading_mode_state(primary_instrument)
                world_model_state = env._prepare_world_model_state(primary_instrument, expert_signals, committee_state)
                governor_state = env._get_governor_state()

                if new_closed:
                    for name, obj in (
                        ("market_state", market_state),
                        ("expert_signals", expert_signals),
                        ("committee_state", committee_state),
                        ("risk_state", risk_state),
                        ("memory_state", memory_state),
                        ("trading_mode_state", trading_mode_state),
                        ("world_model_state", world_model_state),
                        ("governor_state", governor_state),
                    ):
                        h = _hash_obj(obj)
                        prev = last_state_hashes.get(name)
                        last_state_hashes[name] = h
                        if prev == h:
                            same_hash_streak[name] = same_hash_streak.get(name, 0) + 1
                        else:
                            same_hash_streak[name] = 0

                obs = obs_builder.build(
                    market_data=market_state,
                    expert_signals=expert_signals,
                    committee_state=committee_state,
                    risk_state=risk_state,
                    memory_state=memory_state,
                    account_state=account_state,
                    world_model_state=world_model_state,
                    trading_mode_state=trading_mode_state,
                    governor_state=governor_state,
                )

                action_mask = None
                if bool(getattr(ppo_core, "is_discrete_action_space", False)):
                    action_mask = mask_builder.get_action_mask(
                        has_position=bool(live_pos is not None),
                        has_pending_entry=bool(has_pending_entry),
                        has_pending_exit=bool(has_pending_exit),
                        current_dd=float(current_dd),
                        daily_dd=float(daily_dd),
                        daily_trades=int(daily_trades),
                        session_trades=int(session_trades),
                        consecutive_losses=int(consecutive_losses),
                        last_entry_time=last_entry_time,
                        last_loss_time=last_loss_time,
                        current_time=now_dt,
                    )

                _action, _, _ = ppo_core.select_action(
                    obs=np.asarray(obs, dtype=np.float32),
                    deterministic=True,
                    instrument=primary_instrument,
                    action_mask=action_mask,
                )

                decoded = getattr(ppo_core, "last_discrete_action", None)
                if decoded is not None:
                    ppo_intent = str(getattr(decoded, "intent", "hold"))
                    size_mult = float(getattr(decoded, "size_mult", 0.0) or 0.0)
                else:
                    direction_score = float(_action[0]) if len(_action) > 0 else 0.0
                    size_mult = float(max(0.0, min(1.0, float((_action[1] + 1.0) / 2.0)))) if len(_action) > 1 else 0.0
                    if direction_score > float(ppo_core.config.direction_long_threshold):
                        ppo_intent = "long"
                    elif direction_score < float(ppo_core.config.direction_short_threshold):
                        ppo_intent = "short"
                    else:
                        ppo_intent = "hold"

                order_item = _translate_decision_to_intent(intent=ppo_intent, size_mult=size_mult, position=live_pos)
                hard_reasons: List[str] = []
                if bool(args.enforce_hard_rules) and action_mask is not None:
                    try:
                        _allowed, hard_reasons = mask_builder._hard_entry_allowed(  # type: ignore[attr-defined]
                            current_dd=float(current_dd),
                            daily_dd=float(daily_dd),
                            daily_trades=int(daily_trades),
                            session_trades=int(session_trades),
                            consecutive_losses=int(consecutive_losses),
                            last_entry_time=last_entry_time,
                            last_loss_time=last_loss_time,
                            current_time=now_dt,
                        )
                    except Exception:
                        hard_reasons = []

                if bool(args.explain):
                    bar_label = "closed_bar" if bool(args.only_on_new_bar) else "bar"


                    stuck_flags: List[str] = []
                    if new_closed:

                        for k in ("expert_signals", "committee_state", "trading_mode_state"):
                            n = same_hash_streak.get(k, 0)
                            if n >= 2:
                                stuck_flags.append(f"{k}:same_hash_streak={n}")

                    explanation_lines = [
                        f"[DECISION] {primary_instrument} (mt5={primary_mt5_symbol}) tf={env_cfg.primary_timeframe} {bar_label}={decision_bar_ts} latest(forming)={forming_bar_ts}",
                        f"Model: intent={ppo_intent} size={float(size_mult):.2f} discrete={bool(getattr(ppo_core, 'is_discrete_action_space', False))}",
                        _format_position_line(live_pos, now_dt),
                        _format_risk_line(
                            current_dd=float(current_dd),
                            daily_dd=float(daily_dd),
                            daily_trades=int(daily_trades),
                            session_trades=int(session_trades),
                            consecutive_losses=int(consecutive_losses),
                            on_cooldown=bool(on_cooldown),
                            env_cfg=env_cfg,
                        ),
                        f"Pending: entry={has_pending_entry} exit={has_pending_exit}",
                        _format_mask_line(
                            mask_builder=mask_builder,
                            action_mask=action_mask,
                            enforce_hard_rules=bool(args.enforce_hard_rules),
                            hard_reasons=hard_reasons,
                        ),
                        _format_trading_timing(trading_mode_state),
                        _format_primary_experts(expert_signals),
                        _format_htf_experts(expert_signals),
                        _format_committee(committee_state),
                        f"Data: {_summarize_live_data(market_data, primary_instrument, timeframes)}",
                    ]

                    if sig_lines:
                        explanation_lines.append("Sig: " + " | ".join(sig_lines[:6]))

                    if bool(args.only_on_new_bar):
                        explanation_lines.append(f"Diag: decision_idx={decision_idx} (fixed window ⇒ constant; do NOT cache on it)")

                    if stuck_flags:
                        explanation_lines.append("SUSPECT: " + " | ".join(stuck_flags))


                    try:
                        comm_action = str(committee_state.get("action", "flat")).lower()
                        if ppo_intent in ("long", "short") and comm_action in ("long", "short") and comm_action != ppo_intent:
                            explanation_lines.append(f"Conflict: PPO={ppo_intent} vs committee={comm_action}")
                    except Exception:
                        pass

                    logger.info("\n" + "\n".join(explanation_lines))
                else:
                    logger.info(
                        "[PPO] bar=%s | intent=%s | size=%.2f | pos=%s",
                        str(decision_bar_ts),
                        ppo_intent,
                        float(size_mult),
                        ("LONG" if (live_pos and live_pos.side > 0) else ("SHORT" if live_pos else "FLAT")),
                    )
                last_status_log = time.time()

                if bool(args.dump_snapshot):
                    try:
                        snap_dir = Path(str(args.snapshot_dir))
                        snap_dir.mkdir(parents=True, exist_ok=True)
                        inst_slug = _safe_symbol_slug(primary_instrument)
                        fname = f"training_pipeline_{inst_slug}_{env_cfg.primary_timeframe}_{str(decision_bar_ts).replace(':', '').replace(' ', '_')}.json"

                        obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)

                        obs_schema = None
                        try:
                            get_schema = getattr(obs_builder, "get_schema", None)
                            if callable(get_schema):
                                obs_schema = get_schema()
                        except Exception:
                            obs_schema = None

                        obs_by_group = None
                        if isinstance(obs_schema, dict) and isinstance(obs_schema.get("feature_groups"), dict):
                            try:
                                groups = obs_schema.get("feature_groups") or {}
                                obs_by_group = {}
                                for name, sl in groups.items():
                                    if not isinstance(sl, dict):
                                        continue
                                    start = int(sl.get("start", 0))
                                    end = int(sl.get("end", 0))
                                    if 0 <= start <= end <= int(obs_arr.shape[0]):
                                        obs_by_group[str(name)] = obs_arr[start:end].tolist()
                            except Exception:
                                obs_by_group = None

                        mask_summary = None
                        if isinstance(action_mask, np.ndarray):
                            try:
                                mask_summary = mask_builder.get_mask_summary(action_mask)
                            except Exception:
                                mask_summary = None
                        payload = {
                            "meta": {
                                "instrument": primary_instrument,
                                "mt5_instrument": primary_mt5_symbol,
                                "primary_timeframe": env_cfg.primary_timeframe,
                                "decision_bar": decision_bar_ts,
                                "latest_bar": forming_bar_ts,
                                "decision_idx": int(decision_idx),
                                "intent": ppo_intent,
                                "size_mult": float(size_mult),
                                "discrete": bool(getattr(ppo_core, "is_discrete_action_space", False)),
                                "enforce_hard_rules": bool(args.enforce_hard_rules),
                                "only_on_new_bar": bool(args.only_on_new_bar),
                                "execute": bool(args.execute),
                                "hard_block_reasons": list(hard_reasons),
                                "signatures": sig_lines,
                                "hashes": dict(last_state_hashes),
                                "same_hash_streak": dict(same_hash_streak),
                            },
                            "observation_schema": obs_schema,
                            "observation_by_group": obs_by_group,
                            "states": {
                                "market_state": market_state,
                                "expert_signals": expert_signals,
                                "committee_state": committee_state,
                                "risk_state": risk_state,
                                "memory_state": memory_state,
                                "account_state": account_state,
                                "world_model_state": world_model_state,
                                "trading_mode_state": trading_mode_state,
                                "governor_state": governor_state,
                            },
                            "observation": obs_arr.tolist(),
                            "mask": action_mask.tolist() if isinstance(action_mask, np.ndarray) else None,
                            "mask_summary": mask_summary,
                        }
                        (snap_dir / fname).write_text(json.dumps(_to_jsonable(payload), indent=2), encoding="utf-8")
                    except Exception as e:
                        logger.warning("Snapshot dump failed: %s", e)

                if order_item and executor is not None:

                    order_item["instrument"] = primary_mt5_symbol


                    act = str(order_item.get("action", "")).lower()
                    if act.startswith("open_"):
                        has_pending_entry = True
                        has_pending_exit = False
                        pending_since = time.time()
                    elif "close" in act:
                        has_pending_exit = True
                        has_pending_entry = False
                        pending_since = time.time()

                    existing = bus.get("order_queue", "LiveTrainingPipeline", default=[]) or []
                    existing.append(order_item)
                    bus.set("order_queue", existing, module="LiveTrainingPipeline", thesis="PPO training-pipeline intent")

                    await executor.process()


                if executor is not None:
                    try:
                        fills = bus.get("execution_reports", "LiveTrainingPipeline", default=[]) or []
                        for f in fills:
                            act = str(f.get("action", "")).lower().strip()
                            realized = float(f.get("realized_pnl", f.get("pnl", 0.0)) or 0.0)


                            is_close = "close" in act
                            if (not is_close) and (
                                act.startswith("open")
                                or act == "reverse"
                                or act.startswith("reverse")
                                or act.startswith("scale_up")
                            ):
                                daily_trades += 1
                                session_trades += 1
                                last_entry_time = now_dt


                            if is_close:
                                total_trades += 1
                                if realized > 0:
                                    winning_trades += 1
                                    consecutive_wins += 1
                                    consecutive_losses = 0
                                elif realized < 0:
                                    consecutive_losses += 1
                                    consecutive_wins = 0
                                    last_loss_time = now_dt
                    except Exception:
                        pass

            except Exception as e:
                logger.error("Training loop error: %s", e)


            elapsed = time.time() - t0
            await asyncio.sleep(max(0.1, float(args.poll_s) - elapsed))

    finally:
        try:
            connector.disconnect()
        except Exception:
            pass
        try:
            mt5.shutdown()
        except Exception:
            pass

    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live runner using training observation pipeline (A/B vs orchestrator).")
    p.add_argument("--engine", choices=["training", "orchestrator"], default="training")
    p.add_argument("--execute", action="store_true", help="Send orders (default: dry-run).")
    p.add_argument("--model-path", type=str, default=None, help="Path to SB3 .zip model (optional; auto-discovery otherwise).")
    p.add_argument("--instruments", type=str, default="XAUUSD", help="Comma-separated symbols (default: XAUUSD).")
    p.add_argument("--primary-timeframe", type=str, default="M15", help="Primary timeframe (default: M15).")
    p.add_argument("--timeframes", type=str, default="M15,H1,H4,D1", help="Comma-separated timeframes.")
    p.add_argument("--n-bars", type=int, default=1000, help="Bars fetched per timeframe (default: 1000).")
    p.add_argument("--poll-s", type=float, default=2.0, help="Poll interval seconds (default: 2.0).")
    p.add_argument("--only-on-new-bar", action=argparse.BooleanOptionalAction, default=True, help="Only act on new primary bar.")
    p.add_argument(
        "--manage-between-bars",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Executor position management between PPO bar-close decisions (trailing/ExitEngine).",
    )
    p.add_argument(
        "--manage-interval-s",
        type=float,
        default=5.0,
        help="How often to run between-bar position management (seconds).",
    )
    p.add_argument("--enforce-hard-rules", action=argparse.BooleanOptionalAction, default=True, help="Hard mask timing/DD/trade-count rules.")
    p.add_argument(
        "--executor-hold-without-signal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When SmartPositionManager has no fresh expert signals/intents, hold instead of triggering weak-signal exits.",
    )
    p.add_argument("--explain", action=argparse.BooleanOptionalAction, default=True, help="Log rich English decision context.")
    p.add_argument("--dump-snapshot", action=argparse.BooleanOptionalAction, default=False, help="Dump JSON snapshot per decision to snapshot-dir.")
    p.add_argument("--snapshot-dir", type=str, default="logs/explain", help="Directory for --dump-snapshot JSON files.")
    p.add_argument("--initial-balance", type=float, default=None, help="Override initial balance used for DD features.")
    return p


async def _amain() -> int:
    global running
    Path("logs").mkdir(exist_ok=True)
    Path("state").mkdir(exist_ok=True)


    try:
        from config.logging_config import setup_logging
        from config.models import LoggingConfig

        setup_logging(
            LoggingConfig(
                level="INFO",
                debug=False,
                log_dir="logs",
                filename="live_training_pipeline.log",
            )
        )
    except Exception:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler("logs/live_training_pipeline.log", encoding="utf-8"),
            ],
            force=True,
        )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    args = _build_arg_parser().parse_args()
    if args.engine == "orchestrator":
        return await _run_orchestrator_mode()
    return await _run_training_mode(args)


def main() -> int:
    try:
        return asyncio.run(_amain())
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
