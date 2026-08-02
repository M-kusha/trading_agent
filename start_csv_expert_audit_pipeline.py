#!/usr/bin/env python3
"""
CSV Expert Audit Pipeline (Pylance-clean)
========================================

Generates live-like `training_pipeline_*.json` explain snapshots from CSV OHLCV
for one instrument across multiple timeframes (M15/H1/H4/D1 by default).

No future leakage:
- All `states` and `observation` are computed strictly "as-of" the decision bar.
- Future outcomes are stored ONLY under `labels.*` (delta/close at horizons).

Outputs:
- training_pipeline_{SYMBOL}_{PRIMARY_TF}_{YYYY-MM-DD_HHMMSS}.json

Windows example:
  py start_csv_expert_audit_pipeline.py --dataset-dir data\\processed --dataset-metadata data\\processed\\metadata.json --instrument XAUUSD --days 365 --outdir logs\\explain\\csv_sim
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from datetime import time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from envs.core.env_types import PropFirmConfig, load_risk_policy
from envs.prop_firm_env import PropFirmTradingEnv
from modules.meta.ppo_observation_builder import PPOObservationBuilder

# Optional: SmartInfoBus + real voting/world-model modules
try:
    from modules.models.world_model import EnhancedWorldModel
    from modules.utils.info_bus import InfoBusManager
    from modules.voting.experts import MomentumExpert, SeasonalityRiskExpert, ThemeExpert, TrendExpert
    from modules.voting.stages.committee import CommitteeCoordinator
    SMART_MODULES_AVAILABLE = True
except Exception:
    InfoBusManager = None  # type: ignore
    TrendExpert = None  # type: ignore
    MomentumExpert = None  # type: ignore
    ThemeExpert = None  # type: ignore
    SeasonalityRiskExpert = None  # type: ignore
    CommitteeCoordinator = None  # type: ignore
    EnhancedWorldModel = None  # type: ignore
    SMART_MODULES_AVAILABLE = False

LOG = logging.getLogger("CSVExpertAuditPipeline")


# =========================
# Utilities
# =========================

def _safe_symbol_slug(symbol: str) -> str:
    return str(symbol).upper().replace("/", "").replace("\\", "").replace(" ", "")


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, datetime):
        return x.isoformat()
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    try:
        import numpy as np  # type: ignore

        if isinstance(x, np.ndarray):
            return [_to_jsonable(v) for v in x.tolist()]
        if isinstance(x, np.generic):
            return _to_jsonable(x.item())
    except Exception:
        pass
    return str(x)


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    ss = str(s).strip()
    if not ss:
        return None
    try:
        return pd.to_datetime(ss).to_pydatetime()
    except Exception as e:
        raise SystemExit(f"Invalid datetime: {ss} ({type(e).__name__}: {e})")


def _ensure_tz(dt: datetime, tz: ZoneInfo) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def _dt_to_iso(dt: datetime, tz: ZoneInfo) -> str:
    return _ensure_tz(dt, tz).isoformat(timespec="seconds")


def _candle_dict_at_primary(primary_df: pd.DataFrame, idx: int, tz: ZoneInfo) -> Dict[str, Any]:
    """
    Returns a JSON-ready candle dict from primary_df at iloc idx.
    Assumes _load_ohlcv_csv already ensured spread exists (defaults to 0.0).
    """
    t = pd.to_datetime(primary_df["time"].iloc[idx])
    return {
        "ts": _dt_to_iso(t.to_pydatetime(), tz),
        "open": float(primary_df["open"].iloc[idx]),
        "high": float(primary_df["high"].iloc[idx]),
        "low": float(primary_df["low"].iloc[idx]),
        "close": float(primary_df["close"].iloc[idx]),
        "volume": float(primary_df["volume"].iloc[idx]) if "volume" in primary_df.columns else 0.0,
        "spread": float(primary_df["spread"].iloc[idx]) if "spread" in primary_df.columns else 0.0,
    }


def _get_time_col(columns: Sequence[str]) -> str:
    for c in ("time", "timestamp", "datetime", "date"):
        if c in columns:
            return c
    raise ValueError("No time column found (expected one of: time/timestamp/datetime/date)")


def _index_tz(df: pd.DataFrame) -> Optional[Any]:
    """
    Pylance-safe tz accessor: Pandas stubs type df.index as Index[Any] (no .tz),
    so we use getattr.
    """
    return getattr(df.index, "tz", None)


def _timeval_to_minutes(tv: object) -> int:
    """
    Safe conversion of a time-like object with `hour` and `minute` attributes to minutes-of-day.
    Returns 0 on any failure to keep behavior conservative for Pylance typing.
    """
    try:
        return int(tv.hour) * 60 + int(tv.minute)
    except Exception:
        return 0


# =========================
# Policy / config sync
# =========================

def _load_session_policy_into_cfg(cfg: PropFirmConfig) -> None:
    """
    Best-effort: sync session window hours from config/risk_policy.yaml into PropFirmConfig.
    """
    try:
        policy = load_risk_policy() or {}
        if not isinstance(policy, dict):
            return
        sess = policy.get("session_management", {})
        if not isinstance(sess, dict):
            return

        tz = sess.get("timezone")
        if isinstance(tz, str) and tz.strip():
            cfg.tz = tz.strip()

        def _h_to_time(k: str, fallback: dtime) -> dtime:
            v = sess.get(k)
            if v is None:
                return fallback
            try:
                h = int(str(v))
                return dtime(h % 24, 0)
            except Exception:
                return fallback

        cfg.no_new_trades_start = _h_to_time("no_new_trades_start", cfg.no_new_trades_start)
        cfg.no_new_trades_end = _h_to_time("no_new_trades_end", cfg.no_new_trades_end)
        cfg.hard_close_time = _h_to_time("hard_close_hour", cfg.hard_close_time)
        cfg.prime_start = _h_to_time("prime_hours_start", cfg.prime_start)
        cfg.prime_end = _h_to_time("prime_hours_end", cfg.prime_end)

        fem = sess.get("final_exit_window_minutes")
        try:
            cfg.final_exit_window_minutes = int(str(fem))
        except Exception:
            pass
    except Exception:
        return


def _try_call_sync_from_yaml(cfg: PropFirmConfig) -> None:
    """
    Some codebases have PropFirmConfig.sync_from_yaml(), others do not.
    Avoid Pylance "unknown attribute" by using getattr.
    """
    fn = getattr(cfg, "sync_from_yaml", None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


# =========================
# CSV loading / alignment
# =========================

def _resolve_csv_files(
    *,
    dataset_dir: Path,
    dataset_metadata: Optional[Path],
    instrument: str,
    timeframes: List[str],
) -> Dict[str, Path]:
    """
    Returns {timeframe: csv_path}.

    Supports:
    - metadata.json with `files[SYMBOL][TF]`.
    - fallback pattern: <dataset_dir>/<SYMBOL>_<TF>_features.csv
    """
    out: Dict[str, Path] = {}
    sym = str(instrument).upper().strip()

    meta: Dict[str, Any] = {}
    if dataset_metadata and dataset_metadata.exists():
        try:
            meta = json.loads(dataset_metadata.read_text(encoding="utf-8"))
        except Exception as e:
            raise SystemExit(f"Failed to read dataset metadata: {dataset_metadata} ({e})")

    for tf in timeframes:
        tfu = str(tf).upper().strip()
        p: Optional[Path] = None

        meta_files = meta.get("files", {}) if isinstance(meta, dict) else {}
        if isinstance(meta_files, dict):
            sym_files = meta_files.get(sym)
            if isinstance(sym_files, dict):
                v = sym_files.get(tfu)
                if isinstance(v, str) and v:
                    cand = Path(v)
                    # Prefer absolute path, then dataset_dir/cand, then cand as given,
                    # then dataset_dir/cand.name — avoid duplicating dataset_dir twice.
                    if cand.is_absolute():
                        p = cand
                    else:
                        cand_under_dataset = dataset_dir / cand
                        if cand_under_dataset.exists():
                            p = cand_under_dataset
                        elif cand.exists():
                            p = cand
                        elif (dataset_dir / cand.name).exists():
                            p = dataset_dir / cand.name
                        else:
                            p = cand_under_dataset

        if p is None:
            p = dataset_dir / f"{sym}_{tfu}_features.csv"

        if not p.exists():
            raise SystemExit(f"Missing CSV for {sym} {tfu}: {p}")
        out[tfu] = p

    return out


def _load_ohlcv_csv(path: Path, *, tz: ZoneInfo) -> pd.DataFrame:
    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    time_col = _get_time_col(header_cols)

    needed = [time_col, "open", "high", "low", "close", "volume", "spread"]
    usecols = [c for c in needed if c in header_cols]
    df = pd.read_csv(path, usecols=usecols, parse_dates=[time_col])

    df = df.rename(columns={time_col: "time"}).copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()

    t = pd.to_datetime(df["time"], errors="coerce")
    # Make tz-aware
    if getattr(t.dt, "tz", None) is None:
        df["time"] = t.dt.tz_localize(tz, nonexistent="shift_forward", ambiguous="infer")
    else:
        df["time"] = t.dt.tz_convert(tz)

    if "spread" not in df.columns:
        df["spread"] = 0.0

    df = df.sort_values("time")
    df = df.drop_duplicates(subset=["time"], keep="last")
    df = df.set_index("time", drop=False)
    return df


def _time_in_window_minutes(minute_of_day: int, *, start_min: int, end_min: int) -> bool:
    """Scalar window check for [start, end), supports cross-midnight."""
    if start_min == end_min:
        return False
    if start_min < end_min:
        return (minute_of_day >= start_min) and (minute_of_day < end_min)
    return (minute_of_day >= start_min) or (minute_of_day < end_min)


def _compute_block_reasons(decision_time: pd.Timestamp, *, cfg: PropFirmConfig) -> List[str]:
    reasons: List[str] = []
    minutes = int(decision_time.hour) * 60 + int(decision_time.minute)
    weekday = int(decision_time.weekday())

    if weekday >= 5 and (not bool(getattr(cfg, "allow_weekend_holding", False))):
        reasons.append("weekend")

    no_new_trades_start_min = _timeval_to_minutes(cfg.no_new_trades_start)
    no_new_trades_end_min = _timeval_to_minutes(cfg.no_new_trades_end)
    if _time_in_window_minutes(minutes, start_min=no_new_trades_start_min, end_min=no_new_trades_end_min):
        reasons.append("no_new_trades_window")

    hard_close_min = _timeval_to_minutes(cfg.hard_close_time)
    if minutes >= hard_close_min:
        reasons.append("after_hard_close")

    final_exit_start_min = hard_close_min - int(cfg.final_exit_window_minutes)
    if final_exit_start_min <= minutes < hard_close_min:
        reasons.append("final_exit_window")

    return reasons


def _asof_iloc(df: pd.DataFrame, t: pd.Timestamp) -> Optional[int]:
    """
    Return iloc for last CLOSED candle strictly before t (i.e., time < t).

    This is the correct semantics when your decision timestamp `t` is the OPEN of a new bar
    and you want to build observations from fully closed bars only (no leakage).
    """
    if df.empty:
        return None

    idx_tz = _index_tz(df)
    tt = pd.to_datetime(t)

    if tt.tzinfo is None and idx_tz is not None:
        tt = tt.tz_localize(idx_tz)  # type: ignore[arg-type]
    elif tt.tzinfo is not None and idx_tz is not None and str(tt.tzinfo) != str(idx_tz):
        tt = tt.tz_convert(idx_tz)  # type: ignore[arg-type]

    # STRICTLY BEFORE t -> side="left"
    pos_obj = df.index.searchsorted(tt, side="left")
    try:
        pos = int(np.asarray(pos_obj).item())
    except Exception:
        return None

    pos = pos - 1
    if pos < 0:
        return None
    if pos >= len(df):
        return len(df) - 1
    return pos


def _asof_inclusive_iloc(df: pd.DataFrame, t: pd.Timestamp) -> Optional[int]:
    """
    Return iloc for last candle with time <= t (inclusive).
    Kept in case you explicitly need inclusive behavior somewhere else.
    """
    if df.empty:
        return None

    idx_tz = _index_tz(df)
    tt = pd.to_datetime(t)

    if tt.tzinfo is None and idx_tz is not None:
        tt = tt.tz_localize(idx_tz)  # type: ignore[arg-type]
    elif tt.tzinfo is not None and idx_tz is not None and str(tt.tzinfo) != str(idx_tz):
        tt = tt.tz_convert(idx_tz)  # type: ignore[arg-type]

    pos_obj = df.index.searchsorted(tt, side="right")
    try:
        pos = int(np.asarray(pos_obj).item())
    except Exception:
        return None

    pos = pos - 1
    if pos < 0:
        return None
    if pos >= len(df):
        return len(df) - 1
    return pos



def _select_decision_times(
    primary_df: pd.DataFrame,
    *,
    tz: ZoneInfo,
    start_dt: datetime,
    end_dt: datetime,
    cfg: PropFirmConfig,
    warmup_bars: int,
    only_trade_hours: bool,
    exclude_weekends: bool,
    max_horizon_bars: int,
) -> List[pd.Timestamp]:
    """
    Select decision times from primary timeframe with optional trade-hour filtering.
    Avoids pandas-stub pitfalls by iterating scalarly (slower but clean + reliable).
    """
    start = _ensure_tz(start_dt, tz)
    end = _ensure_tz(end_dt, tz)

    times = list(primary_df["time"])
    out: List[pd.Timestamp] = []

    last_allowed_i = len(times) - 1 - int(max_horizon_bars)
    if last_allowed_i <= 0:
        return out

    for i, t in enumerate(times):
        if i < int(warmup_bars):
            continue
        if i > last_allowed_i:
            break

        tt = pd.to_datetime(t)
        if tt.tzinfo is None:
            tt = tt.tz_localize(tz)
        else:
            tt = tt.tz_convert(tz)

        if tt < start or tt > end:
            continue

        weekday = int(tt.weekday())
        if exclude_weekends and weekday >= 5 and (not bool(getattr(cfg, "allow_weekend_holding", False))):
            continue

        if only_trade_hours:
            mins = int(tt.hour) * 60 + int(tt.minute)
            no_new_trades_start_min = _timeval_to_minutes(cfg.no_new_trades_start)
            no_new_trades_end_min = _timeval_to_minutes(cfg.no_new_trades_end)
            hard_close_min = _timeval_to_minutes(cfg.hard_close_time)
            final_exit_start_min = hard_close_min - int(cfg.final_exit_window_minutes)

            if _time_in_window_minutes(mins, start_min=no_new_trades_start_min, end_min=no_new_trades_end_min):
                continue
            if final_exit_start_min <= mins < hard_close_min:
                continue
            if mins >= hard_close_min:
                continue

        out.append(tt)

    return out


# =========================
# SmartInfoBus / real modules helpers (NO LEAKAGE)
# =========================
def _tf_floor(ts: pd.Timestamp, tf: str) -> pd.Timestamp:
    """
    Floor timestamp to timeframe boundary.
    Assumes ts is tz-aware already.
    """
    tfu = str(tf).upper().strip()
    if tfu == "M15":
        return ts.floor("15min")
    if tfu == "H1":
        return ts.floor("H")
    if tfu == "H4":
        return ts.floor("4H")
    if tfu == "D1":
        return ts.normalize()
    # Fallback: no flooring
    return ts


def _forming_bar_from_primary(
    primary_df: pd.DataFrame,
    *,
    t_dec: pd.Timestamp,
    tf: str,
    tz: ZoneInfo,
) -> Optional[Dict[str, Any]]:
    """
    Build a forming (partial) candle for TF using primary bars in [tf_start, t_dec),
    i.e. strictly before decision time. This avoids leakage.

    For example at 10:15:
      - H1 forming bar covers 10:00..10:15 (using closed M15 bars up to 10:15)
      - D1 forming covers 00:00..10:15
    """
    if primary_df.empty:
        return None

    end = pd.to_datetime(t_dec)
    if end.tzinfo is None:
        end = end.tz_localize(tz)
    else:
        end = end.tz_convert(tz)

    start = _tf_floor(end, tf)

    # IMPORTANT: strictly < end to avoid including the just-opened primary bar at t_dec
    mask = (primary_df["time"] >= start) & (primary_df["time"] < end)
    chunk = primary_df.loc[mask]
    if chunk.empty:
        return None

    o = float(chunk["open"].iloc[0]) if "open" in chunk.columns else float(chunk["close"].iloc[0])
    h = float(chunk["high"].max()) if "high" in chunk.columns else float(chunk["close"].max())
    l = float(chunk["low"].min()) if "low" in chunk.columns else float(chunk["close"].min())
    c = float(chunk["close"].iloc[-1])

    vol = float(chunk["volume"].sum()) if "volume" in chunk.columns else float(len(chunk))
    spr = float(chunk["spread"].iloc[-1]) if "spread" in chunk.columns else 0.0

    return {
        "timestamp": _dt_to_iso(end.to_pydatetime(), tz),  # "as-of" time for this partial bar
        "start": _dt_to_iso(start.to_pydatetime(), tz),
        "open": o,
        "high": h,
        "low": l,
        "close": c,
        "volume": vol,
        "spread": spr,
        "is_forming": True,
    }

def _build_historical_prices_asof(
    *,
    data_dict: Dict[str, Dict[str, pd.DataFrame]],
    instrument: str,
    timeframes: List[str],
    t_dec: pd.Timestamp,
    tz: ZoneInfo,
) -> Dict[str, Any]:
    """
    Live-like historical_prices with NO leakage:

    - Closed arrays: bars with time < t_dec
    - closed_bar: last closed bar (safe)
    - forming_bar: partial TF candle built from PRIMARY in [tf_start, t_dec)
    - current_bar: forming_bar if available else closed_bar  <-- key change
      (matches typical MT5 “shift 0” behavior)
    """
    sym = str(instrument).upper().strip()
    out_sym: Dict[str, Any] = {}

    primary_tf = str(timeframes[0]).upper().strip() if timeframes else "M15"
    primary_df = data_dict.get(sym, {}).get(primary_tf)

    for tf in timeframes:
        tfu = str(tf).upper().strip()
        df_tf = data_dict.get(sym, {}).get(tfu)
        if df_tf is None or df_tf.empty:
            continue

        # closed bars only (< t_dec)
        j = _asof_iloc(df_tf, pd.to_datetime(t_dec))
        if j is None:
            continue
        jj = int(j)

        df_slice = df_tf.iloc[: jj + 1]
        if df_slice.empty:
            continue

        def _col(name: str) -> List[float]:
            if name in df_slice.columns:
                return [float(x) for x in df_slice[name].tolist()]
            return [float(x) for x in df_slice["close"].tolist()]

        last = df_slice.iloc[-1]
        closed_bar = {
            "timestamp": _dt_to_iso(pd.to_datetime(last["time"]).to_pydatetime(), tz),
            "open": float(last["open"]) if "open" in df_slice.columns else float(last["close"]),
            "high": float(last["high"]) if "high" in df_slice.columns else float(last["close"]),
            "low": float(last["low"]) if "low" in df_slice.columns else float(last["close"]),
            "close": float(last["close"]),
            "volume": float(last["volume"]) if "volume" in df_slice.columns else 1.0,
            "is_forming": False,
        }

        forming_bar = None
        if primary_df is not None and not primary_df.empty and tfu != primary_tf:
            forming_bar = _forming_bar_from_primary(
                primary_df,
                t_dec=pd.to_datetime(t_dec),
                tf=tfu,
                tz=tz,
            )

        # IMPORTANT: expose forming via current_bar so existing modules update every M15
        current_bar = forming_bar if forming_bar is not None else closed_bar

        out_sym[tfu] = {
            "open": _col("open"),
            "high": _col("high"),
            "low": _col("low"),
            "close": _col("close"),
            "volume": _col("volume") if "volume" in df_slice.columns else [1.0] * len(df_slice),
            "bars_available": len(df_slice),

            # explicit
            "closed_bar": closed_bar,
            "last_closed_bar": closed_bar,

            # live-like
            "forming_bar": forming_bar,
            "forming_available": bool(forming_bar),

            # compatibility / typical MT5 semantics
            "current_bar": current_bar,
        }

    return {sym: out_sym}




def _bus_set_for_modules(
    smart_bus: Any,
    *,
    key: str,
    value: Any,
    module_names: List[str],
    thesis: str = "CSV_SIM",
) -> None:
    """
    Many modules read bus keys under their own module name (self.__class__.__name__).
    Publish the same value under multiple module namespaces to avoid mismatches.
    """
    for m in module_names:
        try:
            smart_bus.set(key, value, module=m, thesis=thesis)
        except Exception:
            pass


def _maybe_set_event_loop(loop: Any) -> None:
    try:
        import asyncio
        asyncio.set_event_loop(loop)
    except Exception:
        return


def _run_real_modules_asof(
    *,
    smart_bus: Any,
    experts_loop: Any,
    instrument: str,
    timeframes: List[str],
    data_dict: Dict[str, Dict[str, pd.DataFrame]],
    t_dec: pd.Timestamp,
    tz: ZoneInfo,
    env: Any,
    obs_builder: Any,
    trend_mod: Any,
    mom_mod: Any,
    theme_mod: Any,
    seas_mod: Any,
    committee_mod: Any,
    wm_mod: Any,
) -> Tuple[Optional[Any], Dict[str, Any]]:
    """
    Publish *as-of* historical_prices (no leakage), run real modules, and build observation
    via obs_builder.build(smart_bus=..., module_name="CSV_SIM").

    Returns:
      (obs_live, live_blob)
    """
    live_blob: Dict[str, Any] = {}
    obs_live: Optional[Any] = None

    module_names = [
        "CSV_SIM",
        "TrendExpert",
        "MomentumExpert",
        "ThemeExpert",
        "SeasonalityRiskExpert",
        "CommitteeCoordinator",
        "EnhancedWorldModel",
    ]

    try:
        _maybe_set_event_loop(experts_loop)

        hist_asof = _build_historical_prices_asof(
            data_dict=data_dict,
            instrument=instrument,
            timeframes=timeframes,
            t_dec=t_dec,
            tz=tz,
        )

        # Publish hist snapshot under multiple namespaces (prevents class-name mismatch)
        _bus_set_for_modules(smart_bus, key="historical_prices", value=hist_asof, module_names=module_names, thesis="CSV_SIM hist_asof")

        # Ensure no stale features from earlier runs
        _bus_set_for_modules(smart_bus, key="features", value={}, module_names=module_names, thesis="CSV_SIM features")

        # Help governor/account pull paths if builder expects them
        try:
            gov = env._get_governor_state()
            smart_bus.set("governor_state", gov, module="CSV_SIM", thesis="CSV_SIM governor_state")
        except Exception:
            pass
        try:
            ts_fn = getattr(env, "_get_trade_statistics", None) or getattr(env, "get_trade_statistics", None)
            if callable(ts_fn):
                smart_bus.set("trade_statistics", ts_fn(), module="CSV_SIM", thesis="CSV_SIM trade_statistics")
        except Exception:
            pass

        # Run modules (capture return values)
        import asyncio

        coros = []
        mods: List[Tuple[str, Any]] = []

        # Some module implementations accept kwargs; some don't. Call conservatively.
        def _call_process(mod: Any, name: str) -> Optional[Any]:
            if mod is None or not hasattr(mod, "process"):
                return None
            fn = mod.process
            if not callable(fn):
                return None
            try:
                return fn(market_data=hist_asof)
            except TypeError:
                try:
                    return fn()
                except Exception:
                    return None
            except Exception:
                return None

        for name, mod in [
            ("TrendExpert", trend_mod),
            ("MomentumExpert", mom_mod),
            ("ThemeExpert", theme_mod),
            ("SeasonalityRiskExpert", seas_mod),
        ]:
            c = _call_process(mod, name)
            if c is not None:
                mods.append((name, mod))
                coros.append(c)

        # Committee & world model typically depend on expert outputs
        for name, mod in [
            ("CommitteeCoordinator", committee_mod),
            ("EnhancedWorldModel", wm_mod),
        ]:
            c = _call_process(mod, name)
            if c is not None:
                mods.append((name, mod))
                coros.append(c)

        module_outputs: Dict[str, Any] = {}
        if coros:
            results = experts_loop.run_until_complete(asyncio.gather(*coros, return_exceptions=True))
            for (name, _m), r in zip(mods, results):
                if isinstance(r, Exception):
                    module_outputs[name] = {"error": f"{type(r).__name__}: {r}"}
                else:
                    module_outputs[name] = r

        # Build observation from bus
        obs_live = obs_builder.build(smart_bus=smart_bus, module_name="CSV_SIM")

        # Best-effort reads for audit
        live_blob = {
            "historical_prices_asof": hist_asof,
            "module_outputs": module_outputs,
            "bus": {
                "expert_votes": smart_bus.get("expert_votes", "CSV_SIM") if hasattr(smart_bus, "get") else None,
                "experts": smart_bus.get("experts", "CSV_SIM") if hasattr(smart_bus, "get") else None,
                "committee_decision": smart_bus.get("committee_decision", "CSV_SIM") if hasattr(smart_bus, "get") else None,
                "world_model_predictions": smart_bus.get("world_model_predictions", "CSV_SIM") if hasattr(smart_bus, "get") else None,
                "governor_state": smart_bus.get("governor_state", "CSV_SIM") if hasattr(smart_bus, "get") else None,
            },
        }

        return obs_live, live_blob
    except Exception as e:
        return None, {"error": f"{type(e).__name__}: {e}"}


# =========================
# Main
# =========================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=str, default="data/processed")
    ap.add_argument("--dataset-metadata", type=str, default=None)
    ap.add_argument("--instrument", type=str, default="XAUUSD")
    ap.add_argument("--primary-timeframe", type=str, default="M15")
    ap.add_argument("--timeframes", type=str, default="M15,H1,H4,D1")

    ap.add_argument("--start", type=str, default=None)
    ap.add_argument("--end", type=str, default=None)
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--warmup-bars", type=int, default=0)

    ap.add_argument("--only-trade-hours", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--exclude-weekends", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--outdir", type=str, default="logs/explain/csv_sim")
    ap.add_argument("--max-bars", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--pretty", action=argparse.BooleanOptionalAction, default=False)

    ap.add_argument(
        "--use-info-bus",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SmartInfoBus + modules when available (default: True)."
    )

    ap.add_argument("--label-bars", type=str, default="1,4,16,32",
                    help="Comma-separated future horizons as primary bars (e.g., 1=+15m,4=+1h,16=+4h,32=+8h).")

    args = ap.parse_args()

    dataset_dir = Path(str(args.dataset_dir))
    dataset_metadata = Path(str(args.dataset_metadata)) if args.dataset_metadata else None

    instrument = str(args.instrument).upper().strip()
    primary_tf = str(args.primary_timeframe).upper().strip()
    timeframes = [str(x).upper().strip() for x in str(args.timeframes).split(",") if str(x).strip()]
    if primary_tf not in timeframes:
        timeframes = [primary_tf] + [tf for tf in timeframes if tf != primary_tf]

    outdir = Path(str(args.outdir))
    outdir.mkdir(parents=True, exist_ok=True)

    # Config
    cfg = PropFirmConfig()
    cfg.instruments = [instrument]
    cfg.primary_timeframe = primary_tf
    _load_session_policy_into_cfg(cfg)
    _try_call_sync_from_yaml(cfg)

    tz = ZoneInfo(str(cfg.tz))

    # Load CSVs
    csv_files = _resolve_csv_files(
        dataset_dir=dataset_dir,
        dataset_metadata=dataset_metadata,
        instrument=instrument,
        timeframes=timeframes,
    )

    data_dict: Dict[str, Dict[str, pd.DataFrame]] = {instrument: {}}
    for tf, p in csv_files.items():
        LOG.info("Loading %s %s from %s", instrument, tf, str(p))
        data_dict[instrument][tf] = _load_ohlcv_csv(p, tz=tz)

    primary_df = data_dict[instrument][primary_tf]
    if len(primary_df) < 10:
        raise SystemExit(f"Not enough primary bars: {len(primary_df)}")

    # Start/end
    end_dt = _parse_dt(args.end)
    if end_dt is None:
        end_dt = pd.to_datetime(primary_df["time"].iloc[-1]).to_pydatetime()
    start_dt = _parse_dt(args.start)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=int(args.days))

    # Warmup
    warmup_bars = int(args.warmup_bars) if int(args.warmup_bars) > 0 else 0
    if warmup_bars <= 0:
        try:
            tmp_env = PropFirmTradingEnv(data_dict, config=cfg, apply_curriculum_overrides=False)
            buf = getattr(tmp_env, "_episode_start_buffer", None)
            if callable(buf):
                try:
                    val = buf()
                    try:
                        warmup_bars = int(str(val)) if val is not None else 300
                    except Exception:
                        warmup_bars = 300
                except Exception:
                    warmup_bars = 300
            else:
                warmup_bars = 300
        except Exception:
            warmup_bars = 300

    # Label horizons
    try:
        label_bars = [int(x.strip()) for x in str(args.label_bars).split(",") if x.strip()]
        label_bars = sorted({b for b in label_bars if b > 0})
    except Exception as e:
        raise SystemExit(f"Invalid --label-bars: {args.label_bars} ({e})")
    if not label_bars:
        label_bars = [1]
    max_h = int(max(label_bars))

    # Select decision times
    decision_times = _select_decision_times(
        primary_df,
        tz=tz,
        start_dt=start_dt,
        end_dt=end_dt,
        cfg=cfg,
        warmup_bars=warmup_bars,
        only_trade_hours=bool(args.only_trade_hours),
        exclude_weekends=bool(args.exclude_weekends),
        max_horizon_bars=max_h,
    )
    if int(args.max_bars) > 0:
        decision_times = decision_times[: int(args.max_bars)]

    LOG.info(
        "Generating snapshots: instrument=%s primary_tf=%s bars=%d (start=%s end=%s warmup=%d max_h=%d)",
        instrument,
        primary_tf,
        len(decision_times),
        str(start_dt),
        str(end_dt),
        int(warmup_bars),
        int(max_h),
    )

    # Env used only for state building (no stepping)
    env = PropFirmTradingEnv(data_dict, config=cfg, apply_curriculum_overrides=False)
    try:
        env.reset()
    except Exception:
        pass

    # SmartInfoBus + modules
    smart_bus = None
    experts_loop = None
    trend_mod = mom_mod = theme_mod = seas_mod = committee_mod = wm_mod = None
    if SMART_MODULES_AVAILABLE and InfoBusManager is not None:
        try:
            smart_bus = InfoBusManager.get_instance()
            import asyncio
            experts_loop = asyncio.new_event_loop()

            try:
                trend_mod = TrendExpert() if TrendExpert is not None else None
            except Exception:
                trend_mod = None
            try:
                mom_mod = MomentumExpert() if MomentumExpert is not None else None
            except Exception:
                mom_mod = None
            try:
                theme_mod = ThemeExpert() if ThemeExpert is not None else None
            except Exception:
                theme_mod = None
            try:
                seas_mod = SeasonalityRiskExpert() if SeasonalityRiskExpert is not None else None
            except Exception:
                seas_mod = None
            try:
                committee_mod = CommitteeCoordinator() if CommitteeCoordinator is not None else None
            except Exception:
                committee_mod = None
            try:
                wm_mod = EnhancedWorldModel() if EnhancedWorldModel is not None else None
            except Exception:
                wm_mod = None
        except Exception:
            smart_bus = None
            experts_loop = None

    LOG.info("SMART_MODULES_AVAILABLE=%s smart_bus=%s experts_loop=%s", SMART_MODULES_AVAILABLE, bool(smart_bus), bool(experts_loop))
    LOG.info(
        "modules present: trend=%s mom=%s theme=%s seas=%s committee=%s world_model=%s",
        bool(trend_mod),
        bool(mom_mod),
        bool(theme_mod),
        bool(seas_mod),
        bool(committee_mod),
        bool(wm_mod),
    )

    obs_builder = PPOObservationBuilder()
    obs_schema: Any = None
    try:
        get_schema = getattr(obs_builder, "get_schema", None)
        if callable(get_schema):
            obs_schema = get_schema()
    except Exception:
        obs_schema = None

    # Flat account state
    initial_balance = float(getattr(cfg, "initial_balance", 100000.0))
    try:
        env.balance = float(initial_balance)
        env.equity = float(initial_balance)
        env.day_start_balance = float(initial_balance)
        env.total_trades = 0
        env.winning_trades = 0
        env.total_pnl = 0.0
        env.daily_trades = 0
        env.consecutive_losses = 0
        env.consecutive_wins = 0
        env._session_trades = 0
        env.session_start_balance = float(initial_balance)
        env.session_pnl = 0.0
    except Exception:
        pass

    inst_slug = _safe_symbol_slug(instrument)
    indent = 2 if bool(args.pretty) else None

    written = 0
    for n, t_dec in enumerate(decision_times, start=1):
        try:
            i = _asof_iloc(primary_df, t_dec)
            if i is None:
                continue
            ii = int(i)

            if ii + max_h >= len(primary_df):
                continue

            try:
                env._episode_instrument = instrument
            except Exception:
                pass
            env.current_step = int(ii)
            env.episode_bars = int(ii)

            block_reasons = _compute_block_reasons(pd.to_datetime(t_dec), cfg=cfg)

            # =========================
            # ENV BUILDERS (training path)
            # =========================
            market_env = env._prepare_market_data(instrument)
            experts_env = env._prepare_expert_signals(instrument)
            committee_env = env._prepare_committee_state(experts_env)
            risk_env = env._prepare_risk_state()
            memory_env = env._prepare_memory_state(instrument)
            mode_env = env._prepare_trading_mode_state(instrument)
            wm_env = env._prepare_world_model_state(instrument, experts_env, committee_env)
            gov_env = env._get_governor_state()
            acct_env = env._prepare_account_state(instrument)

            obs_env = obs_builder.build(
                market_data=market_env,
                expert_signals=experts_env,
                committee_state=committee_env,
                risk_state=risk_env,
                memory_state=memory_env,
                account_state=acct_env,
                world_model_state=wm_env,
                trading_mode_state=mode_env,
                governor_state=gov_env,
            )

            # =========================
            # REAL MODULES (live path, no leakage)
            # =========================
            obs_live = None
            live_blob: Dict[str, Any] = {"enabled": False}

            if smart_bus is not None and experts_loop is not None and bool(args.use_info_bus):
                obs_live, live_blob = _run_real_modules_asof(
                    smart_bus=smart_bus,
                    experts_loop=experts_loop,
                    instrument=instrument,
                    timeframes=timeframes,
                    data_dict=data_dict,
                    t_dec=pd.to_datetime(t_dec),
                    tz=tz,
                    env=env,
                    obs_builder=obs_builder,
                    trend_mod=trend_mod,
                    mom_mod=mom_mod,
                    theme_mod=theme_mod,
                    seas_mod=seas_mod,
                    committee_mod=committee_mod,
                    wm_mod=wm_mod,
                )
                live_blob["enabled"] = True

            decision_bar_ts = pd.to_datetime(primary_df["time"].iloc[ii])
            latest_bar_ts = pd.to_datetime(primary_df["time"].iloc[ii + 1])
            decision_close = float(primary_df["close"].iloc[ii])

            labels: Dict[str, Any] = {
                "decision_bar": _dt_to_iso(decision_bar_ts.to_pydatetime(), tz),
                "decision_close": decision_close,
                "candles": {
                    "decision": _candle_dict_at_primary(primary_df, ii, tz),
                    "latest": _candle_dict_at_primary(primary_df, ii + 1, tz),
                },
            }

            for h in label_bars:
                h_int = int(h)
                t_h = pd.to_datetime(primary_df["time"].iloc[ii + h_int])
                close_h = float(primary_df["close"].iloc[ii + h_int])

                labels[f"next_bar_{h_int}"] = _dt_to_iso(t_h.to_pydatetime(), tz)
                labels[f"next_close_{h_int}"] = close_h
                labels[f"delta_{h_int}"] = float(close_h - decision_close)
                labels["candles"][f"next_{h_int}"] = _candle_dict_at_primary(primary_df, ii + h_int, tz)

            payload = {
                "meta": {
                    "instrument": instrument,
                    "mt5_instrument": instrument,
                    "primary_timeframe": primary_tf,
                    "decision_bar": _dt_to_iso(decision_bar_ts.to_pydatetime(), tz),
                    "latest_bar": _dt_to_iso(latest_bar_ts.to_pydatetime(), tz),
                    "decision_idx": int(ii),
                    "intent": "hold",
                    "size_mult": 0.0,
                    "discrete": False,
                    "enforce_hard_rules": False,
                    "only_on_new_bar": True,
                    "execute": False,
                    "hard_block_reasons": block_reasons,
                    "source": "csv_expert_audit",
                    "label_bars": label_bars,
                    "use_info_bus": bool(args.use_info_bus),
                },
                "observation_schema": _to_jsonable(obs_schema),

                # Keep backwards-compatible top-level states/observation as ENV path
                "states": {
                    "market_state": market_env,
                    "expert_signals": experts_env,
                    "committee_state": committee_env,
                    "risk_state": risk_env,
                    "memory_state": memory_env,
                    "account_state": acct_env,
                    "world_model_state": wm_env,
                    "trading_mode_state": mode_env,
                    "governor_state": gov_env,
                },
                "observation": _to_jsonable(obs_env),

                # New: comparison block
                "compare": {
                    "env": {
                        "states": _to_jsonable({
                            "market_state": market_env,
                            "expert_signals": experts_env,
                            "committee_state": committee_env,
                            "risk_state": risk_env,
                            "memory_state": memory_env,
                            "account_state": acct_env,
                            "world_model_state": wm_env,
                            "trading_mode_state": mode_env,
                            "governor_state": gov_env,
                        }),
                        "observation": _to_jsonable(obs_env),
                    },
                    "live": {
                        "states": _to_jsonable(live_blob),
                        "observation": _to_jsonable(obs_live),
                    },
                },

                "mask": None,
                "mask_summary": None,
                "labels": _to_jsonable(labels),
            }

            fname = f"training_pipeline_{inst_slug}_{primary_tf}_{decision_bar_ts.strftime('%Y-%m-%d_%H%M%S')}.json"
            (outdir / fname).write_text(json.dumps(_to_jsonable(payload), indent=indent), encoding="utf-8")
            written += 1

            if int(args.log_every) > 0 and (n % int(args.log_every) == 0):
                LOG.info("Progress: %d/%d snapshots written", n, len(decision_times))

        except Exception as e:
            LOG.warning("Snapshot failed at t=%s: %s", str(t_dec), str(e))
            continue

    LOG.info("Done. Snapshots written: %d (outdir=%s)", written, str(outdir))
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    raise SystemExit(main())
