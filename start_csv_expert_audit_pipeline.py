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
from datetime import datetime, timedelta, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import numpy as np
from zoneinfo import ZoneInfo

from envs.core.env_types import PropFirmConfig, load_risk_policy
from envs.prop_firm_env import PropFirmTradingEnv
from modules.meta.ppo_observation_builder import PPOObservationBuilder

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
        return int(getattr(tv, "hour")) * 60 + int(getattr(tv, "minute"))
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
    Return iloc for last candle with time <= t (searchsorted).
    Pylance-safe: avoid relying on df.index.tz attribute types.
    """
    if df.empty:
        return None

    idx_tz = _index_tz(df)
    tt = t
    if tt.tzinfo is None and idx_tz is not None:
        tt = tt.tz_localize(idx_tz)  # type: ignore[arg-type]
    elif tt.tzinfo is not None and idx_tz is not None and str(tt.tzinfo) != str(idx_tz):
        tt = tt.tz_convert(idx_tz)  # type: ignore[arg-type]

    # searchsorted works on DatetimeIndex
    pos_obj = df.index.searchsorted(tt, side="right")
    # Convert searchsorted result using numpy to handle scalars, arrays and python ints
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

    # Need next bars for labels up to max horizon
    last_allowed_i = len(times) - 1 - int(max_horizon_bars)
    if last_allowed_i <= 0:
        return out

    for i, t in enumerate(times):
        if i < int(warmup_bars):
            continue
        if i > last_allowed_i:
            break

        tt = pd.to_datetime(t)
        # Ensure in tz (should already be)
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

    # Select decision times (scalar loop => no pandas stub complaints)
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
        env._session_trades = 0  # noqa: SLF001
        env.session_start_balance = float(initial_balance)
        env.session_pnl = 0.0
    except Exception:
        pass

    inst_slug = _safe_symbol_slug(instrument)
    indent = 2 if bool(args.pretty) else None

    written = 0
    for n, t_dec in enumerate(decision_times, start=1):
        try:
            # As-of primary index
            i = _asof_iloc(primary_df, t_dec)
            if i is None:
                continue
            ii = int(i)

            # Ensure horizons available
            if ii + max_h >= len(primary_df):
                continue

            # Set env pointer
            try:
                env._episode_instrument = instrument  # noqa: SLF001
            except Exception:
                pass
            env.current_step = int(ii)
            env.episode_bars = int(ii)

            # Simulated session block reasons
            block_reasons = _compute_block_reasons(pd.to_datetime(t_dec), cfg=cfg)

            # Build states
            market_state = env._prepare_market_data(instrument)
            expert_signals = env._prepare_expert_signals(instrument)
            committee_state = env._prepare_committee_state(expert_signals)
            risk_state = env._prepare_risk_state()
            memory_state = env._prepare_memory_state(instrument)
            trading_mode_state = env._prepare_trading_mode_state(instrument)
            world_model_state = env._prepare_world_model_state(instrument, expert_signals, committee_state)
            governor_state = env._get_governor_state()
            account_state = env._prepare_account_state(instrument)

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

            decision_bar_ts = pd.to_datetime(primary_df["time"].iloc[ii])
            latest_bar_ts = pd.to_datetime(primary_df["time"].iloc[ii + 1])

            decision_close = float(primary_df["close"].iloc[ii])

            labels: Dict[str, Any] = {
                "decision_bar": _dt_to_iso(decision_bar_ts.to_pydatetime(), tz),
                "decision_close": decision_close,
            }
            for h in label_bars:
                h_int = int(h)
                t_h = pd.to_datetime(primary_df["time"].iloc[ii + h_int])
                close_h = float(primary_df["close"].iloc[ii + h_int])
                labels[f"next_bar_{h_int}"] = _dt_to_iso(t_h.to_pydatetime(), tz)
                labels[f"next_close_{h_int}"] = close_h
                labels[f"delta_{h_int}"] = float(close_h - decision_close)

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
                },
                "observation_schema": _to_jsonable(obs_schema),
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
                "observation": _to_jsonable(obs),
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
