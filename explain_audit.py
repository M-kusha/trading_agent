"""
Explain Log Auditor v2 — multi-file, multi-horizon, barrier outcomes

Key upgrades vs v1:
- Builds a unified candle timeline across ALL JSON files (using decision_bar and latest_bar candles).
- Scores predictors on multiple horizons (e.g., 1,4,8 bars ahead).
- Optional ATR-based barrier outcome (TP/SL first hit) to estimate "tradable" correctness.
- Produces CSV + HTML with:
    * accuracy per predictor per horizon
    * blocked vs unblocked breakdown
    * world-model inconsistency stats
    * per-bar drilldown including horizon deltas

Usage (Windows):
  python explain_audit_v2.py --input "C:\\Users\\Kushtrimi\\Desktop\\AI\\logs\\explain" --recursive
  python explain_audit_v2.py --input "C:\\Users\\Kushtrimi\\Desktop\\AI\\logs\\explain" --horizons "1,4,8" --barrier_horizon 8 --atr_mult 1.0
  python explain_audit_v2.py --input "C:\\Users\\Kushtrimi\\Desktop\\AI\\logs\\explain\\training_pipeline_XAUUSD_M15_2026-01-16_*.json"

Notes:
- Candle timeline uses:
    * decision_bar candle from OHLC[-2]
    * latest_bar candle from OHLC[-1]
  across every file. Duplicates are resolved by keeping the most recently seen record.
- Horizons are in number of bars (M15 bars by default). Horizon=4 means 1 hour ahead.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Iterable, cast


# ------------------------- helpers -------------------------

def parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def safe_get(d: Dict[str, Any], path: List[str], default=None):
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0

def dir_from_label(label: Optional[str]) -> Optional[int]:
    if not label:
        return None
    s = label.strip().lower()
    if s in {"bull", "bullish", "long", "buy", "up"}:
        return 1
    if s in {"bear", "bearish", "short", "sell", "down"}:
        return -1
    if s in {"neutral", "hold", "flat"}:
        return 0
    return None

def normalize_action(action: Optional[str]) -> Optional[int]:
    return dir_from_label(action)

def verdict(pred: Optional[int], realized: Optional[int]) -> str:
    if pred is None or realized is None:
        return "NA"
    if pred == 0:
        return "NEUTRAL"
    # realized can be 0; if you consider flat as no-score, filter before calling
    return "RIGHT" if pred == realized else "WRONG"

def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


# ------------------------- data models -------------------------

@dataclass
class Candle:
    ts: datetime
    o: float
    h: float
    l: float
    c: float
    v: float = 0.0

@dataclass
class BarAudit:
    file: str
    instrument: str
    timeframe: str

    decision_bar: str
    latest_bar: str
    decision_ts: Optional[datetime]
    latest_ts: Optional[datetime]

    decision_close: Optional[float]
    latest_close: Optional[float]
    delta_1: Optional[float]
    realized_dir_1: Optional[int]  # 1-bar realized direction

    committee_action: Optional[str]
    committee_pred: Optional[int]

    experts_pred: Dict[str, Optional[int]]
    htf_pred: Dict[str, Optional[int]]

    wm_price_pred: Optional[int]
    wm_scenario_pred: Optional[int]
    wm_inconsistent: bool

    hard_block_reasons: List[str]
    market_regime: Optional[str] = None

    # Optional extra context (for richer offline auditing / dataset building)
    meta_intent: Optional[str] = None
    meta_size_mult: Optional[float] = None
    meta_enforce_hard_rules: Optional[bool] = None

    entry_timing: Dict[str, Any] = field(default_factory=dict)
    risk_state: Dict[str, Any] = field(default_factory=dict)
    governor_state: Dict[str, Any] = field(default_factory=dict)

    observation: Optional[List[float]] = None
    observation_feature_names: Optional[List[str]] = None
    mask: Optional[List[bool]] = None
    mask_summary: Dict[str, Any] = field(default_factory=dict)

    # computed later (multi-horizon)
    horizon_delta: Dict[int, Optional[float]] = field(default_factory=dict)
    horizon_realized: Dict[int, Optional[int]] = field(default_factory=dict)    

    # barrier outcomes later
    # outcome per predictor not stored (we compute summaries), but per-bar we store barrier vs committee as example
    atr_at_decision: Optional[float] = None
    barrier_outcome_committee: Optional[str] = None  # WIN / LOSS / NONE / BOTH / NA


# ------------------------- candle extraction -------------------------

def _extract_ohlcv_at_index(j: Dict[str, Any], tf: str, idx: int) -> Optional[Tuple[float, float, float, float, float]]:
    ms = safe_get(j, ["states", "market_state", tf], {})
    if not isinstance(ms, dict):
        return None
    o = safe_get(ms, ["open"])
    h = safe_get(ms, ["high"])
    l = safe_get(ms, ["low"])
    c = safe_get(ms, ["close"])
    v = safe_get(ms, ["volume"])

    # Ensure the fields are lists before using len/indexing; cast for type-checkers
    if not all(isinstance(x, list) for x in [o, h, l, c]):
        return None

    o_list = cast(List[Any], o)
    h_list = cast(List[Any], h)
    l_list = cast(List[Any], l)
    c_list = cast(List[Any], c)
    v_list = cast(List[Any], v) if isinstance(v, list) else []

    if len(o_list) <= abs(idx) or len(h_list) <= abs(idx) or len(l_list) <= abs(idx) or len(c_list) <= abs(idx):
        return None
    try:
        oo = float(o_list[idx])
        hh = float(h_list[idx])
        ll = float(l_list[idx])
        cc = float(c_list[idx])
        vv = float(v_list[idx]) if isinstance(v_list, list) and len(v_list) > abs(idx) else 0.0
        return oo, hh, ll, cc, vv
    except Exception:
        return None

def extract_candles_from_file(j: Dict[str, Any], tf: str = "M15") -> List[Candle]:
    """
    Pulls exactly two candles per file:
      - decision_bar candle from arrays[-2]
      - latest_bar candle from arrays[-1]
    These two candles, accumulated across files, reconstruct a larger timeline.
    """
    out: List[Candle] = []
    dec_s = safe_get(j, ["meta", "decision_bar"])
    lat_s = safe_get(j, ["meta", "latest_bar"])
    if not isinstance(dec_s, str) or not isinstance(lat_s, str):
        return out

    dec_ts = parse_iso(dec_s)
    lat_ts = parse_iso(lat_s)
    if dec_ts:
        ohlcv = _extract_ohlcv_at_index(j, tf, -2)
        if ohlcv:
            oo, hh, ll, cc, vv = ohlcv
            out.append(Candle(dec_ts, oo, hh, ll, cc, vv))
    if lat_ts:
        ohlcv = _extract_ohlcv_at_index(j, tf, -1)
        if ohlcv:
            oo, hh, ll, cc, vv = ohlcv
            out.append(Candle(lat_ts, oo, hh, ll, cc, vv))
    return out


# ------------------------- audit extraction (per file) -------------------------

def audit_one(path: str, tf: str = "M15") -> Optional[BarAudit]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return None

    instrument = str(safe_get(j, ["meta", "instrument"], ""))
    timeframe = str(safe_get(j, ["meta", "primary_timeframe"], tf)) or tf

    decision_bar = str(safe_get(j, ["meta", "decision_bar"], ""))
    latest_bar = str(safe_get(j, ["meta", "latest_bar"], ""))

    decision_ts = parse_iso(decision_bar)
    latest_ts = parse_iso(latest_bar)

    hard_block = safe_get(j, ["meta", "hard_block_reasons"], [])
    if not isinstance(hard_block, list):
        hard_block = []
    hard_block = [str(x) for x in hard_block]

    # regime tag (if available)
    market_regime = safe_get(j, ["states", "expert_signals", "market", "regime"])
    market_regime = str(market_regime) if isinstance(market_regime, str) else None

    # PPO / observation context (optional fields; present in training_pipeline_*.json dumps)
    meta_intent = safe_get(j, ["meta", "intent"])
    meta_intent_s = str(meta_intent).lower().strip() if isinstance(meta_intent, str) else None

    meta_size_mult = safe_get(j, ["meta", "size_mult"])
    try:
        meta_size_mult_f = float(meta_size_mult) if meta_size_mult is not None else None
    except Exception:
        meta_size_mult_f = None

    meta_enforce_hard_rules = safe_get(j, ["meta", "enforce_hard_rules"])
    meta_enforce_hard_rules_b = bool(meta_enforce_hard_rules) if isinstance(meta_enforce_hard_rules, bool) else None

    entry_timing = safe_get(j, ["states", "trading_mode_state", "entry_timing"], {})
    entry_timing = entry_timing if isinstance(entry_timing, dict) else {}

    risk_state = safe_get(j, ["states", "risk_state"], {})
    risk_state = risk_state if isinstance(risk_state, dict) else {}

    governor_state = safe_get(j, ["states", "governor_state"], {})
    governor_state = governor_state if isinstance(governor_state, dict) else {}

    obs_raw = safe_get(j, ["observation"])
    observation: Optional[List[float]] = None
    if isinstance(obs_raw, list):
        out_obs: List[float] = []
        for x in obs_raw:
            try:
                out_obs.append(float(x))
            except Exception:
                out_obs.append(0.0)
        observation = out_obs

    fn_raw = safe_get(j, ["observation_schema", "feature_names"])
    observation_feature_names: Optional[List[str]] = None
    if isinstance(fn_raw, list) and all(isinstance(x, str) for x in fn_raw):
        observation_feature_names = [str(x) for x in fn_raw]

    # Best-effort fallback for older snapshots without schema. This keeps
    # audits self-describing as long as the observation size matches.
    if observation and not observation_feature_names:
        try:
            from modules.meta.ppo_observation_builder import PPO_OBS_FEATURE_NAMES  # type: ignore

            if len(PPO_OBS_FEATURE_NAMES) == len(observation):
                observation_feature_names = list(PPO_OBS_FEATURE_NAMES)
        except Exception:
            pass

    mask_raw = safe_get(j, ["mask"])
    mask: Optional[List[bool]] = None
    if isinstance(mask_raw, list):
        mask = [bool(x) for x in mask_raw]

    mask_summary_raw = safe_get(j, ["mask_summary"], {})
    mask_summary = mask_summary_raw if isinstance(mask_summary_raw, dict) else {}

    # realized 1-bar direction (within-file)
    decision_close = None
    latest_close = None
    delta_1 = None
    realized_dir_1 = None

    closes = safe_get(j, ["states", "market_state", timeframe, "close"])
    if isinstance(closes, list) and len(closes) >= 2:
        try:
            decision_close = float(closes[-2])
            latest_close = float(closes[-1])
            delta_1 = latest_close - decision_close
            realized_dir_1 = sign(delta_1)
        except Exception:
            pass

    # committee
    committee_action = safe_get(j, ["states", "committee_state", "action"])
    if committee_action is None:
        committee_action = safe_get(j, ["states", "committee_state", "direction"])
    committee_action_s = str(committee_action) if isinstance(committee_action, str) else None
    committee_pred = normalize_action(committee_action_s) if committee_action_s else None

    # experts
    experts_pred: Dict[str, Optional[int]] = {}
    experts = safe_get(j, ["states", "expert_signals", "experts"], {})
    if isinstance(experts, dict):
        for name in ["trend", "momentum", "theme", "seasonality"]:
            direction = safe_get(experts, [name, "direction"])
            p = dir_from_label(direction) if isinstance(direction, str) else None
            experts_pred[name] = p

    # HTF experts
    htf_pred: Dict[str, Optional[int]] = {}
    htf = safe_get(j, ["states", "expert_signals", "htf_experts"], {})
    if isinstance(htf, dict):
        for htf_tf in ["H1", "H4", "D1"]:
            td = safe_get(htf, [htf_tf, "trend_direction"])
            p = dir_from_label(td) if isinstance(td, str) else None
            htf_pred[htf_tf] = p

    # world model preds
    wm_price_pred = None
    price_changes = safe_get(j, ["states", "world_model_state", "market_predictions", "latest_predictions", "price_changes"])
    if isinstance(price_changes, list) and len(price_changes) >= 1:
        try:
            wm_price_pred = sign(float(price_changes[0]))
        except Exception:
            wm_price_pred = None

    wm_scenario_pred = None
    bull_prob = safe_get(j, ["states", "world_model_state", "scenario_generation", "bullish_probability"])
    if isinstance(bull_prob, (int, float)):
        bp = float(bull_prob)
        wm_scenario_pred = 1 if bp > 0.5 else (-1 if bp < 0.5 else 0)

    wm_inconsistent = (
        wm_price_pred is not None and wm_scenario_pred is not None
        and wm_price_pred != 0 and wm_scenario_pred != 0
        and wm_price_pred != wm_scenario_pred
    )

    return BarAudit(
        file=os.path.basename(path),
        instrument=instrument,
        timeframe=timeframe,
        decision_bar=decision_bar,
        latest_bar=latest_bar,
        decision_ts=decision_ts,
        latest_ts=latest_ts,
        decision_close=decision_close,
        latest_close=latest_close,
        delta_1=delta_1,
        realized_dir_1=realized_dir_1,
        committee_action=committee_action_s,
        committee_pred=committee_pred,
        experts_pred=experts_pred,
        htf_pred=htf_pred,
        wm_price_pred=wm_price_pred,
        wm_scenario_pred=wm_scenario_pred,
        wm_inconsistent=wm_inconsistent,
        hard_block_reasons=hard_block,
        market_regime=market_regime,
        meta_intent=meta_intent_s,
        meta_size_mult=meta_size_mult_f,
        meta_enforce_hard_rules=meta_enforce_hard_rules_b,
        entry_timing=entry_timing,
        risk_state=risk_state,
        governor_state=governor_state,
        observation=observation,
        observation_feature_names=observation_feature_names,
        mask=mask,
        mask_summary=mask_summary,
    )


# ------------------------- input collection -------------------------

def collect_inputs(input_arg: str, recursive: bool = False) -> List[str]:
    if os.path.isdir(input_arg):
        if recursive:
            out = []
            for root, _, _files in os.walk(input_arg):
                out.extend(glob.glob(os.path.join(root, "*.json")))
            return sorted(out)
        return sorted(glob.glob(os.path.join(input_arg, "*.json")))
    return sorted(glob.glob(input_arg))


# ------------------------- timeline + multi-horizon realized -------------------------

def build_timeline(paths: List[str], tf: str = "M15") -> Dict[datetime, Candle]:
    """
    Builds a candle map time->Candle across all files.
    If duplicates exist, later files overwrite earlier ones.
    """
    timeline: Dict[datetime, Candle] = {}
    for p in paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
        except Exception:
            continue
        for c in extract_candles_from_file(j, tf=tf):
            timeline[c.ts] = c
    return timeline

def infer_bar_step_minutes(timeline: Dict[datetime, Candle], fallback: int = 15) -> int:
    """
    Best-effort infer typical bar step in minutes from timeline timestamps.
    """
    ts_sorted = sorted(timeline.keys())
    if len(ts_sorted) < 3:
        return fallback
    diffs = []
    for a, b in zip(ts_sorted[:-1], ts_sorted[1:]):
        d = int((b - a).total_seconds() // 60)
        if 1 <= d <= 240:
            diffs.append(d)
    if not diffs:
        return fallback
    # mode-ish: pick the most common diff
    counts: Dict[int, int] = {}
    for d in diffs:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]

def compute_horizon_realized(
    audits: List[BarAudit],
    timeline: Dict[datetime, Candle],
    horizons: List[int],
    step_minutes: int,
) -> None:
    for a in audits:
        if not a.decision_ts or a.decision_close is None:
            for h in horizons:
                a.horizon_delta[h] = None
                a.horizon_realized[h] = None
            continue

        for h in horizons:
            future_ts = a.decision_ts + timedelta(minutes=step_minutes * h)
            future = timeline.get(future_ts)
            if not future:
                a.horizon_delta[h] = None
                a.horizon_realized[h] = None
                continue
            d = future.c - a.decision_close
            a.horizon_delta[h] = d
            a.horizon_realized[h] = sign(d)

def true_range(curr: Candle, prev_close: float) -> float:
    return max(curr.h - curr.l, abs(curr.h - prev_close), abs(curr.l - prev_close))

def atr_at(
    decision_ts: datetime,
    timeline: Dict[datetime, Candle],
    step_minutes: int,
    period: int = 14,
) -> Optional[float]:
    """
    ATR(period) ending at decision_ts, using candles up to and including decision candle.
    Requires at least period candles including decision candle.
    """
    # gather candles backwards: t, t-1, ... t-(period-1)
    candles: List[Candle] = []
    t = decision_ts
    for _ in range(period):
        c = timeline.get(t)
        if not c:
            return None
        candles.append(c)
        t = t - timedelta(minutes=step_minutes)
    candles = list(reversed(candles))
    # need previous close for TR; use first candle open as proxy if prior missing
    trs: List[float] = []
    prev_close = candles[0].o
    for c in candles:
        tr = true_range(c, prev_close)
        trs.append(tr)
        prev_close = c.c
    if not trs:
        return None
    return sum(trs) / float(len(trs))

def barrier_outcome(
    decision_ts: datetime,
    entry: float,
    direction: int,
    timeline: Dict[datetime, Candle],
    step_minutes: int,
    horizon_bars: int,
    atr: float,
    atr_mult: float,
    both_hit_policy: str = "loss",  # "loss" | "na" | "win"
) -> str:
    """
    Simulate TP/SL first-hit over next N bars using candle high/low.
    Returns: WIN / LOSS / NONE / BOTH / NA
    """
    if direction not in (-1, 1):
        return "NA"
    if atr <= 0:
        return "NA"

    dist = atr * atr_mult
    if direction == 1:
        tp = entry + dist
        sl = entry - dist
    else:
        tp = entry - dist
        sl = entry + dist

    for i in range(1, horizon_bars + 1):
        t = decision_ts + timedelta(minutes=step_minutes * i)
        c = timeline.get(t)
        if not c:
            return "NA"

        hit_tp = (c.h >= tp) if direction == 1 else (c.l <= tp)
        hit_sl = (c.l <= sl) if direction == 1 else (c.h >= sl)

        if hit_tp and hit_sl:
            if both_hit_policy == "loss":
                return "LOSS"
            if both_hit_policy == "win":
                return "WIN"
            return "BOTH"
        if hit_tp:
            return "WIN"
        if hit_sl:
            return "LOSS"

    return "NONE"


# ------------------------- scoring -------------------------

PredictorKey = str

def predictors_list() -> List[Tuple[PredictorKey, str]]:
    return [
        ("committee", "Committee"),
        ("expert:trend", "Expert trend"),
        ("expert:momentum", "Expert momentum"),
        ("expert:theme", "Expert theme"),
        ("expert:seasonality", "Expert seasonality"),
        ("htf:H1", "HTF H1 trend"),
        ("htf:H4", "HTF H4 trend"),
        ("htf:D1", "HTF D1 trend"),
        ("wm:price", "World model price_changes[0]"),
        ("wm:scenario", "World model bullish_probability"),
    ]

def pred_for(a: BarAudit, key: PredictorKey) -> Optional[int]:
    if key == "committee":
        return a.committee_pred
    if key.startswith("expert:"):
        name = key.split(":", 1)[1]
        return a.experts_pred.get(name)
    if key.startswith("htf:"):
        tf = key.split(":", 1)[1]
        return a.htf_pred.get(tf)
    if key == "wm:price":
        return a.wm_price_pred
    if key == "wm:scenario":
        return a.wm_scenario_pred
    return None

def acc_for_horizon(
    rows: List[BarAudit],
    key: PredictorKey,
    horizon: int,
    include_flats: bool,
    only_blocked: Optional[bool] = None,
) -> Tuple[int, int, float]:
    right = 0
    total = 0
    for a in rows:
        if only_blocked is not None:
            is_blocked = bool(a.hard_block_reasons)
            if is_blocked != only_blocked:
                continue

        realized = a.horizon_realized.get(horizon)
        if realized is None:
            continue
        if (not include_flats) and realized == 0:
            continue

        pred = pred_for(a, key)
        v = verdict(pred, realized)
        if v in {"RIGHT", "WRONG"}:
            total += 1
            if v == "RIGHT":
                right += 1
    pct = (right / total * 100.0) if total else 0.0
    return right, total, pct

def barrier_acc(
    rows: List[BarAudit],
    key: PredictorKey,
    include_flats: bool,
    only_blocked: Optional[bool],
    barrier_horizon: int,
    atr_period: int,
    atr_mult: float,
    both_hit_policy: str,
    timeline: Dict[datetime, Candle],
    step_minutes: int,
) -> Tuple[int, int, float]:
    """
    Scores predictor by whether its direction would have hit TP before SL (WIN).
    Counts only cases where predictor has direction +/-1 and outcome is WIN or LOSS.
    """
    wins = 0
    total = 0
    for a in rows:
        if only_blocked is not None:
            is_blocked = bool(a.hard_block_reasons)
            if is_blocked != only_blocked:
                continue
        if not a.decision_ts or a.decision_close is None:
            continue
        pred = pred_for(a, key)
        if pred not in (-1, 1):
            continue

        atrv = atr_at(a.decision_ts, timeline, step_minutes, period=atr_period)
        if atrv is None:
            continue

        out = barrier_outcome(
            decision_ts=a.decision_ts,
            entry=a.decision_close,
            direction=pred,
            timeline=timeline,
            step_minutes=step_minutes,
            horizon_bars=barrier_horizon,
            atr=atrv,
            atr_mult=atr_mult,
            both_hit_policy=both_hit_policy,
        )

        if out in {"WIN", "LOSS"}:
            total += 1
            if out == "WIN":
                wins += 1

    pct = (wins / total * 100.0) if total else 0.0
    return wins, total, pct


# ------------------------- outputs -------------------------

def write_csv(
    rows: List[BarAudit],
    out_csv: str,
    horizons: List[int],
    *,
    include_obs: bool = False,
) -> None:
    base_fields = [
        "file", "instrument", "timeframe",
        "decision_bar", "latest_bar",
        "decision_close", "latest_close", "delta_1", "realized_dir_1",
        "market_regime",
        "intent", "size_mult", "enforce_hard_rules",
        "entry_allowed", "entry_quality_long", "entry_quality_short",
        "zone_type", "vol_state", "in_prime_window", "hour_normalized",
        "current_dd", "daily_dd", "trades_today", "risk_per_trade",
        "gov_loss_layer_ratio", "gov_loss_layer_level", "gov_win_streak_ratio",
        "gov_session_pnl_headroom", "gov_session_trade_budget", "gov_session_consec_loss_ratio",
        "gov_session_progress", "gov_pending_order_progress",
        "mask_hold_allowed", "mask_close_allowed", "mask_long_allowed", "mask_short_allowed",
        "mask_total_allowed", "mask_total_actions",
        "blocked", "hard_block_reasons",
        "committee_action", "committee_pred",
        "expert_trend_pred", "expert_momentum_pred", "expert_theme_pred", "expert_seasonality_pred",
        "htf_H1_pred", "htf_H4_pred", "htf_D1_pred",
        "wm_price_pred", "wm_scenario_pred", "wm_inconsistent",
    ]

    obs_fields: List[str] = []
    obs_names: List[str] = []
    if include_obs:
        for a in rows:
            if a.observation and a.observation_feature_names and len(a.observation) == len(a.observation_feature_names):
                obs_names = list(a.observation_feature_names)
                break
        if not obs_names:
            for a in rows:
                if a.observation:
                    obs_names = [str(i) for i in range(len(a.observation))]
                    break
        obs_fields = [f"obs_{n}" for n in obs_names]

    horizon_fields = []
    for h in horizons:
        horizon_fields += [f"delta_h{h}", f"realized_dir_h{h}"]
    fields = base_fields + obs_fields + horizon_fields

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for a in rows:
            timing = a.entry_timing or {}
            risk = a.risk_state or {}
            gov = a.governor_state or {}

            ms = a.mask_summary or {}
            if not ms and a.mask:
                # Best-effort summary from raw mask vector.
                try:
                    total_actions = int(len(a.mask))
                    total_allowed = int(sum(1 for x in a.mask if x))
                    hold_allowed = bool(a.mask[0]) if total_actions > 0 else False
                    close_allowed = bool(a.mask[-1]) if total_actions > 0 else False
                    k = (total_actions - 2) // 2 if (total_actions >= 4 and (total_actions - 2) % 2 == 0) else 0
                    long_allowed = int(sum(1 for x in a.mask[1 : 1 + k] if x)) if k > 0 else 0
                    short_allowed = int(sum(1 for x in a.mask[1 + k : -1] if x)) if k > 0 else 0
                    ms = {
                        "hold_allowed": hold_allowed,
                        "close_allowed": close_allowed,
                        "long_allowed": long_allowed,
                        "short_allowed": short_allowed,
                        "total_allowed": total_allowed,
                        "total_actions": total_actions,
                    }
                except Exception:
                    ms = {}

            row = {
                "file": a.file,
                "instrument": a.instrument,
                "timeframe": a.timeframe,
                "decision_bar": a.decision_bar,
                "latest_bar": a.latest_bar,
                "decision_close": a.decision_close,
                "latest_close": a.latest_close,
                "delta_1": a.delta_1,
                "realized_dir_1": a.realized_dir_1,
                "market_regime": a.market_regime or "",
                "intent": a.meta_intent or "",
                "size_mult": a.meta_size_mult,
                "enforce_hard_rules": a.meta_enforce_hard_rules,
                "entry_allowed": timing.get("entry_allowed"),
                "entry_quality_long": timing.get("entry_quality_long"),
                "entry_quality_short": timing.get("entry_quality_short"),
                "zone_type": timing.get("zone_type"),
                "vol_state": timing.get("vol_state"),
                "in_prime_window": timing.get("in_prime_window"),
                "hour_normalized": timing.get("hour_normalized"),
                "current_dd": risk.get("current_drawdown"),
                "daily_dd": risk.get("daily_drawdown"),
                "trades_today": risk.get("trades_today"),
                "risk_per_trade": risk.get("risk_per_trade"),
                "gov_loss_layer_ratio": gov.get("loss_layer_ratio"),
                "gov_loss_layer_level": gov.get("loss_layer_level"),
                "gov_win_streak_ratio": gov.get("win_streak_ratio"),
                "gov_session_pnl_headroom": gov.get("session_pnl_headroom"),
                "gov_session_trade_budget": gov.get("session_trade_budget"),
                "gov_session_consec_loss_ratio": gov.get("session_consec_loss_ratio"),
                "gov_session_progress": gov.get("session_progress"),
                "gov_pending_order_progress": gov.get("pending_order_progress"),
                "mask_hold_allowed": ms.get("hold_allowed"),
                "mask_close_allowed": ms.get("close_allowed"),
                "mask_long_allowed": ms.get("long_allowed"),
                "mask_short_allowed": ms.get("short_allowed"),
                "mask_total_allowed": ms.get("total_allowed"),
                "mask_total_actions": ms.get("total_actions"),
                "blocked": bool(a.hard_block_reasons),
                "hard_block_reasons": " | ".join(a.hard_block_reasons),    
                "committee_action": a.committee_action,
                "committee_pred": a.committee_pred,
                "expert_trend_pred": a.experts_pred.get("trend"),
                "expert_momentum_pred": a.experts_pred.get("momentum"),    
                "expert_theme_pred": a.experts_pred.get("theme"),
                "expert_seasonality_pred": a.experts_pred.get("seasonality"),
                "htf_H1_pred": a.htf_pred.get("H1"),
                "htf_H4_pred": a.htf_pred.get("H4"),
                "htf_D1_pred": a.htf_pred.get("D1"),
                "wm_price_pred": a.wm_price_pred,
                "wm_scenario_pred": a.wm_scenario_pred,
                "wm_inconsistent": a.wm_inconsistent,
            }

            if include_obs and obs_fields and obs_names:
                obs = a.observation or []
                for i, n in enumerate(obs_names):
                    row[f"obs_{n}"] = obs[i] if i < len(obs) else None

            for h in horizons:
                row[f"delta_h{h}"] = a.horizon_delta.get(h)
                row[f"realized_dir_h{h}"] = a.horizon_realized.get(h)      
            w.writerow(row)

def write_html(
    rows: List[BarAudit],
    out_html: str,
    title: str,
    horizons: List[int],
    include_flats: bool,
    barrier_horizon: Optional[int],
    atr_period: int,
    atr_mult: float,
    both_hit_policy: str,
    timeline: Dict[datetime, Candle],
    step_minutes: int,
) -> None:
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111}
    h1{margin:0 0 6px 0}
    .meta{color:#444;margin:0 0 18px 0}
    table{border-collapse:collapse;width:100%;margin:14px 0}
    th,td{border:1px solid #ddd;padding:8px;font-size:13px}
    th{background:#f6f6f6;text-align:left;position:sticky;top:0}
    .right{background:#eaffea}
    .wrong{background:#ffecec}
    .na{color:#777}
    .warn{background:#fff2cc}
    .small{font-size:12px;color:#444}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
    """

    inconsistent = sum(1 for r in rows if r.wm_inconsistent)
    blocked_count = sum(1 for r in rows if r.hard_block_reasons)
    unblocked_count = len(rows) - blocked_count

    keys = predictors_list()

    lines: List[str] = []
    lines.append("<!doctype html><html><head><meta charset='utf-8'/>")
    lines.append(f"<title>{esc(title)}</title><style>{css}</style></head><body>")
    lines.append(f"<h1>{esc(title)}</h1>")
    lines.append(
        f"<p class='meta'>Bars audited: <b>{len(rows)}</b> | "
        f"Blocked: <b>{blocked_count}</b> | Unblocked: <b>{unblocked_count}</b> | "
        f"WM inconsistency flags: <b>{inconsistent}</b> | "
        f"Bar step: <b>{step_minutes}m</b></p>"
    )

    # Accuracy tables per horizon (all / blocked / unblocked)
    def acc_table(only_blocked: Optional[bool], caption: str) -> None:
        lines.append(f"<h2>{esc(caption)}</h2>")
        lines.append("<table><thead><tr><th>Predictor</th>")
        for h in horizons:
            lines.append(f"<th>H{h} right</th><th>H{h} scored</th><th>H{h} acc</th>")
        if barrier_horizon is not None:
            lines.append("<th>Barrier right</th><th>Barrier scored</th><th>Barrier acc</th>")
        lines.append("</tr></thead><tbody>")

        for k, label in keys:
            lines.append(f"<tr><td>{esc(label)}</td>")
            for h in horizons:
                r, t, p = acc_for_horizon(rows, k, h, include_flats, only_blocked=only_blocked)
                lines.append(f"<td>{r}</td><td>{t}</td><td>{p:.2f}%</td>")
            if barrier_horizon is not None:
                br, bt, bp = barrier_acc(
                    rows, k,
                    include_flats=include_flats,
                    only_blocked=only_blocked,
                    barrier_horizon=barrier_horizon,
                    atr_period=atr_period,
                    atr_mult=atr_mult,
                    both_hit_policy=both_hit_policy,
                    timeline=timeline,
                    step_minutes=step_minutes,
                )
                lines.append(f"<td>{br}</td><td>{bt}</td><td>{bp:.2f}%</td>")
            lines.append("</tr>")

        lines.append("</tbody></table>")

        if barrier_horizon is not None:
            lines.append(
                f"<p class='small'>Barrier setup: ATR({atr_period}) × {atr_mult:.2f}, "
                f"horizon={barrier_horizon} bars, both-hit policy='{esc(both_hit_policy)}'.</p>"
            )

    lines.append("<div class='grid'>")
    lines.append("<div>")
    acc_table(None, "Accuracy — all bars")
    lines.append("</div><div>")
    acc_table(True, "Accuracy — blocked only")
    lines.append("</div>")
    lines.append("</div>")
    acc_table(False, "Accuracy — unblocked only")

    # Per-bar drilldown
    lines.append("<h2>Per-bar detail</h2>")
    lines.append("<p class='small'>Horizon realized direction is computed from the stitched candle timeline, not only within a single file.</p>")

    # headers
    lines.append("<table><thead><tr>")
    lines.append("<th>Decision</th><th>Regime</th><th>Blocked</th><th>File</th>")
    lines.append("<th>Committee</th><th>Trend</th><th>Momentum</th><th>Theme</th><th>H1</th><th>H4</th><th>D1</th><th>WM price</th><th>WM scenario</th><th>Flags</th>")
    for h in horizons:
        lines.append(f"<th>Δ H{h}</th><th>Realized H{h}</th>")
    lines.append("</tr></thead><tbody>")

    def pred_badge(pred: Optional[int]) -> str:
        if pred is None:
            return "<span class='na'>NA</span>"
        if pred == 1:
            return "UP"
        if pred == -1:
            return "DOWN"
        return "FLAT"

    for a in rows:
        flags = []
        if a.wm_inconsistent:
            flags.append("WM_INCONSISTENT")
        if a.hard_block_reasons:
            flags.append("BLOCKED")
        flags_txt = ", ".join(flags)

        lines.append("<tr>")
        lines.append(f"<td>{esc(a.decision_bar)}</td>")
        lines.append(f"<td>{esc(a.market_regime or '')}</td>")
        lines.append(f"<td>{'YES' if a.hard_block_reasons else 'NO'}</td>")
        lines.append(f"<td>{esc(a.file)}</td>")

        # predictor cells (verdict vs H1 horizon by default? We show direction only; accuracy is in summary)
        lines.append(f"<td>{esc(a.committee_action or '')} / {pred_badge(a.committee_pred)}</td>")
        lines.append(f"<td>{pred_badge(a.experts_pred.get('trend'))}</td>")
        lines.append(f"<td>{pred_badge(a.experts_pred.get('momentum'))}</td>")
        lines.append(f"<td>{pred_badge(a.experts_pred.get('theme'))}</td>")
        lines.append(f"<td>{pred_badge(a.htf_pred.get('H1'))}</td>")
        lines.append(f"<td>{pred_badge(a.htf_pred.get('H4'))}</td>")
        lines.append(f"<td>{pred_badge(a.htf_pred.get('D1'))}</td>")
        lines.append(f"<td>{pred_badge(a.wm_price_pred)}</td>")
        lines.append(f"<td>{pred_badge(a.wm_scenario_pred)}</td>")
        lines.append(f"<td class='warn'>{esc(flags_txt)}</td>")

        for h in horizons:
            d = a.horizon_delta.get(h)
            rd = a.horizon_realized.get(h)
            d_txt = "" if d is None else f"{d:+.2f}"
            r_txt = "NA" if rd is None else ("UP" if rd == 1 else ("DOWN" if rd == -1 else "FLAT"))
            lines.append(f"<td>{esc(d_txt)}</td><td>{esc(r_txt)}</td>")    

        lines.append("</tr>")

        # Extra context line (keeps columns unchanged, but exposes observation inputs)
        ctx_parts: List[str] = []
        if a.meta_intent is not None or a.meta_size_mult is not None:
            sm = "" if a.meta_size_mult is None else f"{a.meta_size_mult:.2f}"
            ehr = "" if a.meta_enforce_hard_rules is None else ("ON" if a.meta_enforce_hard_rules else "OFF")
            ctx_parts.append(f"<b>PPO</b>: intent={esc(a.meta_intent or '')} size={esc(sm)} hard_rules={esc(ehr)}")

        if isinstance(a.entry_timing, dict) and a.entry_timing:
            et = a.entry_timing
            ctx_parts.append(
                "<b>Timing</b>: "
                f"entry_allowed={esc(et.get('entry_allowed'))} "
                f"q_long={esc(et.get('entry_quality_long'))} "
                f"q_short={esc(et.get('entry_quality_short'))} "
                f"zone={esc(et.get('zone_type'))} "
                f"vol={esc(et.get('vol_state'))} "
                f"prime={esc(et.get('in_prime_window'))} "
                f"hour={esc(et.get('hour_normalized'))}"
            )

        if isinstance(a.risk_state, dict) and a.risk_state:
            rs = a.risk_state
            ctx_parts.append(
                "<b>Risk</b>: "
                f"dd={esc(rs.get('current_drawdown'))} "
                f"daily_dd={esc(rs.get('daily_drawdown'))} "
                f"trades_today={esc(rs.get('trades_today'))}"
            )

        if isinstance(a.governor_state, dict) and a.governor_state:
            gs = a.governor_state
            ctx_parts.append(
                "<b>Gov</b>: "
                f"loss_layer={esc(gs.get('loss_layer_ratio'))} "
                f"trade_budget={esc(gs.get('session_trade_budget'))} "
                f"pnl_headroom={esc(gs.get('session_pnl_headroom'))}"
            )

        if isinstance(a.mask_summary, dict) and a.mask_summary:
            ms = a.mask_summary
            ctx_parts.append(
                "<b>Mask</b>: "
                f"allowed={esc(ms.get('total_allowed'))}/{esc(ms.get('total_actions'))} "
                f"hold={esc(ms.get('hold_allowed'))} close={esc(ms.get('close_allowed'))} "
                f"long={esc(ms.get('long_allowed'))} short={esc(ms.get('short_allowed'))}"
            )

        if ctx_parts:
            lines.append("<tr>")
            lines.append(
                f"<td colspan='{14 + len(horizons)*2}' class='small'>"
                + " | ".join(ctx_parts)
                + "</td>"
            )
            lines.append("</tr>")

        if a.hard_block_reasons:
            lines.append("<tr>")
            lines.append(f"<td colspan='{14 + len(horizons)*2}' class='small'><b>Hard blocks:</b> {esc(' | '.join(a.hard_block_reasons))}</td>")
            lines.append("</tr>")

    lines.append("</tbody></table>")
    lines.append("</body></html>")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ------------------------- main -------------------------

def parse_horizons(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
            if v > 0:
                out.append(v)
        except Exception:
            continue
    return sorted(set(out)) or [1]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Folder of .json explain logs OR a glob pattern.")
    ap.add_argument("--recursive", action="store_true", help="If input is a folder, search recursively.")
    ap.add_argument("--outdir", default=None, help="Output directory (default: input folder or glob parent).")
    ap.add_argument("--title", default="Explain Audit Report (v2)", help="HTML report title.")
    ap.add_argument("--timeframe", default="M15", help="Market_state timeframe key (default: M15).")
    ap.add_argument("--horizons", default="1,4,8", help="Comma-separated bar horizons to score (default: 1,4,8).")
    ap.add_argument("--include_flats", action="store_true", help="Include flat realized bars in scoring (default: False).")
    ap.add_argument(
        "--include_obs",
        action="store_true",
        help="Include raw observation vector columns (obs_*) in CSV output.",
    )

    # barrier options
    ap.add_argument("--barrier_horizon", type=int, default=0, help="If >0, compute barrier accuracy over N bars.")
    ap.add_argument("--atr_period", type=int, default=14, help="ATR period for barrier sizing.")
    ap.add_argument("--atr_mult", type=float, default=1.0, help="ATR multiplier for TP/SL distance.")
    ap.add_argument("--both_hit_policy", default="loss", choices=["loss", "na", "win"], help="If TP and SL hit in same bar.")

    args = ap.parse_args()

    horizons = parse_horizons(args.horizons)
    tf = args.timeframe

    paths = collect_inputs(args.input, recursive=args.recursive)
    if not paths:
        raise SystemExit(f"No JSON files found for input: {args.input}")

    # outputs
    outdir = args.outdir
    if outdir is None:
        outdir = args.input if os.path.isdir(args.input) else os.path.dirname(os.path.abspath(args.input))
        if not outdir:
            outdir = os.getcwd()
    os.makedirs(outdir, exist_ok=True)

    # Build timeline first (for multi-horizon realized)
    timeline = build_timeline(paths, tf=tf)
    step_minutes = infer_bar_step_minutes(timeline, fallback=15)

    # Build per-file audits
    audits: List[BarAudit] = []
    for p in paths:
        a = audit_one(p, tf=tf)
        if a:
            audits.append(a)

    # Sort by decision time
    audits.sort(key=lambda a: (a.decision_ts is None, a.decision_ts or datetime.min, a.file))

    # Compute multi-horizon realized directions/deltas using stitched timeline
    compute_horizon_realized(audits, timeline, horizons, step_minutes)

    # Barrier example per-bar (committee only), optional
    barrier_h = int(args.barrier_horizon)
    if barrier_h > 0:
        for a in audits:
            if not a.decision_ts or a.decision_close is None:
                continue
            atrv = atr_at(a.decision_ts, timeline, step_minutes, period=int(args.atr_period))
            a.atr_at_decision = atrv
            if atrv is None:
                a.barrier_outcome_committee = "NA"
                continue
            if a.committee_pred not in (-1, 1):
                a.barrier_outcome_committee = "NA"
                continue
            a.barrier_outcome_committee = barrier_outcome(
                decision_ts=a.decision_ts,
                entry=a.decision_close,
                direction=a.committee_pred,
                timeline=timeline,
                step_minutes=step_minutes,
                horizon_bars=barrier_h,
                atr=atrv,
                atr_mult=float(args.atr_mult),
                both_hit_policy=str(args.both_hit_policy),
            )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(outdir, f"explain_audit_v2_{stamp}.csv")
    out_html = os.path.join(outdir, f"explain_audit_v2_{stamp}.html")

    write_csv(audits, out_csv, horizons, include_obs=bool(args.include_obs))

    write_html(
        audits,
        out_html,
        args.title,
        horizons=horizons,
        include_flats=bool(args.include_flats),
        barrier_horizon=(barrier_h if barrier_h > 0 else None),
        atr_period=int(args.atr_period),
        atr_mult=float(args.atr_mult),
        both_hit_policy=str(args.both_hit_policy),
        timeline=timeline,
        step_minutes=step_minutes,
    )

    print(f"[OK] Files scanned : {len(paths)}")
    print(f"[OK] Bars audited  : {len(audits)}")
    print(f"[OK] Timeline bars : {len(timeline)} (stitched)")
    print(f"[OK] CSV  : {out_csv}")
    print(f"[OK] HTML : {out_html}")

if __name__ == "__main__":
    main()
