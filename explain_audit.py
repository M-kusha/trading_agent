#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def minute_key(ts: datetime) -> int:
    if ts.tzinfo is None:

        ts = ts.replace(tzinfo=timezone.utc)
    return int(ts.timestamp() // 60)


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
    if pred == realized:
        return "RIGHT"

    if pred == 0 or realized == 0:
        return "NEUTRAL"
    return "WRONG"


def esc(x: Any) -> str:
    return html.escape("" if x is None else str(x))


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


    decision_close: Optional[float] = None


    committee_action: Optional[str] = None
    committee_pred: Optional[int] = None

    experts_pred: Dict[str, Optional[int]] = field(default_factory=dict)
    htf_pred: Dict[str, Optional[int]] = field(default_factory=dict)
    wm_price_pred: Optional[int] = None
    wm_scenario_pred: Optional[int] = None

    wm_inconsistent: bool = False
    hard_block_reasons: List[str] = field(default_factory=list)


    market_regime: Optional[str] = None
    zone_type: Optional[str] = None
    vol_state: Optional[str] = None
    in_prime_window: Optional[bool] = None
    hour_normalized: Optional[float] = None


    horizon_delta: Dict[int, Optional[float]] = field(default_factory=dict)
    horizon_realized: Dict[int, Optional[int]] = field(default_factory=dict)


def collect_inputs(input_arg: str, recursive: bool = False) -> List[str]:
    if os.path.isdir(input_arg):
        if recursive:
            out: List[str] = []
            for root, _, _files in os.walk(input_arg):
                out.extend(glob.glob(os.path.join(root, "*.json")))
            return sorted(out)
        return sorted(glob.glob(os.path.join(input_arg, "*.json")))
    return sorted(glob.glob(input_arg))


Timeline = Dict[int, Candle]


def _get_time_col(columns: Sequence[str]) -> str:
    cols = [c.strip().lower() for c in columns]
    for c in ("time", "timestamp", "datetime", "date"):
        if c in cols:
            return c
    return cols[0] if cols else "time"


def _parse_dt_any(s: Any) -> Optional[datetime]:
    ss = str(s).strip()
    if not ss:
        return None


    if ss.isdigit():
        try:
            iv = int(ss)
            if len(ss) >= 13:
                return datetime.fromtimestamp(iv / 1000.0, tz=timezone.utc)
            if len(ss) >= 10:
                return datetime.fromtimestamp(iv, tz=timezone.utc)
        except Exception:
            pass


    dt = parse_iso(ss)
    if dt is not None:
        return dt


    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y.%m.%d %H:%M:%S", "%d.%m.%Y %H:%M:%S"):
        try:
            return datetime.strptime(ss, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return None


def _resolve_csv_path(dataset_dir: Path, instrument: str, tf: str, dataset_metadata: Optional[Path]) -> Optional[Path]:
    sym = str(instrument).upper().strip()
    tfu = str(tf).upper().strip()

    meta: Dict[str, Any] = {}
    if dataset_metadata and dataset_metadata.exists():
        try:
            meta = json.loads(dataset_metadata.read_text(encoding="utf-8"))
        except Exception:
            meta = {}


    if isinstance(meta, dict):
        files = meta.get("files", {})
        if isinstance(files, dict):
            sym_files = files.get(sym)
            if isinstance(sym_files, dict):
                v = sym_files.get(tfu)
                if isinstance(v, str) and v:
                    cand = Path(v)
                    if cand.is_absolute() and cand.exists():
                        return cand
                    cand_under = dataset_dir / cand
                    if cand_under.exists():
                        return cand_under
                    if (dataset_dir / cand.name).exists():
                        return dataset_dir / cand.name


    for name in (
        f"{sym}_{tfu}_features.csv",
        f"{sym}_{tfu}.csv",
        f"{sym}_{tfu}_ohlcv.csv",
    ):
        p = dataset_dir / name
        if p.exists():
            return p


    hits = list(dataset_dir.glob(f"{sym}*{tfu}*.csv"))
    return hits[0] if hits else None


def build_timeline_from_csv(
    dataset_dir: str,
    instrument: str,
    tf: str,
    dataset_metadata: Optional[str] = None,
) -> Timeline:
    ds = Path(str(dataset_dir))
    md = Path(str(dataset_metadata)) if dataset_metadata else None
    p = _resolve_csv_path(ds, instrument, tf, md)
    if p is None or (not p.exists()):
        raise SystemExit(f"CSV timeline requested but could not find CSV for {instrument} {tf} under {ds}")

    timeline: Timeline = {}
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit(f"CSV has no header: {p}")

        hdr_map = {c.lower(): c for c in reader.fieldnames}
        time_col = _get_time_col(reader.fieldnames)

        def col(name: str) -> str:
            return hdr_map.get(name.lower(), name)

        time_hdr = col(time_col)

        for row in reader:
            dt = _parse_dt_any(row.get(time_hdr, ""))
            if dt is None:
                continue
            try:
                o = float(row.get(col("open"), ""))
                h = float(row.get(col("high"), ""))
                l = float(row.get(col("low"), ""))
                c = float(row.get(col("close"), ""))
            except Exception:
                continue

            v = 0.0
            try:
                vv = row.get(col("volume"))
                if vv is not None and str(vv).strip():
                    v = float(vv)
            except Exception:
                v = 0.0

            timeline[minute_key(dt)] = Candle(ts=dt, o=o, h=h, l=l, c=c, v=v)

    return timeline


def infer_bar_step_minutes(timeline: Timeline, fallback: int = 15) -> int:
    keys = sorted(timeline.keys())
    if len(keys) < 3:
        return fallback
    diffs: List[int] = []
    for a, b in zip(keys[:-1], keys[1:]):
        d = int(b - a)
        if 1 <= d <= 24 * 60:
            diffs.append(d)
    if not diffs:
        return fallback
    counts: Dict[int, int] = {}
    for d in diffs:
        counts[d] = counts.get(d, 0) + 1
    return max(counts.items(), key=lambda kv: kv[1])[0]


def compute_horizon_realized(audits: List[BarAudit], timeline: Timeline, horizons: List[int], step_minutes: int) -> None:
    for a in audits:
        if not a.decision_ts or a.decision_close is None:
            for h in horizons:
                a.horizon_delta[h] = None
                a.horizon_realized[h] = None
            continue

        dec_key = minute_key(a.decision_ts)
        for h in horizons:
            fut_key = dec_key + int(step_minutes) * int(h)
            future = timeline.get(fut_key)
            if not future:
                a.horizon_delta[h] = None
                a.horizon_realized[h] = None
                continue
            d = float(future.c - a.decision_close)
            a.horizon_delta[h] = d
            a.horizon_realized[h] = sign(d)


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
        ("wm:price", "World model price"),
        ("wm:scenario", "World model scenario"),
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


def natural_horizon_for_predictor(key: PredictorKey) -> Optional[int]:

    if key == "htf:H1":
        return 4
    if key == "htf:H4":
        return 16
    if key == "htf:D1":
        return 96
    return None


def eval_horizon_for_predictor(key: PredictorKey, primary_horizon: int) -> int:
    nh = natural_horizon_for_predictor(key)
    return nh if nh is not None else int(primary_horizon)


def acc_for_horizon(
    rows: List[BarAudit],
    key: PredictorKey,
    horizon: int,
    include_flats: bool,
    only_unblocked: bool,
) -> Tuple[int, int, float]:
    right = 0
    total = 0
    for a in rows:
        if only_unblocked and a.hard_block_reasons:
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


def acc_for_native_next_candle(
    rows: List[BarAudit],
    key: PredictorKey,
    primary_horizon: int,
    include_flats: bool,
    only_unblocked: bool,
) -> Tuple[int, int, float, int]:
    h = eval_horizon_for_predictor(key, primary_horizon)
    r, t, p = acc_for_horizon(rows, key, h, include_flats, only_unblocked)
    return r, t, p, h


def _best_effort_decision_latest_ts(j: Dict[str, Any]) -> Tuple[Optional[datetime], Optional[datetime], str, str]:
    decision_bar = str(safe_get(j, ["meta", "decision_bar"], "")) or ""
    latest_bar = str(safe_get(j, ["meta", "latest_bar"], "")) or ""

    decision_ts = parse_iso(decision_bar) if decision_bar else None
    latest_ts = parse_iso(latest_bar) if latest_bar else None

    if decision_ts is None:
        c_dec = safe_get(j, ["labels", "candles", "decision", "ts"])
        if isinstance(c_dec, str):
            decision_ts = parse_iso(c_dec)
            if not decision_bar:
                decision_bar = c_dec

    if latest_ts is None:
        c_lat = safe_get(j, ["labels", "candles", "latest", "ts"])
        if isinstance(c_lat, str):
            latest_ts = parse_iso(c_lat)
            if not latest_bar:
                latest_bar = c_lat

    return decision_ts, latest_ts, decision_bar, latest_bar


def _parse_env_predictions(j: Dict[str, Any]) -> Tuple[
    Optional[str], Optional[int],
    Dict[str, Optional[int]],
    Dict[str, Optional[int]],
    Optional[int], Optional[int], bool
]:

    committee_action = safe_get(j, ["states", "committee_state", "action"])
    if committee_action is None:
        committee_action = safe_get(j, ["states", "committee_state", "direction"])
    committee_action_s = str(committee_action) if isinstance(committee_action, str) else None
    committee_pred = normalize_action(committee_action_s) if committee_action_s else None


    experts_pred: Dict[str, Optional[int]] = {}
    experts = safe_get(j, ["states", "expert_signals", "experts"], {})
    if isinstance(experts, dict):
        for name in ["trend", "momentum", "theme", "seasonality"]:
            direction = safe_get(experts, [name, "direction"])
            p = dir_from_label(direction) if isinstance(direction, str) else None
            experts_pred[name] = p


    htf_pred: Dict[str, Optional[int]] = {}
    htf = safe_get(j, ["states", "expert_signals", "htf_experts"], {})
    if isinstance(htf, dict):
        for htf_tf in ["H1", "H4", "D1"]:
            td = safe_get(htf, [htf_tf, "trend_direction"])
            p = dir_from_label(td) if isinstance(td, str) else None
            htf_pred[htf_tf] = p


    wm_price_pred = None
    price_changes = safe_get(
        j,
        ["states", "world_model_state", "market_predictions", "latest_predictions", "price_changes"],
    )
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
        wm_price_pred is not None
        and wm_scenario_pred is not None
        and wm_price_pred != 0
        and wm_scenario_pred != 0
        and wm_price_pred != wm_scenario_pred
    )

    return committee_action_s, committee_pred, experts_pred, htf_pred, wm_price_pred, wm_scenario_pred, wm_inconsistent


def audit_one(path: str, tf: str = "M15") -> Optional[BarAudit]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            j = json.load(f)
    except Exception:
        return None

    instrument = str(safe_get(j, ["meta", "instrument"], "")) or ""
    timeframe = str(safe_get(j, ["meta", "primary_timeframe"], tf)) or tf

    decision_ts, latest_ts, decision_bar, latest_bar = _best_effort_decision_latest_ts(j)

    hard_block = safe_get(j, ["meta", "hard_block_reasons"], [])
    if not isinstance(hard_block, list):
        hard_block = []
    hard_block = [str(x) for x in hard_block]

    market_regime = safe_get(j, ["states", "expert_signals", "market", "regime"])
    market_regime = str(market_regime) if isinstance(market_regime, str) else None

    entry_timing = safe_get(j, ["states", "trading_mode_state", "entry_timing"], {})
    entry_timing = entry_timing if isinstance(entry_timing, dict) else {}

    zone_type = entry_timing.get("zone_type")
    vol_state = entry_timing.get("vol_state")
    in_prime = entry_timing.get("in_prime_window")
    hour_norm = entry_timing.get("hour_normalized")

    zone_type_s = str(zone_type) if zone_type is not None else None
    vol_state_s = str(vol_state) if vol_state is not None else None
    in_prime_b = bool(in_prime) if isinstance(in_prime, bool) else None
    try:
        hour_norm_f = float(hour_norm) if isinstance(hour_norm, (int, float)) else None
    except Exception:
        hour_norm_f = None


    decision_close: Optional[float] = None
    dc = safe_get(j, ["labels", "decision_close"])
    if isinstance(dc, (int, float)):
        decision_close = float(dc)

    if decision_close is None:
        c_dec = safe_get(j, ["labels", "candles", "decision"], {})
        if isinstance(c_dec, dict) and "close" in c_dec:
            try:
                decision_close = float(c_dec.get("close", 0.0))
            except Exception:
                decision_close = None

    if decision_close is None:
        closes = safe_get(j, ["states", "market_state", timeframe, "close"])
        if isinstance(closes, list) and len(closes) >= 2:
            try:
                decision_close = float(closes[-2])
            except Exception:
                decision_close = None


    (
        committee_action_s,
        committee_pred,
        experts_pred,
        htf_pred,
        wm_price_pred,
        wm_scenario_pred,
        wm_inconsistent,
    ) = _parse_env_predictions(j)

    return BarAudit(
        file=os.path.basename(path),
        instrument=instrument,
        timeframe=timeframe,
        decision_bar=decision_bar,
        latest_bar=latest_bar,
        decision_ts=decision_ts,
        latest_ts=latest_ts,
        decision_close=decision_close,
        committee_action=committee_action_s,
        committee_pred=committee_pred,
        experts_pred=experts_pred,
        htf_pred=htf_pred,
        wm_price_pred=wm_price_pred,
        wm_scenario_pred=wm_scenario_pred,
        wm_inconsistent=wm_inconsistent,
        hard_block_reasons=hard_block,
        market_regime=market_regime,
        zone_type=zone_type_s,
        vol_state=vol_state_s,
        in_prime_window=in_prime_b,
        hour_normalized=hour_norm_f,
    )


def write_html(
    rows: List[BarAudit],
    out_html: str,
    title: str,
    eval_horizons: List[int],
    include_flats: bool,
    primary_horizon: int,
) -> None:
    css = """
    body{font-family:Segoe UI,Arial,sans-serif;margin:24px;color:#111}
    h1{margin:0 0 6px 0}
    .meta{color:#444;margin:0 0 18px 0;line-height:1.35}
    table{border-collapse:collapse;width:100%;margin:14px 0}
    th,td{border:1px solid #ddd;padding:7px 8px;font-size:13px;vertical-align:top}
    th{background:#f6f6f6;text-align:left;position:sticky;top:0;z-index:1}
    .small{font-size:12px;color:#444}
    .ok{background:#d6ffdd}
    .bad{background:#ffd6d6}
    .warn{background:#fff2cc}
    .na{background:#f3f3f3;color:#666}
    .mono{font-family:Consolas,Menlo,monospace}
    .center{text-align:center}
    """

    total = len(rows)
    blocked = sum(1 for r in rows if r.hard_block_reasons)
    unblocked = total - blocked
    wm_incons = sum(1 for r in rows if r.wm_inconsistent)

    keys = predictors_list()
    key_to_label = dict(keys)

    def dir_txt(pred: Optional[int]) -> str:
        if pred is None:
            return "NA"
        return "UP" if pred == 1 else ("DOWN" if pred == -1 else "FLAT")

    def realized_txt(realized: Optional[int]) -> str:
        if realized is None:
            return "NA"
        return "UP" if realized == 1 else ("DOWN" if realized == -1 else "FLAT")

    def cell_class(pred: Optional[int], realized: Optional[int]) -> str:
        v = verdict(pred, realized)
        if v == "RIGHT":
            return "ok"
        if v == "WRONG":
            return "bad"
        if v == "NEUTRAL":
            return "warn"
        return "na"

    def pred_cell(a: BarAudit, key: PredictorKey) -> str:
        pred = pred_for(a, key)
        h = eval_horizon_for_predictor(key, primary_horizon)
        realized = a.horizon_realized.get(h)
        d = a.horizon_delta.get(h)
        tip = f"{key_to_label.get(key, key)} | eval=H{h} | pred={dir_txt(pred)} | realized={realized_txt(realized)}"
        if d is not None:
            tip += f" | Δ={d:+.2f}"
        cls = cell_class(pred, realized)
        return f"<td class='{cls} center' title='{esc(tip)}'><span class='mono'>{esc(dir_txt(pred))}</span></td>"

    def realized_cell(a: BarAudit, h: int) -> str:
        r = a.horizon_realized.get(h)
        d = a.horizon_delta.get(h)
        txt = realized_txt(r)
        tip = f"Realized H{h}"
        if d is not None:
            tip += f" | Δ={d:+.2f}"
        cls = "na"
        if r is None:
            cls = "na"
        elif r == 0:
            cls = "warn"
        else:
            cls = "ok"
        return f"<td class='{cls} center' title='{esc(tip)}'><span class='mono'>{esc(txt)}</span></td>"

    def summary_table_for_horizon(h: int) -> str:
        lines: List[str] = []
        lines.append(f"<h2>Accuracy — horizon H{h} (all modules graded on the SAME horizon)</h2>")
        lines.append("<table><thead><tr><th>Module</th><th>right</th><th>scored</th><th>acc</th></tr></thead><tbody>")
        for k, label in keys:
            r, t, p = acc_for_horizon(rows, k, h, include_flats, only_unblocked=False)
            lines.append(f"<tr><td>{esc(label)}</td><td>{r}</td><td>{t}</td><td>{p:.2f}%</td></tr>")
        lines.append("</tbody></table>")
        return "\n".join(lines)

    def native_next_candle_table() -> str:

        lines: List[str] = []
        lines.append("<h2>Accuracy — next candle in each module’s timeframe</h2>")
        lines.append("<p class='small'>Committee/Experts/WM are graded on the primary horizon (H"
                    + esc(primary_horizon)
                    + "). HTF H1/H4/D1 are graded on 4/16/96 bars respectively.</p>")
        lines.append("<table><thead><tr><th>Module</th><th>Eval horizon</th><th>right</th><th>scored</th><th>acc</th></tr></thead><tbody>")

        scored_rows: List[Tuple[float, int, int, str, int]] = []
        for k, label in keys:
            r, t, p, eh = acc_for_native_next_candle(rows, k, primary_horizon, include_flats, only_unblocked=False)
            scored_rows.append((p, r, t, label, eh))

        scored_rows.sort(key=lambda x: (x[0], x[2]), reverse=True)
        for p, r, t, label, eh in scored_rows:
            lines.append(f"<tr><td>{esc(label)}</td><td>H{eh}</td><td>{r}</td><td>{t}</td><td>{p:.2f}%</td></tr>")

        lines.append("</tbody></table>")
        return "\n".join(lines)


    lines: List[str] = []
    lines.append("<!doctype html><html><head><meta charset='utf-8'/>")
    lines.append(f"<title>{esc(title)}</title><style>{css}</style></head><body>")
    lines.append(f"<h1>{esc(title)}</h1>")
    lines.append(
        f"<p class='meta'>Bars audited: <b>{total}</b> | "
        f"Blocked: <b>{blocked}</b> | Unblocked: <b>{unblocked}</b><br/>"
        f"Primary timeframe: <b>M15</b> | Primary grading horizon (short-term): <b>H{primary_horizon}</b><br/>"
        f"WM inconsistent (env): <b>{wm_incons}</b></p>"
    )


    lines.append(native_next_candle_table())
    for h in eval_horizons:
        lines.append(summary_table_for_horizon(h))


    show_realized_h = [h for h in [1, 4, 16] if any(h in a.horizon_realized for a in rows)]
    lines.append("<h2>Per-bar detail — color-coded correctness</h2>")
    lines.append("<p class='small'>Each module cell is colored using its own evaluation horizon: "
                 f"Committee/Experts/WM → H{primary_horizon}; HTF H1→H4; HTF H4→H16; HTF D1→H96. "
                 "Hover a cell to see details (horizon, Δ, realized).</p>")

    lines.append("<table><thead><tr>")
    lines.append("<th>Decision</th><th>Regime</th><th>Zone</th><th>Vol</th><th>Prime</th><th>Blocked</th><th>File</th>")
    for h in show_realized_h:
        lines.append(f"<th>Realized H{h}</th>")
    for k, label in keys:
        lines.append(f"<th>{esc(label)}</th>")
    lines.append("</tr></thead><tbody>")

    def prime_txt(b: Optional[bool]) -> str:
        if b is True:
            return "prime"
        if b is False:
            return "off"
        return "NA"

    for a in rows:
        lines.append("<tr>")
        lines.append(f"<td class='mono'>{esc(a.decision_bar)}</td>")
        lines.append(f"<td>{esc(a.market_regime or 'NA')}</td>")
        lines.append(f"<td>{esc(a.zone_type or 'NA')}</td>")
        lines.append(f"<td>{esc(a.vol_state or 'NA')}</td>")
        lines.append(f"<td>{esc(prime_txt(a.in_prime_window))}</td>")
        lines.append(f"<td class='center'>{'YES' if a.hard_block_reasons else 'NO'}</td>")
        lines.append(f"<td class='mono'>{esc(a.file)}</td>")

        for h in show_realized_h:
            lines.append(realized_cell(a, h))

        for k, _label in keys:
            lines.append(pred_cell(a, k))

        lines.append("</tr>")

    lines.append("</tbody></table>")
    lines.append("</body></html>")

    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def parse_horizons(s: str) -> List[int]:
    out: List[int] = []
    for part in str(s).split(","):
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
    ap.add_argument("--title", default="Explain Audit Report (Short-Term)", help="HTML report title.")
    ap.add_argument("--timeframe", default="M15", help="Primary timeframe key (default: M15).")


    ap.add_argument("--primary-horizon", type=int, default=1,
                    help="Horizon (in M15 bars) used to grade committee/experts/WM (default: 1 = next M15 candle).")
    ap.add_argument("--eval-horizons", default="1,4",
                    help="Comma-separated horizons (M15 bars) to produce summary accuracy tables (default: 1,4).")
    ap.add_argument("--include_flats", action="store_true",
                    help="Include flat realized bars in accuracy counts (default: False).")


    ap.add_argument("--dataset-dir", required=True, help="Directory containing processed OHLCV CSVs.")
    ap.add_argument("--instrument", required=True, help="Instrument symbol for CSV timeline (e.g., XAUUSD).")
    ap.add_argument("--dataset-metadata", default=None,
                    help="Optional metadata.json used to resolve per-symbol/per-TF CSV paths.")

    args = ap.parse_args()

    tf = str(args.timeframe).upper().strip() or "M15"
    instrument = str(args.instrument).upper().strip()

    paths = collect_inputs(args.input, recursive=bool(args.recursive))
    if not paths:
        raise SystemExit(f"No JSON files found for input: {args.input}")

    outdir = args.outdir
    if outdir is None:
        outdir = args.input if os.path.isdir(args.input) else os.path.dirname(os.path.abspath(args.input))
        if not outdir:
            outdir = os.getcwd()
    os.makedirs(outdir, exist_ok=True)


    timeline = build_timeline_from_csv(
        dataset_dir=str(args.dataset_dir),
        instrument=instrument,
        tf=tf,
        dataset_metadata=(str(args.dataset_metadata) if args.dataset_metadata else None),
    )

    step_minutes = infer_bar_step_minutes(timeline, fallback=15)


    audits: List[BarAudit] = []
    for p in paths:
        a = audit_one(p, tf=tf)
        if a:
            audits.append(a)

    audits.sort(key=lambda a: (a.decision_ts is None, a.decision_ts or datetime.min, a.file))

    eval_horizons = parse_horizons(args.eval_horizons)
    primary_h = int(args.primary_horizon)


    needed = set(eval_horizons + [primary_h, 4, 16, 96])
    compute_horizon_realized(audits, timeline, sorted(needed), step_minutes)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_html = os.path.join(outdir, f"explain_audit_shortterm_{stamp}.html")

    write_html(
        audits,
        out_html,
        args.title,
        eval_horizons=eval_horizons,
        include_flats=bool(args.include_flats),
        primary_horizon=primary_h,
    )

    print(f"[OK] Files scanned     : {len(paths)}")
    print(f"[OK] Bars audited      : {len(audits)}")
    print(f"[OK] Timeline bars     : {len(timeline)} (epoch-minute keys)")
    print(f"[OK] Inferred bar step : {step_minutes} minutes")
    print(f"[OK] Primary horizon   : {primary_h}")
    print(f"[OK] Eval horizons     : {eval_horizons}")
    print(f"[OK] HTML              : {out_html}")


if __name__ == "__main__":
    main()
