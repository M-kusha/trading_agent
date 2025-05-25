from __future__ import annotations
import os
import json
import datetime
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ────── Import your backend/trader logic here ──────
from .state_backend import StateBackend  # ← adapt as needed

backend = StateBackend()
logger = logging.getLogger("live_api")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

app = FastAPI(title="NeuroTrader Pro API", version="2.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)
trader = None  # assigned in start_api()

# ═════════════════════════ Healthcheck ═════════════════════════
@app.get("/health")
def health():
    try:
        env = trader.env
        return {
            "status": "ok",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "balance": getattr(env, "balance", 0),
            "mode": getattr(env, "mode_manager", None) and env.mode_manager.get_mode(),
            "modules_online": sum(1 for m in env.get_module_status().values() if m.get("enabled")),
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse({"status": "error", "error": str(e)}, status_code=500)

# ════════════════════ Pydantic Models (full) ══════════════════
class Status(BaseModel):
    balance: float
    drawdown: float
    pnl: float
    intuition_norm: float
    clusters: int
    playbook_size: int
    active_positions: int
    risk_exposure: float
    votes: Dict[str, float]
    modules_online: Optional[int] = 0

class ModuleStatus(BaseModel):
    name: str
    enabled: bool
    confidence: float
    online: Optional[bool] = True
    last_error: Optional[str] = None
    last_participation: Optional[str] = None

class ModuleToggle(BaseModel):
    enabled: bool

class DebugToggle(BaseModel):
    debug: bool

class ModeStatus(BaseModel):
    mode: str

class ModeToggle(BaseModel):
    mode: str

# ══════════════════════════ Mode Endpoints ══════════════════════
@app.put("/mode")
def set_mode(req: ModeToggle):
    try:
        if hasattr(trader.env, "mode_manager"):
            trader.env.mode_manager.set_mode(req.mode)
        else:
            trader.env.set_mode(req.mode)
        return {"status": "success", "mode": req.mode}
    except Exception as e:
        logger.error(f"Mode change error: {e}")
        return JSONResponse({"error": str(e)}, status_code=400)

@app.get("/mode")
def get_mode():
    stats = trader.env.mode_manager.get_stats()
    return {
        "mode": stats["mode"],
        "auto": stats["auto"],
        "last_switch_time": stats["last_switch_time"],
        "reason": stats["last_reason"],
        "win_rate": stats["win_rate"],
        "drawdown": stats["drawdown"],
        "volatility": stats["volatility"],
    }


# ════════════════════ Dashboard summary/status ═══════════════════
@app.get("/status", response_model=Status)
def get_status():
    env = trader.env
    try:
        active_pos = len(getattr(env, "open_positions", []))
    except Exception:
        active_pos = 0
    try:
        risk_pct = float(env.risk_system.get_total_exposure()) * 100
    except Exception:
        limits = env.risk_system.get_limits(env.balance).values()
        risk_pct = float(sum(limits)) * 100
    mode = (
        env.mode_manager.get_mode() if hasattr(env, "mode_manager")
        else getattr(env, "mode", "normal")
    )
    modules_online = sum(1 for m in env.get_module_status().values() if m.get("enabled"))
    return Status(
        balance        = round(env.balance, 2),
        drawdown       = round(env.current_drawdown, 4),
        pnl            = env._ep_pnls[-1] if env._ep_pnls else 0.0,
        intuition_norm = float(np.linalg.norm(env.memory_compressor.intuition_vector)),
        clusters       = getattr(getattr(env.mistake_memory, "_kmeans", None), "n_clusters", 0),
        playbook_size  = len(env.playbook_memory._features),
        active_positions = active_pos,
        risk_exposure    = round(risk_pct, 2),
        votes          = env.get_votes_history()[-1] if env.get_votes_history() else {},
        trade_mode     = mode,
        modules_online = modules_online,
    )

# ══════════════════════ Detailed Module Status ═════════════════════
@app.get("/modules/status", response_model=List[ModuleStatus])
def modules_status():
    env = trader.env
    status = env.get_module_status()
    now = datetime.datetime.utcnow().isoformat()
    result = []
    for name, info in status.items():
        m = getattr(env, name, None)
        result.append(ModuleStatus(
            name=name,
            enabled=info.get("enabled", False),
            confidence=info.get("confidence", 0.0),
            online=True if info.get("enabled", False) else False,
            last_error=getattr(m, "last_error", None) if m else None,
            last_participation=getattr(m, "last_run", None) if m and hasattr(m, "last_run") else now,
        ))
    return result

@app.get("/modules/{name}/detail")
def module_detail(name: str):
    env = trader.env
    m = getattr(env, name, None)
    if not m:
        raise HTTPException(404, f"No module named {name}")
    return {
        "name": name,
        "enabled": env.module_enabled.get(name, True),
        "online": True,
        "last_error": getattr(m, "last_error", None),
        "last_participation": getattr(m, "last_run", None),
        "details": m.__dict__,
    }

@app.get("/modules", response_model=List[ModuleStatus])
def get_modules():
    status = trader.env.get_module_status()
    return [
        ModuleStatus(
            name=name,
            enabled=info.get("enabled", False),
            confidence=info.get("confidence", 0.0),
        )
        for name, info in status.items()
    ]

@app.put("/modules/{name}", response_model=ModuleStatus)
def toggle_module(name: str, req: ModuleToggle):
    try:
        trader.env.set_module_enabled(name, req.enabled)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No such module '{name}'")
    info = trader.env.get_module_status().get(name, {})
    return ModuleStatus(name=name, enabled=req.enabled, confidence=info.get("confidence", 0.0))

# ══════════════════════ Trades, Logs, Votes, Trace, etc ═════════════════════
@app.get("/trades")
def get_trades():
    ts = getattr(trader, "trades", [])
    last_votes = trader.env.get_votes_history()[-1] if trader.env.get_votes_history() else {}
    now = datetime.datetime.utcnow().isoformat()
    out = []
    for t in ts:
        out.append(
            {
                "time": now,
                "instrument": t.get("instrument", ""),
                "size": t.get("size", 0.0),
                "pnl": t.get("pnl", 0.0),
                "exit_reason": t.get("exit_reason", ""),
                "votes_by_tf": t.get(
                    "votes_by_tf",
                    {"H1": last_votes, "H4": last_votes, "D1": last_votes},
                ),
                "big_action": t.get("big_action", []),
            }
        )
    return out

@app.get("/logs")
def get_logs():
    path = "live_trading.log"
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()[-100:]
    except Exception as e:
        logger.error(f"Error reading live_trading.log: {e}")
        return []

@app.get("/votes_history")
def votes_history():
    path = "logs/votes_history.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not read votes_history.json: {e}")
        return []

@app.get("/reasoning_trace")
def get_reasoning_trace():
    return trader.env.get_reasoning_trace()

@app.put("/debug", response_model=DebugToggle)
def set_debug(req: DebugToggle):
    trader.env.debug = req.debug
    return req

@app.get("/volatility_profile")
def volatility_profile():
    return trader.env.get_volatility_profile()

@app.get("/genome_metrics")
def genome_metrics():
    return trader.env.get_genome_metrics()

@app.get("/debug/bars")
def debug_bars(n: int = 5) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        curr_vols = trader.env.get_volatility_profile()
        for inst, tfs in trader.env.data.items():
            out[inst] = {}
            for tf, df in tfs.items():
                tail = df.iloc[-n:]
                times = [ts.isoformat() if hasattr(ts, "isoformat") else str(ts) for ts in tail.index]
                rec = {
                    "time": times,
                    "close": tail["close"].tolist() if "close" in df.columns else [],
                }
                rec["volatility"] = (
                    tail["volatility"].tolist()
                    if "volatility" in df.columns
                    else [curr_vols.get(inst, 0.0)] * len(times)
                )
                out[inst][tf] = rec
        return out
    except Exception:
        logger.exception("Error in /debug/bars")
        raise HTTPException(status_code=500, detail="Debug bars failed")

# ══════════════ Market Hologram, Themes, Patterns, World, etc ══════════════
@app.get("/quantum_map")
def get_quantum_map() -> Dict[str, List[Dict[str, Any]]]:
    corr_matrix = trader.env.get_instrument_correlations()
    if not corr_matrix:
        return {"nodes": [{"id": "loading…", "size": 1.0, "color": "#888888"}], "links": []}
    instruments = {i for pair in corr_matrix.keys() for i in pair}
    vols = trader.env.get_volatility_profile()
    nodes = [
        {"id": inst, "size": float(vols.get(inst, 1.0)), "color": f"#{(hash(inst) & 0xFFFFFF):06x}"}
        for inst in instruments
    ]
    links = [
        {"source": i1, "target": i2, "value": abs(float(corr))}
        for (i1, i2), corr in corr_matrix.items()
    ]
    return {"nodes": nodes, "links": links}

@app.get("/neuro_activity")
def get_neuro_activity() -> Dict[str, Any]:
    try:
        return trader.env.get_neuro_activity()
    except Exception:
        logger.exception("Error in /neuro_activity")
        return JSONResponse({"error": "Neuro activity unavailable"}, status_code=503)

@app.get("/market_themes")
def get_market_themes():
    try:
        km = trader.env.theme_detector.km
        return {
            "cluster_centers": km.cluster_centers_.tolist(),
            "feature_labels": [f"feat_{i}" for i in range(km.n_features_in_)],
        }
    except Exception as e:
        logger.error(f"Market themes error: {e}")
        return {}

@app.get("/fractal_patterns")
def get_fractal_patterns():
    try:
        df = trader.env.data[next(iter(trader.env.data))]["D1"]
        return {
            "time": df.index[-100:].astype(str).tolist(),
            "open": df["open"][-100:].tolist(),
            "high": df["high"][-100:].tolist(),
            "low": df["low"][-100:].tolist(),
            "close": df["close"][-100:].tolist(),
            "fractal_times": df.index[-20::5].astype(str).tolist(),
            "fractal_prices": df["close"][-20::5].tolist(),
        }
    except Exception as e:
        logger.error(f"Fractal patterns error: {e}")
        return {}

@app.get("/world_predictions")
def get_world_predictions():
    try:
        actual = (
            trader.env.data["XAU/USD"]["D1"]["close"][-50:].pct_change().dropna().tolist()
        )
        preds = trader.env.world_model.simulate(
            np.array(actual[-30:], np.float32), np.ones(30, np.float32)
        )
        return {"actual": actual[-len(preds):], "predicted": preds[:, 0].tolist()}
    except Exception as e:
        logger.error(f"World predictions error: {e}")
        return {}

# ══════════════════════ Liquidity, Risk, Opponent, Shadow ═════════════════════
@app.get("/liquidity_map")
def get_liquidity_map():
    layer = trader.env.liquidity_layer
    price_levels = [p for p, _ in layer.bids] + [p for p, _ in layer.asks]
    bids         = [q for _, q in layer.bids]
    asks         = [q for _, q in layer.asks]
    return {
        "price_levels": price_levels,
        "bids":         bids,
        "asks":         asks,
    }

@app.get("/stress_test/{scenario}")
def stress_test(scenario: str):
    base_vol = np.mean(list(trader.env.get_volatility_profile().values()) or [0.0])
    factors = {
        "flash-crash":  (2.50, 0.30),
        "rate-spike":   (1.80, 0.50),
        "default-wave": (1.40, 0.65),
    }
    vol_mul, liq_mul = factors.get(scenario, (1.0, 1.0))
    return {
        "volatility": round(base_vol * vol_mul, 4),
        "liquidity":  round(liq_mul, 4),
    }

@app.get("/risk_limits")
def get_risk_limits():
    return trader.env.risk_system.get_limits(trader.env.balance)

@app.get("/risk_exposure")
def get_risk_exposure():
    limits = trader.env.risk_system.get_limits(trader.env.balance)
    return {"labels": list(limits.keys()), "values": list(limits.values())}

@app.get("/opponent_profiles")
def get_opponent_profiles():
    enhancer = trader.env.opp_enhancer
    labels = list(enhancer.pnl.keys())
    features = [enhancer.pnl[m] for m in labels]
    return {"labels": labels, "features": features}

@app.get("/shadow_performance")
def get_shadow_performance():
    pnls = getattr(trader.env, "_ep_pnls", [])
    return [{"step": i, "pnl": float(p)} for i, p in enumerate(pnls)]

# ══════════════════════ Miscellaneous & Advanced ═════════════════════
@app.get("/balance_history")
def balance_history():
    path = "balance_history.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error reading balance_history.json: {e}")
        return []

# (Optional: advanced parametric endpoints, e.g. vol_profile...)

# ══════════════════════ Startup ═════════════════════
def start_api(system):
    global trader
    trader = system
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
