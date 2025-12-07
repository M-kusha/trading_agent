# ─────────────────────────────────────────────────────────────
# File: traindashboard/server.py
# Real-time Training Dashboard API Server (v2.0)
# 
# Streams data directly from InfoBus instance via WebSocket
# Can run standalone OR embedded in training script (same process)
# 
# Comprehensive dashboard with:
# - PPO Learning metrics
# - Trading performance
# - Risk management
# - Memory system
# - Strategy/Bias analysis
# - Position management
# - Executor status
# - Voting/Consensus
# - Module health & thesis tracking
# ─────────────────────────────────────────────────────────────

import asyncio
import json
import time
import sys
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import InfoBus - will use the SAME instance as training when run in same process
from modules.utils.info_bus import InfoBusManager

app = FastAPI(title="Training Dashboard API v2.0")

# Global flag to track if server is running
_server_running = False
_server_thread: Optional[threading.Thread] = None

# CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Connected WebSocket clients
connected_clients: List[WebSocket] = []

def safe_float(val: Any, default: float = 0.0) -> float:
    """Safely convert to float."""
    if val is None:
        return default
    if isinstance(val, dict):
        for k in ['value', 'gate', 'score', 'confidence', 'level', 'scale', 'weight']:
            if k in val:
                try:
                    return float(val[k])
                except:
                    pass
        return default
    try:
        return float(val)
    except:
        return default

def safe_str(val: Any, default: str = "UNKNOWN") -> str:
    """Safely convert to string."""
    if val is None:
        return default
    if isinstance(val, dict):
        for k in ['value', 'regime', 'level', 'state', 'direction', 'action']:
            if k in val:
                return str(val[k])
        return default
    return str(val)

def safe_dict(val: Any, default: Optional[Dict] = None) -> Dict:
    """Safely convert to dict."""
    if default is None:
        default = {}
    if val is None:
        return default
    if isinstance(val, dict):
        return val
    return default

def safe_list(val: Any, default: Optional[List] = None) -> List:
    """Safely convert to list."""
    if default is None:
        default = []
    if val is None:
        return default
    if isinstance(val, list):
        return val
    return default

def get_dashboard_data() -> Dict[str, Any]:
    """Extract all dashboard data from InfoBus - comprehensive version."""
    try:
        bus = InfoBusManager.get_instance()
    except Exception as e:
        return {"error": f"InfoBus not available: {e}", "timestamp": time.time()}
    
    def get(key: str, default: Any = None) -> Any:
        try:
            return bus.get(key, "Dashboard", default=default)
        except:
            return default
    
    def get_with_thesis(key: str) -> Dict[str, Any]:
        """Get value and thesis info for a key."""
        try:
            val = bus.get(key, "Dashboard", default=None)
            # Try to get metadata
            meta = {}
            if hasattr(bus, '_data') and key in bus._data:
                entry = bus._data[key]
                if isinstance(entry, dict):
                    meta = {
                        'module': entry.get('module', 'Unknown'),
                        'thesis': entry.get('thesis', ''),
                        'timestamp': entry.get('timestamp', 0),
                    }
            return {'value': val, 'meta': meta}
        except:
            return {'value': None, 'meta': {}}
    
    # ═══════════════════════════════════════════════════════════════
    # REWARD / ENV METRICS
    # ═══════════════════════════════════════════════════════════════
    reward_components = safe_dict(get("reward_components", {}))
    latest_reward = safe_dict(get("latest_reward_breakdown", {}))
    reward = {
        "step_reward": safe_float(get("step_reward", latest_reward.get("total", 0))),
        "episode_reward": safe_float(get("current_episode_reward", 0)),
        "components": {
            "pnl": safe_float(reward_components.get("pnl", latest_reward.get("pnl", 0))),
            "risk": safe_float(reward_components.get("risk", latest_reward.get("risk", 0))),
            "drawdown": safe_float(reward_components.get("drawdown", latest_reward.get("drawdown", 0))),
            "behavior": safe_float(reward_components.get("behavior", latest_reward.get("behavior", 0))),
        },
        "thesis": latest_reward.get("thesis", ""),
    }

    # ═══════════════════════════════════════════════════════════════
    # LEARNING METRICS (PPO Agent)
    # ═══════════════════════════════════════════════════════════════
    learning = {
        "policy_loss": safe_float(get("policy_loss", 0)),
        "value_loss": safe_float(get("value_loss", 0)),
        "entropy": safe_float(get("entropy_loss", get("entropy", 0))),
        "kl_divergence": safe_float(get("approx_kl", 0)),
        "clip_fraction": safe_float(get("clip_fraction", 0)),
        "explained_variance": safe_float(get("explained_variance", 0)),
        "learning_rate": safe_float(get("learning_rate", 3e-4)),
        "n_updates": int(safe_float(get("n_updates", 0))),
        "current_reward": safe_float(get("current_episode_reward", 0)),
        "mean_reward": safe_float(get("ep_rew_mean", get("episode_reward_mean", 0))),
    }
    
    # ═══════════════════════════════════════════════════════════════
    # PPO AGENT (Intelligent Arbiter)
    # ═══════════════════════════════════════════════════════════════
    ppo_decision = safe_dict(get("ppo_final_decision", {}))
    ppo_multi = safe_dict(get("ppo_multi_decision", {}))
    ppo_stats = safe_dict(get("ppo_instrument_stats", {}))
    training_metrics = safe_dict(get("training_metrics", {}))
    
    ppo = {
        "gate_passed": get("ppo_gate_passed", False),
        "position_size": safe_float(get("ppo_position_size", 0)),
        "final_decision": {
            "action": safe_str(ppo_decision.get("action", ppo_decision.get("direction", "HOLD"))),
            "confidence": safe_float(ppo_decision.get("confidence", 0)),
            "reasoning": ppo_decision.get("reasoning", ""),
            "committee_action": safe_str(ppo_decision.get("committee_action", "")),
            "expert_consensus": safe_dict(ppo_decision.get("expert_consensus", {})),
            "regime": safe_str(ppo_decision.get("regime", "")),
            "trust_score": safe_float(ppo_decision.get("trust_score", 0)),
            "meta": safe_dict(ppo_decision.get("meta", {})),
        },
        "multi_decisions": {},
        "stats": {
            "total_decisions": training_metrics.get("total_decisions", 0),
            "trades_executed": training_metrics.get("trades_executed", 0),
            "accuracy": safe_float(training_metrics.get("accuracy", 0)),
        }
    }
    
    # Per-instrument PPO decisions
    if isinstance(ppo_multi, dict):
        instruments_data = ppo_multi.get("instruments", ppo_multi)
        for symbol, dec in instruments_data.items():
            if isinstance(dec, dict):
                ppo["multi_decisions"][symbol] = {
                    "direction": safe_str(dec.get("direction", dec.get("action", "HOLD"))),
                    "confidence": safe_float(dec.get("confidence", 0)),
                    "gate_passed": dec.get("gate_passed", False),
                    "position_size": safe_float(dec.get("position_size", 0)),
                    "reasoning": dec.get("reasoning", ""),
                    "committee_action": safe_str(dec.get("committee_action", "")),
                    "expert_consensus": safe_dict(dec.get("expert_consensus", {})),
                    "regime": safe_str(dec.get("regime", "")),
                    "trust_score": safe_float(dec.get("trust_score", 0)),
                    "meta": safe_dict(dec.get("meta", {})),
                }
    
    # ═══════════════════════════════════════════════════════════════
    # TRAINING PROGRESS
    # ═══════════════════════════════════════════════════════════════
    progress = {
        "timestep": int(safe_float(get("timestep", get("training_step", 0)))),
        "total_timesteps": int(safe_float(get("total_timesteps", 100000))),
        "episode": int(safe_float(get("episode", get("episodes", 0)))),
        "steps_per_second": safe_float(get("steps_per_second", get("fps", 0))),
    }
    progress["progress_pct"] = (progress["timestep"] / progress["total_timesteps"] * 100) if progress["total_timesteps"] > 0 else 0
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING PERFORMANCE
    # ═══════════════════════════════════════════════════════════════
    balance = safe_float(get("balance", get("env_balance", 100000)))
    initial_balance = safe_float(get("initial_balance", 100000))
    if initial_balance == 0:
        initial_balance = 100000
    
    equity = safe_float(get("equity", balance))
    pnl = balance - initial_balance
    pnl_pct = ((balance / initial_balance) - 1) * 100 if initial_balance > 0 else 0
    drawdown = safe_float(get("current_drawdown", get("env_drawdown", 0)))
    
    trades = get("trades", get("trade_data", []))
    if isinstance(trades, list):
        total_trades = len(trades)
        wins = sum(1 for t in trades if isinstance(t, dict) and safe_float(t.get('pnl', t.get('profit', 0))) > 0)
        losses = sum(1 for t in trades if isinstance(t, dict) and safe_float(t.get('pnl', t.get('profit', 0))) < 0)
    elif isinstance(trades, dict):
        total_trades = len(trades)
        wins = sum(1 for t in trades.values() if isinstance(t, dict) and safe_float(t.get('pnl', t.get('profit', 0))) > 0)
        losses = sum(1 for t in trades.values() if isinstance(t, dict) and safe_float(t.get('pnl', t.get('profit', 0))) < 0)
    else:
        total_trades = int(safe_float(get("total_trades", 0)))
        wins = 0
        losses = 0
    
    completed = wins + losses
    win_rate = (wins / completed * 100) if completed > 0 else 0
    
    trading = {
        "balance": balance,
        "equity": equity,
        "initial_balance": initial_balance,
        "pnl": pnl,
        "pnl_pct": pnl_pct,
        "drawdown": drawdown,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
    }
    
    # ═══════════════════════════════════════════════════════════════
    # RISK MANAGEMENT (DynamicRiskController)
    # ═══════════════════════════════════════════════════════════════
    risk_scaling = safe_dict(get("risk_scaling", {}))
    risk_assessment = safe_dict(get("risk_assessment", {}))
    risk_factors = safe_dict(get("risk_factors", {}))
    risk_analytics = safe_dict(get("risk_analytics", {}))
    
    risk = {
        "level": safe_str(get("risk_level", risk_assessment.get('level', 'UNKNOWN'))),
        "scale": safe_float(get("risk_scale", risk_scaling.get('scale', 1.0))),
        "score": safe_float(get("risk_score", 0.5)),
        "fragility": safe_float(get("fragility", get("fragility_score", 0.5))),
        "volatility": safe_float(get("volatility", get("current_volatility", 0))),
        "trend_strength": safe_float(get("trend_strength", 0)),
        "factors": {
            "market": safe_float(risk_factors.get("market_risk", 0)),
            "position": safe_float(risk_factors.get("position_risk", 0)),
            "volatility": safe_float(risk_factors.get("volatility_risk", 0)),
            "drawdown": safe_float(risk_factors.get("drawdown_risk", 0)),
        },
        "thesis": risk_assessment.get("_thesis", risk_scaling.get("_thesis", "")),
        "alerts": safe_list(get("risk_alerts", []))[:5],
    }
    
    # ═══════════════════════════════════════════════════════════════
    # MEMORY SYSTEM (UnifiedMemory)
    # ═══════════════════════════════════════════════════════════════
    memory_gate_raw = get("memory_gate", {})
    danger_zones = safe_dict(get("danger_zones", {}))
    playbook_recall = safe_dict(get("playbook_recall", {}))
    intuition = safe_dict(get("intuition_vector", {}))
    mistake_avoidance = safe_dict(get("mistake_avoidance", {}))
    memory_rationale = get("memory_rationale", "")
    
    memory = {
        "gate": safe_float(memory_gate_raw.get('gate', memory_gate_raw) if isinstance(memory_gate_raw, dict) else memory_gate_raw, 1.0),
        "vote": safe_str(get("memory_vote", "NEUTRAL")),
        "rationale": memory_rationale if isinstance(memory_rationale, str) else str(memory_rationale),
        "danger_zones": {
            "active": len(danger_zones.get("zones", [])) if isinstance(danger_zones, dict) else 0,
            "severity": safe_float(danger_zones.get("max_severity", 0)),
        },
        "playbook": {
            "match_found": playbook_recall.get("match_found", False),
            "confidence": safe_float(playbook_recall.get("confidence", 0)),
            "pattern": playbook_recall.get("pattern_name", ""),
        },
        "intuition_score": safe_float(intuition.get("score", 0.5)),
        "mistake_signals": len(safe_list(mistake_avoidance.get("active_signals", []))),
        "neural_risk_hint": safe_float(get("neural_risk_hint", 0.5)),
    }
    
    # ═══════════════════════════════════════════════════════════════
    # STRATEGY (BiasAuditor, ThesisEvolution, CurriculumPlanner)
    # ═══════════════════════════════════════════════════════════════
    bias_analysis = safe_dict(get("bias_analysis", {}))
    bias_corrections = safe_dict(get("bias_corrections", {}))
    psychological_state = safe_dict(get("psychological_state", {}))
    curriculum_stage = safe_dict(get("curriculum_stage", {}))
    best_thesis = safe_dict(get("best_thesis", get("market_thesis", {})))
    
    strategy = {
        "bias": {
            "detected": len(bias_analysis.get("detected_biases", [])) if isinstance(bias_analysis, dict) else 0,
            "severity": safe_float(bias_analysis.get("overall_severity", 0)),
            "top_bias": bias_analysis.get("top_bias", {}).get("name", "None") if isinstance(bias_analysis.get("top_bias"), dict) else "None",
            "corrections_applied": len(bias_corrections.get("individual_corrections", {})) if isinstance(bias_corrections, dict) else 0,
        },
        "psychology": {
            "state": safe_str(psychological_state.get("state", "NEUTRAL")),
            "confidence_level": safe_float(psychological_state.get("confidence", 0.5)),
            "tilt_risk": safe_float(psychological_state.get("tilt_risk", 0)),
        },
        "curriculum": {
            "stage": safe_str(curriculum_stage.get("current_stage", curriculum_stage.get("stage", "UNKNOWN"))),
            "progress": safe_float(curriculum_stage.get("progress", 0)),
            "mastery": safe_float(curriculum_stage.get("mastery_score", 0)),
        },
        "thesis": {
            "active": best_thesis.get("thesis", best_thesis.get("name", "None")),
            "confidence": safe_float(best_thesis.get("confidence", 0)),
            "rationale": best_thesis.get("rationale", ""),
        },
    }
    
    # ═══════════════════════════════════════════════════════════════
    # POSITIONS (PositionManager)
    # ═══════════════════════════════════════════════════════════════
    positions_raw = get("positions", get("current_positions", {}))
    position_health = safe_dict(get("position_health", {}))
    portfolio_state = safe_dict(get("portfolio_state", {}))
    
    positions = []
    if isinstance(positions_raw, dict):
        for symbol, pos in positions_raw.items():
            if isinstance(pos, dict) and pos.get('direction', pos.get('side', 'flat')) != 'flat':
                positions.append({
                    "symbol": symbol,
                    "direction": pos.get('direction', pos.get('side', 'unknown')),
                    "size": safe_float(pos.get('size', pos.get('lots', 0))),
                    "entry_price": safe_float(pos.get('entry_price', pos.get('entry', 0))),
                    "current_pnl": safe_float(pos.get('pnl', pos.get('unrealized_pnl', 0))),
                    "duration_mins": safe_float(pos.get('duration_mins', 0)),
                })
    elif isinstance(positions_raw, list):
        for pos in positions_raw:
            if isinstance(pos, dict):
                positions.append({
                    "symbol": pos.get('symbol', pos.get('instrument', 'unknown')),
                    "direction": pos.get('direction', pos.get('side', 'unknown')),
                    "size": safe_float(pos.get('size', pos.get('lots', 0))),
                    "entry_price": safe_float(pos.get('entry_price', pos.get('entry', 0))),
                    "current_pnl": safe_float(pos.get('pnl', pos.get('unrealized_pnl', 0))),
                    "duration_mins": safe_float(pos.get('duration_mins', 0)),
                })
    
    position_data = {
        "positions": positions,
        "count": len(positions),
        "total_exposure": safe_float(portfolio_state.get("total_exposure", 0)),
        "health_score": safe_float(position_health.get("score", 1.0)),
    }
    
    # ═══════════════════════════════════════════════════════════════
    # EXECUTOR
    # ═══════════════════════════════════════════════════════════════
    execution_reports = safe_list(get("execution_reports", []))
    trading_result = safe_dict(get("trading_result", {}))
    account_state = safe_dict(get("account_state", {}))
    pending_orders = safe_list(get("pending_orders", []))
    
    # Executor decision explanations
    executor_decisions = []
    for rpt in execution_reports[-20:]:
        if isinstance(rpt, dict):
            executor_decisions.append({
                "id": rpt.get("id", rpt.get("ticket", "")),
                "symbol": rpt.get("symbol", rpt.get("instrument", "")),
                "action": safe_str(rpt.get("action", rpt.get("side", ""))),
                "volume": safe_float(rpt.get("volume", rpt.get("lots", 0))),
                "price": safe_float(rpt.get("price", rpt.get("open_price", 0))),
                "status": safe_str(rpt.get("status", rpt.get("result", ""))),
                "reason": rpt.get("reason", rpt.get("thesis", "")),
                "gate_reason": rpt.get("gate_reason", ""),
                "risk_reason": rpt.get("risk_reason", ""),
                "timestamp": rpt.get("timestamp", rpt.get("time", "")),
            })

    executor = {
        "last_action": safe_str(trading_result.get("action", "NONE")),
        "last_result": safe_str(trading_result.get("result", "N/A")),
        "pending_orders": len(pending_orders),
        "execution_count": len(execution_reports),
        "decisions": executor_decisions,
        "account": {
            "balance": safe_float(account_state.get("balance", balance)),
            "equity": safe_float(account_state.get("equity", equity)),
            "margin_used": safe_float(account_state.get("margin_used", 0)),
            "margin_free": safe_float(account_state.get("margin_free", 0)),
        },
    }
    
    # ═══════════════════════════════════════════════════════════════
    # RECENT TRADES (fills from Executor)
    # ═══════════════════════════════════════════════════════════════
    recent_trades_raw = get("recent_trades", [])
    recent_trades = []
    if isinstance(recent_trades_raw, list):
        for t in recent_trades_raw[-10:]:
            if isinstance(t, dict):
                # Handle direction: prefer string, convert int side to string
                direction = t.get('direction', t.get('type', ''))
                if not direction or direction == 'unknown':
                    side = t.get('side', 0)
                    if isinstance(side, (int, float)):
                        direction = 'BUY' if side > 0 else 'SELL' if side < 0 else 'HOLD'
                    else:
                        direction = str(side) if side else 'unknown'
                
                # Get price (fills have 'price', not entry/exit)
                price = safe_float(t.get('price', t.get('entry_price', t.get('exit_price', 0))))
                
                recent_trades.append({
                    "symbol": t.get('symbol', t.get('instrument', 'unknown')),
                    "direction": direction,
                    "pnl": safe_float(t.get('pnl', t.get('realized_pnl', t.get('profit', 0)))),
                    "price": price,
                    "entry_price": price,  # For compatibility
                    "exit_price": price,   # For compatibility  
                    "action": t.get('action', t.get('comment', '')),
                    "step": t.get('step', 0),
                    "timestamp": t.get('timestamp', t.get('ts', t.get('close_time', ''))),
                    "lots": safe_float(t.get('lots', t.get('volume', 0))),
                })
    
    # ═══════════════════════════════════════════════════════════════
    # VOTING & CONSENSUS
    # ═══════════════════════════════════════════════════════════════
    vote = safe_dict(get("trade_vote_v2", get("kernel_decision", {})))
    committee_votes = safe_dict(get("committee_votes", {}))
    consensus = safe_dict(get("consensus_result", get("final_consensus", {})))
    collusion = safe_dict(get("collusion_result", {}))
    
    voting = {
        "action": safe_str(vote.get('action', vote.get('direction', 'HOLD'))),
        "confidence": safe_float(vote.get('confidence', 0)),
        "consensus_score": safe_float(get("consensus_score", consensus.get('score', 0))),
        "agreement_score": safe_float(get("agreement_score", 0)),
        "collusion_score": safe_float(collusion.get('score', get("collusion_score", 0))),
        "collusion_detected": collusion.get('detected', False),
        "expert_votes": [],
    }
    
    # Extract expert votes
    if isinstance(committee_votes, dict):
        for expert, vote_data in committee_votes.items():
            if isinstance(vote_data, dict):
                voting["expert_votes"].append({
                    "name": expert,
                    "vote": safe_str(vote_data.get('vote', vote_data.get('action', 'HOLD'))),
                    "confidence": safe_float(vote_data.get('confidence', vote_data.get('weight', 0.5))),
                    "thesis": vote_data.get('thesis', ''),
                })
    
    # Per-instrument decisions
    instrument_decisions = safe_dict(get("committee_decisions_by_instrument", get("arbiter_instrument_signals", {})))
    instruments = {}
    if isinstance(instrument_decisions, dict):
        for symbol, decision in instrument_decisions.items():
            if isinstance(decision, dict):
                instruments[symbol] = {
                    "action": safe_str(decision.get('action', decision.get('direction', 'HOLD'))),
                    "confidence": safe_float(decision.get('confidence', 0)),
                    "consensus": safe_float(decision.get('consensus_score', decision.get('consensus', 0))),
                }
    voting["instruments"] = instruments
    
    # ═══════════════════════════════════════════════════════════════
    # MARKET DATA
    # ═══════════════════════════════════════════════════════════════
    market_regime = safe_str(get("market_regime", "UNKNOWN"))
    market_context = safe_dict(get("market_context", {}))
    liquidity = safe_dict(get("liquidity_score", get("market_liquidity", {})))
    
    market = {
        "regime": market_regime,
        "trend_direction": safe_str(market_context.get("trend", get("trend_direction", "NEUTRAL"))),
        "volatility_level": safe_str(get("volatility_level", "MEDIUM")),
        "liquidity_score": safe_float(liquidity.get("score", liquidity) if isinstance(liquidity, dict) else liquidity, 0.5),
        "session": safe_str(get("session_type", get("trading_session", "UNKNOWN"))),
    }
    
    # Prices
    prices = {}
    for symbol in ['EURUSD', 'XAUUSD', 'EUR_USD', 'XAU_USD']:
        price = get(f"price_{symbol}", get(f"{symbol}_price", 0))
        if price:
            clean_symbol = symbol.replace('_', '')
            prices[clean_symbol] = safe_float(price)
    
    # ═══════════════════════════════════════════════════════════════
    # MODULE HEALTH & THESIS TRACKING
    # ═══════════════════════════════════════════════════════════════
    module_health = safe_dict(get("module_health", {}))
    module_performance = safe_dict(get("module_performance", {}))
    
    modules = []
    key_modules = ['PPOAgent', 'DynamicRiskController', 'UnifiedMemory', 'BiasAuditor', 
                   'PositionManager', 'Executor', 'SlimVotingKernel', 'CommitteeCoordinator']
    
    for name in key_modules:
        health = module_health.get(name, {})
        perf = module_performance.get(name, {})
        modules.append({
            "name": name,
            "status": health.get('status', 'unknown') if isinstance(health, dict) else 'unknown',
            "last_run_ms": safe_float(health.get('last_run_ms', 0) if isinstance(health, dict) else 0),
            "success_rate": safe_float(perf.get('success_rate', 1.0) if isinstance(perf, dict) else 1.0),
        })
    
    # ═══════════════════════════════════════════════════════════════
    # WORLD MODEL PREDICTIONS
    # ═══════════════════════════════════════════════════════════════
    predictions = safe_dict(get("market_predictions", {}))
    prediction_confidence = safe_float(get("prediction_confidence", 0))
    
    world_model = {
        "price_prediction": safe_float(predictions.get("predicted_price_change", 0)),
        "volatility_prediction": safe_float(predictions.get("predicted_volatility", 0)),
        "confidence": prediction_confidence,
        "scenario": safe_str(predictions.get("scenario", "NEUTRAL")),
    }
    
    # ═══════════════════════════════════════════════════════════════
    # TRADING MODE
    # ═══════════════════════════════════════════════════════════════
    trading_mode_raw = get("trading_mode", {})
    mode_config = safe_dict(get("mode_config", {}))
    
    trading_mode = {
        "current": safe_str(trading_mode_raw.get("mode", trading_mode_raw) if isinstance(trading_mode_raw, dict) else trading_mode_raw, "NORMAL"),
        "risk_multiplier": safe_float(mode_config.get("risk_multiplier", 1.0)),
        "max_positions": mode_config.get("max_positions", 3),
        "thesis": mode_config.get("thesis", ""),
    }
    
    payload: Dict[str, Any] = {
        "timestamp": float(time.time()),
        "datetime": datetime.now().isoformat(),
        "learning": learning,
        "reward": reward,
        "ppo": ppo,
        "progress": progress,
        "trading": trading,
        "risk": risk,
        "memory": memory,
        "strategy": strategy,
        "position_data": position_data,
        "executor": executor,
        "recent_trades": recent_trades,
        "voting": voting,
        "market": market,
        "prices": prices,
        "modules": modules,
        "world_model": world_model,
        "trading_mode": trading_mode,
    }

    # Ensure everything is JSON-serializable (avoid numpy / custom objects)
    def json_safe(obj: Any) -> Any:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            if isinstance(obj, dict):
                return {str(k): json_safe(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [json_safe(v) for v in obj]
            if isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            return str(obj)

    return json_safe(payload)


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    frontend_path = Path(__file__).parent / "index.html"
    return FileResponse(frontend_path)


@app.get("/api/data")
async def get_data():
    """REST endpoint for dashboard data."""
    return JSONResponse(get_dashboard_data())


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        bus = InfoBusManager.get_instance()
        return {"status": "ok", "bus_active": True}
    except:
        return {"status": "ok", "bus_active": False}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time streaming."""
    await websocket.accept()
    connected_clients.append(websocket)
    
    try:
        # Send initial data
        await websocket.send_json(get_dashboard_data())
        
        # Stream updates
        while True:
            data = get_dashboard_data()
            await websocket.send_json(data)
            await asyncio.sleep(0.1)  # 10 updates per second
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


# Mount static files (for potential CSS/JS files)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


def _run_server_blocking(host: str = "0.0.0.0", port: int = 8765):
    """Run server in blocking mode (for background thread)."""
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


def start_dashboard_server(host: str = "0.0.0.0", port: int = 8765) -> threading.Thread:
    """
    Start the dashboard server in a background thread.
    Call this from training script to share the same InfoBus instance.
    
    Returns the thread object (daemon thread - will stop when main process exits).
    """
    global _server_running, _server_thread
    
    if _server_running and _server_thread and _server_thread.is_alive():
        print("[Dashboard] Server already running")
        return _server_thread
    
    print("=" * 60)
    print("  TRAINING DASHBOARD SERVER (Background)")
    print("=" * 60)
    print(f"  Open http://localhost:{port} in your browser")
    print("=" * 60)
    
    _server_thread = threading.Thread(
        target=_run_server_blocking,
        args=(host, port),
        daemon=True,  # Will stop when main process exits
        name="DashboardServer"
    )
    _server_thread.start()
    _server_running = True
    
    # Give server a moment to start
    time.sleep(0.5)
    
    return _server_thread


def stop_dashboard_server():
    """Stop the dashboard server (if running)."""
    global _server_running
    _server_running = False
    # Daemon thread will stop automatically when main process exits


def main():
    """Run the dashboard server standalone."""
    print("=" * 60)
    print("  TRAINING DASHBOARD SERVER")
    print("=" * 60)
    print(f"  Open http://localhost:8765 in your browser")
    print("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8765,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
