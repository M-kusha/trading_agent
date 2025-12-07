# ─────────────────────────────────────────────────────────────
# File: modules/executor/executor.py  (revamped debug wiring + full contract return)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time, uuid, math
import datetime as dt
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.lot_calculator import UnifiedLotCalculator, RiskLevel

from .shared.types import PositionSnap, TradeFill
from .shared.utils import SafeBus, round_to_step, resolve_symbol
from .debug.debugger import ExecutorDebugManager
from .adapters.base_adapter import BaseLiveAdapter, LiveAdapterConfig
from .adapters.mt5_adapter import MT5Adapter
from .unified_logger import UnifiedExecutorLogger, ExecutionCycleEntry

# Smart Position Management
from modules.position.smart_position_manager import (
    SmartPositionManager,
    PositionAction,
)


@dataclass
class ExecutorConfig:
    execution_mode: str = "sim"        # 'sim' | 'live'
    live_broker: str = "mt5"
    symbol_overrides: Optional[Dict[str, str]] = None
    
    # Lot sizing now handled by UnifiedLotCalculator
    # These are fallbacks only - the calculator is the source of truth
    lot_step: float = 0.01
    min_lot: float = 0.01              # Use UnifiedLotCalculator for actual min
    contract_size: float = 100000.0
    price_decimals: int = 5

    min_confidence: float = 0.0
    min_intensity: float = 0.0
    ignore_hold: bool = True
    max_orders_per_step: int = 20

    default_spread: float = 0.0
    slippage_pts: float = 0.0
    commission_per_million: float = 0.0

    prefer_order_queue: bool = True
    read_position_decisions: bool = True
    publish_aliases: bool = True
    allow_runtime_switch: bool = True

    debug_enabled: bool = True
    debug_config: Optional[Dict[str, Any]] = None


@module(**module_args(
    "Executor",
    description="Single-writer order executor for sim/live. Reads order_queue and publishes canonical execution state.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=6000,
    critical=True,
))
class Executor(BaseModule):
    debugger: ExecutorDebugManager

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.cfg = ExecutorConfig(**(config or {}))
        super().__init__(config=asdict(self.cfg))
        # Ensure bus exists for post-super setup (may also be created in _initialize)
        self.bus = SafeBus(InfoBusManager.get_instance())
        self.logger = RotatingLogger("Executor", log_path="logs/executor/executor.log", operator_mode=True)

        # mirrors (sim)
        # Prefer explicit config.initial_balance; else try InfoBus environment_config.initial_balance; else safe default
        _cfg_ib = None
        try:
            _cfg_ib = (config or {}).get("initial_balance", None)
        except Exception:
            _cfg_ib = None
        if _cfg_ib is None:
            try:
                env_cfg = self.bus.get("environment_config", "Executor", default={}) or {}
                _cfg_ib = env_cfg.get("initial_balance", None)
            except Exception:
                _cfg_ib = None
        # Additional fallbacks: try existing market_state/portfolio_metrics before defaulting
        if _cfg_ib is None:
            try:
                ms = self.bus.get("market_state", "Executor", default=None)
                if isinstance(ms, dict):
                    _cfg_ib = ms.get("balance", None)
            except Exception:
                pass
        if _cfg_ib is None:
            try:
                pm = self.bus.get("portfolio_metrics", "Executor", default=None)
                if isinstance(pm, dict):
                    _cfg_ib = pm.get("balance", None)
            except Exception:
                pass
        # Try risk_policy.yaml as source of truth before hardcoded fallback
        if _cfg_ib is None:
            try:
                import yaml
                from pathlib import Path
                risk_policy = Path("config/risk_policy.yaml")
                if risk_policy.exists():
                    with open(risk_policy, "r", encoding="utf-8") as f:
                        rp = yaml.safe_load(f) or {}
                    _cfg_ib = rp.get("prop_firm", {}).get("account_size") or rp.get("lot_sizing", {}).get("account_balance")
            except Exception:
                pass
        # Final fallback (should rarely be reached now)
        self.initial_balance: float = float(100_000.0 if _cfg_ib is None else _cfg_ib)
        self.balance: float = float(self.initial_balance)
        self.equity: float = float(self.balance)
        self._last_equity: float = float(self.equity)
        self.positions: Dict[str, PositionSnap] = {}
        self.trades: List[Dict[str, Any]] = []  # Will be trimmed to last 1000
        self.closed_positions: List[Dict[str, Any]] = []  # Track closed positions for win rate
        self.step_idx: int = 0
        self._seen_ids: Set[str] = set()
        self._max_seen_ids: int = 5000  # Memory limit for seen IDs
        self._cumulative_pnl: float = 0.0  # Track cumulative P&L for state persistence
        self._max_trades: int = 1000  # Memory limit for trades list

        # live adapter (ensure initialized; _initialize also handles this during super())
        if not hasattr(self, "adapter"):
            self.adapter: Optional[BaseLiveAdapter] = None
        self._ensure_adapter()
        self._publish_adapter_status()

        # debugger
        dbg_cfg: Dict[str, Any] = {**(self.cfg.debug_config or {}), "enabled": bool(self.cfg.debug_enabled)}
        self.debugger = ExecutorDebugManager(self.bus, config=dbg_cfg)

        # unified logger
        self.unified_logger = UnifiedExecutorLogger(self.logger)

        # Smart Position Manager for intelligent live trading
        # Config is loaded from config/risk_policy.yaml -> smart_position section
        self.smart_position_manager = SmartPositionManager()
        
        # UNIFIED LOT CALCULATOR - Single source of truth for lot sizing
        self.lot_calculator = UnifiedLotCalculator.get_instance()
        self.lot_calculator.publish_lot_config_to_bus()

        # seed bus with empty snapshots
        self._publish_all(exec_fills=[], accepted=[], rejected=[], step_pnl=0.0, realized_step=0.0, unrealized=0.0, reason="startup")

    # ─────────────────────────────────────────────────────────
    # initialization / config update
    # ─────────────────────────────────────────────────────────
    def _initialize(self, **kwargs: Any) -> None:
        # Called by BaseModule.__init__ before this __init__ completes.
        if kwargs.get("config") and not isinstance(self.cfg, ExecutorConfig):
            self.cfg = ExecutorConfig(**kwargs["config"])
        if not hasattr(self, "adapter"):
            self.adapter: Optional[BaseLiveAdapter] = None
        if not hasattr(self, "bus"):
            self.bus = SafeBus(InfoBusManager.get_instance())
        self._ensure_adapter()
        try:
            self._publish_adapter_status()
        except Exception:
            # tolerate early publish issues during bootstrap
            pass
        # update debugger config live
        if hasattr(self, "debugger") and self.debugger:
            if self.cfg.debug_enabled:
                self.debugger.enable()
            else:
                self.debugger.disable()

    # adapter bring-up
    def _ensure_adapter(self, mode: Optional[str] = None) -> None:
        """
        Ensure live adapter is created and connected if needed.

        Args:
            mode: Target execution mode ('live' or 'sim'). If None, uses self.cfg.execution_mode.
                  When called from _resolve_mode(), pass the resolved mode to avoid circular calls.
        """
        target_mode = mode if mode is not None else self.cfg.execution_mode
        if target_mode == "live":
            if (self.adapter is None) or (not self.adapter.is_connected()):
                lac = LiveAdapterConfig(
                    broker=self.cfg.live_broker,
                    account_currency="EUR",
                    symbol_overrides=self.cfg.symbol_overrides,
                    lot_step=self.cfg.lot_step,
                    min_lot=self.cfg.min_lot,
                    contract_size=self.cfg.contract_size,
                    price_decimals=self.cfg.price_decimals,
                )
                self.adapter = MT5Adapter(lac) if self.cfg.live_broker.lower() == "mt5" else None
                if self.adapter and not self.adapter.is_connected():
                    ok = self.adapter.connect()
                    self.logger.info(f"[Executor] Live adapter connect -> {ok}")

    def _build_live_adapter_status(self) -> Dict[str, Any]:
        st = {
            "provider": (self.adapter.cfg.broker if self.adapter else self.cfg.live_broker),
            "connected": bool(self.adapter.is_connected()) if self.adapter else False,
        }
        if self.adapter and self.adapter.is_connected():
            try:
                ai = self.adapter.get_account_info() or {}
                st.update({
                    "balance": float(ai.get("balance", 0.0) or 0.0),
                    "equity": float(ai.get("equity", 0.0) or 0.0),
                })
            except Exception:
                pass
        return st

    def _publish_adapter_status(self) -> None:
        """Expose adapter state for UI/debug and reward-level probes."""
        st = self._build_live_adapter_status()
        self.bus.set("live_adapter_status", st, thesis="executor live adapter status")

    def _resolve_mode(self) -> str:
        """Determine execution mode with strong preference for environment_config=live.

        Precedence:
        1) If environment_config.mode == 'live' -> force 'live' (primary signal when live trading is started)
        2) Else, use bus 'execution_mode' if set to a valid value
        3) Else, fall back to static config default
        Always ensure live adapter is connected before returning 'live'; otherwise fall back to 'sim'.
        """
        if not self.cfg.allow_runtime_switch:
            return self.cfg.execution_mode

        # Read environment config first (authoritative when live trading loop starts)
        envc = self.bus.get("environment_config", "Executor", default={}) or {}
        env_mode = str(envc.get("mode", "")).lower()

        # Then check explicit bus override
        em = self.bus.get("execution_mode", "Executor", default=None)
        em_mode = str(em).lower() if isinstance(em, str) else None

        if env_mode == "live":
            target = "live"
        elif em_mode in ("sim", "live"):
            target = em_mode
        else:
            target = self.cfg.execution_mode

        # Strong override: if the live adapter is already connected, prefer live
        # regardless of a stray execution_mode value elsewhere on the bus.
        if self.adapter and self.adapter.is_connected():
            target = "live"

        if target == "live":
            if not self.adapter or not self.adapter.is_connected():
                # Pass target mode to avoid checking static cfg.execution_mode
                self._ensure_adapter(mode=target)
                if not (self.adapter and self.adapter.is_connected()):
                    return "sim"
        return target

    # ─────────────────────────────────────────────────────────
    # main
    # ─────────────────────────────────────────────────────────
    def _account_snapshot(self, pos_snap: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Canonical account snapshot used by downstream modules.
        Includes *raw* balance/equity so readers can subscribe to simple keys.
        """
        if pos_snap is None:
            mode = self._resolve_mode()
            if mode == "live" and self.adapter:
                pos_snap = self.adapter.sync_positions()
            else:
                # Pass current price to as_bus() for proper display
                # Also update peak_unrealized for trailing profit tracking
                pos_snap = {}
                for k, v in self.positions.items():
                    current_price = self._sim_price(k, v.side)
                    # Calculate current unrealized PnL
                    if current_price is not None and current_price > 0 and v.entry_price > 0 and v.units > 0:
                        current_pnl = (current_price - v.entry_price) * v.side * v.units
                        # Update peak if current PnL is higher
                        if current_pnl > v.peak_unrealized:
                            v.peak_unrealized = current_pnl
                    pos_snap[k] = v.as_bus(last_price=current_price)

        return {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "positions": pos_snap,
            "positions_count": int(len(pos_snap or {})),
            "step": int(self.step_idx),
            "ts": time.time(),
        }

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        t0 = time.time()
        try:
            # use bus step if available to align with env; otherwise monotonic
            val = self.bus.get("step_idx", "Executor", default=None)
            if val is None:
                val = self.bus.get("step_idx", "PositionManager", default=None)
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                self.step_idx = int(val)
            else:
                self.step_idx += 1

            mode = self._resolve_mode()
            self._publish_adapter_status()

            # capture pre PnL snapshot
            balance_before = float(self.balance)
            equity_before = float(self.equity)

            # collect intents
            self.debugger.begin("collect_intents")
            accepted, rejected, q_count, dec_count = self._collect_intents()
            self.debugger.end("collect_intents")

            # execute
            realized_step = 0.0
            unreal_after = 0.0
            positions_after: Dict[str, Any] = {}
            fills: List[Dict[str, Any]] = []

            if mode == "live":
                self.debugger.begin("execute_live")
                # Use smart execution for intelligent position management
                fills, step_pnl = self._execute_live_smart(accepted)
                self.debugger.end("execute_live")
                positions_after = self.adapter.sync_positions() if self.adapter else {}
                acct = self.adapter.get_account_info() if self.adapter else {}
                self.balance = float(acct.get("balance", self.balance))
                self.equity = float(acct.get("equity", self.equity))
            else:
                self.debugger.begin("execute_sim")
                fills, step_pnl, realized_step, unreal_after = self._execute_sim(accepted, want_breakdown=True)
                self.debugger.end("execute_sim")
                # Pass current price to as_bus() for proper display
                # Also update peak_unrealized for trailing profit tracking
                positions_after = {}
                for k, v in self.positions.items():
                    current_price = self._sim_price(k, v.side)
                    # Calculate current unrealized PnL
                    if current_price is not None and current_price > 0 and v.entry_price > 0 and v.units > 0:
                        current_pnl = (current_price - v.entry_price) * v.side * v.units
                        # Update peak if current PnL is higher
                        if current_pnl > v.peak_unrealized:
                            v.peak_unrealized = current_pnl
                    positions_after[k] = v.as_bus(last_price=current_price)

            # publish to bus
            self.debugger.begin("publish_bus")
            self._publish_all(
                exec_fills=fills,
                accepted=accepted,
                rejected=rejected,
                step_pnl=step_pnl,
                realized_step=realized_step,
                unrealized=unreal_after,
            )
            self.debugger.end("publish_bus")
            # Prune consumed orders to avoid duplicate_id churn
            self._prune_order_queue()
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            try:
                self.debugger.record_error(f"process_exception: {e}")
            except Exception:
                pass
            self.logger.error(
                format_operator_message(
                    "[CRASH]", "EXECUTOR PROCESS FAILED", details=str(e), context="executor_process"
                )
            )
            try:
                self.logger.error(f"Traceback: {tb}")
            except Exception:
                pass
            raise

        # Debug: log fills (throttled preview)
        try:
            if getattr(self.cfg, "debug_enabled", False) and getattr(self.cfg, "trace_logging", False):
                limit = max(0, int(getattr(self.cfg, "log_fill_preview", 3)))
                for f in fills[:limit]:
                    self.logger.debug(
                        format_operator_message(
                            icon="[EXEC]",
                            message="Fill",
                            instrument=f.get("instrument"),
                            action=f.get("action"),
                            side=f.get("side"),
                            units=f.get("units"),
                            price=f.get("price"),
                            notional=f.get("notional_eur"),
                            origin=f.get("origin_id"),
                        )
                    )
        except Exception:
            pass

        # debugger report
        try:
            self.debugger.publish(
                step=self.step_idx,
                mode=mode,
                queue_count=q_count,
                decisions_count=dec_count,
                accepted=accepted,
                rejected=rejected,
                fills=fills,
                positions_after=positions_after,
                balance_before=balance_before,
                equity_before=equity_before,
                balance_after=float(self.balance),
                equity_after=float(self.equity),
                realized_step=float(realized_step),
                unreal_after=float(unreal_after),
                step_pnl=float(step_pnl),
                reason="normal",
                extra={"processing_ms": (time.time() - t0) * 1000.0},
            )
        except Exception as e:
            self.debugger.record_error(f"debug_publish_error: {e}")

        # unified logger report
        try:
            if self.cfg.debug_enabled:
                self._log_unified_cycle(
                    mode=mode,
                    q_count=q_count,
                    dec_count=dec_count,
                    accepted=accepted,
                    rejected=rejected,
                    fills=fills,
                    positions_after=positions_after,
                    balance_before=balance_before,
                    equity_before=equity_before,
                    realized_step=realized_step,
                    unreal_after=unreal_after,
                    step_pnl=step_pnl,
                    processing_ms=(time.time() - t0) * 1000.0,
                )
        except Exception as e:
            self.logger.warning(f"Unified logger failed: {e}")

        # REQUIRED outputs for orchestrator contract (return payload)
        recent = self.trades[-50:] if self.trades else []
        order_data = {"accepted": accepted, "rejected": rejected, "step": int(self.step_idx)}
        execution_data = {"fills": fills, "step": int(self.step_idx)}
        market_state = {"balance": float(self.balance), "equity": float(self.equity), "step": int(self.step_idx)}
        portfolio_metrics = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),
            "realized_pnl_step": float(realized_step),
            "unrealized_pnl": float(unreal_after),
            "step": int(self.step_idx),
        }
        trade_data = {
            "trades": list(self.trades),
            "recent_trades": recent,
            "fills": list(fills),
            "step": int(self.step_idx),
        }

        # helpful aliases expected by some readers
        current_positions = dict(positions_after)
        pnl_data = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),
            "realized_pnl_step": float(realized_step),
            "unrealized_pnl": float(unreal_after),
            "step": int(self.step_idx),
        }
        live_adapter_status = self.bus.get(
            "live_adapter_status", "Executor",
            default={"provider": self.cfg.live_broker, "connected": False}
        )

        # Build normalized position_data list (legacy/risk-friendly schema)
        pos_list: List[Dict[str, Any]] = []
        try:
            for inst, p in (positions_after or {}).items():
                notional = float(p.get("notional_eur", 0.0) or 0.0)
                units = float(p.get("units", 0.0) or 0.0)
                entry_price = float(p.get("entry_price", 0.0) or 0.0)
                size = abs(notional) if abs(notional) > 0 else (abs(units * entry_price) if (units and entry_price) else abs(units))
                entry: Dict[str, Any] = {"instrument": inst, "size": float(size)}
                for k, v in p.items():
                    if k != "instrument":
                        entry[k] = v
                pos_list.append(entry)
        except Exception:
            try:
                pos_list = [{"instrument": inst, **(p or {})} for inst, p in (positions_after or {}).items()]
            except Exception:
                pos_list = []

        return {
            "positions": positions_after,
            "trades": self.trades[-200:],
            "recent_trades": recent,
            "order_data": order_data,
            "execution_data": execution_data,
            "execution_reports": fills,
            "portfolio_metrics": portfolio_metrics,
            "trading_result": {"pnl": float(step_pnl)},
            "trade_data": trade_data,
            "market_state": market_state,
            "position_data": {"positions": pos_list, "count": len(pos_list)},
            "current_positions": current_positions,
            "pnl_data": pnl_data,
            "live_adapter_status": live_adapter_status,
            "closed_positions": list(self.closed_positions),  # For win rate tracking (sim)
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),
            "pending_orders": order_data.get("accepted", []),
            "account_state": {
                "balance": float(self.balance),
                "equity": float(self.equity),
                "initial_balance": float(self.initial_balance),
                "step": int(self.step_idx)
            },
            "order_queue": [],
            "processing_time_ms": (time.time() - t0) * 1000.0,
        }

    # ─────────────────────────────────────────────────────────
    # intents
    # ─────────────────────────────────────────────────────────
    def _collect_intents(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        q_count = 0
        dec_count = 0

        # ==========================================================
        # MEMORY INTEGRATION: Check for memory veto before processing
        # ==========================================================
        memory_veto = False
        memory_veto_reasons: List[str] = []
        vetoed_instruments: List[str] = []  # Per-instrument vetoes
        try:
            memory_gate = self.bus.get("memory_gate", "Executor", default=None)
            if isinstance(memory_gate, dict):
                # Global veto
                if memory_gate.get("veto", False):
                    memory_veto = True
                    memory_veto_reasons = memory_gate.get("reasons", ["Memory system vetoed"])
                    self.logger.warning(format_operator_message(
                        icon="🧠",
                        message="MEMORY_VETO_ACTIVE",
                        reasons=memory_veto_reasons[:3],
                    ))
                
                # Per-instrument vetoes (from memory's loss streak tracking)
                vetoed_inst_raw = memory_gate.get("vetoed_instruments", [])
                if isinstance(vetoed_inst_raw, list):
                    vetoed_instruments = [str(inst).upper().replace("/", "_") for inst in vetoed_inst_raw]
                    if vetoed_instruments:
                        self.logger.warning(format_operator_message(
                            icon="🧠",
                            message="MEMORY_INSTRUMENT_VETO",
                            instruments=vetoed_instruments,
                        ))
        except Exception:
            pass

        # ==========================================================
        # v4.2.0: RISK CONTROLLER INTEGRATION
        # Check DynamicRiskController for emergency mode or critical risk
        # NOTE: risk_scale sizing is handled by UnifiedLotCalculator (single source of truth)
        # Here we only check for hard VETO conditions (emergency, critical)
        # ==========================================================
        risk_veto = False
        risk_veto_reason = ""
        try:
            risk_assessment = self.bus.get("risk_assessment", "Executor", default=None)
            risk_level = self.bus.get("risk_level", "Executor", default=None)
            
            # Check for emergency mode - HARD VETO
            if isinstance(risk_assessment, dict):
                if risk_assessment.get("emergency_active", False):
                    risk_veto = True
                    risk_veto_reason = "EMERGENCY_MODE_ACTIVE"
                    self.logger.warning(format_operator_message(
                        icon="🚨",
                        message="RISK_EMERGENCY_MODE",
                        reason="DynamicRiskController in emergency mode - blocking new positions",
                    ))
            
            # Check for critical risk level - HARD VETO
            if isinstance(risk_level, str) and risk_level.upper() == "CRITICAL":
                risk_veto = True
                risk_veto_reason = "RISK_LEVEL_CRITICAL"
                self.logger.warning(format_operator_message(
                    icon="⛔",
                    message="RISK_LEVEL_CRITICAL",
                    reason="DynamicRiskController reports critical risk level",
                ))
            
            # NOTE: risk_scale_multiplier REMOVED - UnifiedLotCalculator handles position sizing
            # Executor only does VETO checks, not sizing adjustments
        except Exception:
            pass

        # ==========================================================
        # STRATEGY MODULE INTEGRATION (v3.0)
        # Apply CurriculumPlannerPlus constraints and BiasAuditor adjustments
        # ==========================================================
        strategy_position_multiplier: float = 1.0
        strategy_max_trades_per_day: int = 50  # Default high limit
        curriculum_stage: str = "Expert"       # Default to no restrictions
        bias_active: List[str] = []
        
        try:
            # BiasAuditor: position size multiplier
            bias_adjustments = self.bus.get("bias_adjustments", "Executor", default=None)
            if isinstance(bias_adjustments, dict):
                strategy_position_multiplier = float(bias_adjustments.get("position_size_multiplier", 1.0) or 1.0)
                strategy_position_multiplier = max(0.1, min(1.0, strategy_position_multiplier))
                
                # Log if bias is reducing position size
                if strategy_position_multiplier < 0.95:
                    bias_analysis = self.bus.get("bias_analysis", "Executor", default={}) or {}
                    ind_biases = bias_analysis.get("individual_biases", {})
                    if isinstance(ind_biases, dict):
                        bias_active = [k for k, v in ind_biases.items() if isinstance(v, dict) and v.get("detected", False)]
                    self.logger.info(format_operator_message(
                        icon="🧠",
                        message="BIAS_POSITION_ADJUSTMENT",
                        multiplier=f"{strategy_position_multiplier:.2f}",
                        active_biases=bias_active[:3] if bias_active else ["psychological"],
                    ))
            
            # CurriculumPlannerPlus: learning constraints
            # NOTE: Only apply curriculum constraints in LIVE mode, not during training (sim)
            # Training needs freedom to explore - curriculum restrictions should only apply to live trading
            learning_constraints = self.bus.get("learning_constraints", "Executor", default=None)
            curriculum_stage_data = self.bus.get("curriculum_stage", "Executor", default=None)
            
            # Determine if we're in live mode (adapter connected)
            is_live_mode = self.adapter and self.adapter.is_connected()
            
            if isinstance(learning_constraints, dict) and is_live_mode:
                # Max position size constraint (applied to strategy_position_multiplier)
                # ONLY in live mode - training should explore full position sizes
                max_pos = float(learning_constraints.get("max_position_size", 1.0) or 1.0)
                if max_pos < 1.0:
                    strategy_position_multiplier = min(strategy_position_multiplier, max_pos)
                
                # Max trades per day constraint (ONLY in live mode)
                strategy_max_trades_per_day = int(learning_constraints.get("max_trades_per_day", 50) or 50)
            elif not is_live_mode:
                # Training mode: allow unlimited trades and full position sizes for exploration
                strategy_max_trades_per_day = 9999  # Effectively unlimited
                # Log only once per session to avoid spam
                pass
            
            if isinstance(curriculum_stage_data, dict):
                curriculum_stage = str(curriculum_stage_data.get("name", "Expert") or "Expert")
                
                # Log curriculum-based constraints (only in live mode)
                if curriculum_stage in ("Foundation", "Basic") and is_live_mode:
                    self.logger.debug(format_operator_message(
                        icon="📚",
                        message="CURRICULUM_CONSTRAINTS_ACTIVE",
                        stage=curriculum_stage,
                        max_position_multiplier=f"{strategy_position_multiplier:.2f}",
                        max_trades=strategy_max_trades_per_day,
                    ))
        except Exception:
            pass
        
        # Track trades today for curriculum enforcement
        trades_today = 0
        try:
            if self.trades:
                today_start = dt.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                trades_today = sum(1 for t in self.trades[-200:] 
                                   if isinstance(t, dict) and t.get("ts", 0) >= today_start.timestamp())
        except Exception:
            pass

        # ==========================================================
        # TRADING MODE INTEGRATION (v4.0)
        # Apply TradingModeManager constraints for position sizing and risk
        # ==========================================================
        trading_mode_name: str = "normal"
        trading_mode_position_scale: float = 1.0
        trading_mode_max_exposure: float = 0.5
        trading_mode_risk_multiplier: float = 1.0
        trading_mode_stop_loss_multiplier: float = 1.0
        
        try:
            trading_mode = self.bus.get("trading_mode", "Executor", default=None)
            mode_config = self.bus.get("mode_config", "Executor", default=None)
            mode_effectiveness = self.bus.get("mode_effectiveness", "Executor", default=None)
            
            if isinstance(trading_mode, str):
                trading_mode_name = trading_mode.lower()
            
            if isinstance(mode_config, dict):
                trading_mode_position_scale = float(mode_config.get("position_scale", 1.0) or 1.0)
                trading_mode_position_scale = max(0.25, min(2.0, trading_mode_position_scale))
                
                trading_mode_max_exposure = float(mode_config.get("max_exposure", 0.5) or 0.5)
                trading_mode_max_exposure = max(0.1, min(1.0, trading_mode_max_exposure))
                
                trading_mode_risk_multiplier = float(mode_config.get("risk_multiplier", 1.0) or 1.0)
                trading_mode_risk_multiplier = max(0.25, min(4.0, trading_mode_risk_multiplier))
                
                trading_mode_stop_loss_multiplier = float(mode_config.get("stop_loss_multiplier", 1.0) or 1.0)
                trading_mode_stop_loss_multiplier = max(0.5, min(2.0, trading_mode_stop_loss_multiplier))
            
            # Log trading mode constraints if not normal
            if trading_mode_name != "normal" or trading_mode_position_scale != 1.0:
                eff_value = None
                if mode_effectiveness is not None:
                    try:
                        eff_value = float(mode_effectiveness) if not isinstance(mode_effectiveness, dict) else float(mode_effectiveness.get("value", mode_effectiveness.get("effectiveness", 0.5)))
                    except (TypeError, ValueError):
                        eff_value = None
                
                self.logger.info(format_operator_message(
                    icon="⚙️",
                    message="TRADING_MODE_CONSTRAINTS",
                    mode=trading_mode_name.upper(),
                    position_scale=f"{trading_mode_position_scale:.2f}",
                    max_exposure=f"{trading_mode_max_exposure:.2f}",
                    risk_multiplier=f"{trading_mode_risk_multiplier:.2f}",
                    effectiveness=f"{eff_value:.2f}" if eff_value else "N/A",
                ))
        except Exception:
            pass
        
        # Helper to apply trading mode position sizing
        def _apply_trading_mode_sizing(intent: Dict[str, Any]) -> Dict[str, Any]:
            """Apply TradingModeManager position scale and max exposure."""
            if trading_mode_position_scale >= 0.99 and trading_mode_position_scale <= 1.01:
                return intent  # No adjustment needed
            
            # Apply position scale to size_eur if present
            if "size_eur" in intent and intent["size_eur"]:
                original_size = float(intent["size_eur"])
                intent["size_eur"] = original_size * trading_mode_position_scale
                intent["_trading_mode_sizing"] = {
                    "original_size": original_size,
                    "position_scale": trading_mode_position_scale,
                    "mode": trading_mode_name,
                    "max_exposure": trading_mode_max_exposure,
                    "risk_multiplier": trading_mode_risk_multiplier,
                }
            
            # Apply scale to units if present
            if "units" in intent and intent.get("units"):
                original_units = float(intent["units"])
                intent["units"] = original_units * trading_mode_position_scale
            
            return intent

        # ==========================================================
        # PORTFOLIO RISK ENFORCEMENT: DISABLED FOR PROP FIRMS
        # Prop firms only care about P&L drawdown, NOT notional exposure.
        # This check was blocking positions based on exposure % which is
        # meaningless - a 0.08 lot XAUUSD position has 33% notional exposure
        # but only 0.08% actual risk with proper stop loss.
        # ==========================================================
        risk_blocked_instruments: Set[str] = set()
        # NOTE: Exposure-based blocking disabled. Prop firms don't monitor this.
        # If you need exposure limits (non-prop firm), set enable_exposure_blocking: true in config

        # Helper to check if instrument is vetoed
        def _is_instrument_vetoed(inst: str) -> bool:
            if memory_veto:
                return True  # Global veto blocks all
            if risk_veto:
                return True  # Risk veto blocks all new positions
            normalized = str(inst).upper().replace("/", "_")
            return normalized in vetoed_instruments
        
        # Helper to get veto reason
        def _get_veto_reason() -> Tuple[str, List[str]]:
            if memory_veto:
                return "memory_veto", memory_veto_reasons
            if risk_veto:
                return "risk_veto", [risk_veto_reason]
            return "instrument_veto", ["Instrument on veto list"]
        
        # Helper to check if instrument is blocked by risk limits
        def _is_risk_blocked(inst: str, action: str) -> bool:
            """DISABLED: Prop firms don't care about notional exposure"""
            return False  # Never block based on exposure for prop firms
        
        # Helper to check curriculum trade limit
        def _exceeds_curriculum_trade_limit() -> bool:
            """Check if today's trades exceed curriculum limit."""
            return trades_today >= strategy_max_trades_per_day
        
        # Helper to apply strategy position size multiplier
        def _apply_strategy_sizing(intent: Dict[str, Any]) -> Dict[str, Any]:
            """Apply BiasAuditor and CurriculumPlanner position size adjustments."""
            if strategy_position_multiplier >= 0.99:
                return intent  # No adjustment needed
            
            # Apply multiplier to size_eur if present
            if "size_eur" in intent and intent["size_eur"]:
                original_size = float(intent["size_eur"])
                intent["size_eur"] = original_size * strategy_position_multiplier
                intent["_strategy_sizing"] = {
                    "original_size": original_size,
                    "multiplier": strategy_position_multiplier,
                    "curriculum_stage": curriculum_stage,
                    "active_biases": bias_active,
                }
            
            # Apply multiplier to units if present
            if "units" in intent and intent.get("units"):
                original_units = float(intent["units"])
                intent["units"] = original_units * strategy_position_multiplier
            
            return intent

        # explicit order_queue
        oq = self.bus.get("order_queue", "Executor", default=[])
        if isinstance(oq, list):
            q_count = len(oq)
            for item in oq[: self.cfg.max_orders_per_step]:
                intent = self._normalize_order_item(item)
                if not intent:
                    rejected.append({"reason": "bad_order_queue_item", "raw": item})
                # Memory/Risk veto check - reject new opening orders (global OR per-instrument)
                elif intent.get("action", "").lower() in ("open_long", "open_short", "buy", "sell", "scale_up"):
                    inst = intent.get("instrument", "")
                    action = intent.get("action", "")
                    # First check memory or risk veto
                    if _is_instrument_vetoed(inst):
                        veto_reason, veto_reasons = _get_veto_reason()
                        rejected.append({
                            "reason": veto_reason,
                            "intent": intent,
                            "veto_reasons": veto_reasons,
                            "vetoed_instrument": inst,
                        })
                        continue
                    # Check curriculum trade limit
                    if _exceeds_curriculum_trade_limit():
                        rejected.append({
                            "reason": "curriculum_trade_limit",
                            "intent": intent,
                            "message": f"Daily trade limit ({strategy_max_trades_per_day}) reached for curriculum stage '{curriculum_stage}'",
                            "trades_today": trades_today,
                            "curriculum_stage": curriculum_stage,
                        })
                        self.logger.info(format_operator_message(
                            icon="📚",
                            message="ORDER_BLOCKED_CURRICULUM_LIMIT",
                            trades_today=trades_today,
                            max_trades=strategy_max_trades_per_day,
                            stage=curriculum_stage,
                        ))
                        continue
                    # Then check portfolio risk limits
                    if _is_risk_blocked(inst, action):
                        rejected.append({
                            "reason": "risk_limit_exceeded",
                            "intent": intent,
                            "blocked_instrument": inst,
                            "message": f"{inst} position exceeds portfolio risk limit - cannot increase exposure",
                        })
                        self.logger.warning(format_operator_message(
                            icon="🛑",
                            message="ORDER_BLOCKED_RISK_LIMIT",
                            instrument=inst,
                            action=action,
                            reason="Position size exceeds limit",
                        ))
                        continue
                    # Apply strategy position sizing (BiasAuditor + CurriculumPlanner)
                    intent = _apply_strategy_sizing(intent)
                    # Apply trading mode position sizing
                    intent = _apply_trading_mode_sizing(intent)
                    if self._passes_filters(intent):
                        if intent["id"] not in self._seen_ids:
                            accepted.append(intent)
                            self._seen_ids.add(intent["id"])
                        else:
                            rejected.append({"reason": "duplicate_id", "intent": intent})
                    else:
                        reason = self._filter_reason(intent)
                        rejected.append({"reason": reason, "intent": intent})
                elif self._passes_filters(intent):
                    if intent["id"] not in self._seen_ids:
                        accepted.append(intent)
                        self._seen_ids.add(intent["id"])
                    else:
                        rejected.append({"reason": "duplicate_id", "intent": intent})
                else:
                    reason = self._filter_reason(intent)
                    rejected.append({"reason": reason, "intent": intent})

        # fallback: position_decision_* (look under PositionManager as well)
        if self.cfg.read_position_decisions:
            env_cfg = (self.bus.get("environment_config", "Executor", default=None)
                    or self.bus.get("environment_config", "PositionManager", default={})
                    or {})
            instruments = env_cfg.get("instruments") or []
            for inst in instruments:
                core = inst.replace("/", "").replace("_", "")

                # try both module spaces and canonical variants
                node = self.bus.get(f"position_decision_{inst}", "Executor", default=None)
                if not (isinstance(node, dict) and node.get("decision")):
                    node = self.bus.get(f"position_decision_{inst}", "PositionManager", default=None)
                if not (isinstance(node, dict) and node.get("decision")):
                    node = self.bus.get(f"position_decision_{core}", "PositionManager", default=None)
                if not (isinstance(node, dict) and node.get("decision")):
                    node = self.bus.get(f"position_decision_{core}", "Executor", default=None)

                if isinstance(node, dict) and node.get("decision"):
                    dec_count += 1
                    dec = str(node["decision"]).lower()
                    if self.cfg.ignore_hold and dec == "hold":
                        continue
                    inten = float(node.get("intensity", 0.0) or 0.0)
                    conf = float(node.get("confidence", 0.0) or 0.0)
                    size_eur = float(node.get("size", 0.0) or 0.0)
                    intent = {
                        "id": f"pm-{inst}-{self.step_idx}-{dec}",
                        "instrument": inst,
                        "action": dec,
                        "intensity": inten,
                        "confidence": conf,
                        "size_eur": size_eur,
                    }
                    if self._passes_filters(intent) and intent["id"] not in self._seen_ids:
                        accepted.append(intent)
                        self._seen_ids.add(intent["id"])
                    else:
                        rejected.append({"reason": self._filter_reason(intent), "intent": intent})

        # Debug summary of intents (compact version - full details in unified logger)
        try:
            self.logger.debug(
                f"[EXEC] Collected: queue={q_count} decisions={dec_count} accepted={len(accepted)} rejected={len(rejected)}"
            )
        except Exception:
            pass

        # Trim _seen_ids to prevent memory leak
        if len(self._seen_ids) > self._max_seen_ids:
            ids_list = list(self._seen_ids)
            self._seen_ids = set(ids_list[-(self._max_seen_ids // 2):])

        return accepted, rejected, q_count, dec_count

    def _prune_order_queue(self) -> None:
        """Remove already-seen order IDs from the shared order_queue to prevent duplicate churn."""
        try:
            q = self.bus.get("order_queue", "Executor", default=[]) or []
            if not isinstance(q, list) or not q:
                return
            filtered: List[Dict[str, Any]] = []
            for item in q:
                oid = None
                try:
                    if isinstance(item, dict):
                        oid = item.get("id")
                except Exception:
                    oid = None
                if oid and oid in self._seen_ids:
                    continue
                filtered.append(item)
            if len(filtered) != len(q):
                try:
                    self.bus.set("order_queue", filtered, thesis="Executor pruned consumed orders")
                except Exception:
                    pass
        except Exception:
            pass

    def _normalize_order_item(self, item: Any) -> Optional[Dict[str, Any]]:
        try:
            if not isinstance(item, dict):
                return None
            oid = str(item.get("id") or f"oq-{uuid.uuid4().hex[:10]}")
            inst = str(item["instrument"])
            raw_action = str(item.get("intent", item.get("action", ""))).lower()
            side_i = int(item.get("side", 0) or 0)
            if raw_action == "open":
                action = "open_long" if side_i >= 0 else "open_short"
            elif raw_action == "scale":
                action = "scale_up" if side_i >= 0 else "scale_down"
            else:
                action = raw_action
            conf = float(item.get("confidence", 0.0) or 0.0)
            inten = float(item.get("intensity", 0.0) or 0.0)
            out = {"id": oid, "instrument": inst, "action": action, "confidence": conf, "intensity": inten}
            if "size_eur" in item and item["size_eur"] is not None:
                out["size_eur"] = float(item["size_eur"])
            elif "units" in item and item["units"] is not None:
                out["units"] = float(item["units"])
            else:
                out["size_eur"] = 0.0
            return out
        except Exception:
            return None

    def _passes_filters(self, intent: Dict[str, Any]) -> bool:
        act = str(intent.get("action", "")).lower()
        if act == "hold" and self.cfg.ignore_hold:
            return False
        # Always allow risk-reduction/exit actions regardless of thresholds
        if act in ("close", "emergency_close", "scale_down"):
            return True
        if float(intent.get("confidence", 0.0)) < self.cfg.min_confidence:
            return False
        if abs(float(intent.get("intensity", 0.0))) < self.cfg.min_intensity and act not in ("close", "emergency_close"):
            return False
        return True

    def _filter_reason(self, intent: Dict[str, Any]) -> str:
        act = str(intent.get("action", "")).lower()
        if act == "hold" and self.cfg.ignore_hold:
            return "ignore_hold"
        if float(intent.get("confidence", 0.0)) < self.cfg.min_confidence:
            return f"low_confidence<{self.cfg.min_confidence}>"
        if abs(float(intent.get("intensity", 0.0))) < self.cfg.min_intensity and act not in ("close", "emergency_close"):
            return f"low_intensity<{self.cfg.min_intensity}>"
        if intent.get("id") in self._seen_ids:
            return "duplicate_id"
        return "filtered"

    # ─────────────────────────────────────────────────────────
    # Symbol-specific contract sizing
    # ─────────────────────────────────────────────────────────
    def _get_contract_size(self, symbol: str) -> float:
        """Get contract size for a symbol (units per 1.0 lot)."""
        sym_upper = (symbol or "").upper().replace("_", "").replace("/", "")

        # Gold/Silver/Crypto have different contract sizes
        if "XAU" in sym_upper or "GOLD" in sym_upper:
            return 100.0  # Gold: 100 oz per lot
        if "XAG" in sym_upper or "SILVER" in sym_upper:
            return 5000.0  # Silver: 5000 oz per lot
        if "BTC" in sym_upper:
            return 1.0  # Bitcoin: 1 BTC per lot
        if "ETH" in sym_upper:
            return 1.0  # Ethereum: 1 ETH per lot

        # Default: Forex 100,000 units per lot
        return float(self.cfg.contract_size)

    def _get_current_volatility(self, symbol: str) -> Optional[float]:
        """
        Get current volatility for a symbol from InfoBus.
        
        Tries multiple sources: volatility_by_instrument, market_conditions, feature data.
        Returns None if unavailable (lot calculator will use defaults).
        """
        try:
            norm_symbol = self._normalize_symbol(symbol)
            
            # Try volatility_by_instrument
            vol_map = self.bus.get("volatility_by_instrument", "Executor", default=None)
            if isinstance(vol_map, dict):
                for key, val in vol_map.items():
                    if self._normalize_symbol(key) == norm_symbol:
                        if isinstance(val, (int, float)):
                            return float(val)
            
            # Try market_conditions
            mc = self.bus.get("market_conditions", "Executor", default=None)
            if isinstance(mc, dict):
                vol = mc.get("volatility")
                if isinstance(vol, (int, float)):
                    return float(vol)
            
            # Try price_data for ATR
            pd = self.bus.get("price_data", "Executor", default=None)
            if isinstance(pd, dict):
                inst_data = pd.get(symbol) or pd.get(norm_symbol)
                if isinstance(inst_data, dict):
                    atr = inst_data.get("atr") or inst_data.get("volatility")
                    if isinstance(atr, (int, float)):
                        return float(atr)
            
        except Exception:
            pass
        
        return None

    # ─────────────────────────────────────────────────────────
    # SIM execution
    # ─────────────────────────────────────────────────────────
    def _sim_price(self, inst: str, side: int) -> Optional[float]:
        sp = (self.bus.get("prices", "Executor", default=None)
              or self.bus.get("prices", "PositionManager", default={})
              or {})
        pd = (self.bus.get("price_data", "Executor", default=None)
              or self.bus.get("price_data", "PositionManager", default={})
              or {})
        px = None
        try:
            node = None
            if isinstance(pd, dict):
                node = pd.get(inst) or pd.get(inst.replace("/", "").replace("_", ""))
            if isinstance(node, dict):
                last = node.get("last", node.get("close"))
                if isinstance(last, (int, float)):
                    px = float(last)
            if px is None and isinstance(sp, dict):
                v = sp.get(inst) or sp.get(inst.replace("/", "").replace("_", ""))
                if isinstance(v, (int, float)):
                    px = float(v)
                elif isinstance(v, dict):
                    for k in ("last", "close", "price", "bid", "ask"):
                        val = v.get(k)
                        if isinstance(val, (int, float)):
                            px = float(val)
                            break
        except Exception:
            px = None
        if px is None:
            return None
        # mid → side-aware exec price
        if self.cfg.default_spread:
            px += (self.cfg.default_spread / 2.0) * (+1 if side > 0 else -1)
        if self.cfg.slippage_pts:
            px += self.cfg.slippage_pts * (+1 if side > 0 else -1)
        return px

    def _units_from(self, size_eur: float, units: float, price: float) -> float:
        if units and units > 0:
            return float(units)
        if size_eur and size_eur > 0 and price > 0:
            return float(size_eur / price)
        return 0.0

    def _track_closed_position(self, position: "PositionSnap", close_price: float, realized_pnl: float, close_reason: str) -> None:
        """Track a fully closed position for win rate and analytics (sim only)."""
        closed_record = {
            "instrument": position.instrument,
            "side": position.side,
            "units": position.units,
            "entry_price": position.entry_price,
            "close_price": close_price,
            "pnl": realized_pnl,
            "profit": realized_pnl,  # Alias for compatibility
            "close_reason": close_reason,
            "close_step": self.step_idx,
            "entry_step": getattr(position, "entry_step", 0),
            "close_time": time.time(),
            "open_time": getattr(position, "open_time", 0.0),
        }
        self.closed_positions.append(closed_record)
        # Keep only last 500 closed positions to avoid memory bloat
        if len(self.closed_positions) > 500:
            self.closed_positions = self.closed_positions[-500:]

    def _execute_sim(self, intents: List[Dict[str, Any]], *, want_breakdown: bool = False) -> Tuple[List[Dict[str, Any]], float, float, float]:
        fills: List[Dict[str, Any]] = []
        realized_step = 0.0

        for intent in intents:
            inst = intent["instrument"]
            action = str(intent["action"]).lower()
            side_from_action = {"open_long": +1, "scale_up": +1, "open_short": -1, "scale_down": -1}.get(action, 0)

            price = self._sim_price(inst, side_from_action)
            if price is None:
                self.debugger.record_error(f"sim_price_unavailable:{inst}")
                continue

            size_eur = float(intent.get("size_eur", 0.0) or 0.0)
            units = float(intent.get("units", 0.0) or 0.0)
            add_units = self._units_from(size_eur, units, price)
            origin_id = intent.get("id", "")

            def commission(notional: float) -> float:
                cpm = float(self.cfg.commission_per_million or 0.0)
                return (abs(notional) / 1_000_000.0) * cpm if cpm > 0 else 0.0

            if action in ("open_long", "open_short"):
                if inst in self.positions and self.positions[inst].side != side_from_action:
                    p = self.positions[inst]
                    realized = (price - p.entry_price) * p.side * p.units - commission(p.units * price)
                    realized_step += realized
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
                        action="close:reverse",
                        side=-p.side,
                        units=p.units,
                        price=price,
                        notional_eur=p.units * price,
                        realized_pnl=realized,
                        origin_id=origin_id,
                        comment="reverse",
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
                    self._track_closed_position(p, price, realized, "reverse")
                    del self.positions[inst]

                if add_units > 0:
                    notional = add_units * price
                    self.positions[inst] = PositionSnap(
                        inst, side_from_action, add_units, price,
                        notional_eur=notional,
                        open_time=time.time(),
                        entry_step=self.step_idx,
                    )
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
                        action=action,
                        side=side_from_action,
                        units=add_units,
                        price=price,
                        notional_eur=notional,
                        realized_pnl=0.0,
                        origin_id=origin_id,
                        comment="open",
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)

            elif action == "scale_up":
                if add_units <= 0:
                    continue
                if inst not in self.positions:
                    notional = add_units * price
                    self.positions[inst] = PositionSnap(
                        inst, +1, add_units, price,
                        notional_eur=notional,
                        open_time=time.time(),
                        entry_step=self.step_idx,
                    )
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
                        action="scale_up->open",
                        side=+1,
                        units=add_units,
                        price=price,
                        notional_eur=notional,
                        origin_id=origin_id,
                        comment="scale_up_open",
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
                else:
                    p = self.positions[inst]
                    if p.side < 0:
                        reduce_u = min(add_units, abs(p.units))
                        realized = (price - p.entry_price) * p.side * reduce_u - commission(reduce_u * price)
                        realized_step += realized
                        p.units -= reduce_u
                        p.notional_eur -= reduce_u * p.entry_price
                        if p.units <= 1e-12:
                            self._track_closed_position(p, price, realized, "scale_up_reduce")
                            del self.positions[inst]
                        fill = TradeFill(
                            id=f"fill-{uuid.uuid4().hex[:10]}",
                            ts=time.time(),
                            step=self.step_idx,
                            instrument=inst,
                            action="scale_up_reduce",
                            side=-p.side,
                            units=reduce_u,
                            price=price,
                            notional_eur=reduce_u * price,
                            realized_pnl=realized,
                            origin_id=origin_id,
                            comment="reduce",
                        ).as_bus()
                        self.trades.append(fill)
                        fills.append(fill)
                    else:
                        new_u = p.units + add_units
                        p.entry_price = (p.entry_price * p.units + price * add_units) / new_u
                        p.units = new_u
                        p.notional_eur += add_units * price
                        fill = TradeFill(
                            id=f"fill-{uuid.uuid4().hex[:10]}",
                            ts=time.time(),
                            step=self.step_idx,
                            instrument=inst,
                            action="scale_up",
                            side=+1,
                            units=add_units,
                            price=price,
                            notional_eur=add_units * price,
                            origin_id=origin_id,
                            comment="scale",
                        ).as_bus()
                        self.trades.append(fill)
                        fills.append(fill)

            elif action == "scale_down":
                if inst not in self.positions or add_units <= 0:
                    continue
                p = self.positions[inst]
                # Reduce exposure regardless of side; realize P&L on reduced portion
                reduce_u = min(add_units, p.units)
                if reduce_u <= 0:
                    continue
                realized = (price - p.entry_price) * p.side * reduce_u - commission(reduce_u * price)
                realized_step += realized
                p.units -= reduce_u
                p.notional_eur -= reduce_u * p.entry_price
                if p.units <= 1e-12:
                    self._track_closed_position(p, price, realized, "scale_down")
                    del self.positions[inst]
                trade_side = -1 if p.side > 0 else +1  # sell to reduce long; buy to reduce short
                fill = TradeFill(
                    id=f"fill-{uuid.uuid4().hex[:10]}",
                    ts=time.time(),
                    step=self.step_idx,
                    instrument=inst,
                    action="scale_down_reduce",
                    side=trade_side,
                    units=reduce_u,
                    price=price,
                    notional_eur=reduce_u * price,
                    realized_pnl=realized,
                    origin_id=origin_id,
                    comment="reduce",
                ).as_bus()
                self.trades.append(fill)
                fills.append(fill)

            elif action in ("close", "emergency_close"):
                if inst in self.positions:
                    p = self.positions[inst]
                    realized = (price - p.entry_price) * p.side * p.units - commission(p.units * price)
                    realized_step += realized
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
                        action=f"close:{action}",
                        side=-p.side,
                        units=p.units,
                        price=price,
                        notional_eur=p.units * price,
                        realized_pnl=realized,
                        origin_id=origin_id,
                        comment="close",
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
                    self._track_closed_position(p, price, realized, action)
                    del self.positions[inst]

        # apply realized → balance
        self.balance += realized_step

        # mark-to-market: calculate unrealized P&L
        unreal = 0.0
        for inst, p in self.positions.items():
            px = self._sim_price(inst, p.side)
            if px is None:
                continue
            unreal += (px - p.entry_price) * p.side * p.units

        # Compute equity as realized balance plus unrealized P&L
        equity_now = float(self.balance + unreal)
        step_pnl = float(equity_now - self._last_equity)

        # Keep balance as realized-only; track equity separately
        self.equity = equity_now
        self._last_equity = equity_now

        # Trim trades list to prevent memory leak
        if len(self.trades) > self._max_trades:
            self.trades = self.trades[-self._max_trades:]

        if want_breakdown:
            return fills, step_pnl, float(realized_step), float(unreal)
        return fills, step_pnl, 0.0, 0.0  # not used

    # ─────────────────────────────────────────────────────────
    # LIVE execution (legacy, direct)
    # ─────────────────────────────────────────────────────────
    def _execute_live(self, intents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        fills: List[Dict[str, Any]] = []
        if not self.adapter or not self.adapter.is_connected():
            return fills, 0.0

        # ═══════════════════════════════════════════════════════════════
        # PROP FIRM SAFETY CHECK - Block ALL new trades if limits breached
        # ═══════════════════════════════════════════════════════════════
        prop_limits = self.lot_calculator.check_prop_firm_limits()
        if not prop_limits.get("can_trade", True):
            self.logger.warning(
                f"[LIVE] 🚫 PROP FIRM BLOCK: Trading halted - {prop_limits.get('warnings', [])}"
            )
            # Only allow close actions, block all opens
            intents = [i for i in intents if str(i.get("action", "")).lower() in ("close", "close_all", "scale_down")]
            if not intents:
                return fills, 0.0

        acct_before = self.adapter.get_account_info()
        eq_before = float(acct_before.get("equity", 0.0) or 0.0)
        try:
            self.logger.info(f"[LIVE] Execute intents: n={len(intents)} | equity_before={eq_before:.2f}")
        except Exception:
            pass

        norm_positions = self._normalized_adapter_positions()

        for intent in intents:
            inst_src = intent["instrument"]
            inst = resolve_symbol(inst_src, self.cfg.symbol_overrides, broker=self.cfg.live_broker)
            norm_inst = self._normalize_symbol(inst)
            action = str(intent["action"]).lower()
            side = {"open_long": +1, "scale_up": +1, "open_short": -1, "scale_down": -1}.get(action, 0)

            price_hint = (self.adapter.get_prices(inst) or {}).get("mid", 0.0) or 1.0
            
            # Get signal strength and volatility for unified lot calculation
            signal_strength = float(intent.get("confidence", intent.get("intensity", 0.5)) or 0.5)
            volatility = self._get_current_volatility(inst)
            
            # ═══════════════════════════════════════════════════════════════
            # UNIFIED LOT CALCULATION - Use the central lot calculator
            # ═══════════════════════════════════════════════════════════════
            lots, lot_details = self.lot_calculator.calculate_lots(
                symbol=inst,
                signal_strength=signal_strength,
                volatility=volatility,
            )
            
            # Log the lot calculation details
            try:
                self.logger.info(
                    f"[LIVE] 📊 LOT_CALC: {inst} | signal={signal_strength:.2f} | "
                    f"lots={lots:.2f} | balance=€{lot_details.get('balance', 0):.0f} | "
                    f"risk={lot_details.get('risk_pct', 0)*100:.1f}% | "
                    f"adjustments={lot_details.get('adjustments', [])}"
                )
            except Exception:
                pass

            origin_id = intent.get("id", "")
            contract_size = self._get_contract_size(inst)

            if action in ("open_long", "open_short", "scale_up"):
                if lots <= 0:
                    try:
                        self.logger.info(f"[LIVE] Skip order: non-positive lots ({lots:.4f}) for {inst}")
                    except Exception:
                        pass
                    continue

                # ANTI-HEDGE CHECK: Prevent opening opposite direction while position exists
                if action in ("open_long", "open_short"):
                    try:
                        pos_snap = norm_positions.get(norm_inst)
                        if pos_snap:
                            existing_side = self._position_side_from_snapshot(pos_snap)
                            if existing_side != 0:
                                is_buy = existing_side > 0
                                # Block if trying to open opposite direction
                                if (existing_side > 0 and side < 0) or (existing_side < 0 and side > 0):
                                    self.logger.warning(
                                        f"[LIVE] 🚫 BLOCKED HEDGE: {inst} wanted={action} but existing={'BUY' if is_buy else 'SELL'}"
                                    )
                                    continue
                                # Block if already have same-direction position
                                if (existing_side > 0 and side > 0) or (existing_side < 0 and side < 0):
                                    self.logger.info(
                                        f"[LIVE] ⏸️ BLOCKED DUPLICATE: {inst} wanted={action}, already have {'BUY' if is_buy else 'SELL'}"
                                    )
                                    continue
                    except Exception as e:
                        self.logger.warning(f"[LIVE] Position check failed: {e}")

                r = self.adapter.market_order(inst, side, lots)
                try:
                    self.logger.info(f"[LIVE] market_order result: {r}")
                except Exception:
                    pass
                if r.get("ok"):
                    px = float(r.get("price", price_hint) or price_hint)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
                        action=action,
                        side=side,
                        units=lots * contract_size,
                        price=px,
                        notional_eur=lots * contract_size * px,
                        realized_pnl=0.0,
                        origin_id=origin_id,
                        comment="live",
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
            elif action == "scale_down":
                if lots <= 0:
                    try:
                        self.logger.info(f"[LIVE] Skip reduce: non-positive lots ({lots:.4f}) for {inst}")
                    except Exception:
                        pass
                    continue
                r = self.adapter.reduce_position(inst, lots, -1 if side > 0 else +1)
                try:
                    self.logger.info(f"[LIVE] reduce_position result: {r}")
                except Exception:
                    pass
                if r.get("ok"):
                    px = float(r.get("price", price_hint) or price_hint)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
                        action="scale_down",
                        side=-1 if side > 0 else +1,
                        units=lots * contract_size,
                        price=px,
                        notional_eur=lots * contract_size * px,
                        realized_pnl=0.0,
                        origin_id=origin_id,
                        comment="live_reduce",
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
            elif action in ("close", "emergency_close"):
                r = self.adapter.close_position(inst)
                try:
                    self.logger.info(f"[LIVE] close_position result: {r}")
                except Exception:
                    pass

        acct_after = self.adapter.get_account_info()
        eq_after = float(acct_after.get("equity", eq_before) or eq_before)
        step_pnl = float(eq_after - eq_before)

        self.balance = float(acct_after.get("balance", self.balance))
        self.equity = float(eq_after)
        self._last_equity = float(eq_after)

        # Trim trades list to prevent memory leak
        if len(self.trades) > self._max_trades:
            self.trades = self.trades[-self._max_trades:]

        try:
            self.logger.info(
                f"[LIVE] Done: equity_after={eq_after:.2f} (Δ={step_pnl:.2f}), fills={len(fills)}"
            )
        except Exception:
            pass
        return fills, step_pnl

    # ─────────────────────────────────────────────────────────
    # SMART LIVE EXECUTION - Intelligent Position Management
    # ─────────────────────────────────────────────────────────
    def _execute_live_smart(self, intents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        """
        Smart live execution with position consolidation and intelligent management.

        Key features:
        1. Syncs actual MT5 positions before any decision
        2. Prevents duplicate positions (max 1 per symbol)
        3. Eliminates hedging (closes opposing positions)
        4. Smart profit-taking and loss-cutting
        5. Converts intents to smart decisions
        """
        fills: List[Dict[str, Any]] = []
        if not self.adapter or not self.adapter.is_connected():
            return fills, 0.0

        # ═══════════════════════════════════════════════════════════════
        # PROP FIRM SAFETY CHECK - Block ALL new trades if limits breached
        # ═══════════════════════════════════════════════════════════════
        prop_limits = self.lot_calculator.check_prop_firm_limits()
        if not prop_limits.get("can_trade", True):
            self.logger.warning(
                f"[SMART] 🚫 PROP FIRM BLOCK: Trading halted - {prop_limits.get('warnings', [])}"
            )
            # Only allow close actions, block all opens
            intents = [i for i in intents if str(i.get("action", "")).lower() in ("close", "close_all", "scale_down")]
            if not intents:
                return fills, 0.0

        acct_before = self.adapter.get_account_info()
        eq_before = float(acct_before.get("equity", 0.0) or 0.0)

        try:
            self.logger.info(
                format_operator_message(
                    "🧠",
                    "SMART_EXECUTE_START",
                    intents=len(intents),
                    equity=f"€{eq_before:.2f}",
                )
            )
        except Exception:
            pass

        # ─────────────────────────────────────────────────────
        # Step 1: Sync actual MT5 positions
        # ─────────────────────────────────────────────────────
        try:
            import MetaTrader5 as mt5  # type: ignore[import]
            raw_positions = list(mt5.positions_get() or [])  # type: ignore[attr-defined]
            mt5_positions = [
                {
                    "symbol": getattr(p, "symbol", ""),
                    "type": getattr(p, "type", 0),
                    "volume": getattr(p, "volume", 0.0),
                    "price_open": getattr(p, "price_open", 0.0),
                    "price_current": getattr(p, "price_current", 0.0),
                    "profit": getattr(p, "profit", 0.0),
                    "time": getattr(p, "time", 0),
                    "ticket": getattr(p, "ticket", 0),
                    "sl": getattr(p, "sl", 0.0),
                    "tp": getattr(p, "tp", 0.0),
                }
                for p in raw_positions
            ]
            self.smart_position_manager.sync_positions(mt5_positions)

            if mt5_positions:
                self.logger.info(
                    format_operator_message(
                        "📊",
                        "MT5_POSITIONS_SYNCED",
                        count=len(mt5_positions),
                        total_pnl=f"€{self.smart_position_manager.get_total_pnl():.2f}",
                    )
                )
        except Exception as e:
            self.logger.warning(f"[SMART] Failed to sync MT5 positions: {e}")
            mt5_positions = []

        # ─────────────────────────────────────────────────────
        # Step 2: Check for and eliminate hedging
        # ─────────────────────────────────────────────────────
        try:
            hedge_cleanup = self.smart_position_manager.needs_hedge_cleanup(mt5_positions)
            if hedge_cleanup:
                self.logger.warning(
                    format_operator_message(
                        "⚠️",
                        "HEDGE_DETECTED",
                        positions_to_close=len(hedge_cleanup),
                    )
                )
                closed_count = 0
                failed_count = 0
                for pos in hedge_cleanup:
                    ticket = pos.get("ticket", 0)
                    symbol = pos.get("symbol", "")
                    if ticket and symbol:
                        try:
                            result = self._close_position_by_ticket(ticket, symbol)
                            if result.get("ok"):
                                self.logger.info(f"[SMART] ✅ Closed hedge ticket {ticket} on {symbol}")
                                # Create proper TradeFill for hedge cleanup
                                fill = TradeFill(
                                    id=f"fill-{uuid.uuid4().hex[:10]}",
                                    ts=time.time(),
                                    step=self.step_idx,
                                    instrument=symbol,
                                    action="hedge_cleanup",
                                    side=0,  # Unknown side for hedge cleanup
                                    units=float(pos.get("volume", 0.0) or 0.0),
                                    price=float(result.get("price", 0) or 0),
                                    notional_eur=0.0,
                                    realized_pnl=float(pos.get("profit", 0.0) or 0.0),
                                    origin_id=f"hedge-{ticket}",
                                    comment="hedge_cleanup",
                                ).as_bus()
                                self.trades.append(fill)
                                fills.append(fill)
                                # Record trade to update cooldowns
                                self.smart_position_manager.record_trade(symbol)
                                closed_count += 1
                            else:
                                error = result.get("error", "unknown")
                                self.logger.error(f"[SMART] ❌ Failed to close hedge ticket {ticket} on {symbol}: {error}")
                                failed_count += 1
                        except Exception as close_err:
                            self.logger.error(f"[SMART] ❌ Exception closing ticket {ticket}: {close_err}")
                            failed_count += 1
                    else:
                        self.logger.warning(f"[SMART] ⚠️ Invalid position data - ticket={ticket}, symbol={symbol}")
                        failed_count += 1

                if closed_count > 0 or failed_count > 0:
                    self.logger.info(f"[SMART] Hedge cleanup: {closed_count} closed, {failed_count} failed")
        except Exception as e:
            self.logger.error(f"[SMART] Hedge cleanup failed: {e}")

        # ─────────────────────────────────────────────────────
        # Step 3: Check existing positions for exits AND scales
        # ─────────────────────────────────────────────────────
        smart_positions = self.smart_position_manager.get_all_positions()

        for raw_symbol, position in smart_positions.items():
            symbol = raw_symbol  # raw broker symbol as used inside SmartPM
            canonical_symbol = self._normalize_symbol(symbol)

            # Get current signal from intents for this specific symbol
            signal_direction = 0
            signal_strength = 0.0

            for intent in intents:
                inst_intent = self._normalize_symbol(intent.get("instrument", ""))
                if inst_intent == canonical_symbol:
                    action = str(intent.get("action", "")).lower()
                    if action in ("open_long", "scale_up"):
                        signal_direction = 1
                    elif action in ("open_short",):
                        signal_direction = -1
                    signal_strength = float(intent.get("intensity", intent.get("confidence", 0.5)) or 0.5)
                    break

            # Fallback to trade_vote_v2 ONLY if it explicitly targets this symbol
            if signal_direction == 0:
                try:
                    trade_vote = self.bus.get("trade_vote_v2", "Executor")
                    if isinstance(trade_vote, dict):
                        vote_symbol = trade_vote.get("symbol", trade_vote.get("instrument", ""))
                        if vote_symbol:
                            vote_symbol_normalized = self._normalize_symbol(vote_symbol)
                            if vote_symbol_normalized == canonical_symbol:
                                vote_action = str(trade_vote.get("action", "")).upper()
                                vote_confidence = float(trade_vote.get("confidence", trade_vote.get("intensity", 0.5)) or 0.0)
                                vote_consensus = float(trade_vote.get("consensus_score", 0.0) or 0.0)

                                position_side = getattr(position, "side", 0)
                                position_is_long = position_side > 0
                                raw_unreal = getattr(position, "unrealized_pnl", getattr(position, "pnl", 0.0))
                                position_is_profitable = raw_unreal > 5.0

                                min_confidence = 0.35
                                min_consensus = 0.55
                                # Slightly stricter bar if profitable and vote is opposing
                                if position_is_profitable:
                                    if (position_is_long and vote_action == "SELL") or \
                                       (not position_is_long and vote_action == "BUY"):
                                        min_confidence = 0.50
                                        min_consensus = 0.65

                                if vote_confidence >= min_confidence and vote_consensus >= min_consensus and vote_action in ("BUY", "SELL"):
                                    if vote_action == "BUY":
                                        signal_direction = 1
                                    elif vote_action == "SELL":
                                        signal_direction = -1
                                    signal_strength = vote_confidence
                except Exception:
                    pass

            decision = self.smart_position_manager.decide(
                symbol=symbol,
                signal_direction=signal_direction,
                signal_strength=signal_strength,
                consensus_confidence=signal_strength,
            )

            # Execute CLOSE/REVERSE decisions
            if decision.action in (PositionAction.CLOSE, PositionAction.REVERSE):
                self.logger.info(
                    format_operator_message(
                        "🎯",
                        "SMART_EXIT",
                        action=decision.action.value,
                        symbol=symbol,
                        reasons=decision.reasons[:2],
                    )
                )
                result = self.adapter.close_position(symbol)
                if result.get("ok"):
                    fills.append({
                        "action": decision.action.value.lower(),
                        "symbol": symbol,
                        "reasons": decision.reasons,
                        "ok": True,
                    })
                    self.smart_position_manager.record_trade(symbol)

            # Execute SCALE_UP decisions
            elif decision.action == PositionAction.SCALE_UP and decision.lots > 0:
                self.logger.info(
                    format_operator_message(
                        "📈",
                        "SMART_SCALE_UP",
                        symbol=symbol,
                        lots=f"+{decision.lots:.2f}",
                        reasons=decision.reasons[:2],
                    )
                )
                result = self.adapter.market_order(symbol, decision.side, decision.lots)
                if result.get("ok"):
                    px = float(result.get("price", 0) or 0)
                    contract_size = self._get_contract_size(symbol)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=symbol,
                        action="scale_up",
                        side=decision.side,
                        units=decision.lots * contract_size,
                        price=px,
                        notional_eur=decision.lots * contract_size * px,
                        realized_pnl=0.0,
                        origin_id="smart_scale",
                        comment="; ".join(decision.reasons[:2]),
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
                    self.smart_position_manager.record_trade(symbol, is_scale=True)

            # Execute SCALE_DOWN decisions
            elif decision.action == PositionAction.SCALE_DOWN and decision.lots > 0:
                self.logger.info(
                    format_operator_message(
                        "📉",
                        "SMART_SCALE_DOWN",
                        symbol=symbol,
                        lots=f"-{decision.lots:.2f}",
                        reasons=decision.reasons[:2],
                    )
                )
                close_side = -decision.side  # Opposite side to reduce
                result = self.adapter.market_order(symbol, close_side, decision.lots)
                if result.get("ok"):
                    px = float(result.get("price", 0) or 0)
                    contract_size = self._get_contract_size(symbol)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=symbol,
                        action="scale_down",
                        side=close_side,
                        units=decision.lots * contract_size,
                        price=px,
                        notional_eur=decision.lots * contract_size * px,
                        realized_pnl=0.0,
                        origin_id="smart_scale",
                        comment="; ".join(decision.reasons[:2]),
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
                    self.smart_position_manager.record_trade(symbol, is_scale=True)

        # ─────────────────────────────────────────────────────
        # Step 4: Process new entry intents with smart filtering
        # ─────────────────────────────────────────────────────
        smart_positions = self.smart_position_manager.get_all_positions()
        smart_pos_by_norm: Dict[str, Tuple[str, Any]] = {}
        try:
            for sym, pos in smart_positions.items():
                smart_pos_by_norm[self._normalize_symbol(sym)] = (sym, pos)
        except Exception:
            smart_pos_by_norm = {}

        adapter_positions_norm = self._normalized_adapter_positions()

        for intent in intents:
            inst_src = intent.get("instrument", "")
            inst = resolve_symbol(inst_src, self.cfg.symbol_overrides, broker=self.cfg.live_broker)
            inst_norm = self._normalize_symbol(inst)
            exec_symbol = smart_pos_by_norm.get(inst_norm, (inst, None))[0]

            action = str(intent.get("action", "")).lower()

            # Explicit close actions from order_queue
            if action in ("close", "emergency_close"):
                # Use smart_symbol if known, otherwise use exec_symbol directly
                # This handles hedged positions that net to 0 (not in smart_pos_by_norm)
                close_symbol = exec_symbol
                if inst_norm in smart_pos_by_norm:
                    close_symbol, _ = smart_pos_by_norm[inst_norm]
                
                self.logger.info(
                    format_operator_message(
                        "🎯",
                        "EXPLICIT_CLOSE",
                        action=action.upper(),
                        symbol=close_symbol,
                        source="order_queue",
                        in_smart_pm=inst_norm in smart_pos_by_norm,
                    )
                )
                result = self.adapter.close_position(close_symbol)
                if result.get("ok"):
                    fills.append({
                        "action": action,
                        "symbol": close_symbol,
                        "source": "order_queue",
                        "realized_pnl": 0.0,  # Live PnL comes from broker
                        "ok": True,
                    })
                    self.smart_position_manager.record_trade(close_symbol)
                else:
                    self.logger.warning(
                        format_operator_message(
                            "⚠️",
                            "CLOSE_FAILED",
                            symbol=close_symbol,
                            error=result.get("error", "unknown"),
                        )
                    )
                continue

            # Determine signal from intent
            signal_direction = {"open_long": 1, "scale_up": 1, "open_short": -1, "scale_down": -1}.get(action, 0)
            signal_strength = float(intent.get("intensity", intent.get("confidence", 0.5)) or 0.5)

            decision = self.smart_position_manager.decide(
                symbol=exec_symbol,
                signal_direction=signal_direction,
                signal_strength=signal_strength,
                consensus_confidence=signal_strength,
            )

            if decision.action == PositionAction.HOLD:
                self.logger.info(
                    format_operator_message(
                        "⏸️",
                        "SMART_HOLD",
                        symbol=exec_symbol,
                        original_action=action,
                        reasons=decision.reasons[:2],
                    )
                )
                continue

            # New opens: enforce anti-hedge at MT5 level (in case SmartPM is stale)
            if decision.action in (PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT):
                try:
                    pos_snap = adapter_positions_norm.get(inst_norm)
                    if pos_snap:
                        existing_side = self._position_side_from_snapshot(pos_snap)
                        if existing_side != 0:
                            is_buy = existing_side > 0
                            if (existing_side > 0 and decision.side < 0) or (existing_side < 0 and decision.side > 0):
                                self.logger.warning(
                                    format_operator_message(
                                        "🚫",
                                        "BLOCKED_HEDGE",
                                        symbol=exec_symbol,
                                        wanted=decision.action.value,
                                        existing="BUY" if is_buy else "SELL",
                                        reason="Would create hedge position",
                                    )
                                )
                                continue
                            if (existing_side > 0 and decision.side > 0) or (existing_side < 0 and decision.side < 0):
                                self.logger.info(
                                    format_operator_message(
                                        "⏸️",
                                        "BLOCKED_DUPLICATE",
                                        symbol=exec_symbol,
                                        wanted=decision.action.value,
                                        existing="BUY" if is_buy else "SELL",
                                        reason="Already have position in same direction",
                                    )
                                )
                                continue
                except Exception as e:
                    self.logger.warning(f"[SMART] MT5 position check failed: {e}")

                # ═══════════════════════════════════════════════════════════════
                # UNIFIED LOT CALCULATION - Use the central lot calculator
                # ═══════════════════════════════════════════════════════════════
                volatility = self._get_current_volatility(exec_symbol)
                lots, lot_details = self.lot_calculator.calculate_lots(
                    symbol=exec_symbol,
                    signal_strength=signal_strength,
                    volatility=volatility,
                )
                
                # Log the unified lot calculation
                try:
                    self.logger.info(
                        f"[SMART] 📊 UNIFIED_LOT_CALC: {exec_symbol} | signal={signal_strength:.2f} | "
                        f"lots={lots:.2f} | balance=€{lot_details.get('balance', 0):.0f} | "
                        f"risk={lot_details.get('risk_pct', 0)*100:.1f}%"
                    )
                except Exception:
                    pass

                self.logger.info(
                    format_operator_message(
                        "🚀",
                        "SMART_OPEN",
                        action=decision.action.value,
                        symbol=exec_symbol,
                        lots=f"{lots:.2f}",
                        reasons=decision.reasons[:2],
                    )
                )

                result = self.adapter.market_order(exec_symbol, decision.side, lots)
                if result.get("ok"):
                    px = float(result.get("price", 0) or 0)
                    contract_size = self._get_contract_size(exec_symbol)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=exec_symbol,
                        action=decision.action.value.lower(),
                        side=decision.side,
                        units=lots * contract_size,
                        price=px,
                        notional_eur=lots * contract_size * px,
                        realized_pnl=0.0,
                        origin_id=intent.get("id", ""),
                        comment="smart_open",
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
                    self.smart_position_manager.record_trade(exec_symbol)

            elif decision.action == PositionAction.SCALE_UP:
                lots = decision.lots or self.adapter.cfg.min_lot

                self.logger.info(
                    format_operator_message(
                        "📈",
                        "SMART_SCALE_UP",
                        symbol=exec_symbol,
                        lots=f"{lots:.2f}",
                        reasons=decision.reasons[:2],
                    )
                )

                result = self.adapter.market_order(exec_symbol, decision.side, lots)
                if result.get("ok"):
                    px = float(result.get("price", 0) or 0)
                    contract_size = self._get_contract_size(exec_symbol)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=exec_symbol,
                        action="scale_up",
                        side=decision.side,
                        units=lots * contract_size,
                        price=px,
                        notional_eur=lots * contract_size * px,
                        realized_pnl=0.0,
                        origin_id=intent.get("id", ""),
                        comment="smart_scale",
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
                    self.smart_position_manager.record_trade(exec_symbol, is_scale=True)

        # ─────────────────────────────────────────────────────
        # Step 5: Update account state
        # ─────────────────────────────────────────────────────
        acct_after = self.adapter.get_account_info()
        eq_after = float(acct_after.get("equity", eq_before) or eq_before)
        step_pnl = float(eq_after - eq_before)

        self.balance = float(acct_after.get("balance", self.balance))
        self.equity = float(eq_after)
        self._last_equity = float(eq_after)

        self.logger.info(
            format_operator_message(
                "✅",
                "SMART_EXECUTE_DONE",
                equity=f"€{eq_after:.2f}",
                pnl=f"€{step_pnl:+.2f}",
                fills=len(fills),
            )
        )

        # Trim trades list to prevent memory leak
        if len(self.trades) > self._max_trades:
            self.trades = self.trades[-self._max_trades:]

        return fills, step_pnl

    # ─────────────────────────────────────────────────────────
    # Symbol helpers / adapter position helpers
    # ─────────────────────────────────────────────────────────
    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for comparison (no separators, upper-case)."""
        return (symbol or "").replace("/", "").replace("_", "").upper()

    def _normalized_adapter_positions(self) -> Dict[str, Dict[str, Any]]:
        """
        Return adapter positions keyed by normalized symbol.

        Handles both dict and list-of-dicts shapes defensively.
        """
        if not self.adapter or not self.adapter.is_connected():
            return {}
        try:
            raw: Any = self.adapter.sync_positions()
        except Exception:
            return {}

        if raw is None:
            return {}

        norm: Dict[str, Dict[str, Any]] = {}

        iterable: List[Tuple[str, Dict[str, Any]]] = []
        if isinstance(raw, dict):
            iterable = list(raw.items())
        elif isinstance(raw, list):
            for p in raw:
                if not isinstance(p, dict):
                    continue
                sym = p.get("instrument") or p.get("symbol")
                if sym:
                    iterable.append((sym, p))

        for sym, snap in iterable:
            try:
                norm[self._normalize_symbol(sym)] = snap
            except Exception:
                continue
        return norm

    def _position_side_from_snapshot(self, snap: Dict[str, Any]) -> int:
        """
        Extract side from a position snapshot.

        Priority:
        - explicit "side" field (+1/-1)
        - MT5-style "type" field (0=BUY, 1=SELL)
        - string "direction" field
        """
        side = 0
        try:
            side = int(snap.get("side", 0) or 0)
        except Exception:
            side = 0
        if side:
            return 1 if side > 0 else -1

        t = snap.get("type", None)
        if t is not None:
            try:
                t_int = int(t)
                # Common MT5 mapping: 0=BUY, 1=SELL
                if t_int == 0:
                    return 1
                if t_int == 1:
                    return -1
            except Exception:
                pass

        d = str(snap.get("direction", "")).lower()
        if d in ("buy", "long", "1", "+1"):
            return 1
        if d in ("sell", "short", "-1"):
            return -1
        return 0

    def _close_position_by_ticket(self, ticket: int, symbol: str) -> Dict[str, Any]:
        """Close a specific position by ticket number."""
        self.logger.debug(f"[CLOSE_TICKET] Attempting to close ticket {ticket} on {symbol}")

        if not self.adapter or not self.adapter.is_connected():
            self.logger.warning("[CLOSE_TICKET] Adapter not connected")
            return {"ok": False, "error": "not_connected"}

        try:
            import MetaTrader5 as mt5  # type: ignore[import]

            # Get position info
            position = mt5.positions_get(ticket=ticket)  # type: ignore[attr-defined]
            if not position:
                self.logger.warning(f"[CLOSE_TICKET] Position {ticket} not found in MT5")
                return {"ok": False, "error": "position_not_found"}

            pos = position[0]
            lots = getattr(pos, "volume", 0.0)
            pos_type = getattr(pos, "type", 0)

            self.logger.info(f"[CLOSE_TICKET] Found position: ticket={ticket}, lots={lots}, type={pos_type}")

            # Close by opening opposite
            close_type = mt5.ORDER_TYPE_SELL if pos_type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": close_type,
                "position": ticket,
                "magic": 123456,
                "comment": "smart_close",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            # Get price
            tick = mt5.symbol_info_tick(symbol)  # type: ignore[attr-defined]
            if tick:
                request["price"] = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            else:
                self.logger.warning(f"[CLOSE_TICKET] No tick data for {symbol}")

            self.logger.info(f"[CLOSE_TICKET] Sending close request: {request}")
            result = mt5.order_send(request)  # type: ignore[attr-defined]

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(f"[CLOSE_TICKET] ✅ Successfully closed ticket {ticket}")
                return {"ok": True, "price": getattr(result, "price", 0)}
            else:
                retcode = getattr(result, "retcode", "unknown") if result else "no_result"
                comment = getattr(result, "comment", "") if result else ""
                self.logger.error(f"[CLOSE_TICKET] ❌ MT5 rejected close: retcode={retcode}, comment={comment}")
                return {"ok": False, "error": f"{retcode}: {comment}"}

        except Exception as e:
            self.logger.error(f"[CLOSE_TICKET] Exception: {e}")
            return {"ok": False, "error": str(e)}

    # ─────────────────────────────────────────────────────────
    # bus publishing
    # ─────────────────────────────────────────────────────────
    def _publish_all(
        self,
        *,
        exec_fills: List[Dict[str, Any]],
        accepted: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
        step_pnl: float,
        realized_step: float = 0.0,
        unrealized: float = 0.0,
        reason: str = "",
    ) -> None:
        pos_snap: Dict[str, Any] = {}
        mode = self._resolve_mode()
        if mode == "live" and self.adapter:
            pos_snap = self.adapter.sync_positions()
        else:
            # Pass current price to as_bus() for proper display
            for inst, p in self.positions.items():
                current_price = self._sim_price(inst, p.side)
                pos_snap[inst] = p.as_bus(last_price=current_price)

        trade_ledger = list(self.trades)
        recent = trade_ledger[-50:] if trade_ledger else []

        order_data = {"accepted": accepted, "rejected": rejected, "step": int(self.step_idx)}
        execution_data = {"fills": exec_fills, "step": int(self.step_idx)}
        portfolio_metrics = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),
            "realized_pnl_step": float(realized_step),
            "unrealized_pnl": float(unrealized),
            "step": int(self.step_idx),
        }
        market_state = {"balance": float(self.balance), "equity": float(self.equity), "step": int(self.step_idx)}

        # Core bus writes
        self.bus.set("positions", pos_snap, thesis="Positions snapshot (executor)")
        self.bus.set("trades", trade_ledger, thesis="Trade ledger (executor)")
        self.bus.set("recent_trades", recent, thesis="Recent fills (executor)")
        self.bus.set("order_data", order_data, thesis="Orders seen this step (executor)")
        self.bus.set("execution_data", execution_data, thesis="Fills this step (executor)")
        self.bus.set("execution_reports", exec_fills, thesis="Fills alias (executor)")

        # CRITICAL: also publish current step fills for memory system
        self.bus.set("current_fills", exec_fills, thesis="Current step fills (executor)")
        try:
            self.bus.set("trade_data", trade_ledger, thesis="alias: trade_data (executor)")
        except Exception:
            pass
        self.bus.set("portfolio_metrics", portfolio_metrics, thesis="Portfolio metrics (executor)")
        self.bus.set("trading_result", {"pnl": float(step_pnl)}, thesis="Per-step Δequity (executor)")
        self.bus.set("market_state", market_state, thesis="Market state (executor)")
        try:
            self.bus.set("current_pnl", float(step_pnl), thesis="alias: current_pnl (executor)")
        except Exception:
            pass

        # Baselines for downstream consumers
        try:
            self.bus.set("pending_orders", order_data.get("accepted", []), thesis="Orders pending execution (baseline)")
            self.bus.set(
                "account_state",
                {
                    "balance": float(self.balance),
                    "equity": float(self.equity),
                    "initial_balance": float(self.initial_balance),
                    "step": int(self.step_idx),
                },
                thesis="Account state (executor)",
            )
        except Exception:
            pass

        # **raw aliases to avoid BUS MISS for simple readers**
        self.bus.set("balance", float(self.balance), thesis="alias: balance (executor)")
        self.bus.set("equity", float(self.equity), thesis="alias: equity (executor)")

        # compact snapshot for consumers that want one read
        self.bus.set("account_snapshot", self._account_snapshot(pos_snap), thesis="Account snapshot (executor)")

        # Helpful aliases
        self.bus.set("current_positions", pos_snap, thesis="alias: current_positions")
        self.bus.set(
            "pnl_data",
            {
                "balance": float(self.balance),
                "equity": float(self.equity),
                "current_pnl": float(step_pnl),
                "realized_pnl_step": float(realized_step),
                "unrealized_pnl": float(unrealized),
                "step": int(self.step_idx),
            },
            thesis="alias: pnl_data",
        )

        # Legacy/risk-friendly alias: position_data { positions: [{instrument, size, ...}], count }
        try:
            pos_list = []
            for inst, p in (pos_snap or {}).items():
                notional = float(p.get("notional_eur", 0.0) or 0.0)
                units = float(p.get("units", 0.0) or 0.0)
                entry_price = float(p.get("entry_price", 0.0) or 0.0)
                size = abs(notional) if abs(notional) > 0 else (abs(units * entry_price) if (units and entry_price) else abs(units))
                entry = {"instrument": inst, "size": float(size)}
                for k, v in p.items():
                    if k != "instrument":
                        entry[k] = v
                pos_list.append(entry)
            self.bus.set("position_data", {"positions": pos_list, "count": len(pos_list)}, thesis="alias: position_data")
        except Exception as e:
            self.debugger.record_error(f"position_data_alias_error: {e}")

        # Simple snapshot log
        self.logger.info(
            format_operator_message(
                "[EXECUTOR]", "SNAPSHOT",
                step=self.step_idx,
                fills=len(exec_fills),
                orders_ok=len(accepted),
                orders_rej=len(rejected),
                balance=f"{self.balance:.2f}",
                equity=f"{self.equity:.2f}",
                step_pnl=f"{step_pnl:.2f}",
                realized_step=f"{realized_step:.2f}",
                unrealized=f"{unrealized:.2f}",
                reason=reason or "ok",
            )
        )

    def _log_unified_cycle(
        self,
        mode: str,
        q_count: int,
        dec_count: int,
        accepted: List[Dict[str, Any]],
        rejected: List[Dict[str, Any]],
        fills: List[Dict[str, Any]],
        positions_after: Dict[str, Any],
        balance_before: float,
        equity_before: float,
        realized_step: float,
        unreal_after: float,
        step_pnl: float,
        processing_ms: float,
    ) -> None:
        """Generate unified execution cycle log."""
        from collections import Counter

        # For sim, positions_before reflects actual PositionSnap map.
        # For live, self.positions is empty (we rely on broker), so this is best-effort.
        positions_before = {k: v for k, v in self.positions.items()}
        before_keys = set(positions_before.keys())
        after_keys = set(positions_after.keys())

        positions_opened = list(after_keys - before_keys)
        positions_closed = list(before_keys - after_keys)
        positions_modified = [
            k for k in (after_keys & before_keys)
            if positions_after.get(k, {}).get("units") != positions_before.get(k, PositionSnap("", 0, 0, 0)).units
        ]

        rejection_reasons = Counter([r.get("reason", "unknown") for r in rejected])
        fills_by_instrument = Counter([f.get("instrument", "N/A") for f in fills])
        total_notional = sum(abs(float(f.get("notional_eur", 0.0))) for f in fills)

        issues = []
        if accepted and not fills:
            issues.append("accepted_but_no_fills")
        if float(step_pnl) > 0 and float(self.equity) < float(equity_before):
            issues.append("pnl_positive_but_equity_down")
        if float(step_pnl) < 0 and float(self.equity) > float(equity_before):
            issues.append("pnl_negative_but_equity_up")

        entry = ExecutionCycleEntry(
            step=self.step_idx,
            mode=mode,
            timestamp=dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            orders_received=q_count + dec_count,
            orders_accepted=len(accepted),
            orders_rejected=len(rejected),
            rejected_reasons=dict(rejection_reasons),
            fills_count=len(fills),
            fills_by_instrument=dict(fills_by_instrument),
            total_notional=total_notional,
            positions_before={},  # Simplified for now
            positions_after=positions_after,
            positions_opened=positions_opened,
            positions_closed=positions_closed,
            positions_modified=positions_modified,
            balance_before=balance_before,
            balance_after=float(self.balance),
            equity_before=equity_before,
            equity_after=float(self.equity),
            realized_pnl=realized_step,
            unrealized_pnl=unreal_after,
            step_pnl=step_pnl,
            trades_this_step=[f for f in fills if f.get("realized_pnl", 0.0) != 0],
            execution_time_ms=processing_ms,
            issues=issues,
            accepted_details=accepted[:10],
            rejected_details=rejected[:10],
            fill_details=fills[:20],
        )

        self.unified_logger.log_execution_cycle(entry)

    # ─────────────────────────────────────────────────────────
    # State Persistence - Save/Load executor state
    # ─────────────────────────────────────────────────────────
    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.

        Saves:
        - Balance and equity
        - Trade history (last 500 trades)
        - Position state (for sim mode)
        - Step counter
        """
        return {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "step_idx": int(self.step_idx),
            "trades": list(self.trades[-500:]) if self.trades else [],
            "positions": {k: v.as_bus() if hasattr(v, "as_bus") else v for k, v in self.positions.items()},
            "_last_equity": float(self._last_equity),
            "_cumulative_pnl": float(self._cumulative_pnl),
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """
        Restore custom state from persistence.
        """
        if not state:
            return

        self.balance = float(state.get("balance", self.balance))
        self.equity = float(state.get("equity", self.equity))
        self.step_idx = int(state.get("step_idx", self.step_idx))
        self._last_equity = float(state.get("_last_equity", self.equity))
        self._cumulative_pnl = float(state.get("_cumulative_pnl", 0.0))

        trades = state.get("trades", [])
        if trades:
            self.trades = list(trades)
            self.logger.info(f"📂 Restored {len(self.trades)} trades from state")

        # Note: positions are synced from MT5 in live mode, so we do not restore them here
        # to avoid conflicts; for sim, positions can be restored externally if needed.
