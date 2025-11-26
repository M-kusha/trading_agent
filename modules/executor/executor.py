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

from .shared.types import PositionSnap, TradeFill
from .shared.utils import SafeBus, round_to_step, resolve_symbol
from .debug.debugger import ExecutorDebugManager
from .adapters.base_adapter import BaseLiveAdapter, LiveAdapterConfig
from .adapters.mt5_adapter import MT5Adapter
from .unified_logger import UnifiedExecutorLogger, ExecutionCycleEntry

# Smart Position Management
from modules.position.smart_position_manager import (
    SmartPositionManager, 
    SmartDecision, 
    PositionAction,
    SmartPositionConfig,
)


@dataclass
class ExecutorConfig:
    execution_mode: str = "sim"        # 'sim' | 'live'
    live_broker: str = "mt5"
    symbol_overrides: Optional[Dict[str, str]] = None
    lot_step: float = 0.01
    min_lot: float = 0.20              # Minimum 0.20 lots for meaningful trades
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
        # Final fallback aligns with environment default (envs/config.py: initial_balance=3000.0)
        self.initial_balance: float = float(3000.0 if _cfg_ib is None else _cfg_ib)
        self.balance: float = float(self.initial_balance)
        self.equity: float = float(self.balance)
        self._last_equity: float = float(self.equity)
        self.positions: Dict[str, PositionSnap] = {}
        self.trades: List[Dict[str, Any]] = []
        self.closed_positions: List[Dict[str, Any]] = []  # Track closed positions for win rate
        self.step_idx: int = 0
        self._seen_ids: Set[str] = set()
        self._cumulative_pnl: float = 0.0  # Track cumulative P&L for state persistence

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
                pos_snap = {}
                for k, v in self.positions.items():
                    current_price = self._sim_price(k, v.side)
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
                positions_after = {}
                for k, v in self.positions.items():
                    current_price = self._sim_price(k, v.side)
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
            "closed_positions": list(self.closed_positions),  # For win rate tracking
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
        try:
            memory_gate = self.bus.get("memory_gate", "Executor", default=None)
            if isinstance(memory_gate, dict) and memory_gate.get("veto", False):
                memory_veto = True
                memory_veto_reasons = memory_gate.get("reasons", ["Memory system vetoed"])
                self.logger.warning(format_operator_message(
                    icon="🧠",
                    message="MEMORY_VETO_ACTIVE",
                    reasons=memory_veto_reasons[:3],
                ))
        except Exception:
            pass

        # explicit order_queue
        oq = self.bus.get("order_queue", "Executor", default=[])
        if isinstance(oq, list):
            q_count = len(oq)
            for item in oq[: self.cfg.max_orders_per_step]:
                intent = self._normalize_order_item(item)
                if not intent:
                    rejected.append({"reason": "bad_order_queue_item", "raw": item})
                # Memory veto check - reject new opening orders
                elif memory_veto and intent.get("action", "").lower() in ("open_long", "open_short", "buy", "sell"):
                    rejected.append({
                        "reason": "memory_veto",
                        "intent": intent,
                        "memory_reasons": memory_veto_reasons
                    })
                elif self._passes_filters(intent):
                    if intent["id"] not in self._seen_ids:
                        accepted.append(intent); self._seen_ids.add(intent["id"])
                    else:
                        rejected.append({"reason": "duplicate_id", "intent": intent})
                else:
                    reason = self._filter_reason(intent)
                    rejected.append({"reason": reason, "intent": intent})

        # consume queue: keep consumption internal to Executor (no write-back here)

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
                        accepted.append(intent); self._seen_ids.add(intent["id"])
                    else:
                        rejected.append({"reason": self._filter_reason(intent), "intent": intent})

        # Debug summary of intents (compact version - full details in unified logger)
        try:
            self.logger.debug(
                f"[EXEC] Collected: queue={q_count} decisions={dec_count} accepted={len(accepted)} rejected={len(rejected)}"
            )
        except Exception:
            pass

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
                    # Best-effort pruning; tolerate owner discipline if enforced
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
            return f"low_confidence<{self.cfg.min_confidence}"
        if abs(float(intent.get("intensity", 0.0))) < self.cfg.min_intensity and act not in ("close", "emergency_close"):
            return f"low_intensity<{self.cfg.min_intensity}"
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
        if 'XAU' in sym_upper or 'GOLD' in sym_upper:
            return 100.0  # Gold: 100 oz per lot
        if 'XAG' in sym_upper or 'SILVER' in sym_upper:
            return 5000.0  # Silver: 5000 oz per lot
        if 'BTC' in sym_upper:
            return 1.0  # Bitcoin: 1 BTC per lot
        if 'ETH' in sym_upper:
            return 1.0  # Ethereum: 1 ETH per lot
        
        # Default: Forex 100,000 units per lot
        return float(self.cfg.contract_size)

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
                            px = float(val); break
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
        """Track a fully closed position for win rate and analytics."""
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
                    self.trades.append(fill); fills.append(fill)
                    self._track_closed_position(p, price, realized, "reverse")
                    del self.positions[inst]

                if add_units > 0:
                    notional = add_units * price
                    self.positions[inst] = PositionSnap(
                        inst, side_from_action, add_units, price,
                        notional_eur=notional,
                        open_time=time.time(),
                        entry_step=self.step_idx
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
                    self.trades.append(fill); fills.append(fill)

            elif action == "scale_up":
                if add_units <= 0:
                    continue
                if inst not in self.positions:
                    notional = add_units * price
                    self.positions[inst] = PositionSnap(
                        inst, +1, add_units, price,
                        notional_eur=notional,
                        open_time=time.time(),
                        entry_step=self.step_idx
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
                    self.trades.append(fill); fills.append(fill)
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
                        self.trades.append(fill); fills.append(fill)
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
                        self.trades.append(fill); fills.append(fill)

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
                self.trades.append(fill); fills.append(fill)

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
                    self.trades.append(fill); fills.append(fill)
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

        if want_breakdown:
            return fills, step_pnl, float(realized_step), float(unreal)
        return fills, step_pnl, 0.0, 0.0  # not used

    # ─────────────────────────────────────────────────────────
    # LIVE execution
    # ─────────────────────────────────────────────────────────
    def _execute_live(self, intents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        fills: List[Dict[str, Any]] = []
        if not self.adapter or not self.adapter.is_connected():
            return fills, 0.0

        acct_before = self.adapter.get_account_info()
        eq_before = float(acct_before.get("equity", 0.0) or 0.0)
        try:
            self.logger.info(f"[LIVE] Execute intents: n={len(intents)} | equity_before={eq_before:.2f}")
        except Exception:
            pass

        for intent in intents:
            inst_src = intent["instrument"]
            inst = resolve_symbol(inst_src, self.cfg.symbol_overrides, broker=self.cfg.live_broker)
            action = str(intent["action"]).lower()
            side = {"open_long": +1, "scale_up": +1, "open_short": -1, "scale_down": -1}.get(action, 0)

            price_hint = (self.adapter.get_prices(inst) or {}).get("mid", 0.0) or 1.0
            units = float(intent.get("units", 0.0) or 0.0)
            size_eur = float(intent.get("size_eur", 0.0) or 0.0)
            if units <= 0 and size_eur > 0 and price_hint > 0:
                units = size_eur / price_hint
            
            # Use symbol-specific contract size
            contract_size = self._get_contract_size(inst)
            lots = max(units / contract_size, 0.0)
            lots = round_to_step(lots, self.adapter.cfg.lot_step)
            lots = max(lots, self.adapter.cfg.min_lot) if lots > 0 else 0.0
            # If the order carries positive size (units/size_eur) but rounding drove lots to 0,
            # enforce the broker min lot so we don't silently drop accepted intents.
            if lots <= 0 and (units > 0 or size_eur > 0):
                try:
                    self.logger.info(
                        f"[LIVE] Enforce min lot: computed_lots=0 -> min_lot={self.adapter.cfg.min_lot:.4f} for {inst}"
                    )
                except Exception:
                    pass
                lots = self.adapter.cfg.min_lot

            origin_id = intent.get("id", "")

            if action in ("open_long", "open_short", "scale_up"):
                if lots <= 0:
                    try:
                        self.logger.info(f"[LIVE] Skip order: non-positive lots ({lots:.4f}) for {inst}")
                    except Exception:
                        pass
                    continue
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
                    self.trades.append(fill); fills.append(fill)
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
                    self.trades.append(fill); fills.append(fill)
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
                                fills.append({
                                    "action": "hedge_cleanup",
                                    "symbol": symbol,
                                    "ticket": ticket,
                                    "ok": True,
                                })
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
        # Read signal from InfoBus since PositionManager may emit HOLD
        # ─────────────────────────────────────────────────────
        for symbol, position in self.smart_position_manager.get_all_positions().items():
            # Get current signal from InfoBus - MUST be per-symbol!
            signal_direction = 0
            signal_strength = 0.0
            
            # First check intents for this specific symbol (most accurate)
            for intent in intents:
                if self._normalize_symbol(intent.get("instrument", "")) == symbol:
                    action = str(intent.get("action", "")).lower()
                    if action in ("open_long", "scale_up"):
                        signal_direction = 1
                    elif action in ("open_short",):
                        signal_direction = -1
                    signal_strength = float(intent.get("intensity", intent.get("confidence", 0.5)) or 0.5)
                    break
            
            # Fallback to trade_vote_v2 ONLY if it matches this symbol
            if signal_direction == 0:
                try:
                    trade_vote = self.bus.get("trade_vote_v2", "Executor")
                    if isinstance(trade_vote, dict):
                        # Check if this vote is for our symbol or is symbol-agnostic
                        vote_symbol = trade_vote.get("symbol", trade_vote.get("instrument", ""))
                        vote_symbol_normalized = self._normalize_symbol(vote_symbol) if vote_symbol else ""
                        
                        # Only apply global vote if no symbol specified or matches
                        if not vote_symbol or vote_symbol_normalized == symbol:
                            vote_action = str(trade_vote.get("action", "")).upper()
                            if vote_action == "BUY":
                                signal_direction = 1
                            elif vote_action == "SELL":
                                signal_direction = -1
                            signal_strength = float(trade_vote.get("confidence", trade_vote.get("intensity", 0.5)) or 0.5)
                except Exception:
                    pass
            
            # Get smart decision for this position
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
                # Scale down = close partial position (opposite side order)
                close_side = -decision.side  # Opposite to reduce
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
        for intent in intents:
            inst_src = intent.get("instrument", "")
            inst = resolve_symbol(inst_src, self.cfg.symbol_overrides, broker=self.cfg.live_broker)
            action = str(intent.get("action", "")).lower()
            
            # Handle explicit close actions from order_queue (e.g., from PositionManager)
            if action in ("close", "emergency_close"):
                # Check if we have a position in this instrument
                existing_positions = self.smart_position_manager.get_all_positions()
                if inst in existing_positions:
                    position = self.positions.get(inst)  # Get our sim position for tracking
                    self.logger.info(
                        format_operator_message(
                            "🎯",
                            "EXPLICIT_CLOSE",
                            action=action.upper(),
                            symbol=inst,
                            source="order_queue",
                        )
                    )
                    result = self.adapter.close_position(inst)
                    if result.get("ok"):
                        # Track the close with proper details
                        close_price = float(result.get("price", 0) or 0)
                        realized_pnl = 0.0
                        if position:
                            realized_pnl = (close_price - position.entry_price) * position.side * position.units
                            self._track_closed_position(position, close_price, realized_pnl, action)
                            # Remove from positions and update balance
                            if inst in self.positions:
                                del self.positions[inst]
                            self.balance += realized_pnl
                        
                        fills.append({
                            "action": action,
                            "symbol": inst,
                            "source": "order_queue",
                            "realized_pnl": realized_pnl,
                            "ok": True,
                        })
                        self.smart_position_manager.record_trade(inst)
                continue
            
            # Determine signal from intent
            signal_direction = {"open_long": 1, "scale_up": 1, "open_short": -1, "scale_down": -1}.get(action, 0)
            signal_strength = float(intent.get("intensity", intent.get("confidence", 0.5)) or 0.5)
            
            # Get smart decision
            decision = self.smart_position_manager.decide(
                symbol=inst,
                signal_direction=signal_direction,
                signal_strength=signal_strength,
                consensus_confidence=signal_strength,
            )
            
            # Only execute if smart manager approves
            if decision.action == PositionAction.HOLD:
                self.logger.info(
                    format_operator_message(
                        "⏸️",
                        "SMART_HOLD",
                        symbol=inst,
                        original_action=action,
                        reasons=decision.reasons[:2],
                    )
                )
                continue
            
            if decision.action in (PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT):
                # Calculate lots
                lots = decision.lots
                if lots <= 0:
                    price_hint = (self.adapter.get_prices(inst) or {}).get("mid", 1.0) or 1.0
                    size_eur = float(intent.get("size_eur", 0.0) or 0.0)
                    contract_size = self._get_contract_size(inst)
                    if size_eur > 0:
                        units = size_eur / price_hint
                        lots = max(units / contract_size, 0.0)
                        lots = round_to_step(lots, self.adapter.cfg.lot_step)
                        lots = max(lots, self.adapter.cfg.min_lot) if lots > 0 else self.adapter.cfg.min_lot
                    else:
                        lots = self.adapter.cfg.min_lot
                
                self.logger.info(
                    format_operator_message(
                        "🚀",
                        "SMART_OPEN",
                        action=decision.action.value,
                        symbol=inst,
                        lots=f"{lots:.2f}",
                        reasons=decision.reasons[:2],
                    )
                )
                
                result = self.adapter.market_order(inst, decision.side, lots)
                if result.get("ok"):
                    px = float(result.get("price", 0) or 0)
                    contract_size = self._get_contract_size(inst)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
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
                    self.smart_position_manager.record_trade(inst)
            
            elif decision.action == PositionAction.SCALE_UP:
                lots = decision.lots or self.adapter.cfg.min_lot
                
                self.logger.info(
                    format_operator_message(
                        "📈",
                        "SMART_SCALE_UP",
                        symbol=inst,
                        lots=f"{lots:.2f}",
                        reasons=decision.reasons[:2],
                    )
                )
                
                result = self.adapter.market_order(inst, decision.side, lots)
                if result.get("ok"):
                    px = float(result.get("price", 0) or 0)
                    contract_size = self._get_contract_size(inst)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
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
                    self.smart_position_manager.record_trade(inst, is_scale=True)

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
        
        return fills, step_pnl

    def _normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for comparison."""
        return symbol.replace("/", "").replace("_", "").upper()

    def _close_position_by_ticket(self, ticket: int, symbol: str) -> Dict[str, Any]:
        """Close a specific position by ticket number."""
        self.logger.debug(f"[CLOSE_TICKET] Attempting to close ticket {ticket} on {symbol}")
        
        if not self.adapter or not self.adapter.is_connected():
            self.logger.warning(f"[CLOSE_TICKET] Adapter not connected")
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
        reason: str = ""
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

        # CRITICAL: Also publish current step fills for memory system
        # Memory needs access to ALL fills including opens (not just closes with pnl)
        self.bus.set("current_fills", exec_fills, thesis="Current step fills (executor)")
        # Alias for readers expecting 'trade_data'
        try:
            self.bus.set("trade_data", trade_ledger, thesis="alias: trade_data (executor)")
        except Exception:
            pass
        self.bus.set("portfolio_metrics", portfolio_metrics, thesis="Portfolio metrics (executor)")
        self.bus.set("trading_result", {"pnl": float(step_pnl)}, thesis="Per-step Δequity (executor)")
        self.bus.set("market_state", market_state, thesis="Market state (executor)")
        # direct alias for simple consumers
        try:
            self.bus.set("current_pnl", float(step_pnl), thesis="alias: current_pnl (executor)")
        except Exception:
            pass

        # Baselines for downstream consumers
        try:
            self.bus.set("pending_orders", order_data.get("accepted", []), thesis="Orders pending execution (baseline)")
            self.bus.set("account_state", {
                "balance": float(self.balance),
                "equity": float(self.equity),
                "initial_balance": float(self.initial_balance),
                "step": int(self.step_idx)
            }, thesis="Account state (executor)")
        except Exception:
            pass

        # **raw aliases to avoid BUS MISS for simple readers**
        self.bus.set("balance", float(self.balance), thesis="alias: balance (executor)")
        self.bus.set("equity", float(self.equity), thesis="alias: equity (executor)")

        # compact snapshot for consumers that want one read
        self.bus.set("account_snapshot", self._account_snapshot(pos_snap), thesis="Account snapshot (executor)")

        # Helpful aliases (write to bus regardless of cfg to satisfy strict readers)
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
                # ‘size’ uses notional if present, else units*entry_price, else units
                notional = float(p.get("notional_eur", 0.0) or 0.0)
                units = float(p.get("units", 0.0) or 0.0)
                entry_price = float(p.get("entry_price", 0.0) or 0.0)
                size = abs(notional) if abs(notional) > 0 else (abs(units * entry_price) if (units and entry_price) else abs(units))
                entry = {"instrument": inst, "size": float(size)}
                # include rest of snapshot for richer consumers
                for k, v in p.items():
                    if k != "instrument":
                        entry[k] = v
                pos_list.append(entry)
            self.bus.set("position_data", {"positions": pos_list, "count": len(pos_list)}, thesis="alias: position_data")
        except Exception as e:
            self.debugger.record_error(f"position_data_alias_error: {e}")

        # Simple snapshot log (keep for quick reference)
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
        """Generate unified execution cycle log"""
        from collections import Counter, defaultdict

        # Detect position changes
        positions_before = {k: v for k, v in self.positions.items()}
        positions_opened = []
        positions_closed = []
        positions_modified = []

        # Positions that existed before
        before_keys = set(positions_before.keys())
        after_keys = set(positions_after.keys())

        positions_opened = list(after_keys - before_keys)
        positions_closed = list(before_keys - after_keys)
        positions_modified = [k for k in (after_keys & before_keys)
                            if positions_after.get(k, {}).get('units') != positions_before.get(k, PositionSnap('', 0, 0, 0)).units]

        # Rejection reasons
        rejection_reasons = Counter([r.get('reason', 'unknown') for r in rejected])

        # Fills by instrument
        fills_by_instrument = Counter([f.get('instrument', 'N/A') for f in fills])

        # Total notional
        total_notional = sum(abs(float(f.get('notional_eur', 0.0))) for f in fills)

        # Detect issues
        issues = []
        if accepted and not fills:
            issues.append("accepted_but_no_fills")
        if float(step_pnl) > 0 and float(self.equity) < float(equity_before):
            issues.append("pnl_positive_but_equity_down")
        if float(step_pnl) < 0 and float(self.equity) > float(equity_before):
            issues.append("pnl_negative_but_equity_up")

        # Build entry
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
            trades_this_step=[f for f in fills if f.get('realized_pnl', 0.0) != 0],
            execution_time_ms=processing_ms,
            issues=issues,
            accepted_details=accepted[:10],
            rejected_details=rejected[:10],
            fill_details=fills[:20],
        )

        # Log it
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
            "positions": {k: v.as_bus() if hasattr(v, 'as_bus') else v for k, v in self.positions.items()},
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
        
        # Restore trades
        trades = state.get("trades", [])
        if trades:
            self.trades = list(trades)
            self.logger.info(f"📂 Restored {len(self.trades)} trades from state")
        
        # Note: positions are synced from MT5 in live mode, so we don't restore them
