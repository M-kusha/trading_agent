

from __future__ import annotations

import datetime as dt
import math
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager
from modules.utils.lot_calculator import UnifiedLotCalculator

try:
    from config import get_trade_limits
    _TRADE_LIMITS = get_trade_limits()
except ImportError:
    _TRADE_LIMITS = {"max_trades_per_day": 20, "training_mode_limit": 9999}


from modules.position.smart_position_manager import (
    ExpertSignal,
    PositionAction,
    PositionFocusContext,
    PositionManagementSignal,
    SmartPositionManager,
)

from .adapters.base_adapter import BaseLiveAdapter, LiveAdapterConfig
from .adapters.mt5_adapter import MT5Adapter
from .debug.debugger import ExecutorDebugManager
from .shared.types import PositionSnap, TradeFill, _parse_timestamp
from .shared.utils import SafeBus, resolve_symbol
from .unified_logger import ExecutionCycleEntry, UnifiedExecutorLogger

try:
    from modules.position.exit_engine import get_exit_engine

    EXIT_ENGINE_AVAILABLE = True
except ImportError:
    get_exit_engine = None  # type: ignore
    EXIT_ENGINE_AVAILABLE = False


def _canon_direction(d: str) -> str:
    d = (d or "").lower().strip()
    if d in ("buy", "long", "bullish", "open_long"):
        return "buy"
    if d in ("sell", "short", "bearish", "open_short"):
        return "sell"
    return "hold"


def _direction_to_side(d: str) -> int:
    d = (d or "").lower().strip()
    if d in ("buy", "long", "bullish", "open_long", "scale_up"):
        return 1
    if d in ("sell", "short", "bearish", "open_short", "scale_down"):
        return -1
    return 0


@dataclass
class ExecutorConfig:
    execution_mode: str = "sim"
    live_broker: str = "mt5"
    symbol_overrides: Optional[Dict[str, str]] = None


    lot_step: float = 0.01
    min_lot: float = 0.01
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


    hold_positions_without_signal: bool = False


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


        self.bus = SafeBus(InfoBusManager.get_instance())
        self.logger = RotatingLogger("Executor", log_path="logs/executor/executor.log", operator_mode=True)


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

        if _cfg_ib is None:
            try:
                from pathlib import Path

                import yaml

                risk_policy = Path("config/risk_policy.yaml")
                if risk_policy.exists():
                    with open(risk_policy, "r", encoding="utf-8") as f:
                        rp = yaml.safe_load(f) or {}
                    _cfg_ib = rp.get("prop_firm", {}).get("account_size") or rp.get("lot_sizing", {}).get(
                        "account_balance"
                    )
            except Exception:
                pass

        self.initial_balance: float = float(100_000.0 if _cfg_ib is None else _cfg_ib)
        self.balance: float = float(self.initial_balance)
        self.equity: float = float(self.balance)
        self._last_equity: float = float(self.equity)
        self.positions: Dict[str, PositionSnap] = {}
        self.trades: List[Dict[str, Any]] = []
        self.closed_positions: List[Dict[str, Any]] = []
        self.step_idx: int = 0
        self._seen_ids: Set[str] = set()
        self._max_seen_ids: int = 5000
        self._cumulative_pnl: float = 0.0
        self._max_trades: int = 1000


        if not hasattr(self, "adapter"):
            self.adapter: Optional[BaseLiveAdapter] = None
        self._ensure_adapter()
        self._publish_adapter_status()


        dbg_cfg: Dict[str, Any] = {**(self.cfg.debug_config or {}), "enabled": bool(self.cfg.debug_enabled)}
        self.debugger = ExecutorDebugManager(self.bus, config=dbg_cfg)


        self.unified_logger = UnifiedExecutorLogger(self.logger)


        self.smart_position_manager = SmartPositionManager()


        self.lot_calculator = UnifiedLotCalculator.get_instance()
        self.lot_calculator.publish_lot_config_to_bus()


        self._publish_all(
            exec_fills=[],
            accepted=[],
            rejected=[],
            step_pnl=0.0,
            realized_step=0.0,
            unrealized=0.0,
            reason="startup",
        )


    def _initialize(self, **kwargs: Any) -> None:

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

            pass

        if hasattr(self, "debugger") and self.debugger:
            if self.cfg.debug_enabled:
                self.debugger.enable()
            else:
                self.debugger.disable()


    def _ensure_adapter(self, mode: Optional[str] = None) -> None:
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
                st.update(
                    {
                        "balance": float(ai.get("balance", 0.0) or 0.0),
                        "equity": float(ai.get("equity", 0.0) or 0.0),
                        "leverage": float(ai.get("leverage", 100.0) or 100.0),
                        "margin": float(ai.get("margin", 0.0) or 0.0),
                        "free_margin": float(ai.get("free_margin", 0.0) or 0.0),
                        "currency": str(ai.get("currency", "EUR")),
                    }
                )
            except Exception:
                pass
        return st

    def _publish_adapter_status(self) -> None:
        st = self._build_live_adapter_status()
        self.bus.set("live_adapter_status", st, thesis="executor live adapter status")

    def _resolve_mode(self) -> str:
        if not self.cfg.allow_runtime_switch:
            return self.cfg.execution_mode


        envc = self.bus.get("environment_config", "Executor", default={}) or {}
        env_mode = str(envc.get("mode", "")).lower()


        em = self.bus.get("execution_mode", "Executor", default=None)
        em_mode = str(em).lower() if isinstance(em, str) else None

        if env_mode == "live":
            target = "live"
        elif em_mode in ("sim", "live"):
            target = em_mode
        else:
            target = self.cfg.execution_mode


        if self.adapter and self.adapter.is_connected():
            target = "live"

        if target == "live":
            if not self.adapter or not self.adapter.is_connected():

                self._ensure_adapter(mode=target)
                if not (self.adapter and self.adapter.is_connected()):
                    return "sim"
        return target


    def _account_snapshot(self, pos_snap: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if pos_snap is None:
            mode = self._resolve_mode()
            if mode == "live" and self.adapter:
                pos_snap = self.adapter.sync_positions()
            else:
                pos_snap = {}
                for k, v in self.positions.items():
                    current_price = self._sim_price(k, v.side)

                    if (
                        current_price is not None
                        and current_price > 0
                        and v.entry_price > 0
                        and v.units > 0
                    ):
                        current_pnl = (current_price - v.entry_price) * v.side * v.units

                        if current_pnl > v.peak_unrealized:
                            v.peak_unrealized = current_pnl
                    pos_snap[k] = v.as_bus(last_price=current_price)

        return {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "positions": pos_snap,
            "positions_count": len(pos_snap or {}),
            "step": int(self.step_idx),
            "ts": time.time(),
        }

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        t0 = time.time()
        try:

            val = self.bus.get("step_idx", "Executor", default=None)
            if val is None:
                val = self.bus.get("step_idx", "PositionManager", default=None)
            if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
                self.step_idx = int(val)
            else:
                self.step_idx += 1

            mode = self._resolve_mode()
            self._publish_adapter_status()


            balance_before = float(self.balance)
            equity_before = float(self.equity)


            self.debugger.begin("collect_intents")
            accepted, rejected, q_count, dec_count = self._collect_intents()
            self.debugger.end("collect_intents")


            realized_step = 0.0
            unreal_after = 0.0
            positions_after: Dict[str, Any] = {}
            fills: List[Dict[str, Any]] = []

            if mode == "live":
                self.debugger.begin("execute_live")

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
                positions_after = {}
                for k, v in self.positions.items():
                    current_price = self._sim_price(k, v.side)
                    if (
                        current_price is not None
                        and current_price > 0
                        and v.entry_price > 0
                        and v.units > 0
                    ):
                        current_pnl = (current_price - v.entry_price) * v.side * v.units
                        if current_pnl > v.peak_unrealized:
                            v.peak_unrealized = current_pnl
                    positions_after[k] = v.as_bus(last_price=current_price)


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


        recent = self.trades[-50:] if self.trades else []
        order_data = {"accepted": accepted, "rejected": rejected, "step": int(self.step_idx)}
        execution_data = {"fills": fills, "step": int(self.step_idx)}
        initial_balance = float(getattr(self, "initial_balance", 0.0) or 0.0)
        balance = float(self.balance)
        equity = float(self.equity)
        drawdown = max(0.0, (initial_balance - balance) / max(initial_balance, 1e-9)) if initial_balance > 0 else 0.0
        total_closed = len(self.closed_positions)
        wins = 0
        if total_closed > 0:
            try:
                wins = sum(1 for p in self.closed_positions if float(p.get("pnl", p.get("profit", 0.0)) or 0.0) > 0.0)
            except Exception:
                wins = 0
        win_rate = (wins / total_closed) if total_closed > 0 else 0.5
        pnl_trend = float(step_pnl) / max(initial_balance * 0.01, 1e-9) if initial_balance > 0 else 0.0
        pnl_trend = float(max(-1.0, min(1.0, pnl_trend)))

        market_state = {
            "balance": balance,
            "equity": equity,
            "step": int(self.step_idx),
            "initial_balance": initial_balance,
            "drawdown": float(drawdown),
            "win_rate": float(win_rate),
            "pnl_trend": float(pnl_trend),
        }
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
            "live_adapter_status",
            "Executor",
            default={"provider": self.cfg.live_broker, "connected": False},
        )


        pos_list: List[Dict[str, Any]] = []
        try:
            for inst, p in (positions_after or {}).items():
                notional = float(p.get("notional_eur", 0.0) or 0.0)
                units = float(p.get("units", 0.0) or 0.0)
                entry_price = float(p.get("entry_price", 0.0) or 0.0)
                size = (
                    abs(notional)
                    if abs(notional) > 0
                    else (abs(units * entry_price) if (units and entry_price) else abs(units))
                )
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


        position_focus = PositionFocusContext.from_positions(positions_after)
        position_focus_dict = position_focus.to_dict()

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
            "closed_positions": list(self.closed_positions),
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),
            "pending_orders": order_data.get("accepted", []),
            "account_state": {
                "balance": float(self.balance),
                "equity": float(self.equity),
                "initial_balance": float(self.initial_balance),
                "step": int(self.step_idx),
            },
            "order_queue": [],
            "processing_time_ms": (time.time() - t0) * 1000.0,


            "position_focus_context": position_focus_dict,
        }


    def _collect_intents(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, int]:
        accepted: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []
        q_count = 0
        dec_count = 0


        memory_veto = False
        memory_veto_reasons: List[str] = []
        vetoed_instruments: List[str] = []
        try:
            memory_gate = self.bus.get("memory_gate", "Executor", default=None)
            if isinstance(memory_gate, dict):

                if memory_gate.get("veto", False):
                    memory_veto = True
                    memory_veto_reasons = memory_gate.get("reasons", ["Memory system vetoed"])
                    self.logger.warning(
                        format_operator_message(
                            icon="🧠",
                            message="MEMORY_VETO_ACTIVE",
                            reasons=memory_veto_reasons[:3],
                        )
                    )


                vetoed_inst_raw = memory_gate.get("vetoed_instruments", [])
                if isinstance(vetoed_inst_raw, list):
                    vetoed_instruments = [
                        str(inst).upper().replace("/", "_") for inst in vetoed_inst_raw
                    ]
                    if vetoed_instruments:
                        self.logger.warning(
                            format_operator_message(
                                icon="🧠",
                                message="MEMORY_INSTRUMENT_VETO",
                                instruments=vetoed_instruments,
                            )
                        )
        except Exception:
            pass


        risk_veto = False
        risk_veto_reason = ""
        try:
            risk_assessment = self.bus.get("risk_assessment", "Executor", default=None)
            risk_level = self.bus.get("risk_level", "Executor", default=None)


            if isinstance(risk_assessment, dict):
                if risk_assessment.get("emergency_active", False):
                    risk_veto = True
                    risk_veto_reason = "EMERGENCY_MODE_ACTIVE"
                    self.logger.warning(
                        format_operator_message(
                            icon="🚨",
                            message="RISK_EMERGENCY_MODE",
                            reason="DynamicRiskController in emergency mode - blocking new positions",
                        )
                    )


            if isinstance(risk_level, str) and risk_level.upper() == "CRITICAL":
                risk_veto = True
                risk_veto_reason = "RISK_LEVEL_CRITICAL"
                self.logger.warning(
                    format_operator_message(
                        icon="⛔",
                        message="RISK_LEVEL_CRITICAL",
                        reason="DynamicRiskController reports critical risk level",
                    )
                )

        except Exception:
            pass


        strategy_position_multiplier: float = 1.0

        strategy_max_trades_per_day: int = _TRADE_LIMITS.get("max_trades_per_day", 20)
        curriculum_stage: str = "Expert"
        bias_active: List[str] = []

        try:

            bias_adjustments = self.bus.get("bias_adjustments", "Executor", default=None)
            if isinstance(bias_adjustments, dict):
                strategy_position_multiplier = float(
                    bias_adjustments.get("position_size_multiplier", 1.0) or 1.0
                )
                strategy_position_multiplier = max(0.1, min(1.0, strategy_position_multiplier))


                if strategy_position_multiplier < 0.95:
                    bias_analysis = self.bus.get("bias_analysis", "Executor", default={}) or {}
                    ind_biases = bias_analysis.get("individual_biases", {})
                    if isinstance(ind_biases, dict):
                        bias_active = [
                            k
                            for k, v in ind_biases.items()
                            if isinstance(v, dict) and v.get("detected", False)
                        ]
                    self.logger.info(
                        format_operator_message(
                            icon="🧠",
                            message="BIAS_POSITION_ADJUSTMENT",
                            multiplier=f"{strategy_position_multiplier:.2f}",
                            active_biases=bias_active[:3] if bias_active else ["psychological"],
                        )
                    )


            learning_constraints = self.bus.get("learning_constraints", "Executor", default=None)
            curriculum_stage_data = self.bus.get("curriculum_stage", "Executor", default=None)


            is_live_mode = bool(self.adapter and self.adapter.is_connected())

            if isinstance(learning_constraints, dict) and is_live_mode:

                max_pos = float(learning_constraints.get("max_position_size", 1.0) or 1.0)
                if max_pos < 1.0:
                    strategy_position_multiplier = min(strategy_position_multiplier, max_pos)


                default_max_trades = _TRADE_LIMITS.get("max_trades_per_day", 20)
                strategy_max_trades_per_day = int(
                    learning_constraints.get("max_trades_per_day", default_max_trades) or default_max_trades
                )
            elif not is_live_mode:

                strategy_max_trades_per_day = 1000

            if isinstance(curriculum_stage_data, dict):
                curriculum_stage = str(curriculum_stage_data.get("name", "Expert") or "Expert")

                if curriculum_stage in ("Foundation", "Basic") and is_live_mode:
                    self.logger.debug(
                        format_operator_message(
                            icon="📚",
                            message="CURRICULUM_CONSTRAINTS_ACTIVE",
                            stage=curriculum_stage,
                            max_position_multiplier=f"{strategy_position_multiplier:.2f}",
                            max_trades=strategy_max_trades_per_day,
                        )
                    )
        except Exception:
            pass


        trades_today = 0
        try:
            if self.trades:
                today_start = dt.datetime.now().replace(
                    hour=0, minute=0, second=0, microsecond=0
                )

                def _is_new_entry(t: dict) -> bool:
                    action = str(t.get("action", "")).lower()
                    comment = str(t.get("comment", "")).lower()


                    if any(x in action for x in ["scale", "close", "exit", "reverse"]):
                        return False
                    if any(x in action for x in ["open", "long", "short"]):
                        return True
                    if comment == "open":
                        return True
                    return False

                trades_today = sum(
                    1
                    for t in self.trades[-200:]
                    if isinstance(t, dict)
                    and t.get("ts", 0) >= today_start.timestamp()
                    and _is_new_entry(t)
                )
        except Exception:
            pass


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

                trading_mode_risk_multiplier = float(
                    mode_config.get("risk_multiplier", 1.0) or 1.0
                )
                trading_mode_risk_multiplier = max(0.25, min(4.0, trading_mode_risk_multiplier))

                trading_mode_stop_loss_multiplier = float(
                    mode_config.get("stop_loss_multiplier", 1.0) or 1.0
                )
                trading_mode_stop_loss_multiplier = max(
                    0.5, min(2.0, trading_mode_stop_loss_multiplier)
                )


            if trading_mode_name != "normal" or abs(trading_mode_position_scale - 1.0) > 1e-6:
                eff_value = None
                if mode_effectiveness is not None:
                    try:
                        eff_value = (
                            float(mode_effectiveness)
                            if not isinstance(mode_effectiveness, dict)
                            else float(
                                mode_effectiveness.get(
                                    "value", mode_effectiveness.get("effectiveness", 0.5)
                                )
                            )
                        )
                    except (TypeError, ValueError):
                        eff_value = None

                self.logger.info(
                    format_operator_message(
                        icon="⚙️",
                        message="TRADING_MODE_CONSTRAINTS",
                        mode=trading_mode_name.upper(),
                        position_scale=f"{trading_mode_position_scale:.2f}",
                        max_exposure=f"{trading_mode_max_exposure:.2f}",
                        risk_multiplier=f"{trading_mode_risk_multiplier:.2f}",
                        effectiveness=f"{eff_value:.2f}" if eff_value is not None else "N/A",
                    )
                )
        except Exception:
            pass

        def _apply_trading_mode_sizing(intent: Dict[str, Any]) -> Dict[str, Any]:
            if 0.99 <= trading_mode_position_scale <= 1.01:
                return intent

            if intent.get("size_eur"):
                original_size = float(intent["size_eur"])
                intent["size_eur"] = original_size * trading_mode_position_scale
                intent["_trading_mode_sizing"] = {
                    "original_size": original_size,
                    "position_scale": trading_mode_position_scale,
                    "mode": trading_mode_name,
                    "max_exposure": trading_mode_max_exposure,
                    "risk_multiplier": trading_mode_risk_multiplier,
                }

            if "units" in intent and intent.get("units"):
                original_units = float(intent["units"])
                intent["units"] = original_units * trading_mode_position_scale

            return intent


        def _is_instrument_vetoed(inst: str) -> bool:
            if memory_veto:
                return True
            if risk_veto:
                return True
            normalized = str(inst).upper().replace("/", "_")
            return normalized in vetoed_instruments

        def _get_veto_reason() -> Tuple[str, List[str]]:
            if memory_veto:
                return "memory_veto", memory_veto_reasons
            if risk_veto:
                return "risk_veto", [risk_veto_reason]
            return "instrument_veto", ["Instrument on veto list"]

        def _is_risk_blocked(inst: str, action: str) -> bool:
            return False

        def _exceeds_curriculum_trade_limit() -> bool:
            return trades_today >= strategy_max_trades_per_day

        def _apply_strategy_sizing(intent: Dict[str, Any]) -> Dict[str, Any]:
            if strategy_position_multiplier >= 0.99:
                return intent

            if intent.get("size_eur"):
                original_size = float(intent["size_eur"])
                intent["size_eur"] = original_size * strategy_position_multiplier
                intent["_strategy_sizing"] = {
                    "original_size": original_size,
                    "multiplier": strategy_position_multiplier,
                    "curriculum_stage": curriculum_stage,
                    "active_biases": bias_active,
                }

            if "units" in intent and intent.get("units"):
                original_units = float(intent["units"])
                intent["units"] = original_units * strategy_position_multiplier

            return intent


        oq = self.bus.get("order_queue", "Executor", default=[]) or []
        if isinstance(oq, list):
            q_count = len(oq)
            for item in oq[: self.cfg.max_orders_per_step]:
                intent = self._normalize_order_item(item)
                if not intent:
                    rejected.append({"reason": "bad_order_queue_item", "raw": item})

                elif intent.get("action", "").lower() in (
                    "open_long",
                    "open_short",
                    "buy",
                    "sell",
                    "scale_up",
                ):
                    inst = intent.get("instrument", "")
                    action = intent.get("action", "")

                    if _is_instrument_vetoed(inst):
                        veto_reason, veto_reasons = _get_veto_reason()
                        rejected.append(
                            {
                                "reason": veto_reason,
                                "intent": intent,
                                "veto_reasons": veto_reasons,
                                "vetoed_instrument": inst,
                            }
                        )
                        continue

                    if _exceeds_curriculum_trade_limit():
                        rejected.append(
                            {
                                "reason": "curriculum_trade_limit",
                                "intent": intent,
                                "message": f"Daily trade limit ({strategy_max_trades_per_day}) "
                                f"reached for curriculum stage '{curriculum_stage}'",
                                "trades_today": trades_today,
                                "curriculum_stage": curriculum_stage,
                            }
                        )
                        self.logger.info(
                            format_operator_message(
                                icon="📚",
                                message="ORDER_BLOCKED_CURRICULUM_LIMIT",
                                trades_today=trades_today,
                                max_trades=strategy_max_trades_per_day,
                                stage=curriculum_stage,
                            )
                        )
                        continue

                    if _is_risk_blocked(inst, action):
                        rejected.append(
                            {
                                "reason": "risk_limit_exceeded",
                                "intent": intent,
                                "blocked_instrument": inst,
                                "message": f"{inst} position exceeds portfolio risk limit - cannot increase exposure",
                            }
                        )
                        self.logger.warning(
                            format_operator_message(
                                icon="🛑",
                                message="ORDER_BLOCKED_RISK_LIMIT",
                                instrument=inst,
                                action=action,
                                reason="Position size exceeds limit",
                            )
                        )
                        continue

                    intent = _apply_strategy_sizing(intent)
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


        if self.cfg.read_position_decisions:
            env_cfg = (
                self.bus.get("environment_config", "Executor", default=None)
                or self.bus.get("environment_config", "PositionManager", default={})
                or {}
            )
            instruments = env_cfg.get("instruments") or []


            if not instruments:
                instruments = ["XAUUSD"]


            seen_normalized = set()
            unique_instruments = []
            for inst in instruments:
                normalized = inst.replace("/", "").replace("_", "").upper()
                if normalized not in seen_normalized:
                    seen_normalized.add(normalized)
                    unique_instruments.append(inst)
            instruments = unique_instruments


            self.logger.debug(f"[EXEC] Checking position_decisions for instruments: {instruments}")

            for inst in instruments:
                core = inst.replace("/", "").replace("_", "")


                node = self.bus.get(f"position_decision_{inst}", "Executor", default=None)
                self.logger.debug(f"[EXEC] position_decision_{inst} (Executor): {type(node).__name__} = {node}")

                if not (isinstance(node, dict) and node.get("decision")):
                    node = self.bus.get(
                        f"position_decision_{inst}", "PositionManager", default=None
                    )
                    self.logger.debug(f"[EXEC] position_decision_{inst} (PositionManager): {type(node).__name__} = {node}")

                if not (isinstance(node, dict) and node.get("decision")):
                    node = self.bus.get(
                        f"position_decision_{core}", "PositionManager", default=None
                    )
                    self.logger.debug(f"[EXEC] position_decision_{core} (PositionManager): {type(node).__name__} = {node}")

                if not (isinstance(node, dict) and node.get("decision")):
                    node = self.bus.get(
                        f"position_decision_{core}", "Executor", default=None
                    )
                    self.logger.debug(f"[EXEC] position_decision_{core} (Executor): {type(node).__name__} = {node}")

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


        try:
            self.logger.debug(
                f"[EXEC] Collected: queue={q_count} decisions={dec_count} "
                f"accepted={len(accepted)} rejected={len(rejected)}"
            )
        except Exception:
            pass


        if len(self._seen_ids) > self._max_seen_ids:
            ids_list = list(self._seen_ids)
            self._seen_ids = set(ids_list[-(self._max_seen_ids // 2) :])

        return accepted, rejected, q_count, dec_count

    def _prune_order_queue(self) -> None:
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
            out: Dict[str, Any] = {
                "id": oid,
                "instrument": inst,
                "action": action,
                "confidence": conf,
                "intensity": inten,
            }
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

        if act in ("close", "emergency_close", "scale_down"):
            return True
        if float(intent.get("confidence", 0.0)) < self.cfg.min_confidence:
            return False
        if abs(float(intent.get("intensity", 0.0))) < self.cfg.min_intensity and act not in (
            "close",
            "emergency_close",
        ):
            return False
        return True

    def _filter_reason(self, intent: Dict[str, Any]) -> str:
        act = str(intent.get("action", "")).lower()
        if act == "hold" and self.cfg.ignore_hold:
            return "ignore_hold"
        if float(intent.get("confidence", 0.0)) < self.cfg.min_confidence:
            return f"low_confidence<{self.cfg.min_confidence}>"
        if abs(float(intent.get("intensity", 0.0))) < self.cfg.min_intensity and act not in (
            "close",
            "emergency_close",
        ):
            return f"low_intensity<{self.cfg.min_intensity}>"
        if intent.get("id") in self._seen_ids:
            return "duplicate_id"
        return "filtered"


    def _get_contract_size(self, symbol: str) -> float:
        sym_upper = (symbol or "").upper().replace("_", "").replace("/", "")

        if "XAU" in sym_upper or "GOLD" in sym_upper:
            return 100.0
        if "XAG" in sym_upper or "SILVER" in sym_upper:
            return 5000.0
        if "BTC" in sym_upper:
            return 1.0
        if "ETH" in sym_upper:
            return 1.0


        return float(self.cfg.contract_size)

    def _get_current_volatility(self, symbol: str) -> Optional[float]:
        try:
            norm_symbol = self._normalize_symbol(symbol)


            vol_map = self.bus.get("volatility_by_instrument", "Executor", default=None)
            if isinstance(vol_map, dict):
                for key, val in vol_map.items():
                    if self._normalize_symbol(key) == norm_symbol:
                        if isinstance(val, (int, float)):
                            return float(val)


            mc = self.bus.get("market_conditions", "Executor", default=None)
            if isinstance(mc, dict):
                vol = mc.get("volatility")
                if isinstance(vol, (int, float)):
                    return float(vol)


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


    def _check_sim_exits(self) -> Tuple[List[Dict[str, Any]], float]:
        fills: List[Dict[str, Any]] = []
        realized = 0.0


        try:
            spm_cfg = self.smart_position_manager.config
            hard_stop_eur = float(spm_cfg.hard_stop_loss_eur)  # type: ignore[attr-defined]
            trailing_activation = float(spm_cfg.profit_take_activation_eur)  # type: ignore[attr-defined]
            trailing_pct = float(spm_cfg.profit_take_trail_pct)  # type: ignore[attr-defined]
            time_decay_hours = float(spm_cfg.time_decay_hours)  # type: ignore[attr-defined]
            time_decay_stop = float(spm_cfg.time_decay_stop_eur)  # type: ignore[attr-defined]
        except Exception:

            hard_stop_eur = 150.0
            trailing_activation = 100.0
            trailing_pct = 0.30
            time_decay_hours = 4.0
            time_decay_stop = 60.0

        positions_to_close: List[Tuple[str, str, float]] = []

        for inst, pos in self.positions.items():
            price = self._sim_price(inst, pos.side)
            if price is None:
                continue

            unrealized_pnl = (price - pos.entry_price) * pos.side * pos.units

            if unrealized_pnl > pos.peak_unrealized:
                pos.peak_unrealized = unrealized_pnl


            open_ts = _parse_timestamp(pos.open_time)
            age_hours = (time.time() - open_ts) / 3600.0 if open_ts else 0.0

            exit_reason = None


            if unrealized_pnl <= -hard_stop_eur:
                exit_reason = f"HARD_STOP: Loss €{unrealized_pnl:.2f} exceeds -€{hard_stop_eur:.0f}"

            elif pos.peak_unrealized >= trailing_activation:
                retrace = (
                    (pos.peak_unrealized - unrealized_pnl) / pos.peak_unrealized
                    if pos.peak_unrealized > 0
                    else 0
                )
                if retrace >= trailing_pct:
                    exit_reason = (
                        f"TRAILING_PROFIT: Retraced {retrace*100:.1f}% from peak "
                        f"€{pos.peak_unrealized:.2f}"
                    )

            elif age_hours >= time_decay_hours and unrealized_pnl <= -time_decay_stop:
                exit_reason = (
                    f"TIME_DECAY: Position {age_hours:.1f}h old with loss €{unrealized_pnl:.2f}"
                )

            if exit_reason:
                positions_to_close.append((inst, exit_reason, unrealized_pnl))

        for inst, reason, pnl in positions_to_close:
            pos = self.positions.get(inst)
            if not pos:
                continue

            price = self._sim_price(inst, pos.side)
            if price is None:
                continue

            commission = 0.0
            if getattr(self.cfg, "commission_per_million", 0.0):
                notional = pos.units * price
                commission = (
                    abs(notional) / 1_000_000.0
                ) * float(self.cfg.commission_per_million)

            realized_pnl = (price - pos.entry_price) * pos.side * pos.units - commission
            realized += realized_pnl

            fill = TradeFill(
                id=f"fill-{uuid.uuid4().hex[:10]}",
                ts=time.time(),
                step=self.step_idx,
                instrument=inst,
                action="exit:auto",
                side=-pos.side,
                units=pos.units,
                price=price,
                notional_eur=pos.units * price,
                realized_pnl=realized_pnl,
                origin_id="sim_exit_check",
                comment=reason,
            ).as_bus()
            self.trades.append(fill)
            fills.append(fill)

            self._track_closed_position(pos, price, realized_pnl, "auto_exit")
            del self.positions[inst]

            try:
                self.logger.info(
                    f"[SIM] 🛑 AUTO EXIT: {inst} | {reason} | Realized: €{realized_pnl:.2f}"
                )
            except Exception:
                pass

        return fills, realized

    def _sim_price(self, inst: str, side: int) -> Optional[float]:
        sp = (
            self.bus.get("prices", "Executor", default=None)
            or self.bus.get("prices", "PositionManager", default={})
            or {}
        )
        pd = (
            self.bus.get("price_data", "Executor", default=None)
            or self.bus.get("price_data", "PositionManager", default={})
            or {}
        )
        px: Optional[float] = None
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


        direction = 0
        if side > 0:
            direction = 1
        elif side < 0:
            direction = -1

        if self.cfg.default_spread and direction:
            px += (self.cfg.default_spread / 2.0) * direction
        if self.cfg.slippage_pts and direction:
            px += self.cfg.slippage_pts * direction
        return px

    def _units_from(self, size_eur: float, units: float, price: float) -> float:
        if units and units > 0:
            return float(units)
        if size_eur and size_eur > 0 and price > 0:
            return float(size_eur / price)
        return 0.0

    def _get_decision_context(self, instrument: str) -> Dict[str, Any]:
        context = {
            "ppo_direction": None,
            "expert_direction": None,
            "ppo_confidence": None,
            "was_ppo_led": False,
        }

        try:

            ppo_decision = self.bus.get("ppo_final_decision", "Executor", default={}) or {}
            if isinstance(ppo_decision, dict):
                ppo_dir = ppo_decision.get("direction", "flat")
                context["ppo_direction"] = (
                    ppo_dir if ppo_dir in ("long", "short", "flat") else "flat"
                )
                context["ppo_confidence"] = float(
                    ppo_decision.get("confidence", 0.0) or 0.0
                )


            autonomy_state = self.bus.get(
                "ppo_autonomy_state", "Executor", default={}
            ) or {}
            if isinstance(autonomy_state, dict):


                context["was_ppo_led"] = True


            committee = self.bus.get("committee_decision", "Executor", default={}) or {}
            if isinstance(committee, dict):
                committee_action = str(committee.get("action", "hold")).lower()
                if committee_action in ("long", "buy", "bullish"):
                    context["expert_direction"] = "long"
                elif committee_action in ("short", "sell", "bearish"):
                    context["expert_direction"] = "short"
                else:
                    context["expert_direction"] = "flat"
        except Exception:
            pass

        return context

    def _track_closed_position(
        self,
        position: "PositionSnap",
        close_price: float,
        realized_pnl: float,
        close_reason: str,
    ) -> None:
        closed_record = {
            "instrument": position.instrument,
            "side": position.side,
            "units": position.units,
            "entry_price": position.entry_price,
            "close_price": close_price,
            "pnl": realized_pnl,
            "profit": realized_pnl,
            "close_reason": close_reason,
            "close_step": self.step_idx,
            "entry_step": getattr(position, "entry_step", 0),
            "close_time": time.time(),
            "open_time": getattr(position, "open_time", 0.0),
        }
        self.closed_positions.append(closed_record)
        if len(self.closed_positions) > 500:
            self.closed_positions = self.closed_positions[-500:]


        self._update_memory_on_trade_close(closed_record)


        try:
            if EXIT_ENGINE_AVAILABLE and get_exit_engine:
                exit_engine = get_exit_engine()
                exit_engine.reset_peak(position.instrument)
                self.logger.debug(
                    f"[Executor] Reset exit peak for {position.instrument} after close"
                )
        except Exception as e:
            self.logger.debug(f"[Executor] Failed to reset exit peak: {e}")


        self._notify_ppo_autonomy(position, realized_pnl)

    def _notify_ppo_autonomy(self, position: "PositionSnap", realized_pnl: float) -> None:
        try:
            ppo_direction = getattr(position, "ppo_direction", None)
            expert_direction = getattr(position, "expert_direction", None)
            ppo_confidence = getattr(position, "ppo_confidence", None)
            was_ppo_led = getattr(position, "was_ppo_led", False)

            if ppo_direction is None and expert_direction is None:
                return

            trade_outcome = {
                "instrument": position.instrument,
                "ppo_direction": ppo_direction or "flat",
                "expert_direction": expert_direction or "flat",
                "pnl": realized_pnl,
                "ppo_confidence": ppo_confidence or 0.0,
                "was_ppo_led": bool(was_ppo_led),
                "timestamp": time.time(),
            }

            self.bus.set(
                "trade_outcome_for_autonomy",
                trade_outcome,
                module="Executor",
                thesis=(
                    f"Trade closed: {position.instrument} "
                    f"PnL={realized_pnl:.2f}, PPO={ppo_direction}, Expert={expert_direction}"
                ),
            )

        except Exception as e:
            self.logger.debug(f"Failed to notify PPO autonomy: {e}")

    def _update_memory_on_trade_close(self, closed_record: Dict[str, Any]) -> None:
        try:


            self.bus.set(
                "latest_closed_trade",
                closed_record,
                module="Executor",
                thesis=f"Trade closed: {closed_record.get('instrument')} PnL={closed_record.get('pnl', 0):.2f}"
            )

            self.logger.debug(
                f"[Executor] Published closed trade for memory: {closed_record.get('instrument')} "
                f"PnL={closed_record.get('pnl', 0):.2f}"
            )

        except Exception as e:

            self.logger.debug(f"[Executor] Trade publish failed (non-fatal): {e}")

    def _execute_sim(
        self,
        intents: List[Dict[str, Any]],
        *,
        want_breakdown: bool = False,
    ) -> Tuple[List[Dict[str, Any]], float, float, float]:
        fills: List[Dict[str, Any]] = []
        realized_step = 0.0


        exit_fills, exit_realized = self._check_sim_exits()
        fills.extend(exit_fills)
        realized_step += exit_realized

        for intent in intents:
            inst = intent["instrument"]
            action = str(intent["action"]).lower()
            side_from_action = {
                "open_long": +1,
                "scale_up": +1,
                "open_short": -1,
                "scale_down": -1,
            }.get(action, 0)

            price = self._sim_price(inst, side_from_action)
            if price is None:
                self.debugger.record_error(f"sim_price_unavailable:{inst}")
                continue

            size_eur = float(intent.get("size_eur", 0.0) or 0.0)
            units = float(intent.get("units", 0.0) or 0.0)
            add_units = self._units_from(size_eur, units, price)
            origin_id = intent.get("id", "")

            def commission(notional: float) -> float:
                cpm = float(getattr(self.cfg, "commission_per_million", 0.0) or 0.0)
                return (abs(notional) / 1_000_000.0) * cpm if cpm > 0 else 0.0

            if action in ("open_long", "open_short"):
                if inst in self.positions and self.positions[inst].side != side_from_action:
                    p = self.positions[inst]
                    realized = (
                        (price - p.entry_price) * p.side * p.units
                        - commission(p.units * price)
                    )
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
                    ctx = self._get_decision_context(inst)
                    self.positions[inst] = PositionSnap(
                        inst,
                        side_from_action,
                        add_units,
                        price,
                        notional_eur=notional,
                        open_time=time.time(),
                        entry_step=self.step_idx,
                        ppo_direction=ctx["ppo_direction"],
                        expert_direction=ctx["expert_direction"],
                        ppo_confidence=ctx["ppo_confidence"],
                        was_ppo_led=ctx["was_ppo_led"],
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
                    ctx = self._get_decision_context(inst)
                    self.positions[inst] = PositionSnap(
                        inst,
                        +1,
                        add_units,
                        price,
                        notional_eur=notional,
                        open_time=time.time(),
                        entry_step=self.step_idx,
                        ppo_direction=ctx["ppo_direction"],
                        expert_direction=ctx["expert_direction"],
                        ppo_confidence=ctx["ppo_confidence"],
                        was_ppo_led=ctx["was_ppo_led"],
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
                        realized = (
                            (price - p.entry_price) * p.side * reduce_u
                            - commission(reduce_u * price)
                        )
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
                        p.entry_price = (
                            p.entry_price * p.units + price * add_units
                        ) / new_u
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
                reduce_u = min(add_units, p.units)
                if reduce_u <= 0:
                    continue
                realized = (
                    (price - p.entry_price) * p.side * reduce_u
                    - commission(reduce_u * price)
                )
                realized_step += realized
                p.units -= reduce_u
                p.notional_eur -= reduce_u * p.entry_price
                if p.units <= 1e-12:
                    self._track_closed_position(p, price, realized, "scale_down")
                    del self.positions[inst]
                trade_side = -1 if p.side > 0 else +1
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
                    realized = (
                        (price - p.entry_price) * p.side * p.units
                        - commission(p.units * price)
                    )
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


        self.balance += realized_step


        unreal = 0.0
        for inst, p in self.positions.items():
            px = self._sim_price(inst, p.side)
            if px is None:
                continue
            unreal += (px - p.entry_price) * p.side * p.units

        equity_now = float(self.balance + unreal)
        step_pnl = float(equity_now - self._last_equity)

        self.equity = equity_now
        self._last_equity = equity_now

        if len(self.trades) > self._max_trades:
            self.trades = self.trades[-self._max_trades :]

        if want_breakdown:
            return fills, step_pnl, float(realized_step), float(unreal)
        return fills, step_pnl, 0.0, 0.0


    def _execute_live_smart(self, intents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], float]:
        fills: List[Dict[str, Any]] = []
        if not self.adapter or not self.adapter.is_connected():
            return fills, 0.0

        prop_limits = self.lot_calculator.check_prop_firm_limits()


        if prop_limits.get("must_close_all", False):
            self.logger.critical(
                f"[SMART] 🚨🚨🚨 EMERGENCY CLOSE ALL! {prop_limits.get('warnings', [])}"
            )

            try:
                import MetaTrader5 as mt5
                positions = mt5.positions_get()  # type: ignore[attr-defined]
                if positions:
                    for pos in positions:
                        self._emergency_close_position(pos)
            except Exception as e:
                self.logger.error(f"[SMART] Emergency close failed: {e}")
            return fills, 0.0

        if not prop_limits.get("can_trade", True):
            self.logger.warning(
                f"[SMART] 🚫 PROP FIRM BLOCK: Trading halted - {prop_limits.get('warnings', [])}"
            )
            intents = [
                i
                for i in intents
                if str(i.get("action", "")).lower()
                in ("close", "close_all", "scale_down")
            ]
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


        try:
            ppo_multi = self.bus.get("ppo_multi_decision", "Executor", default=None)
            if isinstance(ppo_multi, dict):
                instruments_data = ppo_multi.get("instruments", {})
                for inst, dec in instruments_data.items():
                    if isinstance(dec, dict):
                        direction = dec.get("direction", "flat")
                        dir_score = dec.get("direction_score", 0.0)
                        gate_passed = dec.get("gate_passed", False)
                        action_intent = (dec.get("meta") or {}).get("action_intent", "unknown")


                        if gate_passed or direction == "short" or dir_score < -0.2:
                            self.logger.info(
                                f"[PPO→EXEC] {inst}: direction={direction} dir_score={dir_score:.3f} "
                                f"gate_passed={gate_passed} action_intent={action_intent}"
                            )
        except Exception:
            pass


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
                                self.logger.info(
                                    f"[SMART] ✅ Closed hedge ticket {ticket} on {symbol}"
                                )
                                fill = TradeFill(
                                    id=f"fill-{uuid.uuid4().hex[:10]}",
                                    ts=time.time(),
                                    step=self.step_idx,
                                    instrument=symbol,
                                    action="hedge_cleanup",
                                    side=0,
                                    units=float(pos.get("volume", 0.0) or 0.0),
                                    price=float(result.get("price", 0) or 0),
                                    notional_eur=0.0,
                                    realized_pnl=float(pos.get("profit", 0.0) or 0.0),
                                    origin_id=f"hedge-{ticket}",
                                    comment="hedge_cleanup",
                                ).as_bus()
                                self.trades.append(fill)
                                fills.append(fill)
                                self.smart_position_manager.record_trade(symbol)
                                closed_count += 1
                            else:
                                error = result.get("error", "unknown")
                                self.logger.error(
                                    f"[SMART] ❌ Failed to close hedge ticket {ticket} on {symbol}: {error}"
                                )
                                failed_count += 1
                        except Exception as close_err:
                            self.logger.error(
                                f"[SMART] ❌ Exception closing ticket {ticket}: {close_err}"
                            )
                            failed_count += 1
                    else:
                        self.logger.warning(
                            f"[SMART] ⚠️ Invalid position data - ticket={ticket}, symbol={symbol}"
                        )
                        failed_count += 1

                if closed_count > 0 or failed_count > 0:
                    self.logger.info(
                        f"[SMART] Hedge cleanup: {closed_count} closed, {failed_count} failed"
                    )
        except Exception as e:
            self.logger.error(f"[SMART] Hedge cleanup failed: {e}")


        smart_positions = self.smart_position_manager.get_all_positions()

        for raw_symbol, position in smart_positions.items():
            symbol = raw_symbol
            canonical_symbol = self._normalize_symbol(symbol)

            mgmt_signal = self._build_position_management_signal(
                symbol=symbol,
                canonical_symbol=canonical_symbol,
                position=position,
                intents=intents,
            )

            if mgmt_signal.expert_signals:
                decision = self.smart_position_manager.manage_position(mgmt_signal)
            else:
                signal_direction = 0
                signal_strength = 0.0
                matched_intent = False

                for intent in intents:
                    inst_intent = self._normalize_symbol(intent.get("instrument", ""))
                    if inst_intent == canonical_symbol:
                        action = str(intent.get("action", "")).lower()
                        if action in ("open_long", "scale_up"):
                            signal_direction = 1
                        elif action in ("open_short",):
                            signal_direction = -1
                        intent_strength = float(
                            intent.get("intensity", intent.get("confidence", 0.5)) or 0.5
                        )

                        ppo_pos = self.bus.get("ppo_position_size", "Executor", default=None)
                        if ppo_pos is not None:
                            try:
                                ppo_str = float(ppo_pos)
                                signal_strength = max(intent_strength, ppo_str) if ppo_str > 0 else intent_strength
                            except (TypeError, ValueError):
                                signal_strength = intent_strength
                        else:
                            signal_strength = intent_strength
                        matched_intent = True
                        break

                if (not matched_intent) and bool(getattr(self.cfg, "hold_positions_without_signal", False)):
                    decision = self.smart_position_manager.manage_position(mgmt_signal)
                else:
                    decision = self.smart_position_manager.decide(
                        symbol=symbol,
                        signal_direction=signal_direction,
                        signal_strength=signal_strength,
                        consensus_confidence=signal_strength,
                    )


            if decision.action == PositionAction.CLOSE:
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
                    fills.append(
                        {
                            "action": decision.action.value.lower(),
                            "symbol": symbol,
                            "reasons": decision.reasons,
                            "ok": True,
                        }
                    )
                    self.smart_position_manager.record_trade(symbol)


            elif decision.action == PositionAction.REVERSE:
                self.logger.info(
                    format_operator_message(
                        "🔁",
                        "SMART_REVERSE",
                        symbol=symbol,
                        side=("BUY" if decision.side > 0 else "SELL"),
                        reasons=decision.reasons[:2],
                    )
                )

                close_result = self.adapter.close_position(symbol)
                if not close_result.get("ok"):
                    self.logger.warning(
                        format_operator_message(
                            "⚠️",
                            "SMART_REVERSE_CLOSE_FAILED",
                            symbol=symbol,
                            error=close_result.get("error", "unknown"),
                        )
                    )
                    continue

                fills.append(
                    {
                        "action": "reverse_close",
                        "symbol": symbol,
                        "reasons": decision.reasons,
                        "ok": True,
                    }
                )
                self.smart_position_manager.record_trade(symbol)


                lots = 0.0
                try:
                    volatility = self._get_current_volatility(symbol)
                    lots, _ = self.lot_calculator.calculate_lots(
                        symbol=symbol,
                        signal_strength=float(max(0.1, decision.confidence)),
                        volatility=volatility,
                    )
                except Exception:
                    lots = float(decision.lots or self.adapter.cfg.min_lot)

                lots = max(float(lots), float(self.adapter.cfg.min_lot))

                open_result = self.adapter.market_order(symbol, decision.side, lots)
                if open_result.get("ok"):
                    px = float(open_result.get("price", 0) or 0)
                    contract_size = self._get_contract_size(symbol)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=symbol,
                        action="reverse",
                        side=decision.side,
                        units=lots * contract_size,
                        price=px,
                        notional_eur=lots * contract_size * px,
                        realized_pnl=0.0,
                        origin_id="smart_reverse",
                        comment="; ".join(decision.reasons[:2]),
                    ).as_bus()
                    self.trades.append(fill)
                    fills.append(fill)
                    self.smart_position_manager.record_trade(symbol)
                else:
                    self.logger.warning(
                        format_operator_message(
                            "⚠️",
                            "SMART_REVERSE_OPEN_FAILED",
                            symbol=symbol,
                            lots=f"{lots:.2f}",
                            error=open_result.get("error", "unknown"),
                        )
                    )


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
                close_side = -decision.side

                ticket = getattr(position, "ticket", 0)
                if ticket:

                    result = self._partial_close_by_ticket(ticket, symbol, decision.lots)
                else:

                    self.logger.warning(f"[SCALE_DOWN] No ticket for {symbol}, falling back to market_order")
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


            elif decision.action in (
                PositionAction.ADJUST_SL,
                PositionAction.ADJUST_TP,
                PositionAction.TIGHTEN_PROTECTION,
            ):
                ticket = getattr(position, "ticket", 0)
                if ticket and decision.new_sl is not None:
                    try:
                        result = self.adapter.modify_position(
                            ticket=ticket,
                            sl=decision.new_sl,
                            tp=decision.new_tp if decision.new_tp else None,
                        )
                        if result.get("ok"):
                            self.logger.info(
                                format_operator_message(
                                    "🔧",
                                    "POSITION_ADJUSTED",
                                    action=decision.action.value,
                                    symbol=symbol,
                                    new_sl=(
                                        f"{decision.new_sl:.5f}"
                                        if decision.new_sl
                                        else "unchanged"
                                    ),
                                    new_tp=(
                                        f"{decision.new_tp:.5f}"
                                        if decision.new_tp
                                        else "unchanged"
                                    ),
                                    expert_support=f"{decision.expert_support_ratio:.0%}",
                                    reasons=decision.reasons[:2],
                                )
                            )
                            fills.append(
                                {
                                    "action": decision.action.value.lower(),
                                    "symbol": symbol,
                                    "new_sl": decision.new_sl,
                                    "new_tp": decision.new_tp,
                                    "expert_support": decision.expert_support_ratio,
                                    "ok": True,
                                }
                            )
                        else:
                            self.logger.warning(
                                f"[SMART] Failed to adjust {symbol}: {result.get('error', 'unknown')}"
                            )
                    except Exception as e:
                        self.logger.warning(f"[SMART] Position adjustment failed: {e}")
                else:

                    if not ticket:
                        self.logger.warning(
                            f"[SMART] ⚠️ Cannot adjust {symbol} SL/TP: no ticket found on position"
                        )
                    elif decision.new_sl is None:
                        self.logger.warning(
                            f"[SMART] ⚠️ Cannot adjust {symbol}: new_sl is None"
                        )


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


            if action in ("close", "emergency_close"):
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
                    fills.append(
                        {
                            "action": action,
                            "symbol": close_symbol,
                            "source": "order_queue",
                            "realized_pnl": 0.0,
                            "ok": True,
                        }
                    )
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

            signal_direction = {
                "open_long": 1,
                "scale_up": 1,
                "open_short": -1,
                "scale_down": -1,
            }.get(action, 0)


            intent_strength = float(
                intent.get("intensity", intent.get("confidence", 0.5)) or 0.5
            )


            ppo_position_size = self.bus.get("ppo_position_size", "Executor", default=None)
            if ppo_position_size is not None:
                try:
                    ppo_strength = float(ppo_position_size)
                    if ppo_strength > 0.0:

                        signal_strength = max(intent_strength, ppo_strength)
                    else:
                        signal_strength = intent_strength
                except (TypeError, ValueError):
                    signal_strength = intent_strength
            else:
                signal_strength = intent_strength

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


            if decision.action in (PositionAction.OPEN_LONG, PositionAction.OPEN_SHORT):
                try:
                    pos_snap = adapter_positions_norm.get(inst_norm)
                    if pos_snap:
                        existing_side = self._position_side_from_snapshot(pos_snap)
                        if existing_side != 0:
                            is_buy = existing_side > 0
                            if (existing_side > 0 and decision.side < 0) or (
                                existing_side < 0 and decision.side > 0
                            ):
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
                            if (existing_side > 0 and decision.side > 0) or (
                                existing_side < 0 and decision.side < 0
                            ):
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

                volatility = self._get_current_volatility(exec_symbol)
                lots, lot_details = self.lot_calculator.calculate_lots(
                    symbol=exec_symbol,
                    signal_strength=signal_strength,
                    volatility=volatility,
                )

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

        if len(self.trades) > self._max_trades:
            self.trades = self.trades[-self._max_trades :]

        return fills, step_pnl


    def _build_position_management_signal(
        self,
        symbol: str,
        canonical_symbol: str,
        position: Any,
        intents: List[Dict[str, Any]],
    ) -> PositionManagementSignal:
        expert_signals: List[ExpertSignal] = []
        position_side = getattr(position, "side", 0)
        position_action = "BUY" if position_side > 0 else "SELL"


        try:
            committee_votes = self.bus.get("committee_votes", "Executor")
            if isinstance(committee_votes, list):
                for vote in committee_votes:
                    if isinstance(vote, dict):
                        expert_name = vote.get("member", vote.get("expert_name", "Unknown"))
                        action = str(vote.get("action", "HOLD")).upper()
                        confidence = float(vote.get("confidence", 0.5) or 0.5)
                        reasoning = vote.get("reasoning", vote.get("thesis", ""))

                        supports_position = (
                            action == position_action
                            or action == "HOLD"
                            or (position_side > 0 and action == "BUY")
                            or (position_side < 0 and action == "SELL")
                        )

                        expert_signals.append(
                            ExpertSignal(
                                expert_name=expert_name,
                                action=action,
                                confidence=confidence,
                                reasoning=str(reasoning)[:200] if reasoning else "",
                                supports_position=supports_position,
                            )
                        )
        except Exception:
            pass


        try:
            instrument_signals = self.bus.get("instrument_signals", "Executor")
            if isinstance(instrument_signals, dict):
                inst_signal = instrument_signals.get(canonical_symbol) or instrument_signals.get(symbol)
                if isinstance(inst_signal, dict):
                    inst_action = str(inst_signal.get("action", "HOLD")).upper()
                    inst_conf = float(inst_signal.get("confidence", 0.5) or 0.5)
                    supports_position = inst_action in (position_action, "HOLD")
                    expert_signals.append(
                        ExpertSignal(
                            expert_name="InstrumentAlignment",
                            action=inst_action,
                            confidence=inst_conf,
                            reasoning=inst_signal.get("reason", ""),
                            supports_position=supports_position,
                        )
                    )
        except Exception:
            pass


        consensus_action = "HOLD"
        consensus_confidence = 0.5
        consensus_score = 0.5
        fragility = 0.5
        regime = "unknown"
        volatility_level = "medium"

        try:
            final_decision = self.bus.get("final_decision", "Executor")
            if isinstance(final_decision, dict):
                raw_action = str(final_decision.get("action", "HOLD")).lower()

                if raw_action in ("buy", "long", "bullish", "open_long"):
                    consensus_action = "BUY"
                elif raw_action in ("sell", "short", "bearish", "open_short"):
                    consensus_action = "SELL"
                else:
                    consensus_action = "HOLD"
                consensus_confidence = float(
                    final_decision.get("confidence", 0.5) or 0.5
                )


            ppo_decision = self.bus.get("ppo_final_decision", "Executor", default=None)
            if isinstance(ppo_decision, dict):
                ppo_dir = str(ppo_decision.get("direction", "flat")).lower()
                ppo_gate = ppo_decision.get("gate_passed", False)
                ppo_conf = float(ppo_decision.get("confidence", 0.0) or 0.0)


                if ppo_gate and ppo_conf > 0.5:
                    if ppo_dir in ("long", "buy"):
                        consensus_action = "BUY"
                        consensus_confidence = max(consensus_confidence, ppo_conf)
                    elif ppo_dir in ("short", "sell"):
                        consensus_action = "SELL"
                        consensus_confidence = max(consensus_confidence, ppo_conf)

            consensus_result = self.bus.get("consensus_result", "Executor")
            if isinstance(consensus_result, dict):
                consensus_score = float(
                    consensus_result.get(
                        "consensus_score", consensus_result.get("agreement_score", 0.5)
                    )
                    or 0.5
                )

            frag = self.bus.get("fragility", "Executor")
            if frag is not None:
                fragility = float(frag)

            reg = self.bus.get("market_regime", "Executor")
            if reg:
                regime = str(reg)

            vol_data = self.bus.get("volatility_data", "Executor")
            if isinstance(vol_data, dict):
                volatility_level = str(
                    vol_data.get("level", vol_data.get("volatility_level", "medium"))
                )
        except Exception:
            pass

        return PositionManagementSignal(
            symbol=symbol,
            expert_signals=expert_signals,
            consensus_action=consensus_action,
            consensus_confidence=consensus_confidence,
            consensus_score=consensus_score,
            fragility=fragility,
            regime=regime,
            volatility_level=volatility_level,
            position_side=position_side,
            position_pnl=float(getattr(position, "unrealized_pnl", 0) or 0),
            position_age_hours=float(getattr(position, "age_hours", 0) or 0),
        )


    def _normalize_symbol(self, symbol: str) -> str:
        return (symbol or "").replace("/", "").replace("_", "").upper()

    def _normalized_adapter_positions(self) -> Dict[str, Dict[str, Any]]:
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

    def _emergency_close_position(self, pos) -> bool:
        try:
            import MetaTrader5 as mt5

            ticket = getattr(pos, "ticket", 0)
            symbol = getattr(pos, "symbol", "")
            lots = getattr(pos, "volume", 0.0)
            pos_type = getattr(pos, "type", 0)

            self.logger.warning(
                f"[EMERGENCY] 🚨 Closing position: ticket={ticket}, symbol={symbol}, lots={lots}"
            )

            close_type = (
                mt5.ORDER_TYPE_SELL
                if pos_type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

            tick = mt5.symbol_info_tick(symbol)  # type: ignore[attr-defined]
            if not tick:
                self.logger.error(f"[EMERGENCY] No tick for {symbol}")
                return False

            price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": lots,
                "type": close_type,
                "position": ticket,
                "price": price,
                "magic": 123456,
                "comment": "EMERGENCY_9PCT",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)  # type: ignore[attr-defined]

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.warning(f"[EMERGENCY] ✅ CLOSED ticket {ticket}")
                return True
            else:
                retcode = getattr(result, "retcode", "unknown") if result else "no_result"
                self.logger.error(f"[EMERGENCY] ❌ Failed to close {ticket}: {retcode}")
                return False

        except Exception as e:
            self.logger.error(f"[EMERGENCY] Exception closing position: {e}")
            return False

    def _close_position_by_ticket(self, ticket: int, symbol: str) -> Dict[str, Any]:
        self.logger.debug(f"[CLOSE_TICKET] Attempting to close ticket {ticket} on {symbol}")

        if not self.adapter or not self.adapter.is_connected():
            self.logger.warning("[CLOSE_TICKET] Adapter not connected")
            return {"ok": False, "error": "not_connected"}

        try:
            import MetaTrader5 as mt5  # type: ignore[import]

            position = mt5.positions_get(ticket=ticket)  # type: ignore[attr-defined]
            if not position:
                self.logger.warning(f"[CLOSE_TICKET] Position {ticket} not found in MT5")
                return {"ok": False, "error": "position_not_found"}

            pos = position[0]
            lots = getattr(pos, "volume", 0.0)
            pos_type = getattr(pos, "type", 0)

            self.logger.info(
                f"[CLOSE_TICKET] Found position: ticket={ticket}, lots={lots}, type={pos_type}"
            )

            close_type = (
                mt5.ORDER_TYPE_SELL
                if pos_type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

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

            tick = mt5.symbol_info_tick(symbol)  # type: ignore[attr-defined]
            if tick:
                request["price"] = (
                    tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                )
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
                self.logger.error(
                    f"[CLOSE_TICKET] ❌ MT5 rejected close: retcode={retcode}, comment={comment}"
                )
                return {"ok": False, "error": f"{retcode}: {comment}"}

        except Exception as e:
            self.logger.error(f"[CLOSE_TICKET] Exception: {e}")
            return {"ok": False, "error": str(e)}

    def _partial_close_by_ticket(self, ticket: int, symbol: str, lots_to_close: float) -> Dict[str, Any]:
        self.logger.debug(f"[PARTIAL_CLOSE] Attempting partial close: ticket={ticket}, symbol={symbol}, lots={lots_to_close}")

        if not self.adapter or not self.adapter.is_connected():
            self.logger.warning("[PARTIAL_CLOSE] Adapter not connected")
            return {"ok": False, "error": "not_connected"}

        try:
            import MetaTrader5 as mt5  # type: ignore[import]

            position = mt5.positions_get(ticket=ticket)  # type: ignore[attr-defined]
            if not position:
                self.logger.warning(f"[PARTIAL_CLOSE] Position {ticket} not found in MT5")
                return {"ok": False, "error": "position_not_found"}

            pos = position[0]
            current_lots = getattr(pos, "volume", 0.0)
            pos_type = getattr(pos, "type", 0)


            actual_close_lots = min(lots_to_close, current_lots)
            if actual_close_lots <= 0:
                self.logger.warning(f"[PARTIAL_CLOSE] Invalid lots to close: {lots_to_close} (current: {current_lots})")
                return {"ok": False, "error": "invalid_lots"}

            self.logger.info(
                f"[PARTIAL_CLOSE] Position: ticket={ticket}, current_lots={current_lots}, "
                f"closing_lots={actual_close_lots}, type={'BUY' if pos_type == 0 else 'SELL'}"
            )


            close_type = (
                mt5.ORDER_TYPE_SELL
                if pos_type == mt5.POSITION_TYPE_BUY
                else mt5.ORDER_TYPE_BUY
            )

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": round(actual_close_lots, 2),
                "type": close_type,
                "position": ticket,
                "magic": 123456,
                "comment": "scale_down",
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            tick = mt5.symbol_info_tick(symbol)  # type: ignore[attr-defined]
            if tick:
                request["price"] = (
                    tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
                )
            else:
                self.logger.warning(f"[PARTIAL_CLOSE] No tick data for {symbol}")

            self.logger.info(f"[PARTIAL_CLOSE] Sending request: {request}")
            result = mt5.order_send(request)  # type: ignore[attr-defined]

            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info(
                    f"[PARTIAL_CLOSE] ✅ Successfully closed {actual_close_lots} lots on ticket {ticket}. "
                    f"Remaining: {current_lots - actual_close_lots:.2f} lots"
                )
                return {"ok": True, "price": getattr(result, "price", 0), "closed_lots": actual_close_lots}
            else:
                retcode = getattr(result, "retcode", "unknown") if result else "no_result"
                comment = getattr(result, "comment", "") if result else ""
                self.logger.error(
                    f"[PARTIAL_CLOSE] ❌ MT5 rejected: retcode={retcode}, comment={comment}"
                )
                return {"ok": False, "error": f"{retcode}: {comment}"}

        except Exception as e:
            self.logger.error(f"[PARTIAL_CLOSE] Exception: {e}")
            return {"ok": False, "error": str(e)}


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
        initial_balance = float(getattr(self, "initial_balance", 0.0) or 0.0)
        balance = float(self.balance)
        equity = float(self.equity)
        drawdown = max(0.0, (initial_balance - balance) / max(initial_balance, 1e-9)) if initial_balance > 0 else 0.0
        total_closed = len(self.closed_positions)
        wins = 0
        if total_closed > 0:
            try:
                wins = sum(1 for p in self.closed_positions if float(p.get("pnl", p.get("profit", 0.0)) or 0.0) > 0.0)
            except Exception:
                wins = 0
        win_rate = (wins / total_closed) if total_closed > 0 else 0.5
        pnl_trend = float(step_pnl) / max(initial_balance * 0.01, 1e-9) if initial_balance > 0 else 0.0
        pnl_trend = float(max(-1.0, min(1.0, pnl_trend)))

        market_state = {
            "balance": balance,
            "equity": equity,
            "step": int(self.step_idx),
            "initial_balance": initial_balance,
            "drawdown": float(drawdown),
            "win_rate": float(win_rate),
            "pnl_trend": float(pnl_trend),
        }

        self.bus.set("positions", pos_snap, thesis="Positions snapshot (executor)")


        position_focus = PositionFocusContext.from_positions(pos_snap)
        self.bus.set(
            "position_focus_context",
            position_focus.to_dict(),
            thesis=(
                f"Position focus mode: {'ACTIVE' if position_focus.focus_mode_active else 'INACTIVE'} "
                f"({len(pos_snap)} positions)"
                if pos_snap
                else "No positions - looking for new trades"
            ),
        )

        self.bus.set("trades", trade_ledger, thesis="Trade ledger (executor)")
        self.bus.set("recent_trades", recent, thesis="Recent fills (executor)")
        self.bus.set("order_data", order_data, thesis="Orders seen this step (executor)")
        self.bus.set("execution_data", execution_data, thesis="Fills this step (executor)")
        self.bus.set("execution_reports", exec_fills, thesis="Fills alias (executor)")


        self.bus.set("current_fills", exec_fills, thesis="Current step fills (executor)")


        self.bus.set("current_fills", exec_fills, thesis="Current step fills (executor)")


        closed_positions = list(self.closed_positions)
        self.bus.set("closed_positions", closed_positions, thesis="Closed positions (executor)")


        self.bus.set("portfolio_metrics", portfolio_metrics, thesis="Portfolio metrics (executor)")
        self.bus.set("market_state", market_state, thesis="Market/account state (executor)")

        pnl_data = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),
            "realized_pnl_step": float(realized_step),
            "unrealized_pnl": float(unrealized),
            "step": int(self.step_idx),
        }
        self.bus.set("pnl_data", pnl_data, thesis="P&L alias (executor)")


        trade_data = {
            "trades": trade_ledger,
            "recent_trades": recent,
            "fills": exec_fills,
            "step": int(self.step_idx),
        }
        self.bus.set("trade_data", trade_data, thesis="Trade data bundle (executor)")

        pos_list: List[Dict[str, Any]] = []
        try:
            for inst, p in (pos_snap or {}).items():
                notional = float(p.get("notional_eur", 0.0) or 0.0)
                units = float(p.get("units", 0.0) or 0.0)
                entry_price = float(p.get("entry_price", 0.0) or 0.0)
                size = (
                    abs(notional)
                    if abs(notional) > 0
                    else (abs(units * entry_price) if (units and entry_price) else abs(units))
                )
                row: Dict[str, Any] = {"instrument": inst, "size": float(size)}
                for k, v in p.items():
                    if k != "instrument":
                        row[k] = v
                pos_list.append(row)
        except Exception:
            try:
                pos_list = [{"instrument": inst, **(p or {})} for inst, p in (pos_snap or {}).items()]
            except Exception:
                pos_list = []

        position_data = {"positions": pos_list, "count": len(pos_list)}
        self.bus.set("position_data", position_data, thesis="Position data bundle (executor)")
        self.bus.set("current_positions", pos_snap, thesis="Current positions alias (executor)")


        account_state = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "initial_balance": float(self.initial_balance),
            "step": int(self.step_idx),
        }
        self.bus.set("account_state", account_state, thesis="Account state (executor)")


        self.bus.set("current_pnl", float(step_pnl), thesis="Current P&L (executor)")
        self.bus.set("balance", float(self.balance), thesis="Account balance (executor)")
        self.bus.set("pending_orders", accepted, thesis="Pending orders (executor)")


        existing_queue = self.bus.get("order_queue", "Executor", default=None)
        if existing_queue is None:
            self.bus.set("order_queue", [], thesis="Order queue fallback (executor)")

        trading_result = {"pnl": float(step_pnl), "step": int(self.step_idx)}
        self.bus.set("trading_result", trading_result, thesis="Per-step trading result (executor)")


        live_adapter_status = self.bus.get(
            "live_adapter_status",
            "Executor",
            default={"provider": self.cfg.live_broker, "connected": False},
        )
        self.bus.set(
            "live_adapter_status",
            live_adapter_status,
            thesis="Live adapter status (executor refresh)",
        )


    def _log_unified_cycle(
        self,
        *,
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
        if not getattr(self, "unified_logger", None):
            return

        try:
            try:
                from inspect import signature  # type: ignore
            except Exception:
                signature = None  # type: ignore


            payload = {

                "step": int(self.step_idx),
                "mode": str(mode),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),

                "orders_received": int(q_count),
                "orders_accepted": len(accepted),
                "orders_rejected": len(rejected),
                "rejected_reasons": {},

                "fills_count": len(fills),
                "fills_by_instrument": {},
                "total_notional": 0.0,

                "positions_before": {},
                "positions_after": positions_after or {},
                "positions_opened": [],
                "positions_closed": [],
                "positions_modified": [],

                "balance_before": float(balance_before),
                "balance_after": float(self.balance),
                "equity_before": float(equity_before),
                "equity_after": float(self.equity),
                "realized_pnl": float(realized_step),
                "unrealized_pnl": float(unreal_after),
                "step_pnl": float(step_pnl),

                "trades_this_step": [],

                "execution_time_ms": float(processing_ms),
                "issues": [],

                "accepted_details": list(accepted) if accepted else [],
                "rejected_details": list(rejected) if rejected else [],
                "fill_details": list(fills) if fills else [],
            }

            entry = None
            try:
                if ExecutionCycleEntry is not None:
                    if signature is not None:
                        sig = signature(ExecutionCycleEntry)  # type: ignore[arg-type]
                        allowed = {k for k in sig.parameters.keys() if k != "self"}
                        filtered = {k: v for k, v in payload.items() if k in allowed}
                    else:
                        filtered = payload
                    entry = ExecutionCycleEntry(**filtered)  # type: ignore[call-arg]
            except Exception as e:
                try:
                    self.logger.warning(f"ExecutionCycleEntry construction failed: {e}")
                except Exception:
                    pass
                entry = None

            if entry is not None:
                try:
                    self.unified_logger.log_execution_cycle(entry)  # type: ignore[attr-defined]
                    return
                except Exception as e:
                    try:
                        self.logger.warning(f"UnifiedExecutorLogger.log_execution_cycle failed: {e}")
                    except Exception:
                        pass


            try:
                self.logger.info(
                    f"[EXEC-CYCLE] step={self.step_idx} mode={mode} "
                    f"q={q_count} dec={dec_count} acc={len(accepted)} rej={len(rejected)} "
                    f"fills={len(fills)} pnl={step_pnl:.2f} ms={processing_ms:.1f}"
                )
            except Exception:
                pass

        except Exception as outer:
            try:
                self.logger.warning(f"_log_unified_cycle failed: {outer}")
            except Exception:
                pass
