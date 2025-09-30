# ─────────────────────────────────────────────────────────────
# File: modules/executor/executor.py  (revamped debug wiring + full contract return)
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import time, uuid, math
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


@dataclass
class ExecutorConfig:
    execution_mode: str = "sim"        # 'sim' | 'live'
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
        self.balance: float = float(10_000 if _cfg_ib is None else _cfg_ib)
        self.equity: float = float(self.balance)
        self._last_equity: float = float(self.equity)
        self.positions: Dict[str, PositionSnap] = {}
        self.trades: List[Dict[str, Any]] = []
        self.step_idx: int = 0
        self._seen_ids: Set[str] = set()

        # live adapter (ensure initialized; _initialize also handles this during super())
        if not hasattr(self, "adapter"):
            self.adapter: Optional[BaseLiveAdapter] = None
        self._ensure_adapter()
        self._publish_adapter_status()

        # debugger
        dbg_cfg: Dict[str, Any] = {**(self.cfg.debug_config or {}), "enabled": bool(self.cfg.debug_enabled)}
        self.debugger = ExecutorDebugManager(self.bus, config=dbg_cfg)

        # seed bus with empty snapshots
        self._publish_all(exec_fills=[], accepted=[], rejected=[], step_pnl=0.0, reason="startup")

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
    def _ensure_adapter(self) -> None:
        if self.cfg.execution_mode == "live":
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
        if not self.cfg.allow_runtime_switch:
            return self.cfg.execution_mode
        em = self.bus.get("execution_mode", "Executor", default=None)
        if isinstance(em, str) and em.lower() in ("sim", "live"):
            target = em.lower()
        else:
            envc = self.bus.get("environment_config", "Executor", default={}) or {}
            target = str(envc.get("mode", self.cfg.execution_mode)).lower()
            if target not in ("sim", "live"):
                target = self.cfg.execution_mode
        if target == "live":
            if not self.adapter or not self.adapter.is_connected():
                self._ensure_adapter()
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
                pos_snap = {k: v.as_bus() for k, v in self.positions.items()}

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
                fills, step_pnl = self._execute_live(accepted)
                self.debugger.end("execute_live")
                positions_after = self.adapter.sync_positions() if self.adapter else {}
                acct = self.adapter.get_account_info() if self.adapter else {}
                self.balance = float(acct.get("balance", self.balance))
                self.equity = float(acct.get("equity", self.equity))
            else:
                self.debugger.begin("execute_sim")
                fills, step_pnl, realized_step, unreal_after = self._execute_sim(accepted, want_breakdown=True)
                self.debugger.end("execute_sim")
                positions_after = {k: v.as_bus() for k, v in self.positions.items()}

            # publish to bus
            self.debugger.begin("publish_bus")
            self._publish_all(exec_fills=fills, accepted=accepted, rejected=rejected, step_pnl=step_pnl)
            self.debugger.end("publish_bus")
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

        # REQUIRED outputs for orchestrator contract (return payload)
        recent = self.trades[-50:] if self.trades else []
        order_data = {"accepted": accepted, "rejected": rejected, "step": int(self.step_idx)}
        execution_data = {"fills": fills, "step": int(self.step_idx)}
        market_state = {"balance": float(self.balance), "equity": float(self.equity), "step": int(self.step_idx)}
        portfolio_metrics = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),
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
            # Fall back to a basic projection if anything goes wrong
            try:
                pos_list = [{"instrument": inst, **(p or {})} for inst, p in (positions_after or {}).items()]
            except Exception:
                pos_list = []

        # also include raw balance/equity at top-level for strict readers
        return {
            # core snapshots
            "positions": positions_after,
            "trades": self.trades[-200:],
            "recent_trades": recent,

            # per-step order & execution rollups
            "order_data": order_data,
            "execution_data": execution_data,
            "execution_reports": fills,

            # portfolio telemetry
            "portfolio_metrics": portfolio_metrics,
            "trading_result": {"pnl": float(step_pnl)},

            # rollups for downstream consumers
            "trade_data": trade_data,
            "market_state": market_state,

            # FIX: Contract-required position_data (canonical publisher) — normalized list schema
            "position_data": {"positions": pos_list, "count": len(pos_list)},

            # helpful aliases (explicitly returned to satisfy strict orchestrators)
            "current_positions": current_positions,
            "pnl_data": pnl_data,

            # live routing/health surface
            "live_adapter_status": live_adapter_status,

            # **raw aliases to avoid BUS MISS for simple readers**
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),

            # baselines
            "pending_orders": order_data.get("accepted", []),
            "account_state": {"balance": float(self.balance), "equity": float(self.equity), "step": int(self.step_idx)},

            # housekeeping
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

        # explicit order_queue
        oq = self.bus.get("order_queue", "Executor", default=[])
        if isinstance(oq, list):
            q_count = len(oq)
            for item in oq[: self.cfg.max_orders_per_step]:
                intent = self._normalize_order_item(item)
                if not intent:
                    rejected.append({"reason": "bad_order_queue_item", "raw": item})
                elif self._passes_filters(intent):
                    if intent["id"] not in self._seen_ids:
                        accepted.append(intent); self._seen_ids.add(intent["id"])
                    else:
                        rejected.append({"reason": "duplicate_id", "intent": intent})
                else:
                    reason = self._filter_reason(intent)
                    rejected.append({"reason": reason, "intent": intent})

        # consume queue (do not write canonical key back to the bus; owner is PositionManager)
        # Keep consumption internal to Executor

        # fallback: position_decision_* from environment_config instruments
        if self.cfg.read_position_decisions:
            env_cfg = self.bus.get("environment_config", "Executor", default={}) or {}
            instruments = env_cfg.get("instruments") or []
            for inst in instruments:
                node = self.bus.get(f"position_decision_{inst}", "Executor", default=None)
                if not isinstance(node, dict) or not node.get("decision"):
                    node = self.bus.get(f"position_decision_{inst.replace('/','').replace('_','')}", "Executor", default=None)
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

        return accepted, rejected, q_count, dec_count

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
        if intent["action"] == "hold" and self.cfg.ignore_hold:
            return False
        if float(intent.get("confidence", 0.0)) < self.cfg.min_confidence:
            return False
        if abs(float(intent.get("intensity", 0.0))) < self.cfg.min_intensity and intent["action"] not in ("close", "emergency_close"):
            return False
        return True

    def _filter_reason(self, intent: Dict[str, Any]) -> str:
        if intent["action"] == "hold" and self.cfg.ignore_hold:
            return "ignore_hold"
        if float(intent.get("confidence", 0.0)) < self.cfg.min_confidence:
            return f"low_confidence<{self.cfg.min_confidence}"
        if abs(float(intent.get("intensity", 0.0))) < self.cfg.min_intensity and intent["action"] not in ("close", "emergency_close"):
            return f"low_intensity<{self.cfg.min_intensity}"
        if intent.get("id") in self._seen_ids:
            return "duplicate_id"
        return "filtered"

    # ─────────────────────────────────────────────────────────
    # SIM execution
    # ─────────────────────────────────────────────────────────
    def _sim_price(self, inst: str, side: int) -> Optional[float]:
        sp = self.bus.get("prices", "Executor", default={}) or {}
        pd = self.bus.get("price_data", "Executor", default={}) or {}
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
            px += (self.cfg.default_spread / 2.0) * (+1 if side < 0 else -1)
        if self.cfg.slippage_pts:
            px += self.cfg.slippage_pts * (+1 if side < 0 else -1)
        return px

    def _units_from(self, size_eur: float, units: float, price: float) -> float:
        if units and units > 0:
            return float(units)
        if size_eur and size_eur > 0 and price > 0:
            return float(size_eur / price)
        return 0.0

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
                    del self.positions[inst]

                if add_units > 0:
                    notional = add_units * price
                    self.positions[inst] = PositionSnap(inst, side_from_action, add_units, price, notional_eur=notional)
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
                    self.positions[inst] = PositionSnap(inst, +1, add_units, price, notional_eur=notional)
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
                if p.side > 0:
                    reduce_u = min(add_units, p.units)
                    realized = (price - p.entry_price) * p.side * reduce_u - commission(reduce_u * price)
                    realized_step += realized
                    p.units -= reduce_u
                    p.notional_eur -= reduce_u * p.entry_price
                    if p.units <= 1e-12:
                        del self.positions[inst]
                    side = -1
                else:
                    new_u = p.units + add_units
                    p.entry_price = (p.entry_price * p.units + price * add_units) / new_u
                    p.units = new_u
                    p.notional_eur += add_units * price
                    side = -1
                fill = TradeFill(
                    id=f"fill-{uuid.uuid4().hex[:10]}",
                    ts=time.time(),
                    step=self.step_idx,
                    instrument=inst,
                    action="scale_down" if side == -1 else "scale_down_reduce",
                    side=side,
                    units=add_units,
                    price=price,
                    notional_eur=add_units * price,
                    origin_id=origin_id,
                    comment="scale_down",
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
                    del self.positions[inst]

        # apply realized → balance
        self.balance += realized_step

        # mark-to-market
        unreal = 0.0
        for inst, p in self.positions.items():
            px = self._sim_price(inst, p.side)
            if px is None:
                continue
            unreal += (px - p.entry_price) * p.side * p.units

        equity_now = self.balance + unreal
        step_pnl = float(equity_now - self._last_equity)
        self._last_equity = equity_now
        self.equity = equity_now

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

        for intent in intents:
            inst_src = intent["instrument"]
            inst = resolve_symbol(inst_src, self.cfg.symbol_overrides)
            action = str(intent["action"]).lower()
            side = {"open_long": +1, "scale_up": +1, "open_short": -1, "scale_down": -1}.get(action, 0)

            price_hint = (self.adapter.get_prices(inst) or {}).get("mid", 0.0) or 1.0
            units = float(intent.get("units", 0.0) or 0.0)
            size_eur = float(intent.get("size_eur", 0.0) or 0.0)
            if units <= 0 and size_eur > 0 and price_hint > 0:
                units = size_eur / price_hint
            lots = max(units / self.adapter.cfg.contract_size, 0.0)
            lots = round_to_step(lots, self.adapter.cfg.lot_step)
            lots = max(lots, self.adapter.cfg.min_lot) if lots > 0 else 0.0

            origin_id = intent.get("id", "")

            if action in ("open_long", "open_short", "scale_up"):
                if lots <= 0:
                    continue
                r = self.adapter.market_order(inst, side, lots)
                if r.get("ok"):
                    px = float(r.get("price", price_hint) or price_hint)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
                        action=action,
                        side=side,
                        units=lots * self.adapter.cfg.contract_size,
                        price=px,
                        notional_eur=lots * self.adapter.cfg.contract_size * px,
                        realized_pnl=0.0,
                        origin_id=origin_id,
                        comment="live",
                    ).as_bus()
                    self.trades.append(fill); fills.append(fill)
            elif action == "scale_down":
                if lots <= 0:
                    continue
                r = self.adapter.reduce_position(inst, lots, -1 if side > 0 else +1)
                if r.get("ok"):
                    px = float(r.get("price", price_hint) or price_hint)
                    fill = TradeFill(
                        id=f"fill-{uuid.uuid4().hex[:10]}",
                        ts=time.time(),
                        step=self.step_idx,
                        instrument=inst,
                        action="scale_down",
                        side=-1 if side > 0 else +1,
                        units=lots * self.adapter.cfg.contract_size,
                        price=px,
                        notional_eur=lots * self.adapter.cfg.contract_size * px,
                        realized_pnl=0.0,
                        origin_id=origin_id,
                        comment="live_reduce",
                    ).as_bus()
                    self.trades.append(fill); fills.append(fill)
            elif action in ("close", "emergency_close"):
                self.adapter.close_position(inst)

        acct_after = self.adapter.get_account_info()
        eq_after = float(acct_after.get("equity", eq_before) or eq_before)
        step_pnl = float(eq_after - eq_before)

        self.balance = float(acct_after.get("balance", self.balance))
        self.equity = float(eq_after)
        self._last_equity = float(eq_after)

        return fills, step_pnl

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
        reason: str = ""
    ) -> None:
        pos_snap: Dict[str, Any] = {}
        mode = self._resolve_mode()
        if mode == "live" and self.adapter:
            pos_snap = self.adapter.sync_positions()
        else:
            for inst, p in self.positions.items():
                pos_snap[inst] = p.as_bus()

        trade_ledger = list(self.trades)
        recent = trade_ledger[-50:] if trade_ledger else []

        order_data = {"accepted": accepted, "rejected": rejected, "step": int(self.step_idx)}
        execution_data = {"fills": exec_fills, "step": int(self.step_idx)}
        portfolio_metrics = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "current_pnl": float(step_pnl),
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
            self.bus.set("account_state", {"balance": float(self.balance), "equity": float(self.equity), "step": int(self.step_idx)}, thesis="Account state (executor)")
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
            {"balance": self.balance, "equity": self.equity, "current_pnl": step_pnl, "step": self.step_idx},
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
                reason=reason or "ok",
            )
        )
