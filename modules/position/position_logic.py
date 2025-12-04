# -------------------------------------------------------------
# File: modules/position/position_logic.py
# PositionManager — decision logic + sizing + profit rules
# (subclasses PositionManagerBase from position_base.py)
#
# EXIT LOGIC: Uses unified ExitStrategyEngine for consistency
# with SmartPositionManager (live trading). Both systems use the
# same exit strategies configured in risk_policy.yaml.
# -------------------------------------------------------------

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from modules.contracts import module_args
from modules.core.error_pinpointer import create_error_handler
from modules.core.module_base import module
from modules.utils.audit_utils import format_operator_message

from .position_base import (
    PositionManagerBase,
    PositionDecision,
    PositionDecisionResult,
    SignalContext,
)
from .position_logger import UnifiedPositionLogger, PositionLogEntry
from .exit_engine import (
    ExitStrategyEngine,
    ExitDecision,
    ExitReason,
    PositionContext,
    get_exit_engine,
)


@module(
    **module_args(
        "PositionManager",
        description=(
            "Position decision maker with trailing take-profit & hard loss-cut; "
            "Env/Executor handle execution."
        ),
        error_handling=True,
        hot_reload=True,
        timeout_ms=3000,
    )
)
class PositionManager(PositionManagerBase):
    """
    Decision layer:
      - Opens only on adequate signal.
      - Scales and sizes via volatility / portfolio health / memory.
      - Can close positions on:
          * Trailing take-profit (PnL retrace from peak + favorability degradation)
          * Emergency conditions (drawdown / loss streak / exposure / liquidity)
          * Agent intent (direction flip or very weak signal).
    """

    # ==========================================================
    # Public pipeline hooks (called by PositionManagerBase)
    # ==========================================================
    @create_error_handler("process_market_signals")
    def process_market_signals(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, PositionDecisionResult]:
        decisions: Dict[str, PositionDecisionResult] = {}

        # Portfolio & regime assessment
        portfolio_health = self._assess_portfolio_health()
        market_regime = self._assess_market_regime(market_data)
        if market_regime is not None:
            # Make a shallow copy so we don't surprise upstream callers
            market_data = dict(market_data)
            market_data["market_regime"] = market_regime

        # Log portfolio stats with unified logger
        if self.debug and hasattr(self, "unified_logger"):
            self.unified_logger.log_portfolio_stats(portfolio_health)

        # Per-instrument decisions
        for instrument in self.instruments:
            try:
                ctx = self._extract_signal_context(instrument, market_data, portfolio_health)

                # Update short signal history (for favorability slope / trailing logic)
                self.signal_history[instrument].append(ctx.market_intensity)
                if len(self.signal_history[instrument]) > 50:
                    self.signal_history[instrument].pop(0)

                # Maintain trailing P&L peak (for analytics / logging only).
                # ExitStrategyEngine is the single source of truth for exit peaks.
                self.update_profit_tracker(instrument)

                dr = self._make_position_decision(ctx)
                decisions[instrument] = dr
                self.last_decisions[instrument] = dr

            except Exception as e:
                # Failsafe: never crash the whole pipeline for a single instrument
                if self.debug:
                    self.logger.warning(
                        f"[PositionManager] Decision pipeline failed for {instrument}: {e}"
                    )
                decisions[instrument] = PositionDecisionResult(
                    decision=PositionDecision.HOLD,
                    intensity=0.0,
                    size=0.0,
                    confidence=0.0,
                    rationale={
                        "stage": "error",
                        "factors": [f"Exception in decision loop: {e}"],
                    },
                    risk_factors={},
                    context=SignalContext(instrument=instrument),
                )

        if self.debug:
            self._flush_logs()

        return decisions

    # ==========================================================
    # Core decision logic
    # ==========================================================
    def _make_position_decision(self, context: SignalContext) -> PositionDecisionResult:
        """
        Position decision logic.

        Handles:
        0. PPOAgent intelligent arbiter gate (final GO/NO-GO decision).
        1. Memory veto / danger-zone gating (pre-entry).
        2. Emergency conditions (hard exits / no new exposure).
        3. Exit strategies (via unified ExitStrategyEngine).
        4. Agent-driven exits (direction flips, weak signals).
        5. New entries with portfolio health & hedge prevention.
        """
        exit_engine = get_exit_engine()
        instrument = context.instrument

        has_position = self._has_position_for_instrument(instrument)

        # If instrument is flat, clear any stale exit-engine peak.
        if not has_position:
            try:
                exit_engine.reset_peak(instrument)
            except Exception:
                pass

        decision = PositionDecision.HOLD
        intensity = 0.0
        size = 0.0
        confidence = 0.5
        rationale: Dict[str, Any] = {"stage": "initial", "factors": []}
        risk_factors: Dict[str, float] = {}

        # ======================================================
        # PPO INTELLIGENT ARBITER GATE (highest priority)
        # PPOAgent has seen committee consensus, risk, memory and made final decision
        # ======================================================
        ppo_gate_passed = self.smart_bus.get("ppo_gate_passed", "PositionManager", default=True)
        ppo_final_decision = self.smart_bus.get("ppo_final_decision", "PositionManager", default={})
        ppo_position_size = self.smart_bus.get("ppo_position_size", "PositionManager", default=None)

        # If PPOAgent explicitly blocked the trade, respect that decision
        if ppo_gate_passed is False and not has_position:
            rationale["stage"] = "ppo_arbiter_veto"
            rationale["factors"].append(
                f"PPOAgent intelligent arbiter vetoed: {ppo_final_decision.get('reasoning', 'No reason provided')}"
            )
            rationale["ppo_decision"] = ppo_final_decision
            return self._finalize_decision(
                instrument,
                decision,
                0.0,
                0.0,
                0.3,  # Low confidence in holding
                rationale,
                risk_factors,
                context,
            )

        # ======================================================
        # MEMORY INTEGRATION: Read all memory signals upfront
        # ======================================================
        memory_data = self._get_memory_intelligence()

        # ---------- MEMORY VETO CHECK (highest priority gate for new entries)
        if memory_data.get("veto", False) and not has_position:
            rationale["stage"] = "memory_veto"
            rationale["factors"].extend(
                memory_data.get("veto_reasons", ["Memory system vetoed this trade"])
            )
            rationale["memory_data"] = memory_data
            return self._finalize_decision(
                instrument,
                decision,
                0.0,
                0.0,
                0.3,
                rationale,
                risk_factors,
                context,
            )

        # ---------- Fast emergency gate (can close, prevents new risk)
        if self._check_emergency_conditions(context):
            if has_position:
                # Close existing exposure with full notional size
                close_notional = self._get_position_notional_eur(instrument)
                decision = PositionDecision.EMERGENCY_CLOSE
                intensity = 1.0
                confidence = 0.95
                rationale["stage"] = "emergency"
                rationale["factors"].append("Emergency conditions detected")

                # Reset exit engine peak because position will be force-closed.
                try:
                    exit_engine.reset_peak(instrument)
                except Exception:
                    pass

                return self._finalize_decision(
                    instrument,
                    decision,
                    intensity,
                    close_notional,
                    confidence,
                    rationale,
                    risk_factors,
                    context,
                )

            # No open position: do not open anything in an emergency
            rationale["stage"] = "emergency_hold"
            rationale["factors"].append("Emergency conditions; no open position to close")
            return self._finalize_decision(
                instrument,
                decision,
                0.0,
                0.0,
                0.6,
                rationale,
                risk_factors,
                context,
            )

        # ---------- MEMORY DANGER ZONE CHECK (before opening new positions)
        if not has_position and memory_data.get("in_danger_zone", False):
            danger_similarity = float(memory_data.get("danger_similarity", 0.0) or 0.0)
            if danger_similarity > 0.7:
                rationale["stage"] = "memory_danger_zone"
                rationale["factors"].append(
                    f"Memory danger zone: {danger_similarity:.1%} similarity to past losses"
                )
                rationale["memory_data"] = memory_data
                return self._finalize_decision(
                    instrument,
                    decision,
                    0.0,
                    0.0,
                    0.4,
                    rationale,
                    risk_factors,
                    context,
                )

        # ---------- Signal & portfolio gates
        min_sig = float(self.Cval("min_signal_threshold", 0.20))
        sig_strength = abs(float(context.market_intensity))

        # Portfolio health brake for opening new exposure
        portfolio_health_score = self._calculate_portfolio_health_score(context)
        health_floor = 0.30
        if not has_position and portfolio_health_score < health_floor:
            rationale["stage"] = "portfolio_health"
            rationale["factors"].append(
                f"Portfolio health {portfolio_health_score:.3f} below floor {health_floor:.2f} for new exposure"
            )
            return self._finalize_decision(
                instrument,
                decision,
                0.0,
                0.0,
                0.5,
                rationale,
                risk_factors,
                context,
            )

        # ======================================================
        # EXISTING POSITION: exits / holds
        # ======================================================
        if has_position:
            pos_data = self._get_position_for_instrument(instrument)

            # Infer current position side: +1 long, -1 short
            position_side = 1
            if isinstance(pos_data, dict):
                try:
                    side_val = pos_data.get("side", 0)
                    if isinstance(side_val, (int, float)) and side_val != 0:
                        position_side = int(np.sign(side_val))
                    else:
                        units_val = float(pos_data.get("units", 0.0) or 0.0)
                        if units_val != 0:
                            position_side = int(np.sign(units_val))
                except Exception:
                    position_side = 1
            else:
                # Fallback: read from SmartBus positions map
                try:
                    positions = self.smart_bus.get("positions", "PositionManager") or {}
                    inst_norm = self._normalize_instrument(instrument)
                    for pos_key, p in positions.items():
                        if self._normalize_instrument(str(pos_key)) == inst_norm and isinstance(p, dict):
                            side_val = p.get("side", 0)
                            if isinstance(side_val, (int, float)) and side_val != 0:
                                position_side = int(np.sign(side_val))
                            break
                except Exception:
                    pass

            # ================================================================
            # UNIFIED EXIT STRATEGY ENGINE
            # Uses same logic as SmartPositionManager for consistency
            # Strategies: hard_stop > soft_stop > time_decay > trailing > momentum > signal
            # ================================================================
            exit_decision = self._evaluate_exit_strategies(
                instrument=instrument,
                pos_data=pos_data,
                position_side=position_side,
                context=context,
            )

            if exit_decision.should_exit:
                decision = PositionDecision.CLOSE
                intensity = exit_decision.urgency
                confidence = exit_decision.confidence
                rationale["stage"] = f"exit_{exit_decision.reason.name.lower()}"
                rationale["factors"].append(
                    exit_decision.details.get("message", str(exit_decision.reason.name))
                )
                rationale["exit_details"] = exit_decision.to_dict()
                close_notional = self._get_position_notional_eur(instrument)

                # Reset exit engine peak tracking for this instrument
                try:
                    exit_engine.reset_peak(instrument)
                except Exception:
                    pass

                return self._finalize_decision(
                    instrument,
                    decision,
                    intensity,
                    close_notional,
                    confidence,
                    rationale,
                    risk_factors,
                    context,
                )

            # Otherwise HOLD – no exit triggered
            rationale["stage"] = "hold_existing"
            rationale["factors"].append(
                f"Holding position; exit_check={exit_decision.reason.name}, signal={sig_strength:.3f}"
            )
            return self._finalize_decision(
                instrument,
                decision,
                0.0,
                0.0,
                0.6,
                rationale,
                risk_factors,
                context,
            )

        # ======================================================
        # NO POSITION: evaluate entry signals
        # ======================================================
        if sig_strength < min_sig:
            rationale["stage"] = "signal_filter"
            rationale["factors"].append(
                f"Signal {sig_strength:.3f} below minimum threshold {min_sig:.2f} for new entry"
            )
            return self._finalize_decision(
                instrument,
                decision,
                0.0,
                0.0,
                0.5,
                rationale,
                risk_factors,
                context,
            )

        # ======================================================
        # VOTING DIRECTION: per-instrument first, global vote fallback
        # ======================================================
        voting_direction: Optional[int] = None
        voting_confidence = 0.0

        # 1) Try per-instrument signal from FinalArbiter (preferred)
        instrument_signals = self.smart_bus.get("instrument_signals", "PositionManager") or {}
        inst_signal: Any = None
        if isinstance(instrument_signals, dict):
            inst_signal = instrument_signals.get(instrument)
            if inst_signal is None:
                # Try normalized key match (EURUSD / EUR_USD / EUR/USD)
                inst_norm = self._normalize_instrument(instrument)
                for key, val in instrument_signals.items():
                    try:
                        if self._normalize_instrument(str(key)) == inst_norm:
                            inst_signal = val
                            break
                    except Exception:
                        continue

        if isinstance(inst_signal, dict):
            raw_action = (
                inst_signal.get("action")
                or inst_signal.get("decision")
                or inst_signal.get("direction")
                or ""
            )
            inst_action = str(raw_action).upper()
            inst_confidence = float(
                inst_signal.get("confidence", inst_signal.get("weight", 0.0)) or 0.0
            )

            if inst_action in ("BUY", "LONG", "SELL", "SHORT") and inst_confidence > 0.3:
                voting_direction = 1 if inst_action in ("BUY", "LONG") else -1
                voting_confidence = inst_confidence
                rationale["factors"].append(
                    f"Using per-instrument arbiter signal: {inst_action} (conf={inst_confidence:.2f})"
                )
            elif inst_action == "HOLD":
                rationale["stage"] = "instrument_hold"
                rationale["factors"].append(
                    f"Per-instrument arbiter signal is HOLD for {instrument}"
                )
                return self._finalize_decision(
                    instrument,
                    decision,
                    0.0,
                    0.0,
                    0.4,
                    rationale,
                    risk_factors,
                    context,
                )

        # 2) Fallback to global trade_vote_v2 if no per-instrument signal
        if voting_direction is None:
            trade_vote = self.smart_bus.get("trade_vote_v2", "PositionManager")
            if isinstance(trade_vote, dict):
                raw_action = trade_vote.get("action") or trade_vote.get("direction")
                if raw_action is not None:
                    vote_action = str(raw_action).upper()
                    if vote_action in ("BUY", "SELL"):
                        voting_direction = 1 if vote_action == "BUY" else -1
                        voting_confidence = float(
                            trade_vote.get("confidence", 0.5) or 0.5
                        )
                        rationale["factors"].append(
                            f"Using global trade_vote_v2 (fallback): {vote_action}"
                        )

        # Use voting direction if available; otherwise fall back to raw agent direction
        effective_direction = (
            voting_direction if voting_direction is not None else context.market_direction
        )
        if effective_direction == 0:
            rationale["stage"] = "no_direction"
            rationale["factors"].append("No reliable directional consensus; holding flat")
            return self._finalize_decision(
                instrument,
                decision,
                0.0,
                0.0,
                0.4,
                rationale,
                risk_factors,
                context,
            )

        # ======================================================
        # HEDGE PREVENTION: avoid long/short conflicts across instruments
        # ======================================================
        portfolio_direction = self._get_portfolio_direction()
        if portfolio_direction != 0 and portfolio_direction != int(np.sign(effective_direction)):
            rationale["stage"] = "hedge_prevention"
            rationale["factors"].append(
                "Blocked new position to avoid portfolio hedging: "
                f"portfolio is {'LONG' if portfolio_direction > 0 else 'SHORT'}, "
                f"signal is {'LONG' if effective_direction > 0 else 'SHORT'}"
            )
            return self._finalize_decision(
                instrument,
                decision,
                0.0,
                0.0,
                0.3,
                rationale,
                risk_factors,
                context,
            )

        # ======================================================
        # New position: direction, confidence, sizing
        # ======================================================
        decision = (
            PositionDecision.OPEN_LONG if effective_direction > 0 else PositionDecision.OPEN_SHORT
        )
        intensity = sig_strength
        confidence = self._calculate_confidence(context, decision)

        size = self._calculate_position_size(context, intensity, confidence)

        # Enforce min viable notional; otherwise hold
        min_size_pct = float(self.Cval("min_size_pct", 0.01))
        min_notional = context.balance * min_size_pct

        if abs(size) < min_notional:
            if sig_strength > (min_sig + 0.10):
                # Strong signal but tiny size – force minimum notional in the signal direction
                size = np.sign(size or effective_direction) * min_notional
            else:
                rationale["stage"] = "sizing"
                rationale["factors"].append(
                    f"Computed size {size:.2f} below minimum notional {min_notional:.2f}; holding"
                )
                decision = PositionDecision.HOLD
                intensity = 0.0
                size = 0.0
                return self._finalize_decision(
                    instrument,
                    decision,
                    intensity,
                    size,
                    confidence,
                    rationale,
                    risk_factors,
                    context,
                )

        rationale["stage"] = "new_position"
        direction_label = "BULLISH" if effective_direction > 0 else "BEARISH"
        rationale["factors"].append(
            f"{direction_label} signal {sig_strength:.3f} for new position (size≈{size:.2f} EUR)"
        )
        return self._finalize_decision(
            instrument,
            decision,
            intensity,
            size,
            confidence,
            rationale,
            risk_factors,
            context,
        )

    # ==========================================================
    # Decision helpers
    # ==========================================================
    def _normalize_instrument(self, inst: str) -> str:
        """Normalize instrument to standard format (no separators, uppercase)."""
        return inst.replace("/", "").replace("_", "").upper()

    def _has_position_for_instrument(self, instrument: str) -> bool:
        """
        Check if there's an open position for this instrument.
        Handles format variations: EUR/USD, EURUSD, EUR_USD.
        """
        if not self.open_positions:
            return False

        # Direct key match
        if instrument in self.open_positions:
            return True

        inst_norm = self._normalize_instrument(instrument)
        for pos_key in self.open_positions.keys():
            if self._normalize_instrument(str(pos_key)) == inst_norm:
                return True

        return False

    def _get_position_for_instrument(self, instrument: str) -> Optional[Dict[str, Any]]:
        """Get position data for instrument, handling format variations."""
        if not self.open_positions:
            return None

        # Direct key match
        if instrument in self.open_positions:
            return self.open_positions[instrument]

        inst_norm = self._normalize_instrument(instrument)
        for pos_key, pos_data in self.open_positions.items():
            if self._normalize_instrument(str(pos_key)) == inst_norm:
                return pos_data

        return None

    def _get_portfolio_direction(self) -> int:
        """
        Get the net directional bias of all open positions.

        Uses notional-weighted sign of positions (based on 'side' or 'units').

        Returns:
            1  -> portfolio net LONG
           -1  -> portfolio net SHORT
            0  -> flat / hedged / no positions
        """
        if not self.open_positions:
            return 0

        net = 0.0
        for pos_data in self.open_positions.values():
            try:
                side_val = pos_data.get("side", 0)
                if isinstance(side_val, (int, float)) and side_val != 0:
                    sign = float(np.sign(side_val))
                else:
                    units_val = float(pos_data.get("units", 0.0) or 0.0)
                    if units_val == 0:
                        continue
                    sign = float(np.sign(units_val))

                notional = float(pos_data.get("size", 0.0) or 0.0)
                if notional <= 0.0:
                    # Fallback: approximate from units * price
                    units_val = float(pos_data.get("units", 0.0) or 0.0)
                    price_open = float(
                        pos_data.get("price_open", pos_data.get("entry_price", 0.0)) or 0.0
                    )
                    notional = abs(units_val * price_open)

                if notional <= 0.0:
                    continue

                net += sign * notional
            except Exception:
                continue

        if net > 0:
            return 1
        if net < 0:
            return -1
        return 0

    def _get_position_notional_eur(self, instrument: str) -> float:
        """Return absolute notional exposure for an instrument in EUR."""
        pos = self._get_position_for_instrument(instrument)
        if not isinstance(pos, dict):
            return 0.0

        try:
            size = float(pos.get("size", 0.0) or 0.0)
            if size != 0.0:
                return abs(size)

            units = float(pos.get("units", 0.0) or 0.0)
            price_open = float(pos.get("price_open", pos.get("entry_price", 0.0)) or 0.0)
            if units != 0.0 and price_open > 0.0:
                return abs(units * price_open)

            # Conservative fallback: cap by configured max_position_pct of balance
            balance, _ = self._read_balance_and_drawdown()
            return abs(balance * float(self.Cval("max_position_pct", 0.10)))
        except Exception:
            return 0.0

    def _finalize_decision(
        self,
        instrument: str,
        decision: PositionDecision,
        intensity: float,
        size: float,
        confidence: float,
        rationale: Dict[str, Any],
        risk_factors: Dict[str, float],
        context: SignalContext,
    ) -> PositionDecisionResult:
        # Clip intensity/confidence
        intensity = float(np.clip(intensity, 0.0, 1.0))
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Ensure risk_factors is populated
        if not risk_factors:
            risk_factors = self._assess_risk_factors(context)

        # Calculate aggregate risk score for unified logging
        risk_score = (
            sum(risk_factors.values()) / max(len(risk_factors), 1) if risk_factors else 0.0
        )

        # Get current price from context or market data
        current_price = 0.0
        try:
            if getattr(context, "current_price", 0.0):
                current_price = float(context.current_price)
            else:
                price_data = self.smart_bus.get("price_data", "PositionManager") or {}
                inst_price = price_data.get(instrument, {})
                if isinstance(inst_price, dict):
                    current_price = float(
                        inst_price.get("last", inst_price.get("close", 0.0))
                    )
                elif isinstance(inst_price, (int, float)):
                    current_price = float(inst_price)
        except Exception:
            pass

        # Unified logging for non-HOLD decisions
        if self.debug and decision != PositionDecision.HOLD and hasattr(
            self, "unified_logger"
        ):
            try:
                voting_signals = self.unified_logger.get_voting_signals()

                log_entry = PositionLogEntry(
                    instrument=instrument,
                    decision=decision.value,
                    intensity=intensity,
                    size_eur=float(size),
                    confidence=confidence,
                    signal_strength=abs(context.market_intensity),
                    volatility=context.volatility,
                    trend_strength=context.trend_strength,
                    current_price=current_price,
                    portfolio_health=self._portfolio_health_score,
                    exposure_ratio=context.current_exposure,
                    balance=context.balance,
                    drawdown=context.drawdown,
                    risk_score=risk_score,
                    committee_consensus=voting_signals.get("committee_consensus"),
                    trade_vote=voting_signals.get("trade_vote"),
                    consensus_strength=voting_signals.get("consensus_strength"),
                    stage=rationale.get("stage", "unknown"),
                    factors=rationale.get("factors", []),
                    risk_factors=risk_factors,
                    will_execute=True,
                    blocked_reason=None,
                )

                self.unified_logger.log_decision_summary(log_entry)

            except Exception as e:
                # Fallback to simple log if unified logger fails
                self.logger.warning(f"Unified logger failed for {instrument}: {e}")
                self.logger.info(
                    format_operator_message(
                        icon="[DECISION]",
                        message=decision.value,
                        instrument=instrument,
                        size_eur=f"{size:.2f}",
                        confidence=f"{confidence:.1%}",
                    )
                )

        # Debugger (CSV/JSON) – forensic only
        debug_ctx = {
            "volatility": context.volatility,
            "current_exposure": context.current_exposure,
            "drawdown": context.drawdown,
            "balance": context.balance,
        }

        if hasattr(self, "debugger") and self.debugger.enabled and decision != PositionDecision.HOLD:
            try:
                self.debugger.log_decision(
                    instrument=instrument,
                    decision=decision.value,
                    intensity=intensity,
                    size=size,
                    confidence=confidence,
                    context=debug_ctx,
                    rationale=rationale,
                    portfolio_health=self._portfolio_health_score,
                )
            except Exception:
                pass

        return PositionDecisionResult(
            decision=decision,
            intensity=intensity,
            size=float(size),
            confidence=confidence,
            rationale=rationale,
            risk_factors=risk_factors,
            context=context,
        )

    # ==========================================================
    # Context extraction (lean + robust)
    # ==========================================================
    def _extract_signal_context(
        self, instrument: str, market_data: Dict[str, Any], portfolio_health: Dict[str, float]
    ) -> SignalContext:
        inst_data = market_data.get(instrument, {}) or {}

        # ------------------------------------------------------
        # Signal extraction priority:
        # 1. inst_data["intensity"] (from instrument_signals via SmartBus)
        # 2. agent_action multi-instrument vector
        # 3. Default to 0.0
        # ------------------------------------------------------
        raw_intensity = 0.0

        # Priority 1: pre-parsed intensity from instrument_signals (via position_base)
        if "intensity" in inst_data:
            raw_intensity = inst_data.get("intensity", 0.0)

        # Priority 2: fallback to agent_action with multi-instrument parsing
        if abs(raw_intensity) < 1e-6:
            try:
                agent_action = self.smart_bus.get("agent_action", "PositionManager")
                if isinstance(agent_action, (list, tuple, np.ndarray)):
                    action_arr = np.asarray(agent_action, dtype=np.float32).flatten()
                    n_instruments = len(self.instruments)

                    try:
                        inst_idx = self.instruments.index(instrument)
                    except ValueError:
                        inst_idx = -1

                    if inst_idx >= 0 and action_arr.size > 0:
                        if action_arr.size >= 2 * n_instruments:
                            contiguous = action_arr[:n_instruments]
                            interleaved = action_arr[0 : 2 * n_instruments : 2]
                            if np.mean(np.abs(contiguous)) >= np.mean(np.abs(interleaved)):
                                raw_intensity = float(contiguous[inst_idx])
                            else:
                                raw_intensity = float(interleaved[inst_idx])
                        elif action_arr.size >= n_instruments:
                            raw_intensity = float(action_arr[inst_idx])
                        elif inst_idx < action_arr.size:
                            raw_intensity = float(action_arr[inst_idx])
            except Exception:
                pass

        market_intensity = float(
            raw_intensity if isinstance(raw_intensity, (int, float)) else 0.0
        )
        market_direction = int(np.sign(market_intensity))

        vol_floor = float(self.Cval("min_volatility", 0.015))
        volatility = float(inst_data.get("volatility", vol_floor) or vol_floor)
        volatility = max(volatility, vol_floor)

        trend_strength = float(inst_data.get("trend_strength", 0.0) or 0.0)
        momentum = float(inst_data.get("momentum", 0.0) or 0.0)
        volume_profile = float(inst_data.get("volume_profile", 1.0) or 1.0)

        regime = market_data.get("market_regime", None)
        if regime is None:
            mc = self.smart_bus.get("market_regime", "PositionManager")
            regime = mc or "normal"

        market_conditions = self.smart_bus.get("market_conditions", "PositionManager") or {}
        session = inst_data.get("session") or market_conditions.get("session", "unknown")

        # Correlation penalty (if available)
        correlation_penalty = 0.0
        correlation_data = self.smart_bus.get("correlation_matrix", "PositionManager")
        if isinstance(correlation_data, dict):
            try:
                node = correlation_data.get(instrument)
                if isinstance(node, dict) and "avg_correlation" in node:
                    ac = node.get("avg_correlation")
                    if isinstance(ac, (int, float)):
                        correlation_penalty = min(abs(float(ac)) * 0.5, 0.8)
            except Exception:
                correlation_penalty = 0.0

        # Liquidity
        liquidity_score = self._get_liquidity(instrument)

        balance_val = float(portfolio_health.get("balance", self.C.initial_balance))
        dd_val = float(portfolio_health.get("drawdown", 0.0))
        exposure_ratio = float(portfolio_health.get("exposure_ratio", 0.0))

        # Get current price (for logging / context)
        current_price = 0.0
        try:
            price_data = self.smart_bus.get("price_data", "PositionManager") or {}
            inst_price = price_data.get(instrument, {})
            if isinstance(inst_price, dict):
                current_price = float(inst_price.get("last", inst_price.get("close", 0.0)))
            elif isinstance(inst_price, (int, float)):
                current_price = float(inst_price)
        except Exception:
            pass

        return SignalContext(
            instrument=instrument,
            market_intensity=market_intensity,
            market_direction=market_direction,
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            volume_profile=volume_profile,
            correlation_penalty=correlation_penalty,
            regime=str(regime),
            liquidity_score=liquidity_score,
            session=session,
            current_exposure=exposure_ratio,
            drawdown=dd_val,
            balance=balance_val,
            current_price=current_price,
            step_idx=0,
            timestamp=self._utc_stamp(),
        )

    def _utc_stamp(self) -> str:
        import datetime as _dt

        return _dt.datetime.utcnow().isoformat() + "Z"

    # ==========================================================
    # Risk & confidence
    # ==========================================================
    def _calculate_confidence(self, context: SignalContext, decision: PositionDecision) -> float:
        base_confidence = 0.5
        signal_conf = min(abs(context.market_intensity) * 1.2, 0.4)
        trend_conf = min(abs(context.trend_strength) * 0.3, 0.2)
        vol_penalty = min(context.volatility / 0.05, 0.2)
        health_boost = self._calculate_portfolio_health_score(context) * 0.2
        decision_adj = 0.0
        if decision in [PositionDecision.CLOSE, PositionDecision.EMERGENCY_CLOSE]:
            decision_adj = 0.1
        elif decision == PositionDecision.SCALE_DOWN:
            decision_adj = 0.05
        total = (
            base_confidence
            + signal_conf
            + trend_conf
            - vol_penalty
            + health_boost
            + decision_adj
        )
        return float(np.clip(total, 0.1, 1.0))

    def _calculate_portfolio_health_score(self, context: SignalContext) -> float:
        """
        Lightweight per-context health score.

        Combines the portfolio-wide health computed in the base class with
        inexpensive, local penalties from the provided context.
        """
        try:
            base_health = float(getattr(self, "_portfolio_health_score", 1.0))

            dd_penalty = max(0.0, 1.0 - float(np.clip(context.drawdown, 0.0, 1.0)) * 1.5)
            conc_cap = float(self.Cval("max_instrument_concentration", 0.30))
            conc_ratio = float(context.current_exposure) / max(conc_cap, 1e-9)
            conc_penalty = max(0.0, 1.0 - float(np.clip(conc_ratio, 0.0, 2.0)))

            score = (base_health * 0.7) + (dd_penalty * 0.15) + (conc_penalty * 0.15)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return float(np.clip(float(getattr(self, "_portfolio_health_score", 1.0)), 0.0, 1.0))

    def _assess_risk_factors(self, context: SignalContext) -> Dict[str, float]:
        rf: Dict[str, float] = {}
        vol_base = float(self.Cval("min_volatility", 0.015))
        vol_excess = max(0.0, float(context.volatility) - vol_base)
        rf["volatility"] = float(np.clip(vol_excess / 0.05, 0.0, 0.5))

        rf["correlation"] = float(np.clip(abs(context.correlation_penalty), 0.0, 1.0))
        rf["drawdown"] = float(np.clip(context.drawdown * 2.0, 0.0, 0.8))

        denom = max(float(self.Cval("max_instrument_concentration", 0.30)), 1e-9)
        rf["concentration"] = float(np.clip(context.current_exposure / denom, 0.0, 0.9))

        if context.session == "closed":
            session_risk = 0.3
        elif context.session == "asian":
            session_risk = 0.1
        else:
            session_risk = 0.0
        rf["session"] = session_risk

        return rf

    # ==========================================================
    # Sizing
    # ==========================================================
    def _calculate_position_size(
        self, context: SignalContext, intensity: float, confidence: float
    ) -> float:
        return self.calculate_size(
            volatility=context.volatility,
            intensity=float(np.clip(intensity, -1.0, 1.0)),
            balance=context.balance,
            drawdown=context.drawdown,
            correlation=context.correlation_penalty,
            current_exposure=context.current_exposure,
        )

    def calculate_size(
        self,
        volatility: float,
        intensity: float,
        balance: float,
        drawdown: float,
        correlation: Optional[float] = None,
        current_exposure: Optional[float] = None,
    ) -> float:
        vol_floor = float(self.Cval("min_volatility", 0.015))
        volatility = max(float(np.nan_to_num(volatility, nan=vol_floor)), vol_floor)
        intensity = float(np.nan_to_num(np.clip(intensity, -1.0, 1.0), nan=0.0))
        balance = max(float(balance), 100.0)
        drawdown = float(np.nan_to_num(drawdown, nan=0.0))

        # Read max position % directly from risk_policy.yaml -> lot_sizing.max_exposure_pct
        # This aligns with PortfolioRiskSystem's enforcement limit (single source of truth)
        default_max_pct = 0.05  # 5% fallback
        try:
            from pathlib import Path
            import yaml
            risk_policy_path = Path("config/risk_policy.yaml")
            if risk_policy_path.exists():
                with open(risk_policy_path, "r", encoding="utf-8") as f:
                    policy = yaml.safe_load(f) or {}
                lot_sizing = policy.get("lot_sizing", {})
                if lot_sizing.get("max_exposure_pct") is not None:
                    default_max_pct = float(lot_sizing["max_exposure_pct"])
        except Exception:
            pass

        risk_pct = max(
            float(
                self._adaptive_params.get(
                    "dynamic_max_pct",
                    self.Cval("max_position_pct", default_max_pct),
                )
            ),
            0.01,
        )
        risk_budget = balance * risk_pct
        vol_adjusted_budget = risk_budget / volatility
        base_size = intensity * vol_adjusted_budget

        # Health & tolerance modifiers
        portfolio_health = self._portfolio_health_score
        adjusted_size = (
            base_size
            * max(0.1, portfolio_health)
            * float(self._adaptive_params.get("risk_tolerance", 1.0))
        )

        # Trading mode risk multiplier (if available)
        try:
            mode_config = self.smart_bus.get("mode_config", "PositionManager") or {}
            risk_multiplier = float(mode_config.get("risk_multiplier", 1.0))
            adjusted_size *= risk_multiplier
            if self.debug and risk_multiplier != 1.0:
                self.logger.info(
                    format_operator_message(
                        icon="🎛️",
                        message="Trading mode risk adjustment applied",
                        multiplier=f"{risk_multiplier:.2f}x",
                    )
                )
        except Exception:
            pass

        # Correlation penalty
        if correlation is not None:
            corr_penalty = 1.0 - min(abs(float(correlation)) * 0.3, 0.5)
            adjusted_size *= corr_penalty

        # Loss-streak brake
        if self.consecutive_losses >= self.Cval("max_consecutive_losses", 5):
            streak_reduction = max(0.1, float(self.Cval("loss_reduction", 0.5)))
            adjusted_size *= streak_reduction
            if self.debug:
                self.logger.info(
                    format_operator_message(
                        "SELL",
                        "LOSS_STREAK_REDUCTION",
                        reduction_factor=f"{streak_reduction:.2f}",
                        consecutive_losses=self.consecutive_losses,
                    )
                )

        # MEMORY INTEGRATION: Apply memory risk multiplier
        try:
            memory_gate = self.smart_bus.get("memory_gate", "PositionManager")
            if isinstance(memory_gate, dict):
                mem_risk_mult = float(memory_gate.get("risk_multiplier", 1.0))
                if mem_risk_mult < 1.0:
                    adjusted_size *= mem_risk_mult
                    if self.debug:
                        self.logger.info(
                            format_operator_message(
                                icon="🧠",
                                message="MEMORY_SIZE_ADJUSTMENT",
                                multiplier=f"{mem_risk_mult:.2f}x",
                                reasons=memory_gate.get("reasons", [])[:2],
                            )
                        )
        except Exception:
            pass

        # PPO INTELLIGENT ARBITER: Apply PPO position sizing recommendation
        try:
            ppo_position_size = self.smart_bus.get("ppo_position_size", "PositionManager")
            if ppo_position_size is not None and ppo_position_size > 0:
                # PPOAgent provides a multiplier (0.0 to 1.0) based on confidence
                ppo_mult = float(np.clip(ppo_position_size, 0.0, 1.0))
                adjusted_size *= ppo_mult
                if self.debug:
                    self.logger.info(
                        format_operator_message(
                            icon="🤖",
                            message="PPO_ARBITER_SIZE_ADJUSTMENT",
                            multiplier=f"{ppo_mult:.2f}x",
                            source="intelligent arbiter confidence scaling",
                        )
                    )
        except Exception:
            pass

        abs_size = abs(adjusted_size)
        min_viable_size = balance * float(self.Cval("min_size_pct", 0.01))

        if abs_size < min_viable_size and abs(intensity) > 0.3:
            adjusted_size = np.sign(adjusted_size or intensity) * min_viable_size
        elif abs_size < min_viable_size:
            adjusted_size = 0.0

        max_single_position = balance * risk_pct
        final_size = float(np.clip(adjusted_size, -max_single_position, max_single_position))
        return float(np.nan_to_num(final_size, nan=0.0, posinf=0.0, neginf=0.0))

    def _get_liquidity(self, instrument: str) -> float:
        """
        Fetch a normalized liquidity score [0.0, 1.0] for an instrument.

        Tries several SmartBus keys commonly published by the Market module,
        with graceful fallbacks.
        """
        try:
            liq_map = self.smart_bus.get("liquidity_score_by_instrument", "PositionManager")
            if isinstance(liq_map, dict):
                v = liq_map.get(instrument)
                if isinstance(v, (int, float)):
                    return float(np.clip(v, 0.0, 1.0))

            v = self.smart_bus.get("liquidity_score", "PositionManager")
            if isinstance(v, (int, float)):
                return float(np.clip(v, 0.0, 1.0))
            if isinstance(v, dict):
                cand = v.get(instrument)
                if isinstance(cand, (int, float)):
                    return float(np.clip(cand, 0.0, 1.0))
                cand = v.get("score") or v.get("value")
                if isinstance(cand, (int, float)):
                    return float(np.clip(cand, 0.0, 1.0))

            mc = self.smart_bus.get("market_conditions", "PositionManager") or {}
            if isinstance(mc, dict):
                lv = mc.get("liquidity_score")
                if isinstance(lv, (int, float)):
                    return float(np.clip(lv, 0.0, 1.0))
        except Exception:
            pass

        return 0.5

    # ==========================================================
    # Memory Intelligence Integration
    # ==========================================================
    def _get_memory_intelligence(self) -> Dict[str, Any]:
        """
        Gather actionable intelligence from the unified memory system.

        Returns a consolidated dict with:
        - Gate signals (veto, risk_multiplier)
        - Danger zone detection
        - Expected PnL from similar trades
        - Avoidance signals and playbook insights.
        """
        result: Dict[str, Any] = {
            "veto": False,
            "risk_multiplier": 1.0,
            "veto_reasons": [],
            "in_danger_zone": False,
            "danger_similarity": 0.0,
            "avoidance_signal": 0.0,
            "expected_pnl": 0.0,
            "playbook_confidence": 0.5,
            "signed_bias": 0.0,
            "neural_risk_hint": 0.5,
            "loss_prob": 0.0,
            "intervention_type": "none",
            "intervention_strength": 0.0,
        }

        try:
            # 1) memory_gate - Primary gate signal (veto + size control)
            memory_gate = self.smart_bus.get("memory_gate", "PositionManager")
            if isinstance(memory_gate, dict):
                result["veto"] = bool(memory_gate.get("veto", False))
                result["risk_multiplier"] = float(memory_gate.get("risk_multiplier", 1.0))
                result["veto_reasons"] = memory_gate.get("reasons", [])
                result["danger_similarity"] = float(memory_gate.get("danger_similarity", 0.0))
                result["loss_prob"] = float(memory_gate.get("loss_prob", 0.0))

            # 2) memory_vote - Ensemble contribution
            memory_vote = self.smart_bus.get("memory_vote", "PositionManager")
            if isinstance(memory_vote, dict):
                result["signed_bias"] = float(memory_vote.get("signed_bias", 0.0))
                result["expected_pnl"] = float(memory_vote.get("expected_pnl", 0.0))
                result["playbook_confidence"] = float(memory_vote.get("confidence", 0.5))
                result["neural_risk_hint"] = float(memory_vote.get("neural_risk_hint", 0.5))

            # 3) danger_zones - Direct danger zone check
            danger_zones = self.smart_bus.get("danger_zones", "PositionManager")
            if isinstance(danger_zones, dict):
                zones = danger_zones.get("zones", [])
                zone_count = danger_zones.get("zone_count", 0)
                if zone_count > 0 or len(zones) > 0:
                    result["in_danger_zone"] = True
                    max_sim = 0.0
                    for z in zones[:5]:
                        if isinstance(z, dict):
                            max_sim = max(max_sim, float(z.get("similarity", 0.0)))
                    result["danger_similarity"] = max(result["danger_similarity"], max_sim)

            # 4) mistake_avoidance - Avoidance signal
            mistake_avoidance = self.smart_bus.get("mistake_avoidance", "PositionManager")
            if isinstance(mistake_avoidance, dict):
                result["avoidance_signal"] = float(
                    mistake_avoidance.get("avoidance_signal", 0.0)
                )
                if result["avoidance_signal"] > 0.6:
                    result["in_danger_zone"] = True

            # 5) playbook_recall - Historical pattern insights
            playbook_recall = self.smart_bus.get("playbook_recall", "PositionManager")
            if isinstance(playbook_recall, dict):
                if result["expected_pnl"] == 0.0:
                    result["expected_pnl"] = float(playbook_recall.get("expected_pnl", 0.0))
                if result["playbook_confidence"] == 0.5:
                    result["playbook_confidence"] = float(
                        playbook_recall.get("confidence", 0.5)
                    )

            # 6) intuition_vector - Compressed pattern intelligence
            intuition = self.smart_bus.get("intuition_vector", "PositionManager")
            if isinstance(intuition, dict):
                strength = float(intuition.get("strength", 0.0))
                if strength < -0.3:
                    result["avoidance_signal"] = max(
                        result["avoidance_signal"],
                        abs(strength),
                    )

            # 7) loss_prevention - Loss prevention metrics
            loss_prevention = self.smart_bus.get("loss_prevention", "PositionManager")
            if isinstance(loss_prevention, dict):
                effectiveness = float(
                    loss_prevention.get("avoidance_effectiveness", 0.0)
                )
                if effectiveness > 0.5:
                    result["risk_multiplier"] *= (1.0 - effectiveness * 0.3)

            if self.debug and (
                result["veto"]
                or result["in_danger_zone"]
                or result["avoidance_signal"] > 0.3
            ):
                self.logger.info(
                    format_operator_message(
                        icon="🧠",
                        message="MEMORY_INTELLIGENCE",
                        veto=result["veto"],
                        danger_zone=result["in_danger_zone"],
                        danger_sim=f"{result['danger_similarity']:.2f}",
                        avoidance=f"{result['avoidance_signal']:.2f}",
                        risk_mult=f"{result['risk_multiplier']:.2f}",
                        expected_pnl=f"{result['expected_pnl']:.2f}",
                    )
                )

        except Exception as e:
            if self.debug:
                self.logger.warning(f"Memory intelligence fetch failed: {e}")

        return result

    # ==========================================================
    # Unified Exit Strategy Evaluation
    # ==========================================================
    def _evaluate_exit_strategies(
        self,
        instrument: str,
        pos_data: Optional[Dict[str, Any]],
        position_side: int,
        context: SignalContext,
    ) -> ExitDecision:
        """
        Evaluate all exit strategies using the unified ExitStrategyEngine.

        This ensures consistency with SmartPositionManager (live trading).
        Both systems now use the same exit logic from risk_policy.yaml.
        """
        exit_engine = get_exit_engine()

        # Extract position data
        unrealized_pnl = 0.0
        entry_price = 0.0
        open_time = time.time() - 3600  # Default to 1 hour ago
        lots = 0.0
        position_id = ""

        if isinstance(pos_data, dict):
            unrealized_pnl = float(
                pos_data.get("unrealized_pnl", pos_data.get("pnl", 0.0)) or 0.0
            )
            entry_price = float(
                pos_data.get("entry_price", pos_data.get("price_open", 0.0)) or 0.0
            )
            # Robust open_time parsing - handle various formats
            raw_open_time = pos_data.get("open_time", pos_data.get("time"))
            if raw_open_time is None:
                open_time = time.time() - 3600  # Default 1 hour ago
            elif isinstance(raw_open_time, (int, float)):
                open_time = float(raw_open_time)
            elif isinstance(raw_open_time, str):
                # Try parsing ISO format or numeric string
                try:
                    open_time = float(raw_open_time)
                except ValueError:
                    try:
                        import datetime
                        # Handle ISO format like "2025-12-04T10:30:00"
                        dt = datetime.datetime.fromisoformat(raw_open_time.replace("Z", "+00:00"))
                        open_time = dt.timestamp()
                    except Exception:
                        open_time = time.time() - 3600
            else:
                open_time = time.time() - 3600
            lots = float(
                pos_data.get("lots", pos_data.get("volume", pos_data.get("units", 0.0))) or 0.0
            )
            position_id = str(pos_data.get("ticket", ""))

        # Get tracked peak from _profit_tracker for trailing profit logic
        # This is the actual peak PnL seen since position opened
        tracked_peak = self._profit_tracker.peak(instrument)
        if tracked_peak <= 0.0:
            tracked_peak = max(unrealized_pnl, 0.0)  # Fallback to current if no peak tracked
        
        # Engine peak for logging comparison
        engine_peak = exit_engine.get_peak(instrument)
        if engine_peak is None or engine_peak == 0.0:
            engine_peak = tracked_peak

        # Get current price
        current_price = context.current_price if hasattr(context, "current_price") else 0.0
        if current_price == 0.0:
            try:
                price_data = self.smart_bus.get("price_data", "PositionManager") or {}
                inst_price = price_data.get(instrument, {})
                if isinstance(inst_price, dict):
                    current_price = float(inst_price.get("last", inst_price.get("close", 0.0)))
                elif isinstance(inst_price, (int, float)):
                    current_price = float(inst_price)
            except Exception:
                pass

        # Get ATR for dynamic trailing
        atr = None
        try:
            market_data = self.smart_bus.get("market_data", "PositionManager") or {}
            inst_data = market_data.get(instrument, {})
            if isinstance(inst_data, dict):
                atr = inst_data.get("atr") or inst_data.get("ATR")

            if atr is None:
                indicators = self.smart_bus.get("technical_indicators", "PositionManager") or {}
                inst_ind = indicators.get(instrument, {})
                if isinstance(inst_ind, dict):
                    atr = inst_ind.get("atr") or inst_ind.get("ATR")
        except Exception:
            pass

        # Get market regime
        regime = context.regime if hasattr(context, "regime") else "normal"
        if not regime:
            regime = self.smart_bus.get("market_regime", "PositionManager", default="normal") or "normal"

        # Get account-level context for EMERGENCY exits
        account_drawdown_pct = context.drawdown if hasattr(context, "drawdown") else None
        daily_loss_eur = None
        daily_loss_limit_eur = None
        total_open_risk_eur = None
        try:
            # Daily P&L from bus
            daily_pnl = self.smart_bus.get("daily_pnl", "PositionManager")
            if daily_pnl is not None:
                daily_loss_eur = float(daily_pnl)

            # Daily loss limit
            risk_limits = self.smart_bus.get("risk_limits", "PositionManager") or {}
            if isinstance(risk_limits, dict):
                daily_loss_limit_eur = risk_limits.get("daily_loss_limit_eur")
                if daily_loss_limit_eur is not None:
                    daily_loss_limit_eur = float(daily_loss_limit_eur)

            # Total open risk (sum of negative unrealized P&L)
            positions = self.smart_bus.get("positions", "PositionManager") or {}
            if isinstance(positions, dict):
                total_open_risk_eur = sum(
                    abs(float(p.get("unrealized_pnl", 0) or 0))
                    for p in positions.values()
                    if isinstance(p, dict)
                    and float(p.get("unrealized_pnl", 0) or 0) < 0
                )
        except Exception:
            pass

        # Consensus confidence from voting if available
        consensus_confidence = float(min(max(abs(context.market_intensity), 0.0), 1.0))
        try:
            trade_vote = self.smart_bus.get("trade_vote_v2", "PositionManager")
            if isinstance(trade_vote, dict):
                conf = trade_vote.get("confidence") or trade_vote.get("consensus_confidence")
                if conf is not None:
                    consensus_confidence = float(conf)
        except Exception:
            pass

        # Build position context for exit engine.
        # peak_pnl uses tracked peak from _profit_tracker for proper trailing profit logic
        pos_ctx = PositionContext(
            symbol=instrument,
            side=position_side,
            unrealized_pnl=unrealized_pnl,
            peak_pnl=tracked_peak,
            entry_price=entry_price,
            current_price=current_price,
            open_time=open_time,
            lots=lots,
            position_id=position_id,
            atr=float(atr) if atr is not None else None,
            volatility=context.volatility,
            regime=str(regime),
            signal_direction=context.market_direction,
            signal_strength=abs(context.market_intensity),
            consensus_confidence=consensus_confidence,
            account_drawdown_pct=account_drawdown_pct,
            daily_loss_eur=daily_loss_eur,
            daily_loss_limit_eur=daily_loss_limit_eur,
            total_open_risk_eur=total_open_risk_eur,
        )

        # Evaluate all exit strategies
        exit_decision = exit_engine.evaluate(pos_ctx)

        # Log exit evaluation (non-HOLD only)
        if self.debug and exit_decision.should_exit:
            self.logger.info(
                format_operator_message(
                    icon="🚪",
                    message="EXIT_TRIGGERED",
                    instrument=instrument,
                    reason=exit_decision.reason.name,
                    confidence=f"{exit_decision.confidence:.2f}",
                    urgency=f"{exit_decision.urgency:.2f}",
                    pnl=f"€{unrealized_pnl:.2f}",
                    peak=f"€{engine_peak:.2f}",
                    details=exit_decision.details.get("message", "")[:80],
                )
            )

        return exit_decision

    # ==========================================================
    # Emergency conditions
    # ==========================================================
    # ==========================================================
    # Emergency conditions
    # ==========================================================
    def _check_emergency_conditions(self, context: SignalContext) -> bool:
        """
        Fast local emergency gate (per-tick).

        This is intentionally stricter and earlier than the account-level
        EMERGENCY logic inside ExitStrategyEngine:

        - Here we protect *per-tick* against:
            * too much exposure on a single instrument,
            * account drawdown getting close to prop limits,
            * long loss streaks,
            * trading in very illiquid conditions.

        Thresholds are read from risk_policy.position_manager and fall back
        to prop-firm friendly defaults if not configured.
        """
        # Prop-firm style defaults; will be overridden by risk_policy.yaml:
        #   position_manager.emergency_exposure_trigger: 0.20
        #   position_manager.emergency_drawdown_trigger: 0.035
        #   position_manager.max_consecutive_losses: 3
        emergency_exposure_threshold = float(
            self.Cval("emergency_exposure_trigger", 0.20)
        )
        drawdown_trigger = float(
            self.Cval("emergency_drawdown_trigger", 0.035)
        )
        max_losses = int(
            self.Cval("max_consecutive_losses", 3)
        )
        liquidity_floor = float(
            self.Cval("emergency_liquidity_threshold", 0.30)
        )

        triggers = {
            "drawdown": context.drawdown >= drawdown_trigger,
            "loss_streak": self.consecutive_losses >= max_losses,
            "exposure": context.current_exposure >= emergency_exposure_threshold,
            "liquidity": context.liquidity_score <= liquidity_floor,
        }

        active = bool(
            triggers["drawdown"]
            or triggers["loss_streak"]
            or triggers["exposure"]
            or triggers["liquidity"]
        )
        if not active:
            return False

        # Detect executor activity to prevent noisy emergencies when nothing can act
        executor_active = False
        try:
            md = getattr(self.smart_bus, "get_with_metadata", None)
            keys = ("execution_data", "executor_debug", "positions")
            for key in keys:
                if md is not None:
                    rec = md(key, "PositionManager")
                    if (
                        rec
                        and rec.value is not None
                        and hasattr(rec, "age_seconds")
                        and rec.age_seconds() < 10.0
                    ):
                        executor_active = True
                        break
                else:
                    val = self.smart_bus.get(key, "PositionManager")
                    if val:
                        executor_active = True
                        break
        except Exception:
            pass

        suppress = bool(self.Cval("suppress_emergency_without_executor", True)) and not executor_active

        # Log once per approximate-tick (based on timestamp)
        stamp = getattr(self, "_last_emergency_diag_stamp", None)
        current_stamp = context.timestamp or f"t_{int(time.time())}"
        if stamp != current_stamp:
            try:
                self.logger.warning(
                    format_operator_message(
                        icon="[ALERT]" if not suppress else "[INFO]",
                        message="Emergency condition evaluated"
                        + (" (SUPPRESSED)" if suppress else ""),
                        drawdown=f"{context.drawdown:.4f}",
                        drawdown_trigger=drawdown_trigger,
                        consecutive_losses=self.consecutive_losses,
                        loss_streak_trigger=triggers["loss_streak"],
                        exposure=f"{context.current_exposure:.4f}",
                        exposure_trigger=triggers["exposure"],
                        exposure_threshold=emergency_exposure_threshold,
                        liquidity=f"{context.liquidity_score:.3f}",
                        liquidity_trigger=triggers["liquidity"],
                        liquidity_floor=liquidity_floor,
                        executor_active=executor_active,
                        suppressed=suppress,
                    )
                )
            except Exception:
                pass
            self._last_emergency_diag_stamp = current_stamp

        return False if suppress else True

        