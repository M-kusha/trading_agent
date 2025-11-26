# -------------------------------------------------------------
# File: modules/position/position_logic.py
# PositionManager — decision logic + sizing + profit rules
# (subclasses PositionManagerBase from position_base.py)
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


@module(
    **module_args(
        "PositionManager",
        description="Position decision maker with trailing take-profit & hard loss-cut; Env/Executor handle execution.",
        error_handling=True,
        hot_reload=True,
        timeout_ms=3000,
    )
)
class PositionManager(PositionManagerBase):
    """
    Decision layer:
      - Opens only on adequate signal.
      - Scales-up conservatively (cooldown + exposure checks).
      - Hard loss-cut at -C.hard_loss_eur (default: 100 EUR).
      - Trailing take-profit: activate above C.take_profit_min_eur and close on
        ≥ C.take_profit_trailing_pct retrace from P&L peak *when favors degrade*.
    """

    # ==========================================================
    # Public pipeline hooks (called by PositionManagerBase)
    # ==========================================================
    @create_error_handler("process_market_signals")
    def process_market_signals(self, market_data: Dict[str, Any]) -> Dict[str, PositionDecisionResult]:
        decisions: Dict[str, PositionDecisionResult] = {}

        # Portfolio assessment
        portfolio_health = self._assess_portfolio_health()
        market_regime = self._assess_market_regime(market_data)

        # Log portfolio stats with unified logger
        if self.debug and hasattr(self, 'unified_logger'):
            self.unified_logger.log_portfolio_stats(portfolio_health)

        # Per-instrument decisions
        for instrument in self.instruments:
            ctx = self._extract_signal_context(instrument, market_data, portfolio_health)
            # update short signal history (for favorability slope)
            self.signal_history[instrument].append(ctx.market_intensity)
            if len(self.signal_history[instrument]) > 50:
                self.signal_history[instrument].pop(0)

            # Maintain trailing P&L peak
            self.update_profit_tracker(instrument)

            dr = self._make_position_decision(ctx)
            decisions[instrument] = dr
            self.last_decisions[instrument] = dr

        if self.debug:
            self._flush_logs()

        return decisions

    # ==========================================================
    # Core decision logic
    # ==========================================================
    def _make_position_decision(self, context: SignalContext) -> PositionDecisionResult:
        """
        Simplified decision logic - ENTRY DECISIONS ONLY.
        
        Exit logic (profit-taking, loss-cutting, trailing stops) is handled
        by SmartPositionManager in the Executor for live trading.
        
        This module focuses on:
        1. Memory veto/danger zone checks (pre-entry filtering)
        2. Emergency conditions
        3. Signal-based entry decisions (OPEN_LONG, OPEN_SHORT)
        4. Portfolio health gates
        """
        instrument = context.instrument
        has_position = instrument in self.open_positions

        decision = PositionDecision.HOLD
        intensity = 0.0
        size = 0.0
        confidence = 0.5
        rationale: Dict[str, Any] = {"stage": "initial", "factors": []}
        risk_factors: Dict[str, float] = {}

        # ==========================================================
        # MEMORY INTEGRATION: Read all memory signals upfront
        # ==========================================================
        memory_data = self._get_memory_intelligence()
        
        # ---------- MEMORY VETO CHECK (highest priority gate)
        if memory_data.get("veto", False) and not has_position:
            rationale["stage"] = "memory_veto"
            rationale["factors"].extend(memory_data.get("veto_reasons", ["Memory system vetoed this trade"]))
            rationale["memory_data"] = memory_data
            return self._finalize_decision(instrument, decision, 0.0, 0.0, 0.3, rationale, risk_factors, context)

        # ---------- Fast emergency gate (only closes / reduces)
        if self._check_emergency_conditions(context):
            if has_position:
                decision = PositionDecision.EMERGENCY_CLOSE
                intensity = 1.0
                confidence = 0.9
                rationale["stage"] = "emergency"
                rationale["factors"].append("Emergency conditions detected")
                return self._finalize_decision(instrument, decision, intensity, size, confidence, rationale, risk_factors, context)

            # No position: just hold if emergency with no exposure
            rationale["stage"] = "emergency_hold"
            rationale["factors"].append("Emergency conditions; no open position")
            return self._finalize_decision(instrument, decision, 0.0, 0.0, 0.6, rationale, risk_factors, context)

        # ---------- MEMORY DANGER ZONE CHECK (before opening new positions)
        if not has_position and memory_data.get("in_danger_zone", False):
            danger_similarity = memory_data.get("danger_similarity", 0.0)
            if danger_similarity > 0.7:
                rationale["stage"] = "memory_danger_zone"
                rationale["factors"].append(f"Memory danger zone: {danger_similarity:.1%} similarity to past losses")
                rationale["memory_data"] = memory_data
                return self._finalize_decision(instrument, decision, 0.0, 0.0, 0.4, rationale, risk_factors, context)

        # ---------- Signal & portfolio gates
        min_sig = float(self.Cval("min_signal_threshold", 0.20))
        sig_strength = abs(context.market_intensity)

        # Portfolio health brake for opening new exposure
        portfolio_health_score = self._calculate_portfolio_health_score(context)
        health_floor = 0.30
        if not has_position and portfolio_health_score < health_floor:
            rationale["stage"] = "portfolio_health"
            rationale["factors"].append(f"Portfolio health {portfolio_health_score:.3f} too low to open")
            return self._finalize_decision(instrument, decision, 0.0, 0.0, 0.5, rationale, risk_factors, context)

        # ==========================================================
        # EXISTING POSITION: Let SmartPositionManager handle exits
        # ==========================================================
        if has_position:
            # Just HOLD - exit logic is in SmartPositionManager (Executor)
            # We only emit EMERGENCY_CLOSE (above) for critical situations
            rationale["stage"] = "hold_existing"
            rationale["factors"].append("Position exists; exits handled by SmartPositionManager")
            return self._finalize_decision(instrument, decision, 0.0, 0.0, 0.6, rationale, risk_factors, context)

        # ==========================================================
        # NO POSITION: Evaluate entry signals
        # ==========================================================
        if sig_strength < min_sig:
            rationale["stage"] = "signal_filter"
            rationale["factors"].append(f"Signal {sig_strength:.3f} below threshold {min_sig:.2f}")
            return self._finalize_decision(instrument, decision, 0.0, 0.0, 0.5, rationale, risk_factors, context)

        # Use voting direction when available, fallback to signal direction
        trade_vote = self.smart_bus.get("trade_vote_v2", "PositionManager")
        voting_direction = None
        if isinstance(trade_vote, dict) and trade_vote.get("action") in ("BUY", "buy", "SELL", "sell"):
            vote_action = str(trade_vote.get("action", "")).upper()
            voting_direction = 1 if vote_action == "BUY" else -1
            rationale["factors"].append(f"Using voting direction: {vote_action}")
        
        # Use voting direction if available and confident, else fallback to signal
        effective_direction = voting_direction if voting_direction is not None else context.market_direction
        decision = PositionDecision.OPEN_LONG if effective_direction > 0 else PositionDecision.OPEN_SHORT
        intensity = sig_strength
        confidence = self._calculate_confidence(context, decision)
        size = self._calculate_position_size(context, intensity, confidence)

        # Enforce min viable notional; otherwise hold
        min_size_pct = float(self.Cval("min_size_pct", 0.01))
        if abs(size) < context.balance * min_size_pct:
            if sig_strength > (min_sig + 0.10):
                size = context.balance * min_size_pct
            else:
                rationale["stage"] = "sizing"
                rationale["factors"].append("Computed size below minimum; holding")
                decision = PositionDecision.HOLD
                intensity = 0.0
                size = 0.0
                return self._finalize_decision(instrument, decision, intensity, size, confidence, rationale, risk_factors, context)

        rationale["stage"] = "new_position"
        rationale["factors"].append(f"Strong signal {sig_strength:.3f} for new position")
        return self._finalize_decision(instrument, decision, intensity, size, confidence, rationale, risk_factors, context)

    # ==========================================================
    # Decision helpers
    # ==========================================================
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

        # Calculate risk score for unified logging
        risk_score = sum(risk_factors.values()) / max(len(risk_factors), 1) if risk_factors else 0.0

        # Get current price from context or market data
        current_price = 0.0
        try:
            # Try to get price from context first
            if hasattr(context, 'current_price') and context.current_price:
                current_price = context.current_price
            else:
                # Fall back to market data or smart bus
                inst_data = {}
                try:
                    market_data = self.smart_bus.get("price_data", "PositionManager") or {}
                    inst_data = market_data.get(instrument, {})
                    current_price = float(inst_data.get("last", inst_data.get("close", 0.0)))
                except:
                    pass
        except:
            pass

        # Unified logging for non-HOLD decisions
        if self.debug and decision != PositionDecision.HOLD and hasattr(self, 'unified_logger'):
            try:
                # Get voting signals
                voting_signals = self.unified_logger.get_voting_signals()

                # Create log entry
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
                    committee_consensus=voting_signals.get('committee_consensus'),
                    trade_vote=voting_signals.get('trade_vote'),
                    consensus_strength=voting_signals.get('consensus_strength'),
                    stage=rationale.get("stage", "unknown"),
                    factors=rationale.get("factors", []),
                    risk_factors=risk_factors,
                    will_execute=True,
                    blocked_reason=None
                )

                # Log the unified decision summary
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

        # Integrated debugger (CSV/JSON) - Only log to CSV for forensics, not to console/file
        # The unified logger above handles all human-readable output
        debug_ctx = {
            "volatility": context.volatility,
            "current_exposure": context.current_exposure,
            "drawdown": context.drawdown,
            "balance": context.balance,
        }

        # Only log to CSV if debugger is enabled
        if hasattr(self, 'debugger') and self.debugger.enabled and decision != PositionDecision.HOLD:
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

        raw_intensity = inst_data.get("intensity", 0.0)
        market_intensity = float(raw_intensity if isinstance(raw_intensity, (int, float)) else 0.0)
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

        # Get current price
        current_price = 0.0
        try:
            price_data = self.smart_bus.get("price_data", "PositionManager") or {}
            inst_price = price_data.get(instrument, {})
            if isinstance(inst_price, dict):
                current_price = float(inst_price.get("last", inst_price.get("close", 0.0)))
            elif isinstance(inst_price, (int, float)):
                current_price = float(inst_price)
        except:
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
        total = base_confidence + signal_conf + trend_conf - vol_penalty + health_boost + decision_adj
        return float(np.clip(total, 0.1, 1.0))

    def _calculate_portfolio_health_score(self, context: SignalContext) -> float:
        """Lightweight per-context health score.

        Combines the portfolio-wide health computed in the base class with
        a couple of inexpensive, local penalties from the provided context.

        This intentionally stays simple and robust: if upstream publishers
        haven't provided some values yet, we fall back to the last known
        portfolio score maintained by PositionManagerBase.
        """
        try:
            base_health = float(getattr(self, "_portfolio_health_score", 1.0))
            # Penalize for drawdown and concentration relative to configured caps.
            dd_penalty = max(0.0, 1.0 - float(np.clip(context.drawdown, 0.0, 1.0)) * 1.5)
            conc_cap = float(self.Cval("max_instrument_concentration", 0.30))
            conc_ratio = float(context.current_exposure) / max(conc_cap, 1e-9)
            conc_penalty = max(0.0, 1.0 - float(np.clip(conc_ratio, 0.0, 2.0)))

            # Blend: emphasize base health, lightly adjust with local context.
            score = (base_health * 0.7) + (dd_penalty * 0.15) + (conc_penalty * 0.15)
            return float(np.clip(score, 0.0, 1.0))
        except Exception:
            return float(np.clip(float(getattr(self, "_portfolio_health_score", 1.0)), 0.0, 1.0))

    def _assess_risk_factors(self, context: SignalContext) -> Dict[str, float]:
        rf: Dict[str, float] = {}
        rf["volatility"] = float(min((context.volatility - self.Cval("min_volatility", 0.015)) / 0.05, 0.5))
        rf["correlation"] = float(context.correlation_penalty)
        rf["drawdown"] = float(min(context.drawdown * 2.0, 0.8))
        denom = max(self.Cval("max_instrument_concentration", 0.30), 1e-9)
        rf["concentration"] = float(min(context.current_exposure / denom, 0.9))
        rf["session"] = 0.3 if context.session == "closed" else (0.1 if context.session == "asian" else 0.0)
        return rf

    # ==========================================================
    # Sizing
    # ==========================================================
    def _calculate_position_size(self, context: SignalContext, intensity: float, confidence: float) -> float:
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

        risk_pct = max(float(self._adaptive_params.get("dynamic_max_pct", self.Cval("max_position_pct", 0.10))), 0.01)
        risk_budget = balance * risk_pct
        vol_adjusted_budget = risk_budget / volatility
        base_size = intensity * vol_adjusted_budget

        # Health & tolerance modifiers
        portfolio_health = self._portfolio_health_score
        adjusted_size = base_size * max(0.1, portfolio_health) * float(self._adaptive_params.get("risk_tolerance", 1.0))

        # Trading mode risk multiplier (if available)
        try:
            mode_config = self.smart_bus.get('mode_config', 'PositionManager') or {}
            risk_multiplier = float(mode_config.get('risk_multiplier', 1.0))
            adjusted_size *= risk_multiplier
            if self.debug and risk_multiplier != 1.0:
                self.logger.info(format_operator_message(
                    icon="🎛️",
                    message="Trading mode risk adjustment applied",
                    multiplier=f"{risk_multiplier:.2f}x",
                ))
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

        # ==========================================================
        # MEMORY INTEGRATION: Apply memory risk multiplier
        # ==========================================================
        try:
            memory_gate = self.smart_bus.get('memory_gate', 'PositionManager')
            if isinstance(memory_gate, dict):
                mem_risk_mult = float(memory_gate.get('risk_multiplier', 1.0))
                if mem_risk_mult < 1.0:
                    adjusted_size *= mem_risk_mult
                    if self.debug:
                        self.logger.info(format_operator_message(
                            icon="🧠",
                            message="MEMORY_SIZE_ADJUSTMENT",
                            multiplier=f"{mem_risk_mult:.2f}x",
                            reasons=memory_gate.get('reasons', [])[:2],
                        ))
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
        """Fetch a normalized liquidity score [0.0, 1.0] for an instrument.

        Tries several SmartBus keys commonly published by the Market module,
        with graceful fallbacks. Keeps computation cheap and safe for hot paths.
        """
        try:
            # 1) Instrument-specific map
            liq_map = self.smart_bus.get("liquidity_score_by_instrument", "PositionManager")
            if isinstance(liq_map, dict):
                v = liq_map.get(instrument)
                if isinstance(v, (int, float)):
                    return float(np.clip(v, 0.0, 1.0))

            # 2) Direct score
            v = self.smart_bus.get("liquidity_score", "PositionManager")
            if isinstance(v, (int, float)):
                return float(np.clip(v, 0.0, 1.0))
            if isinstance(v, dict):
                # Some publishers encapsulate a value or per-instrument map
                cand = v.get(instrument)
                if isinstance(cand, (int, float)):
                    return float(np.clip(cand, 0.0, 1.0))
                cand = v.get("score") or v.get("value")
                if isinstance(cand, (int, float)):
                    return float(np.clip(cand, 0.0, 1.0))

            # 3) Market conditions container
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
        Gather all actionable intelligence from the unified memory system.
        
        Reads multiple memory bus keys to provide:
        - Gate signals (veto, risk_multiplier)
        - Danger zone detection
        - Expected PnL from similar trades
        - Avoidance signals
        - Pattern recognition insights
        - Playbook recommendations
        
        Returns a consolidated dict for position decision logic.
        """
        result: Dict[str, Any] = {
            # Gate signals
            "veto": False,
            "risk_multiplier": 1.0,
            "veto_reasons": [],
            # Danger zones
            "in_danger_zone": False,
            "danger_similarity": 0.0,
            "avoidance_signal": 0.0,
            # Playbook insights
            "expected_pnl": 0.0,
            "playbook_confidence": 0.5,
            "signed_bias": 0.0,
            # Neural insights
            "neural_risk_hint": 0.5,
            # Loss prediction
            "loss_prob": 0.0,
            # Intervention recommendation
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
                    # Use max similarity from zones if available
                    max_sim = 0.0
                    for z in zones[:5]:
                        if isinstance(z, dict):
                            max_sim = max(max_sim, float(z.get("similarity", 0.0)))
                    result["danger_similarity"] = max(result["danger_similarity"], max_sim)
            
            # 4) mistake_avoidance - Avoidance signal
            mistake_avoidance = self.smart_bus.get("mistake_avoidance", "PositionManager")
            if isinstance(mistake_avoidance, dict):
                result["avoidance_signal"] = float(mistake_avoidance.get("avoidance_signal", 0.0))
                # Strong avoidance signal can trigger danger zone
                if result["avoidance_signal"] > 0.6:
                    result["in_danger_zone"] = True
            
            # 5) playbook_recall - Historical pattern insights
            playbook_recall = self.smart_bus.get("playbook_recall", "PositionManager")
            if isinstance(playbook_recall, dict):
                if result["expected_pnl"] == 0.0:
                    result["expected_pnl"] = float(playbook_recall.get("expected_pnl", 0.0))
                if result["playbook_confidence"] == 0.5:
                    result["playbook_confidence"] = float(playbook_recall.get("confidence", 0.5))
            
            # 6) intuition_vector - Compressed pattern intelligence
            intuition = self.smart_bus.get("intuition_vector", "PositionManager")
            if isinstance(intuition, dict):
                strength = float(intuition.get("strength", 0.0))
                # Negative intuition strength indicates loss-aligned patterns
                if strength < -0.3:
                    result["avoidance_signal"] = max(result["avoidance_signal"], abs(strength))
            
            # 7) loss_prevention - Loss prevention metrics
            loss_prevention = self.smart_bus.get("loss_prevention", "PositionManager")
            if isinstance(loss_prevention, dict):
                effectiveness = float(loss_prevention.get("avoidance_effectiveness", 0.0))
                # If avoidance has been effective, trust it more
                if effectiveness > 0.5:
                    result["risk_multiplier"] *= (1.0 - effectiveness * 0.3)
            
            # Log memory intelligence if debug enabled
            if self.debug and (result["veto"] or result["in_danger_zone"] or result["avoidance_signal"] > 0.3):
                self.logger.info(format_operator_message(
                    icon="🧠",
                    message="MEMORY_INTELLIGENCE",
                    veto=result["veto"],
                    danger_zone=result["in_danger_zone"],
                    danger_sim=f"{result['danger_similarity']:.2f}",
                    avoidance=f"{result['avoidance_signal']:.2f}",
                    risk_mult=f"{result['risk_multiplier']:.2f}",
                    expected_pnl=f"{result['expected_pnl']:.2f}",
                ))
                
        except Exception as e:
            if self.debug:
                self.logger.warning(f"Memory intelligence fetch failed: {e}")
        
        return result

    # ==========================================================
    # Emergency conditions
    # ==========================================================
    def _check_emergency_conditions(self, context: SignalContext) -> bool:
        """Return True if any emergency condition is met (drawdown/loss streak/exposure)."""
        triggers = {
            "drawdown": context.drawdown > float(self.Cval("emergency_drawdown_trigger", 0.15)),
            "loss_streak": self.consecutive_losses >= int(self.Cval("max_consecutive_losses", 5)),
            "exposure": context.current_exposure > float(self.Cval("max_instrument_concentration", 0.30)) * 1.5,
            "liquidity": context.liquidity_score < 0.30,
        }
        active = bool(triggers["drawdown"] or triggers["loss_streak"] or triggers["exposure"])
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
                    if rec and rec.value is not None and hasattr(rec, "age_seconds") and rec.age_seconds() < 10.0:
                        executor_active = True
                        break
                else:
                    val = self.smart_bus.get(key, "PositionManager", default=None)
                    if val:
                        executor_active = True
                        break
        except Exception:
            pass

        suppress = bool(self.Cval("suppress_emergency_without_executor", True)) and not executor_active

        # Log once per tick-ish
        stamp = getattr(self, "_last_emergency_diag_stamp", None)
        current_stamp = context.timestamp or f"t_{int(time.time())}"
        if stamp != current_stamp:
            try:
                self.logger.warning(
                    format_operator_message(
                        icon="[ALERT]" if not suppress else "[INFO]",
                        message="Emergency condition evaluated" + (" (SUPPRESSED)" if suppress else ""),
                        drawdown=f"{context.drawdown:.4f}",
                        consecutive_losses=self.consecutive_losses,
                        loss_streak_trigger=triggers["loss_streak"],
                        drawdown_trigger=triggers["drawdown"],
                        exposure=f"{context.current_exposure:.4f}",
                        exposure_trigger=triggers["exposure"],
                        liquidity=f"{context.liquidity_score:.3f}",
                        liquidity_trigger=triggers["liquidity"],
                        executor_active=executor_active,
                        suppressed=suppress,
                    )
                )
            except Exception:
                pass
            self._last_emergency_diag_stamp = current_stamp

        return False if suppress else True
