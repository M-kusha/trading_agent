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
        instrument = context.instrument
        has_position = instrument in self.open_positions

        decision = PositionDecision.HOLD
        intensity = 0.0
        size = 0.0
        confidence = 0.5
        rationale: Dict[str, Any] = {"stage": "initial", "factors": []}
        risk_factors: Dict[str, float] = {}

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

        # ---------- Hard risk limits & trailing take-profit (when position exists)
        if has_position:
            pnl_eur = self._get_unrealised_pnl_from_bus(instrument)
            hard_loss_eur = float(self.Cval("hard_loss_eur", 100.0))

            # 1) Hard stop-loss (absolute)
            if pnl_eur <= -abs(hard_loss_eur):
                decision = PositionDecision.CLOSE
                intensity = 1.0
                confidence = 0.95
                rationale["stage"] = "risk_management"
                rationale["factors"].append(f"Hard loss-cut triggered at EUR {pnl_eur:.2f} ≤ -{hard_loss_eur:.0f}")
                return self._finalize_decision(instrument, decision, intensity, 0.0, confidence, rationale, risk_factors, context)

            # 2) Trailing take-profit with favorability check
            trailing_pct = float(self.Cval("take_profit_trailing_pct", 0.15))
            min_activation_eur = float(self.Cval("take_profit_min_eur", 150.0))

            favors_down = self.favors_trend_down(instrument, context.market_intensity, lookback=6, eps=0.03)
            if self.should_close_for_trailing_profit(instrument, trailing_pct, min_activation_eur, favors_down):
                decision = PositionDecision.CLOSE
                intensity = 0.9
                confidence = 0.9
                rationale["stage"] = "take_profit_trailing"
                peak = self._profit_tracker.peak(instrument)
                cur = self._profit_tracker.last(instrument)
                drop_pct = (peak - cur) / max(1e-9, peak)
                rationale["factors"].append(
                    f"Trailing TP: peak={peak:.2f}, now={cur:.2f}, drop={drop_pct:.1%} ≥ {trailing_pct:.1%}, favors_down"
                )
                return self._finalize_decision(instrument, decision, intensity, 0.0, confidence, rationale, risk_factors, context)

        # ---------- Decide open / scale / reduce
        if not has_position:
            if sig_strength < min_sig:
                rationale["stage"] = "signal_filter"
                rationale["factors"].append(f"Signal {sig_strength:.3f} below threshold {min_sig:.2f}")
                return self._finalize_decision(instrument, decision, 0.0, 0.0, 0.5, rationale, risk_factors, context)

            decision = PositionDecision.OPEN_LONG if context.market_direction > 0 else PositionDecision.OPEN_SHORT
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

            rationale["stage"] = "new_position"
            rationale["factors"].append(f"Strong signal {sig_strength:.3f} for new position")
            return self._finalize_decision(instrument, decision, intensity, size, confidence, rationale, risk_factors, context)

        # From here on: we have a position; choose scale-up / scale-down / close
        current_side = int(np.sign(self.open_positions[instrument].get("side", 0)))
        signal_aligns = (current_side > 0 and context.market_direction > 0) or (current_side < 0 and context.market_direction < 0)
        pnl_eur = self._get_unrealised_pnl_from_bus(instrument)

        # Scale-up rules: only when strong, not too frequent, not near concentration limits, and preferably with non-negative P&L
        if signal_aligns:
            scale_th = float(self.Cval("position_scale_threshold", max(min_sig + 0.10, 0.35)))
            if sig_strength > scale_th and self.scale_cooldown_ok(instrument) and pnl_eur >= -0.01 * self.Cval("hard_loss_eur", 100.0):
                # also avoid scaling if already near concentration cap
                max_conc = float(self.Cval("max_instrument_concentration", 0.30))
                if context.current_exposure < (max_conc * 0.9):
                    decision = PositionDecision.SCALE_UP
                    # lighter intensity for adds
                    intensity = min(0.8 * sig_strength, 0.9)
                    confidence = self._calculate_confidence(context, decision)
                    size = max(self._calculate_position_size(context, intensity, confidence) * 0.5, 0.0)
                    if size > 0:
                        rationale["stage"] = "scale_up"
                        rationale["factors"].append(f"Signal {sig_strength:.3f} aligns; exposure OK; cooldown OK")
                        # arm cooldown to avoid spam scaling
                        self.arm_scale_cooldown(instrument, seconds=float(self.Cval("scale_cooldown_seconds", 15.0)))
                        return self._finalize_decision(instrument, decision, intensity, size, confidence, rationale, risk_factors, context)

        # Opposing signals: reduce or close
        if not signal_aligns and sig_strength > 0.50:
            if sig_strength >= 0.80:
                decision = PositionDecision.CLOSE
                intensity = 0.9
                confidence = 0.8
                rationale["stage"] = "close_reverse"
                rationale["factors"].append(f"Strong opposing signal {sig_strength:.3f}")
                return self._finalize_decision(instrument, decision, intensity, 0.0, confidence, rationale, risk_factors, context)
            else:
                decision = PositionDecision.SCALE_DOWN
                intensity = 0.6
                confidence = 0.65
                size = max(self._calculate_position_size(context, intensity, confidence) * 0.5, 0.0)
                rationale["stage"] = "scale_down"
                rationale["factors"].append(f"Opposing signal {sig_strength:.3f}, reducing exposure")
                return self._finalize_decision(instrument, decision, intensity, size, confidence, rationale, risk_factors, context)

        # Risk-based nudge: if risk metrics high, consider mild scale-down
        risk_factors = self._assess_risk_factors(context)
        if max(risk_factors.values() or [0.0]) > 0.75 and context.current_exposure > 0.0:
            decision = PositionDecision.SCALE_DOWN
            intensity = 0.4
            confidence = 0.6
            size = max(self._calculate_position_size(context, intensity, confidence) * 0.4, 0.0)
            rationale["stage"] = "risk_management_reduce"
            rationale["factors"].append("High risk factors; trimming exposure")
            return self._finalize_decision(instrument, decision, intensity, size, confidence, rationale, risk_factors, context)

        # Nothing compelling — hold
        rationale["stage"] = "hold"
        rationale["factors"].append("No actionable change")
        return self._finalize_decision(instrument, decision, 0.0, 0.0, 0.55, rationale, risk_factors, context)

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
