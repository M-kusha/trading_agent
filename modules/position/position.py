# modules/position/position_manager.py
import numpy as np
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# --- Optional live-broker connector -----------------------------------------
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # still works in back-test / unit-test mode

from modules.trading_modes.trading_mode import TradingModeManager


class PositionDecision(Enum):
    HOLD = "hold"
    OPEN_LONG = "open_long"
    OPEN_SHORT = "open_short"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    CLOSE = "close"
    EMERGENCY_CLOSE = "emergency_close"


@dataclass
class SignalContext:
    """Container for all signal inputs to position decisions"""
    instrument: str
    market_intensity: float = 0.0
    market_direction: int = 0  # -1, 0, 1
    volatility: float = 0.02
    trend_strength: float = 0.0
    momentum: float = 0.0
    volume_profile: float = 1.0
    correlation_penalty: float = 0.0
    
    # Market regime context
    regime: str = "normal"  # normal, volatile, trending, ranging
    liquidity_score: float = 1.0
    
    # Portfolio context
    current_exposure: float = 0.0
    drawdown: float = 0.0
    balance: float = 10000.0


@dataclass
class PositionDecisionResult:
    """Result of position decision process"""
    decision: PositionDecision
    intensity: float  # 0.0 to 1.0
    size: float
    confidence: float
    rationale: Dict[str, Any]
    risk_factors: Dict[str, float]


class PositionManager:

    def __init__(
        self,
        initial_balance: float,
        instruments: List[str],
        max_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        loss_reduction: float = 0.2,
        max_instrument_concentration: float = 0.25,
        min_volatility: float = 0.015,
        # Tunable exit thresholds:
        hard_loss_eur: float = 30.0,
        trail_pct: float = 0.10,
        trail_abs_eur: float = 10.0,
        pips_tolerance: int = 20,
        min_size_pct: float = 0.01,
        # Signal interpretation parameters:
        min_signal_threshold: float = 0.15,  # Minimum signal to consider
        position_scale_threshold: float = 0.30,  # Threshold for scaling positions
        emergency_close_threshold: float = 0.85,  # Threshold for emergency closes
        confidence_decay: float = 0.95,  # Confidence decay per step
        debug: bool = True,
    ):
        self.mode_manager = TradingModeManager(initial_mode="safe", window=50)
        self.initial_balance = float(initial_balance)
        self.instruments = instruments
        self.default_max_pct = float(max_pct)
        self.max_pct = float(max_pct)
        self.min_size_pct = float(min_size_pct)
        self.debug = debug

        # Signal interpretation parameters
        self.min_signal_threshold = float(min_signal_threshold)
        self.position_scale_threshold = float(position_scale_threshold)
        self.emergency_close_threshold = float(emergency_close_threshold)
        self.confidence_decay = float(confidence_decay)

        # Loss‐streak breaker
        self.consecutive_losses = 0
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.loss_reduction = float(loss_reduction)

        # Concentration & volatility floor
        self.max_instrument_concentration = float(max_instrument_concentration)
        self.min_volatility = float(min_volatility)

        # Exit thresholds (constructor parameters)
        self.hard_loss_eur = float(hard_loss_eur)
        self.trail_pct = float(trail_pct)
        self.trail_abs_eur = float(trail_abs_eur)
        self.pips_tolerance = int(pips_tolerance)

        self.open_positions: Dict[str, Dict[str, float]] = {}
        self.env: Optional[Any] = None  # to be set externally

        # Decision tracking and state
        self.last_decisions: Dict[str, PositionDecisionResult] = {}
        self.position_confidence: Dict[str, float] = {}
        self.signal_history: Dict[str, List[float]] = {inst: [] for inst in instruments}
        
        # Internals for audit/explanation
        self._forced_action = None
        self._forced_conf = None
        self.last_rationale: Dict[str, Any] = {}
        self.last_confidence_components: Dict[str, Any] = {}

        # Professional logger
        import os
        self.logger = logging.getLogger("PositionManager")
        if not self.logger.handlers:
            log_dir = os.path.join("logs", "position")
            os.makedirs(log_dir, exist_ok=True)
            handler = logging.FileHandler("logs/position/position_manager.log")
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

        self.logger.info(f"[PositionManager] Initialized with {len(instruments)} instruments")
        self.logger.info(f"[PositionManager] Signal thresholds: min={min_signal_threshold}, scale={position_scale_threshold}")

    # ------------- Evolutionary Logic ----------------------
    def mutate(self, std: float = 0.05):
        """Mutate parameters for evolutionary optimization"""
        self.max_pct += np.random.normal(0, std)
        self.max_pct = np.clip(self.max_pct, 0.01, 0.25)
        self.max_instrument_concentration += np.random.normal(0, std)
        self.max_instrument_concentration = np.clip(
            self.max_instrument_concentration, 0.05, 0.5
        )
        self.loss_reduction += np.random.normal(0, std)
        self.loss_reduction = np.clip(self.loss_reduction, 0.05, 1.0)
        self.min_volatility += np.random.normal(0, std)
        self.min_volatility = np.clip(self.min_volatility, 0.001, 0.10)
        self.max_consecutive_losses += int(np.random.choice([-1, 0, 1]))
        self.max_consecutive_losses = int(np.clip(self.max_consecutive_losses, 1, 20))
        
        # Mutate signal thresholds
        self.min_signal_threshold += np.random.normal(0, std * 0.5)
        self.min_signal_threshold = np.clip(self.min_signal_threshold, 0.05, 0.5)
        
        if self.debug:
            self.logger.info("[PositionManager] Mutated parameters.")

    def crossover(self, other: "PositionManager"):
        """Create offspring via crossover with another PositionManager"""
        child = copy.deepcopy(self)
        for attr in [
            "max_pct",
            "max_instrument_concentration", 
            "loss_reduction",
            "min_volatility",
            "max_consecutive_losses",
            "min_signal_threshold",
            "position_scale_threshold",
        ]:
            if np.random.rand() > 0.5:
                setattr(child, attr, getattr(other, attr))
        if self.debug:
            self.logger.info("[PositionManager] Crossover complete.")
        return child

    # ------------------------------------------------------

    def set_env(self, env: Any):
        """Set environment reference"""
        self.env = env

    def reset(self):
        """Reset position manager state"""
        self.max_pct = self.default_max_pct
        self.consecutive_losses = 0
        self.open_positions.clear()
        self.last_decisions.clear()
        self.position_confidence.clear()
        for inst in self.instruments:
            self.signal_history[inst].clear()
        
        self._forced_action = None
        self._forced_conf = None
        self.last_rationale.clear()
        self.last_confidence_components.clear()
        
        if self.max_pct < 1e-5:
            self.logger.warning(
                f"[PositionManager] max_pct was {self.max_pct:.6f} at reset, restoring to default {self.default_max_pct:.4f}"
            )
            self.max_pct = self.default_max_pct

    def step(self, **kwargs):
        """Execute one step of position management"""
        env = kwargs.get("env", None)
        if env:
            self.env = env
        
        # Ensure minimum allocation capability
        min_cap = 0.01  # 1% minimal allocation
        if self.max_pct < min_cap:
            self.logger.warning(
                f"[PositionManager] max_pct={self.max_pct:.6f} below min_cap={min_cap:.4f} – restoring"
            )
            self.max_pct = self.default_max_pct

        # Live‐mode: sync & apply exit rules
        if self.env and getattr(self.env, "live_mode", False):
            self._sync_live_positions()
            self._apply_exit_rules()

        # Decay position confidence over time
        for inst in self.position_confidence:
            self.position_confidence[inst] *= self.confidence_decay

        if self.debug:
            self.logger.debug(
                f"Step | Open positions: {len(self.open_positions)}, "
                f"Consecutive losses: {self.consecutive_losses}, Max %: {self.max_pct:.3f}"
            )

    # ────────────────────────────────────────────────────────
    #              Core Decision Making Logic
    # ────────────────────────────────────────────────────────
    
    def process_market_signals(self, market_data: Dict[str, Any]) -> Dict[str, PositionDecisionResult]:
        """
        Main entry point for processing market signals into position decisions.
        
        This method implements the hierarchical decision making:
        1. Strategic: Assess market regime and portfolio health
        2. Tactical: Make instrument-specific decisions  
        3. Execution: Apply risk management and sizing
        """
        decisions = {}
        
        # 1. Strategic Layer: Portfolio-level assessment
        portfolio_health = self._assess_portfolio_health()
        market_regime = self._assess_market_regime(market_data)
        
        self.logger.debug(f"Portfolio health: {portfolio_health}, Market regime: {market_regime}")
        
        # 2. Tactical Layer: Per-instrument decisions
        for instrument in self.instruments:
            # Extract signal context for this instrument
            signal_context = self._extract_signal_context(instrument, market_data, portfolio_health)
            
            # Make position decision
            decision_result = self._make_position_decision(signal_context)
            
            # Store decision and update tracking
            decisions[instrument] = decision_result
            self.last_decisions[instrument] = decision_result
            
            # Update signal history
            self.signal_history[instrument].append(signal_context.market_intensity)
            if len(self.signal_history[instrument]) > 50:  # Keep last 50 signals
                self.signal_history[instrument].pop(0)
                
            if decision_result.decision != PositionDecision.HOLD:
                self.logger.info(
                    f"[{instrument}] Decision: {decision_result.decision.value} | "
                    f"Intensity: {decision_result.intensity:.3f} | "
                    f"Size: {decision_result.size:.2f} | "
                    f"Confidence: {decision_result.confidence:.3f}"
                )
        
        return decisions

    def _assess_portfolio_health(self) -> Dict[str, float]:
        """Assess overall portfolio health metrics"""
        balance = getattr(self.env, 'balance', self.initial_balance) if self.env else self.initial_balance
        drawdown = getattr(self.env, 'current_drawdown', 0.0) if self.env else 0.0
        
        # Calculate total exposure
        total_exposure = 0.0
        for pos_data in self.open_positions.values():
            if "size" in pos_data:
                total_exposure += abs(pos_data["size"])
            else:
                # Live mode: estimate exposure
                total_exposure += abs(pos_data.get("lots", 0)) * pos_data.get("price_open", 1) * 100_000
        
        exposure_ratio = total_exposure / max(balance, 1.0)
        
        # Health score components
        dd_health = max(0.0, 1.0 - drawdown * 2.0)  # Penalize drawdown
        exposure_health = max(0.0, 1.0 - exposure_ratio / self.max_instrument_concentration)
        streak_health = max(0.1, 1.0 - self.consecutive_losses / self.max_consecutive_losses)
        
        return {
            "drawdown_health": dd_health,
            "exposure_health": exposure_health,
            "streak_health": streak_health,
            "overall_health": (dd_health + exposure_health + streak_health) / 3.0,
            "total_exposure": total_exposure,
            "exposure_ratio": exposure_ratio
        }

    def _assess_market_regime(self, market_data: Dict[str, Any]) -> str:
        """Determine current market regime"""
        # Extract volatility indicators
        avg_volatility = 0.0
        trend_strength = 0.0
        momentum_count = 0
        
        for instrument in self.instruments:
            inst_data = market_data.get(instrument, {})
            
            # Get volatility (from market data or default)
            vol = inst_data.get('volatility', self.min_volatility) 
            avg_volatility += vol
            
            # Get trend indicators
            if 'trend_strength' in inst_data:
                trend_strength += abs(inst_data['trend_strength'])
            
            # Count momentum signals
            if abs(inst_data.get('momentum', 0.0)) > 0.3:
                momentum_count += 1
        
        if self.instruments:
            avg_volatility /= len(self.instruments)
            trend_strength /= len(self.instruments)
        
        # Classify regime
        if avg_volatility > 0.04:
            return "volatile"
        elif trend_strength > 0.5:
            return "trending" 
        elif momentum_count >= len(self.instruments) * 0.6:
            return "momentum"
        else:
            return "ranging"

    def _extract_signal_context(self, instrument: str, market_data: Dict[str, Any], portfolio_health: Dict[str, float]) -> SignalContext:
        """Extract and structure signal context for decision making"""
        inst_data = market_data.get(instrument, {})
        
        # Extract market signals
        market_intensity = float(inst_data.get('intensity', 0.0))
        market_direction = int(np.sign(market_intensity))
        volatility = max(float(inst_data.get('volatility', self.min_volatility)), self.min_volatility)
        trend_strength = float(inst_data.get('trend_strength', 0.0))
        momentum = float(inst_data.get('momentum', 0.0))
        volume_profile = float(inst_data.get('volume_profile', 1.0))
        
        # Calculate correlation penalty
        correlation_penalty = 0.0
        if self.env and hasattr(self.env, 'get_current_correlation'):
            try:
                correlation = self.env.get_current_correlation()
                correlation_penalty = min(abs(correlation) * 0.5, 0.8)
            except:
                correlation_penalty = 0.0
        
        # Get portfolio context
        balance = getattr(self.env, 'balance', self.initial_balance) if self.env else self.initial_balance
        drawdown = getattr(self.env, 'current_drawdown', 0.0) if self.env else 0.0
        
        # Get liquidity score
        liquidity_score = 1.0
        if self.env and hasattr(self.env, 'liquidity_layer'):
            try:
                liquidity_score = self.env.liquidity_layer.current_score()
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
            current_exposure=portfolio_health["exposure_ratio"],
            drawdown=drawdown,
            balance=balance,
            liquidity_score=liquidity_score
        )

    def _make_position_decision(self, context: SignalContext) -> PositionDecisionResult:
        """
        Core position decision logic implementing hierarchical decision making
        """
        instrument = context.instrument
        has_position = instrument in self.open_positions
        
        # Initialize decision components
        decision = PositionDecision.HOLD
        intensity = 0.0
        size = 0.0
        confidence = 0.5
        risk_factors = {}
        rationale = {"stage": "initial", "factors": []}
        
        # Get absolute signal strength
        signal_strength = abs(context.market_intensity)
        signal_direction = np.sign(context.market_intensity)
        
        # Stage 1: Emergency conditions check
        if self._check_emergency_conditions(context):
            if has_position:
                decision = PositionDecision.EMERGENCY_CLOSE
                intensity = 1.0
                confidence = 0.9
                rationale["stage"] = "emergency"
                rationale["factors"].append("Emergency conditions detected")
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors)
        
        # Stage 2: Signal strength filtering
        if signal_strength < self.min_signal_threshold:
            rationale["stage"] = "signal_filter"
            rationale["factors"].append(f"Signal strength {signal_strength:.3f} below threshold {self.min_signal_threshold}")
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors)
        
        # Stage 3: Portfolio health checks
        portfolio_health_score = self._calculate_portfolio_health_score(context)
        if portfolio_health_score < 0.3:
            rationale["stage"] = "portfolio_health"
            rationale["factors"].append(f"Portfolio health {portfolio_health_score:.3f} too low")
            # Still allow closes but no new positions
            if has_position and signal_strength > 0.7:
                decision = PositionDecision.CLOSE
                intensity = 0.8
                confidence = 0.7
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors)
        
        # Stage 4: Position-specific decision logic
        if not has_position:
            # New position logic
            if signal_strength >= self.min_signal_threshold:
                decision = PositionDecision.OPEN_LONG if signal_direction > 0 else PositionDecision.OPEN_SHORT
                intensity = signal_strength
                confidence = self._calculate_confidence(context, decision)
                size = self._calculate_position_size(context, intensity, confidence)
                rationale["stage"] = "new_position"
                rationale["factors"].append(f"Strong signal {signal_strength:.3f} for new position")
        else:
            # Existing position management
            current_side = self.open_positions[instrument].get("side", 0)
            position_pnl, _ = self._calc_unrealised_pnl(instrument, self.open_positions[instrument])
            
            # Check if signal aligns with current position
            signal_aligns = (current_side > 0 and signal_direction > 0) or (current_side < 0 and signal_direction < 0)
            
            if signal_aligns and signal_strength > self.position_scale_threshold:
                # Scale up position
                decision = PositionDecision.SCALE_UP
                intensity = min(signal_strength * 0.8, 0.9)  # Conservative scaling
                confidence = self._calculate_confidence(context, decision)
                size = self._calculate_position_size(context, intensity, confidence) * 0.5  # Smaller scale
                rationale["stage"] = "scale_up"
                rationale["factors"].append(f"Signal {signal_strength:.3f} aligns with position, scaling up")
                
            elif not signal_aligns and signal_strength > 0.5:
                # Consider closing or reversing
                if signal_strength > 0.8:
                    decision = PositionDecision.CLOSE
                    intensity = 0.9
                    confidence = 0.8
                    rationale["stage"] = "close_reverse"
                    rationale["factors"].append(f"Strong opposing signal {signal_strength:.3f}")
                else:
                    decision = PositionDecision.SCALE_DOWN
                    intensity = 0.6
                    confidence = 0.6
                    rationale["stage"] = "scale_down"
                    rationale["factors"].append(f"Opposing signal {signal_strength:.3f}, reducing exposure")
            
            elif position_pnl < -self.hard_loss_eur * 0.5:  # Approaching hard loss
                decision = PositionDecision.CLOSE
                intensity = 0.8
                confidence = 0.9
                rationale["stage"] = "risk_management"
                rationale["factors"].append(f"Position approaching loss limit: {position_pnl:.2f}")
        
        # Stage 5: Final risk adjustments
        risk_factors = self._assess_risk_factors(context)
        final_intensity = intensity * (1.0 - max(risk_factors.values()))
        final_confidence = confidence * portfolio_health_score
        
        # Ensure minimum viable size or zero
        if decision in [PositionDecision.OPEN_LONG, PositionDecision.OPEN_SHORT, PositionDecision.SCALE_UP]:
            if size < context.balance * self.min_size_pct and final_intensity > 0.3:
                size = context.balance * self.min_size_pct
            elif size < context.balance * self.min_size_pct:
                size = 0.0
                decision = PositionDecision.HOLD
                rationale["factors"].append("Size too small, holding instead")
        
        return PositionDecisionResult(
            decision=decision,
            intensity=final_intensity,
            size=size,
            confidence=final_confidence,
            rationale=rationale,
            risk_factors=risk_factors
        )

    def _check_emergency_conditions(self, context: SignalContext) -> bool:
        """Check for emergency conditions requiring immediate action"""
        emergency_conditions = [
            context.drawdown > 0.15,  # 15% drawdown
            self.consecutive_losses >= self.max_consecutive_losses,
            context.current_exposure > self.max_instrument_concentration * 1.5,
            context.liquidity_score < 0.3
        ]
        return any(emergency_conditions)

    def _calculate_portfolio_health_score(self, context: SignalContext) -> float:
        """Calculate overall portfolio health score"""
        drawdown_component = max(0.0, 1.0 - context.drawdown * 3.0)
        exposure_component = max(0.0, 1.0 - context.current_exposure / self.max_instrument_concentration)
        streak_component = max(0.1, 1.0 - self.consecutive_losses / self.max_consecutive_losses)
        liquidity_component = context.liquidity_score
        
        return (drawdown_component + exposure_component + streak_component + liquidity_component) / 4.0

    def _calculate_confidence(self, context: SignalContext, decision: PositionDecision) -> float:
        """Calculate confidence in the position decision"""
        base_confidence = 0.5
        
        # Signal strength contribution
        signal_confidence = min(abs(context.market_intensity) * 1.2, 0.4)
        
        # Trend alignment
        trend_confidence = min(abs(context.trend_strength) * 0.3, 0.2)
        
        # Volatility penalty (high vol = lower confidence)
        vol_penalty = min(context.volatility / 0.05, 0.2)
        
        # Portfolio health contribution
        health_boost = self._calculate_portfolio_health_score(context) * 0.2
        
        # Decision-specific adjustments
        decision_adjustment = 0.0
        if decision in [PositionDecision.CLOSE, PositionDecision.EMERGENCY_CLOSE]:
            decision_adjustment = 0.1  # Higher confidence in exits
        elif decision == PositionDecision.SCALE_DOWN:
            decision_adjustment = 0.05
        
        total_confidence = base_confidence + signal_confidence + trend_confidence - vol_penalty + health_boost + decision_adjustment
        return float(np.clip(total_confidence, 0.1, 1.0))

    def _assess_risk_factors(self, context: SignalContext) -> Dict[str, float]:
        """Assess various risk factors that might reduce position size"""
        risk_factors = {}
        
        # Volatility risk
        risk_factors["volatility"] = min((context.volatility - self.min_volatility) / 0.05, 0.5)
        
        # Correlation risk
        risk_factors["correlation"] = context.correlation_penalty
        
        # Drawdown risk
        risk_factors["drawdown"] = min(context.drawdown * 2.0, 0.8)
        
        # Concentration risk
        risk_factors["concentration"] = min(context.current_exposure / self.max_instrument_concentration, 0.9)
        
        # Liquidity risk
        risk_factors["liquidity"] = max(0.0, 1.0 - context.liquidity_score)
        
        return risk_factors

    # ────────────────────────────────────────────────────────────────
    #  Enhanced sizing engine with hierarchical decision support
    # ────────────────────────────────────────────────────────────────
    def calculate_size(
        self,
        volatility: float,
        intensity: float,
        balance: float,
        drawdown: float,
        correlation: Optional[float] = None,
        current_exposure: Optional[float] = None,
    ) -> float:
        """
        Enhanced position sizing that integrates with hierarchical decision making
        """
        # Input sanitization
        volatility = max(float(np.nan_to_num(volatility, nan=self.min_volatility)), self.min_volatility)
        intensity = float(np.nan_to_num(intensity, nan=0.0))
        balance = max(float(balance), 100.0)  # Minimum balance
        drawdown = float(np.nan_to_num(drawdown, nan=0.0))
        
        # Clip intensity to reasonable range
        intensity = np.clip(intensity, -1.0, 1.0)
        
        # Base risk budget calculation
        risk_pct = max(self.max_pct, 0.01)  # Ensure minimum risk allocation
        risk_budget = balance * risk_pct
        
        # Volatility-adjusted base size
        vol_adjusted_budget = risk_budget / volatility
        base_size = intensity * vol_adjusted_budget
        
        # Apply portfolio health modifiers
        portfolio_health = self._calculate_portfolio_health_score(
            SignalContext(
                instrument="",  # Not used in health calculation
                volatility=volatility,
                drawdown=drawdown,
                balance=balance,
                current_exposure=current_exposure or 0.0,
                liquidity_score=1.0  # Default
            )
        )
        
        # Health-based size adjustment
        health_multiplier = max(0.1, portfolio_health)  # Never go completely to zero
        adjusted_size = base_size * health_multiplier
        
        # Mode-dependent adjustments
        mode = getattr(self.mode_manager, "current_mode", "safe")
        if mode == "aggressive":
            adjusted_size *= 1.2
        elif mode == "conservative": 
            adjusted_size *= 0.7
        
        # Correlation penalty
        if correlation is not None:
            corr_penalty = 1.0 - min(abs(correlation) * 0.3, 0.5)  # Less aggressive penalty
            adjusted_size *= corr_penalty
        
        # Loss streak reduction
        if self.consecutive_losses >= self.max_consecutive_losses:
            streak_reduction = max(0.1, self.loss_reduction)  # Never reduce below 10%
            adjusted_size *= streak_reduction
            self.logger.info(f"Loss streak reduction applied: {streak_reduction:.2f}")
        
        # Ensure minimum viable size or zero
        abs_size = abs(adjusted_size)
        min_viable_size = balance * self.min_size_pct
        
        if abs_size < min_viable_size and abs(intensity) > 0.3:
            # Strong signal but small size - use minimum
            adjusted_size = np.sign(adjusted_size or intensity) * min_viable_size
        elif abs_size < min_viable_size:
            # Weak signal and small size - zero out
            adjusted_size = 0.0
        
        # Final safety bounds
        max_single_position = balance * risk_pct
        final_size = np.clip(adjusted_size, -max_single_position, max_single_position)
        
        return float(np.nan_to_num(final_size, nan=0.0, posinf=0.0, neginf=0.0))

    def _calculate_position_size(self, context: SignalContext, intensity: float, confidence: float) -> float:
        """Calculate position size using the enhanced sizing logic"""
        return self.calculate_size(
            volatility=context.volatility,
            intensity=intensity,
            balance=context.balance,
            drawdown=context.drawdown,
            correlation=context.correlation_penalty,
            current_exposure=context.current_exposure
        )

    # ────────────────────────────────────────────────────────
    #              Live‐sync positions from broker
    # ────────────────────────────────────────────────────────
    def _sync_live_positions(self):
        """Sync positions from live broker"""
        broker_positions: List[Dict[str, Any]] = []

        # 1) env.broker
        if self.env and hasattr(self.env, "broker") and self.env.broker is not None:
            try:
                broker_positions = self.env.broker.get_positions()
            except Exception as exc:
                self.logger.warning("env.broker.get_positions failed: %s", exc)

        # 2) MT5 fallback
        elif mt5 is not None:
            raw = mt5.positions_get() or []
            for p in raw:
                broker_positions.append(
                    dict(
                        instrument=f"{p.symbol[:3]}/{p.symbol[3:]}",
                        ticket=p.ticket,
                        side=1 if p.type == mt5.POSITION_TYPE_BUY else -1,
                        lots=p.volume,
                        price_open=p.price_open,
                    )
                )

        if not broker_positions:
            return

        new_positions: Dict[str, Dict[str, Any]] = {}
        for pos in broker_positions:
            inst = pos["instrument"]
            d = {
                "ticket": pos["ticket"],
                "side": pos["side"],
                "lots": pos["lots"],
                "price_open": pos["price_open"],
                "peak_profit": self.open_positions.get(inst, {}).get("peak_profit", 0.0),
                "size": pos["lots"],
            }
            new_positions[inst] = d

        self.open_positions = new_positions
        if self.debug:
            self.logger.debug(f"Synced live positions: {self.open_positions}")

    # ──────────────────────────────────────────────────────────────
    #     Exit logic – hard‐loss & hybrid trailing‐profit
    # ──────────────────────────────────────────────────────────────
    def _apply_exit_rules(self):
        """Apply automated exit rules to open positions"""
        for inst, data in list(self.open_positions.items()):
            pnl_eur, _ = self._calc_unrealised_pnl(inst, data)
            # update peak
            if pnl_eur > data["peak_profit"]:
                data["peak_profit"] = pnl_eur

            # hard‐loss
            if pnl_eur <= -self.hard_loss_eur:
                self._close_position(inst, "hard_loss")
                continue

            # trailing‐profit
            if data["peak_profit"] > 0:
                drawdown_eur = data["peak_profit"] - pnl_eur
                trigger = max(
                    data["peak_profit"] * self.trail_pct,
                    self.trail_abs_eur
                )
                if drawdown_eur >= trigger:
                    self._close_position(inst, "trail_stop")

    def _close_position(self, inst: str, reason: str):
        """Close a position via broker or simulation"""
        # env.broker
        if self.env and getattr(self.env, "broker", None):
            ok = self.env.broker.close_position(inst, comment=reason)
            if ok:
                self.logger.info("Closed %s via env.broker (%s)", inst, reason)
                self.open_positions.pop(inst, None)
            else:
                self.logger.error("env.broker.close_position failed for %s (%s)", inst, reason)
            return

        # MT5 close
        if mt5 is not None:
            data = self.open_positions[inst]
            side = data["side"]
            lots = data["lots"]
            sym = inst.replace("/", "")
            tick = mt5.symbol_info_tick(sym)
            price = (tick.bid if side > 0 else tick.ask) if tick else 0.0

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       sym,
                "volume":       lots,
                "type":         (mt5.ORDER_TYPE_SELL if side > 0 else mt5.ORDER_TYPE_BUY),
                "price":        price,
                "deviation":    self.pips_tolerance,
                "position":     data["ticket"],
                "magic":        10001,
                "comment":      f"auto-exit:{reason}",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            res = mt5.order_send(request)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info("Closed %s ticket %d (%s)", inst, data["ticket"], reason)
                self.open_positions.pop(inst, None)
            else:
                self.logger.error("order_send close failed for %s: %s", inst, res)
            return

        # backtest fallback
        self.logger.info("Marked %s closed (sim) – %s", inst, reason)
        self.open_positions.pop(inst, None)

    def _calc_unrealised_pnl(
        self,
        inst: str,
        data: Dict[str, Any],
    ) -> Tuple[float, float]:
        """Calculate unrealized P&L for a position"""
        sym = inst.replace("/", "")
        # price
        price = None
        if self.env and getattr(self.env, "broker", None):
            price = self.env.broker.get_price(sym, side=data["side"])
        elif mt5 is not None:
            tick = mt5.symbol_info_tick(sym)
            price = tick.bid if data["side"] > 0 else tick.ask
        if price is None or not np.isfinite(price):
            return 0.0, 0.0

        # contract size
        contract_size = 100_000
        if mt5 is not None:
            info = mt5.symbol_info(sym)
            if info and info.trade_contract_size:
                contract_size = info.trade_contract_size

        points = (price - data["price_open"]) * data["side"]
        pnl_eur = points * contract_size * data["lots"]
        pnl_pct = pnl_eur / (abs(data["price_open"]) * contract_size * data["lots"])
        return float(pnl_eur), float(pnl_pct)

    # ────────────────────────────────────────────────────────────────
    #  Enhanced Action and Confidence Methods
    # ────────────────────────────────────────────────────────────────
    
    def propose_action(self, obs: Any) -> np.ndarray:
        """
        Propose trading actions based on independent signal interpretation.
        
        This method no longer depends on env.meta_agent, breaking the circular dependency.
        Instead, it uses the hierarchical decision making process.
        """
        if self._forced_action is not None:
            return np.array(
                [self._forced_action] * len(self.instruments) * 2, dtype=np.float32
            )
        
        signals: List[float] = []
        
        # Extract market data from observation or environment
        market_data = self._extract_market_data_from_obs(obs)
        
        # Process signals through hierarchical decision making
        decisions = self.process_market_signals(market_data)
        
        # Convert decisions to action signals
        for inst in self.instruments:
            decision_result = decisions.get(inst, None)
            
            if decision_result is None:
                # Fallback: use basic signal extraction
                intensity = 0.0
                duration = 1.0
            else:
                # Map decision to intensity
                intensity = self._map_decision_to_intensity(decision_result)
                duration = 1.0  # Standard duration
                
                # Update position confidence tracking
                self.position_confidence[inst] = decision_result.confidence
            
            if self.debug:
                self.logger.debug(f"Propose action: {inst}: intensity={intensity:.3f}, dur={duration}")
            
            signals.extend([intensity, duration])
        
        return np.array(signals, dtype=np.float32)

    def _extract_market_data_from_obs(self, obs: Any) -> Dict[str, Any]:
        """Extract market data from observation or environment"""
        market_data = {}
        
        # Try to get data from environment first
        if self.env:
            # Get market data from various environment components
            if hasattr(self.env, 'market_data'):
                market_data = getattr(self.env, 'market_data', {})
            elif hasattr(self.env, 'get_market_data'):
                try:
                    market_data = self.env.get_market_data()
                except:
                    pass
            
            # Extract from individual components
            for inst in self.instruments:
                inst_data = market_data.get(inst, {})
                
                # Try to get signals from various sources
                if hasattr(self.env, 'indicators') and self.env.indicators:
                    try:
                        # Get latest indicator values
                        if hasattr(self.env.indicators, 'get_latest'):
                            latest = self.env.indicators.get_latest(inst)
                            if latest:
                                inst_data.update(latest)
                    except:
                        pass
                
                # Get volatility from various sources
                if 'volatility' not in inst_data:
                    if hasattr(self.env, 'get_volatility'):
                        try:
                            inst_data['volatility'] = self.env.get_volatility(inst)
                        except:
                            inst_data['volatility'] = self.min_volatility
                    else:
                        inst_data['volatility'] = self.min_volatility
                
                # Generate synthetic intensity if not available
                if 'intensity' not in inst_data:
                    inst_data['intensity'] = self._generate_synthetic_intensity(inst, inst_data)
                
                market_data[inst] = inst_data
        
        # Fallback: generate synthetic data if no environment data
        if not market_data:
            for inst in self.instruments:
                market_data[inst] = {
                    'intensity': self._generate_synthetic_intensity(inst, {}),
                    'volatility': self.min_volatility,
                    'trend_strength': 0.0,
                    'momentum': 0.0,
                    'volume_profile': 1.0
                }
        
        return market_data

    def _generate_synthetic_intensity(self, instrument: str, inst_data: Dict[str, Any]) -> float:
        """Generate synthetic intensity signal for bootstrapping"""
        # Use signal history to create momentum-based intensity
        history = self.signal_history.get(instrument, [])
        
        if len(history) >= 3:
            # Use recent trend
            recent_avg = np.mean(history[-3:])
            older_avg = np.mean(history[-6:-3]) if len(history) >= 6 else recent_avg
            momentum = recent_avg - older_avg
            base_intensity = np.clip(momentum * 2.0, -0.8, 0.8)
        else:
            # Generate weak random signal for bootstrapping
            base_intensity = np.random.normal(0, 0.1)
        
        # Add some technical indicator influence if available
        if 'rsi' in inst_data:
            rsi = inst_data['rsi']
            if rsi < 30:
                base_intensity += 0.2  # Oversold
            elif rsi > 70:
                base_intensity -= 0.2  # Overbought
        
        if 'macd' in inst_data:
            macd = inst_data.get('macd', 0.0)
            base_intensity += np.clip(macd * 0.5, -0.3, 0.3)
        
        # Ensure reasonable bounds
        return float(np.clip(base_intensity, -1.0, 1.0))

    def _map_decision_to_intensity(self, decision_result: PositionDecisionResult) -> float:
        """Map position decision to trading intensity"""
        decision = decision_result.decision
        base_intensity = decision_result.intensity
        
        if decision == PositionDecision.HOLD:
            return 0.0
        elif decision == PositionDecision.OPEN_LONG:
            return base_intensity
        elif decision == PositionDecision.OPEN_SHORT:
            return -base_intensity
        elif decision == PositionDecision.SCALE_UP:
            # For scaling, use moderate intensity
            return base_intensity * 0.6
        elif decision == PositionDecision.SCALE_DOWN:
            return -base_intensity * 0.4
        elif decision in [PositionDecision.CLOSE, PositionDecision.EMERGENCY_CLOSE]:
            # For closes, use opposite of current position direction
            return -base_intensity * 0.8
        
        return 0.0

    def confidence(self, obs: Any) -> float:
        """
        Calculate overall confidence based on portfolio and position health.
        Now independent of circular dependencies.
        """
        if self._forced_conf is not None:
            return float(self._forced_conf)

        # Calculate confidence components
        confidence_components = []
        
        # Portfolio health confidence
        if self.env:
            balance = getattr(self.env, 'balance', self.initial_balance)
            drawdown = getattr(self.env, 'current_drawdown', 0.0)
        else:
            balance = self.initial_balance
            drawdown = 0.0
        
        # Portfolio health score
        portfolio_health = self._calculate_portfolio_health_score(
            SignalContext(
                instrument="",
                drawdown=drawdown,
                balance=balance,
                current_exposure=self._calculate_current_exposure_ratio(),
                liquidity_score=self._get_liquidity_score()
            )
        )
        confidence_components.append(portfolio_health)
        
        # Position-specific confidence
        if self.position_confidence:
            avg_position_conf = np.mean(list(self.position_confidence.values()))
            confidence_components.append(avg_position_conf)
        else:
            confidence_components.append(0.5)  # Neutral when no positions
        
        # Signal quality confidence
        signal_quality = self._assess_signal_quality()
        confidence_components.append(signal_quality)
        
        # Mode-based confidence
        mode = getattr(self.mode_manager, "current_mode", "safe")
        mode_confidence = 0.8 if mode == "safe" else (0.9 if mode == "aggressive" else 0.6)
        confidence_components.append(mode_confidence)
        
        # Calculate weighted average
        weights = [0.3, 0.3, 0.25, 0.15]  # Portfolio, positions, signals, mode
        final_confidence = np.average(confidence_components, weights=weights)
        
        # Store components for audit
        self.last_confidence_components = {
            "portfolio_health": portfolio_health,
            "position_confidence": confidence_components[1],
            "signal_quality": signal_quality,
            "mode_confidence": mode_confidence,
            "final_confidence": final_confidence
        }
        
        if self.debug:
            self.logger.debug(f"Confidence components: {self.last_confidence_components}")

        return float(np.clip(final_confidence, 0.1, 1.0))

    def _calculate_current_exposure_ratio(self) -> float:
        """Calculate current exposure as ratio of balance"""
        if not self.open_positions:
            return 0.0
        
        balance = getattr(self.env, 'balance', self.initial_balance) if self.env else self.initial_balance
        total_exposure = 0.0
        
        for pos_data in self.open_positions.values():
            if "size" in pos_data:
                total_exposure += abs(pos_data["size"])
            else:
                # Live mode estimation
                total_exposure += abs(pos_data.get("lots", 0)) * pos_data.get("price_open", 1) * 100_000
        
        return total_exposure / max(balance, 1.0)

    def _get_liquidity_score(self) -> float:
        """Get current liquidity score"""
        if self.env and hasattr(self.env, 'liquidity_layer'):
            try:
                return self.env.liquidity_layer.current_score()
            except:
                pass
        return 1.0  # Default good liquidity

    def _assess_signal_quality(self) -> float:
        """Assess the quality of recent signals"""
        if not self.signal_history:
            return 0.5
        
        quality_scores = []
        for inst, history in self.signal_history.items():
            if len(history) >= 5:
                # Check signal consistency and strength
                recent_signals = history[-5:]
                signal_strength = np.mean(np.abs(recent_signals))
                signal_consistency = 1.0 - np.std(recent_signals)
                inst_quality = (signal_strength + max(0, signal_consistency)) / 2.0
                quality_scores.append(inst_quality)
        
        return np.mean(quality_scores) if quality_scores else 0.5

    # ────────────────────────────────────────────────────────────────
    #  API and Interface Methods
    # ────────────────────────────────────────────────────────────────

    def force_action(self, value: float):
        """Force a specific action value for testing/debugging"""
        self._forced_action = float(value)

    def force_confidence(self, value: float):
        """Force a specific confidence value for testing/debugging"""
        self._forced_conf = float(value)

    def clear_forced(self):
        """Clear any forced values"""
        self._forced_action = None
        self._forced_conf = None

    def get_observation_components(self) -> np.ndarray:
        """Get observation components for environment"""
        dd = getattr(self.env, "current_drawdown", 0.0) if self.env else 0.0
        conf = self.confidence(None)
        exposure = self._calculate_current_exposure_ratio()
        return np.array([float(dd), float(conf), float(exposure)], dtype=np.float32)

    def get_last_rationale(self) -> Dict[str, Any]:
        """Get rationale from last decision"""
        rationales = {}
        for inst, decision in self.last_decisions.items():
            rationales[inst] = decision.rationale
        return rationales

    def get_last_confidence_components(self):
        """Get detailed confidence breakdown"""
        return self.last_confidence_components.copy()

    def get_full_audit(self):
        """Get comprehensive audit information"""
        return {
            "positions": copy.deepcopy(self.open_positions),
            "last_decisions": {k: {
                "decision": v.decision.value,
                "intensity": v.intensity,
                "confidence": v.confidence,
                "rationale": v.rationale
            } for k, v in self.last_decisions.items()},
            "position_confidence": copy.deepcopy(self.position_confidence),
            "last_confidence_components": self.get_last_confidence_components(),
            "consecutive_losses": self.consecutive_losses,
            "max_pct": self.max_pct,
            "max_instrument_concentration": self.max_instrument_concentration,
            "signal_history_summary": {
                k: {
                    "length": len(v),
                    "recent_avg": np.mean(v[-5:]) if len(v) >= 5 else 0.0,
                    "recent_std": np.std(v[-5:]) if len(v) >= 5 else 0.0
                } for k, v in self.signal_history.items()
            }
        }

    def get_state(self) -> Dict[str, Any]:
        """Get serializable state for persistence"""
        return {
            "positions": copy.deepcopy(self.open_positions),
            "max_pct": float(self.max_pct),
            "consecutive_losses": int(self.consecutive_losses),
            "position_confidence": copy.deepcopy(self.position_confidence),
            "signal_history": {k: v[-20:] for k, v in self.signal_history.items()},  # Keep last 20
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialized data"""
        self.open_positions = copy.deepcopy(state.get("positions", {}))
        self.max_pct = float(state.get("max_pct", self.default_max_pct))
        self.consecutive_losses = int(state.get("consecutive_losses", 0))
        self.position_confidence = copy.deepcopy(state.get("position_confidence", {}))
        
        # Restore signal history
        saved_history = state.get("signal_history", {})
        for inst in self.instruments:
            if inst in saved_history:
                self.signal_history[inst] = saved_history[inst]
            else:
                self.signal_history[inst] = []
        
        if self.debug:
            self.logger.info(
                f"Restored state: max_pct={self.max_pct}, "
                f"losses={self.consecutive_losses}, "
                f"positions={len(self.open_positions)}, "
                f"avg_confidence={np.mean(list(self.position_confidence.values())) if self.position_confidence else 0:.3f}"
            )