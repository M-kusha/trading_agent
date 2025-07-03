# ─────────────────────────────────────────────────────────────
# File: modules/position/position_manager.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
import copy
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import datetime
import random

# Optional live-broker connector
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Still works in back-test / unit-test mode

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import TradingMixin, RiskMixin, AnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, extract_standard_context
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
    session: str = "unknown"
    
    # Portfolio context
    current_exposure: float = 0.0
    drawdown: float = 0.0
    balance: float = 10000.0
    
    # InfoBus context
    step_idx: int = 0
    timestamp: str = ""


@dataclass
class PositionDecisionResult:
    """Result of position decision process"""
    decision: PositionDecision
    intensity: float  # 0.0 to 1.0
    size: float
    confidence: float
    rationale: Dict[str, Any]
    risk_factors: Dict[str, float]
    context: SignalContext


class PositionManager(Module, TradingMixin, RiskMixin, AnalysisMixin):

    def __init__(
        self,
        initial_balance: float,
        instruments: List[str],
        max_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        loss_reduction: float = 0.2,
        max_instrument_concentration: float = 0.25,
        min_volatility: float = 0.015,
        hard_loss_eur: float = 30.0,
        trail_pct: float = 0.10,
        trail_abs_eur: float = 10.0,
        pips_tolerance: int = 20,
        min_size_pct: float = 0.01,
        min_signal_threshold: float = 0.15,
        position_scale_threshold: float = 0.30,
        emergency_close_threshold: float = 0.85,
        confidence_decay: float = 0.95,
        debug: bool = True,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # ─── PREPARE ALL ATTRIBUTES NEEDED BY _initialize_module_state ───
        self.instruments = instruments
        self.initial_balance = float(initial_balance)
        self.max_pct = float(max_pct)
        self.default_max_pct = self.max_pct
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.loss_reduction = float(loss_reduction)
        self.max_instrument_concentration = float(max_instrument_concentration)
        self.min_volatility = float(min_volatility)
        self.hard_loss_eur = float(hard_loss_eur)
        self.trail_pct = float(trail_pct)
        self.trail_abs_eur = float(trail_abs_eur)
        self.pips_tolerance = int(pips_tolerance)
        self.min_size_pct = float(min_size_pct)
        self.min_signal_threshold = float(min_signal_threshold)
        self.position_scale_threshold = float(position_scale_threshold)
        self.emergency_close_threshold = float(emergency_close_threshold)
        self.confidence_decay = float(confidence_decay)
        # Store genome for later evolution (optional)
        self.genome = genome or {}

        # ─── NOW INITIALIZE THE BASE MODULE (which calls your _initialize_module_state) ───
        config = ModuleConfig(
            debug=debug,
            max_history=200,
            **kwargs
        )
        super().__init__(config)

        # ─── FOLLOW-UP SETUP ───────────────────────────────────────────────
        # (You can re-run your genome‐based override here if desired)
        if genome:
            self._initialize_genome_parameters(
                genome, self.initial_balance, self.max_pct,
                self.max_consecutive_losses, self.loss_reduction,
                self.max_instrument_concentration, self.min_volatility,
                self.hard_loss_eur, self.trail_pct, self.trail_abs_eur,
                self.min_signal_threshold, self.position_scale_threshold,
                self.emergency_close_threshold, self.confidence_decay
            )

        # Trading‐mode manager and position tracking
        self.mode_manager = TradingModeManager(initial_mode="safe", window=50)
        self._initialize_position_tracking()
        self.env = None

        self.log_operator_info(
            "Enhanced position manager initialized",
            instruments_count=len(self.instruments),
            initial_balance=f"€{self.initial_balance:,.0f}",
            max_position_pct=f"{self.max_pct:.1%}",
            min_signal_threshold=f"{self.min_signal_threshold:.2f}",
            max_consecutive_losses=self.max_consecutive_losses,
            hard_loss_limit=f"€{self.hard_loss_eur:.0f}",
            trail_stop_pct=f"{self.trail_pct:.1%}"
        )


    def _initialize_genome_parameters(self, genome: Optional[Dict], initial_balance: float,
                                    max_pct: float, max_consecutive_losses: int,
                                    loss_reduction: float, max_instrument_concentration: float,
                                    min_volatility: float, hard_loss_eur: float,
                                    trail_pct: float, trail_abs_eur: float,
                                    min_signal_threshold: float, position_scale_threshold: float,
                                    emergency_close_threshold: float, confidence_decay: float):
        """Initialize genome-based parameters"""
        if genome:
            self.initial_balance = float(genome.get("initial_balance", initial_balance))
            self.max_pct = float(genome.get("max_pct", max_pct))
            self.max_consecutive_losses = int(genome.get("max_consecutive_losses", max_consecutive_losses))
            self.loss_reduction = float(genome.get("loss_reduction", loss_reduction))
            self.max_instrument_concentration = float(genome.get("max_instrument_concentration", max_instrument_concentration))
            self.min_volatility = float(genome.get("min_volatility", min_volatility))
            self.hard_loss_eur = float(genome.get("hard_loss_eur", hard_loss_eur))
            self.trail_pct = float(genome.get("trail_pct", trail_pct))
            self.trail_abs_eur = float(genome.get("trail_abs_eur", trail_abs_eur))
            self.min_signal_threshold = float(genome.get("min_signal_threshold", min_signal_threshold))
            self.position_scale_threshold = float(genome.get("position_scale_threshold", position_scale_threshold))
            self.emergency_close_threshold = float(genome.get("emergency_close_threshold", emergency_close_threshold))
            self.confidence_decay = float(genome.get("confidence_decay", confidence_decay))
            self.risk_multiplier = float(genome.get("risk_multiplier", 1.0))
            self.correlation_threshold = float(genome.get("correlation_threshold", 0.7))
        else:
            self.initial_balance = float(initial_balance)
            self.max_pct = float(max_pct)
            self.max_consecutive_losses = int(max_consecutive_losses)
            self.loss_reduction = float(loss_reduction)
            self.max_instrument_concentration = float(max_instrument_concentration)
            self.min_volatility = float(min_volatility)
            self.hard_loss_eur = float(hard_loss_eur)
            self.trail_pct = float(trail_pct)
            self.trail_abs_eur = float(trail_abs_eur)
            self.min_signal_threshold = float(min_signal_threshold)
            self.position_scale_threshold = float(position_scale_threshold)
            self.emergency_close_threshold = float(emergency_close_threshold)
            self.confidence_decay = float(confidence_decay)
            self.risk_multiplier = 1.0
            self.correlation_threshold = 0.7

        # Store genome for evolution
        self.genome = {
            "initial_balance": self.initial_balance,
            "max_pct": self.max_pct,
            "max_consecutive_losses": self.max_consecutive_losses,
            "loss_reduction": self.loss_reduction,
            "max_instrument_concentration": self.max_instrument_concentration,
            "min_volatility": self.min_volatility,
            "hard_loss_eur": self.hard_loss_eur,
            "trail_pct": self.trail_pct,
            "trail_abs_eur": self.trail_abs_eur,
            "min_signal_threshold": self.min_signal_threshold,
            "position_scale_threshold": self.position_scale_threshold,
            "emergency_close_threshold": self.emergency_close_threshold,
            "confidence_decay": self.confidence_decay,
            "risk_multiplier": self.risk_multiplier,
            "correlation_threshold": self.correlation_threshold
        }
        
        # Dynamic parameters (reset on each episode)
        self.default_max_pct = self.max_pct

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_trading_state()
        self._initialize_risk_state()
        self._initialize_analysis_state()
        
        # Position manager specific state
        self.consecutive_losses = 0
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        
        # Enhanced tracking
        self._decision_history = deque(maxlen=100)
        self._portfolio_health_history = deque(maxlen=50)
        self._exposure_history = deque(maxlen=100)
        self._performance_analytics = defaultdict(list)
        
        # Decision tracking
        self.last_decisions: Dict[str, PositionDecisionResult] = {}
        self.position_confidence: Dict[str, float] = {}
        self.signal_history: Dict[str, List[float]] = {inst: [] for inst in self.instruments}
        
        # Performance metrics
        self._portfolio_health_score = 1.0
        self._total_exposure_ratio = 0.0
        self._decision_quality_score = 0.5
        self._risk_management_score = 1.0
        
        # Adaptive parameters
        self._adaptive_params = {
            'dynamic_max_pct': self.max_pct,
            'signal_sensitivity': 1.0,
            'risk_tolerance': 1.0,
            'confidence_threshold': 0.5
        }

    def _initialize_position_tracking(self):
        """Initialize position-specific tracking"""
        
        # Position metadata tracking
        self._position_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking per position
        self._position_performance: Dict[str, Dict[str, Any]] = {}
        
        # Exit rule tracking
        self._exit_signals: Dict[str, List[Dict[str, Any]]] = {}
        
        # Live trading state
        self._forced_action = None
        self._forced_conf = None
        self._last_sync_time = None

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_trading_state()
        self._reset_risk_state()
        self._reset_analysis_state()
        
        # Reset position manager state
        self.max_pct = self.default_max_pct
        self.consecutive_losses = 0
        self.open_positions.clear()
        self.last_decisions.clear()
        self.position_confidence.clear()
        
        # Reset tracking
        self._decision_history.clear()
        self._portfolio_health_history.clear()
        self._exposure_history.clear()
        self._performance_analytics.clear()
        
        # Reset signal history
        for inst in self.instruments:
            self.signal_history[inst].clear()
        
        # Reset position tracking
        self._position_metadata.clear()
        self._position_performance.clear()
        self._exit_signals.clear()
        
        # Reset forced values
        self._forced_action = None
        self._forced_conf = None
        
        # Reset performance metrics
        self._portfolio_health_score = 1.0
        self._total_exposure_ratio = 0.0
        self._decision_quality_score = 0.5
        self._risk_management_score = 1.0
        
        # Reset adaptive parameters
        self._adaptive_params = {
            'dynamic_max_pct': self.max_pct,
            'signal_sensitivity': 1.0,
            'risk_tolerance': 1.0,
            'confidence_threshold': 0.5
        }
        
        # Ensure minimum allocation capability
        if self.max_pct < 1e-5:
            self.log_operator_warning(
                f"max_pct was {self.max_pct:.6f} at reset, restoring to default",
                default_max_pct=f"{self.default_max_pct:.4f}"
            )
            self.max_pct = self.default_max_pct

    def set_env(self, env: Any):
        """Set environment reference"""
        self.env = env

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Set environment if provided
        env = kwargs.get("env", None)
        if env:
            self.env = env
        
        # Process market signals if InfoBus available
        if info_bus:
            decisions = self.process_market_signals_from_info_bus(info_bus)
            self._track_decision_performance(decisions, info_bus)
        
        # Apply position management rules
        self._apply_position_management()
        
        # Update performance metrics
        self._update_position_performance()

    def process_market_signals_from_info_bus(self, info_bus: InfoBus) -> Dict[str, PositionDecisionResult]:
        """Main entry point for processing InfoBus market signals into position decisions"""
        
        # Extract market data from InfoBus
        market_data = self._extract_market_data_from_info_bus(info_bus)
        
        # Process using enhanced signal processing
        decisions = self.process_market_signals(market_data, info_bus)
        
        # Update InfoBus with position decisions
        self._update_info_bus_with_decisions(info_bus, decisions)
        
        return decisions

    def _extract_market_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract market data from InfoBus for position decision making"""
        
        market_data = {}
        
        # Extract standard context
        context = extract_standard_context(info_bus)
        
        # Extract prices and market context
        prices = info_bus.get('prices', {})
        market_context = info_bus.get('market_context', {})
        votes = info_bus.get('votes', [])
        
        # Process per instrument
        for instrument in self.instruments:
            inst_data = {}
            
            # Price information
            current_price = prices.get(instrument, 0.0)
            inst_data['current_price'] = current_price
            
            # Extract intensity from votes
            inst_votes = [v for v in votes if v.get('instrument') == instrument]
            if inst_votes:
                # Average vote intensity
                intensities = []
                for vote in inst_votes:
                    action = vote.get('action', 0)
                    if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                        intensities.append(action[0])
                    else:
                        intensities.append(float(action) if action else 0.0)
                
                inst_data['intensity'] = np.mean(intensities) if intensities else 0.0
                inst_data['vote_count'] = len(inst_votes)
                inst_data['avg_confidence'] = np.mean([v.get('confidence', 0.5) for v in inst_votes])
            else:
                inst_data['intensity'] = 0.0
                inst_data['vote_count'] = 0
                inst_data['avg_confidence'] = 0.5
            
            # Market regime and volatility
            inst_data['regime'] = context.get('regime', 'unknown')
            inst_data['session'] = context.get('session', 'unknown')
            inst_data['volatility_level'] = context.get('volatility_level', 'medium')
            
            # Extract volatility value
            if isinstance(market_context.get('volatility'), dict):
                inst_data['volatility'] = market_context['volatility'].get(instrument, self.min_volatility)
            else:
                inst_data['volatility'] = self.min_volatility
            
            # Extract trend and momentum from market context
            if isinstance(market_context.get('trend_strength'), dict):
                inst_data['trend_strength'] = market_context['trend_strength'].get(instrument, 0.0)
            else:
                inst_data['trend_strength'] = 0.0
            
            inst_data['momentum'] = 0.0  # Can be enhanced with more data
            inst_data['volume_profile'] = 1.0  # Default
            
            market_data[instrument] = inst_data
        
        return market_data

    def _update_info_bus_with_decisions(self, info_bus: InfoBus, decisions: Dict[str, PositionDecisionResult]):
        """Update InfoBus with position manager decisions"""
        
        # Add position manager data to module_data
        if 'module_data' not in info_bus:
            info_bus['module_data'] = {}
        
        info_bus['module_data']['position_manager'] = {
            'decisions': {
                inst: {
                    'decision': result.decision.value,
                    'intensity': result.intensity,
                    'size': result.size,
                    'confidence': result.confidence,
                    'rationale': result.rationale
                }
                for inst, result in decisions.items()
            },
            'portfolio_health': self._portfolio_health_score,
            'total_exposure': self._total_exposure_ratio,
            'consecutive_losses': self.consecutive_losses,
            'position_count': len(self.open_positions)
        }

    def process_market_signals(self, market_data: Dict[str, Any], 
                             info_bus: Optional[InfoBus] = None) -> Dict[str, PositionDecisionResult]:
        """
        Main entry point for processing market signals into position decisions.
        
        This method implements the hierarchical decision making:
        1. Strategic: Assess market regime and portfolio health
        2. Tactical: Make instrument-specific decisions  
        3. Execution: Apply risk management and sizing
        """
        decisions = {}
        
        # 1. Strategic Layer: Portfolio-level assessment
        portfolio_health = self._assess_portfolio_health(info_bus)
        market_regime = self._assess_market_regime(market_data)
        
        self.log_operator_info(
            f"Portfolio assessment",
            health_score=f"{portfolio_health['overall_health']:.3f}",
            market_regime=market_regime,
            exposure_ratio=f"{portfolio_health['exposure_ratio']:.2%}",
            consecutive_losses=self.consecutive_losses
        )
        
        # 2. Tactical Layer: Per-instrument decisions
        for instrument in self.instruments:
            # Extract signal context for this instrument
            signal_context = self._extract_signal_context(
                instrument, market_data, portfolio_health, info_bus
            )
            
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
                self.log_operator_info(
                    f"Position decision: {instrument}",
                    decision=decision_result.decision.value,
                    intensity=f"{decision_result.intensity:.3f}",
                    size=f"€{decision_result.size:.0f}",
                    confidence=f"{decision_result.confidence:.3f}",
                    rationale=decision_result.rationale.get('stage', 'unknown')
                )
        
        return decisions

    def _assess_portfolio_health(self, info_bus: Optional[InfoBus] = None) -> Dict[str, float]:
        """Assess overall portfolio health metrics"""
        
        # Extract balance and drawdown
        if info_bus:
            risk_snapshot = info_bus.get('risk', {})
            balance = risk_snapshot.get('balance', self.initial_balance)
            drawdown = risk_snapshot.get('current_drawdown', 0.0)
        elif self.env:
            balance = getattr(self.env, 'balance', self.initial_balance)
            drawdown = getattr(self.env, 'current_drawdown', 0.0)
        else:
            balance = self.initial_balance
            drawdown = 0.0
        
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
        
        # Risk management score
        risk_violations = len([alert for alert in self._risk_alerts if alert.get('severity') in ['HIGH', 'CRITICAL']])
        risk_health = max(0.1, 1.0 - risk_violations / 10.0)
        
        overall_health = (dd_health + exposure_health + streak_health + risk_health) / 4.0
        
        health_metrics = {
            "drawdown_health": dd_health,
            "exposure_health": exposure_health,
            "streak_health": streak_health,
            "risk_health": risk_health,
            "overall_health": overall_health,
            "total_exposure": total_exposure,
            "exposure_ratio": exposure_ratio,
            "balance": balance,
            "drawdown": drawdown
        }
        
        # Update portfolio health score
        self._portfolio_health_score = overall_health
        self._total_exposure_ratio = exposure_ratio
        
        # Track portfolio health history
        self._portfolio_health_history.append({
            'timestamp': datetime.datetime.now().isoformat(),
            'health_score': overall_health,
            'exposure_ratio': exposure_ratio,
            'drawdown': drawdown,
            'balance': balance
        })
        
        return health_metrics

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

    def _extract_signal_context(self, instrument: str, market_data: Dict[str, Any], 
                               portfolio_health: Dict[str, float], 
                               info_bus: Optional[InfoBus] = None) -> SignalContext:
        """Extract and structure signal context for decision making"""
        
        inst_data = market_data.get(instrument, {})
        
        # Extract market signals
        market_intensity = float(inst_data.get('intensity', 0.0))
        market_direction = int(np.sign(market_intensity))
        volatility = max(float(inst_data.get('volatility', self.min_volatility)), self.min_volatility)
        trend_strength = float(inst_data.get('trend_strength', 0.0))
        momentum = float(inst_data.get('momentum', 0.0))
        volume_profile = float(inst_data.get('volume_profile', 1.0))
        
        # Market regime context
        regime = inst_data.get('regime', 'normal')
        session = inst_data.get('session', 'unknown')
        
        # Calculate correlation penalty
        correlation_penalty = 0.0
        if info_bus:
            # Extract correlation from InfoBus if available
            try:
                correlation_data = info_bus.get('module_data', {}).get('correlation', {})
                if correlation_data:
                    correlation_penalty = min(abs(correlation_data.get(instrument, 0.0)) * 0.5, 0.8)
            except:
                correlation_penalty = 0.0
        
        # Get liquidity score
        liquidity_score = 1.0
        if info_bus:
            market_status = info_bus.get('market_status', {})
            liquidity_score = market_status.get('liquidity_score', 1.0)
        
        # Create context with InfoBus data
        context = SignalContext(
            instrument=instrument,
            market_intensity=market_intensity,
            market_direction=market_direction,
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            volume_profile=volume_profile,
            correlation_penalty=correlation_penalty,
            regime=regime,
            liquidity_score=liquidity_score,
            session=session,
            current_exposure=portfolio_health["exposure_ratio"],
            drawdown=portfolio_health["drawdown"],
            balance=portfolio_health["balance"],
            step_idx=info_bus.get('step_idx', 0) if info_bus else 0,
            timestamp=info_bus.get('timestamp', datetime.datetime.now().isoformat()) if info_bus else datetime.datetime.now().isoformat()
        )
        
        return context

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
                
                self.log_operator_warning(
                    f"Emergency close triggered: {instrument}",
                    drawdown=f"{context.drawdown:.1%}",
                    consecutive_losses=self.consecutive_losses,
                    exposure=f"{context.current_exposure:.1%}"
                )
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors, context)
        
        # Stage 2: Signal strength filtering
        if signal_strength < self.min_signal_threshold:
            rationale["stage"] = "signal_filter"
            rationale["factors"].append(f"Signal strength {signal_strength:.3f} below threshold {self.min_signal_threshold}")
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors, context)
        
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
                rationale["factors"].append("Closing due to poor portfolio health")
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors, context)
        
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
                rationale["factors"].append(f"Position approaching loss limit: €{position_pnl:.2f}")
        
        # Stage 5: Final risk adjustments
        risk_factors = self._assess_risk_factors(context)
        final_intensity = intensity * (1.0 - max(risk_factors.values()) if risk_factors else 1.0)
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
            risk_factors=risk_factors,
            context=context
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
        
        # Session risk (trading outside optimal hours)
        session_risk = 0.0
        if context.session == "closed":
            session_risk = 0.3
        elif context.session == "asian":
            session_risk = 0.1  # Lower liquidity
        risk_factors["session"] = session_risk
        
        return risk_factors

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
        risk_pct = max(self._adaptive_params['dynamic_max_pct'], 0.01)  # Ensure minimum risk allocation
        risk_budget = balance * risk_pct
        
        # Volatility-adjusted base size
        vol_adjusted_budget = risk_budget / volatility
        base_size = intensity * vol_adjusted_budget
        
        # Apply portfolio health modifiers
        portfolio_health = self._portfolio_health_score
        
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
            self.log_operator_info(f"Loss streak reduction applied", reduction_factor=f"{streak_reduction:.2f}")
        
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

    def _apply_position_management(self):
        """Apply position management rules including live sync and exits"""
        
        # Ensure minimum allocation capability
        min_cap = 0.01  # 1% minimal allocation
        if self._adaptive_params['dynamic_max_pct'] < min_cap:
            self.log_operator_warning(
                f"Dynamic max_pct below minimum",
                current=f"{self._adaptive_params['dynamic_max_pct']:.6f}",
                minimum=f"{min_cap:.4f}"
            )
            self._adaptive_params['dynamic_max_pct'] = self.default_max_pct

        # Live‐mode: sync & apply exit rules
        if self.env and getattr(self.env, "live_mode", False):
            self._sync_live_positions()
            self._apply_exit_rules()

        # Decay position confidence over time
        for inst in self.position_confidence:
            self.position_confidence[inst] *= self.confidence_decay

    def _track_decision_performance(self, decisions: Dict[str, PositionDecisionResult], 
                                   info_bus: InfoBus):
        """Track decision performance for continuous improvement"""
        
        decision_record = {
            'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
            'step_idx': info_bus.get('step_idx', 0),
            'decisions': {
                inst: {
                    'decision': result.decision.value,
                    'intensity': result.intensity,
                    'confidence': result.confidence,
                    'stage': result.rationale.get('stage', 'unknown')
                }
                for inst, result in decisions.items()
            },
            'portfolio_health': self._portfolio_health_score,
            'total_exposure': self._total_exposure_ratio,
            'consecutive_losses': self.consecutive_losses
        }
        
        self._decision_history.append(decision_record)
        
        # Update decision quality metrics
        active_decisions = [d for d in decisions.values() if d.decision != PositionDecision.HOLD]
        if active_decisions:
            avg_confidence = np.mean([d.confidence for d in active_decisions])
            self._decision_quality_score = avg_confidence
            
        # Update performance metrics
        self._update_performance_metric('decision_quality', self._decision_quality_score)
        self._update_performance_metric('portfolio_health', self._portfolio_health_score)

    def _update_position_performance(self):
        """Update comprehensive position performance metrics"""
        
        try:
            # Calculate current exposure
            current_exposure = self._calculate_current_exposure_ratio()
            self._total_exposure_ratio = current_exposure
            
            # Update exposure history
            self._exposure_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'exposure_ratio': current_exposure,
                'position_count': len(self.open_positions),
                'consecutive_losses': self.consecutive_losses
            })
            
            # Calculate risk management score
            recent_risk_alerts = len([a for a in self._risk_alerts if a.get('severity') in ['HIGH', 'CRITICAL']])
            self._risk_management_score = max(0.1, 1.0 - recent_risk_alerts / 5.0)
            
            # Update adaptive parameters based on performance
            self._adapt_parameters()
            
            # Update performance metrics
            self._update_performance_metric('total_exposure', current_exposure)
            self._update_performance_metric('risk_management_score', self._risk_management_score)
            self._update_performance_metric('consecutive_losses', self.consecutive_losses)
            
        except Exception as e:
            self.log_operator_warning(f"Position performance update failed: {e}")

    def _adapt_parameters(self):
        """Adapt position management parameters based on performance"""
        
        try:
            # Adapt max_pct based on recent performance
            if len(self._decision_history) >= 10:
                recent_decisions = list(self._decision_history)[-10:]
                avg_portfolio_health = np.mean([d['portfolio_health'] for d in recent_decisions])
                
                if avg_portfolio_health > 0.8:
                    # Good performance, slightly increase risk tolerance
                    self._adaptive_params['dynamic_max_pct'] = min(
                        self.max_pct * 1.1, 
                        self.max_pct * 1.5
                    )
                elif avg_portfolio_health < 0.4:
                    # Poor performance, reduce risk
                    self._adaptive_params['dynamic_max_pct'] = max(
                        self.max_pct * 0.7,
                        self.max_pct * 0.3
                    )
                else:
                    # Neutral performance, gradual return to default
                    current = self._adaptive_params['dynamic_max_pct']
                    self._adaptive_params['dynamic_max_pct'] = current * 0.95 + self.max_pct * 0.05
            
            # Adapt signal sensitivity based on decision success
            if len(self._decision_history) >= 5:
                recent_confidences = []
                for decision_record in list(self._decision_history)[-5:]:
                    for decision_data in decision_record['decisions'].values():
                        if decision_data['decision'] != 'hold':
                            recent_confidences.append(decision_data['confidence'])
                
                if recent_confidences:
                    avg_confidence = np.mean(recent_confidences)
                    if avg_confidence > 0.7:
                        self._adaptive_params['signal_sensitivity'] = min(1.2, self._adaptive_params['signal_sensitivity'] * 1.02)
                    elif avg_confidence < 0.4:
                        self._adaptive_params['signal_sensitivity'] = max(0.7, self._adaptive_params['signal_sensitivity'] * 0.98)
            
            # Adapt risk tolerance based on consecutive losses
            if self.consecutive_losses == 0:
                self._adaptive_params['risk_tolerance'] = min(1.3, self._adaptive_params['risk_tolerance'] * 1.01)
            elif self.consecutive_losses >= 3:
                self._adaptive_params['risk_tolerance'] = max(0.5, self._adaptive_params['risk_tolerance'] * 0.95)
                
        except Exception as e:
            self.log_operator_warning(f"Parameter adaptation failed: {e}")

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
                self.log_operator_warning(f"Broker position sync failed: {exc}")

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
        self._last_sync_time = datetime.datetime.now()
        
        self.log_operator_info(f"Live positions synced", position_count=len(self.open_positions))

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
                self.log_operator_info(f"Position closed via broker: {inst}", reason=reason)
                self.open_positions.pop(inst, None)
            else:
                self.log_operator_error(f"Broker close failed: {inst}", reason=reason)
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
                self.log_operator_info(f"Position closed via MT5: {inst}", ticket=data["ticket"], reason=reason)
                self.open_positions.pop(inst, None)
            else:
                self.log_operator_error(f"MT5 close failed: {inst}", error=str(res))
            return

        # backtest fallback
        self.log_operator_info(f"Position marked closed (simulation): {inst}", reason=reason)
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
            if tick:
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

    # ════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with position metrics"""
        
        try:
            # Portfolio health metrics
            portfolio_health = self._portfolio_health_score
            total_exposure = self._total_exposure_ratio
            decision_quality = self._decision_quality_score
            
            # Position statistics
            position_count = len(self.open_positions)
            avg_position_confidence = np.mean(list(self.position_confidence.values())) if self.position_confidence else 0.5
            
            # Risk metrics
            consecutive_losses_ratio = self.consecutive_losses / max(self.max_consecutive_losses, 1)
            risk_management_score = self._risk_management_score
            
            # Adaptive parameters
            dynamic_risk_ratio = self._adaptive_params['dynamic_max_pct'] / self.max_pct
            signal_sensitivity = self._adaptive_params['signal_sensitivity']
            
            # Recent performance indicators
            recent_decision_count = 0
            if self._decision_history:
                recent_decisions = list(self._decision_history)[-5:]
                recent_decision_count = sum(
                    len([d for d in record['decisions'].values() if d['decision'] != 'hold'])
                    for record in recent_decisions
                ) / max(len(recent_decisions), 1)
            
            # Balance information
            if self.env:
                balance = getattr(self.env, 'balance', self.initial_balance)
                drawdown = getattr(self.env, 'current_drawdown', 0.0)
            else:
                balance = self.initial_balance
                drawdown = 0.0
            
            balance_ratio = balance / self.initial_balance
            
            # Combine all components
            observation = np.array([
                portfolio_health,
                total_exposure,
                decision_quality,
                float(position_count) / 10.0,  # Normalize
                avg_position_confidence,
                consecutive_losses_ratio,
                risk_management_score,
                dynamic_risk_ratio,
                signal_sensitivity,
                recent_decision_count / 5.0,  # Normalize
                balance_ratio,
                drawdown
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
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
        
        # Extract market data from observation or InfoBus
        if info_bus:
            market_data = self._extract_market_data_from_info_bus(info_bus)
        else:
            market_data = self._extract_market_data_from_obs(obs)
        
        # Process signals through hierarchical decision making
        decisions = self.process_market_signals(market_data, info_bus)
        
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
            
            self.log_operator_info(
                f"Action proposal: {inst}",
                intensity=f"{intensity:.3f}",
                duration=f"{duration:.1f}",
                decision=decision_result.decision.value if decision_result else "hold"
            )
            
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

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """
        Calculate overall confidence based on portfolio and position health.
        Now independent of circular dependencies.
        """
        if self._forced_conf is not None:
            return float(self._forced_conf)

        # Calculate confidence components
        confidence_components = []
        
        # Portfolio health confidence
        confidence_components.append(self._portfolio_health_score)
        
        # Position-specific confidence
        if self.position_confidence:
            avg_position_conf = np.mean(list(self.position_confidence.values()))
            confidence_components.append(avg_position_conf)
        else:
            confidence_components.append(0.5)  # Neutral when no positions
        
        # Signal quality confidence
        signal_quality = self._assess_signal_quality()
        confidence_components.append(signal_quality)
        
        # Risk management confidence
        confidence_components.append(self._risk_management_score)
        
        # Mode-based confidence
        mode = getattr(self.mode_manager, "current_mode", "safe")
        mode_confidence = 0.8 if mode == "safe" else (0.9 if mode == "aggressive" else 0.6)
        confidence_components.append(mode_confidence)
        
        # Decision quality confidence
        confidence_components.append(self._decision_quality_score)
        
        # Calculate weighted average
        weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]  # Portfolio, positions, signals, risk, mode, decisions
        final_confidence = np.average(confidence_components, weights=weights)
        
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

    # ════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome"""
        self.max_pct = float(np.clip(genome.get("max_pct", self.max_pct), 0.01, 0.25))
        self.max_consecutive_losses = int(np.clip(genome.get("max_consecutive_losses", self.max_consecutive_losses), 1, 20))
        self.loss_reduction = float(np.clip(genome.get("loss_reduction", self.loss_reduction), 0.05, 1.0))
        self.max_instrument_concentration = float(np.clip(genome.get("max_instrument_concentration", self.max_instrument_concentration), 0.05, 0.5))
        self.min_volatility = float(np.clip(genome.get("min_volatility", self.min_volatility), 0.001, 0.10))
        self.hard_loss_eur = float(np.clip(genome.get("hard_loss_eur", self.hard_loss_eur), 10.0, 100.0))
        self.trail_pct = float(np.clip(genome.get("trail_pct", self.trail_pct), 0.05, 0.3))
        self.trail_abs_eur = float(np.clip(genome.get("trail_abs_eur", self.trail_abs_eur), 5.0, 50.0))
        self.min_signal_threshold = float(np.clip(genome.get("min_signal_threshold", self.min_signal_threshold), 0.05, 0.5))
        self.position_scale_threshold = float(np.clip(genome.get("position_scale_threshold", self.position_scale_threshold), 0.2, 0.8))
        self.emergency_close_threshold = float(np.clip(genome.get("emergency_close_threshold", self.emergency_close_threshold), 0.7, 0.95))
        self.confidence_decay = float(np.clip(genome.get("confidence_decay", self.confidence_decay), 0.90, 0.99))
        self.risk_multiplier = float(np.clip(genome.get("risk_multiplier", self.risk_multiplier), 0.5, 2.0))
        self.correlation_threshold = float(np.clip(genome.get("correlation_threshold", self.correlation_threshold), 0.3, 0.9))
        
        self.genome = {
            "max_pct": self.max_pct,
            "max_consecutive_losses": self.max_consecutive_losses,
            "loss_reduction": self.loss_reduction,
            "max_instrument_concentration": self.max_instrument_concentration,
            "min_volatility": self.min_volatility,
            "hard_loss_eur": self.hard_loss_eur,
            "trail_pct": self.trail_pct,
            "trail_abs_eur": self.trail_abs_eur,
            "min_signal_threshold": self.min_signal_threshold,
            "position_scale_threshold": self.position_scale_threshold,
            "emergency_close_threshold": self.emergency_close_threshold,
            "confidence_decay": self.confidence_decay,
            "risk_multiplier": self.risk_multiplier,
            "correlation_threshold": self.correlation_threshold
        }
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance-based adjustments"""
        g = self.genome.copy()
        mutations = []
        
        if np.random.rand() < mutation_rate:
            old_val = g["max_pct"]
            g["max_pct"] = float(np.clip(old_val + np.random.uniform(-0.02, 0.02), 0.01, 0.25))
            mutations.append(f"max_pct: {old_val:.3f} → {g['max_pct']:.3f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["max_consecutive_losses"]
            g["max_consecutive_losses"] = int(np.clip(old_val + np.random.choice([-1, 0, 1]), 1, 20))
            mutations.append(f"max_losses: {old_val} → {g['max_consecutive_losses']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["loss_reduction"]
            g["loss_reduction"] = float(np.clip(old_val + np.random.uniform(-0.1, 0.1), 0.05, 1.0))
            mutations.append(f"loss_reduction: {old_val:.2f} → {g['loss_reduction']:.2f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["min_signal_threshold"]
            g["min_signal_threshold"] = float(np.clip(old_val + np.random.uniform(-0.05, 0.05), 0.05, 0.5))
            mutations.append(f"signal_threshold: {old_val:.2f} → {g['min_signal_threshold']:.2f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["hard_loss_eur"]
            g["hard_loss_eur"] = float(np.clip(old_val + np.random.uniform(-5, 5), 10.0, 100.0))
            mutations.append(f"hard_loss: €{old_val:.0f} → €{g['hard_loss_eur']:.0f}")
        
        if mutations:
            self.log_operator_info(f"Position manager mutation applied", changes=", ".join(mutations))
            
        self.set_genome(g)
        
    def crossover(self, other: "PositionManager") -> "PositionManager":
        """Enhanced crossover with performance-based selection"""
        if not isinstance(other, PositionManager):
            self.log_operator_warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_performance = self._portfolio_health_score
        other_performance = other._portfolio_health_score
        
        # Favor higher performance parent
        if self_performance > other_performance:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {k: (self.genome[k] if np.random.rand() < bias else other.genome[k]) for k in self.genome}
        
        child = PositionManager(
            initial_balance=self.initial_balance,
            instruments=self.instruments,
            genome=new_g,
            debug=self.config.debug
        )
        
        # Inherit beneficial state from better parent
        if self_performance > other_performance:
            if self.signal_history:
                child.signal_history = copy.deepcopy(self.signal_history)
        else:
            if other.signal_history:
                child.signal_history = copy.deepcopy(other.signal_history)
        
        return child

    # ════════════════════════════════════════════════════════════════
    # API AND INTERFACE METHODS
    # ════════════════════════════════════════════════════════════════

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

    def get_last_rationale(self) -> Dict[str, Any]:
        """Get rationale from last decision"""
        rationales = {}
        for inst, decision in self.last_decisions.items():
            rationales[inst] = decision.rationale
        return rationales

    def get_full_audit(self) -> Dict[str, Any]:
        """Get comprehensive audit information"""
        return {
            "positions": copy.deepcopy(self.open_positions),
            "last_decisions": {k: {
                "decision": v.decision.value,
                "intensity": v.intensity,
                "confidence": v.confidence,
                "rationale": v.rationale,
                "risk_factors": v.risk_factors
            } for k, v in self.last_decisions.items()},
            "position_confidence": copy.deepcopy(self.position_confidence),
            "consecutive_losses": self.consecutive_losses,
            "adaptive_params": copy.deepcopy(self._adaptive_params),
            "performance_metrics": {
                'portfolio_health': self._portfolio_health_score,
                'total_exposure': self._total_exposure_ratio,
                'decision_quality': self._decision_quality_score,
                'risk_management': self._risk_management_score
            },
            "signal_history_summary": {
                k: {
                    "length": len(v),
                    "recent_avg": np.mean(v[-5:]) if len(v) >= 5 else 0.0,
                    "recent_std": np.std(v[-5:]) if len(v) >= 5 else 0.0
                } for k, v in self.signal_history.items()
            },
            "genome": self.genome.copy()
        }

    # ════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check basic data consistency
            if not isinstance(self.open_positions, dict):
                return False
            if not isinstance(self.instruments, list):
                return False
            if not isinstance(self.signal_history, dict):
                return False
                
            # Check parameter bounds
            if not (0.0 < self.max_pct <= 1.0):
                return False
            if not (0.0 < self.min_volatility <= 1.0):
                return False
            if not (0 <= self.consecutive_losses <= 100):
                return False
                
            # Check adaptive parameters
            for param_name, param_value in self._adaptive_params.items():
                if not np.isfinite(param_value):
                    return False
                    
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        position_details = {
            'position_info': {
                'open_positions': len(self.open_positions),
                'total_exposure_ratio': self._total_exposure_ratio,
                'consecutive_losses': self.consecutive_losses,
                'portfolio_health_score': self._portfolio_health_score
            },
            'performance_info': {
                'decision_quality_score': self._decision_quality_score,
                'risk_management_score': self._risk_management_score,
                'signal_quality': self._assess_signal_quality()
            },
            'configuration_info': {
                'max_pct': self.max_pct,
                'min_signal_threshold': self.min_signal_threshold,
                'hard_loss_eur': self.hard_loss_eur,
                'trail_pct': self.trail_pct,
                'instruments_count': len(self.instruments)
            },
            'adaptive_params': self._adaptive_params.copy(),
            'mode_info': {
                'current_mode': getattr(self.mode_manager, "current_mode", "unknown"),
                'trading_mode_manager': self.mode_manager is not None
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(position_details)
            return base_details
        
        return position_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        return {
            "positions": copy.deepcopy(self.open_positions),
            "genome": self.genome.copy(),
            "adaptive_params": self._adaptive_params.copy(),
            "consecutive_losses": self.consecutive_losses,
            "position_confidence": copy.deepcopy(self.position_confidence),
            "signal_history": {k: v[-20:] for k, v in self.signal_history.items()},  # Keep last 20
            "performance_metrics": {
                'portfolio_health_score': self._portfolio_health_score,
                'total_exposure_ratio': self._total_exposure_ratio,
                'decision_quality_score': self._decision_quality_score,
                'risk_management_score': self._risk_management_score
            },
            "decision_history": list(self._decision_history)[-10:],  # Keep recent decisions
            "portfolio_health_history": list(self._portfolio_health_history)[-20:],
            "exposure_history": list(self._exposure_history)[-30:],
            "last_decisions": {
                k: {
                    'decision': v.decision.value,
                    'intensity': v.intensity,
                    'confidence': v.confidence,
                    'rationale': v.rationale
                } for k, v in self.last_decisions.items()
            },
            "position_metadata": copy.deepcopy(self._position_metadata),
            "mode_manager_state": {
                'current_mode': getattr(self.mode_manager, "current_mode", "safe"),
                'window': getattr(self.mode_manager, "window", 50)
            }
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore core position data
        self.open_positions = copy.deepcopy(module_state.get("positions", {}))
        self.consecutive_losses = int(module_state.get("consecutive_losses", 0))
        self.position_confidence = copy.deepcopy(module_state.get("position_confidence", {}))
        
        # Restore genome and parameters
        self.set_genome(module_state.get("genome", self.genome))
        self._adaptive_params = module_state.get("adaptive_params", self._adaptive_params)
        
        # Restore signal history
        saved_history = module_state.get("signal_history", {})
        for inst in self.instruments:
            if inst in saved_history:
                self.signal_history[inst] = saved_history[inst]
            else:
                self.signal_history[inst] = []
        
        # Restore performance metrics
        performance_metrics = module_state.get("performance_metrics", {})
        self._portfolio_health_score = performance_metrics.get('portfolio_health_score', 1.0)
        self._total_exposure_ratio = performance_metrics.get('total_exposure_ratio', 0.0)
        self._decision_quality_score = performance_metrics.get('decision_quality_score', 0.5)
        self._risk_management_score = performance_metrics.get('risk_management_score', 1.0)
        
        # Restore tracking data
        self._decision_history = deque(module_state.get("decision_history", []), maxlen=100)
        self._portfolio_health_history = deque(module_state.get("portfolio_health_history", []), maxlen=50)
        self._exposure_history = deque(module_state.get("exposure_history", []), maxlen=100)
        
        # Restore last decisions
        last_decisions_data = module_state.get("last_decisions", {})
        self.last_decisions.clear()
        for inst, decision_data in last_decisions_data.items():
            # Reconstruct PositionDecisionResult
            try:
                decision = PositionDecision(decision_data['decision'])
                rationale = decision_data.get('rationale', {})
                risk_factors = {}
                context = SignalContext(instrument=inst)  # Minimal context
                
                result = PositionDecisionResult(
                    decision=decision,
                    intensity=decision_data.get('intensity', 0.0),
                    size=0.0,  # Not stored
                    confidence=decision_data.get('confidence', 0.5),
                    rationale=rationale,
                    risk_factors=risk_factors,
                    context=context
                )
                self.last_decisions[inst] = result
            except:
                pass  # Skip invalid decisions
        
        # Restore position metadata
        self._position_metadata = copy.deepcopy(module_state.get("position_metadata", {}))
        
        # Restore mode manager state
        mode_state = module_state.get("mode_manager_state", {})
        if hasattr(self.mode_manager, 'current_mode'):
            self.mode_manager.current_mode = mode_state.get('current_mode', 'safe')

    def get_position_manager_report(self) -> str:
        """Generate operator-friendly position manager report"""
        
        # Portfolio status
        if self._portfolio_health_score > 0.8:
            portfolio_status = "🚀 Excellent"
        elif self._portfolio_health_score > 0.6:
            portfolio_status = "✅ Good"
        elif self._portfolio_health_score > 0.4:
            portfolio_status = "⚡ Fair"
        else:
            portfolio_status = "⚠️ Poor"
        
        # Risk status
        if self.consecutive_losses == 0:
            risk_status = "🟢 Safe"
        elif self.consecutive_losses < self.max_consecutive_losses // 2:
            risk_status = "🟡 Caution"
        else:
            risk_status = "🔴 High Risk"
        
        # Active decisions
        active_decisions = len([d for d in self.last_decisions.values() if d.decision != PositionDecision.HOLD])
        
        # Recent performance
        recent_avg_confidence = 0.0
        if self._decision_history:
            recent_decisions = list(self._decision_history)[-5:]
            all_confidences = []
            for record in recent_decisions:
                for decision_data in record['decisions'].values():
                    if decision_data['decision'] != 'hold':
                        all_confidences.append(decision_data['confidence'])
            recent_avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return f"""
📊 ENHANCED POSITION MANAGER
═══════════════════════════════════════
💼 Portfolio: {portfolio_status} ({self._portfolio_health_score:.3f})
⚠️ Risk Status: {risk_status}
📈 Exposure: {self._total_exposure_ratio:.1%}
📍 Open Positions: {len(self.open_positions)}

🔧 RISK PARAMETERS
• Max Position %: {self.max_pct:.1%} (dynamic: {self._adaptive_params['dynamic_max_pct']:.1%})
• Hard Loss Limit: €{self.hard_loss_eur:.0f}
• Trail Stop: {self.trail_pct:.1%} / €{self.trail_abs_eur:.0f}
• Signal Threshold: {self.min_signal_threshold:.2f}
• Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}

📊 PERFORMANCE METRICS
• Decision Quality: {self._decision_quality_score:.3f}
• Risk Management: {self._risk_management_score:.3f}
• Signal Quality: {self._assess_signal_quality():.3f}
• Recent Confidence: {recent_avg_confidence:.3f}

🎯 ADAPTIVE PARAMETERS
• Signal Sensitivity: {self._adaptive_params['signal_sensitivity']:.2f}
• Risk Tolerance: {self._adaptive_params['risk_tolerance']:.2f}
• Confidence Threshold: {self._adaptive_params['confidence_threshold']:.2f}

💡 RECENT ACTIVITY
• Active Decisions: {active_decisions}
• Decision History: {len(self._decision_history)} records
• Portfolio Health Trend: {len([h for h in self._portfolio_health_history if h['health_score'] > 0.7])} good periods
• Mode: {getattr(self.mode_manager, "current_mode", "unknown").upper()}

🔄 INSTRUMENTS ({len(self.instruments)})
{chr(10).join([f"• {inst}: {len(self.signal_history.get(inst, []))} signals, confidence: {self.position_confidence.get(inst, 0.5):.2f}" for inst in self.instruments[:5]])}
        """

    # Maintain backward compatibility
    def step(self, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, **kwargs)

    def get_state(self) -> Dict[str, Any]:
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state: Dict[str, Any]):
        """Backward compatibility state method"""
        super().set_state(state)