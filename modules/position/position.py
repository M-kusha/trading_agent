# ─────────────────────────────────────────────────────────────
# File: modules/position/position_manager.py
# Enhanced with SmartInfoBus infrastructure integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import copy
import time
import threading
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from collections import deque, defaultdict
import datetime

# Optional live-broker connector
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # Still works in back-test / unit-test mode

# New infrastructure imports
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class PositionConfig:
    """Configuration for Position Manager"""
    initial_balance: float = 10000.0
    max_pct: float = 0.10
    max_consecutive_losses: int = 5
    loss_reduction: float = 0.2
    max_instrument_concentration: float = 0.25
    min_volatility: float = 0.015
    hard_loss_eur: float = 30.0
    trail_pct: float = 0.10
    trail_abs_eur: float = 10.0
    pips_tolerance: int = 20
    min_size_pct: float = 0.01
    min_signal_threshold: float = 0.15
    position_scale_threshold: float = 0.30
    emergency_close_threshold: float = 0.85
    confidence_decay: float = 0.95
    debug: bool = True
    
    # Performance thresholds
    max_processing_time_ms: float = 100
    circuit_breaker_threshold: int = 3


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
    
    # SmartInfoBus context
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


@module(
    name="PositionManager",
    version="3.0.0",
    category="position",
    provides=[
        "position_decisions", "portfolio_state", "risk_metrics", "position_analysis",
        "positions", "pending_orders", "position_data", "trades", "recent_trades",
        "current_pnl", "balance", "equity"
    ],
    requires=["market_data", "trading_signals", "risk_score"],
    description="Advanced position management with dynamic risk scaling and portfolio optimization",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    is_voting_member=True
)
class PositionManager(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):

    def _initialize(self, **kwargs):
        """
        Real implementation of the required _initialize method for module system compatibility.
        Initializes advanced systems, genome parameters, position state, and tracking.
        Accepts optional config, instruments, genome, and env from kwargs for flexible hot-reload and orchestration.
        """
        config = kwargs.get('config', None)
        instruments = kwargs.get('instruments', None)
        genome = kwargs.get('genome', None)
        env = kwargs.get('env', None)

        # Ensure config is properly initialized
        if not hasattr(self, 'config') or self.config is None:
            self.config = config or PositionConfig()
        elif config is not None:
            self.config = config
            
        # Ensure config is the right type
        if not isinstance(self.config, PositionConfig):
            if isinstance(self.config, dict):
                self.config = PositionConfig(**self.config)
            else:
                self.config = PositionConfig()
        
        if not hasattr(self, 'instruments') or self.instruments is None:
            self.instruments = instruments or ["XAU/USD", "EUR/USD"]
        elif instruments is not None:
            self.instruments = instruments
            
        if not hasattr(self, 'genome') or self.genome is None:
            self.genome = genome or {}
        elif genome is not None:
            self.genome = genome
            
        if env is not None:
            self.env = env

        self._initialize_advanced_systems()
        self._initialize_genome_parameters(self.genome)
        self._initialize_position_state()
        self._initialize_position_tracking()
        self.logger.info(
            format_operator_message(
                "[INIT]", "POSITION_MANAGER_REINITIALIZED",
                instruments_count=len(self.instruments),
                initial_balance=f"€{self.config.initial_balance:,.0f}",
                max_position_pct=f"{self.config.max_pct:.1%}",
                details=f"PositionManager _initialize called"
            )
        )

    def __init__(
        self,
        config: Optional[PositionConfig] = None,
        instruments: Optional[List[str]] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.instruments = instruments or ["XAU/USD", "EUR/USD", ]
        
        super().__init__()
        
        # Set config after super() call to avoid BaseModule interference
        self.config = config or PositionConfig()
        
        # Ensure config is the right type
        if not isinstance(self.config, PositionConfig):
            if isinstance(self.config, dict):
                self.config = PositionConfig(**self.config)
            else:
                self.config = PositionConfig()
        
        self._initialize_advanced_systems()
        self._initialize_genome_parameters(genome)
        self._initialize_position_state()
        self._initialize_position_tracking()
        
        # Start monitoring after all initialization is complete
        self._start_monitoring()
        
        self.env = None
        
        self.logger.info(
            format_operator_message(
                "🏦", "POSITION_MANAGER_INITIALIZED",
                instruments_count=len(self.instruments),
                initial_balance=f"€{self.config.initial_balance:,.0f}",
                max_position_pct=f"{self.config.max_pct:.1%}",
                details=f"Smart position management active"
            )
        )

    def _initialize_advanced_systems(self):
        """Initialize advanced systems for position management"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="PositionManager", 
            log_path="logs/position.log", 
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PositionManager", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for position operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }

    def _start_monitoring(self):
        """Start background monitoring for position management"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', False):
                try:
                    self._update_position_health()
                    time.sleep(30)
                except:
                    pass
        
        self._monitoring_active = True
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters"""
        if genome:
            # Override config with genome values
            for key, value in genome.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # Store genome for evolution
        self.genome = genome or {}
        
        # Set derived parameters
        self.risk_multiplier = self.genome.get("risk_multiplier", 1.0)
        self.correlation_threshold = self.genome.get("correlation_threshold", 0.7)
        
        # Dynamic parameters (reset on each episode)  
        self.default_max_pct = self.config.max_pct

    def _initialize_position_state(self):
        """Initialize position management state"""
        # Core position state
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
            'dynamic_max_pct': self.config.max_pct,
            'signal_sensitivity': 1.0,
            'risk_tolerance': 1.0,
            'confidence_threshold': 0.5
        }
        
        # Live trading state
        self._forced_action = None
        self._forced_conf = None
        self._last_sync_time = None

    def _initialize_position_tracking(self):
        """Initialize position-specific tracking"""
        # Position metadata tracking
        self._position_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking per position
        self._position_performance: Dict[str, Dict[str, Any]] = {}
        
        # Exit rule tracking
        self._exit_signals: Dict[str, List[Dict[str, Any]]] = {}

    async def process(self, **inputs) -> Dict[str, Any]:
        """Main processing method for SmartInfoBus integration"""
        start_time = time.time()
        
        try:
            # Extract market data from SmartInfoBus
            market_data = self._extract_market_data_from_smartbus()
            
            if not market_data:
                return self._create_fallback_response("No market data available")
            
            # Process position decisions
            decisions = self.process_market_signals(market_data)
            
            # Update SmartInfoBus with results
            await self._update_smartbus_with_decisions(decisions)
            
            # Generate thesis
            thesis = await self._generate_position_thesis(market_data, decisions)
            
            # Performance tracking
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('PositionManager', 'process', processing_time, True)
            
            return {
                'decisions': decisions,
                'portfolio_health': self._portfolio_health_score,
                'exposure_ratio': self._total_exposure_ratio,
                'processing_time_ms': processing_time,
                'thesis': thesis
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "PositionManager")
            self.logger.error(f"Position processing failed: {e}")
            return self._create_fallback_response(f"Processing error: {str(e)}")

    def _extract_market_data_from_smartbus(self) -> Optional[Dict[str, Any]]:
        """Extract market data from SmartInfoBus"""
        market_data = {}
        
        # Get trading signals
        trading_signal = self.smart_bus.get('trading_signal', 'PositionManager')
        if trading_signal:
            market_data['trading_signal'] = trading_signal
        
        # Get risk data
        risk_score = self.smart_bus.get('risk_score', 'PositionManager')
        if risk_score:
            market_data['risk_score'] = risk_score
        
        # Get market regime
        market_regime = self.smart_bus.get('market_regime', 'PositionManager')
        if market_regime:
            market_data['market_regime'] = market_regime
        
        # Get per-instrument data
        for instrument in self.instruments:
            inst_data = {}
            
            # Get market data for instrument
            price_data = self.smart_bus.get(f'price_{instrument}', 'PositionManager')
            if price_data:
                inst_data['current_price'] = price_data.get('price', 0.0)
                inst_data['volatility'] = price_data.get('volatility', self.config.min_volatility)
            
            # Get technical indicators
            indicators = self.smart_bus.get(f'indicators_{instrument}', 'PositionManager')
            if indicators:
                inst_data['trend_strength'] = indicators.get('trend_strength', 0.0)
                inst_data['momentum'] = indicators.get('momentum', 0.0)
                inst_data['rsi'] = indicators.get('rsi', 50.0)
            
            # Get signal intensity
            signal = self.smart_bus.get(f'signal_{instrument}', 'PositionManager')
            if signal:
                inst_data['intensity'] = signal.get('intensity', 0.0)
                inst_data['confidence'] = signal.get('confidence', 0.5)
            else:
                inst_data['intensity'] = 0.0
                inst_data['confidence'] = 0.5
            
            market_data[instrument] = inst_data
        
        return market_data if market_data else None

    async def _update_smartbus_with_decisions(self, decisions: Dict[str, PositionDecisionResult]):
        """Update SmartInfoBus with position decisions"""
        for instrument, decision in decisions.items():
            self.smart_bus.set(
                f'position_decision_{instrument}',
                {
                    'decision': decision.decision.value,
                    'intensity': decision.intensity,
                    'size': decision.size,
                    'confidence': decision.confidence,
                    'risk_factors': decision.risk_factors
                },
                module='PositionManager',
                thesis=f"Position decision for {instrument}: {decision.decision.value} with {decision.confidence:.2f} confidence"
            )
        
        # Set portfolio state
        self.smart_bus.set(
            'portfolio_state',
            {
                'health_score': self._portfolio_health_score,
                'exposure_ratio': self._total_exposure_ratio,
                'open_positions': len(self.open_positions),
                'decision_quality': self._decision_quality_score
            },
            module='PositionManager',
            thesis="Current portfolio health and exposure metrics"
        )

    async def _generate_position_thesis(self, market_data: Dict[str, Any], decisions: Dict[str, PositionDecisionResult]) -> str:
        """Generate comprehensive thesis for position decisions"""
        return f"""Position Management Analysis:

Portfolio Health: {self._portfolio_health_score:.2f}
Total Exposure: {self._total_exposure_ratio:.1%}
Active Decisions: {len(decisions)}
Risk Management: {self._risk_management_score:.2f}

Key factors considered: market regime, risk score, portfolio balance, and signal quality."""

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error conditions"""
        return {
            'decisions': {},
            'portfolio_health': self._portfolio_health_score,
            'exposure_ratio': self._total_exposure_ratio,
            'error': reason,
            'processing_time_ms': 0,
            'thesis': f"Position manager fallback: {reason}"
        }

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        
        # Reset position manager state
        self.config.max_pct = self.default_max_pct
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
            'dynamic_max_pct': self.config.max_pct,
            'signal_sensitivity': 1.0,
            'risk_tolerance': 1.0,
            'confidence_threshold': 0.5
        }
        
        # Ensure minimum allocation capability
        if self.config.max_pct < 1e-5:
            self.logger.warning(
                format_operator_message(
                    "[WARN]", "MAX_PCT_TOO_LOW",
                    current=f"{self.config.max_pct:.6f}",
                    default=f"{self.default_max_pct:.4f}",
                    action="Restoring to default"
                )
            )
            self.config.max_pct = self.default_max_pct

    def set_env(self, env: Any):
        """Set environment reference"""
        self.env = env

    @create_error_handler("process_market_signals")
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
        
        self.logger.info(
            format_operator_message(
                "[STATS]", "PORTFOLIO_ASSESSMENT",
                health_score=f"{portfolio_health['overall_health']:.3f}",
                market_regime=market_regime,
                exposure_ratio=f"{portfolio_health['exposure_ratio']:.2%}",
                consecutive_losses=self.consecutive_losses
            )
        )
        
        # 2. Tactical Layer: Per-instrument decisions
        for instrument in self.instruments:
            # Extract signal context for this instrument
            signal_context = self._extract_signal_context(
                instrument, market_data, portfolio_health
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
                self.logger.info(
                    format_operator_message(
                        "[MONEY]", "POSITION_DECISION",
                        instrument=instrument,
                        decision=decision_result.decision.value,
                        intensity=f"{decision_result.intensity:.3f}",
                        size=f"€{decision_result.size:.0f}",
                        confidence=f"{decision_result.confidence:.3f}",
                        rationale=decision_result.rationale.get('stage', 'unknown')
                    )
                )
        
        return decisions

    def _assess_portfolio_health(self) -> Dict[str, float]:
        """Assess overall portfolio health metrics"""
        
        # Extract balance and drawdown
        balance = self.config.initial_balance
        drawdown = 0.0
        
        # Try to get from SmartInfoBus
        portfolio_metrics = self.smart_bus.get('portfolio_metrics', 'PositionManager')
        if portfolio_metrics:
            balance = portfolio_metrics.get('balance', self.config.initial_balance)
            drawdown = portfolio_metrics.get('drawdown', 0.0)
        elif self.env:
            balance = getattr(self.env, 'balance', self.config.initial_balance)
            drawdown = getattr(self.env, 'current_drawdown', 0.0)
        
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
        exposure_health = max(0.0, 1.0 - exposure_ratio / self.config.max_instrument_concentration)
        streak_health = max(0.1, 1.0 - self.consecutive_losses / self.config.max_consecutive_losses)
        
        # Risk management score from SmartInfoBus
        risk_score = self.smart_bus.get('risk_score', 'PositionManager')
        risk_health = 1.0 - (risk_score.get('risk_level', 0.0) if risk_score else 0.0)
        
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
        # Check if market regime already provided
        if 'market_regime' in market_data:
            return market_data['market_regime']
        
        # Extract volatility indicators
        avg_volatility = 0.0
        trend_strength = 0.0
        momentum_count = 0
        
        for instrument in self.instruments:
            inst_data = market_data.get(instrument, {})
            
            # Get volatility (from market data or default)
            vol = inst_data.get('volatility', self.config.min_volatility) 
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
                               portfolio_health: Dict[str, float]) -> SignalContext:
        """Extract and structure signal context for decision making"""
        
        inst_data = market_data.get(instrument, {})
        
        # Extract market signals
        market_intensity = float(inst_data.get('intensity', 0.0))
        market_direction = int(np.sign(market_intensity))
        volatility = max(float(inst_data.get('volatility', self.config.min_volatility)), self.config.min_volatility)
        trend_strength = float(inst_data.get('trend_strength', 0.0))
        momentum = float(inst_data.get('momentum', 0.0))
        volume_profile = float(inst_data.get('volume_profile', 1.0))
        
        # Market regime context
        regime = market_data.get('market_regime', 'normal')
        session = inst_data.get('session', 'unknown')
        
        # Calculate correlation penalty
        correlation_penalty = 0.0
        correlation_data = self.smart_bus.get('correlation_matrix', 'PositionManager')
        if correlation_data and instrument in correlation_data:
            correlation_penalty = min(abs(correlation_data[instrument].get('avg_correlation', 0.0)) * 0.5, 0.8)
        
        # Get liquidity score
        market_liquidity = self.smart_bus.get('market_liquidity', 'PositionManager')
        liquidity_score = market_liquidity.get(instrument, 1.0) if market_liquidity else 1.0
        
        # Create context with SmartInfoBus data
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
            step_idx=0,  # Will be updated from SmartInfoBus
            timestamp=datetime.datetime.now().isoformat()
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
                
                self.logger.warning(
                    format_operator_message(
                        "[ALERT]", "EMERGENCY_CLOSE",
                        instrument=instrument,
                        drawdown=f"{context.drawdown:.1%}",
                        consecutive_losses=self.consecutive_losses,
                        exposure=f"{context.current_exposure:.1%}"
                    )
                )
            return PositionDecisionResult(decision, intensity, size, confidence, rationale, risk_factors, context)
        
        # Stage 2: Signal strength filtering
        if signal_strength < self.config.min_signal_threshold:
            rationale["stage"] = "signal_filter"
            rationale["factors"].append(f"Signal strength {signal_strength:.3f} below threshold {self.config.min_signal_threshold}")
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
            if signal_strength >= self.config.min_signal_threshold:
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
            
            if signal_aligns and signal_strength > self.config.position_scale_threshold:
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
            
            elif position_pnl < -self.config.hard_loss_eur * 0.5:  # Approaching hard loss
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
            if size < context.balance * self.config.min_size_pct and final_intensity > 0.3:
                size = context.balance * self.config.min_size_pct
            elif size < context.balance * self.config.min_size_pct:
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
            self.consecutive_losses >= self.config.max_consecutive_losses,
            context.current_exposure > self.config.max_instrument_concentration * 1.5,
            context.liquidity_score < 0.3
        ]
        return any(emergency_conditions)

    def _calculate_portfolio_health_score(self, context: SignalContext) -> float:
        """Calculate overall portfolio health score"""
        drawdown_component = max(0.0, 1.0 - context.drawdown * 3.0)
        exposure_component = max(0.0, 1.0 - context.current_exposure / self.config.max_instrument_concentration)
        streak_component = max(0.1, 1.0 - self.consecutive_losses / self.config.max_consecutive_losses)
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
        risk_factors["volatility"] = min((context.volatility - self.config.min_volatility) / 0.05, 0.5)
        
        # Correlation risk
        risk_factors["correlation"] = context.correlation_penalty
        
        # Drawdown risk
        risk_factors["drawdown"] = min(context.drawdown * 2.0, 0.8)
        
        # Concentration risk
        risk_factors["concentration"] = min(context.current_exposure / self.config.max_instrument_concentration, 0.9)
        
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
        volatility = max(float(np.nan_to_num(volatility, nan=self.config.min_volatility)), self.config.min_volatility)
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
        
        # Risk tolerance adjustments
        risk_tolerance = self._adaptive_params.get('risk_tolerance', 1.0)
        adjusted_size *= risk_tolerance
        
        # Correlation penalty
        if correlation is not None:
            corr_penalty = 1.0 - min(abs(correlation) * 0.3, 0.5)  # Less aggressive penalty
            adjusted_size *= corr_penalty
        
        # Loss streak reduction
        if self.consecutive_losses >= self.config.max_consecutive_losses:
            streak_reduction = max(0.1, self.config.loss_reduction)  # Never reduce below 10%
            adjusted_size *= streak_reduction
            self.logger.info(
                format_operator_message(
                    "📉", "LOSS_STREAK_REDUCTION",
                    reduction_factor=f"{streak_reduction:.2f}",
                    consecutive_losses=self.consecutive_losses
                )
            )
        
        # Ensure minimum viable size or zero
        abs_size = abs(adjusted_size)
        min_viable_size = balance * self.config.min_size_pct
        
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

    def _apply_position_management(self) -> None:
        """Apply position management rules including live sync and exits"""
        
        # Ensure minimum allocation capability
        min_cap = 0.01  # 1% minimal allocation
        if self._adaptive_params['dynamic_max_pct'] < min_cap:
            self.logger.warning(
                format_operator_message(
                    "[WARN]", "DYNAMIC_MAX_PCT_LOW",
                    current=f"{self._adaptive_params['dynamic_max_pct']:.6f}",
                    minimum=f"{min_cap:.4f}",
                    action="Resetting to default"
                )
            )
            self._adaptive_params['dynamic_max_pct'] = self.default_max_pct

        # Live‐mode: sync & apply exit rules
        if self.env and getattr(self.env, "live_mode", False):
            self._sync_live_positions()
            self._apply_exit_rules()

        # Decay position confidence over time
        for inst in self.position_confidence:
            self.position_confidence[inst] *= self.config.confidence_decay

    def _update_position_health(self) -> None:
        """Update position health metrics"""
        try:
            # Defensive check for initialization race condition
            if not hasattr(self, 'open_positions') or not hasattr(self, '_exposure_history'):
                return
                
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
            risk_data = self.smart_bus.get('risk_score', 'PositionManager')
            if risk_data:
                risk_level = risk_data.get('risk_level', 0.5)
                self._risk_management_score = max(0.1, 1.0 - risk_level)
            
            # Update adaptive parameters based on performance
            self._adapt_parameters()
            
            # Update SmartInfoBus with health metrics
            self.smart_bus.set(
                'position_health',
                {
                    'portfolio_health': self._portfolio_health_score,
                    'exposure_ratio': current_exposure,
                    'risk_management_score': self._risk_management_score,
                    'consecutive_losses': self.consecutive_losses
                },
                module='PositionManager',
                thesis="Position manager health update"
            )
            
        except Exception as e:
            self.logger.warning(f"Position health update failed: {e}")

    def _adapt_parameters(self) -> None:
        """Adapt position management parameters based on performance"""
        
        try:
            # Adapt max_pct based on recent performance
            if len(self._decision_history) >= 10:
                recent_decisions = list(self._decision_history)[-10:]
                avg_portfolio_health = np.mean([d['portfolio_health'] for d in recent_decisions])
                
                if avg_portfolio_health > 0.8:
                    # Good performance, slightly increase risk tolerance
                    self._adaptive_params['dynamic_max_pct'] = min(
                        self.config.max_pct * 1.1, 
                        self.config.max_pct * 1.5
                    )
                elif avg_portfolio_health < 0.4:
                    # Poor performance, reduce risk
                    self._adaptive_params['dynamic_max_pct'] = max(
                        self.config.max_pct * 0.7,
                        self.config.max_pct * 0.3
                    )
                else:
                    # Neutral performance, gradual return to default
                    current = self._adaptive_params['dynamic_max_pct']
                    self._adaptive_params['dynamic_max_pct'] = current * 0.95 + self.config.max_pct * 0.05
            
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
            self.logger.warning(f"Parameter adaptation failed: {e}")

    def _sync_live_positions(self) -> None:
        """Sync positions from live broker"""
        broker_positions: List[Dict[str, Any]] = []

        # 1) env.broker
        if self.env and hasattr(self.env, "broker") and self.env.broker is not None:
            try:
                broker_positions = self.env.broker.get_positions()
            except Exception as exc:
                self.logger.warning(f"Broker position sync failed: {exc}")

        # 2) MT5 fallback
        elif mt5 is not None:
            raw = getattr(mt5, 'positions_get', lambda: [])() or []
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
        
        self.logger.info(
            format_operator_message(
                "[RELOAD]", "POSITIONS_SYNCED",
                position_count=len(self.open_positions),
                timestamp=self._last_sync_time.isoformat()
            )
        )

    def _apply_exit_rules(self) -> None:
        """Apply automated exit rules to open positions"""
        for inst, data in list(self.open_positions.items()):
            pnl_eur, _ = self._calc_unrealised_pnl(inst, data)
            # update peak
            if pnl_eur > data["peak_profit"]:
                data["peak_profit"] = pnl_eur

            # hard‐loss
            if pnl_eur <= -self.config.hard_loss_eur:
                self._close_position(inst, "hard_loss")
                continue

            # trailing‐profit
            if data["peak_profit"] > 0:
                drawdown_eur = data["peak_profit"] - pnl_eur
                trigger = max(
                    data["peak_profit"] * self.config.trail_pct,
                    self.config.trail_abs_eur
                )
                if drawdown_eur >= trigger:
                    self._close_position(inst, "trail_stop")

    def _close_position(self, inst: str, reason: str) -> None:
        """Close a position via broker or simulation"""
        # env.broker
        if self.env and getattr(self.env, "broker", None):
            ok = self.env.broker.close_position(inst, comment=reason)
            if ok:
                self.logger.info(
                    format_operator_message(
                        "[RED]", "POSITION_CLOSED",
                        instrument=inst,
                        reason=reason,
                        via="broker"
                    )
                )
                self.open_positions.pop(inst, None)
            else:
                self.logger.error(f"Broker close failed: {inst}, reason: {reason}")
            return

        # MT5 close
        if mt5 is not None:
            data = self.open_positions[inst]
            side = data["side"]
            lots = data["lots"]
            sym = inst.replace("/", "")
            tick = getattr(mt5, 'symbol_info_tick', lambda x: None)(sym)
            price = (tick.bid if side > 0 else tick.ask) if tick else 0.0

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       sym,
                "volume":       lots,
                "type":         (mt5.ORDER_TYPE_SELL if side > 0 else mt5.ORDER_TYPE_BUY),
                "price":        price,
                "deviation":    self.config.pips_tolerance,
                "position":     data["ticket"],
                "magic":        10001,
                "comment":      f"auto-exit:{reason}",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            order_send = getattr(mt5, 'order_send', None)
            res = order_send(request) if order_send else None
            if res and getattr(res, 'retcode', -1) == getattr(mt5, 'TRADE_RETCODE_DONE', 10009):
                self.logger.info(
                    format_operator_message(
                        "[RED]", "POSITION_CLOSED_MT5",
                        instrument=inst,
                        ticket=data["ticket"],
                        reason=reason
                    )
                )
                self.open_positions.pop(inst, None)
            else:
                self.logger.error(f"MT5 close failed: {inst}, error: {str(res)}")
            return

        # backtest fallback
        self.logger.info(
            format_operator_message(
                "[RED]", "POSITION_CLOSED_SIM",
                instrument=inst,
                reason=reason,
                mode="simulation"
            )
        )
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
            tick = getattr(mt5, 'symbol_info_tick', lambda x: None)(sym)
            if tick:
                price = tick.bid if data["side"] > 0 else tick.ask
        if price is None or not np.isfinite(price):
            return 0.0, 0.0

        # contract size
        contract_size = 100_000
        if mt5 is not None:
            info = getattr(mt5, 'symbol_info', lambda x: None)(sym)
            if info and getattr(info, 'trade_contract_size', None):
                contract_size = info.trade_contract_size

        points = (price - data["price_open"]) * data["side"]
        pnl_eur = points * contract_size * data["lots"]
        pnl_pct = pnl_eur / (abs(data["price_open"]) * contract_size * data["lots"])
        return float(pnl_eur), float(pnl_pct)

    def _calculate_current_exposure_ratio(self) -> float:
        """Calculate current exposure as ratio of balance"""
        if not self.open_positions:
            return 0.0
        
        balance = self.config.initial_balance
        
        # Try to get current balance from SmartInfoBus
        portfolio_metrics = self.smart_bus.get('portfolio_metrics', 'PositionManager')
        if portfolio_metrics:
            balance = portfolio_metrics.get('balance', self.config.initial_balance)
        elif self.env:
            balance = getattr(self.env, 'balance', self.config.initial_balance)
        
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
        
        return float(np.mean(quality_scores)) if quality_scores else 0.5

    # ════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get current state for persistence"""
        return {
            'config': self.config.__dict__,
            'genome': self.genome,
            'open_positions': self.open_positions,
            'portfolio_health_score': self._portfolio_health_score,
            'exposure_ratio': self._total_exposure_ratio,
            'consecutive_losses': self.consecutive_losses,
            'adaptive_params': self._adaptive_params,
            'position_confidence': self.position_confidence,
            'signal_history': {k: v[-20:] for k, v in self.signal_history.items()},  # Keep last 20
            'decision_history': list(self._decision_history)[-20:],  # Keep recent history
            'portfolio_health_history': list(self._portfolio_health_history)[-20:],
            'last_decisions': {
                k: {
                    'decision': v.decision.value,
                    'intensity': v.intensity,
                    'confidence': v.confidence,
                    'rationale': v.rationale
                } for k, v in self.last_decisions.items()
            },
            'success_count': getattr(self, 'success_count', 0),
            'failure_count': getattr(self, 'failure_count', 0)
        }

    def set_state(self, state: Dict[str, Any]):
        """Set state for hot-reload"""
        if 'config' in state:
            for key, value in state['config'].items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        if 'genome' in state:
            self.genome = state['genome']
        if 'open_positions' in state:
            self.open_positions = state['open_positions']
        if 'portfolio_health_score' in state:
            self._portfolio_health_score = state['portfolio_health_score']
        if 'exposure_ratio' in state:
            self._total_exposure_ratio = state['exposure_ratio']
        if 'consecutive_losses' in state:
            self.consecutive_losses = state['consecutive_losses']
        if 'adaptive_params' in state:
            self._adaptive_params = state['adaptive_params']
        if 'position_confidence' in state:
            self.position_confidence = state['position_confidence']
        if 'signal_history' in state:
            for inst, history in state['signal_history'].items():
                if inst in self.signal_history:
                    self.signal_history[inst] = history
        
        # Restore decision history
        if 'decision_history' in state:
            self._decision_history = deque(state['decision_history'], maxlen=100)
        if 'portfolio_health_history' in state:
            self._portfolio_health_history = deque(state['portfolio_health_history'], maxlen=50)
        
        # Restore last decisions
        if 'last_decisions' in state:
            self.last_decisions.clear()
            for inst, decision_data in state['last_decisions'].items():
                try:
                    decision = PositionDecision(decision_data['decision'])
                    context = SignalContext(instrument=inst)  # Minimal context
                    result = PositionDecisionResult(
                        decision=decision,
                        intensity=decision_data.get('intensity', 0.0),
                        size=0.0,
                        confidence=decision_data.get('confidence', 0.5),
                        rationale=decision_data.get('rationale', {}),
                        risk_factors={},
                        context=context
                    )
                    self.last_decisions[inst] = result
                except:
                    pass
        
        # Restore counts
        if 'success_count' in state:
            self.success_count = state['success_count']
        if 'failure_count' in state:
            self.failure_count = state['failure_count']

    # ════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome"""
        self.config.max_pct = float(np.clip(genome.get("max_pct", self.config.max_pct), 0.01, 0.25))
        self.config.max_consecutive_losses = int(np.clip(genome.get("max_consecutive_losses", self.config.max_consecutive_losses), 1, 20))
        self.config.loss_reduction = float(np.clip(genome.get("loss_reduction", self.config.loss_reduction), 0.05, 1.0))
        self.config.max_instrument_concentration = float(np.clip(genome.get("max_instrument_concentration", self.config.max_instrument_concentration), 0.05, 0.5))
        self.config.min_volatility = float(np.clip(genome.get("min_volatility", self.config.min_volatility), 0.001, 0.10))
        self.config.hard_loss_eur = float(np.clip(genome.get("hard_loss_eur", self.config.hard_loss_eur), 10.0, 100.0))
        self.config.trail_pct = float(np.clip(genome.get("trail_pct", self.config.trail_pct), 0.05, 0.3))
        self.config.trail_abs_eur = float(np.clip(genome.get("trail_abs_eur", self.config.trail_abs_eur), 5.0, 50.0))
        self.config.min_signal_threshold = float(np.clip(genome.get("min_signal_threshold", self.config.min_signal_threshold), 0.05, 0.5))
        self.config.position_scale_threshold = float(np.clip(genome.get("position_scale_threshold", self.config.position_scale_threshold), 0.2, 0.8))
        self.config.emergency_close_threshold = float(np.clip(genome.get("emergency_close_threshold", self.config.emergency_close_threshold), 0.7, 0.95))
        self.config.confidence_decay = float(np.clip(genome.get("confidence_decay", self.config.confidence_decay), 0.90, 0.99))
        self.risk_multiplier = float(np.clip(genome.get("risk_multiplier", self.risk_multiplier), 0.5, 2.0))
        self.correlation_threshold = float(np.clip(genome.get("correlation_threshold", self.correlation_threshold), 0.3, 0.9))
        
        self.genome = {
            "max_pct": self.config.max_pct,
            "max_consecutive_losses": self.config.max_consecutive_losses,
            "loss_reduction": self.config.loss_reduction,
            "max_instrument_concentration": self.config.max_instrument_concentration,
            "min_volatility": self.config.min_volatility,
            "hard_loss_eur": self.config.hard_loss_eur,
            "trail_pct": self.config.trail_pct,
            "trail_abs_eur": self.config.trail_abs_eur,
            "min_signal_threshold": self.config.min_signal_threshold,
            "position_scale_threshold": self.config.position_scale_threshold,
            "emergency_close_threshold": self.config.emergency_close_threshold,
            "confidence_decay": self.config.confidence_decay,
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
            self.logger.info(
                format_operator_message(
                    "🧬", "MUTATION_APPLIED",
                    changes=", ".join(mutations)
                )
            )
            
        self.set_genome(g)
        
    def crossover(self, other: "PositionManager") -> "PositionManager":
        """Enhanced crossover with performance-based selection"""
        if not isinstance(other, PositionManager):
            self.logger.warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_performance = getattr(self, '_portfolio_health_score', 0.5)
        other_performance = getattr(other, '_portfolio_health_score', 0.5)
        
        # Favor higher performance parent
        if self_performance > other_performance:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {k: (self.genome[k] if np.random.rand() < bias else getattr(other, 'genome', {}).get(k, v)) for k, v in self.genome.items()}
        
        child = PositionManager(
            **{
                'config': self.config,
                'instruments': getattr(self, 'instruments', []),
                'genome': new_g
            }
        )
        
        # Inherit beneficial state from better parent
        if self_performance > other_performance:
            self_signal_history = getattr(self, 'signal_history', {})
            if self_signal_history:
                setattr(child, 'signal_history', copy.deepcopy(self_signal_history))
        else:
            other_signal_history = getattr(other, 'signal_history', {})
            if other_signal_history:
                setattr(child, 'signal_history', copy.deepcopy(other_signal_history))
        
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
            "genome": self.genome.copy(),
            "circuit_breaker": self.circuit_breaker.copy()
        }

    def get_position_manager_report(self) -> str:
        """Generate operator-friendly position manager report"""
        
        # Portfolio status
        if self._portfolio_health_score > 0.8:
            portfolio_status = "[ROCKET] Excellent"
        elif self._portfolio_health_score > 0.6:
            portfolio_status = "[OK] Good"
        elif self._portfolio_health_score > 0.4:
            portfolio_status = "[FAST] Fair"
        else:
            portfolio_status = "[WARN] Poor"
        
        # Risk status
        if self.consecutive_losses == 0:
            risk_status = "[GREEN] Safe"
        elif self.consecutive_losses < self.config.max_consecutive_losses // 2:
            risk_status = "[YELLOW] Caution"
        else:
            risk_status = "[RED] High Risk"
        
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
[STATS] ENHANCED POSITION MANAGER
═══════════════════════════════════════
💼 Portfolio: {portfolio_status} ({self._portfolio_health_score:.3f})
[WARN] Risk Status: {risk_status}
[CHART] Exposure: {self._total_exposure_ratio:.1%}
📍 Open Positions: {len(self.open_positions)}

[TOOL] RISK PARAMETERS
• Max Position %: {self.config.max_pct:.1%} (dynamic: {self._adaptive_params['dynamic_max_pct']:.1%})
• Hard Loss Limit: €{self.config.hard_loss_eur:.0f}
• Trail Stop: {self.config.trail_pct:.1%} / €{self.config.trail_abs_eur:.0f}
• Signal Threshold: {self.config.min_signal_threshold:.2f}
• Consecutive Losses: {self.consecutive_losses}/{self.config.max_consecutive_losses}

[STATS] PERFORMANCE METRICS
• Decision Quality: {self._decision_quality_score:.3f}
• Risk Management: {self._risk_management_score:.3f}
• Signal Quality: {self._assess_signal_quality():.3f}
• Recent Confidence: {recent_avg_confidence:.3f}

[TARGET] ADAPTIVE PARAMETERS
• Signal Sensitivity: {self._adaptive_params['signal_sensitivity']:.2f}
• Risk Tolerance: {self._adaptive_params['risk_tolerance']:.2f}
• Confidence Threshold: {self._adaptive_params['confidence_threshold']:.2f}

💡 RECENT ACTIVITY
• Active Decisions: {active_decisions}
• Decision History: {len(self._decision_history)} records
• Portfolio Health Trend: {len([h for h in self._portfolio_health_history if h['health_score'] > 0.7])} good periods
• Circuit Breaker: {self.circuit_breaker['state']}

[RELOAD] INSTRUMENTS ({len(self.instruments)})
{chr(10).join([f"• {inst}: {len(self.signal_history.get(inst, []))} signals, confidence: {self.position_confidence.get(inst, 0.5):.2f}" for inst in self.instruments[:5]])}
        """

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
            consecutive_losses_ratio = self.consecutive_losses / max(self.config.max_consecutive_losses, 1)
            risk_management_score = self._risk_management_score
            
            # Adaptive parameters
            dynamic_risk_ratio = self._adaptive_params['dynamic_max_pct'] / self.config.max_pct
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
            balance = self.config.initial_balance
            drawdown = 0.0
            
            portfolio_metrics = self.smart_bus.get('portfolio_metrics', 'PositionManager')
            if portfolio_metrics:
                balance = portfolio_metrics.get('balance', self.config.initial_balance)
                drawdown = portfolio_metrics.get('drawdown', 0.0)
            elif self.env:
                balance = getattr(self.env, 'balance', self.config.initial_balance)
                drawdown = getattr(self.env, 'current_drawdown', 0.0)
            
            balance_ratio = balance / self.config.initial_balance
            
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
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """
        Propose trading actions based on independent signal interpretation.
        
        This method uses the hierarchical decision making process with SmartInfoBus.
        """
        try:
            obs = inputs.get('obs', None)
            
            if self._forced_action is not None:
                action_array = np.array(
                    [self._forced_action] * len(self.instruments) * 2, dtype=np.float32
                )
                return {
                    'action_type': 'position_management',
                    'action_array': action_array.tolist(),
                    'forced': True,
                    'confidence': self._forced_conf or 0.8
                }
            
            signals: List[float] = []
            
            # Extract market data from SmartInfoBus
            market_data = self._extract_market_data_from_smartbus()
            
            if not market_data:
                # Fallback: generate neutral signals
                action_array = np.zeros(len(self.instruments) * 2, dtype=np.float32)
                return {
                    'action_type': 'position_management',
                    'action_array': action_array.tolist(),
                    'fallback_reason': 'no_market_data',
                    'confidence': 0.1
                }
            
            # Process signals through hierarchical decision making
            decisions = self.process_market_signals(market_data)
            
            # Convert decisions to action signals
            action_details = []
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
                
                action_details.append({
                    'instrument': inst,
                    'intensity': intensity,
                    'duration': duration,
                    'decision': decision_result.decision.value if decision_result else 'hold',
                    'confidence': decision_result.confidence if decision_result else 0.5
                })
                
                self.logger.info(
                    format_operator_message(
                        "📡", "ACTION_PROPOSAL",
                        instrument=inst,
                        intensity=f"{intensity:.3f}",
                        duration=f"{duration:.1f}",
                        decision=decision_result.decision.value if decision_result else "hold"
                    )
                )
                
                signals.extend([intensity, duration])
            
            action_array = np.array(signals, dtype=np.float32)
            overall_confidence = self.confidence() if hasattr(self, 'confidence') else 0.5
            
            return {
                'action_type': 'position_management',
                'action_array': action_array.tolist(),
                'action_details': action_details,
                'confidence': overall_confidence,
                'market_data_available': True,
                'decisions_count': len(decisions),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            action_array = np.zeros(len(self.instruments) * 2, dtype=np.float32)
            return {
                'action_type': 'position_management',
                'action_array': action_array.tolist(),
                'error': str(e),
                'confidence': 0.1
            }

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in position management decisions"""
        try:
            # Use the existing confidence calculation logic
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
            
            # Circuit breaker confidence
            circuit_confidence = 1.0 if self.circuit_breaker['state'] == 'CLOSED' else 0.2
            confidence_components.append(circuit_confidence)
            
            # Decision quality confidence
            confidence_components.append(self._decision_quality_score)
            
            # Calculate weighted average
            weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]  # Portfolio, positions, signals, risk, circuit, decisions
            final_confidence = np.average(confidence_components, weights=weights)
            
            return float(np.clip(final_confidence, 0.1, 1.0))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5  # Default moderate confidence

    # Legacy method for backward compatibility
    def propose_action_legacy(self, obs: Any = None) -> np.ndarray:
        """Legacy propose_action method that returns numpy array"""
        try:
            # Try to run async method
            result = asyncio.run(self.propose_action(obs=obs))
            return np.array(result.get('action_array', []), dtype=np.float32)
        except RuntimeError:
            # If we're already in an event loop, create a task instead
            try:
                loop = asyncio.get_running_loop()
                task = loop.create_task(self.propose_action(obs=obs))
                # For this legacy method, we'll provide a simple fallback
                return np.zeros(len(self.instruments) * 2, dtype=np.float32)
            except Exception:
                # Final fallback
                return np.zeros(len(self.instruments) * 2, dtype=np.float32)

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

    def confidence(self, obs: Any = None) -> float:
        """
        Calculate overall confidence based on portfolio and position health.
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
        
        # Circuit breaker confidence
        circuit_confidence = 1.0 if self.circuit_breaker['state'] == 'CLOSED' else 0.2
        confidence_components.append(circuit_confidence)
        
        # Decision quality confidence
        confidence_components.append(self._decision_quality_score)
        
        # Calculate weighted average
        weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]  # Portfolio, positions, signals, risk, circuit, decisions
        final_confidence = np.average(confidence_components, weights=weights)
        
        return float(np.clip(final_confidence, 0.1, 1.0))

    # Backward compatibility
    def step(self, **kwargs):
        """Backward compatibility step method"""
        # Run synchronous version of process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(self.process(**kwargs))
        loop.close()
        return result