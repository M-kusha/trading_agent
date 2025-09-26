"""
⚙️ Enhanced Trading Mode Manager with SmartInfoBus Integration v3.1
Intelligent trading mode switching based on comprehensive market analysis and performance tracking
"""

from __future__ import annotations

import asyncio
import time
from modules.contracts import module_args
import numpy as np
import datetime
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

# ═══════════════════════════════════════════════════════════════════
# MODERN SMARTINFOBUS IMPORTS
# ═══════════════════════════════════════════════════════════════════
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


# ─────────────────────────────────────────────────────────────
# Typed configuration (lint-safe) + dict bridge for BaseModule
# ─────────────────────────────────────────────────────────────

@dataclass
class TradingModeManagerConfig:
    initial_mode: str = "normal"
    window: int = 20
    auto_mode: bool = True
    min_persistence: int = 5
    context_sensitivity: float = 0.8
    performance_weight: float = 0.4
    risk_weight: float = 0.3
    consensus_weight: float = 0.2
    market_context_weight: float = 0.1
    regime_awareness: bool = True
    session_awareness: bool = True
    volatility_scaling: bool = True
    debug: bool = True
    # circuit breaker
    breaker_error_window: int = 10
    breaker_open_threshold: int = 5
    breaker_cooldown_sec: float = 20.0
    # namespaced health keys (single-writer style)
    status_key: str = "trading_mode_manager_status"
    health_key: str = "trading_mode_manager_health"


@module(**module_args(
    "TradingModeManager",
    description="Intelligent trading mode switching based on comprehensive market analysis and performance tracking",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class TradingModeManager(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    ⚙️ PRODUCTION-GRADE Trading Mode Manager v3.1

    Intelligent trading mode management system with:
    - Adaptive mode switching based on market conditions and performance
    - Comprehensive risk assessment and performance tracking
    - Market regime and session awareness for contextual decisions
    - SmartInfoBus zero-wiring architecture (single-writer discipline)
    - Real-time effectiveness monitoring and optimization
    """

    # Enhanced trading modes with comprehensive definitions
    TRADING_MODES = {
        "safe": {
            "description": "Conservative risk management with capital preservation focus",
            "risk_multiplier": 0.5,
            "max_exposure": 0.3,
            "win_rate_threshold": 0.0,
            "drawdown_limit": 0.15,
            "consensus_requirement": 0.0,
            "volatility_tolerance": "low"
        },
        "normal": {
            "description": "Balanced trading approach with moderate risk-reward",
            "risk_multiplier": 1.0,
            "max_exposure": 0.6,
            "win_rate_threshold": 0.40,
            "drawdown_limit": 0.10,
            "consensus_requirement": 0.30,
            "volatility_tolerance": "medium"
        },
        "aggressive": {
            "description": "Increased risk for higher returns with active position management",
            "risk_multiplier": 1.5,
            "max_exposure": 0.8,
            "win_rate_threshold": 0.55,
            "drawdown_limit": 0.08,
            "consensus_requirement": 0.50,
            "volatility_tolerance": "medium-high"
        },
        "extreme": {
            "description": "Maximum risk for maximum returns - requires exceptional conditions",
            "risk_multiplier": 2.0,
            "max_exposure": 1.0,
            "win_rate_threshold": 0.65,
            "drawdown_limit": 0.05,
            "consensus_requirement": 0.65,
            "volatility_tolerance": "high"
        }
    }

    # ── lifecycle ────────────────────────────────────────────

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        # Typed config; keep dict copy for BaseModule internals
        cfg_dict = (config or {}).copy()
        # Merge user config into defaults safely
        merged = {**asdict(TradingModeManagerConfig()), **cfg_dict}
        self._cfg = TradingModeManagerConfig(**merged)
        self.config = cfg_dict

        # breaker state
        self._breaker_state = "CLOSED"   # CLOSED | OPEN | HALF_OPEN
        self._last_failure_ts: float = 0.0
        self._recent_failures: deque[bool] = deque(maxlen=self._cfg.breaker_error_window)

        # Initialize systems
        self._initialize_advanced_systems()

        # Parent init (calls _initialize)
        super().__init__()

    def _initialize(self):
        """Initialize advanced trading mode management systems"""
        # Initialize base mixins / state helpers
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()

        # Enhanced mode configuration
        self.initial_mode = self._cfg.initial_mode if self._cfg.initial_mode in self.TRADING_MODES else 'normal'
        self.window = int(self._cfg.window)
        self.auto_mode = bool(self._cfg.auto_mode)
        self.min_persistence = int(self._cfg.min_persistence)
        self.context_sensitivity = float(self._cfg.context_sensitivity)
        self.performance_weight = float(self._cfg.performance_weight)
        self.risk_weight = float(self._cfg.risk_weight)
        self.consensus_weight = float(self._cfg.consensus_weight)
        self.market_context_weight = float(self._cfg.market_context_weight)
        self.regime_awareness = bool(self._cfg.regime_awareness)
        self.session_awareness = bool(self._cfg.session_awareness)
        self.volatility_scaling = bool(self._cfg.volatility_scaling)
        self.debug = bool(self._cfg.debug)

        # Core state management
        self.current_mode = self.initial_mode
        self.mode_persistence = 0
        self.last_mode_change = None
        self.last_change_reason = ""

        # Enhanced state tracking
        self.stats_history = deque(maxlen=self.window * 2)
        self.mode_history = deque(maxlen=100)
        self.decision_trace = deque(maxlen=200)
        self.performance_history = deque(maxlen=200)

        # Market context awareness
        self.market_regime = "unknown"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        self.market_schedule = self.config.get('market_schedule')

        # Performance analytics with comprehensive tracking
        self.mode_analytics = defaultdict(lambda: defaultdict(list))
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.session_performance = defaultdict(lambda: defaultdict(list))

        # Enhanced mode switching statistics
        self.mode_stats = {
            "total_switches": 0,
            "auto_switches": 0,
            "manual_switches": 0,
            "current_mode_duration": 0,
            "mode_effectiveness": 0.5,
            "switching_accuracy": 0.0,
            "average_mode_duration": 0.0,
            "best_performing_mode": "normal",
            "total_uptime": 0,
            "session_start": datetime.datetime.now().isoformat()
        }

        # Enhanced decision factors with intelligence
        self.decision_factors = {
            "performance_score": 0.5,
            "risk_score": 0.5,
            "consensus_score": 0.5,
            "market_context_score": 0.5,
            "volatility_score": 0.5,
            "regime_score": 0.5,
            "session_score": 0.5,
            "trend_score": 0.5,
            "stability_score": 0.5
        }

        # Adaptive mode thresholds
        self.mode_thresholds = self._initialize_adaptive_thresholds()

        # Learning and adaptation systems
        self.learning_history = deque(maxlen=50)
        self.threshold_adaptations = deque(maxlen=50)
        self.effectiveness_tracking = defaultdict(list)

        # Circuit breaker
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False

        # Mode intelligence parameters
        self.mode_intelligence = {
            'adaptation_speed': 0.1,
            'confidence_threshold': 0.7,
            'stability_requirement': 0.8,
            'performance_memory': 0.9,
            'risk_sensitivity': 0.8,
            'consensus_importance': 0.6
        }

        # Generate initialization thesis
        self._generate_initialization_thesis()

        version = getattr(self.metadata, 'version', '3.1.0') if self.metadata else '3.1.0'
        self.logger.info(format_operator_message(
            icon="⚙️",
            message=f"Trading Mode Manager v{version} initialized",
            initial_mode=self.current_mode,
            auto_mode=self.auto_mode,
            window=self.window,
            regime_awareness=self.regime_awareness
        ))

        # Post initial namespaced health
        self._post_health_status()

    # ── mixin-required overrides ───────────────────────────

    async def propose_action(self, **inputs: Any) -> Dict[str, Any]:
        """Provide a compact trading-mode action proposal.

        Contract:
        - Never returns None (satisfies SmartInfoBusTradingMixin typing)
        - Minimal, side-effect-free; uses current state and lightweight scoring
        """
        try:
            # Lightweight confidence using existing factors; fall back safely
            consensus = float(self.decision_factors.get('consensus_score', 0.5))
            context = float(self.decision_factors.get('market_context_score', 0.5))
            stability = float(self.decision_factors.get('stability_score', 0.5))
            performance = float(self.decision_factors.get('performance_score', 0.5))
            base_conf = float(np.clip(0.35 * consensus + 0.35 * context + 0.15 * stability + 0.15 * performance, 0.0, 1.0))

            action: Dict[str, Any] = {
                'module': 'TradingModeManager',
                'mode': self.current_mode,
                'confidence': base_conf,
                'timestamp': datetime.datetime.now().isoformat(),
                'reason': 'lightweight_proposal',
            }
            return action
        except Exception as e:
            # Never return None; degrade gracefully
            if hasattr(self, 'logger'):
                self.logger.warning(f"propose_action degraded: {e}")
            return {
                'module': 'TradingModeManager',
                'mode': getattr(self, 'current_mode', 'normal'),
                'confidence': 0.5,
                'timestamp': datetime.datetime.now().isoformat(),
                'reason': 'fallback',
            }

    async def calculate_confidence(self, action: Dict[str, Any], **inputs: Any) -> float:
        """Compute confidence for a given action; must return float (non-optional)."""
        try:
            if isinstance(action, dict) and 'confidence' in action:
                v = action.get('confidence')
                if isinstance(v, (int, float, np.generic)):
                    return float(np.clip(v, 0.0, 1.0))
            # Derive from current decision factors
            consensus = float(self.decision_factors.get('consensus_score', 0.5))
            context = float(self.decision_factors.get('market_context_score', 0.5))
            stability = float(self.decision_factors.get('stability_score', 0.5))
            return float(np.clip(0.5 * consensus + 0.5 * context * 0.9 + 0.1 * stability, 0.0, 1.0))
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"calculate_confidence degraded: {e}")
            return 0.5

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="TradingModeManager",
            log_path="logs/trading/trading_mode_manager.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("TradingModeManager", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    def _post_health_status(self):
        """Write compact health and status snapshots (namespaced keys)."""
        try:
            status = {
                'initialized': True,
                'auto_mode': self.auto_mode,
                'current_mode': self.current_mode,
                'breaker_state': self._breaker_state,
                'ts': datetime.datetime.now().isoformat()
            }
            self.smart_bus.set(self._cfg.status_key, status, module='TradingModeManager',
                               thesis="TradingModeManager status heartbeat")
            self.smart_bus.set(self._cfg.health_key, self._get_health_metrics(), module='TradingModeManager',
                               thesis="TradingModeManager health metrics")
            # Also maintain explicit initialization heartbeat key for contract consumers
            init_view = {
                'status': 'initialized' if not self.is_disabled else 'disabled',
                'timestamp': datetime.datetime.now().isoformat(),
                'current_mode': self.current_mode,
                'auto_mode': self.auto_mode
            }
            try:
                self.smart_bus.set('trading_mode_manager_initialization', init_view, module='TradingModeManager', thesis='Initialization heartbeat')
            except Exception:
                pass
        except Exception as e:
            self.logger.warning(f"[MONITOR] health update failed: {e}")

    # ── initialization helpers (referenced above) ──────────

    def _initialize_adaptive_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Create per-mode adaptive thresholds with sensible defaults.

        This is a lightweight, deterministic initializer to satisfy static typing
        and provide reasonable starting thresholds.
        """
        try:
            thresholds: Dict[str, Dict[str, float]] = {}
            for mode, cfg in self.TRADING_MODES.items():
                # Seed thresholds using TRADING_MODES guidance
                thresholds[mode] = {
                    'min_win_rate': float(cfg.get('win_rate_threshold', 0.4)),
                    'max_drawdown': float(cfg.get('drawdown_limit', 0.1)),
                    'max_exposure': float(cfg.get('max_exposure', 0.6)),
                    'min_consensus': float(cfg.get('consensus_requirement', 0.3)),
                    'performance_threshold': 0.5,
                    'stability_requirement': 0.5,
                }
            return thresholds
        except Exception:
            # Fallback safe defaults
            return {
                'safe': {'min_win_rate': 0.0, 'max_drawdown': 0.15, 'max_exposure': 0.3, 'min_consensus': 0.0, 'performance_threshold': 0.4, 'stability_requirement': 0.6},
                'normal': {'min_win_rate': 0.4, 'max_drawdown': 0.10, 'max_exposure': 0.6, 'min_consensus': 0.30, 'performance_threshold': 0.5, 'stability_requirement': 0.5},
                'aggressive': {'min_win_rate': 0.55, 'max_drawdown': 0.08, 'max_exposure': 0.8, 'min_consensus': 0.50, 'performance_threshold': 0.6, 'stability_requirement': 0.4},
                'extreme': {'min_win_rate': 0.65, 'max_drawdown': 0.05, 'max_exposure': 1.0, 'min_consensus': 0.65, 'performance_threshold': 0.7, 'stability_requirement': 0.3},
            }

    def _generate_initialization_thesis(self) -> None:
        """Publish a concise initialization thesis to the logger and bus."""
        try:
            thesis = format_operator_message(
                icon="⚙️",
                message="TradingModeManager initialization",
                initial_mode=self.initial_mode,
                window=self.window,
                auto_mode=self.auto_mode,
                regime_awareness=self.regime_awareness,
                session_awareness=self.session_awareness,
            )
            # Log it
            self.logger.info(thesis)
            # Optionally publish to InfoBus as part of mode status
            status = {
                'initial_mode': self.initial_mode,
                'window': self.window,
                'auto_mode': self.auto_mode,
                'ts': datetime.datetime.now().isoformat(),
            }
            try:
                self.smart_bus.set(self._cfg.status_key, {**status, '_thesis': thesis}, module='TradingModeManager', thesis='Initialization thesis')
            except Exception:
                # Non-fatal if bus is unavailable during init
                pass
        except Exception as e:
            self.logger.warning(f"Initialization thesis generation failed: {e}")

    # ── core process ─────────────────────────────────────────

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive mode management

        Returns:
            Dict containing mode status, analytics, recommendations, and _thesis
        """
        start_time = time.time()
        try:
            # Circuit breaker check
            if self.is_disabled or self._breaker_state == "OPEN":
                if (time.time() - self._last_failure_ts) >= self._cfg.breaker_cooldown_sec:
                    self._breaker_state = "HALF_OPEN"
                return self._generate_disabled_response() if self.is_disabled else self._generate_breaker_response()

            # Get comprehensive market data from SmartInfoBus
            market_data = await self._get_comprehensive_market_data()

            # Update market context awareness
            await self._update_market_context_comprehensive(market_data)

            # Extract and analyze performance data
            performance_data = await self._extract_performance_data_comprehensive(market_data)

            # Update performance statistics with new data
            await self._update_performance_statistics_comprehensive(performance_data, market_data)

            # Perform intelligent mode decision analysis
            mode_decision = await self._make_intelligent_mode_decision_comprehensive(performance_data, market_data)

            # Apply mode decision with comprehensive tracking
            mode_change_result = await self._apply_mode_decision_comprehensive(mode_decision, market_data)

            # Analyze current mode effectiveness
            effectiveness_analysis = await self._analyze_mode_effectiveness_comprehensive(performance_data, market_data)

            # Update adaptive thresholds based on market conditions
            threshold_updates = await self._update_adaptive_thresholds_comprehensive(performance_data, market_data)

            # Generate mode recommendations
            recommendations = await self._generate_intelligent_mode_recommendations(mode_decision, effectiveness_analysis)

            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_mode_thesis(mode_decision, effectiveness_analysis)

            # Create comprehensive results (include initialization view for contract compliance)
            results = {
                'trading_mode': self.current_mode,
                'mode_config': self._get_mode_configuration(),
                'mode_stats': self._get_comprehensive_mode_stats(),
                'mode_effectiveness': float(effectiveness_analysis.get('current_effectiveness', 0.5)),
                'decision_factors': {k: float(v) if isinstance(v, (int, float, np.generic)) else v
                                     for k, v in self.decision_factors.items()},
                'mode_thresholds': self._safe_thresholds_copy(),
                'market_context': self._get_market_context_summary(),
                'mode_recommendations': list(recommendations),
                'mode_decision_analysis': mode_decision,
                'health_metrics': self._get_health_metrics(),
                '_thesis': thesis,
                'trading_mode_manager_initialization': self._get_tmm_init_view()
            }

            # Update SmartInfoBus with comprehensive thesis (single-writer keys)
            await self._update_smartinfobus_comprehensive(results, thesis)

            # NOTE: performance_data is provided by SessionManager, NOT TradingModeManager
            # Removed illegal publication to stop provider ownership conflicts

            # Record performance metrics
            processing_time = int((time.time() - start_time) * 1000)
            self.performance_tracker.record_metric('TradingModeManager', 'process_time', processing_time, True)

            # Update mode statistics book-keeping
            self._update_mode_performance_metrics()

            # breaker: success path
            self._recent_failures.append(False)
            if self._breaker_state == "HALF_OPEN":
                # close after a good cycle
                self._breaker_state = "CLOSED"

            # Reset error count on successful processing
            self.error_count = 0
            self._post_health_status()

            # NOTE: performance_data is provided by SessionManager, NOT TradingModeManager
            # Removed from return to stop provider ownership conflicts

            return results

        except Exception as e:
            # error path
            self._recent_failures.append(True)
            self._last_failure_ts = time.time()
            if list(self._recent_failures).count(True) >= self._cfg.breaker_open_threshold:
                self._breaker_state = "OPEN"

            return await self._handle_processing_error(e, start_time)

    # ── data access ──────────────────────────────────────────

    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data using modern SmartInfoBus patterns"""
        try:
            return {
                'recent_trades': self.smart_bus.get('recent_trades', 'TradingModeManager') or [],
                'risk_metrics': self.smart_bus.get('risk_metrics', 'TradingModeManager') or {},
                'votes': self.smart_bus.get('votes', 'TradingModeManager') or [],
                'positions': self.smart_bus.get('positions', 'TradingModeManager') or [],
                'market_context': self.smart_bus.get('market_context', 'TradingModeManager') or {},
                'session_metrics': self.smart_bus.get('session_metrics', 'TradingModeManager') or {},
                'strategy_performance': self.smart_bus.get('strategy_performance', 'TradingModeManager') or {},
                'trading_performance': self.smart_bus.get('trading_performance', 'TradingModeManager') or {},
                'market_regime': self.smart_bus.get('market_regime', 'TradingModeManager') or 'unknown',
                'volatility_data': self.smart_bus.get('volatility_data', 'TradingModeManager') or {},
                'economic_calendar': self.smart_bus.get('economic_calendar', 'TradingModeManager') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "TradingModeManager")
            self.logger.warning(f"Market data retrieval incomplete: {error_context}")
            return self._get_safe_market_defaults()

    async def _update_market_context_comprehensive(self, market_data: Dict[str, Any]):
        """Update comprehensive market context awareness"""
        try:
            old_regime = self.market_regime
            old_volatility = self.volatility_regime
            old_session = self.market_session

            # Update regime tracking
            market_context = market_data.get('market_context', {}) or {}
            self.market_regime = market_data.get('market_regime', 'unknown') or market_context.get('regime', 'unknown')
            self.volatility_regime = market_context.get('volatility_level', 'medium')
            self.market_session = market_context.get('session', market_context.get('trading_session', 'unknown'))

            # Detect significant changes
            regime_changed = self.market_regime != old_regime and old_regime != 'unknown'
            volatility_changed = self.volatility_regime != old_volatility and old_volatility != 'unknown'
            session_changed = self.market_session != old_session and old_session != 'unknown'

            # Log and adapt if needed
            if regime_changed or volatility_changed:
                impact_assessment = self._assess_market_change_impact(regime_changed, volatility_changed, session_changed)
                self.logger.info(format_operator_message(
                    icon="🌊",
                    message="Market context change detected",
                    regime_change=f"{old_regime} → {self.market_regime}" if regime_changed else "unchanged",
                    volatility_change=f"{old_volatility} → {self.volatility_regime}" if volatility_changed else "unchanged",
                    impact=impact_assessment,
                    current_mode=self.current_mode
                ))
                if impact_assessment in ['high', 'extreme']:
                    await self._trigger_emergency_threshold_adaptation(market_data)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "market_context_update")
            self.logger.warning(f"Market context update failed: {error_context}")

    def _assess_market_change_impact(self, regime_changed: bool, volatility_changed: bool, session_changed: bool) -> str:
        """Assess the impact level of market changes"""
        try:
            impact_score = 0
            if regime_changed:
                impact_score += 3
            if volatility_changed:
                impact_score += 2 if self.volatility_regime in ['high', 'extreme'] else 1
            if session_changed:
                impact_score += 1

            if impact_score >= 4:
                return 'extreme'
            elif impact_score >= 3:
                return 'high'
            elif impact_score >= 2:
                return 'medium'
            else:
                return 'low'
        except Exception:
            return 'medium'

    async def _trigger_emergency_threshold_adaptation(self, market_data: Dict[str, Any]):
        """Trigger emergency adaptation of thresholds due to significant market changes"""
        try:
            _ = self._calculate_emergency_adaptation_factor(market_data)
            for mode in self.mode_thresholds:
                thresholds = self.mode_thresholds[mode]
                # Make thresholds more conservative during high volatility/uncertainty
                if self.volatility_regime in ['high', 'extreme'] or self.market_regime == 'unknown':
                    thresholds['max_drawdown'] = float(max(0.01, thresholds['max_drawdown'] * 0.8))
                    thresholds['min_win_rate'] = float(min(1.0, thresholds['min_win_rate'] * 1.1))
                    thresholds['min_consensus'] = float(min(1.0, thresholds['min_consensus'] * 1.2))
                # Regime-specific
                if self.market_regime == 'volatile':
                    thresholds['stability_requirement'] = float(min(1.0, thresholds['stability_requirement'] * 1.3))
                elif self.market_regime in ['trending', 'momentum']:
                    thresholds['performance_threshold'] = float(max(0.0, thresholds['performance_threshold'] * 0.9))

            self.threshold_adaptations.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'type': 'emergency',
                'regime': self.market_regime,
                'volatility': self.volatility_regime
            })
            self.logger.info(format_operator_message(
                icon="[FAST]",
                message="Emergency threshold adaptation triggered",
                volatility_regime=self.volatility_regime,
                market_regime=self.market_regime
            ))
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "emergency_adaptation")
            self.logger.warning(f"Emergency threshold adaptation failed: {error_context}")

    def _calculate_emergency_adaptation_factor(self, market_data: Dict[str, Any]) -> float:
        """Calculate emergency adaptation factor based on market stress"""
        try:
            stress_factors = 0
            if self.volatility_regime == 'extreme':
                stress_factors += 3
            elif self.volatility_regime == 'high':
                stress_factors += 2
            if self.market_regime == 'unknown':
                stress_factors += 2
            elif self.market_regime == 'volatile':
                stress_factors += 1
            recent_trades = market_data.get('recent_trades', [])
            if recent_trades:
                recent_pnls = [t.get('pnl', 0) for t in recent_trades[-5:]]
                if recent_pnls and np.mean(recent_pnls) < -20:
                    stress_factors += 2
            return max(0.5, min(1.5, 1.0 + (stress_factors - 3) * 0.1))
        except Exception:
            return 1.0

    # ── performance extraction ───────────────────────────────

    async def _extract_performance_data_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive performance data with enhanced analytics"""
        try:
            performance_data: Dict[str, Any] = {}
            # Trades
            recent_trades = market_data.get('recent_trades', []) or []
            performance_data['recent_trades'] = recent_trades
            performance_data['trade_count'] = len(recent_trades)

            if recent_trades:
                pnls = [float(trade.get('pnl', 0) or 0) for trade in recent_trades]
                wins = sum(1 for p in pnls if p > 0)
                performance_data['win_rate'] = wins / max(1, len(pnls))
                performance_data['total_pnl'] = float(sum(pnls))
                performance_data['avg_pnl'] = float(np.mean(pnls))
                performance_data['pnl_std'] = float(np.std(pnls)) if len(pnls) > 1 else 0.0
                performance_data['max_win'] = float(max(pnls))
                performance_data['max_loss'] = float(min(pnls))
                performance_data['profit_factor'] = self._calculate_profit_factor(pnls)
                if len(pnls) >= 5:
                    recent_5 = pnls[-5:]
                    performance_data['recent_trend'] = float(np.mean(recent_5))
                    performance_data['trend_consistency'] = self._calculate_trend_consistency(recent_5)
                else:
                    performance_data['recent_trend'] = 0.0
                    performance_data['trend_consistency'] = 0.5
            else:
                performance_data.update({
                    'win_rate': 0.5, 'total_pnl': 0.0, 'avg_pnl': 0.0,
                    'pnl_std': 0.0, 'max_win': 0.0, 'max_loss': 0.0, 'profit_factor': 1.0,
                    'recent_trend': 0.0, 'trend_consistency': 0.5
                })

            # Risk metrics
            risk_metrics = market_data.get('risk_metrics', {}) or {}
            performance_data['current_balance'] = float(risk_metrics.get('balance', risk_metrics.get('equity', 10000)) or 10000.0)
            performance_data['drawdown'] = max(0.0, float(risk_metrics.get('current_drawdown', 0.0) or 0.0))
            performance_data['max_drawdown'] = max(0.0, float(risk_metrics.get('max_drawdown', 0.0) or 0.0))
            performance_data['risk_score'] = float(risk_metrics.get('risk_score', 0.5) or 0.5)

            # Consensus
            votes_data = market_data.get('votes', []) or []
            confidences: List[float] = []
            if isinstance(votes_data, list):
                for v in votes_data:
                    if isinstance(v, dict):
                        confidences.append(float(v.get('confidence', v.get('score', 0.5)) or 0.5))
                    elif isinstance(v, (int, float, np.generic)):
                        confidences.append(float(v))
            elif isinstance(votes_data, dict):
                for v in votes_data.values():
                    if isinstance(v, dict):
                        confidences.append(float(v.get('confidence', v.get('score', 0.5)) or 0.5))
                    elif isinstance(v, (int, float, np.generic)):
                        confidences.append(float(v))
            if confidences:
                performance_data['consensus'] = float(np.mean(confidences))
                performance_data['vote_agreement'] = float(1.0 - (np.std(confidences) if len(confidences) > 1 else 0.0))
                performance_data['vote_count'] = int(len(confidences))
                performance_data['consensus_strength'] = float(min(performance_data['consensus'], performance_data['vote_agreement']))
            else:
                performance_data.update({'consensus': 0.5, 'vote_agreement': 0.5, 'vote_count': 0, 'consensus_strength': 0.5})

            # Volatility (robust aggregation: handle dicts like {symbol: {atr, volatility}} or numeric values)
            volatility_data = market_data.get('volatility_data', {}) or {}
            vol_values: List[float] = []
            try:
                if isinstance(volatility_data, dict):
                    for v in volatility_data.values():
                        if isinstance(v, dict):
                            val = v.get('volatility', v.get('atr', None))
                            if isinstance(val, (int, float, np.generic)):
                                vol_values.append(float(val))
                        elif isinstance(v, (int, float, np.generic)):
                            vol_values.append(float(v))
                elif isinstance(volatility_data, list):
                    for v in volatility_data:
                        if isinstance(v, dict):
                            val = v.get('volatility', v.get('atr', None))
                            if isinstance(val, (int, float, np.generic)):
                                vol_values.append(float(val))
                        elif isinstance(v, (int, float, np.generic)):
                            vol_values.append(float(v))
                # Fallback single value
                elif isinstance(volatility_data, (int, float, np.generic)):
                    vol_values.append(float(volatility_data))
            except Exception:
                # On any parsing error, keep vol_values as collected so far
                pass

            performance_data['volatility'] = float(np.mean(vol_values)) if vol_values else 0.02
            performance_data['volatility_regime_score'] = self._get_volatility_regime_score()

            # Exposure (robust to positions being dict keyed by symbol or list of dicts)
            positions_raw = market_data.get('positions', []) or []

            pos_list: List[Dict[str, Any]] = []
            if isinstance(positions_raw, dict):
                for v in positions_raw.values():
                    if isinstance(v, dict):
                        pos_list.append(v)
            elif isinstance(positions_raw, list):
                for v in positions_raw:
                    if isinstance(v, dict):
                        pos_list.append(v)

            def _to_float_safe(x: Any) -> float:
                try:
                    if isinstance(x, (int, float, np.generic)):
                        return float(x)
                    if isinstance(x, str):
                        x = x.strip()
                        return float(x) if x else 0.0
                except Exception:
                    return 0.0
                return 0.0

            def _pos_exposure_value(p: Dict[str, Any]) -> float:
                # Prefer explicit notionals/exposure, else fall back to size/units, else derive units*price
                for key in ('notional', 'notional_eur', 'notional_usd', 'notional_value', 'exposure',
                            'size', 'units', 'quantity', 'qty', 'volume', 'amount'):
                    if key in p:
                        return _to_float_safe(p.get(key))
                units = _to_float_safe(p.get('units', p.get('size', 0)))
                price = _to_float_safe(p.get('price', p.get('entry_price', 0)))
                if units and price:
                    return units * price
                return units

            total_exposure = float(sum(abs(_pos_exposure_value(pos)) for pos in pos_list))
            performance_data['exposure'] = total_exposure
            performance_data['position_count'] = int(len(pos_list))
            performance_data['exposure_ratio'] = float(min(1.0, total_exposure / max(performance_data['current_balance'], 1)))

            # Strategy performance
            strategy_performance = market_data.get('strategy_performance', {}) or {}
            performance_data['strategy_effectiveness'] = float(strategy_performance.get('effectiveness_score', 0.5) or 0.5)
            performance_data['strategy_confidence'] = float(strategy_performance.get('confidence_score', 0.5) or 0.5)

            # Session performance
            session_metrics = market_data.get('session_metrics', {}) or {}
            performance_data['session_pnl'] = float(session_metrics.get('session_pnl', 0.0) or 0.0)
            performance_data['session_trades'] = int(session_metrics.get('session_trades', 0) or 0)

            # Sharpe
            if len(recent_trades) > 5:
                returns = [float(trade.get('pnl', 0) or 0) / max(performance_data['current_balance'], 1.0) for trade in recent_trades]
                performance_data['sharpe'] = float((np.sqrt(252) * np.mean(returns) / np.std(returns)) if np.std(returns) > 0 else 0.0)
            else:
                performance_data['sharpe'] = 0.0

            return performance_data

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_data_extraction")
            self.logger.warning(f"Performance data extraction failed: {error_context}")
            return self._get_safe_performance_defaults()

    def _calculate_profit_factor(self, pnls: List[float]) -> float:
        """Calculate profit factor from PnL list"""
        try:
            if not pnls:
                return 1.0
            gross_profit = sum(p for p in pnls if p > 0)
            gross_loss = abs(sum(p for p in pnls if p < 0))
            if gross_loss == 0:
                return float('inf') if gross_profit > 0 else 1.0
            return float(gross_profit / gross_loss)
        except Exception:
            return 1.0

    def _calculate_trend_consistency(self, pnls: List[float]) -> float:
        """Calculate trend consistency score"""
        try:
            if len(pnls) < 2:
                return 0.5
            directions = []
            for i in range(1, len(pnls)):
                if pnls[i] > pnls[i-1]:
                    directions.append(1)
                elif pnls[i] < pnls[i-1]:
                    directions.append(-1)
                else:
                    directions.append(0)
            if not directions:
                return 0.5
            direction_changes = sum(
                1 for i in range(1, len(directions))
                if directions[i] != directions[i-1] and directions[i] != 0 and directions[i-1] != 0
            )
            consistency = 1.0 - (direction_changes / max(1, len(directions) - 1))
            return float(max(0.0, min(1.0, consistency)))
        except Exception:
            return 0.5

    def _get_volatility_regime_score(self) -> float:
        """Get score based on current volatility regime"""
        volatility_scores = {
            'low': 0.8, 'medium': 0.7, 'high': 0.4, 'extreme': 0.2, 'unknown': 0.5,
            'very_low': 0.9, 'medium_high': 0.5
        }
        return float(volatility_scores.get(self.volatility_regime, 0.5))

    # ── performance stats ────────────────────────────────────

    async def _update_performance_statistics_comprehensive(self, performance_data: Dict[str, Any],
                                                           market_data: Dict[str, Any]):
        """Update comprehensive performance statistics with enhanced tracking"""
        try:
            stats_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'mode': self.current_mode,
                'win_rate': float(performance_data.get('win_rate', 0.5)),
                'avg_pnl': float(performance_data.get('avg_pnl', 0.0)),
                'total_pnl': float(performance_data.get('total_pnl', 0.0)),
                'drawdown': float(performance_data.get('drawdown', 0.0)),
                'max_drawdown': float(performance_data.get('max_drawdown', 0.0)),
                'consensus': float(performance_data.get('consensus', 0.5)),
                'consensus_strength': float(performance_data.get('consensus_strength', 0.5)),
                'volatility': float(performance_data.get('volatility', 0.02)),
                'volatility_regime_score': float(performance_data.get('volatility_regime_score', 0.5)),
                'sharpe': float(performance_data.get('sharpe', 0.0)),
                'profit_factor': float(performance_data.get('profit_factor', 1.0)),
                'trade_count': int(performance_data.get('trade_count', 0)),
                'exposure': float(performance_data.get('exposure', 0.0)),
                'exposure_ratio': float(performance_data.get('exposure_ratio', 0.0)),
                'strategy_effectiveness': float(performance_data.get('strategy_effectiveness', 0.5)),
                'regime': self.market_regime,
                'volatility_level': self.volatility_regime,
                'session': self.market_session,
                'recent_trend': float(performance_data.get('recent_trend', 0.0)),
                'trend_consistency': float(performance_data.get('trend_consistency', 0.5))
            }
            self.stats_history.append(stats_entry)

            # Mode analytics
            mp = self.mode_analytics[self.current_mode]
            mp['win_rates'].append(stats_entry['win_rate'])
            mp['pnl_values'].append(stats_entry['avg_pnl'])
            mp['total_pnl_values'].append(stats_entry['total_pnl'])
            mp['drawdowns'].append(stats_entry['drawdown'])
            mp['sharpe_ratios'].append(stats_entry['sharpe'])
            mp['profit_factors'].append(stats_entry['profit_factor'])
            mp['timestamps'].append(stats_entry['timestamp'])

            # Regime/session analytics
            if self.regime_awareness and self.market_regime != 'unknown':
                rp = self.regime_performance[self.market_regime]
                rp['modes'].append(self.current_mode)
                rp['performance'].append(stats_entry['avg_pnl'])
                rp['effectiveness'].append(stats_entry['strategy_effectiveness'])
                rp['timestamps'].append(stats_entry['timestamp'])

            if self.session_awareness and self.market_session != 'unknown':
                sp = self.session_performance[self.market_session]
                sp['modes'].append(self.current_mode)
                sp['performance'].append(stats_entry['avg_pnl'])
                sp['win_rates'].append(stats_entry['win_rate'])
                sp['timestamps'].append(stats_entry['timestamp'])

            # Effectiveness tracking
            eff = self._calculate_mode_effectiveness(stats_entry)
            self.effectiveness_tracking[self.current_mode].append({
                'timestamp': stats_entry['timestamp'],
                'effectiveness': float(eff),
                'performance_score': stats_entry['avg_pnl'],
                'risk_score': float(max(0.0, 1.0 - stats_entry['drawdown'])),
                'consistency_score': stats_entry['trend_consistency']
            })

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_statistics_update")
            self.logger.warning(f"Performance statistics update failed: {error_context}")

    def _calculate_mode_effectiveness(self, stats_entry: Dict[str, Any]) -> float:
        """Calculate effectiveness score for current mode"""
        try:
            performance_component = float(np.tanh(stats_entry.get('avg_pnl', 0.0) / 50.0) * 0.5 + 0.5)
            drawdown = float(stats_entry.get('drawdown', 0.0))
            risk_component = float(max(0.0, 1.0 - drawdown * 5))
            consistency_component = float(stats_entry.get('trend_consistency', 0.5))
            consensus_component = float(stats_entry.get('consensus_strength', 0.5))
            effectiveness = (
                0.4 * performance_component +
                0.3 * risk_component +
                0.2 * consistency_component +
                0.1 * consensus_component
            )
            return float(max(0.0, min(1.0, effectiveness)))
        except Exception:
            return 0.5

    # ── decision making ──────────────────────────────────────

    async def _make_intelligent_mode_decision_comprehensive(self, performance_data: Dict[str, Any],
                                                            market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make comprehensive intelligent mode decision with advanced analysis"""
        try:
            decision: Dict[str, Any] = {
                'current_mode': self.current_mode,
                'recommended_mode': self.current_mode,
                'confidence': 0.5,
                'reasoning': [],
                'decision_factors': {},
                'should_change': False,
                'analysis_details': {},
                'risk_assessment': {},
                'market_alignment': {}
            }

            # Market closed → safe mode
            if not self._is_market_open():
                decision.update({
                    'recommended_mode': 'safe',
                    'confidence': 1.0,
                    'reasoning': ['Market is closed - safety mode required'],
                    'should_change': self.current_mode != 'safe'
                })
                return decision

            # Auto mode guard
            if not self.auto_mode:
                decision['reasoning'] = ['Auto mode disabled - maintaining current mode']
                return decision

            # Persistence requirement
            if self.mode_persistence < self.min_persistence:
                decision['reasoning'] = [f'Mode persistence required ({self.mode_persistence}/{self.min_persistence})']
                return decision

            # Factors
            await self._calculate_decision_factors_comprehensive(performance_data, market_data)
            decision['decision_factors'] = self.decision_factors.copy()

            # Scores & assessments
            mode_scores = await self._calculate_mode_scores_comprehensive(performance_data, market_data)
            decision['analysis_details']['mode_scores'] = mode_scores

            risk_assessment = await self._perform_comprehensive_risk_assessment(performance_data, market_data)
            decision['risk_assessment'] = risk_assessment

            market_alignment = await self._assess_market_alignment_comprehensive(performance_data, market_data)
            decision['market_alignment'] = market_alignment

            # Best mode
            best_mode_analysis = self._find_optimal_mode_with_confidence(mode_scores, risk_assessment, market_alignment)
            recommended_mode = best_mode_analysis['mode']
            confidence = float(best_mode_analysis['confidence'])

            # Reasoning & change analysis
            reasoning = await self._generate_mode_reasoning_comprehensive(
                performance_data, market_data, mode_scores, risk_assessment, market_alignment
            )
            change_analysis = self._analyze_mode_change_necessity(recommended_mode, confidence, mode_scores, risk_assessment)

            decision.update({
                'recommended_mode': recommended_mode,
                'confidence': confidence,
                'reasoning': reasoning,
                'should_change': change_analysis['should_change'],
                'change_urgency': change_analysis.get('urgency', 'normal'),
                'improvement_potential': float(change_analysis.get('improvement', 0.0)),
                'analysis_details': {
                    **decision['analysis_details'],
                    'best_mode_analysis': best_mode_analysis,
                    'change_analysis': change_analysis
                }
            })
            return decision

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "intelligent_mode_decision")
            self.logger.error(f"Intelligent mode decision failed: {error_context}")
            return {
                'recommended_mode': self.current_mode,
                'confidence': 0.3,
                'should_change': False,
                'reasoning': [f'Decision analysis error: {error_context}']
            }

    async def _calculate_decision_factors_comprehensive(self, performance_data: Dict[str, Any],
                                                        market_data: Dict[str, Any]):
        """Calculate comprehensive decision factors with enhanced intelligence"""
        try:
            # Performance factor with trend analysis
            if len(self.stats_history) >= 5:
                recent_stats = list(self.stats_history)[-5:]
                avg_win_rate = float(np.mean([s['win_rate'] for s in recent_stats]))
                avg_pnl = float(np.mean([s['avg_pnl'] for s in recent_stats]))
                trend = float(np.mean([s.get('recent_trend', 0.0) for s in recent_stats]))
                win_rate_score = avg_win_rate
                pnl_score = float(np.tanh(avg_pnl / 100.0) * 0.5 + 0.5)
                trend_score_normalized = float(np.tanh(trend / 50.0) * 0.5 + 0.5)
                self.decision_factors['performance_score'] = (
                    0.4 * win_rate_score +
                    0.4 * pnl_score +
                    0.2 * trend_score_normalized
                )
            else:
                self.decision_factors['performance_score'] = 0.5

            # Risk factor
            drawdown = float(performance_data.get('drawdown', 0.0))
            max_drawdown = float(performance_data.get('max_drawdown', 0.0))
            volatility = float(performance_data.get('volatility', 0.02))
            exposure_ratio = float(performance_data.get('exposure_ratio', 0.0))
            drawdown_score = max(0.0, 1.0 - drawdown * 5)
            max_drawdown_score = max(0.0, 1.0 - max_drawdown * 3)
            volatility_score = max(0.0, 1.0 - volatility * 20)
            exposure_score = max(0.0, 1.0 - exposure_ratio)
            self.decision_factors['risk_score'] = (
                0.3 * drawdown_score +
                0.3 * max_drawdown_score +
                0.2 * volatility_score +
                0.2 * exposure_score
            )

            # Consensus
            consensus = float(performance_data.get('consensus', 0.5))
            consensus_strength = float(performance_data.get('consensus_strength', 0.5))
            vote_count = int(performance_data.get('vote_count', 0))
            vote_confidence = min(1.0, vote_count / 5.0)
            self.decision_factors['consensus_score'] = (
                0.4 * consensus +
                0.4 * consensus_strength +
                0.2 * vote_confidence
            )

            # Context
            regime_score = self._get_regime_score_enhanced(self.market_regime)
            volatility_level_score = self._get_volatility_level_score_enhanced(self.volatility_regime)
            session_score = self._get_session_score_enhanced(self.market_session)
            self.decision_factors['market_context_score'] = (
                0.4 * regime_score +
                0.4 * volatility_level_score +
                0.2 * session_score
            )
            # Additional
            self.decision_factors['volatility_score'] = volatility_score
            self.decision_factors['regime_score'] = regime_score
            self.decision_factors['session_score'] = session_score
            self.decision_factors['trend_score'] = float(performance_data.get('trend_consistency', 0.5))
            self.decision_factors['stability_score'] = self._calculate_stability_score(performance_data)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "decision_factors_calculation")
            self.logger.warning(f"Decision factors calculation failed: {error_context}")

    def _get_regime_score_enhanced(self, regime: str) -> float:
        regime_scores = {
            'trending': 0.85, 'breakout': 0.8, 'momentum': 0.75, 'normal': 0.7,
            'ranging': 0.6, 'consolidation': 0.55, 'volatile': 0.35, 'uncertain': 0.3,
            'reversal': 0.4, 'unknown': 0.5
        }
        return float(regime_scores.get(regime, 0.5))

    def _get_volatility_level_score_enhanced(self, vol_level: str) -> float:
        vol_scores = {
            'very_low': 0.9, 'low': 0.8, 'medium': 0.7, 'medium_high': 0.5,
            'high': 0.3, 'extreme': 0.15, 'unknown': 0.5
        }
        return float(vol_scores.get(vol_level, 0.5))

    def _get_session_score_enhanced(self, session: str) -> float:
        session_scores = {
            'overlap_london_new_york': 0.9, 'london': 0.8, 'new_york': 0.8,
            'asian': 0.6, 'sydney': 0.55, 'weekend': 0.2, 'holiday': 0.25,
            'rollover': 0.3, 'unknown': 0.5
        }
        return float(session_scores.get(session, 0.5))

    def _calculate_stability_score(self, performance_data: Dict[str, Any]) -> float:
        """Calculate system stability score based on various factors"""
        try:
            factors: List[float] = []
            if len(self.stats_history) >= 5:
                recent_pnls = [float(s.get('avg_pnl', 0.0)) for s in list(self.stats_history)[-5:]]
                denom = abs(float(np.mean(recent_pnls))) + 10.0
                pnl_stability = 1.0 - (float(np.std(recent_pnls)) / denom)
                factors.append(max(0.0, min(1.0, float(pnl_stability))))
            drawdown = float(performance_data.get('drawdown', 0.0))
            factors.append(max(0.0, 1.0 - drawdown * 3))
            factors.append(float(performance_data.get('trend_consistency', 0.5)))
            persistence_stability = min(1.0, self.mode_persistence / max(1.0, (self.min_persistence * 2)))
            factors.append(float(persistence_stability))
            return float(np.mean(factors)) if factors else 0.5
        except Exception:
            return 0.5

    async def _calculate_mode_scores_comprehensive(self, performance_data: Dict[str, Any],
                                                   market_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive scores for each trading mode"""
        try:
            mode_scores: Dict[str, float] = {}
            for mode in self.TRADING_MODES:
                score = await self._calculate_single_mode_score_comprehensive(mode, performance_data, market_data)
                mode_scores[mode] = float(score)
            return mode_scores
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "comprehensive_mode_scores")
            self.logger.warning(f"Comprehensive mode scores calculation failed: {error_context}")
            return {mode: 0.5 for mode in self.TRADING_MODES}

    async def _calculate_single_mode_score_comprehensive(self, mode: str, performance_data: Dict[str, Any],
                                                         market_data: Dict[str, Any]) -> float:
        """Calculate comprehensive score for a single mode with enhanced logic"""
        try:
            mode_config = self.TRADING_MODES[mode]
            thresholds = self.mode_thresholds[mode]
            drawdown = float(performance_data.get('drawdown', 0.0))
            if drawdown > float(thresholds['max_drawdown']):
                return 0.1
            exposure_ratio = float(performance_data.get('exposure_ratio', 0.0))
            if exposure_ratio > float(thresholds['max_exposure']):
                return 0.2
            if len(self.stats_history) >= 3:
                recent_stats = list(self.stats_history)[-3:]
                avg_wr = float(np.mean([s['win_rate'] for s in recent_stats]))
                if avg_wr < float(thresholds['min_win_rate']):
                    return 0.15

            performance_score = float(self.decision_factors['performance_score'])
            risk_score = float(self.decision_factors['risk_score'])
            consensus_score = float(self.decision_factors['consensus_score'])
            market_score = float(self.decision_factors['market_context_score'])
            stability_score = float(self.decision_factors['stability_score'])

            # Base score
            score = 0.5

            if mode == 'safe':
                score = (
                    risk_score * 0.4 +
                    stability_score * 0.3 +
                    performance_score * 0.2 +
                    market_score * 0.1
                )
                if drawdown > 0.05 or performance_score < 0.4 or self.volatility_regime in ['high', 'extreme']:
                    score += 0.3

            elif mode == 'normal':
                score = (
                    performance_score * self.performance_weight +
                    risk_score * self.risk_weight +
                    consensus_score * self.consensus_weight +
                    market_score * self.market_context_weight
                ) + stability_score * 0.1

            elif mode == 'aggressive':
                score = (
                    performance_score * 0.5 +
                    consensus_score * 0.3 +
                    market_score * 0.15 +
                    risk_score * 0.05
                )
                if (performance_score < 0.6 or consensus_score < 0.5 or
                        self.volatility_regime in ['high', 'extreme']):
                    score *= 0.6
                if self.market_regime in ['trending', 'breakout', 'momentum']:
                    score += 0.15

            elif mode == 'extreme':
                score = (
                    performance_score * 0.6 +
                    consensus_score * 0.25 +
                    market_score * 0.15
                )
                req = [
                    performance_score >= 0.7,
                    consensus_score >= 0.65,
                    self.decision_factors['regime_score'] >= 0.7,
                    self.volatility_regime not in ['high', 'extreme'],
                    stability_score >= 0.6
                ]
                if not all(req):
                    score *= 0.3
                if self.market_regime not in ['trending', 'breakout', 'momentum']:
                    score *= 0.5

            if self.volatility_scaling:
                score *= self._get_volatility_adjustment_factor()

            if self.effectiveness_tracking.get(mode):
                recent_eff = np.mean([e['effectiveness'] for e in self.effectiveness_tracking[mode][-3:]])
                score += (float(recent_eff) - 0.5) * 0.2

            return float(np.clip(score, 0.0, 1.0))

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, f"single_mode_score_{mode}")
            self.logger.warning(f"Single mode score calculation failed for {mode}: {error_context}")
            return 0.5

    def _get_volatility_adjustment_factor(self) -> float:
        """Get volatility adjustment factor for mode scoring"""
        adjustments = {
            'very_low': 1.1, 'low': 1.05, 'medium': 1.0,
            'medium_high': 0.95, 'high': 0.85, 'extreme': 0.7, 'unknown': 1.0
        }
        return float(adjustments.get(self.volatility_regime, 1.0))

    # ── risk & alignment ─────────────────────────────────────

    async def _perform_comprehensive_risk_assessment(self, performance_data: Dict[str, Any],
                                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive risk assessment for mode decision"""
        try:
            risk_assessment = {
                'overall_risk_level': 'medium',
                'risk_factors': [],
                'risk_mitigations': [],
                'risk_score_by_mode': {}
            }
            drawdown_risk = self._assess_drawdown_risk(performance_data)
            volatility_risk = self._assess_volatility_risk(performance_data, market_data)
            exposure_risk = self._assess_exposure_risk(performance_data)
            consensus_risk = self._assess_consensus_risk(performance_data)
            market_risk = self._assess_market_risk(market_data)
            risk_factors = [drawdown_risk, volatility_risk, exposure_risk, consensus_risk, market_risk]
            high_risk = [rf for rf in risk_factors if rf['level'] in ['high', 'extreme']]

            if any(rf['level'] == 'extreme' for rf in risk_factors):
                risk_assessment['overall_risk_level'] = 'extreme'
            elif len(high_risk) >= 2:
                risk_assessment['overall_risk_level'] = 'high'
            elif len(high_risk) >= 1:
                risk_assessment['overall_risk_level'] = 'medium_high'
            elif any(rf['level'] == 'medium' for rf in risk_factors):
                risk_assessment['overall_risk_level'] = 'medium'
            else:
                risk_assessment['overall_risk_level'] = 'low'

            risk_assessment['risk_factors'] = [rf for rf in risk_factors if rf['level'] != 'low']
            risk_assessment['risk_mitigations'] = self._generate_risk_mitigations(high_risk)

            for mode in self.TRADING_MODES:
                risk_assessment['risk_score_by_mode'][mode] = self._calculate_mode_risk_score(mode, risk_factors)

            return risk_assessment

        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "comprehensive_risk_assessment")
            return {'overall_risk_level': 'medium', 'risk_factors': [], 'risk_score_by_mode': {}}

    def _assess_drawdown_risk(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        drawdown = float(performance_data.get('drawdown', 0.0))
        max_drawdown = float(performance_data.get('max_drawdown', 0.0))
        if max_drawdown > 0.2 or drawdown > 0.1:
            level = 'extreme' if max_drawdown > 0.25 else 'high'
        elif max_drawdown > 0.08 or drawdown > 0.05:
            level = 'medium'
        else:
            level = 'low'
        return {
            'type': 'drawdown', 'level': level,
            'current_value': drawdown, 'max_value': max_drawdown,
            'description': f'Current drawdown: {drawdown:.1%}, Max: {max_drawdown:.1%}'
        }

    def _assess_volatility_risk(self, performance_data: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        volatility = float(performance_data.get('volatility', 0.02))
        volatility_regime = self.volatility_regime
        if volatility_regime == 'extreme' or volatility > 0.08:
            level = 'extreme'
        elif volatility_regime == 'high' or volatility > 0.05:
            level = 'high'
        elif volatility_regime == 'medium_high' or volatility > 0.03:
            level = 'medium'
        else:
            level = 'low'
        return {
            'type': 'volatility', 'level': level,
            'current_value': volatility, 'regime': volatility_regime,
            'description': f'Volatility: {volatility:.2%}, Regime: {volatility_regime}'
        }

    def _assess_exposure_risk(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        exposure_ratio = float(performance_data.get('exposure_ratio', 0.0))
        position_count = int(performance_data.get('position_count', 0))
        if exposure_ratio > 0.9 or position_count > 15:
            level = 'extreme'
        elif exposure_ratio > 0.7 or position_count > 10:
            level = 'high'
        elif exposure_ratio > 0.5 or position_count > 6:
            level = 'medium'
        else:
            level = 'low'
        return {
            'type': 'exposure', 'level': level,
            'exposure_ratio': exposure_ratio, 'position_count': position_count,
            'description': f'Exposure: {exposure_ratio:.1%}, Positions: {position_count}'
        }

    def _assess_consensus_risk(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        consensus = float(performance_data.get('consensus', 0.5))
        consensus_strength = float(performance_data.get('consensus_strength', 0.5))
        vote_count = int(performance_data.get('vote_count', 0))
        if consensus < 0.3 or consensus_strength < 0.3 or vote_count < 2:
            level = 'high'
        elif consensus < 0.4 or consensus_strength < 0.4 or vote_count < 3:
            level = 'medium'
        else:
            level = 'low'
        return {
            'type': 'consensus', 'level': level,
            'consensus': consensus, 'strength': consensus_strength, 'vote_count': vote_count,
            'description': f'Consensus: {consensus:.1%}, Strength: {consensus_strength:.1%}, Votes: {vote_count}'
        }

    def _assess_market_risk(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        regime = self.market_regime
        volatility_regime = self.volatility_regime
        session = self.market_session
        high_risk_regimes = ['volatile', 'uncertain', 'unknown']
        high_risk_volatility = ['high', 'extreme']
        low_liquidity_sessions = ['weekend', 'holiday', 'rollover']
        if (regime in high_risk_regimes and volatility_regime in high_risk_volatility) or session in low_liquidity_sessions:
            level = 'extreme'
        elif regime in high_risk_regimes or volatility_regime in high_risk_volatility:
            level = 'high'
        elif regime in ['reversal'] or volatility_regime == 'medium_high':
            level = 'medium'
        else:
            level = 'low'
        return {
            'type': 'market', 'level': level,
            'regime': regime, 'volatility_regime': volatility_regime, 'session': session,
            'description': f'Regime: {regime}, Volatility: {volatility_regime}, Session: {session}'
        }

    def _generate_risk_mitigations(self, high_risk_factors: List[Dict[str, Any]]) -> List[str]:
        mitigations: List[str] = []
        for rf in high_risk_factors:
            t = rf['type']; lvl = rf['level']
            if t == 'drawdown':
                mitigations.append("Switch to safe mode immediately to preserve capital" if lvl == 'extreme'
                                   else "Reduce position sizing and consider defensive positions")
            elif t == 'volatility':
                mitigations.append("Halt new trades until volatility subsides" if lvl == 'extreme'
                                   else "Use wider stops and reduce leverage")
            elif t == 'exposure':
                mitigations.append("Reduce position count and total exposure")
            elif t == 'consensus':
                mitigations.append("Wait for stronger committee consensus before increasing risk")
            elif t == 'market':
                mitigations.append("Adjust strategy for current market regime and session")
        return mitigations

    def _calculate_mode_risk_score(self, mode: str, risk_factors: List[Dict[str, Any]]) -> float:
        try:
            risk_tolerance = float(self.TRADING_MODES[mode]['risk_multiplier'])
            risk_penalty = 0.0
            for rf in risk_factors:
                level = rf['level']
                base = {'low': 0.0, 'medium': 0.1, 'high': 0.3, 'extreme': 0.6}.get(level, 0.1)
                risk_penalty += base / max(0.1, risk_tolerance)
            return float(max(0.0, 1.0 - risk_penalty))
        except Exception:
            return 0.5

    async def _assess_market_alignment_comprehensive(self, performance_data: Dict[str, Any],
                                                     market_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            alignment = {
                'regime_alignment': {},
                'volatility_alignment': {},
                'session_alignment': {},
                'overall_alignment': 0.5
            }
            for mode in self.TRADING_MODES:
                tol = self.TRADING_MODES[mode]['volatility_tolerance']
                alignment['regime_alignment'][mode] = self._score_regime_alignment(mode, self.market_regime)
                alignment['volatility_alignment'][mode] = self._score_volatility_alignment(tol, self.volatility_regime)
                alignment['session_alignment'][mode] = self._score_session_alignment(mode, self.market_session)

            scores = []
            for mode in self.TRADING_MODES:
                s = 0.5 * alignment['regime_alignment'][mode] + 0.3 * alignment['volatility_alignment'][mode] + 0.2 * alignment['session_alignment'][mode]
                scores.append(s)
            alignment['overall_alignment'] = float(np.mean(scores)) if scores else 0.5
            alignment['best_aligned_mode'] = max(self.TRADING_MODES.keys(), key=lambda m: (
                0.5 * alignment['regime_alignment'][m] +
                0.3 * alignment['volatility_alignment'][m] +
                0.2 * alignment['session_alignment'][m]
            ))
            return alignment
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "market_alignment_assessment")
            return {'overall_alignment': 0.5, 'regime_alignment': {}, 'volatility_alignment': {}, 'session_alignment': {}}

    def _score_regime_alignment(self, mode: str, regime: str) -> float:
        regime_mode_scores = {
            'trending': {'extreme': 0.9, 'aggressive': 0.8, 'normal': 0.6, 'safe': 0.3},
            'breakout': {'extreme': 0.85, 'aggressive': 0.8, 'normal': 0.5, 'safe': 0.4},
            'momentum': {'extreme': 0.8, 'aggressive': 0.75, 'normal': 0.6, 'safe': 0.4},
            'volatile': {'safe': 0.8, 'normal': 0.5, 'aggressive': 0.3, 'extreme': 0.1},
            'ranging': {'normal': 0.8, 'aggressive': 0.6, 'safe': 0.7, 'extreme': 0.4},
            'reversal': {'safe': 0.7, 'normal': 0.6, 'aggressive': 0.4, 'extreme': 0.2},
            'uncertain': {'safe': 0.9, 'normal': 0.5, 'aggressive': 0.2, 'extreme': 0.1},
            'unknown': {'safe': 0.8, 'normal': 0.6, 'aggressive': 0.4, 'extreme': 0.2}
        }
        return float(regime_mode_scores.get(regime, {}).get(mode, 0.5))

    def _score_volatility_alignment(self, mode_tolerance: str, volatility_regime: str) -> float:
        tolerance_scores = {
            'low': {'very_low': 0.9, 'low': 0.8, 'medium': 0.5, 'medium_high': 0.3, 'high': 0.1, 'extreme': 0.05},
            'medium': {'very_low': 0.7, 'low': 0.8, 'medium': 0.9, 'medium_high': 0.7, 'high': 0.4, 'extreme': 0.2},
            'medium-high': {'very_low': 0.6, 'low': 0.7, 'medium': 0.8, 'medium_high': 0.9, 'high': 0.6, 'extreme': 0.3},
            'high': {'very_low': 0.5, 'low': 0.6, 'medium': 0.7, 'medium_high': 0.8, 'high': 0.9, 'extreme': 0.6}
        }
        return float(tolerance_scores.get(mode_tolerance, {}).get(volatility_regime, 0.5))

    def _score_session_alignment(self, mode: str, session: str) -> float:
        session_mode_scores = {
            'overlap_london_new_york': {'extreme': 0.9, 'aggressive': 0.85, 'normal': 0.8, 'safe': 0.6},
            'london': {'extreme': 0.8, 'aggressive': 0.8, 'normal': 0.8, 'safe': 0.7},
            'new_york': {'extreme': 0.8, 'aggressive': 0.8, 'normal': 0.8, 'safe': 0.7},
            'asian': {'aggressive': 0.6, 'normal': 0.7, 'safe': 0.8, 'extreme': 0.4},
            'sydney': {'normal': 0.6, 'safe': 0.7, 'aggressive': 0.5, 'extreme': 0.3},
            'weekend': {'safe': 0.9, 'normal': 0.3, 'aggressive': 0.1, 'extreme': 0.05},
            'holiday': {'safe': 0.8, 'normal': 0.4, 'aggressive': 0.2, 'extreme': 0.1},
            'rollover': {'safe': 0.7, 'normal': 0.5, 'aggressive': 0.3, 'extreme': 0.1}
        }
        return float(session_mode_scores.get(session, {}).get(mode, 0.6))

    def _find_optimal_mode_with_confidence(self, mode_scores: Dict[str, float],
                                           risk_assessment: Dict[str, Any],
                                           market_alignment: Dict[str, Any]) -> Dict[str, Any]:
        """Find optimal mode with confidence assessment"""
        try:
            final_scores: Dict[str, float] = {}
            for mode in self.TRADING_MODES:
                base_score = float(mode_scores.get(mode, 0.5))
                risk_score = float(risk_assessment.get('risk_score_by_mode', {}).get(mode, 0.5))
                regime_align = float(market_alignment.get('regime_alignment', {}).get(mode, 0.5))
                volatility_align = float(market_alignment.get('volatility_alignment', {}).get(mode, 0.5))
                session_align = float(market_alignment.get('session_alignment', {}).get(mode, 0.5))
                alignment_score = (0.5 * regime_align + 0.3 * volatility_align + 0.2 * session_align)
                final_scores[mode] = float(0.5 * base_score + 0.3 * risk_score + 0.2 * alignment_score)

            best_mode = max(final_scores.items(), key=lambda x: x[1])
            sorted_scores = sorted(final_scores.values(), reverse=True)
            sep = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
            confidence = float(min(0.95, 0.5 + sep)) if len(sorted_scores) >= 2 else float(best_mode[1])

            return {'mode': best_mode[0], 'confidence': confidence, 'score': float(best_mode[1]),
                    'all_scores': final_scores, 'score_separation': sep}
        except Exception as e:
            _ = self.error_pinpointer.analyze_error(e, "optimal_mode_finding")
            return {'mode': self.current_mode, 'confidence': 0.5, 'score': 0.5}

    def _analyze_mode_change_necessity(self, recommended_mode: str, confidence: float,
                                       mode_scores: Dict[str, float],
                                       risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if mode change is necessary with enhanced logic"""
        try:
            current_score = float(mode_scores.get(self.current_mode, 0.5))
            recommended_score = float(mode_scores.get(recommended_mode, current_score))
            improvement = float(recommended_score - current_score)

            change_threshold = 0.1
            if risk_assessment.get('overall_risk_level') in ['high', 'extreme']:
                change_threshold = 0.05
            elif self.mode_persistence < self.min_persistence * 2:
                change_threshold = 0.15

            urgency = 'normal'
            if improvement > 0.5:
                urgency = 'critical'
            elif improvement > 0.3:
                urgency = 'high'
            elif risk_assessment.get('overall_risk_level') == 'extreme':
                urgency = 'emergency'

            should_change = (recommended_mode != self.current_mode and
                             improvement > change_threshold and
                             confidence > self.mode_intelligence['confidence_threshold'])

            return {
                'should_change': bool(should_change),
                'improvement': improvement,
                'change_threshold': float(change_threshold),
                'urgency': urgency,
                'confidence_met': bool(confidence > self.mode_intelligence['confidence_threshold']),
                'risk_justification': bool(risk_assessment.get('overall_risk_level') in ['high', 'extreme'])
            }
        except Exception:
            return {'should_change': False, 'improvement': 0.0, 'urgency': 'normal'}

    # ── newly implemented production-grade helpers ───────────

    async def _generate_mode_reasoning_comprehensive(self, performance_data: Dict[str, Any], market_data: Dict[str, Any],
                                                     mode_scores: Dict[str, float], risk_assessment: Dict[str, Any],
                                                     market_alignment: Dict[str, Any]) -> List[str]:
        """Generate comprehensive reasoning for mode decision"""
        try:
            reasons: List[str] = []
            reasons.append(f"Performance score={self.decision_factors['performance_score']:.2f}, "
                           f"Risk score={self.decision_factors['risk_score']:.2f}, "
                           f"Consensus={self.decision_factors['consensus_score']:.2f}, "
                           f"Context={self.decision_factors['market_context_score']:.2f}")
            reasons.append(f"Regime={self.market_regime}, Volatility={self.volatility_regime}, Session={self.market_session}")
            if risk_assessment.get('risk_factors'):
                rf_short = ", ".join([rf['type'] for rf in risk_assessment['risk_factors'][:3]])
                reasons.append(f"Risk factors: {rf_short}")
            if 'best_aligned_mode' in market_alignment:
                reasons.append(f"Best aligned mode by context: {market_alignment['best_aligned_mode']}")
            top_modes = sorted(mode_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            reasons.append("Top mode scores: " + ", ".join([f"{m}={s:.2f}" for m, s in top_modes]))
            return reasons
        except Exception:
            return ["Mode decision based on comprehensive analysis"]

    async def _apply_mode_decision_comprehensive(self, decision: Dict[str, Any], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply mode decision with comprehensive tracking"""
        try:
            changed = False
            old_mode = self.current_mode
            if decision.get('should_change'):
                self.current_mode = decision['recommended_mode']
                self.mode_persistence = 0
                self.mode_stats['total_switches'] += 1
                self.mode_stats['auto_switches'] += 1
                self.mode_stats['current_mode_duration'] = 0
                self.last_mode_change = datetime.datetime.now().isoformat()
                self.last_change_reason = "; ".join(decision.get('reasoning', [])[:2])
                changed = True

                self.mode_history.append({
                    'timestamp': self.last_mode_change,
                    'from_mode': old_mode,
                    'to_mode': self.current_mode,
                    'reason': self.last_change_reason,
                    'confidence': float(decision.get('confidence', 0.5)),
                    'auto': True,
                    'context': {
                        'regime': self.market_regime,
                        'volatility': self.volatility_regime,
                        'session': self.market_session
                    }
                })
                self.logger.info(format_operator_message(
                    icon="🎛️",
                    message="Auto mode change executed",
                    from_mode=old_mode,
                    to_mode=self.current_mode,
                    confidence=f"{decision.get('confidence', 0.0):.2f}"
                ))
            else:
                # increase persistence if no change
                self.mode_persistence += 1

            # Decision trace
            self.decision_trace.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'current_mode': old_mode,
                'recommended_mode': decision.get('recommended_mode', old_mode),
                'changed': changed,
                'confidence': float(decision.get('confidence', 0.5))
            })
            return {"mode_changed": changed, "from": old_mode, "to": self.current_mode}
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "apply_mode_decision")
            self.logger.warning(f"Apply mode decision failed: {error_context}")
            return {"mode_changed": False}

    async def _analyze_mode_effectiveness_comprehensive(self, performance_data: Dict[str, Any],
                                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mode effectiveness comprehensively"""
        try:
            if not self.effectiveness_tracking.get(self.current_mode):
                eff = 0.5
            else:
                recent = self.effectiveness_tracking[self.current_mode][-5:]
                eff = float(np.mean([e['effectiveness'] for e in recent]))
            self.mode_stats['mode_effectiveness'] = float(eff)
            # Best performing mode (last 20 effectiveness points)
            best = self.current_mode
            best_score = eff
            for mode, hist in self.effectiveness_tracking.items():
                if hist:
                    score = float(np.mean([e['effectiveness'] for e in hist[-20:]]))
                    if score > best_score:
                        best_score = score
                        best = mode
            self.mode_stats['best_performing_mode'] = best
            return {"current_effectiveness": eff, "best_mode": best, "best_score": best_score}
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "effectiveness_analysis")
            self.logger.warning(f"Effectiveness analysis failed: {error_context}")
            return {"current_effectiveness": 0.5}

    async def _update_adaptive_thresholds_comprehensive(self, performance_data: Dict[str, Any],
                                                        market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update adaptive thresholds comprehensively (slow drift, volatility-aware)"""
        try:
            updates = {}
            # Drift step
            alpha = float(self.mode_intelligence.get('adaptation_speed', 0.1)) * 0.1
            for mode, th in self.mode_thresholds.items():
                # Use recent effectiveness to slightly relax/tighten
                if self.effectiveness_tracking.get(mode):
                    eff = float(np.mean([e['effectiveness'] for e in self.effectiveness_tracking[mode][-5:]]))
                else:
                    eff = 0.5
                # If effective, relax slightly, else tighten slightly
                sign = 1 if eff > 0.55 else -1
                # Drawdown tolerance
                th['max_drawdown'] = float(np.clip(th['max_drawdown'] * (1 + sign * alpha * 0.1), 0.02, 0.2))
                # Win-rate requirement
                th['min_win_rate'] = float(np.clip(th['min_win_rate'] * (1 - sign * alpha * 0.1), 0.0, 0.8))
                # Consensus requirement
                th['min_consensus'] = float(np.clip(th['min_consensus'] * (1 - sign * alpha * 0.1), 0.0, 0.8))
                updates[mode] = dict(th)

            self.threshold_adaptations.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'type': 'periodic',
                'updates': updates
            })
            return {"thresholds_updated':": True, "updates": updates}
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "threshold_updates")
            self.logger.warning(f"Threshold update failed: {error_context}")
            return {"thresholds_updated": False}

    async def _generate_intelligent_mode_recommendations(self, mode_decision: Dict[str, Any],
                                                         effectiveness_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent mode recommendations"""
        try:
            recs: List[str] = []
            eff = float(effectiveness_analysis.get('current_effectiveness', 0.5))
            if mode_decision.get('should_change'):
                recs.append(f"Switch to {mode_decision['recommended_mode']} (confidence {mode_decision.get('confidence', 0.5):.2f}).")
            if eff < 0.4:
                recs.append("Review strategy parameters; effectiveness is low.")
            if self.decision_factors.get('risk_score', 0.5) < 0.5:
                recs.append("Tighten risk: reduce exposure or increase stop discipline.")
            if self.decision_factors.get('consensus_score', 0.5) < 0.45:
                recs.append("Wait for stronger committee consensus before scaling risk.")
            if not recs:
                recs.append("Maintain current mode; conditions stable.")
            return recs
        except Exception:
            return ["Continue current mode approach"]

    async def _generate_comprehensive_mode_thesis(self, mode_decision: Dict[str, Any],
                                                  effectiveness_analysis: Dict[str, Any]) -> str:
        """Generate comprehensive mode thesis"""
        try:
            parts = [
                f"Mode: {self.current_mode.upper()} | Effectiveness {effectiveness_analysis.get('current_effectiveness', 0.5):.2f}.",
                f"Factors → Perf {self.decision_factors['performance_score']:.2f}, "
                f"Risk {self.decision_factors['risk_score']:.2f}, "
                f"Consensus {self.decision_factors['consensus_score']:.2f}, "
                f"Context {self.decision_factors['market_context_score']:.2f}.",
                f"Market: regime={self.market_regime}, vol={self.volatility_regime}, session={self.market_session}."
            ]
            if mode_decision.get('should_change'):
                parts.append(f"Recommendation: switch to {mode_decision['recommended_mode']} "
                             f"(confidence {mode_decision.get('confidence', 0.5):.2f}, "
                             f"urgency {mode_decision.get('change_urgency', 'normal')}).")
            else:
                parts.append("No mode change recommended; persistence or thresholds not met.")
            return " ".join(parts)
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Trading mode management proceeding; thesis error: {error_context}"

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus comprehensively (single-writer keys only)"""
        try:
            # Publish ONLY what we provide
            self.smart_bus.set('trading_mode', results.get('trading_mode', self.current_mode),
                               module='TradingModeManager', thesis=thesis)
            self.smart_bus.set('mode_config', results.get('mode_config', {}),
                               module='TradingModeManager', thesis="Mode configuration updated")
            self.smart_bus.set('mode_stats', results.get('mode_stats', {}),
                               module='TradingModeManager', thesis="Mode stats updated")
            self.smart_bus.set('mode_effectiveness', results.get('mode_effectiveness', 0.5),
                               module='TradingModeManager', thesis="Mode effectiveness updated")
            self.smart_bus.set('decision_factors', results.get('decision_factors', {}),
                               module='TradingModeManager', thesis="Decision factors updated")
            self.smart_bus.set('mode_thresholds', results.get('mode_thresholds', {}),
                               module='TradingModeManager', thesis="Mode thresholds updated")
            # Do not write 'market_context' (owned by MarketDataProvider). If needed, embed context in mode_stats.
            try:
                mc = results.get('market_context', {})
                if isinstance(mc, dict) and mc:
                    stats = results.get('mode_stats', {}) or {}
                    stats = dict(stats)
                    stats.setdefault('context', mc)
                    self.smart_bus.set('mode_stats', stats,
                                       module='TradingModeManager', thesis="Mode stats updated (with context)")
            except Exception:
                pass
            # Do not publish contested key 'mode_recommendations' (canonical owner: OpponentModeEnhancer)
            # If needed for dashboards, include recommendations under namespaced stats instead.
            recs = results.get('mode_recommendations', [])
            if isinstance(recs, list) and recs:
                stats = results.get('mode_stats', {}) or {}
                stats = dict(stats)
                stats['recommendations'] = list(recs)
                self.smart_bus.set('mode_stats', stats,
                                   module='TradingModeManager', thesis="Mode stats updated (with recommendations)")

            # Namespaced health/status
            self._post_health_status()

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, 'smartinfobus_update')
            self.logger.warning(f"SmartInfoBus update failed: {error_context}")

    # ── utilities & payload shaping ──────────────────────────

    def _is_market_open(self) -> bool:
        """Check if market is open based on schedule"""
        if not self.market_schedule:
            return True
        try:
            import pytz
            timezone = self.market_schedule.get('timezone', 'UTC')
            tz = pytz.timezone(timezone)
            now = datetime.datetime.now(tz)
            weekday = now.weekday()
            if weekday in self.market_schedule.get('close_days', [5, 6]):
                return False
            if 'holidays' in self.market_schedule:
                today_str = now.strftime('%Y-%m-%d')
                if today_str in self.market_schedule['holidays']:
                    return False
            hour = now.hour
            open_hour = int(self.market_schedule.get('open_hour', 0))
            close_hour = int(self.market_schedule.get('close_hour', 23))
            if hour < open_hour or hour >= close_hour:
                return False
            return True
        except Exception as e:
            self.logger.warning(f"Market schedule check failed: {e}")
            return True

    def _get_mode_configuration(self) -> Dict[str, Any]:
        """Get current mode configuration"""
        cfg = self.TRADING_MODES[self.current_mode]
        return {
            'mode': self.current_mode,
            'auto': self.auto_mode,
            'persistence': self.mode_persistence,
            'effectiveness': float(self.mode_stats.get('mode_effectiveness', 0.5)),
            'description': cfg.get('description', ''),
            'risk_multiplier': float(cfg.get('risk_multiplier', 1.0)),
            'max_exposure': float(cfg.get('max_exposure', 0.5))
        }

    def _safe_thresholds_copy(self) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for m, t in self.mode_thresholds.items():
            out[m] = {k: (float(v) if isinstance(v, (int, float, np.generic)) else v) for k, v in t.items()}
        return out

    def _get_comprehensive_mode_stats(self) -> Dict[str, Any]:
        """Get comprehensive mode statistics"""
        return {
            **self.mode_stats,
            'decision_factors': self.decision_factors.copy(),
            'rolling_stats': self._calculate_rolling_stats(),
            'mode_analytics_summary': self._get_mode_analytics_summary()
        }

    def _get_mode_analytics_summary(self) -> Dict[str, Any]:
        """Get summary of mode analytics"""
        summary: Dict[str, Any] = {}
        for mode, analytics in self.mode_analytics.items():
            if analytics.get('win_rates'):
                summary[mode] = {
                    'avg_win_rate': float(np.mean(analytics['win_rates'])),
                    'avg_pnl': float(np.mean(analytics['pnl_values'])) if analytics.get('pnl_values') else 0.0,
                    'total_periods': int(len(analytics['win_rates'])),
                    'best_sharpe': float(max(analytics.get('sharpe_ratios', [0.0]))),
                    'best_profit_factor': float(max(analytics.get('profit_factors', [1.0])))
                }
        return summary

    def _get_market_context_summary(self) -> Dict[str, Any]:
        """Get market context summary"""
        return {
            'regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'session': self.market_session,
            'market_open': bool(self._is_market_open()),
            'regime_score': float(self.decision_factors.get('regime_score', 0.5)),
            'volatility_score': float(self.decision_factors.get('volatility_score', 0.5)),
            'session_score': float(self.decision_factors.get('session_score', 0.5))
        }

    def _calculate_rolling_stats(self) -> Dict[str, Any]:
        """Calculate rolling statistics"""
        if not self.stats_history:
            return {
                'win_rate': 0.5, 'avg_pnl': 0.0, 'total_pnl': 0.0, 'drawdown': 0.0, 'consensus': 0.5,
                'volatility': 0.02, 'trade_count': 0, 'sharpe': 0.0, 'profit_factor': 1.0
            }
        recent_stats = list(self.stats_history)[-self.window:]
        return {
            'win_rate': float(np.mean([s.get('win_rate', 0.5) for s in recent_stats])),
            'avg_pnl': float(np.mean([s.get('avg_pnl', 0.0) for s in recent_stats])),
            'total_pnl': float(np.sum([s.get('total_pnl', 0.0) for s in recent_stats])),
            'drawdown': float(max([s.get('drawdown', 0.0) for s in recent_stats])),
            'consensus': float(np.mean([s.get('consensus', 0.5) for s in recent_stats])),
            'volatility': float(np.mean([s.get('volatility', 0.02) for s in recent_stats])),
            'trade_count': int(sum([s.get('trade_count', 0) for s in recent_stats])),
            'sharpe': float(np.mean([s.get('sharpe', 0.0) for s in recent_stats])),
            'profit_factor': float(np.mean([s.get('profit_factor', 1.0) for s in recent_stats]))
        }

    def _update_mode_performance_metrics(self):
        """Update mode performance metrics"""
        self.mode_stats['current_mode_duration'] += 1
        self.mode_stats['total_uptime'] += 1
        self.performance_tracker.record_metric('TradingModeManager', 'mode_duration', self.mode_stats['current_mode_duration'])
        self.performance_tracker.record_metric('TradingModeManager', 'mode_effectiveness', self.mode_stats['mode_effectiveness'])
        self.performance_tracker.record_metric('TradingModeManager', 'mode_persistence', self.mode_persistence)

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        try:
            session_hours = (datetime.datetime.now() - datetime.datetime.fromisoformat(self.mode_stats['session_start'])).total_seconds() / 3600.0
        except Exception:
            session_hours = 0.0
        return {
            'module_name': 'TradingModeManager',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': int(self.error_count),
            'circuit_breaker_threshold': int(self.circuit_breaker_threshold),
            'breaker_state': self._breaker_state,
            'current_mode': self.current_mode,
            'auto_mode_enabled': bool(self.auto_mode),
            'mode_persistence': int(self.mode_persistence),
            'total_switches': int(self.mode_stats.get('total_switches', 0)),
            'mode_effectiveness': float(self.mode_stats.get('mode_effectiveness', 0.5)),
            'decision_confidence': float(self.decision_factors.get('performance_score', 0.5)),
            'market_alignment': float(self.decision_factors.get('market_context_score', 0.5)),
            'session_duration_hours': float(session_hours)
        }

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "TradingModeManager")

        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]",
                message="Trading Mode Manager disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))

        processing_time = int((time.time() - start_time) * 1000)
        self.performance_tracker.record_metric('TradingModeManager', 'process_time', processing_time, False)

        payload = {
            'trading_mode': self.current_mode,
            'mode_config': self._get_mode_configuration(),
            'mode_stats': {'error': str(error_context), **self._get_comprehensive_mode_stats()},
            'mode_effectiveness': 0.0,
            'decision_factors': {'error': str(error_context), **self.decision_factors},
            'mode_thresholds': self._safe_thresholds_copy(),
            'market_context': {'error': str(error_context), **self._get_market_context_summary()},
            'mode_recommendations': ["Investigate trading mode manager errors"],
            'health_metrics': {'status': 'error', 'error_context': str(error_context), **self._get_health_metrics()},
            '_thesis': f'TradingModeManager error: {error_context}',
            'trading_mode_manager_initialization': self._get_tmm_init_view()
        }

        # Try to publish minimal info
        try:
            await self._update_smartinfobus_comprehensive(payload, payload['_thesis'])
        except Exception:
            pass

        return payload

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        return {
            'recent_trades': [], 'risk_metrics': {}, 'votes': [], 'positions': [],
            'market_context': {}, 'session_metrics': {}, 'strategy_performance': {},
            'trading_performance': {}, 'market_regime': 'unknown', 'volatility_data': {},
            'economic_calendar': {}
        }

    def _get_safe_performance_defaults(self) -> Dict[str, Any]:
        return {
            'recent_trades': [], 'trade_count': 0, 'win_rate': 0.5, 'total_pnl': 0.0,
            'avg_pnl': 0.0, 'pnl_std': 0.0, 'max_win': 0.0, 'max_loss': 0.0, 'profit_factor': 1.0,
            'recent_trend': 0.0, 'trend_consistency': 0.5, 'current_balance': 10000.0,
            'drawdown': 0.0, 'max_drawdown': 0.0, 'risk_score': 0.5, 'consensus': 0.5,
            'vote_agreement': 0.5, 'vote_count': 0, 'consensus_strength': 0.5,
            'volatility': 0.02, 'volatility_regime_score': 0.5, 'exposure': 0.0,
            'position_count': 0, 'exposure_ratio': 0.0, 'strategy_effectiveness': 0.5,
            'strategy_confidence': 0.5, 'session_pnl': 0.0, 'session_trades': 0, 'sharpe': 0.0
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'trading_mode': self.current_mode,
            'mode_config': {'status': 'disabled'},
            'mode_stats': {'status': 'disabled', **self._get_comprehensive_mode_stats()},
            'mode_effectiveness': 0.0,
            'decision_factors': {'status': 'disabled'},
            'mode_thresholds': self._safe_thresholds_copy(),
            'market_context': {'status': 'disabled', **self._get_market_context_summary()},
            'mode_recommendations': ["Restart trading mode manager system"],
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered', **self._get_health_metrics()},
            '_thesis': 'TradingModeManager disabled due to circuit breaker',
            'trading_mode_manager_initialization': self._get_tmm_init_view()
        }

    def _generate_breaker_response(self) -> Dict[str, Any]:
        """Response when breaker OPEN but module not fully disabled"""
        return {
            'trading_mode': self.current_mode,
            'mode_config': self._get_mode_configuration(),
            'mode_stats': self._get_comprehensive_mode_stats(),
            'mode_effectiveness': float(self.mode_stats.get('mode_effectiveness', 0.5)),
            'decision_factors': self.decision_factors.copy(),
            'mode_thresholds': self._safe_thresholds_copy(),
            'market_context': self._get_market_context_summary(),
            'mode_recommendations': ["Breaker OPEN: skipping decision cycle until cooldown"],
            'health_metrics': {**self._get_health_metrics(), 'breaker_state': self._breaker_state},
            '_thesis': 'Circuit breaker OPEN due to repeated failures; cooldown in effect.',
            'trading_mode_manager_initialization': self._get_tmm_init_view()
        }

    def _get_tmm_init_view(self) -> Dict[str, Any]:
        """Safely read or synthesize initialization view for contract compliance"""
        try:
            init_view = self.smart_bus.get('trading_mode_manager_initialization', 'TradingModeManager')
            if isinstance(init_view, dict) and init_view:
                return init_view
        except Exception:
            pass
        return {
            'status': 'initialized' if not getattr(self, 'is_disabled', False) else 'disabled',
            'timestamp': datetime.datetime.now().isoformat(),
            'current_mode': getattr(self, 'current_mode', 'normal'),
            'auto_mode': getattr(self, 'auto_mode', True)
        }

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS
    # ═══════════════════════════════════════════════════════════════════

    def set_mode(self, mode: str, reason: str = "Manual override") -> None:
        """Set trading mode manually"""
        if mode not in self.TRADING_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {list(self.TRADING_MODES.keys())}")
        old_mode = self.current_mode
        self.current_mode = mode
        self.auto_mode = False
        self.mode_persistence = 0
        self.last_mode_change = datetime.datetime.now().isoformat()
        self.last_change_reason = reason
        self.mode_stats['total_switches'] += 1
        self.mode_stats['manual_switches'] += 1
        self.mode_stats['current_mode_duration'] = 0
        self.mode_history.append({
            'timestamp': self.last_mode_change, 'from_mode': old_mode, 'to_mode': self.current_mode,
            'reason': reason, 'confidence': 1.0, 'auto': False, 'context': {}
        })
        self.logger.info(format_operator_message(
            icon="🎛️",
            message="Manual mode change executed",
            from_mode=old_mode,
            to_mode=self.current_mode,
            reason=reason
        ))

    def set_auto_mode(self, auto: bool) -> None:
        old_auto = self.auto_mode
        self.auto_mode = bool(auto)
        self.logger.info(format_operator_message(
            icon="⚙️",
            message=f"Auto mode {'enabled' if auto else 'disabled'}",
            previous=f"{'enabled' if old_auto else 'disabled'}",
            current_mode=self.current_mode
        ))

    def get_mode(self) -> str:
        return self.current_mode

    def get_mode_stats(self) -> Dict[str, Any]:
        return self._get_comprehensive_mode_stats()

    def get_observation_components(self) -> np.ndarray:
        """Return mode features for RL observation"""
        try:
            mode_encoding = np.zeros(len(self.TRADING_MODES), dtype=np.float32)
            mode_index = list(self.TRADING_MODES.keys()).index(self.current_mode)
            mode_encoding[mode_index] = 1.0
            additional_features = np.array([
                float(self.auto_mode),
                float(self.mode_persistence) / max(self.min_persistence, 1),
                float(self.decision_factors.get('performance_score', 0.5)),
                float(self.decision_factors.get('risk_score', 0.5)),
                float(self.decision_factors.get('consensus_score', 0.5)),
                float(self.decision_factors.get('market_context_score', 0.5)),
                float(self.decision_factors.get('stability_score', 0.5)),
                float(self.mode_stats.get('mode_effectiveness', 0.5)),
                float(self._get_volatility_regime_score()),
                float(self._get_regime_score_enhanced(self.market_regime))
            ], dtype=np.float32)
            observation = np.concatenate([mode_encoding, additional_features])
            if np.any(~np.isfinite(observation)):
                self.logger.error(f"Invalid mode observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            return observation
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "observation_generation")
            self.logger.error(f"Mode observation generation failed: {error_context}")
            default_encoding = np.zeros(len(self.TRADING_MODES), dtype=np.float32)
            default_encoding[1] = 1.0
            default_additional = np.array([1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)
            return np.concatenate([default_encoding, default_additional])

    def get_trading_mode_report(self) -> str:
        """Generate comprehensive trading mode report"""
        mode_emoji = {'safe': '[SAFE]', 'normal': '[BALANCE]', 'aggressive': '[FAST]', 'extreme': '[ROCKET]'}
        current_emoji = mode_emoji.get(self.current_mode, '❓')
        mode_config = self.TRADING_MODES.get(self.current_mode, {})
        mode_description = mode_config.get('description', 'Unknown mode')
        effectiveness = float(self.mode_stats.get('mode_effectiveness', 0.5))
        if effectiveness > 0.8: eff_status = "[OK] Excellent"
        elif effectiveness > 0.65: eff_status = "[FAST] Good"
        elif effectiveness > 0.5: eff_status = "[WARN] Fair"
        elif effectiveness > 0.35: eff_status = "🔶 Poor"
        else: eff_status = "[ALERT] Critical"

        change_lines = []
        for change in list(self.mode_history)[-3:]:
            timestamp = change['timestamp'][:19].replace('T', ' ')
            from_mode = change['from_mode']; to_mode = change['to_mode']
            auto = '[BOT]' if change['auto'] else '👤'
            confidence = change.get('confidence', 0.5)
            change_lines.append(f"  {auto} {timestamp}: {from_mode} → {to_mode} ({confidence:.1%})")

        factor_lines = []
        for factor, value in self.decision_factors.items():
            v = float(value) if isinstance(value, (int, float, np.generic)) else 0.5
            if v > 0.8: emoji = "[GREEN]"
            elif v > 0.6: emoji = "[OK]"
            elif v > 0.5: emoji = "[FAST]"
            elif v > 0.3: emoji = "[WARN]"
            else: emoji = "[ALERT]"
            factor_name = factor.replace('_', ' ').title()
            factor_lines.append(f"  {emoji} {factor_name}: {v:.1%}")

        rolling_stats = self._calculate_rolling_stats()
        analytics_summary = self._get_mode_analytics_summary()
        mode_performance_lines = []
        for mode, stats in analytics_summary.items():
            emoji = mode_emoji.get(mode, '❓')
            win_rate = float(stats.get('avg_win_rate', 0.5))
            avg_pnl = float(stats.get('avg_pnl', 0.0))
            periods = int(stats.get('total_periods', 0))
            mode_performance_lines.append(f"  {emoji} {mode.title()}: {win_rate:.1%} WR, €{avg_pnl:+.1f} avg, {periods} periods")

        market_open_status = '[GREEN] Open' if self._is_market_open() else '[RED] Closed'
        regime_score = float(self.decision_factors.get('regime_score', 0.5))
        volatility_score = float(self.decision_factors.get('volatility_score', 0.5))

        current_thresholds = self.mode_thresholds.get(self.current_mode, {})
        threshold_lines = []
        for key, value in current_thresholds.items():
            if key == 'max_drawdown':
                threshold_lines.append(f"  📉 Max Drawdown: {float(value):.1%}")
            elif key == 'min_win_rate':
                threshold_lines.append(f"  [TARGET] Min Win Rate: {float(value):.1%}")
            elif key == 'min_consensus':
                threshold_lines.append(f"  🤝 Min Consensus: {float(value):.1%}")
            elif key == 'max_exposure':
                threshold_lines.append(f"  [STATS] Max Exposure: {float(value):.1%}")

        return f"""
⚙️ TRADING MODE MANAGER v3.1
═══════════════════════════════════════════════════════════════
{current_emoji} Current Mode: {self.current_mode.upper()} - {mode_description}
[TARGET] Mode Effectiveness: {eff_status} ({effectiveness:.1%})
[BOT] Auto Mode: {'[OK] Enabled' if self.auto_mode else '[FAIL] Disabled'}
[TIME] Mode Persistence: {self.mode_persistence}/{self.min_persistence} periods
[RELOAD] Total Switches: {self.mode_stats.get('total_switches', 0)} (Auto: {self.mode_stats.get('auto_switches', 0)}, Manual: {self.mode_stats.get('manual_switches', 0)})

[STATS] CURRENT MODE CONFIGURATION
• Risk Multiplier: {mode_config.get('risk_multiplier', 1.0):.1f}x
• Max Exposure: {mode_config.get('max_exposure', 0.5):.1%}
• Win Rate Threshold: {mode_config.get('win_rate_threshold', 0.5):.1%}
• Drawdown Limit: {mode_config.get('drawdown_limit', 0.1):.1%}
• Consensus Requirement: {mode_config.get('consensus_requirement', 0.3):.1%}
• Volatility Tolerance: {mode_config.get('volatility_tolerance', 'medium').title()}

🎛️ DECISION WEIGHTS & SENSITIVITY
• Performance Weight: {self.performance_weight:.1%}
• Risk Weight: {self.risk_weight:.1%}
• Consensus Weight: {self.consensus_weight:.1%}
• Market Context Weight: {self.market_context_weight:.1%}
• Context Sensitivity: {self.context_sensitivity:.1%}

[CHART] ROLLING STATISTICS (Last {self.window} periods)
• Win Rate: {rolling_stats['win_rate']:.1%}
• Average PnL: €{rolling_stats['avg_pnl']:+.2f}
• Total PnL: €{rolling_stats['total_pnl']:+.2f}
• Max Drawdown: {rolling_stats['drawdown']:.1%}
• Consensus Strength: {rolling_stats['consensus']:.1%}
• Market Volatility: {rolling_stats['volatility']:.2%}
• Trade Count: {rolling_stats['trade_count']}
• Sharpe Ratio: {rolling_stats['sharpe']:.2f}
• Profit Factor: {rolling_stats['profit_factor']:.2f}

[TARGET] DECISION FACTORS (Current Analysis)
{chr(10).join(factor_lines) if factor_lines else "  📭 No decision factors available"}

[STATS] MARKET CONTEXT & REGIME ANALYSIS
• Market Regime: {self.market_regime.title()} (Score: {regime_score:.1%})
• Volatility Level: {self.volatility_regime.title()} (Score: {volatility_score:.1%})
• Trading Session: {self.market_session.title()}
• Market Status: {market_open_status}
• Regime Awareness: {'[OK] Enabled' if self.regime_awareness else '[FAIL] Disabled'}
• Session Awareness: {'[OK] Enabled' if self.session_awareness else '[FAIL] Disabled'}
• Volatility Scaling: {'[OK] Enabled' if self.volatility_scaling else '[FAIL] Disabled'}

[TARGET] CURRENT MODE THRESHOLDS
{chr(10).join(threshold_lines) if threshold_lines else "  📭 No thresholds configured"}

[STATS] MODE PERFORMANCE ANALYTICS
{chr(10).join(mode_performance_lines) if mode_performance_lines else "  📭 No mode performance data available"}

📜 RECENT MODE CHANGES
{chr(10).join(change_lines) if change_lines else "  📭 No recent mode changes"}

🧠 INTELLIGENCE & LEARNING
• Adaptation Speed: {self.mode_intelligence.get('adaptation_speed', 0.1):.1%}
• Confidence Threshold: {self.mode_intelligence.get('confidence_threshold', 0.7):.1%}
• Stability Requirement: {self.mode_intelligence.get('stability_requirement', 0.8):.1%}
• Performance Memory: {self.mode_intelligence.get('performance_memory', 0.9):.1%}
• Learning Records: {len(self.learning_history)}
• Threshold Adaptations: {len(self.threshold_adaptations)}

[TARGET] LAST DECISION CONTEXT
• Change Reason: {self.last_change_reason or 'No recent changes'}
• Last Change Time: {self.last_mode_change[:19].replace('T', ' ') if self.last_mode_change else 'Never'}
• Current Duration: {self.mode_stats.get('current_mode_duration', 0)} periods
• System Uptime: {self.mode_stats.get('total_uptime', 0)} periods

[TOOL] HEALTH & STATUS
• Errors: {self.error_count}/{self.circuit_breaker_threshold} | Breaker: {self._breaker_state}
• Status: {'[ALERT] DISABLED' if self.is_disabled else '[OK] OPERATIONAL'}
• Session Duration: {self._get_health_metrics().get('session_duration_hours', 0.0):.1f} hours
        """
