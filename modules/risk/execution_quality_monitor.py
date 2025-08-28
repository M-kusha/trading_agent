# ─────────────────────────────────────────────────────────────
# File: modules/risk/execution_quality_monitor.py
# [ROCKET] PRODUCTION-READY Enhanced Execution Quality Monitor
# Advanced execution monitoring with SmartInfoBus integration and intelligent training mode
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


class ExecutionMode(Enum):
    """Execution quality monitoring modes"""
    TRAINING = "training"
    CALIBRATION = "calibration"
    NORMAL = "normal"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class ExecutionQualityConfig:
    """Configuration for Execution Quality Monitor"""
    slip_limit: float = 0.002
    latency_limit: int = 1000
    min_fill_rate: float = 0.95
    stats_window: int = 50
    slippage_percentile: int = 95
    latency_percentile: int = 95
    spread_threshold: float = 0.01
    execution_timeout: int = 5000
    quality_threshold: float = 0.7
    degradation_threshold: float = 0.5

    # Performance thresholds
    max_processing_time_ms: float = 150
    circuit_breaker_threshold: int = 5
    min_execution_quality: float = 0.3

    # Adaptation parameters
    adaptive_learning_rate: float = 0.02
    quality_sensitivity: float = 1.0


@module(
    name="ExecutionQualityMonitor",
    version="4.0.1",
    category="risk",
    provides=["execution_quality", "execution_analytics", "quality_metrics", "execution_alerts"],
    requires=["execution_data", "trade_data", "order_data", "market_data"],
    description="Advanced execution quality monitoring with intelligent context-aware analysis and training mode",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    is_voting_member=True
)
class ExecutionQualityMonitor(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    [ROCKET] Advanced execution quality monitor with SmartInfoBus integration.
    Monitors execution metrics including slippage, latency, fill rates, and spreads
    with intelligent context-aware analysis.
    """

    # NOTE ABOUT CONFIG:
    # - self._cfg is the typed ExecutionQualityConfig used throughout this class.
    # - self.config is kept as a dict for BaseModule/infra compatibility.

    def __init__(
        self,
        config: Optional[Union[ExecutionQualityConfig, Dict[str, Any]]] = None,
        training_mode: bool = True,
        **kwargs: Any
    ):
        # Build the typed config (_cfg)
        if isinstance(config, ExecutionQualityConfig):
            self._cfg = config
        elif isinstance(config, dict):
            self._cfg = ExecutionQualityConfig(**config)
        else:
            self._cfg = ExecutionQualityConfig()

        self.training_mode = training_mode

        # Initialize BaseModule (it expects self.config to be a dict)
        super().__init__()

        # Mirror dataclass to dict for BaseModule / ecosystem compatibility
        self.config = dict(self._cfg.__dict__)  # type: ignore[assignment]

        # Initialize advanced systems & state
        self._initialize_advanced_systems()
        self._initialize_execution_state()

        self.logger.info(
            format_operator_message(
                "[FAST]", "EXECUTION_MONITOR_READY",
                training_mode=self.training_mode,
                stats_window=self._cfg.stats_window,
                config_loaded=True
            )
        )

    # ─────────────────────────────────────────────────────────────
    # INIT HELPERS
    # ─────────────────────────────────────────────────────────────

    def _initialize_advanced_systems(self):
        """Initialize advanced systems for execution monitoring"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="ExecutionQualityMonitor",
            log_path="logs/risk/execution_quality_monitor.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("ExecutionQualityMonitor", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

        # Circuit breaker for execution operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self._cfg.circuit_breaker_threshold
        }

        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        # Do not start monitoring yet; wait until execution state is initialized to avoid races

    def _initialize_execution_state(self):
        """Initialize execution quality state"""
        # Initialize mixin states
        self._initialize_risk_state()
        self._initialize_trading_state()
        self._initialize_state_management()

        # Current operational mode
        self.current_mode = ExecutionMode.TRAINING if self.training_mode else ExecutionMode.NORMAL
        self.mode_start_time = datetime.datetime.now()

        # Enhanced histories
        self.slippage_history = deque(maxlen=self._cfg.stats_window)
        self.latency_history = deque(maxlen=self._cfg.stats_window)
        self.fill_history = deque(maxlen=self._cfg.stats_window)
        self.spread_history = deque(maxlen=self._cfg.stats_window)
        self.quality_history = deque(maxlen=self._cfg.stats_window)

        # Instrument-specific tracking
        self.instrument_metrics: Dict[str, Dict[str, deque]] = defaultdict(
            lambda: {
                'slippage': deque(maxlen=20),
                'latency': deque(maxlen=20),
                'fill_rate': deque(maxlen=20),
                'spread': deque(maxlen=20)
            }
        )

        # Current quality metrics
        self.quality_score = 1.0
        self.execution_count = 0
        self.degraded_executions = 0

        # Market context awareness
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"

        # Issue tracking with enhanced categorization
        self.issues: Dict[str, List[Dict[str, Any]]] = {
            "slippage": [],
            "latency": [],
            "fill_rate": [],
            "spread": [],
            "timeout": [],
            "partial_fill": []
        }

        # Performance analytics
        self.execution_analytics = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self.session_performance = defaultdict(lambda: defaultdict(list))

        # Comprehensive metrics
        self.comprehensive_metrics = {
            "avg_slippage": 0.0,
            "avg_latency": 0.0,
            "avg_fill_rate": 1.0,
            "avg_spread": 0.0,
            "total_executions": 0,
            "success_rate": 1.0,
            "degradation_rate": 0.0,
            "quality_trend": 0.0
        }

        # Broker/venue performance tracking
        self.venue_performance: Dict[str, Dict[str, Any]] = {}

        # Alert thresholds and escalation
        self.quality_alerts = deque(maxlen=10)
        self.escalation_count = 0
        self.last_escalation: Optional[datetime.datetime] = None

        # Adaptive parameters
        self._adaptive_params = {
            'dynamic_threshold_scaling': 1.0,
            'context_sensitivity': 1.0,
            'quality_adaptation_confidence': 0.5
        }
        # Now that all attributes exist, start monitoring loop safely
        self._start_monitoring()

    def _start_monitoring(self):
        """Start background monitoring for execution quality"""

        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_execution_health()
                    self._analyze_execution_effectiveness()
                    self._adapt_quality_parameters()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Execution monitoring error: {e}")

        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    def _initialize(self) -> None:
        """Initialize module with SmartInfoBus integration"""
        try:
            # Set initial execution quality status
            initial_status = {
                "current_mode": self.current_mode.value,
                "quality_score": self.quality_score,
                "training_mode": self.training_mode,
                "execution_count": self.execution_count
            }

            self.smart_bus.set(
                'execution_quality',
                initial_status,
                module='ExecutionQualityMonitor',
                thesis="Initial execution quality monitor status"
            )

        except Exception as e:
            self.logger.error(f"Execution monitor initialization failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # MAIN PROCESS
    # ─────────────────────────────────────────────────────────────

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process execution quality assessment with enhanced analytics"""
        start_time = time.time()

        try:
            # Extract execution data from SmartInfoBus
            execution_data = await self._extract_execution_data(**inputs)

            if not execution_data:
                return await self._handle_no_data_fallback()

            # Update market context
            context_result = await self._update_market_context_async(execution_data)

            # Process executions comprehensively
            processing_result = await self._process_executions_comprehensive(execution_data)

            # Generate training data if needed
            training_result = {}
            if self.training_mode and self._should_generate_training_data():
                training_result = await self._generate_realistic_training_data(execution_data)

            # Analyze quality trends
            trends_result = await self._analyze_execution_quality_trends()

            # Check for degradation and alerts
            alerts_result = await self._check_quality_degradation_and_alerts()

            # Update comprehensive metrics
            metrics_result = await self._update_comprehensive_metrics()

            # Update operational mode
            mode_result = await self._update_operational_mode()

            # Combine results
            result = {
                **context_result,
                **processing_result,
                **training_result,
                **trends_result,
                **alerts_result,
                **metrics_result,
                **mode_result
            }

            # Generate thesis
            thesis = await self._generate_execution_thesis(execution_data, result)

            # Build provided outputs for return payload
            execution_quality_data = {
                'current_mode': self.current_mode.value,
                'quality_score': self.quality_score,
                'training_mode': self.training_mode,
                'execution_count': self.execution_count,
                'degraded_executions': self.degraded_executions,
                'timestamp': datetime.datetime.now().isoformat()
            }

            analytics_data = {
                'comprehensive_metrics': self.comprehensive_metrics.copy(),
                'issues': {k: len(v) for k, v in self.issues.items() if v},
                'quality_alerts': len(self.quality_alerts),
                'escalation_count': self.escalation_count,
                'regime_performance': {
                    regime: {
                        'quality_scores': len(data['quality_scores']),
                        'avg_quality': float(np.mean(data['quality_scores'][-10:])) if data['quality_scores'] else 0.0
                    }
                    for regime, data in self.regime_performance.items()
                }
            }

            metrics_data = {
                'quality_score': self.quality_score,
                'avg_slippage': self.comprehensive_metrics["avg_slippage"],
                'avg_latency': self.comprehensive_metrics["avg_latency"],
                'avg_fill_rate': self.comprehensive_metrics["avg_fill_rate"],
                'avg_spread': self.comprehensive_metrics["avg_spread"],
                'success_rate': self.comprehensive_metrics["success_rate"],
                'degradation_rate': self.comprehensive_metrics["degradation_rate"]
            }

            alerts_data = {
                'quality_alerts': len(self.quality_alerts),
                'escalation_count': self.escalation_count,
                'critical_issues': sum(1 for issues in self.issues.values() for _ in issues),
                'last_escalation': self.last_escalation.isoformat() if self.last_escalation else None,
                'current_issues': {k: len(v) for k, v in self.issues.items() if v}
            }

            # Update SmartInfoBus
            await self._update_execution_smart_bus(result, thesis)

            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)

            # Ensure returned payload complies with provides contract
            result.update({
                'execution_quality': execution_quality_data,
                'execution_analytics': analytics_data,
                'quality_metrics': metrics_data,
                'execution_alerts': alerts_data,
                '_thesis': thesis,
                'success': True
            })

            return result

        except Exception as e:
            return await self._handle_execution_error(e, start_time)

    # ─────────────────────────────────────────────────────────────
    # DATA EXTRACTION & CONVERSION
    # ─────────────────────────────────────────────────────────────

    async def _extract_execution_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract comprehensive execution data from SmartInfoBus"""
        try:
            # Get execution data from SmartInfoBus
            execution_data = self.smart_bus.get('execution_data', 'ExecutionQualityMonitor') or {}
            executions = execution_data.get('executions', [])

            # Get trade data
            trade_data = self.smart_bus.get('trade_data', 'ExecutionQualityMonitor') or {}
            recent_trades = trade_data.get('recent_trades', [])

            # Get order data
            order_data = self.smart_bus.get('order_data', 'ExecutionQualityMonitor') or {}
            orders = order_data.get('orders', [])

            # Get market data
            market_data = self.smart_bus.get('market_data', 'ExecutionQualityMonitor') or {}
            spreads = market_data.get('spreads', {})

            # Get direct inputs
            trade_executions = inputs.get('trade_executions', inputs.get('trades', recent_trades))
            order_attempts = inputs.get('order_attempts', inputs.get('orders', orders))
            spread_data = inputs.get('spread_data', spreads)

            # Convert trades to executions if needed
            converted_executions = []
            for trade in trade_executions:
                execution = self._convert_trade_to_execution(trade)
                if execution:
                    converted_executions.append(execution)

            return {
                'executions': executions + converted_executions,
                'orders': order_attempts,
                'spreads': spread_data,
                'market_data': market_data,
                'timestamp': datetime.datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to extract execution data: {e}")
            return None

    def _convert_trade_to_execution(self, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert trade data to execution format"""
        try:
            execution = {
                'instrument': trade.get('symbol', trade.get('instrument', 'UNKNOWN')),
                'size': abs(trade.get('size', trade.get('volume', 0))),
                'side': trade.get('side', 'BUY' if trade.get('size', 0) > 0 else 'SELL'),
                'timestamp': trade.get('timestamp', datetime.datetime.now().isoformat()),
                'execution_price': float(trade.get('price', trade.get('fill_price', 0)) or 0.0),
            }

            # Extract execution quality metrics
            sl = self._extract_slippage(trade)
            if sl is not None:
                execution['slippage'] = float(sl)

            lat = self._extract_latency(trade)
            if lat is not None:
                execution['latency_ms'] = float(lat)

            sp = self._extract_spread(trade)
            if sp is not None:
                execution['spread'] = float(sp)

            execution['fill_status'] = self._extract_fill_status(trade)

            # Add venue information if available
            execution['venue'] = trade.get('broker', trade.get('venue', 'unknown'))

            return execution

        except Exception as e:
            self.logger.warning(f"Trade conversion failed: {e}")
            return None

    def _extract_slippage(self, trade: Dict[str, Any]) -> Optional[float]:
        """Extract slippage from trade data with multiple fallback methods"""
        # Method 1: Direct slippage field
        for field in ["slippage", "slip", "price_diff", "execution_slippage"]:
            if field in trade and trade[field] is not None:
                try:
                    return float(trade[field])
                except Exception:
                    pass

        # Method 2: Calculate from expected vs actual price
        expected_price = trade.get("expected_price", trade.get("order_price"))
        actual_price = trade.get("actual_price", trade.get("fill_price", trade.get("price")))
        try:
            if expected_price is not None and actual_price is not None:
                expected_price = float(expected_price)
                actual_price = float(actual_price)
                if expected_price > 0:
                    return abs((actual_price - expected_price) / expected_price)
        except Exception:
            pass

        # Method 3: Estimate from market impact
        size = trade.get('size', trade.get('volume', 0))
        try:
            if size and abs(float(size)) > 0.1:  # Significant size
                return abs(float(size)) * 0.0001  # 1 pip per lot estimation
        except Exception:
            pass

        return None

    def _extract_latency(self, trade: Dict[str, Any]) -> Optional[float]:
        """Extract latency from trade data with multiple methods"""
        # Method 1: Direct latency fields
        for field in ["latency_ms", "latency", "execution_time_ms", "fill_time_ms"]:
            if field in trade and trade[field] is not None:
                try:
                    return float(trade[field])
                except Exception:
                    pass

        # Method 2: Calculate from timestamps
        order_time = trade.get("order_time", trade.get("submit_time"))
        fill_time = trade.get("fill_time", trade.get("execution_time"))

        if order_time is not None and fill_time is not None:
            try:
                if isinstance(order_time, str):
                    order_time = datetime.datetime.fromisoformat(order_time.replace('Z', '+00:00'))
                if isinstance(fill_time, str):
                    fill_time = datetime.datetime.fromisoformat(fill_time.replace('Z', '+00:00'))

                if isinstance(order_time, datetime.datetime) and isinstance(fill_time, datetime.datetime):
                    latency_seconds = (fill_time - order_time).total_seconds()
                    return max(0.0, latency_seconds * 1000.0)  # Convert to milliseconds
            except Exception:
                pass

        return None

    def _extract_spread(self, trade: Dict[str, Any]) -> Optional[float]:
        """Extract spread from trade data"""
        # Method 1: Direct spread fields
        for field in ["spread", "bid_ask_spread", "market_spread"]:
            if field in trade and trade[field] is not None:
                try:
                    return float(trade[field])
                except Exception:
                    pass

        # Method 2: Calculate from bid/ask
        bid = trade.get("bid_price", trade.get("bid"))
        ask = trade.get("ask_price", trade.get("ask"))

        try:
            if bid is not None and ask is not None:
                bid, ask = float(bid), float(ask)
                if bid > 0 and ask > bid:
                    return ask - bid
        except Exception:
            pass

        return None

    def _extract_fill_status(self, trade: Dict[str, Any]) -> str:
        """Extract fill status from trade data"""
        # Check various status indicators
        status_fields = ["status", "state", "fill_status", "order_status"]
        for field in status_fields:
            status = str(trade.get(field, "")).lower()
            if status in ["filled", "completed", "executed"]:
                return "filled"
            elif status in ["partial", "partially_filled"]:
                return "partial"
            elif status in ["rejected", "cancelled", "failed"]:
                return "failed"

        # Check fill quantities
        try:
            order_qty = float(trade.get("quantity", trade.get("size", trade.get("volume", 0))) or 0.0)
            filled_qty = float(trade.get("filled_quantity", trade.get("filled_size", trade.get("executed_quantity", order_qty))) or 0.0)
            if order_qty > 0 and filled_qty >= 0:
                fill_ratio = filled_qty / order_qty if order_qty > 0 else 1.0
                if fill_ratio >= 0.99:
                    return "filled"
                elif fill_ratio >= 0.01:
                    return "partial"
        except Exception:
            pass

        return "unknown"

    # ─────────────────────────────────────────────────────────────
    # ANALYSIS
    # ─────────────────────────────────────────────────────────────

    async def _update_market_context_async(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market context awareness asynchronously"""
        try:
            # Extract market context from SmartInfoBus
            market_context = self.smart_bus.get('market_context', 'ExecutionQualityMonitor') or {}

            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = market_context.get('regime', self.market_regime)
            self.volatility_regime = market_context.get('volatility_level', self.volatility_regime)
            self.market_session = market_context.get('session', self.market_session)

            # Log regime changes
            if self.market_regime != old_regime:
                self.logger.info(
                    format_operator_message(
                        "[STATS]", "MARKET_REGIME_CHANGE",
                        old_regime=old_regime,
                        new_regime=self.market_regime,
                        volatility=self.volatility_regime,
                        session=self.market_session,
                        context="market_context"
                    )
                )

            return {
                'market_context_updated': True,
                'current_regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'market_session': self.market_session
            }

        except Exception as e:
            self.logger.error(f"Market context update failed: {e}")
            return {'market_context_updated': False, 'error': str(e)}

    async def _process_executions_comprehensive(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process executions with comprehensive analysis"""
        try:
            executions = execution_data.get('executions', [])
            orders = execution_data.get('orders', [])
            spreads = execution_data.get('spreads', {})

            # Clear previous issues
            for issue_type in self.issues:
                self.issues[issue_type].clear()

            execution_count = 0
            processing_results = []

            # Process individual executions
            for execution in executions:
                try:
                    result = await self._analyze_single_execution_async(execution)
                    processing_results.append(result)
                    execution_count += 1
                except Exception as e:
                    self.logger.warning(f"Execution analysis failed: {e}")

            # Process fill rates from orders
            fill_rate_result = {}
            if orders:
                fill_rate_result = await self._analyze_fill_rates_async(orders)

            # Process spread data
            spread_result = {}
            if spreads:
                spread_result = await self._analyze_spread_data_async(spreads)

            # Update execution count
            self.execution_count += execution_count
            self.comprehensive_metrics["total_executions"] = self.execution_count

            return {
                'executions_processed': True,
                'execution_count': execution_count,
                'total_executions': self.execution_count,
                'issues_detected': {k: len(v) for k, v in self.issues.items() if v},
                **fill_rate_result,
                **spread_result
            }

        except Exception as e:
            self.logger.error(f"Execution processing failed: {e}")
            return {'executions_processed': False, 'error': str(e)}

    async def _analyze_single_execution_async(self, execution: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze individual execution with context awareness"""
        try:
            instrument = execution.get('instrument', 'UNKNOWN')
            analysis_result: Dict[str, Any] = {}

            # Analyze slippage
            slippage = execution.get('slippage')
            if slippage is not None:
                slippage = abs(float(slippage))
                self.slippage_history.append(slippage)
                self.instrument_metrics[instrument]['slippage'].append(slippage)

                # Context-aware slippage limits
                adjusted_limit = self._get_context_adjusted_slip_limit()

                if slippage > adjusted_limit:
                    issue = {
                        'instrument': instrument,
                        'value': slippage,
                        'limit': adjusted_limit,
                        'context': {
                            'regime': self.market_regime,
                            'volatility': self.volatility_regime,
                            'session': self.market_session
                        },
                        'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                    }
                    self.issues["slippage"].append(issue)
                    analysis_result['slippage_violation'] = True

                    self.logger.warning(
                        format_operator_message(
                            "[WARN]", "HIGH_SLIPPAGE",
                            instrument=instrument,
                            slippage=f"{slippage:.5f}",
                            limit=f"{adjusted_limit:.5f}",
                            context="execution_analysis"
                        )
                    )

            # Analyze latency
            latency = execution.get('latency_ms')
            if latency is not None:
                latency = float(latency)
                self.latency_history.append(latency)
                self.instrument_metrics[instrument]['latency'].append(latency)

                # Context-aware latency limits
                adjusted_limit = self._get_context_adjusted_latency_limit()

                if latency > adjusted_limit:
                    issue = {
                        'instrument': instrument,
                        'value': latency,
                        'limit': adjusted_limit,
                        'context': {
                            'regime': self.market_regime,
                            'volatility': self.volatility_regime,
                            'session': self.market_session
                        },
                        'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                    }
                    self.issues["latency"].append(issue)
                    analysis_result['latency_violation'] = True

                    self.logger.warning(
                        format_operator_message(
                            "[TIME]", "HIGH_LATENCY",
                            instrument=instrument,
                            latency=f"{latency:.0f}ms",
                            limit=f"{adjusted_limit:.0f}ms",
                            context="execution_analysis"
                        )
                    )

            # Analyze spread
            spread = execution.get('spread')
            if spread is not None:
                spread = float(spread)
                self.spread_history.append(spread)
                self.instrument_metrics[instrument]['spread'].append(spread)

                threshold = self._get_context_adjusted_spread_threshold(instrument)

                if spread > threshold:
                    issue = {
                        'instrument': instrument,
                        'value': spread,
                        'threshold': threshold,
                        'context': {
                            'regime': self.market_regime,
                            'volatility': self.volatility_regime,
                            'session': self.market_session
                        },
                        'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                    }
                    self.issues["spread"].append(issue)
                    analysis_result['spread_violation'] = True

            # Analyze fill status
            fill_status = execution.get('fill_status', 'unknown')
            if fill_status == 'partial':
                issue = {
                    'instrument': instrument,
                    'status': fill_status,
                    'context': {
                        'regime': self.market_regime,
                        'volatility': self.volatility_regime,
                        'session': self.market_session
                    },
                    'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                }
                self.issues["partial_fill"].append(issue)
                analysis_result['partial_fill'] = True
            elif fill_status == 'failed':
                issue = {
                    'instrument': instrument,
                    'status': fill_status,
                    'context': {
                        'regime': self.market_regime,
                        'volatility': self.volatility_regime,
                        'session': self.market_session
                    },
                    'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                }
                self.issues["fill_rate"].append(issue)
                analysis_result['fill_failure'] = True

            return analysis_result

        except Exception as e:
            self.logger.warning(f"Single execution analysis failed: {e}")
            return {}

    # ─────────────────────────────────────────────────────────────
    # CONTEXT-AWARE LIMITS
    # ─────────────────────────────────────────────────────────────

    def _get_context_adjusted_slip_limit(self) -> float:
        """Get context-adjusted slippage limit"""
        base_limit = self._cfg.slip_limit

        # Adjust for volatility
        if self.volatility_regime == 'high':
            base_limit *= 2.0
        elif self.volatility_regime == 'extreme':
            base_limit *= 3.0
        elif self.volatility_regime == 'low':
            base_limit *= 0.7

        # Adjust for market regime
        if self.market_regime == 'volatile':
            base_limit *= 1.5
        elif self.market_regime == 'trending':
            base_limit *= 0.8

        # Adjust for session
        if self.market_session in ['asian', 'rollover']:
            base_limit *= 1.3  # Less liquidity

        return float(base_limit * self._adaptive_params['dynamic_threshold_scaling'])

    def _get_context_adjusted_latency_limit(self) -> float:
        """Get context-adjusted latency limit"""
        base_limit = float(self._cfg.latency_limit)

        # Adjust for session
        if self.market_session == 'asian':
            base_limit *= 1.5  # Higher latency expected
        elif self.market_session == 'rollover':
            base_limit *= 1.3

        # Adjust for volatility
        if self.volatility_regime in ['high', 'extreme']:
            base_limit *= 1.4  # Higher latency during volatility

        return float(base_limit * self._adaptive_params['dynamic_threshold_scaling'])

    def _get_context_adjusted_spread_threshold(self, instrument: str) -> float:
        """Get context-adjusted spread threshold"""
        # Base thresholds by instrument type
        inst_upper = instrument.upper()
        if 'XAU' in inst_upper or 'GOLD' in inst_upper:
            base_threshold = 1.0  # $1 for gold
        elif any(curr in inst_upper for curr in ['EUR', 'USD', 'GBP', 'JPY']):
            base_threshold = 0.0003  # 3 pips for major pairs
        else:
            base_threshold = self._cfg.spread_threshold

        # Adjust for volatility
        if self.volatility_regime == 'high':
            base_threshold *= 2.0
        elif self.volatility_regime == 'extreme':
            base_threshold *= 3.0

        # Adjust for session
        if self.market_session in ['asian', 'rollover']:
            base_threshold *= 1.5

        return float(base_threshold)

    # ─────────────────────────────────────────────────────────────
    # AGGREGATIONS
    # ─────────────────────────────────────────────────────────────

    async def _analyze_fill_rates_async(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze fill rates from order data asynchronously"""
        try:
            if not orders:
                return {'fill_rates_analyzed': False}

            filled_equiv = 0.0
            total_orders = len(orders)

            for order in orders:
                fill_status = self._extract_fill_status(order)
                if fill_status == 'filled':
                    filled_equiv += 1.0
                elif fill_status == 'partial':
                    filled_equiv += 0.5  # Count partial fills as half

            fill_rate = filled_equiv / total_orders if total_orders > 0 else 1.0
            self.fill_history.append(fill_rate)

            # Context-aware fill rate expectations
            expected_fill_rate = self._get_context_adjusted_fill_rate()

            if fill_rate < expected_fill_rate:
                issue = {
                    'fill_rate': fill_rate,
                    'expected': expected_fill_rate,
                    'filled_equiv': filled_equiv,
                    'total_orders': total_orders,
                    'context': {
                        'regime': self.market_regime,
                        'volatility': self.volatility_regime,
                        'session': self.market_session
                    },
                    'timestamp': datetime.datetime.now().isoformat()
                }
                self.issues["fill_rate"].append(issue)

                self.logger.warning(
                    format_operator_message(
                        "📉", "LOW_FILL_RATE",
                        fill_rate=f"{fill_rate:.1%}",
                        expected=f"{expected_fill_rate:.1%}",
                        orders=f"{filled_equiv:.1f}/{total_orders}",
                        context="execution_analysis"
                    )
                )

            return {
                'fill_rates_analyzed': True,
                'fill_rate': fill_rate,
                'expected_fill_rate': expected_fill_rate,
                'total_orders': total_orders,
                'filled_orders_equivalent': filled_equiv
            }

        except Exception as e:
            self.logger.warning(f"Fill rate analysis failed: {e}")
            return {'fill_rates_analyzed': False, 'error': str(e)}

    def _get_context_adjusted_fill_rate(self) -> float:
        """Get context-adjusted expected fill rate"""
        base_rate = self._cfg.min_fill_rate

        # Adjust for volatility
        if self.volatility_regime == 'extreme':
            base_rate -= 0.05  # Lower expectations during extreme volatility
        elif self.volatility_regime == 'high':
            base_rate -= 0.02

        # Adjust for session
        if self.market_session in ['asian', 'rollover']:
            base_rate -= 0.03  # Lower liquidity sessions

        return float(max(0.8, base_rate))  # Never go below 80%

    async def _analyze_spread_data_async(self, spreads: Dict[str, float]) -> Dict[str, Any]:
        """Analyze spread data with context awareness asynchronously"""
        try:
            spreads_analyzed = 0
            violations = []

            for instrument, spread in spreads.items():
                if spread is None:
                    continue
                try:
                    spread_val = float(spread)
                except Exception:
                    continue

                if spread_val <= 0:
                    continue

                self.spread_history.append(spread_val)
                self.instrument_metrics[instrument]['spread'].append(spread_val)
                spreads_analyzed += 1

                # Context-aware spread thresholds
                threshold = self._get_context_adjusted_spread_threshold(instrument)

                if spread_val > threshold:
                    violations.append(instrument)
                    issue = {
                        'instrument': instrument,
                        'spread': spread_val,
                        'threshold': threshold,
                        'context': {
                            'regime': self.market_regime,
                            'volatility': self.volatility_regime,
                            'session': self.market_session
                        },
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    self.issues["spread"].append(issue)

            return {
                'spreads_analyzed': True,
                'spreads_count': spreads_analyzed,
                'spread_violations': len(violations),
                'violation_instruments': violations
            }

        except Exception as e:
            self.logger.warning(f"Spread analysis failed: {e}")
            return {'spreads_analyzed': False, 'error': str(e)}

    # ─────────────────────────────────────────────────────────────
    # TRAINING DATA (OPTIONAL)
    # ─────────────────────────────────────────────────────────────

    def _should_generate_training_data(self) -> bool:
        """Determine if we should generate training data"""
        return (
            self.training_mode and
            len(self.slippage_history) < 10 and
            self.execution_count < 5
        )

    async def _generate_realistic_training_data(self, execution_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate realistic execution data for training"""
        try:
            # Generate realistic slippage
            base_slippage = 0.0002

            # Adjust for context
            if self.volatility_regime == 'high':
                base_slippage *= 2.0
            elif self.volatility_regime == 'extreme':
                base_slippage *= 3.0

            if self.market_session in ['asian', 'rollover']:
                base_slippage *= 1.3

            # Generate with realistic distribution
            realistic_slippage = abs(float(np.random.gamma(2, base_slippage)))
            self.slippage_history.append(realistic_slippage)

            # Generate realistic latency
            base_latency = 250.0  # 250ms base

            if self.market_session == 'asian':
                base_latency += 100
            if self.volatility_regime in ['high', 'extreme']:
                base_latency += 100

            realistic_latency = max(50.0, float(np.random.gamma(3, base_latency / 3)))
            self.latency_history.append(realistic_latency)

            # Generate realistic fill rate
            base_fill_rate = 0.96

            if self.volatility_regime == 'extreme':
                base_fill_rate -= 0.05
            elif self.volatility_regime == 'high':
                base_fill_rate -= 0.02

            realistic_fill_rate = float(np.random.beta(20 * base_fill_rate, 20 * (1 - base_fill_rate)))
            self.fill_history.append(realistic_fill_rate)

            # Generate realistic spread
            base_spread = 0.00015  # 1.5 pips

            if self.volatility_regime == 'high':
                base_spread *= 2.0
            elif self.volatility_regime == 'extreme':
                base_spread *= 3.0

            realistic_spread = abs(float(np.random.gamma(2, base_spread)))
            self.spread_history.append(realistic_spread)

            return {
                'training_data_generated': True,
                'synthetic_slippage': realistic_slippage,
                'synthetic_latency': realistic_latency,
                'synthetic_fill_rate': realistic_fill_rate,
                'synthetic_spread': realistic_spread
            }

        except Exception as e:
            self.logger.warning(f"Training data generation failed: {e}")
            return {'training_data_generated': False, 'error': str(e)}

    # ─────────────────────────────────────────────────────────────
    # TRENDS & METRICS
    # ─────────────────────────────────────────────────────────────

    async def _analyze_execution_quality_trends(self) -> Dict[str, Any]:
        """Analyze execution quality trends"""
        try:
            # Calculate current quality score
            await self._calculate_comprehensive_quality_score_async()

            # Add to history
            self.quality_history.append(self.quality_score)

            # Analyze trends
            quality_trend = 0.0
            if len(self.quality_history) >= 10:
                recent_scores = list(self.quality_history)[-10:]
                older_scores = list(self.quality_history)[-20:-10] if len(self.quality_history) >= 20 else []

                current_avg = float(np.mean(recent_scores))
                previous_avg = float(np.mean(older_scores)) if older_scores else current_avg

                quality_trend = current_avg - previous_avg
                self.comprehensive_metrics["quality_trend"] = quality_trend

            # Update regime and session performance
            await self._update_regime_session_performance_async()

            return {
                'quality_trends_analyzed': True,
                'current_quality_score': self.quality_score,
                'quality_trend': quality_trend,
                'quality_history_size': len(self.quality_history)
            }

        except Exception as e:
            self.logger.warning(f"Quality trend analysis failed: {e}")
            return {'quality_trends_analyzed': False, 'error': str(e)}

    async def _calculate_comprehensive_quality_score_async(self) -> None:
        """Calculate comprehensive execution quality score asynchronously"""
        try:
            scores: List[float] = []
            weights: List[float] = []

            # Slippage score
            if self.slippage_history:
                percentile_slippage = float(np.percentile(list(self.slippage_history), self._cfg.slippage_percentile))
                slippage_score = max(0.0, 1.0 - (percentile_slippage / (self._cfg.slip_limit * 2)))
                scores.append(slippage_score)
                weights.append(0.3)

            # Latency score
            if self.latency_history:
                percentile_latency = float(np.percentile(list(self.latency_history), self._cfg.latency_percentile))
                latency_score = max(0.0, 1.0 - (percentile_latency / (self._cfg.latency_limit * 2)))
                scores.append(latency_score)
                weights.append(0.3)

            # Fill rate score
            if self.fill_history:
                avg_fill_rate = float(np.mean(list(self.fill_history)[-20:]))
                fill_score = max(0.0, min(1.0, avg_fill_rate))
                scores.append(fill_score)
                weights.append(0.25)

            # Spread score
            if self.spread_history:
                avg_spread = float(np.mean(list(self.spread_history)[-20:]))
                spread_score = max(0.0, 1.0 - (avg_spread / (self._cfg.spread_threshold * 2 if self._cfg.spread_threshold > 0 else 1e-6)))
                scores.append(spread_score)
                weights.append(0.15)

            # Calculate weighted average
            if scores and weights:
                self.quality_score = float(np.average(scores, weights=weights))
            else:
                self.quality_score = 1.0

            # Apply issue penalties
            total_issues = sum(len(issues) for issues in self.issues.values())
            if total_issues > 0:
                issue_penalty = min(0.3, total_issues * 0.05)
                self.quality_score = max(0.1, self.quality_score - issue_penalty)

            # Check for degraded executions
            if self.quality_score < self._cfg.degradation_threshold:
                self.degraded_executions += 1

        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {e}")
            self.quality_score = 0.5  # Conservative fallback

    async def _update_regime_session_performance_async(self) -> None:
        """Update regime and session performance tracking asynchronously"""
        try:
            # Update regime performance
            if self.market_regime and self.market_regime != 'unknown':
                regime_data = self.regime_performance[self.market_regime]
                regime_data['quality_scores'].append(self.quality_score)
                if self.slippage_history:
                    regime_data.setdefault('avg_slippage', []).append(float(np.mean(list(self.slippage_history)[-5:])))
                if self.latency_history:
                    regime_data.setdefault('avg_latency', []).append(float(np.mean(list(self.latency_history)[-5:])))

            # Update session performance
            if self.market_session and self.market_session != 'unknown':
                session_data = self.session_performance[self.market_session]
                session_data['quality_scores'].append(self.quality_score)
                if self.fill_history:
                    session_data.setdefault('fill_rates', []).append(float(np.mean(list(self.fill_history)[-5:])))

        except Exception as e:
            self.logger.warning(f"Regime/session performance update failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # ALERTS & MODES
    # ─────────────────────────────────────────────────────────────

    async def _check_quality_degradation_and_alerts(self) -> Dict[str, Any]:
        """Check for quality degradation and generate alerts"""
        try:
            alerts_generated = []

            # Check for significant quality degradation
            if self.quality_score < self._cfg.degradation_threshold:
                alert = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'quality_score': self.quality_score,
                    'threshold': self._cfg.degradation_threshold,
                    'context': {
                        'regime': self.market_regime,
                        'volatility': self.volatility_regime,
                        'session': self.market_session
                    },
                    'issues': {k: len(v) for k, v in self.issues.items()},
                    'severity': 'critical' if self.quality_score < 0.3 else 'warning'
                }

                self.quality_alerts.append(alert)
                alerts_generated.append(alert)

                self.logger.error(
                    format_operator_message(
                        "[ALERT]", "EXECUTION_QUALITY_DEGRADATION",
                        quality_score=f"{self.quality_score:.2f}",
                        threshold=f"{self._cfg.degradation_threshold:.2f}",
                        regime=self.market_regime,
                        context="quality_degradation"
                    )
                )

                # Check for escalation
                if (
                    self.quality_score < 0.3 and
                    (self.last_escalation is None or
                     (datetime.datetime.now() - self.last_escalation).total_seconds() > 300)  # 5 min cooldown
                ):
                    self.escalation_count += 1
                    self.last_escalation = datetime.datetime.now()

                    self.logger.error(
                        format_operator_message(
                            "[ALERT]", "EXECUTION_QUALITY_ESCALATION",
                            escalation_number=self.escalation_count,
                            quality_score=f"{self.quality_score:.2f}",
                            degraded_executions=self.degraded_executions,
                            context="quality_escalation"
                        )
                    )

            return {
                'quality_degradation_checked': True,
                'alerts_generated': len(alerts_generated),
                'total_quality_alerts': len(self.quality_alerts),
                'escalation_count': self.escalation_count,
                'quality_below_threshold': self.quality_score < self._cfg.degradation_threshold
            }

        except Exception as e:
            self.logger.warning(f"Quality degradation check failed: {e}")
            return {'quality_degradation_checked': False, 'error': str(e)}

    async def _update_comprehensive_metrics(self) -> Dict[str, Any]:
        """Update comprehensive execution metrics"""
        try:
            # Update averages
            if self.slippage_history:
                self.comprehensive_metrics["avg_slippage"] = float(np.mean(self.slippage_history))
            if self.latency_history:
                self.comprehensive_metrics["avg_latency"] = float(np.mean(self.latency_history))
            if self.fill_history:
                self.comprehensive_metrics["avg_fill_rate"] = float(np.mean(self.fill_history))
            if self.spread_history:
                self.comprehensive_metrics["avg_spread"] = float(np.mean(self.spread_history))

            # Update rates
            self.comprehensive_metrics["success_rate"] = (
                (self.execution_count - self.degraded_executions) / max(self.execution_count, 1)
            )
            self.comprehensive_metrics["degradation_rate"] = (
                self.degraded_executions / max(self.execution_count, 1)
            )

            return {
                'comprehensive_metrics_updated': True,
                'avg_slippage': self.comprehensive_metrics["avg_slippage"],
                'avg_latency': self.comprehensive_metrics["avg_latency"],
                'avg_fill_rate': self.comprehensive_metrics["avg_fill_rate"],
                'success_rate': self.comprehensive_metrics["success_rate"],
                'degradation_rate': self.comprehensive_metrics["degradation_rate"]
            }

        except Exception as e:
            self.logger.warning(f"Comprehensive metrics update failed: {e}")
            return {'comprehensive_metrics_updated': False, 'error': str(e)}

    async def _update_operational_mode(self) -> Dict[str, Any]:
        """Update operational mode based on execution quality"""
        try:
            old_mode = self.current_mode

            # Determine new mode based on quality conditions
            if self.quality_score < 0.3 or self.escalation_count > 3:
                new_mode = ExecutionMode.EMERGENCY
            elif self.quality_score < 0.5 or len(self.quality_alerts) > 5:
                new_mode = ExecutionMode.CRITICAL
            elif self.quality_score < self._cfg.degradation_threshold:
                new_mode = ExecutionMode.DEGRADED
            elif self.training_mode and self.execution_count < 50:
                new_mode = ExecutionMode.TRAINING
            elif self.execution_count < 100:
                new_mode = ExecutionMode.CALIBRATION
            else:
                new_mode = ExecutionMode.NORMAL

            # Update mode if changed
            mode_changed = False
            if new_mode != old_mode:
                self.current_mode = new_mode
                self.mode_start_time = datetime.datetime.now()
                mode_changed = True

                self.logger.info(
                    format_operator_message(
                        "[RELOAD]", "EXECUTION_MODE_CHANGE",
                        old_mode=old_mode.value,
                        new_mode=new_mode.value,
                        quality_score=f"{self.quality_score:.2f}",
                        execution_count=self.execution_count,
                        context="mode_transition"
                    )
                )

            return {
                'mode_updated': True,
                'current_mode': self.current_mode.value,
                'mode_changed': mode_changed,
                'old_mode': old_mode.value if mode_changed else None,
                'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds()
            }

        except Exception as e:
            self.logger.warning(f"Mode update failed: {e}")
            return {'mode_updated': False, 'error': str(e)}

    async def _generate_execution_thesis(self, execution_data: Dict[str, Any],
                                         result: Dict[str, Any]) -> str:
        """Generate comprehensive execution thesis"""
        try:
            # Core metrics
            quality_score = self.quality_score
            mode = self.current_mode.value
            execution_count = result.get('execution_count', 0)

            thesis_parts = [
                f"Execution Quality: {mode.upper()} mode with {quality_score:.1%} quality score",
                f"Processing: {execution_count} executions analyzed"
            ]

            # Quality assessment
            if quality_score < 0.5:
                thesis_parts.append("DEGRADED: Quality below acceptable threshold")
            elif quality_score > 0.8:
                thesis_parts.append("EXCELLENT: High execution quality maintained")

            # Issue analysis
            total_issues = sum(len(v) for v in self.issues.values())
            if total_issues > 0:
                issue_types = [k for k, v in self.issues.items() if v]
                thesis_parts.append(f"ISSUES: {total_issues} problems in {', '.join(issue_types[:2])}")

            # Performance metrics
            if self.comprehensive_metrics["avg_slippage"] > 0:
                thesis_parts.append(f"Slippage: {self.comprehensive_metrics['avg_slippage']:.4f} avg")

            if self.comprehensive_metrics["avg_latency"] > 0:
                thesis_parts.append(f"Latency: {self.comprehensive_metrics['avg_latency']:.0f}ms avg")

            # Market context
            thesis_parts.append(f"Context: {self.market_regime.upper()} regime, {self.volatility_regime.upper()} volatility")

            # Training status
            if self.training_mode:
                thesis_parts.append(f"TRAINING: {self.execution_count} total executions processed")

            return " | ".join(thesis_parts)

        except Exception as e:
            return f"Execution thesis generation failed: {str(e)} - Core execution monitoring functional"

    async def _update_execution_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with execution results"""
        try:
            # Execution quality
            execution_quality_data = {
                'current_mode': self.current_mode.value,
                'quality_score': self.quality_score,
                'training_mode': self.training_mode,
                'execution_count': self.execution_count,
                'degraded_executions': self.degraded_executions,
                'timestamp': datetime.datetime.now().isoformat()
            }

            self.smart_bus.set(
                'execution_quality',
                execution_quality_data,
                module='ExecutionQualityMonitor',
                thesis=thesis
            )

            # Execution analytics
            analytics_data = {
                'comprehensive_metrics': self.comprehensive_metrics.copy(),
                'issues': {k: len(v) for k, v in self.issues.items() if v},
                'quality_alerts': len(self.quality_alerts),
                'escalation_count': self.escalation_count,
                'regime_performance': {
                    regime: {
                        'quality_scores': len(data['quality_scores']),
                        'avg_quality': float(np.mean(data['quality_scores'][-10:])) if data['quality_scores'] else 0.0
                    }
                    for regime, data in self.regime_performance.items()
                }
            }

            self.smart_bus.set(
                'execution_analytics',
                analytics_data,
                module='ExecutionQualityMonitor',
                thesis="Comprehensive execution quality analytics and performance tracking"
            )

            # Quality metrics
            metrics_data = {
                'quality_score': self.quality_score,
                'avg_slippage': self.comprehensive_metrics["avg_slippage"],
                'avg_latency': self.comprehensive_metrics["avg_latency"],
                'avg_fill_rate': self.comprehensive_metrics["avg_fill_rate"],
                'avg_spread': self.comprehensive_metrics["avg_spread"],
                'success_rate': self.comprehensive_metrics["success_rate"],
                'degradation_rate': self.comprehensive_metrics["degradation_rate"]
            }

            self.smart_bus.set(
                'quality_metrics',
                metrics_data,
                module='ExecutionQualityMonitor',
                thesis="Real-time execution quality metrics and performance indicators"
            )

            # Execution alerts
            alerts_data = {
                'quality_alerts': len(self.quality_alerts),
                'escalation_count': self.escalation_count,
                'critical_issues': sum(1 for issues in self.issues.values() for _ in issues),
                'last_escalation': self.last_escalation.isoformat() if self.last_escalation else None,
                'current_issues': {k: len(v) for k, v in self.issues.items() if v}
            }

            self.smart_bus.set(
                'execution_alerts',
                alerts_data,
                module='ExecutionQualityMonitor',
                thesis="Execution quality alerts and escalation tracking"
            )

        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    # ─────────────────────────────────────────────────────────────
    # FALLBACKS & ERRORS
    # ─────────────────────────────────────────────────────────────

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no execution data is available"""
        self.logger.warning("No execution data available - maintaining current state")

        thesis = "No execution data available - maintaining current state"
        execution_quality_data = {
            'current_mode': self.current_mode.value,
            'quality_score': self.quality_score,
            'training_mode': self.training_mode,
            'execution_count': self.execution_count,
            'degraded_executions': self.degraded_executions,
            'timestamp': datetime.datetime.now().isoformat()
        }
        analytics_data = {
            'comprehensive_metrics': self.comprehensive_metrics.copy(),
            'issues': {k: len(v) for k, v in self.issues.items() if v},
            'quality_alerts': len(self.quality_alerts),
            'escalation_count': self.escalation_count
        }
        metrics_data = {
            'quality_score': self.quality_score,
            'avg_slippage': self.comprehensive_metrics.get("avg_slippage", 0.0),
            'avg_latency': self.comprehensive_metrics.get("avg_latency", 0.0),
            'avg_fill_rate': self.comprehensive_metrics.get("avg_fill_rate", 1.0),
            'avg_spread': self.comprehensive_metrics.get("avg_spread", 0.0),
            'success_rate': self.comprehensive_metrics.get("success_rate", 1.0),
            'degradation_rate': self.comprehensive_metrics.get("degradation_rate", 0.0)
        }
        alerts_data = {
            'quality_alerts': len(self.quality_alerts),
            'escalation_count': self.escalation_count,
            'critical_issues': sum(1 for issues in self.issues.values() for _ in issues),
            'last_escalation': self.last_escalation.isoformat() if self.last_escalation else None,
            'current_issues': {k: len(v) for k, v in self.issues.items() if v}
        }

        return {
            'execution_quality': execution_quality_data,
            'execution_analytics': analytics_data,
            'quality_metrics': metrics_data,
            'execution_alerts': alerts_data,
            '_thesis': thesis,
            'success': True,
            'fallback_reason': 'no_execution_data'
        }

    async def _handle_execution_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle execution monitoring errors"""
        processing_time = (time.time() - start_time) * 1000

        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()

        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self._health_status = 'warning'

        # Log error with context
        _ = self.error_pinpointer.analyze_error(error, "ExecutionQualityMonitor")
        explanation = self.english_explainer.explain_error(
            "ExecutionQualityMonitor", str(error), "execution quality monitoring"
        )

        self.logger.error(
            format_operator_message(
                "[CRASH]", "EXECUTION_MONITOR_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                circuit_breaker_state=self.circuit_breaker['state'],
                context="execution_error"
            )
        )

        # Record failure
        self._record_failure(error)

        return self._create_error_fallback_response(f"error: {str(error)}")

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        thesis = f"Execution monitor error fallback: {reason}"
        execution_quality_data = {
            'current_mode': ExecutionMode.EMERGENCY.value,
            'quality_score': 0.1,
            'training_mode': self.training_mode,
            'execution_count': self.execution_count,
            'degraded_executions': self.degraded_executions,
            'timestamp': datetime.datetime.now().isoformat()
        }
        analytics_data = {
            'comprehensive_metrics': self.comprehensive_metrics.copy(),
            'issues': {k: len(v) for k, v in self.issues.items() if v},
            'quality_alerts': len(self.quality_alerts),
            'escalation_count': self.escalation_count
        }
        metrics_data = {
            'quality_score': 0.1,
            'avg_slippage': self.comprehensive_metrics.get("avg_slippage", 0.0),
            'avg_latency': self.comprehensive_metrics.get("avg_latency", 0.0),
            'avg_fill_rate': self.comprehensive_metrics.get("avg_fill_rate", 1.0),
            'avg_spread': self.comprehensive_metrics.get("avg_spread", 0.0),
            'success_rate': self.comprehensive_metrics.get("success_rate", 0.0),
            'degradation_rate': self.comprehensive_metrics.get("degradation_rate", 1.0)
        }
        alerts_data = {
            'quality_alerts': len(self.quality_alerts),
            'escalation_count': self.escalation_count,
            'critical_issues': sum(1 for issues in self.issues.values() for _ in issues),
            'last_escalation': self.last_escalation.isoformat() if self.last_escalation else None,
            'current_issues': {k: len(v) for k, v in self.issues.items() if v}
        }

        return {
            'execution_quality': execution_quality_data,
            'execution_analytics': analytics_data,
            'quality_metrics': metrics_data,
            'execution_alerts': alerts_data,
            '_thesis': thesis,
            'success': False,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    # ─────────────────────────────────────────────────────────────
    # HEALTH, EFFECTIVENESS & ADAPTATION
    # ─────────────────────────────────────────────────────────────

    def _update_execution_health(self):
        """Update execution health metrics"""
        try:
            # Check execution quality
            if self.quality_score < self._cfg.min_execution_quality:
                self._health_status = 'warning'
            else:
                self._health_status = 'healthy'

            # Check circuit breaker
            if self.circuit_breaker['state'] == 'OPEN':
                self._health_status = 'warning'

            # Check for excessive degradation
            if self.degraded_executions > self.execution_count * 0.5:
                self._health_status = 'warning'

            self._last_health_check = time.time()

        except Exception as e:
            self.logger.error(f"Execution health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_execution_effectiveness(self):
        """Analyze execution monitoring effectiveness"""
        try:
            if self.execution_count >= 20:
                effectiveness = self.quality_score

                if effectiveness > 0.8:
                    self.logger.info(
                        format_operator_message(
                            "[TARGET]", "HIGH_EXECUTION_EFFECTIVENESS",
                            quality_score=f"{effectiveness:.2f}",
                            execution_count=self.execution_count,
                            context="execution_analysis"
                        )
                    )
                elif effectiveness < 0.4:
                    self.logger.warning(
                        format_operator_message(
                            "[WARN]", "LOW_EXECUTION_EFFECTIVENESS",
                            quality_score=f"{effectiveness:.2f}",
                            degraded_executions=self.degraded_executions,
                            context="execution_analysis"
                        )
                    )

        except Exception as e:
            self.logger.error(f"Execution effectiveness analysis failed: {e}")

    def _adapt_quality_parameters(self):
        """Continuous quality parameter adaptation"""
        try:
            # Adapt threshold scaling based on recent performance
            if len(self.quality_history) >= 10:
                recent_quality = float(np.mean(list(self.quality_history)[-10:]))

                if recent_quality < 0.5:  # Poor quality
                    self._adaptive_params['dynamic_threshold_scaling'] = min(
                        1.5, self._adaptive_params['dynamic_threshold_scaling'] * 1.02
                    )
                elif recent_quality > 0.8:  # Good quality
                    self._adaptive_params['dynamic_threshold_scaling'] = max(
                        0.7, self._adaptive_params['dynamic_threshold_scaling'] * 0.995
                    )

            # Adapt context sensitivity
            total_issues = sum(len(v) for v in self.issues.values())
            if total_issues > 10:
                self._adaptive_params['context_sensitivity'] = min(
                    1.5, self._adaptive_params['context_sensitivity'] * 1.01
                )
            elif total_issues == 0:
                self._adaptive_params['context_sensitivity'] = max(
                    0.8, self._adaptive_params['context_sensitivity'] * 0.999
                )

        except Exception as e:
            self.logger.warning(f"Quality parameter adaptation failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'ExecutionQualityMonitor', 'execution_monitoring', processing_time, True
        )

        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'ExecutionQualityMonitor', 'execution_monitoring', 0, False
        )

    # ─────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE
    # ─────────────────────────────────────────────────────────────

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get detailed execution statistics"""
        stats: Dict[str, Any] = {}

        try:
            # Slippage statistics
            if self.slippage_history:
                slips = list(self.slippage_history)
                stats["slippage"] = {
                    "mean": float(np.mean(slips)),
                    "std": float(np.std(slips)),
                    "max": float(np.max(slips)),
                    "min": float(np.min(slips)),
                    "p95": float(np.percentile(slips, 95)),
                    "p99": float(np.percentile(slips, 99)),
                    "count": len(slips),
                    "violations": len([s for s in slips if s > self._cfg.slip_limit])
                }

            # Latency statistics
            if self.latency_history:
                latencies = list(self.latency_history)
                stats["latency"] = {
                    "mean": float(np.mean(latencies)),
                    "std": float(np.std(latencies)),
                    "max": float(np.max(latencies)),
                    "min": float(np.min(latencies)),
                    "p95": float(np.percentile(latencies, 95)),
                    "p99": float(np.percentile(latencies, 99)),
                    "count": len(latencies),
                    "violations": len([l for l in latencies if l > self._cfg.latency_limit])
                }

            # Fill rate statistics
            if self.fill_history:
                fills = list(self.fill_history)
                stats["fill_rate"] = {
                    "mean": float(np.mean(fills)),
                    "std": float(np.std(fills)),
                    "min": float(np.min(fills)),
                    "max": float(np.max(fills)),
                    "current": float(fills[-1]) if fills else 1.0,
                    "below_threshold_count": len([f for f in fills if f < self._cfg.min_fill_rate]),
                    "count": len(fills)
                }

            # Quality statistics
            if self.quality_history:
                qualities = list(self.quality_history)
                stats["quality"] = {
                    "current": float(self.quality_score),
                    "mean": float(np.mean(qualities)),
                    "trend": float(self.comprehensive_metrics.get('quality_trend', 0.0)),
                    "degraded_count": int(self.degraded_executions),
                    "degradation_rate": float(self.comprehensive_metrics.get('degradation_rate', 0.0))
                }

        except Exception as e:
            self.logger.error(f"Error calculating execution stats: {e}")

        return stats

    def get_observation_components(self) -> np.ndarray:
        """Return execution quality metrics as observation"""
        try:
            has_issues = float(any(len(issues) > 0 for issues in self.issues.values()))

            recent_slippage = 0.0
            recent_latency = 0.0
            recent_fill_rate = 1.0
            recent_spread = 0.0

            if self.slippage_history:
                recent_slippage = float(np.mean(list(self.slippage_history)[-10:]))
            if self.latency_history:
                recent_latency = float(np.mean(list(self.latency_history)[-10:]))
            if self.fill_history:
                recent_fill_rate = float(np.mean(list(self.fill_history)[-10:]))
            if self.spread_history:
                recent_spread = float(np.mean(list(self.spread_history)[-10:]))

            return np.array([
                float(self.quality_score),
                has_issues,
                float(np.clip(recent_slippage / max(self._cfg.slip_limit, 1e-8), 0.0, 10.0)),
                float(np.clip(recent_latency / max(self._cfg.latency_limit, 1.0), 0.0, 10.0)),
                float(np.clip(recent_fill_rate, 0.0, 1.0)),
                float(np.clip(recent_spread / max(self._cfg.spread_threshold, 1e-8), 0.0, 10.0)),
                float(self.degraded_executions / max(self.execution_count, 1)),
                float(self.escalation_count / 10.0),  # Normalize escalations
                float(1.0 if self.current_mode in [ExecutionMode.CRITICAL, ExecutionMode.EMERGENCY] else 0.0),
                float(self.training_mode)
            ], dtype=np.float32)

        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'quality_score': self.quality_score,
            'execution_count': self.execution_count,
            'training_mode': self.training_mode
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    def get_execution_quality_report(self) -> str:
        """Generate operator-friendly execution quality report"""

        # Quality status indicators
        if self.quality_score < 0.5:
            quality_status = "[ALERT] Critical"
        elif self.quality_score < 0.7:
            quality_status = "[WARN] Poor"
        elif self.quality_score < 0.9:
            quality_status = "[FAST] Good"
        else:
            quality_status = "[OK] Excellent"

        # Mode status
        mode_emoji = {
            ExecutionMode.TRAINING: "🎓",
            ExecutionMode.CALIBRATION: "[TOOL]",
            ExecutionMode.NORMAL: "[OK]",
            ExecutionMode.DEGRADED: "[WARN]",
            ExecutionMode.CRITICAL: "[ALERT]",
            ExecutionMode.EMERGENCY: "🆘"
        }

        mode_status = f"{mode_emoji.get(self.current_mode, '❓')} {self.current_mode.value.upper()}"

        # Health status
        health_emoji = "[OK]" if self._health_status == 'healthy' else "[WARN]"
        cb_status = "[RED] OPEN" if self.circuit_breaker['state'] == 'OPEN' else "[GREEN] CLOSED"

        # Issue summary
        issue_lines = []
        for issue_type, issues in self.issues.items():
            if issues:
                count = len(issues)
                emoji = "[ALERT]" if count > 3 else "[WARN]" if count > 1 else "[FAST]"
                issue_lines.append(f"  {emoji} {issue_type.replace('_', ' ').title()}: {count} issues")

        return f"""
[FAST] ENHANCED EXECUTION QUALITY MONITOR v4.0
═══════════════════════════════════════════════════
[TARGET] Quality Status: {quality_status} ({self.quality_score:.1%})
[TOOL] Monitor Mode: {mode_status}
[STATS] Market Regime: {self.market_regime.title()}
[CRASH] Volatility Level: {self.volatility_regime.title()}
🕐 Market Session: {self.market_session.title()}
🎓 Training Mode: {'[OK] Active' if self.training_mode else '[FAIL] Inactive'}

[HEALTH] SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Execution Count: {self.execution_count:,}

[STATS] EXECUTION METRICS
• Quality Score: {self.quality_score:.1%}
• Success Rate: {self.comprehensive_metrics['success_rate']:.1%}
• Degraded Executions: {self.degraded_executions}
• Degradation Rate: {self.comprehensive_metrics['degradation_rate']:.1%}

[CHART] PERFORMANCE STATISTICS
• Avg Slippage: {self.comprehensive_metrics['avg_slippage']:.5f} (limit: {self._cfg.slip_limit:.5f})
• Avg Latency: {self.comprehensive_metrics['avg_latency']:.0f}ms (limit: {self._cfg.latency_limit}ms)
• Avg Fill Rate: {self.comprehensive_metrics['avg_fill_rate']:.1%} (min: {self._cfg.min_fill_rate:.1%})
• Avg Spread: {self.comprehensive_metrics['avg_spread']:.5f}
• Quality Trend: {self.comprehensive_metrics['quality_trend']:+.2f}

[WARN] CURRENT ISSUES
{chr(10).join(issue_lines) if issue_lines else "  [OK] No current execution issues"}

[ALERT] ESCALATIONS & ALERTS
• Escalation Count: {self.escalation_count}
• Quality Alerts: {len(self.quality_alerts)}
• Last Escalation: {self.last_escalation.strftime('%H:%M:%S') if self.last_escalation else 'None'}

💡 THRESHOLDS & LIMITS
• Slippage Limit: {self._cfg.slip_limit:.5f}
• Latency Limit: {self._cfg.latency_limit}ms
• Min Fill Rate: {self._cfg.min_fill_rate:.1%}
• Spread Threshold: {self._cfg.spread_threshold:.5f}
• Quality Threshold: {self._cfg.quality_threshold:.1%}
• Degradation Threshold: {self._cfg.degradation_threshold:.1%}

[TOOL] SYSTEM STATUS
• Statistics Window: {self._cfg.stats_window} records
• History Sizes: S:{len(self.slippage_history)} L:{len(self.latency_history)} F:{len(self.fill_history)} Q:{len(self.quality_history)}
        """

    # ─────────────────────────────────────────────────────────────
    # LEGACY COMPATIBILITY
    # ─────────────────────────────────────────────────────────────

    def step(self, **kwargs) -> Dict[str, Any]:
        """Legacy step interface for backward compatibility"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            result = loop.run_until_complete(self.process(**kwargs))
            return result
        finally:
            loop.close()

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        # Reset history
        self.slippage_history.clear()
        self.latency_history.clear()
        self.fill_history.clear()
        self.spread_history.clear()
        self.quality_history.clear()

        # Reset instrument-specific metrics
        self.instrument_metrics.clear()

        # Reset quality metrics
        self.quality_score = 1.0
        self.execution_count = 0
        self.degraded_executions = 0

        # Reset market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"

        # Reset issues
        for issue_type in self.issues:
            self.issues[issue_type].clear()

        # Reset analytics
        self.execution_analytics.clear()
        self.regime_performance.clear()
        self.session_performance.clear()

        # Reset comprehensive metrics
        self.comprehensive_metrics = {
            "avg_slippage": 0.0,
            "avg_latency": 0.0,
            "avg_fill_rate": 1.0,
            "avg_spread": 0.0,
            "total_executions": 0,
            "success_rate": 1.0,
            "degradation_rate": 0.0,
            "quality_trend": 0.0
        }

        # Reset venue performance
        self.venue_performance.clear()

        # Reset alerts
        self.quality_alerts.clear()
        self.escalation_count = 0
        self.last_escalation = None

        # Reset mode
        self.current_mode = ExecutionMode.TRAINING if self.training_mode else ExecutionMode.NORMAL
        self.mode_start_time = datetime.datetime.now()

        # Reset circuit breaker
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['state'] = 'CLOSED'
        self._health_status = 'healthy'

        # Reset adaptive parameters
        self._adaptive_params = {
            'dynamic_threshold_scaling': 1.0,
            'context_sensitivity': 1.0,
            'quality_adaptation_confidence': 0.5
        }

        self.logger.info("[RELOAD] Enhanced Execution Quality Monitor reset - all state cleared")

    # ─────────────────────────────────────────────────────────────
    # BASEMODULE ABSTRACT METHOD IMPLEMENTATIONS
    # ─────────────────────────────────────────────────────────────

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in execution quality assessment"""
        try:
            # Base confidence from execution quality score
            execution_quality = float(self.quality_score)

            # Mode adjustment factor
            mode_confidence = {
                ExecutionMode.TRAINING: 0.6,
                ExecutionMode.CALIBRATION: 0.7,
                ExecutionMode.NORMAL: 1.0,
                ExecutionMode.DEGRADED: 0.4,
                ExecutionMode.CRITICAL: 0.2,
                ExecutionMode.EMERGENCY: 0.1
            }.get(self.current_mode, 0.5)

            # Data quality factor
            latency_count = len(self.latency_history) if hasattr(self, 'latency_history') else 0
            data_quality = min(latency_count / max(self._cfg.stats_window, 1), 1.0)

            # Recent performance consistency
            if hasattr(self, 'quality_history') and len(self.quality_history) > 2:
                recent_qualities = list(self.quality_history)[-5:]
                denom = max(float(np.mean(recent_qualities)), 0.1)
                consistency = 1.0 - float(np.std(recent_qualities)) / denom
            else:
                consistency = 0.5

            # Health status factor
            health_factor = 1.0 if self._health_status == 'healthy' else 0.3

            # Combine factors
            confidence = (
                execution_quality * 0.4 +
                mode_confidence * 0.25 +
                data_quality * 0.2 +
                consistency * 0.1 +
                health_factor * 0.05
            )

            # Ensure valid range
            return float(max(0.1, min(0.95, float(confidence))))

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.4  # Conservative default

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose execution quality action for trading system optimization"""
        try:
            # Analyze current execution quality
            overall_quality = float(self.quality_score)
            current_mode = self.current_mode
            alert_count = len(self.quality_alerts)

            # Get specific quality metrics from comprehensive_metrics
            sl_limit = max(self._cfg.slip_limit, 1e-6)
            lat_limit = max(float(self._cfg.latency_limit), 1.0)

            slippage_quality = 1.0 - min(1.0, float(self.comprehensive_metrics.get('avg_slippage', 0.0)) / sl_limit)
            latency_quality = 1.0 - min(1.0, float(self.comprehensive_metrics.get('avg_latency', 0.0)) / lat_limit)
            fill_rate_quality = float(self.comprehensive_metrics.get('avg_fill_rate', 1.0))

            # Determine action based on execution quality
            if overall_quality < 0.3 or current_mode in [ExecutionMode.CRITICAL, ExecutionMode.EMERGENCY]:
                action_type = 'emergency_mode'
                signal_strength = 0.95
                reasoning = f"Critical execution quality ({overall_quality:.3f}) or emergency mode"
            elif overall_quality < 0.5 or current_mode == ExecutionMode.DEGRADED:
                action_type = 'reduce_execution_risk'
                signal_strength = 0.8
                reasoning = f"Poor execution quality ({overall_quality:.3f}) requires risk reduction"
            elif slippage_quality < 0.4:
                action_type = 'optimize_slippage'
                signal_strength = 0.7
                reasoning = f"High slippage detected (quality: {slippage_quality:.3f})"
            elif latency_quality < 0.4:
                action_type = 'optimize_latency'
                signal_strength = 0.6
                reasoning = f"High latency detected (quality: {latency_quality:.3f})"
            elif fill_rate_quality < 0.6:
                action_type = 'improve_fills'
                signal_strength = 0.5
                reasoning = f"Poor fill rates (quality: {fill_rate_quality:.3f})"
            elif alert_count > 3:
                action_type = 'review_alerts'
                signal_strength = 0.4
                reasoning = f"Multiple quality alerts ({alert_count}) require attention"
            elif overall_quality > 0.8:
                action_type = 'maintain_quality'
                signal_strength = 0.3
                reasoning = f"Excellent execution quality ({overall_quality:.3f})"
            else:
                action_type = 'monitor'
                signal_strength = 0.4
                reasoning = f"Normal execution quality ({overall_quality:.3f})"

            return {
                'action': action_type,
                'signal_strength': signal_strength,
                'reasoning': reasoning,
                'execution_metrics': {
                    'overall_quality': overall_quality,
                    'slippage_quality': slippage_quality,
                    'latency_quality': latency_quality,
                    'fill_rate_quality': fill_rate_quality,
                    'current_mode': current_mode.value,
                    'alert_count': alert_count
                },
                'system_status': {
                    'health_status': self._health_status,
                    'training_mode': self.training_mode,
                    'circuit_breaker_state': self.circuit_breaker.get('state', 'UNKNOWN')
                },
                'confidence': await self.calculate_confidence({}, **inputs)
            }

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Action proposal failed: {e}")
            return {
                'action': 'abstain',
                'signal_strength': 0.0,
                'reasoning': f'Execution quality error: {str(e)}',
                'confidence': 0.1
            }
