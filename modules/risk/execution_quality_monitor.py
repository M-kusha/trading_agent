# ─────────────────────────────────────────────────────────────
# File: modules/risk/execution_quality_monitor.py
# Enhanced Execution Quality Monitor with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import copy
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import RiskMixin, AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class ExecutionQualityMonitor(Module, RiskMixin, AnalysisMixin, StateManagementMixin):
    """
    Enhanced execution quality monitor with InfoBus integration.
    Monitors execution metrics including slippage, latency, fill rates, and spreads
    with intelligent context-aware analysis.
    """

    # Enhanced default configuration
    ENHANCED_DEFAULTS = {
        "slip_limit": 0.002,
        "latency_limit": 1000,
        "min_fill_rate": 0.95,
        "stats_window": 50,
        "slippage_percentile": 95,
        "latency_percentile": 95,
        "spread_threshold": 0.01,
        "execution_timeout": 5000,
        "quality_threshold": 0.7,
        "degradation_threshold": 0.5
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        training_mode: bool = True,
        debug: bool = False,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 100),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_risk_state()
        self._initialize_analysis_state()
        
        # Merge configuration with enhanced defaults
        self.execution_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if config:
            self.execution_config.update(config)
        
        # Core parameters
        self.slip_limit = float(self.execution_config["slip_limit"])
        self.latency_limit = int(self.execution_config["latency_limit"])
        self.min_fill_rate = float(self.execution_config["min_fill_rate"])
        self.stats_window = int(self.execution_config["stats_window"])
        self.slippage_percentile = int(self.execution_config["slippage_percentile"])
        self.latency_percentile = int(self.execution_config["latency_percentile"])
        self.spread_threshold = float(self.execution_config["spread_threshold"])
        self.execution_timeout = int(self.execution_config["execution_timeout"])
        self.quality_threshold = float(self.execution_config["quality_threshold"])
        self.degradation_threshold = float(self.execution_config["degradation_threshold"])
        
        # Execution mode
        self.training_mode = training_mode
        
        # Enhanced statistics tracking
        self.slippage_history = deque(maxlen=self.stats_window)
        self.latency_history = deque(maxlen=self.stats_window)
        self.fill_history = deque(maxlen=self.stats_window)
        self.spread_history = deque(maxlen=self.stats_window)
        self.quality_history = deque(maxlen=self.stats_window)
        
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
        self.last_escalation = None
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "ExecutionQualityMonitor",
            "logs/risk/execution_quality_monitor.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("ExecutionQualityMonitor")
        
        self.log_operator_info(
            "⚡ Enhanced Execution Quality Monitor initialized",
            slip_limit=f"{self.slip_limit:.4f}",
            latency_limit=f"{self.latency_limit}ms",
            min_fill_rate=f"{self.min_fill_rate:.1%}",
            training_mode=training_mode,
            stats_window=self.stats_window
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_analysis_state()
        
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
        
        self.log_operator_info("🔄 Execution Quality Monitor reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using fallback mode")
            self._process_legacy_step(**kwargs)
            return
        
        # Extract comprehensive context
        context = extract_standard_context(info_bus)
        
        # Update market context awareness
        self._update_market_context(context, info_bus)
        
        # Extract execution data from InfoBus
        execution_data = self._extract_execution_data_from_info_bus(info_bus)
        
        # Process executions with context awareness
        self._process_executions_comprehensive(execution_data, context, info_bus)
        
        # Generate realistic training data if needed
        if self.training_mode and self._should_generate_training_data():
            self._generate_realistic_training_data(context)
        
        # Analyze execution quality trends
        self._analyze_execution_quality_trends(context)
        
        # Check for quality degradation and alerts
        self._check_quality_degradation_and_alerts(context)
        
        # Update comprehensive metrics
        self._update_comprehensive_metrics()
        
        # Update InfoBus with results
        self._update_info_bus(info_bus)
        
        # Record audit for significant events
        self._record_execution_audit(info_bus, context, execution_data)
        
        # Update performance metrics
        self._update_execution_performance_metrics()

    def _extract_execution_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract comprehensive execution data from InfoBus"""
        
        data = {
            'executions': [],
            'orders': [],
            'spreads': {},
            'market_data': {},
            'venue_info': {}
        }
        
        try:
            # Get executions from recent trades
            recent_trades = info_bus.get('recent_trades', [])
            for trade in recent_trades:
                execution = self._convert_trade_to_execution(trade)
                if execution:
                    data['executions'].append(execution)
            
            # Get orders
            pending_orders = info_bus.get('pending_orders', [])
            completed_orders = info_bus.get('completed_orders', [])
            data['orders'] = pending_orders + completed_orders
            
            # Get market data for spread calculation
            prices = info_bus.get('prices', {})
            market_context = info_bus.get('market_context', {})
            
            # Extract spreads if available
            if 'spreads' in market_context:
                data['spreads'] = market_context['spreads']
            elif 'bid_ask_spreads' in info_bus:
                data['spreads'] = info_bus['bid_ask_spreads']
            
            # Get venue information
            broker_info = info_bus.get('broker_info', {})
            if broker_info:
                data['venue_info'] = {
                    'broker': broker_info.get('name', 'unknown'),
                    'server': broker_info.get('server', 'unknown'),
                    'connection_quality': broker_info.get('connection_quality', 1.0)
                }
            
            # Market data
            data['market_data'] = {
                'prices': prices,
                'volatility': market_context.get('volatility', {}),
                'liquidity': market_context.get('liquidity_score', 1.0)
            }
            
        except Exception as e:
            self.log_operator_warning(f"Execution data extraction failed: {e}")
        
        return data

    def _convert_trade_to_execution(self, trade: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert trade data to execution format"""
        
        try:
            execution = {
                'instrument': trade.get('symbol', trade.get('instrument', 'UNKNOWN')),
                'size': abs(trade.get('size', trade.get('volume', 0))),
                'side': trade.get('side', 'BUY' if trade.get('size', 0) > 0 else 'SELL'),
                'timestamp': trade.get('timestamp', datetime.datetime.now().isoformat()),
                'execution_price': trade.get('price', trade.get('fill_price', 0)),
            }
            
            # Extract execution quality metrics
            execution['slippage'] = self._extract_slippage(trade)
            execution['latency_ms'] = self._extract_latency(trade)
            execution['spread'] = self._extract_spread(trade)
            execution['fill_status'] = self._extract_fill_status(trade)
            
            # Add venue information if available
            execution['venue'] = trade.get('broker', trade.get('venue', 'unknown'))
            
            return execution
            
        except Exception as e:
            self.log_operator_warning(f"Trade conversion failed: {e}")
            return None

    def _extract_slippage(self, trade: Dict[str, Any]) -> Optional[float]:
        """Extract slippage from trade data with multiple fallback methods"""
        
        # Method 1: Direct slippage field
        for field in ["slippage", "slip", "price_diff", "execution_slippage"]:
            if field in trade and trade[field] is not None:
                return float(trade[field])
        
        # Method 2: Calculate from expected vs actual price
        expected_price = trade.get("expected_price", trade.get("order_price"))
        actual_price = trade.get("actual_price", trade.get("fill_price", trade.get("price")))
        
        if expected_price is not None and actual_price is not None:
            expected_price = float(expected_price)
            actual_price = float(actual_price)
            if expected_price > 0:
                return (actual_price - expected_price) / expected_price
        
        # Method 3: Estimate from market impact
        size = trade.get('size', trade.get('volume', 0))
        if size and abs(size) > 0.1:  # Significant size
            # Estimate slippage based on size (simplified)
            return abs(size) * 0.0001  # 1 pip per lot estimation
        
        return None

    def _extract_latency(self, trade: Dict[str, Any]) -> Optional[float]:
        """Extract latency from trade data with multiple methods"""
        
        # Method 1: Direct latency fields
        for field in ["latency_ms", "latency", "execution_time_ms", "fill_time_ms"]:
            if field in trade and trade[field] is not None:
                return float(trade[field])
        
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
                    return max(0, latency_seconds * 1000)  # Convert to milliseconds
            except Exception:
                pass
        
        # Method 3: Estimate based on market conditions
        if self.training_mode:
            # Generate realistic latency for training
            base_latency = 200  # 200ms base
            if self.market_session == "asian":
                base_latency += 100  # Higher latency for Asian session
            elif self.volatility_regime == "high":
                base_latency += 150  # Higher latency during volatility
            
            return float(base_latency + np.random.gamma(2, 50))  # Add some variation
        
        return None

    def _extract_spread(self, trade: Dict[str, Any]) -> Optional[float]:
        """Extract spread from trade data"""
        
        # Method 1: Direct spread fields
        for field in ["spread", "bid_ask_spread", "market_spread"]:
            if field in trade and trade[field] is not None:
                return float(trade[field])
        
        # Method 2: Calculate from bid/ask
        bid = trade.get("bid_price", trade.get("bid"))
        ask = trade.get("ask_price", trade.get("ask"))
        
        if bid is not None and ask is not None:
            bid, ask = float(bid), float(ask)
            if bid > 0 and ask > bid:
                return ask - bid
        
        # Method 3: Estimate from instrument type
        instrument = trade.get('instrument', trade.get('symbol', ''))
        if 'XAU' in instrument or 'GOLD' in instrument:
            return 0.5  # Typical gold spread in USD
        elif 'EUR' in instrument or 'USD' in instrument:
            return 0.00015  # Typical forex spread
        
        return None

    def _extract_fill_status(self, trade: Dict[str, Any]) -> str:
        """Extract fill status from trade data"""
        
        # Check various status indicators
        status_fields = ["status", "state", "fill_status", "order_status"]
        for field in status_fields:
            status = trade.get(field, "")
            if status in ["filled", "completed", "executed"]:
                return "filled"
            elif status in ["partial", "partially_filled"]:
                return "partial"
            elif status in ["rejected", "cancelled", "failed"]:
                return "failed"
        
        # Check fill quantities
        order_qty = trade.get("quantity", trade.get("size", trade.get("volume", 0)))
        filled_qty = trade.get("filled_quantity", trade.get("filled_size", trade.get("executed_quantity", order_qty)))
        
        if order_qty > 0 and filled_qty > 0:
            fill_ratio = filled_qty / order_qty
            if fill_ratio >= 0.99:
                return "filled"
            elif fill_ratio >= 0.01:
                return "partial"
        
        return "unknown"

    def _update_market_context(self, context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Update market context awareness"""
        
        try:
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = context.get('regime', 'unknown')
            self.volatility_regime = context.get('volatility_level', 'medium')
            self.market_session = context.get('session', 'unknown')
            
            # Log regime changes
            if self.market_regime != old_regime:
                self.log_operator_info(
                    f"📊 Market regime change: {old_regime} → {self.market_regime}",
                    volatility=self.volatility_regime,
                    session=self.market_session
                )
            
        except Exception as e:
            self.log_operator_warning(f"Market context update failed: {e}")

    def _process_executions_comprehensive(self, execution_data: Dict[str, Any], 
                                        context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Process executions with comprehensive analysis"""
        
        try:
            executions = execution_data.get('executions', [])
            orders = execution_data.get('orders', [])
            spreads = execution_data.get('spreads', {})
            
            # Clear previous issues
            for issue_type in self.issues:
                self.issues[issue_type].clear()
            
            execution_count = 0
            
            # Process individual executions
            for execution in executions:
                try:
                    self._analyze_single_execution(execution, context)
                    execution_count += 1
                except Exception as e:
                    self.log_operator_warning(f"Execution analysis failed: {e}")
            
            # Process fill rates from orders
            if orders:
                self._analyze_fill_rates(orders, context)
            
            # Process spread data
            if spreads:
                self._analyze_spread_data(spreads, context)
            
            # Update execution count
            self.execution_count += execution_count
            self.comprehensive_metrics["total_executions"] = self.execution_count
            
        except Exception as e:
            self.log_operator_error(f"Execution processing failed: {e}")

    def _analyze_single_execution(self, execution: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Analyze individual execution with context awareness"""
        
        try:
            instrument = execution.get('instrument', 'UNKNOWN')
            
            # Analyze slippage
            slippage = execution.get('slippage')
            if slippage is not None:
                slippage = abs(float(slippage))
                self.slippage_history.append(slippage)
                self.instrument_metrics[instrument]['slippage'].append(slippage)
                
                # Context-aware slippage limits
                adjusted_limit = self._get_context_adjusted_slip_limit(context)
                
                if slippage > adjusted_limit:
                    issue = {
                        'instrument': instrument,
                        'value': slippage,
                        'limit': adjusted_limit,
                        'context': context.copy(),
                        'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                    }
                    self.issues["slippage"].append(issue)
                    self.log_operator_warning(
                        f"⚠️ High slippage: {instrument} {slippage:.5f} > {adjusted_limit:.5f}"
                    )
            
            # Analyze latency
            latency = execution.get('latency_ms')
            if latency is not None:
                latency = float(latency)
                self.latency_history.append(latency)
                self.instrument_metrics[instrument]['latency'].append(latency)
                
                # Context-aware latency limits
                adjusted_limit = self._get_context_adjusted_latency_limit(context)
                
                if latency > adjusted_limit:
                    issue = {
                        'instrument': instrument,
                        'value': latency,
                        'limit': adjusted_limit,
                        'context': context.copy(),
                        'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                    }
                    self.issues["latency"].append(issue)
                    self.log_operator_warning(
                        f"⏱️ High latency: {instrument} {latency:.0f}ms > {adjusted_limit:.0f}ms"
                    )
            
            # Analyze spread
            spread = execution.get('spread')
            if spread is not None:
                spread = float(spread)
                self.spread_history.append(spread)
                self.instrument_metrics[instrument]['spread'].append(spread)
                
                if spread > self.spread_threshold:
                    issue = {
                        'instrument': instrument,
                        'value': spread,
                        'threshold': self.spread_threshold,
                        'context': context.copy(),
                        'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                    }
                    self.issues["spread"].append(issue)
            
            # Analyze fill status
            fill_status = execution.get('fill_status', 'unknown')
            if fill_status == 'partial':
                issue = {
                    'instrument': instrument,
                    'status': fill_status,
                    'context': context.copy(),
                    'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                }
                self.issues["partial_fill"].append(issue)
            elif fill_status == 'failed':
                issue = {
                    'instrument': instrument,
                    'status': fill_status,
                    'context': context.copy(),
                    'timestamp': execution.get('timestamp', datetime.datetime.now().isoformat())
                }
                self.issues["fill_rate"].append(issue)
            
        except Exception as e:
            self.log_operator_warning(f"Single execution analysis failed: {e}")

    def _get_context_adjusted_slip_limit(self, context: Dict[str, Any]) -> float:
        """Get context-adjusted slippage limit"""
        
        base_limit = self.slip_limit
        
        # Adjust for volatility
        vol_level = context.get('volatility_level', 'medium')
        if vol_level == 'high':
            base_limit *= 2.0
        elif vol_level == 'extreme':
            base_limit *= 3.0
        elif vol_level == 'low':
            base_limit *= 0.7
        
        # Adjust for market regime
        regime = context.get('regime', 'unknown')
        if regime == 'volatile':
            base_limit *= 1.5
        elif regime == 'trending':
            base_limit *= 0.8
        
        # Adjust for session
        session = context.get('session', 'unknown')
        if session in ['asian', 'rollover']:
            base_limit *= 1.3  # Less liquidity
        
        return base_limit

    def _get_context_adjusted_latency_limit(self, context: Dict[str, Any]) -> float:
        """Get context-adjusted latency limit"""
        
        base_limit = self.latency_limit
        
        # Adjust for session
        session = context.get('session', 'unknown')
        if session == 'asian':
            base_limit *= 1.5  # Higher latency expected
        elif session == 'rollover':
            base_limit *= 1.3
        
        # Adjust for volatility
        vol_level = context.get('volatility_level', 'medium')
        if vol_level in ['high', 'extreme']:
            base_limit *= 1.4  # Higher latency during volatility
        
        return base_limit

    def _analyze_fill_rates(self, orders: List[Dict[str, Any]], context: Dict[str, Any]) -> None:
        """Analyze fill rates from order data"""
        
        try:
            if not orders:
                return
            
            filled_orders = 0
            total_orders = len(orders)
            
            for order in orders:
                fill_status = self._extract_fill_status(order)
                if fill_status == 'filled':
                    filled_orders += 1
                elif fill_status == 'partial':
                    filled_orders += 0.5  # Count partial fills as half
            
            fill_rate = filled_orders / total_orders if total_orders > 0 else 1.0
            self.fill_history.append(fill_rate)
            
            # Context-aware fill rate expectations
            expected_fill_rate = self._get_context_adjusted_fill_rate(context)
            
            if fill_rate < expected_fill_rate:
                issue = {
                    'fill_rate': fill_rate,
                    'expected': expected_fill_rate,
                    'filled_orders': filled_orders,
                    'total_orders': total_orders,
                    'context': context.copy(),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                self.issues["fill_rate"].append(issue)
                self.log_operator_warning(
                    f"📉 Low fill rate: {fill_rate:.1%} < {expected_fill_rate:.1%} ({filled_orders}/{total_orders})"
                )
            
        except Exception as e:
            self.log_operator_warning(f"Fill rate analysis failed: {e}")

    def _get_context_adjusted_fill_rate(self, context: Dict[str, Any]) -> float:
        """Get context-adjusted expected fill rate"""
        
        base_rate = self.min_fill_rate
        
        # Adjust for volatility
        vol_level = context.get('volatility_level', 'medium')
        if vol_level == 'extreme':
            base_rate -= 0.05  # Lower expectations during extreme volatility
        elif vol_level == 'high':
            base_rate -= 0.02
        
        # Adjust for session
        session = context.get('session', 'unknown')
        if session in ['asian', 'rollover']:
            base_rate -= 0.03  # Lower liquidity sessions
        
        return max(0.8, base_rate)  # Never go below 80%

    def _analyze_spread_data(self, spreads: Dict[str, float], context: Dict[str, Any]) -> None:
        """Analyze spread data with context awareness"""
        
        try:
            for instrument, spread in spreads.items():
                if spread is not None and spread > 0:
                    spread = float(spread)
                    self.spread_history.append(spread)
                    self.instrument_metrics[instrument]['spread'].append(spread)
                    
                    # Context-aware spread thresholds
                    threshold = self._get_context_adjusted_spread_threshold(instrument, context)
                    
                    if spread > threshold:
                        issue = {
                            'instrument': instrument,
                            'spread': spread,
                            'threshold': threshold,
                            'context': context.copy(),
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                        self.issues["spread"].append(issue)
            
        except Exception as e:
            self.log_operator_warning(f"Spread analysis failed: {e}")

    def _get_context_adjusted_spread_threshold(self, instrument: str, context: Dict[str, Any]) -> float:
        """Get context-adjusted spread threshold"""
        
        # Base thresholds by instrument type
        if 'XAU' in instrument or 'GOLD' in instrument:
            base_threshold = 1.0  # $1 for gold
        elif any(curr in instrument for curr in ['EUR', 'USD', 'GBP', 'JPY']):
            base_threshold = 0.0003  # 3 pips for major pairs
        else:
            base_threshold = self.spread_threshold
        
        # Adjust for volatility
        vol_level = context.get('volatility_level', 'medium')
        if vol_level == 'high':
            base_threshold *= 2.0
        elif vol_level == 'extreme':
            base_threshold *= 3.0
        
        # Adjust for session
        session = context.get('session', 'unknown')
        if session in ['asian', 'rollover']:
            base_threshold *= 1.5
        
        return base_threshold

    def _should_generate_training_data(self) -> bool:
        """Determine if we should generate training data"""
        
        return (
            self.training_mode and 
            len(self.slippage_history) < 10 and 
            self.execution_count < 5
        )

    def _generate_realistic_training_data(self, context: Dict[str, Any]) -> None:
        """Generate realistic execution data for training"""
        
        try:
            # Generate realistic slippage
            base_slippage = 0.0002
            
            # Adjust for context
            vol_level = context.get('volatility_level', 'medium')
            if vol_level == 'high':
                base_slippage *= 2.0
            elif vol_level == 'extreme':
                base_slippage *= 3.0
            
            session = context.get('session', 'unknown')
            if session in ['asian', 'rollover']:
                base_slippage *= 1.3
            
            # Generate with realistic distribution
            realistic_slippage = abs(np.random.gamma(2, base_slippage))
            self.slippage_history.append(realistic_slippage)
            
            # Generate realistic latency
            base_latency = 250  # 250ms base
            
            if session == 'asian':
                base_latency += 100
            if vol_level in ['high', 'extreme']:
                base_latency += 100
            
            realistic_latency = max(50, np.random.gamma(3, base_latency / 3))
            self.latency_history.append(realistic_latency)
            
            # Generate realistic fill rate
            base_fill_rate = 0.96
            
            if vol_level == 'extreme':
                base_fill_rate -= 0.05
            elif vol_level == 'high':
                base_fill_rate -= 0.02
            
            realistic_fill_rate = np.random.beta(20 * base_fill_rate, 20 * (1 - base_fill_rate))
            self.fill_history.append(realistic_fill_rate)
            
            # Generate realistic spread
            base_spread = 0.00015  # 1.5 pips
            
            if vol_level == 'high':
                base_spread *= 2.0
            elif vol_level == 'extreme':
                base_spread *= 3.0
            
            realistic_spread = abs(np.random.gamma(2, base_spread))
            self.spread_history.append(realistic_spread)
            
        except Exception as e:
            self.log_operator_warning(f"Training data generation failed: {e}")

    def _analyze_execution_quality_trends(self, context: Dict[str, Any]) -> None:
        """Analyze execution quality trends"""
        
        try:
            # Calculate current quality score
            self._calculate_comprehensive_quality_score()
            
            # Add to history
            self.quality_history.append(self.quality_score)
            
            # Analyze trends
            if len(self.quality_history) >= 10:
                recent_scores = list(self.quality_history)[-10:]
                older_scores = list(self.quality_history)[-20:-10] if len(self.quality_history) >= 20 else []
                
                current_avg = np.mean(recent_scores)
                previous_avg = np.mean(older_scores) if older_scores else current_avg
                
                self.comprehensive_metrics["quality_trend"] = current_avg - previous_avg
            
            # Update regime and session performance
            self._update_regime_session_performance(context)
            
        except Exception as e:
            self.log_operator_warning(f"Quality trend analysis failed: {e}")

    def _calculate_comprehensive_quality_score(self) -> None:
        """Calculate comprehensive execution quality score"""
        
        try:
            scores = []
            weights = []
            
            # Slippage score
            if self.slippage_history:
                avg_slippage = np.mean(list(self.slippage_history)[-20:])
                percentile_slippage = np.percentile(list(self.slippage_history), self.slippage_percentile)
                slippage_score = max(0, 1.0 - (percentile_slippage / (self.slip_limit * 2)))
                scores.append(slippage_score)
                weights.append(0.3)
            
            # Latency score
            if self.latency_history:
                avg_latency = np.mean(list(self.latency_history)[-20:])
                percentile_latency = np.percentile(list(self.latency_history), self.latency_percentile)
                latency_score = max(0, 1.0 - (percentile_latency / (self.latency_limit * 2)))
                scores.append(latency_score)
                weights.append(0.3)
            
            # Fill rate score
            if self.fill_history:
                avg_fill_rate = np.mean(list(self.fill_history)[-20:])
                fill_score = avg_fill_rate
                scores.append(fill_score)
                weights.append(0.25)
            
            # Spread score
            if self.spread_history:
                avg_spread = np.mean(list(self.spread_history)[-20:])
                spread_score = max(0, 1.0 - (avg_spread / (self.spread_threshold * 2)))
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
            if self.quality_score < self.degradation_threshold:
                self.degraded_executions += 1
            
        except Exception as e:
            self.log_operator_error(f"Quality score calculation failed: {e}")
            self.quality_score = 0.5  # Conservative fallback

    def _update_regime_session_performance(self, context: Dict[str, Any]) -> None:
        """Update regime and session performance tracking"""
        
        try:
            regime = context.get('regime', 'unknown')
            session = context.get('session', 'unknown')
            
            # Update regime performance
            if regime != 'unknown':
                regime_data = self.regime_performance[regime]
                regime_data['quality_scores'].append(self.quality_score)
                if self.slippage_history:
                    regime_data['avg_slippage'].append(np.mean(list(self.slippage_history)[-5:]))
                if self.latency_history:
                    regime_data['avg_latency'].append(np.mean(list(self.latency_history)[-5:]))
            
            # Update session performance
            if session != 'unknown':
                session_data = self.session_performance[session]
                session_data['quality_scores'].append(self.quality_score)
                if self.fill_history:
                    session_data['fill_rates'].append(np.mean(list(self.fill_history)[-5:]))
            
        except Exception as e:
            self.log_operator_warning(f"Regime/session performance update failed: {e}")

    def _check_quality_degradation_and_alerts(self, context: Dict[str, Any]) -> None:
        """Check for quality degradation and generate alerts"""
        
        try:
            # Check for significant quality degradation
            if self.quality_score < self.degradation_threshold:
                alert = {
                    'timestamp': datetime.datetime.now().isoformat(),
                    'quality_score': self.quality_score,
                    'threshold': self.degradation_threshold,
                    'context': context.copy(),
                    'issues': {k: len(v) for k, v in self.issues.items()},
                    'severity': 'critical' if self.quality_score < 0.3 else 'warning'
                }
                
                self.quality_alerts.append(alert)
                
                self.log_operator_error(
                    f"🚨 Execution quality degradation",
                    quality_score=f"{self.quality_score:.2f}",
                    threshold=f"{self.degradation_threshold:.2f}",
                    regime=context.get('regime', 'unknown')
                )
                
                # Check for escalation
                if (self.quality_score < 0.3 and 
                    (self.last_escalation is None or 
                     (datetime.datetime.now() - self.last_escalation).total_seconds() > 300)):  # 5 min cooldown
                    
                    self.escalation_count += 1
                    self.last_escalation = datetime.datetime.now()
                    
                    self.log_operator_error(
                        f"🚨 ESCALATION #{self.escalation_count}: Critical execution quality failure",
                        quality_score=f"{self.quality_score:.2f}",
                        degraded_executions=self.degraded_executions
                    )
            
        except Exception as e:
            self.log_operator_warning(f"Quality degradation check failed: {e}")

    def _update_comprehensive_metrics(self) -> None:
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
            
        except Exception as e:
            self.log_operator_warning(f"Comprehensive metrics update failed: {e}")

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with execution quality results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'execution_quality_monitor', {
            'quality_score': self.quality_score,
            'comprehensive_metrics': self.comprehensive_metrics.copy(),
            'current_issues': {k: len(v) for k, v in self.issues.items()},
            'execution_count': self.execution_count,
            'degraded_executions': self.degraded_executions,
            'escalation_count': self.escalation_count,
            'training_mode': self.training_mode,
            'market_regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'recent_quality_trend': self.comprehensive_metrics.get('quality_trend', 0.0)
        })
        
        # Update risk snapshot
        InfoBusUpdater.update_risk_snapshot(info_bus, {
            'execution_quality': self.quality_score,
            'execution_degraded': self.quality_score < self.degradation_threshold,
            'execution_issues': sum(len(issues) for issues in self.issues.values())
        })
        
        # Add alerts for quality issues
        total_issues = sum(len(issues) for issues in self.issues.values())
        
        if self.quality_score < 0.5:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Critical execution quality: {self.quality_score:.1%}",
                severity="critical",
                module="ExecutionQualityMonitor"
            )
        elif self.quality_score < 0.7:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Poor execution quality: {self.quality_score:.1%}",
                severity="warning",
                module="ExecutionQualityMonitor"
            )
        
        if total_issues > 5:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Multiple execution issues: {total_issues} problems detected",
                severity="warning",
                module="ExecutionQualityMonitor"
            )

    def _record_execution_audit(self, info_bus: InfoBus, context: Dict[str, Any], 
                               execution_data: Dict[str, Any]) -> None:
        """Record comprehensive audit trail"""
        
        # Only audit when there are issues or periodically
        should_audit = (
            sum(len(issues) for issues in self.issues.values()) > 0 or
            self.quality_score < 0.8 or
            len(self.quality_alerts) > 0 or
            info_bus.get('step_idx', 0) % 100 == 0
        )
        
        if should_audit:
            audit_data = {
                'quality_metrics': {
                    'quality_score': self.quality_score,
                    'execution_count': self.execution_count,
                    'degraded_executions': self.degraded_executions,
                    'success_rate': self.comprehensive_metrics['success_rate']
                },
                'comprehensive_metrics': self.comprehensive_metrics.copy(),
                'current_issues': {k: len(v) for k, v in self.issues.items()},
                'market_context': context.copy(),
                'execution_summary': {
                    'total_executions': len(execution_data.get('executions', [])),
                    'total_orders': len(execution_data.get('orders', [])),
                    'spreads_analyzed': len(execution_data.get('spreads', {}))
                },
                'alerts': len(self.quality_alerts),
                'escalations': self.escalation_count,
                'training_mode': self.training_mode
            }
            
            severity = "critical" if self.quality_score < 0.5 else "warning" if self.quality_score < 0.8 else "info"
            
            self.audit_tracker.record_event(
                event_type="execution_quality_assessment",
                module="ExecutionQualityMonitor",
                details=audit_data,
                severity=severity
            )

    def _update_execution_performance_metrics(self) -> None:
        """Update performance metrics"""
        
        # Update performance metrics
        self._update_performance_metric('quality_score', self.quality_score)
        self._update_performance_metric('execution_count', self.execution_count)
        self._update_performance_metric('degraded_executions', self.degraded_executions)
        self._update_performance_metric('escalation_count', self.escalation_count)
        
        # Update comprehensive metrics
        for metric_name, metric_value in self.comprehensive_metrics.items():
            self._update_performance_metric(f'exec_{metric_name}', metric_value)

    def _process_legacy_step(self, **kwargs) -> None:
        """Process legacy step parameters for backward compatibility"""
        
        try:
            # Process legacy execution data
            trade_executions = kwargs.get('trade_executions', kwargs.get('trades', []))
            order_attempts = kwargs.get('order_attempts', kwargs.get('orders', []))
            spread_data = kwargs.get('spread_data', {})
            
            if trade_executions:
                for execution in trade_executions:
                    self._analyze_single_execution(execution, {'regime': 'unknown'})
                    self.execution_count += 1
            
            if order_attempts:
                self._analyze_fill_rates(order_attempts, {'regime': 'unknown'})
            
            if spread_data:
                self._analyze_spread_data(spread_data, {'regime': 'unknown'})
            
            # Calculate quality score
            self._calculate_comprehensive_quality_score()
            
            # Generate training data if needed
            if self.training_mode and self._should_generate_training_data():
                self._generate_realistic_training_data({'regime': 'unknown'})
            
        except Exception as e:
            self.log_operator_error(f"Legacy step processing failed: {e}")

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get detailed execution statistics"""
        
        stats = {}
        
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
                    "violations": len([s for s in slips if s > self.slip_limit])
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
                    "violations": len([l for l in latencies if l > self.latency_limit])
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
                    "below_threshold_count": len([f for f in fills if f < self.min_fill_rate]),
                    "count": len(fills)
                }
            
            # Spread statistics
            if self.spread_history:
                spreads = list(self.spread_history)
                stats["spread"] = {
                    "mean": float(np.mean(spreads)),
                    "std": float(np.std(spreads)),
                    "max": float(np.max(spreads)),
                    "min": float(np.min(spreads)),
                    "p95": float(np.percentile(spreads, 95)),
                    "count": len(spreads),
                    "violations": len([s for s in spreads if s > self.spread_threshold])
                }
            
            # Quality statistics
            if self.quality_history:
                qualities = list(self.quality_history)
                stats["quality"] = {
                    "current": self.quality_score,
                    "mean": float(np.mean(qualities)),
                    "trend": self.comprehensive_metrics.get('quality_trend', 0.0),
                    "degraded_count": self.degraded_executions,
                    "degradation_rate": self.comprehensive_metrics.get('degradation_rate', 0.0)
                }
            
        except Exception as e:
            self.log_operator_error(f"Error calculating execution stats: {e}")
        
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
                recent_slippage = np.mean(list(self.slippage_history)[-10:])
            if self.latency_history:
                recent_latency = np.mean(list(self.latency_history)[-10:])
            if self.fill_history:
                recent_fill_rate = np.mean(list(self.fill_history)[-10:])
            if self.spread_history:
                recent_spread = np.mean(list(self.spread_history)[-10:])
            
            return np.array([
                float(self.quality_score),
                has_issues,
                float(np.clip(recent_slippage / max(self.slip_limit, 1e-8), 0.0, 10.0)),
                float(np.clip(recent_latency / max(self.latency_limit, 1), 0.0, 10.0)),
                float(np.clip(recent_fill_rate, 0.0, 1.0)),
                float(np.clip(recent_spread / max(self.spread_threshold, 1e-8), 0.0, 10.0)),
                float(self.degraded_executions / max(self.execution_count, 1)),
                float(self.escalation_count / 10.0)  # Normalize escalations
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_execution_quality_report(self) -> str:
        """Generate operator-friendly execution quality report"""
        
        # Quality status indicators
        if self.quality_score < 0.5:
            quality_status = "🚨 Critical"
        elif self.quality_score < 0.7:
            quality_status = "⚠️ Poor"
        elif self.quality_score < 0.9:
            quality_status = "⚡ Good"
        else:
            quality_status = "✅ Excellent"
        
        # Issue summary
        issue_lines = []
        for issue_type, issues in self.issues.items():
            if issues:
                count = len(issues)
                emoji = "🚨" if count > 3 else "⚠️" if count > 1 else "⚡"
                issue_lines.append(f"  {emoji} {issue_type.replace('_', ' ').title()}: {count} issues")
        
        # Recent alerts
        alert_lines = []
        for alert in list(self.quality_alerts)[-3:]:
            timestamp = alert['timestamp'][:19]
            severity = alert['severity']
            emoji = "🚨" if severity == 'critical' else "⚠️"
            alert_lines.append(f"  {emoji} {timestamp}: Quality {alert['quality_score']:.1%}")
        
        # Instrument performance
        instrument_lines = []
        for instrument, metrics in list(self.instrument_metrics.items())[:5]:  # Show top 5
            if metrics['slippage']:
                avg_slip = np.mean(list(metrics['slippage'])[-5:])
                avg_latency = np.mean(list(metrics['latency'])[-5:]) if metrics['latency'] else 0
                
                if avg_slip > self.slip_limit * 1.5:
                    emoji = "🔴"
                elif avg_slip > self.slip_limit:
                    emoji = "🟡"
                else:
                    emoji = "🟢"
                
                instrument_lines.append(
                    f"  {emoji} {instrument}: slip {avg_slip:.5f}, latency {avg_latency:.0f}ms"
                )
        
        return f"""
⚡ EXECUTION QUALITY MONITOR
═══════════════════════════════════════
🎯 Quality Status: {quality_status} ({self.quality_score:.1%})
📊 Market Regime: {self.market_regime.title()}
💥 Volatility Level: {self.volatility_regime.title()}
🕐 Market Session: {self.market_session.title()}
🎓 Training Mode: {'✅ Active' if self.training_mode else '❌ Inactive'}

📊 EXECUTION METRICS
• Quality Score: {self.quality_score:.1%}
• Total Executions: {self.execution_count:,}
• Success Rate: {self.comprehensive_metrics['success_rate']:.1%}
• Degraded Executions: {self.degraded_executions}
• Degradation Rate: {self.comprehensive_metrics['degradation_rate']:.1%}

📈 PERFORMANCE STATISTICS
• Avg Slippage: {self.comprehensive_metrics['avg_slippage']:.5f} (limit: {self.slip_limit:.5f})
• Avg Latency: {self.comprehensive_metrics['avg_latency']:.0f}ms (limit: {self.latency_limit}ms)
• Avg Fill Rate: {self.comprehensive_metrics['avg_fill_rate']:.1%} (min: {self.min_fill_rate:.1%})
• Avg Spread: {self.comprehensive_metrics['avg_spread']:.5f}
• Quality Trend: {self.comprehensive_metrics['quality_trend']:+.2f}

⚠️ CURRENT ISSUES
{chr(10).join(issue_lines) if issue_lines else "  ✅ No current execution issues"}

🚨 ESCALATIONS & ALERTS
• Escalation Count: {self.escalation_count}
• Quality Alerts: {len(self.quality_alerts)}
• Last Escalation: {self.last_escalation.strftime('%H:%M:%S') if self.last_escalation else 'None'}

📋 INSTRUMENT PERFORMANCE
{chr(10).join(instrument_lines) if instrument_lines else "  📭 No instrument data available"}

📜 RECENT QUALITY ALERTS
{chr(10).join(alert_lines) if alert_lines else "  📭 No recent quality alerts"}

💡 THRESHOLDS & LIMITS
• Slippage Limit: {self.slip_limit:.5f}
• Latency Limit: {self.latency_limit}ms
• Min Fill Rate: {self.min_fill_rate:.1%}
• Spread Threshold: {self.spread_threshold:.5f}
• Quality Threshold: {self.quality_threshold:.1%}
• Degradation Threshold: {self.degradation_threshold:.1%}

🔧 SYSTEM STATUS
• Statistics Window: {self.stats_window} records
• Slippage History: {len(self.slippage_history)} records
• Latency History: {len(self.latency_history)} records
• Fill Rate History: {len(self.fill_history)} records
• Quality History: {len(self.quality_history)} records
        """

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        self._process_legacy_step(**kwargs)

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for legacy compatibility"""
        return {
            "limits": {
                "slip_limit": self.slip_limit,
                "latency_limit": self.latency_limit,
                "min_fill_rate": self.min_fill_rate
            },
            "training_mode": self.training_mode,
            "execution_count": self.execution_count,
            "quality_score": float(self.quality_score),
            "comprehensive_metrics": self.comprehensive_metrics.copy(),
            "statistics": self.get_execution_stats(),
            "history_sizes": {
                "slippage": len(self.slippage_history),
                "latency": len(self.latency_history),
                "fill_rate": len(self.fill_history),
                "spread": len(self.spread_history),
                "quality": len(self.quality_history)
            },
            "current_issues": {k: len(v) for k, v in self.issues.items()},
            "audit_summary": {
                "quality_alerts": len(self.quality_alerts),
                "escalation_count": self.escalation_count,
                "degraded_executions": self.degraded_executions
            }
        }