import time
import datetime
from typing import Dict, Any
from modules.utils.audit_utils import format_operator_message

class TradeMapVisualizer:
    def __init__(self, smart_bus, error_pinpointer, logger, performance_tracker, circuit_breaker_threshold=5):
        self.smart_bus = smart_bus
        self.error_pinpointer = error_pinpointer
        self.logger = logger
        self.performance_tracker = performance_tracker
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.is_disabled = False
        self.error_count = 0
        self._chart_cache = {}
        self._chart_history = []
        self.viz_stats = {
            'charts_generated': 0,
            'trade_charts': 0,
            'performance_charts': 0,
            'dashboard_charts': 0
        }

    async def process(self, **inputs) -> Dict[str, Any]:
        """Modern async processing with comprehensive visualization"""
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive market data from SmartInfoBus
            market_data = await self._get_comprehensive_market_data()
            
            # Generate visualizations if conditions are met
            chart_results = await self._generate_comprehensive_charts(market_data)

            # Build required provides for contract compliance
            thesis = f"Generated {chart_results.get('charts_generated', 0)} charts: {', '.join(chart_results.get('chart_types', []))}"
            chart_data = chart_results.get('chart_data', {})
            chart_results.update({
                'chart_history': self._chart_history[-20:],
                'chart_statistics': {
                    'charts_generated': self.viz_stats.get('charts_generated', 0),
                    'trade_charts': self.viz_stats.get('trade_charts', 0),
                    'performance_charts': self.viz_stats.get('performance_charts', 0),
                    'dashboard_charts': self.viz_stats.get('dashboard_charts', 0),
                    'last_generation': datetime.datetime.now().isoformat()
                },
                'trade_charts': chart_data.get('trade_map', {}) or {},
                'performance_charts': chart_data.get('performance', {}) or {},
                'dashboard_charts': chart_data.get('dashboard', {}) or {},
                '_thesis': thesis,
            })
            # Ensure keys required by contract exist even if generator omitted them
            chart_results.setdefault('chart_cache', dict(self._chart_cache))
            chart_results.setdefault('visualization_reports', [])
            
            # Update SmartInfoBus with visualization data
            await self._update_smartinfobus_comprehensive(chart_results)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('TradeMapVisualizer', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return chart_results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_market_data(self):
        """Placeholder for method to get market data"""
        return {}

    async def _generate_comprehensive_charts(self, market_data):
        """Placeholder for method to generate charts"""
        return {
            'charts_generated': 1,
            'chart_types': ['line'],
            'chart_data': {}
        }

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any]):
        """Update SmartInfoBus with comprehensive visualization data"""
        try:
            thesis = results.get('_thesis') or f"Generated {results.get('charts_generated', 0)} charts: {', '.join(results.get('chart_types', []))}"
            
            # Publish individual provided keys
            self.smart_bus.set('trade_charts', results.get('trade_charts', {}), module='TradeMapVisualizer', thesis=thesis)
            self.smart_bus.set('performance_charts', results.get('performance_charts', {}), module='TradeMapVisualizer', thesis=thesis)
            self.smart_bus.set('dashboard_charts', results.get('dashboard_charts', {}), module='TradeMapVisualizer', thesis=thesis)
            self.smart_bus.set('chart_statistics', results.get('chart_statistics', {}), module='TradeMapVisualizer', thesis=thesis)
            self.smart_bus.set('chart_cache', results.get('chart_cache', {}), module='TradeMapVisualizer', thesis=thesis)
            self.smart_bus.set('chart_history', results.get('chart_history', []), module='TradeMapVisualizer', thesis=thesis)
            self.smart_bus.set('visualization_reports', results.get('visualization_reports', []), module='TradeMapVisualizer', thesis=thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.warning(f"SmartInfoBus update failed: {error_context}")

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "TradeMapVisualizer")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]",
                message="TradeMapVisualizer disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        thesis = f"TradeMapVisualizer encountered an error and applied safe fallbacks: {error_context}"
        return {
            'charts_generated': 0,
            'error': str(error_context),
            'status': 'error',
            'chart_cache': dict(self._chart_cache),
            'chart_history': self._chart_history[-20:],
            'chart_statistics': {
                'charts_generated': self.viz_stats.get('charts_generated', 0),
                'trade_charts': self.viz_stats.get('trade_charts', 0),
                'performance_charts': self.viz_stats.get('performance_charts', 0),
                'dashboard_charts': self.viz_stats.get('dashboard_charts', 0),
                'last_generation': datetime.datetime.now().isoformat()
            },
            'trade_charts': {},
            'performance_charts': {},
            'dashboard_charts': {},
            'visualization_reports': [],
            '_thesis': thesis,
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'charts_generated': 0,
            'status': 'disabled',
            'reason': 'circuit_breaker_triggered',
            'chart_cache': dict(self._chart_cache),
            'chart_history': self._chart_history[-20:],
            'chart_statistics': {
                'charts_generated': self.viz_stats.get('charts_generated', 0),
                'trade_charts': self.viz_stats.get('trade_charts', 0),
                'performance_charts': self.viz_stats.get('performance_charts', 0),
                'dashboard_charts': self.viz_stats.get('dashboard_charts', 0),
                'last_generation': datetime.datetime.now().isoformat()
            },
            'trade_charts': {},
            'performance_charts': {},
            'dashboard_charts': {},
            'visualization_reports': [],
            '_thesis': 'TradeMapVisualizer is temporarily disabled due to repeated errors (circuit breaker). It will remain in a safe state until manual intervention or auto-recovery.'
        }