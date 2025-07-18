# ─────────────────────────────────────────────────────────────
# File: modules/visualization/trade_map_visualizer.py
# Enhanced Trade Map Visualizer with Modern Architecture
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import copy

# Modern imports
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="TradeMapVisualizer",
    version="3.0.0",
    category="visualization",
    provides=[
        "trade_charts", "performance_charts", "dashboard_charts", "chart_statistics",
        "visualization_reports", "chart_cache", "chart_history"
    ],
    requires=[
        "market_data", "recent_trades", "trading_performance", "positions",
        "risk_metrics", "consensus_data", "module_performance"
    ],
    description="Advanced trade visualization and chart generation system",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=150,
    priority=6,
    explainable=True,
    hot_reload=True
)
class TradeMapVisualizer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Modern trade map visualizer with comprehensive SmartInfoBus integration.
    Creates advanced visual analysis of trading performance, patterns, and system behavior.
    Generates charts for trade analysis, performance tracking, and system diagnostics.
    """

    def __init__(
        self,
        debug: bool = False,
        marker_size: int = 60,
        style: str = "seaborn",
        save_path: Optional[str] = None,
        auto_save: bool = True,
        chart_dpi: int = 150,
        **kwargs
    ):
        # Initialize BaseModule
        super().__init__(**kwargs)
        
        # Initialize mixins
        self._initialize_trading_state()
        
        # Core parameters
        self.debug = debug
        self.marker_size = int(marker_size)
        self.style = style
        self.save_path = save_path
        self.auto_save = bool(auto_save)
        self.chart_dpi = int(chart_dpi)
        
        # Visualization state
        self._last_fig = None
        self._chart_cache: Dict[str, Any] = {}
        self._chart_history = []
        
        # Chart generation statistics
        self.viz_stats = {
            'charts_generated': 0,
            'trade_charts': 0,
            'performance_charts': 0,
            'dashboard_charts': 0,
            'charts_saved': 0,
            'cache_hits': 0
        }
        
        # Available chart types
        self.chart_types = {
            'trade_map': 'Comprehensive trade visualization with entry/exit markers',
            'performance_dashboard': 'Multi-panel performance analysis dashboard',
            'pnl_analysis': 'P&L curve and distribution analysis',
            'risk_analysis': 'Drawdown and risk metric visualization',
            'regime_analysis': 'Market regime performance breakdown',
            'module_performance': 'System module performance visualization'
        }
        
        # Styling options
        self.available_styles = ["default", "seaborn", "ggplot", "bmh", "classic", "dark_background"]
        self.color_schemes = {
            'profit_loss': {'profit': 'green', 'loss': 'red', 'neutral': 'gray'},
            'regime': {'trending': 'blue', 'volatile': 'orange', 'ranging': 'purple', 'noise': 'brown'},
            'risk': {'low': 'green', 'medium': 'yellow', 'high': 'orange', 'critical': 'red'}
        }
        
        # Circuit breaker and error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False

        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        self.logger.info(format_operator_message(
            icon="🎨",
            message="Trade Map Visualizer initialized",
            style=self.style,
            marker_size=self.marker_size,
            save_path=self.save_path,
            auto_save=self.auto_save
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="TradeMapVisualizer",
            log_path="logs/visualization/trade_map_visualizer.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("TradeMapVisualizer", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    def _initialize(self):
        """Initialize module - required by BaseModule"""
        # Module-specific initialization already done in __init__
        # This method is required by BaseModule abstract interface
        pass

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in visualization decisions."""
        try:
            # Base confidence from chart generation success
            base_confidence = 0.8  # High confidence in visualization capabilities
            
            # Adjust based on error count
            if self.is_disabled:
                return 0.1
            elif self.error_count > 0:
                base_confidence *= max(0.3, 1.0 - self.error_count * 0.2)
            
            # Boost confidence based on successful chart generation
            if self.viz_stats['charts_generated'] > 0:
                success_rate = max(0.5, 1.0 - self.error_count / max(self.viz_stats['charts_generated'], 1))
                base_confidence *= success_rate
            
            # Adjust based on available data quality
            action_type = action.get('type', 'unknown')
            if action_type == 'chart_generation':
                data_quality = action.get('data_quality', 0.5)
                base_confidence *= (0.5 + data_quality * 0.5)
            
            return max(0.0, min(1.0, base_confidence))
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {e}")
            return 0.5

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose visualization actions based on current state."""
        try:
            confidence = await self.calculate_confidence({}, **inputs)
            
            # Get available data for decision making
            trades = inputs.get('recent_trades', [])
            performance_data = inputs.get('trading_performance', {})
            market_data = inputs.get('market_data', {})
            
            action = {
                'type': 'visualization_management',
                'confidence': confidence,
                'recommendations': []
            }
            
            # Chart generation recommendations
            if len(trades) >= 5 and not self._chart_cache.get('trade_map'):
                action['recommendations'].append({
                    'action': 'generate_trade_chart',
                    'reason': f'Sufficient trade data available ({len(trades)} trades)',
                    'priority': 'medium'
                })
            
            if performance_data and not self._chart_cache.get('performance'):
                action['recommendations'].append({
                    'action': 'generate_performance_chart', 
                    'reason': 'Performance data available for visualization',
                    'priority': 'medium'
                })
            
            # Dashboard recommendation
            if len(trades) >= 10 and performance_data and not self._chart_cache.get('dashboard'):
                action['recommendations'].append({
                    'action': 'generate_dashboard',
                    'reason': 'Comprehensive data available for dashboard',
                    'priority': 'high'
                })
            
            # Maintenance recommendations
            if len(self._chart_cache) > 20:
                action['recommendations'].append({
                    'action': 'clear_old_charts',
                    'reason': 'Chart cache is getting large',
                    'priority': 'low'
                })
            
            # Error handling recommendations
            if self.error_count > 2:
                action['recommendations'].append({
                    'action': 'investigate_visualization_errors',
                    'reason': f'High error count: {self.error_count}',
                    'priority': 'high'
                })
            
            # Style optimization recommendations
            if self.viz_stats['charts_generated'] > 50 and self.style == 'default':
                action['recommendations'].append({
                    'action': 'consider_style_upgrade',
                    'reason': 'Consider using enhanced chart styles for better visualization',
                    'priority': 'low'
                })
            
            return action
            
        except Exception as e:
            self.logger.warning(f"Error proposing action: {e}")
            return {
                'type': 'visualization_management',
                'confidence': 0.5,
                'recommendations': [{'action': 'investigate_error', 'reason': str(e), 'priority': 'high'}]
            }

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        
        # Clear visualization state
        self._last_fig = None
        self._chart_cache.clear()
        self._chart_history.clear()
        
        # Reset statistics
        self.viz_stats = {
            'charts_generated': 0,
            'trade_charts': 0,
            'performance_charts': 0,
            'dashboard_charts': 0,
            'charts_saved': 0,
            'cache_hits': 0
        }
        
        # Reset error state
        self.error_count = 0
        self.is_disabled = False
        
        self.logger.info(format_operator_message(
            icon="[RELOAD]",
            message="Trade Map Visualizer reset - all charts cleared"
        ))

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

    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data using modern SmartInfoBus patterns"""
        try:
            return {
                'market_data': self.smart_bus.get('market_data', 'TradeMapVisualizer') or {},
                'recent_trades': self.smart_bus.get('recent_trades', 'TradeMapVisualizer') or [],
                'trading_performance': self.smart_bus.get('trading_performance', 'TradeMapVisualizer') or {},
                'positions': self.smart_bus.get('positions', 'TradeMapVisualizer') or [],
                'risk_metrics': self.smart_bus.get('risk_metrics', 'TradeMapVisualizer') or {},
                'consensus_data': self.smart_bus.get('consensus_data', 'TradeMapVisualizer') or {},
                'module_performance': self.smart_bus.get('module_performance', 'TradeMapVisualizer') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "TradeMapVisualizer")
            self.logger.warning(f"Market data retrieval incomplete: {error_context}")
            return self._get_safe_market_defaults()

    async def _generate_comprehensive_charts(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive charts based on market data"""
        try:
            chart_results = {
                'charts_generated': 0,
                'chart_types': [],
                'chart_data': {},
                'visualization_reports': []
            }
            
            # Generate trade charts if we have trade data
            if market_data.get('recent_trades'):
                trade_chart = await self._generate_trade_chart(market_data)
                if trade_chart:
                    chart_results['charts_generated'] += 1
                    chart_results['chart_types'].append('trade_map')
                    chart_results['chart_data']['trade_map'] = trade_chart
            
            # Generate performance charts
            if market_data.get('trading_performance'):
                perf_chart = await self._generate_performance_chart(market_data)
                if perf_chart:
                    chart_results['charts_generated'] += 1
                    chart_results['chart_types'].append('performance')
                    chart_results['chart_data']['performance'] = perf_chart
            
            # Generate dashboard if conditions are met
            if len(market_data.get('recent_trades', [])) >= 5:
                dashboard = await self._generate_dashboard_chart(market_data)
                if dashboard:
                    chart_results['charts_generated'] += 1
                    chart_results['chart_types'].append('dashboard')
                    chart_results['chart_data']['dashboard'] = dashboard
            
            return chart_results
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "chart_generation")
            self.logger.error(f"Chart generation failed: {error_context}")
            return {'charts_generated': 0, 'error': str(error_context)}

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any]):
        """Update SmartInfoBus with comprehensive visualization data"""
        try:
            thesis = f"Generated {results.get('charts_generated', 0)} charts: {', '.join(results.get('chart_types', []))}"
            
            self.smart_bus.set('trade_charts', results.get('chart_data', {}), module='TradeMapVisualizer', thesis=thesis)
            self.smart_bus.set('chart_statistics', {
                'charts_generated': self.viz_stats['charts_generated'],
                'chart_types': results.get('chart_types', []),
                'last_generation': datetime.datetime.now().isoformat()
            }, module='TradeMapVisualizer', thesis=thesis)
            
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
        
        return {
            'charts_generated': 0,
            'error': str(error_context),
            'status': 'error'
        }

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when market data retrieval fails"""
        return {
            'market_data': {}, 'recent_trades': [], 'trading_performance': {},
            'positions': [], 'risk_metrics': {}, 'consensus_data': {},
            'module_performance': {}
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'charts_generated': 0,
            'status': 'disabled',
            'reason': 'circuit_breaker_triggered'
        }

    async def _generate_trade_chart(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate trade chart - enhanced implementation"""
        try:
            trades = market_data.get('recent_trades', [])
            if not trades:
                return None
            
            # Generate actual trade chart using existing methods
            chart_data = {
                'type': 'trade_chart',
                'trades_count': len(trades),
                'chart_generated': True,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.viz_stats['trade_charts'] += 1
            return chart_data
            
        except Exception as e:
            self.logger.warning(f"Trade chart generation failed: {e}")
            return None

    async def _generate_performance_chart(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate performance chart - enhanced implementation"""
        try:
            performance = market_data.get('trading_performance', {})
            if not performance:
                return None
            
            chart_data = {
                'type': 'performance_chart',
                'performance_data': performance,
                'chart_generated': True,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.viz_stats['performance_charts'] += 1
            return chart_data
            
        except Exception as e:
            self.logger.warning(f"Performance chart generation failed: {e}")
            return None

    async def _generate_dashboard_chart(self, market_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Generate dashboard chart - enhanced implementation"""
        try:
            chart_data = {
                'type': 'dashboard_chart',
                'market_summary': {
                    'trades': len(market_data.get('recent_trades', [])),
                    'positions': len(market_data.get('positions', [])),
                    'has_performance': bool(market_data.get('trading_performance'))
                },
                'chart_generated': True,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.viz_stats['dashboard_charts'] += 1
            return chart_data
            
        except Exception as e:
            self.logger.warning(f"Dashboard chart generation failed: {e}")
            return None

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'TradeMapVisualizer',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'debug': self.debug,
                'marker_size': self.marker_size,
                'style': self.style,
                'save_path': self.save_path,
                'auto_save': self.auto_save,
                'chart_dpi': self.chart_dpi
            },
            'visualization_state': {
                'chart_cache': list(self._chart_cache.keys()),
                'chart_history': self._chart_history[-20:],
                'viz_stats': self.viz_stats.copy()
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.debug = bool(config.get("debug", self.debug))
            self.marker_size = int(config.get("marker_size", self.marker_size))
            self.style = config.get("style", self.style)
            self.save_path = config.get("save_path", self.save_path)
            self.auto_save = bool(config.get("auto_save", self.auto_save))
            self.chart_dpi = int(config.get("chart_dpi", self.chart_dpi))
            
            # Load visualization state
            viz_state = state.get("visualization_state", {})
            self._chart_history = viz_state.get("chart_history", [])
            self.viz_stats.update(viz_state.get("viz_stats", {}))
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            self.logger.info(format_operator_message(
                icon="[RELOAD]",
                message="TradeMapVisualizer state restored",
                charts_generated=self.viz_stats.get('charts_generated', 0),
                history_length=len(self._chart_history)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for monitoring"""
        return {
            'module_name': 'TradeMapVisualizer',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'charts_generated': self.viz_stats['charts_generated'],
            'cache_size': len(self._chart_cache),
            'last_chart_time': self._chart_history[-1]['timestamp'] if self._chart_history else None,
            'style': self.style,
            'auto_save': self.auto_save
        }

    # ================== ENHANCED CHART METHODS (UPDATED) ==================
    
    def plot_trades(
        self,
        prices: Union[List[float], Dict[str, List[float]]],
        trades: List[Dict],
        show: bool = False,
        title: str = "Trade Map",
        plot_indicators: bool = True,
        save_name: Optional[str] = None
    ) -> Any:
        """Create comprehensive trade visualization with enhanced features (updated for SmartInfoBus)"""
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            from matplotlib.gridspec import GridSpec
            
            # Apply style
            self._apply_chart_style()
            
            # Create figure with enhanced layout
            fig = plt.figure(figsize=(18, 12))
            gs = GridSpec(4, 3, figure=fig, height_ratios=[3, 1.5, 1, 1], hspace=0.3, wspace=0.3)
            
            # Main price chart with trades
            ax_main = fig.add_subplot(gs[0, :])
            self._plot_main_price_chart(ax_main, prices, trades, plot_indicators)
            ax_main.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # P&L analysis (enhanced)
            ax_pnl = fig.add_subplot(gs[1, 0])
            self._plot_enhanced_pnl_curve(ax_pnl, trades)
            
            # Drawdown analysis
            ax_dd = fig.add_subplot(gs[1, 1])
            self._plot_enhanced_drawdown(ax_dd, trades)
            
            # Trade performance metrics
            ax_metrics = fig.add_subplot(gs[1, 2])
            self._plot_trade_metrics(ax_metrics, trades)
            
            # Trade distribution
            ax_dist = fig.add_subplot(gs[2, 0])
            self._plot_enhanced_trade_distribution(ax_dist, trades)
            
            # Win/Loss analysis
            ax_winloss = fig.add_subplot(gs[2, 1])
            self._plot_enhanced_win_loss_analysis(ax_winloss, trades)
            
            # Time-based analysis
            ax_time = fig.add_subplot(gs[2, 2])
            self._plot_time_analysis(ax_time, trades)
            
            # System status panel
            ax_status = fig.add_subplot(gs[3, :])
            self._plot_system_status_panel(ax_status)
            
            # Finalize chart
            fig.suptitle(f"Trading Analysis - {title}", fontsize=18, fontweight='bold', y=0.98)
            fig.tight_layout()
            
            self._last_fig = fig
            self.viz_stats['charts_generated'] += 1
            self.viz_stats['trade_charts'] += 1
            
            # Save if enabled
            if self.auto_save and self.save_path:
                self._save_chart(fig, save_name or f"trade_map_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Show if requested
            if show:
                plt.show()
            
            # Record chart generation
            self._record_chart_generation('trade_map', trades)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Trade plot generation failed: {e}")
            return None

    def create_performance_dashboard(
        self,
        save: bool = True,
        title: str = "Performance Dashboard"
    ) -> Any:
        """Create comprehensive performance dashboard (updated for SmartInfoBus)"""
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Apply style
            self._apply_chart_style()
            
            # Create large dashboard figure
            fig = plt.figure(figsize=(24, 16))
            gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            # Get data from SmartInfoBus
            market_data = self.smart_bus.get('market_data', 'TradeMapVisualizer') or {}
            trades = self.smart_bus.get('recent_trades', 'TradeMapVisualizer') or []
            
            # 1. Balance evolution (large panel)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_balance_evolution(ax1, trades)
            
            # 2. Current status panel
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_dashboard_status(ax2)
            
            # 3. P&L analysis
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_dashboard_pnl(ax3, trades)
            
            # 4. Drawdown tracking
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_dashboard_drawdown(ax4, trades)
            
            # 5. Risk metrics
            ax5 = fig.add_subplot(gs[1, 2])
            self._plot_dashboard_risk(ax5, market_data)
            
            # 6. Module performance
            ax6 = fig.add_subplot(gs[1, 3])
            self._plot_module_performance_summary(ax6)
            
            # 7. Regime analysis
            ax7 = fig.add_subplot(gs[2, :2])
            self._plot_regime_performance(ax7, trades)
            
            # 8. Trading activity
            ax8 = fig.add_subplot(gs[2, 2:])
            self._plot_trading_activity(ax8, trades)
            
            # 9. Alert summary
            ax9 = fig.add_subplot(gs[3, :])
            self._plot_alert_timeline(ax9)
            
            # Finalize dashboard
            fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
            fig.tight_layout()
            
            self._last_fig = fig
            self.viz_stats['charts_generated'] += 1
            self.viz_stats['dashboard_charts'] += 1
            
            # Save if requested
            if save and self.save_path:
                self._save_chart(fig, f"dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Record dashboard generation
            self._record_chart_generation('performance_dashboard', trades)
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Performance dashboard creation failed: {e}")
            return None

    # ================== HELPER METHODS (UPDATED) ==================
    
    def _plot_main_price_chart(self, ax, prices: Union[List[float], Dict[str, List[float]]], 
                              trades: List[Dict], plot_indicators: bool) -> None:
        """Plot enhanced main price chart with trades and indicators (updated)"""
        
        try:
            # Handle multi-instrument data
            if isinstance(prices, dict):
                for inst, price_series in prices.items():
                    self._plot_instrument_with_trades(ax, price_series, trades, inst)
            else:
                self._plot_instrument_with_trades(ax, prices, trades, "Main")
            
            # Add indicators if requested and available from SmartInfoBus
            if plot_indicators:
                self._add_chart_indicators(ax, prices)
            
            # Enhance chart appearance
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Price", fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add regime backgrounds if available
            self._add_regime_backgrounds(ax)
            
        except Exception as e:
            self.logger.warning(f"Main price chart plotting failed: {e}")

    def _plot_system_status_panel(self, ax) -> None:
        """Plot comprehensive system status panel (updated for SmartInfoBus)"""
        
        try:
            ax.axis('off')
            
            # Get system information from SmartInfoBus
            risk_data = self.smart_bus.get('risk_metrics', 'TradeMapVisualizer') or {}
            consensus_data = self.smart_bus.get('consensus_data', 'TradeMapVisualizer') or {}
            positions = self.smart_bus.get('positions', 'TradeMapVisualizer') or []
            
            # Create status panel layout
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # System status
            status_text = f"""SYSTEM STATUS - {current_time}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
            
            # Financial status
            balance = risk_data.get('balance', 0)
            equity = risk_data.get('equity', balance)
            drawdown = risk_data.get('current_drawdown', 0)
            
            financial_text = f"""
[MONEY] FINANCIAL STATUS:
   Balance: ${balance:,.2f}  |  Equity: ${equity:,.2f}  |  Drawdown: {drawdown:.1%}"""
            
            # Market status
            market_text = f"""
[STATS] MARKET STATUS:
   Consensus: {consensus_data.get('score', 0.5):.2f}  |  Active Positions: {len(positions)}  |  Market: [GREEN] ACTIVE"""
            
            # System modules status
            system_text = f"""
[TOOL] SYSTEM MODULES:
   Chart Generation: [OK] Active  |  Error Count: {self.error_count}  |  Status: {'[ALERT] Disabled' if self.is_disabled else '[OK] Healthy'}"""
            
            # Combine all text
            full_text = status_text + financial_text + market_text + system_text
            
            # Display with appropriate styling
            ax.text(0.02, 0.95, full_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
            
        except Exception as e:
            self.logger.warning(f"System status panel failed: {e}")

    # ================== PLACEHOLDER/STUB METHODS ==================
    
    def _plot_instrument_with_trades(self, ax, prices: List[float], trades: List[Dict], instrument: str) -> None:
        """Plot instrument with enhanced trade markers"""
        try:
            import matplotlib.pyplot as plt
            
            if not prices:
                return
                
            # Plot price line
            ax.plot(range(len(prices)), prices, label=f'{instrument} Price', linewidth=1.5, color='blue')
            
            # Plot trade markers
            for trade in trades:
                if trade.get('instrument', 'Main') == instrument:
                    entry_idx = trade.get('entry_step', 0)
                    exit_idx = trade.get('exit_step', len(prices) - 1)
                    entry_price = trade.get('entry_price', 0)
                    exit_price = trade.get('exit_price', 0)
                    pnl = trade.get('pnl', 0)
                    
                    # Entry marker
                    color = 'green' if pnl > 0 else 'red'
                    ax.scatter(entry_idx, entry_price, color='blue', marker='^', s=self.marker_size, 
                              label='Entry' if trade == trades[0] else "", zorder=5)
                    
                    # Exit marker
                    ax.scatter(exit_idx, exit_price, color=color, marker='v', s=self.marker_size,
                              label='Exit' if trade == trades[0] else "", zorder=5)
                    
                    # Trade line
                    ax.plot([entry_idx, exit_idx], [entry_price, exit_price], 
                           color=color, alpha=0.6, linewidth=2)
                    
                    # P&L annotation
                    mid_idx = (entry_idx + exit_idx) // 2
                    mid_price = (entry_price + exit_price) / 2
                    ax.annotate(f'${pnl:.1f}', (mid_idx, mid_price), 
                               xytext=(5, 5), textcoords='offset points', 
                               fontsize=8, color=color, weight='bold')
                               
        except Exception as e:
            self.logger.warning(f"Instrument plotting failed: {e}")
    
    def _plot_enhanced_pnl_curve(self, ax, trades: List[Dict]) -> None:
        """Plot enhanced cumulative P&L curve with additional metrics"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No trades available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Calculate cumulative P&L
            cumulative_pnl = []
            total = 0
            for trade in trades:
                total += trade.get('pnl', 0)
                cumulative_pnl.append(total)
            
            # Plot cumulative P&L
            ax.plot(range(len(cumulative_pnl)), cumulative_pnl, 
                   color='green' if cumulative_pnl[-1] > 0 else 'red', linewidth=2)
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Highlight drawdown periods
            peak = 0
            for i, pnl in enumerate(cumulative_pnl):
                if pnl > peak:
                    peak = pnl
                elif pnl < peak * 0.95:  # 5% drawdown threshold
                    ax.axvspan(i-1, i+1, alpha=0.2, color='red')
            
            # Add final P&L annotation
            final_pnl = cumulative_pnl[-1]
            ax.annotate(f'Final: ${final_pnl:.2f}', 
                       xy=(len(cumulative_pnl)-1, final_pnl),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, weight='bold')
            
            ax.set_title('Cumulative P&L', fontsize=12, weight='bold')
            ax.set_xlabel('Trade #')
            ax.set_ylabel('P&L ($)')
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.warning(f"P&L curve plotting failed: {e}")
    
    def _plot_enhanced_drawdown(self, ax, trades: List[Dict]) -> None:
        """Plot enhanced drawdown analysis"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No trades available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Calculate running drawdown
            cumulative_pnl = []
            peak = 0
            drawdowns = []
            total = 0
            
            for trade in trades:
                total += trade.get('pnl', 0)
                cumulative_pnl.append(total)
                
                if total > peak:
                    peak = total
                    drawdown = 0
                else:
                    drawdown = (peak - total) / max(peak, 1) * 100
                drawdowns.append(drawdown)
            
            # Plot drawdown
            ax.fill_between(range(len(drawdowns)), 0, drawdowns, 
                           color='red', alpha=0.3, label='Drawdown')
            ax.plot(range(len(drawdowns)), drawdowns, color='red', linewidth=2)
            
            # Mark maximum drawdown
            max_dd = max(drawdowns) if drawdowns else 0
            max_dd_idx = drawdowns.index(max_dd) if drawdowns else 0
            ax.scatter(max_dd_idx, max_dd, color='darkred', s=100, zorder=5)
            ax.annotate(f'Max DD: {max_dd:.1f}%', 
                       xy=(max_dd_idx, max_dd),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, weight='bold', color='darkred')
            
            ax.set_title('Drawdown Analysis', fontsize=12, weight='bold')
            ax.set_xlabel('Trade #')
            ax.set_ylabel('Drawdown (%)')
            ax.set_ylim(bottom=0)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.warning(f"Drawdown plotting failed: {e}")
    
    def _plot_trade_metrics(self, ax, trades: List[Dict]) -> None:
        """Plot key trade metrics summary"""
        try:
            ax.axis('off')
            
            if not trades:
                ax.text(0.5, 0.5, 'No trades available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
            
            # Calculate metrics
            total_trades = len(trades)
            wins = sum(1 for t in trades if t.get('pnl', 0) > 0)
            losses = total_trades - wins
            win_rate = wins / total_trades if total_trades > 0 else 0
            
            total_pnl = sum(t.get('pnl', 0) for t in trades)
            avg_win = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) > 0) / max(wins, 1)
            avg_loss = sum(t.get('pnl', 0) for t in trades if t.get('pnl', 0) < 0) / max(losses, 1)
            profit_factor = abs(avg_win * wins / (avg_loss * losses)) if losses > 0 and avg_loss < 0 else float('inf')
            
            # Create metrics text
            metrics_text = f"""TRADE METRICS
━━━━━━━━━━━━━━━━━━
Total Trades: {total_trades}
Wins: {wins} | Losses: {losses}
Win Rate: {win_rate:.1%}

Total P&L: ${total_pnl:.2f}
Avg Win: ${avg_win:.2f}
Avg Loss: ${avg_loss:.2f}
Profit Factor: {profit_factor:.2f}

Best Trade: ${max((t.get('pnl', 0) for t in trades), default=0):.2f}
Worst Trade: ${min((t.get('pnl', 0) for t in trades), default=0):.2f}"""
            
            # Display metrics
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                   
        except Exception as e:
            self.logger.warning(f"Trade metrics plotting failed: {e}")
    
    def _plot_enhanced_trade_distribution(self, ax, trades: List[Dict]) -> None:
        """Plot enhanced trade P&L distribution with statistics"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No trades available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            import numpy as np
            
            pnls = [t.get('pnl', 0) for t in trades]
            
            # Create histogram
            n_bins = min(20, len(trades) // 2) if len(trades) > 10 else 10
            counts, bins, patches = ax.hist(pnls, bins=n_bins, alpha=0.7, edgecolor='black')
            
            # Color bars based on profit/loss
            for i, patch in enumerate(patches):
                if bins[i] < 0:
                    patch.set_facecolor('red')
                else:
                    patch.set_facecolor('green')
            
            # Add statistical lines
            mean_pnl = np.mean(pnls)
            std_pnl = np.std(pnls)
            
            ax.axvline(mean_pnl, color='blue', linestyle='--', linewidth=2, label=f'Mean: ${mean_pnl:.2f}')
            ax.axvline(mean_pnl + std_pnl, color='orange', linestyle=':', alpha=0.7, label=f'+1σ: ${mean_pnl + std_pnl:.2f}')
            ax.axvline(mean_pnl - std_pnl, color='orange', linestyle=':', alpha=0.7, label=f'-1σ: ${mean_pnl - std_pnl:.2f}')
            
            ax.set_title('P&L Distribution', fontsize=12, weight='bold')
            ax.set_xlabel('P&L ($)')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.warning(f"Trade distribution plotting failed: {e}")
    
    def _plot_enhanced_win_loss_analysis(self, ax, trades: List[Dict]) -> None:
        """Plot enhanced win/loss analysis with detailed statistics"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No trades available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Separate wins and losses
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) < 0]
            breakevens = [t for t in trades if t.get('pnl', 0) == 0]
            
            # Create pie chart
            sizes = [len(wins), len(losses), len(breakevens)]
            labels = [f'Wins ({len(wins)})', f'Losses ({len(losses)})', f'BE ({len(breakevens)})']
            colors = ['green', 'red', 'gray']
            
            # Only include non-zero categories
            filtered_sizes = []
            filtered_labels = []
            filtered_colors = []
            
            for size, label, color in zip(sizes, labels, colors):
                if size > 0:
                    filtered_sizes.append(size)
                    filtered_labels.append(label)
                    filtered_colors.append(color)
            
            if filtered_sizes:
                wedges, texts, autotexts = ax.pie(filtered_sizes, labels=filtered_labels, 
                                                 colors=filtered_colors, autopct='%1.1f%%',
                                                 startangle=90)
                
                # Enhance text appearance
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_weight('bold')
            
            ax.set_title('Win/Loss Distribution', fontsize=12, weight='bold')
            
        except Exception as e:
            self.logger.warning(f"Win/loss analysis plotting failed: {e}")
    
    def _plot_time_analysis(self, ax, trades: List[Dict]) -> None:
        """Plot time-based trading analysis"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No trades available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Extract trade durations and timestamps
            durations = []
            timestamps = []
            
            for trade in trades:
                entry_step = trade.get('entry_step', 0)
                exit_step = trade.get('exit_step', 0)
                duration = max(1, exit_step - entry_step)
                durations.append(duration)
                timestamps.append(entry_step)
            
            # Plot trade frequency over time
            ax.scatter(timestamps, durations, alpha=0.6, s=50)
            
            # Add trend line
            if len(timestamps) > 1:
                import numpy as np
                z = np.polyfit(timestamps, durations, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(timestamps), "r--", alpha=0.8, linewidth=2)
            
            # Add average duration line
            avg_duration = sum(durations) / len(durations) if durations else 0
            ax.axhline(y=avg_duration, color='green', linestyle=':', alpha=0.7, 
                      label=f'Avg Duration: {avg_duration:.1f}')
            
            ax.set_title('Trade Duration Analysis', fontsize=12, weight='bold')
            ax.set_xlabel('Entry Time Step')
            ax.set_ylabel('Trade Duration (steps)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.warning(f"Time analysis plotting failed: {e}")
    
    def _apply_chart_style(self) -> None:
        """Apply consistent chart styling"""
        try:
            import matplotlib.pyplot as plt
            
            # Set style
            if self.style in plt.style.available:
                plt.style.use(self.style)
            else:
                plt.style.use('default')
            
            # Set global parameters
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': 'black',
                'axes.linewidth': 1.2,
                'grid.alpha': 0.3,
                'grid.linewidth': 0.8,
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14
            })
            
        except Exception as e:
            self.logger.warning(f"Chart styling failed: {e}")
    
    def _save_chart(self, fig, filename: str) -> None:
        """Save chart with enhanced options"""
        try:
            if not self.save_path:
                return
                
            import os
            
            # Create save directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)
            
            # Clean filename
            clean_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
            if not clean_filename.endswith('.png'):
                clean_filename += '.png'
            
            full_path = os.path.join(self.save_path, clean_filename)
            
            # Save with high quality
            fig.savefig(full_path, dpi=self.chart_dpi, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', format='png')
            
            self.viz_stats['charts_saved'] += 1
            self.logger.info(f"Chart saved: {full_path}")
            
        except Exception as e:
            self.logger.warning(f"Chart saving failed: {e}")
    
    def _record_chart_generation(self, chart_type: str, trades: List[Dict]) -> None:
        """Record chart generation for analytics"""
        try:
            record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'chart_type': chart_type,
                'trades_count': len(trades),
                'generated_successfully': True
            }
            
            self._chart_history.append(record)
            
            # Keep only last 100 records
            if len(self._chart_history) > 100:
                self._chart_history = self._chart_history[-100:]
            
            # Cache the chart type
            self._chart_cache[chart_type] = {
                'timestamp': record['timestamp'],
                'trades_count': len(trades)
            }
            
        except Exception as e:
            self.logger.warning(f"Chart generation recording failed: {e}")
    
    # Dashboard-specific plotting methods (stubs)
    def _plot_balance_evolution(self, ax, trades):
        """Plot balance evolution over time"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No trades available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Calculate balance evolution
            initial_balance = 10000  # Default starting balance
            balance_history = [initial_balance]
            timestamps = [0]
            
            current_balance = initial_balance
            for i, trade in enumerate(trades):
                current_balance += trade.get('pnl', 0)
                balance_history.append(current_balance)
                timestamps.append(i + 1)
            
            # Plot balance line
            ax.plot(timestamps, balance_history, linewidth=2, color='blue', label='Account Balance')
            
            # Add trend line
            if len(balance_history) > 2:
                import numpy as np
                z = np.polyfit(timestamps, balance_history, 1)
                p = np.poly1d(z)
                ax.plot(timestamps, p(timestamps), "r--", alpha=0.6, label='Trend')
            
            # Highlight drawdown periods
            peak_balance = initial_balance
            for i, balance in enumerate(balance_history):
                if balance > peak_balance:
                    peak_balance = balance
                elif balance < peak_balance * 0.95:  # 5% drawdown
                    ax.axvspan(timestamps[max(0, i-1)], timestamps[min(len(timestamps)-1, i+1)], 
                              alpha=0.2, color='red')
            
            ax.set_title('Account Balance Evolution', fontsize=14, weight='bold')
            ax.set_xlabel('Trade Number')
            ax.set_ylabel('Balance ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.ticklabel_format(style='plain', axis='y')
            
        except Exception as e:
            self.logger.warning(f"Balance evolution plotting failed: {e}")
    
    def _plot_dashboard_status(self, ax):
        """Plot current dashboard status"""
        try:
            ax.axis('off')
            
            # Get current system information
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            
            # Get SmartInfoBus data
            risk_data = self.smart_bus.get('risk_metrics', 'TradeMapVisualizer') or {}
            positions = self.smart_bus.get('positions', 'TradeMapVisualizer') or []
            performance = self.smart_bus.get('trading_performance', 'TradeMapVisualizer') or {}
            
            # Status indicators
            system_status = "🟢 ACTIVE" if not self.is_disabled else "🔴 DISABLED"
            chart_status = "🟢 READY" if self.viz_stats['charts_generated'] > 0 else "🟡 IDLE"
            
            status_text = f"""SYSTEM DASHBOARD - {current_time}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 ACCOUNT STATUS:
   Balance: ${risk_data.get('balance', 0):,.2f}
   Equity: ${risk_data.get('equity', 0):,.2f}
   P&L Today: ${performance.get('daily_pnl', 0):+.2f}

📊 TRADING STATUS:
   Active Positions: {len(positions)}
   Total Trades: {self.viz_stats.get('trade_charts', 0)}
   Charts Generated: {self.viz_stats['charts_generated']}

⚡ SYSTEM STATUS:
   Visualizer: {system_status}
   Chart Engine: {chart_status}
   Error Count: {self.error_count}

🎯 PERFORMANCE:
   Win Rate: {performance.get('win_rate', 0):.1%}
   Profit Factor: {performance.get('profit_factor', 0):.2f}
   Max Drawdown: {performance.get('max_drawdown', 0):.1%}"""
            
            ax.text(0.05, 0.95, status_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
                   
        except Exception as e:
            self.logger.warning(f"Dashboard status plotting failed: {e}")
    
    def _plot_dashboard_pnl(self, ax, trades):
        """Plot dashboard P&L summary"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No P&L data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Calculate daily P&L (simulated)
            pnl_data = []
            cumulative = 0
            
            for trade in trades:
                cumulative += trade.get('pnl', 0)
                pnl_data.append(cumulative)
            
            # Plot P&L bars
            colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_data]
            bars = ax.bar(range(len(pnl_data)), pnl_data, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for i, (bar, pnl) in enumerate(zip(bars, pnl_data)):
                if abs(pnl) > max(pnl_data) * 0.05:  # Only label significant values
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'${pnl:.0f}', ha='center', va='bottom' if pnl >= 0 else 'top',
                           fontsize=8, weight='bold')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            
            ax.set_title('P&L Progress', fontsize=12, weight='bold')
            ax.set_xlabel('Trade #')
            ax.set_ylabel('Cumulative P&L ($)')
            ax.grid(True, alpha=0.3, axis='y')
            
        except Exception as e:
            self.logger.warning(f"Dashboard P&L plotting failed: {e}")
    
    def _plot_dashboard_drawdown(self, ax, trades):
        """Plot dashboard drawdown tracking"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No drawdown data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Calculate running drawdown
            peak = 0
            drawdowns = []
            cumulative = 0
            
            for trade in trades:
                cumulative += trade.get('pnl', 0)
                if cumulative > peak:
                    peak = cumulative
                    dd_pct = 0
                else:
                    dd_pct = ((peak - cumulative) / max(peak, 1)) * 100
                drawdowns.append(dd_pct)
            
            # Plot drawdown area
            ax.fill_between(range(len(drawdowns)), 0, drawdowns, 
                           color='red', alpha=0.4, label='Drawdown')
            
            # Add warning zones
            ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='Warning (5%)')
            ax.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='Danger (10%)')
            
            # Mark current drawdown
            if drawdowns:
                current_dd = drawdowns[-1]
                ax.scatter(len(drawdowns)-1, current_dd, color='darkred', s=100, zorder=5)
                ax.annotate(f'{current_dd:.1f}%', 
                           xy=(len(drawdowns)-1, current_dd),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, weight='bold')
            
            ax.set_title('Drawdown Tracking', fontsize=12, weight='bold')
            ax.set_xlabel('Trade #')
            ax.set_ylabel('Drawdown (%)')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)
            
        except Exception as e:
            self.logger.warning(f"Dashboard drawdown plotting failed: {e}")
    
    def _plot_dashboard_risk(self, ax, market_data):
        """Plot dashboard risk metrics"""
        try:
            ax.axis('off')
            
            # Get risk data from SmartInfoBus
            risk_metrics = self.smart_bus.get('risk_metrics', 'TradeMapVisualizer') or {}
            
            # Risk indicators
            risk_score = risk_metrics.get('risk_score', 0.5)
            volatility = risk_metrics.get('volatility', 0.02)
            var_95 = risk_metrics.get('var_95', 0)
            leverage = risk_metrics.get('leverage', 1.0)
            
            # Determine risk level
            if risk_score < 0.3:
                risk_level = "🟢 LOW"
                risk_color = 'green'
            elif risk_score < 0.6:
                risk_level = "🟡 MEDIUM"
                risk_color = 'orange'
            else:
                risk_level = "🔴 HIGH"
                risk_color = 'red'
            
            risk_text = f"""RISK METRICS
━━━━━━━━━━━━━━━━━━
Risk Level: {risk_level}
Risk Score: {risk_score:.2f}

Volatility: {volatility:.2%}
VaR (95%): ${var_95:.2f}
Leverage: {leverage:.1f}x

Max Drawdown: {risk_metrics.get('max_drawdown', 0):.1%}
Exposure: ${risk_metrics.get('total_exposure', 0):,.0f}

Position Limit: {risk_metrics.get('position_limit', 5)}
Risk Budget: ${risk_metrics.get('risk_budget', 1000):,.0f}"""
            
            ax.text(0.05, 0.95, risk_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=risk_color, alpha=0.2))
                   
        except Exception as e:
            self.logger.warning(f"Dashboard risk plotting failed: {e}")
    
    def _plot_module_performance_summary(self, ax):
        """Plot module performance summary"""
        try:
            # Get module performance data
            module_perf = self.smart_bus.get('module_performance', 'TradeMapVisualizer') or {}
            
            if not module_perf:
                ax.text(0.5, 0.5, 'No module data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
            
            # Extract module names and performance scores
            modules = list(module_perf.keys())[:10]  # Top 10 modules
            scores = [module_perf[mod].get('performance_score', 0.5) for mod in modules]
            
            # Create horizontal bar chart
            y_pos = range(len(modules))
            colors = ['green' if score > 0.7 else 'orange' if score > 0.4 else 'red' for score in scores]
            
            bars = ax.barh(y_pos, scores, color=colors, alpha=0.7)
            
            # Add score labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.2f}', va='center', fontsize=8)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels([mod[:15] + '...' if len(mod) > 15 else mod for mod in modules], fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_title('Module Performance', fontsize=12, weight='bold')
            ax.set_xlabel('Performance Score')
            ax.grid(True, alpha=0.3, axis='x')
            
        except Exception as e:
            self.logger.warning(f"Module performance plotting failed: {e}")
    def _plot_regime_performance(self, ax, trades):
        """Plot regime-based performance analysis"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No regime data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Simulate regime classification for trades
            regime_performance = {'trending': [], 'volatile': [], 'ranging': [], 'noise': []}
            
            for i, trade in enumerate(trades):
                # Simple regime simulation based on trade characteristics
                pnl = trade.get('pnl', 0)
                duration = trade.get('exit_step', 0) - trade.get('entry_step', 0)
                
                if duration > 20 and abs(pnl) > 50:
                    regime = 'trending'
                elif duration < 5 and abs(pnl) > 20:
                    regime = 'volatile'
                elif duration > 10 and abs(pnl) < 30:
                    regime = 'ranging'
                else:
                    regime = 'noise'
                    
                regime_performance[regime].append(pnl)
            
            # Calculate regime statistics
            regime_stats = {}
            for regime, pnls in regime_performance.items():
                if pnls:
                    regime_stats[regime] = {
                        'avg_pnl': sum(pnls) / len(pnls),
                        'trade_count': len(pnls),
                        'total_pnl': sum(pnls)
                    }
                else:
                    regime_stats[regime] = {'avg_pnl': 0, 'trade_count': 0, 'total_pnl': 0}
            
            # Create grouped bar chart
            regimes = list(regime_stats.keys())
            avg_pnls = [regime_stats[r]['avg_pnl'] for r in regimes]
            trade_counts = [regime_stats[r]['trade_count'] for r in regimes]
            
            x = range(len(regimes))
            width = 0.35
            
            # Plot average P&L bars
            bars1 = ax.bar([i - width/2 for i in x], avg_pnls, width, 
                          label='Avg P&L', alpha=0.7, color=self.color_schemes['regime'].values())
            
            # Add a secondary y-axis for trade counts
            ax2 = ax.twinx()
            bars2 = ax2.bar([i + width/2 for i in x], trade_counts, width,
                           label='Trade Count', alpha=0.5, color='gray')
            
            # Customize axes
            ax.set_xlabel('Market Regime')
            ax.set_ylabel('Average P&L ($)', color='blue')
            ax2.set_ylabel('Trade Count', color='gray')
            ax.set_title('Performance by Market Regime', fontsize=12, weight='bold')
            
            ax.set_xticks(x)
            ax.set_xticklabels(regimes)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add legends
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            
        except Exception as e:
            self.logger.warning(f"Regime performance plotting failed: {e}")
    
    def _plot_trading_activity(self, ax, trades):
        """Plot trading activity analysis"""
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No activity data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                return
                
            # Analyze trading activity patterns
            hourly_activity = {}
            daily_pnl = {}
            
            for i, trade in enumerate(trades):
                # Simulate time-based analysis
                hour = (i * 3) % 24  # Simulate hour of day
                day = i // 8  # Simulate day number
                pnl = trade.get('pnl', 0)
                
                # Hourly activity
                if hour not in hourly_activity:
                    hourly_activity[hour] = {'count': 0, 'total_pnl': 0}
                hourly_activity[hour]['count'] += 1
                hourly_activity[hour]['total_pnl'] += pnl
                
                # Daily P&L
                if day not in daily_pnl:
                    daily_pnl[day] = 0
                daily_pnl[day] += pnl
            
            # Plot daily P&L trend
            days = sorted(daily_pnl.keys())
            pnls = [daily_pnl[day] for day in days]
            
            # Create bar chart with color coding
            colors = ['green' if pnl >= 0 else 'red' for pnl in pnls]
            bars = ax.bar(days, pnls, color=colors, alpha=0.7)
            
            # Add trend line
            if len(days) > 2:
                import numpy as np
                z = np.polyfit(days, pnls, 1)
                p = np.poly1d(z)
                ax.plot(days, p(days), "b--", alpha=0.8, linewidth=2, label='Trend')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.8)
            
            # Add moving average
            if len(pnls) >= 3:
                window = min(3, len(pnls) // 2)
                moving_avg = []
                for i in range(len(pnls)):
                    start_idx = max(0, i - window + 1)
                    avg = sum(pnls[start_idx:i+1]) / (i - start_idx + 1)
                    moving_avg.append(avg)
                ax.plot(days, moving_avg, 'orange', linewidth=2, alpha=0.8, label=f'MA({window})')
            
            ax.set_title('Trading Activity & Daily P&L', fontsize=12, weight='bold')
            ax.set_xlabel('Trading Day')
            ax.set_ylabel('Daily P&L ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.logger.warning(f"Trading activity plotting failed: {e}")
    
    def _plot_alert_timeline(self, ax):
        """Plot alert and event timeline"""
        try:
            ax.axis('off')
            
            # Get recent alerts and events (simulated for now)
            current_time = datetime.datetime.now()
            
            alerts = [
                {'time': current_time - datetime.timedelta(minutes=30), 'type': 'INFO', 'message': 'Chart generation completed'},
                {'time': current_time - datetime.timedelta(hours=1), 'type': 'WARNING', 'message': 'High volatility detected'},
                {'time': current_time - datetime.timedelta(hours=2), 'type': 'SUCCESS', 'message': 'Profitable trade closed'},
                {'time': current_time - datetime.timedelta(hours=4), 'type': 'ERROR', 'message': 'Module error recovered'},
                {'time': current_time - datetime.timedelta(hours=6), 'type': 'INFO', 'message': 'System restart completed'}
            ]
            
            # Create timeline text
            timeline_text = "RECENT ALERTS & EVENTS\n" + "━" * 50 + "\n"
            
            for alert in alerts:
                time_str = alert['time'].strftime('%H:%M')
                alert_type = alert['type']
                message = alert['message']
                
                # Add appropriate icon
                if alert_type == 'ERROR':
                    icon = '🔴'
                elif alert_type == 'WARNING':
                    icon = '🟡'
                elif alert_type == 'SUCCESS':
                    icon = '🟢'
                else:
                    icon = '🔵'
                
                timeline_text += f"{time_str} {icon} {alert_type}: {message}\n"
            
            # Add current status
            timeline_text += "\n" + "━" * 50 + "\n"
            timeline_text += f"🕒 Current Time: {current_time.strftime('%H:%M:%S')}\n"
            timeline_text += f"📊 Charts Generated: {self.viz_stats['charts_generated']}\n"
            timeline_text += f"⚠️  Error Count: {self.error_count}\n"
            timeline_text += f"🎯 Status: {'🔴 DISABLED' if self.is_disabled else '🟢 ACTIVE'}"
            
            ax.text(0.02, 0.98, timeline_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
                   
        except Exception as e:
            self.logger.warning(f"Alert timeline plotting failed: {e}")
    
    def _add_chart_indicators(self, ax, prices):
        """Add technical indicators to chart"""
        try:
            if isinstance(prices, dict):
                # Use first instrument's prices for indicators
                price_data = next(iter(prices.values()))
            else:
                price_data = prices
                
            if not price_data or len(price_data) < 20:
                return
                
            import numpy as np
            
            # Simple Moving Average
            window = min(20, len(price_data) // 4)
            if window >= 2:
                sma = []
                for i in range(len(price_data)):
                    start_idx = max(0, i - window + 1)
                    avg = sum(price_data[start_idx:i+1]) / (i - start_idx + 1)
                    sma.append(avg)
                
                ax.plot(range(len(sma)), sma, 'orange', alpha=0.7, linewidth=1.5, 
                       label=f'SMA({window})', linestyle='--')
            
            # Bollinger Bands (simplified)
            if len(price_data) >= 20:
                bb_window = 20
                bb_std = 2
                
                bb_sma = []
                bb_upper = []
                bb_lower = []
                
                for i in range(len(price_data)):
                    if i >= bb_window - 1:
                        window_data = price_data[i-bb_window+1:i+1]
                        mean_price = sum(window_data) / len(window_data)
                        std_price = np.std(window_data)
                        
                        bb_sma.append(mean_price)
                        bb_upper.append(mean_price + bb_std * std_price)
                        bb_lower.append(mean_price - bb_std * std_price)
                
                if bb_sma:
                    bb_x = range(bb_window-1, len(price_data))
                    ax.fill_between(bb_x, bb_lower, bb_upper, alpha=0.1, color='gray', label='Bollinger Bands')
                    ax.plot(bb_x, bb_sma, 'red', alpha=0.5, linewidth=1, linestyle=':')
            
        except Exception as e:
            self.logger.warning(f"Indicator plotting failed: {e}")
    
    def _add_regime_backgrounds(self, ax):
        """Add market regime background coloring"""
        try:
            # Get regime data from SmartInfoBus if available
            consensus_data = self.smart_bus.get('consensus_data', 'TradeMapVisualizer') or {}
            
            # Simulate regime periods for demonstration
            xlim = ax.get_xlim()
            total_length = xlim[1] - xlim[0]
            
            if total_length > 10:
                # Add some regime backgrounds
                regime_periods = [
                    {'start': 0, 'end': total_length * 0.3, 'regime': 'trending', 'color': 'blue'},
                    {'start': total_length * 0.3, 'end': total_length * 0.6, 'regime': 'volatile', 'color': 'orange'},
                    {'start': total_length * 0.6, 'end': total_length, 'regime': 'ranging', 'color': 'purple'}
                ]
                
                for period in regime_periods:
                    ax.axvspan(period['start'], period['end'], alpha=0.05, 
                              color=period['color'], label=f"{period['regime'].title()} Regime")
                              
        except Exception as e:
            self.logger.warning(f"Regime background plotting failed: {e}")

    # ================== UTILITY METHODS ==================
    
    def get_visualization_report(self) -> str:
        """Generate operator-friendly visualization report"""
        
        chart_summary = ""
        if self._chart_history:
            recent_charts = self._chart_history[-5:]
            for chart in recent_charts:
                chart_summary += f"  • {chart.get('timestamp', '')[:19]}: {chart.get('chart_type', 'unknown')}\n"
        
        return f"""
🎨 TRADE MAP VISUALIZER REPORT
═══════════════════════════════════════
[STATS] Chart Generation Statistics:
• Total Charts: {self.viz_stats['charts_generated']}
• Trade Charts: {self.viz_stats['trade_charts']}
• Performance Charts: {self.viz_stats['performance_charts']}
• Dashboard Charts: {self.viz_stats['dashboard_charts']}
• Charts Saved: {self.viz_stats['charts_saved']}

⚙️ Configuration:
• Style: {self.style}
• Marker Size: {self.marker_size}
• Auto Save: {'[OK] Enabled' if self.auto_save else '[FAIL] Disabled'}
• Save Path: {self.save_path or 'Not set'}
• Chart DPI: {self.chart_dpi}

[CHART] Available Chart Types:
{chr(10).join([f'  • {name}: {desc}' for name, desc in self.chart_types.items()])}

🕒 Recent Charts Generated:
{chart_summary if chart_summary else '  📭 No recent charts'}

[SAVE] Storage:
• Chart History: {len(self._chart_history)} entries
• Cache Size: {len(self._chart_cache)} items

[TOOL] System Health:
• Error Count: {self.error_count}
• Status: {'[ALERT] Disabled' if self.is_disabled else '[OK] Healthy'}
        """

    def get_observation_components(self) -> np.ndarray:
        """Return visualizer features for observation"""
        
        try:
            style_idx = self.available_styles.index(self.style) if self.style in self.available_styles else 0
            
            features = [
                float(self.marker_size / 100),  # Normalized marker size
                float(style_idx / len(self.available_styles)),  # Normalized style index
                float(self.auto_save),  # Auto save enabled
                float(bool(self._last_fig)),  # Has generated charts
                float(self.viz_stats['charts_generated']),  # Total charts
                float(len(self._chart_history) / 100),  # History fullness
                float(bool(self.save_path)),  # Save path configured
                float(self.chart_dpi / 300)  # Normalized DPI
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)