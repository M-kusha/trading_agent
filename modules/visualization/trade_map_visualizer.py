from modules.core.mixins import SmartInfoBusStateMixin, SmartInfoBusTradingMixin
# ─────────────────────────────────────────────────────────────
# File: modules/visualization/trade_map_visualizer.py
# Enhanced Trade Map Visualizer with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
import copy

from modules.core.core import Module, ModuleConfig, audit_step
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class TradeMapVisualizer(Module, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Enhanced trade map visualizer with InfoBus integration.
    Creates comprehensive visual analysis of trading performance, patterns, and system behavior.
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
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 100),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
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
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "TradeMapVisualizer",
            "logs/visualization/trade_map_visualizer.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("TradeMapVisualizer")
        
        self.log_operator_info(
            "🎨 Trade Map Visualizer initialized",
            style=self.style,
            marker_size=self.marker_size,
            save_path=self.save_path,
            auto_save=self.auto_save
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
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
        
        self.log_operator_info("🔄 Trade Map Visualizer reset - all charts cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited functionality")
            return
        
        # Extract visualization data
        context = extract_standard_context(info_bus)
        viz_data = self._extract_visualization_data_from_info_bus(info_bus)
        
        # Generate auto-charts if enabled
        if self.debug and viz_data.get('should_generate_chart', False):
            self._generate_auto_chart(viz_data, context)
        
        # Update InfoBus with visualizer status
        self._update_info_bus(info_bus)

    def _extract_visualization_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract data needed for visualization from InfoBus"""
        
        data = {}
        
        try:
            # Get visualization interface data
            viz_interface_data = info_bus.get('module_data', {}).get('visualization_interface', {})
            data['interface_data'] = viz_interface_data
            
            # Get trading data
            recent_trades = info_bus.get('recent_trades', [])
            data['trades'] = recent_trades
            
            # Get position data
            positions = InfoBusExtractor.get_positions(info_bus)
            data['positions'] = positions
            
            # Get price data if available
            prices = info_bus.get('prices', {})
            data['prices'] = prices
            
            # Get performance metrics
            risk_data = info_bus.get('risk', {})
            data['balance'] = risk_data.get('balance', 0)
            data['equity'] = risk_data.get('equity', 0)
            data['drawdown'] = risk_data.get('current_drawdown', 0)
            
            # Determine if we should generate a chart
            step = info_bus.get('step_idx', 0)
            data['should_generate_chart'] = (step % 100 == 0)  # Every 100 steps
            
        except Exception as e:
            self.log_operator_warning(f"Visualization data extraction failed: {e}")
            data = {
                'interface_data': {},
                'trades': [],
                'positions': [],
                'prices': {},
                'balance': 0,
                'equity': 0,
                'drawdown': 0,
                'should_generate_chart': False
            }
        
        return data

    def plot_trades(
        self,
        prices: Union[List[float], Dict[str, List[float]]],
        trades: List[Dict],
        info_bus: Optional[InfoBus] = None,
        show: bool = False,
        title: str = "Trade Map",
        plot_indicators: bool = True,
        save_name: Optional[str] = None
    ) -> Any:
        """Create comprehensive trade visualization with enhanced features"""
        
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
            self._plot_main_price_chart(ax_main, prices, trades, plot_indicators, info_bus)
            ax_main.set_title(title, fontsize=16, fontweight='bold', pad=20)
            
            # P&L analysis (enhanced)
            ax_pnl = fig.add_subplot(gs[1, 0])
            self._plot_enhanced_pnl_curve(ax_pnl, trades)
            
            # Drawdown analysis
            ax_dd = fig.add_subplot(gs[1, 1])
            self._plot_enhanced_drawdown(ax_dd, trades, info_bus)
            
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
            self._plot_system_status_panel(ax_status, info_bus)
            
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
            self._record_chart_generation('trade_map', trades, info_bus)
            
            return fig
            
        except Exception as e:
            self.log_operator_error(f"Trade plot generation failed: {e}")
            return None

    def _plot_main_price_chart(self, ax, prices: Union[List[float], Dict[str, List[float]]], 
                              trades: List[Dict], plot_indicators: bool, info_bus: Optional[InfoBus]) -> None:
        """Plot enhanced main price chart with trades and indicators"""
        
        try:
            # Handle multi-instrument data
            if isinstance(prices, dict):
                for inst, price_series in prices.items():
                    self._plot_instrument_with_trades(ax, price_series, trades, inst)
            else:
                self._plot_instrument_with_trades(ax, prices, trades, "Main")
            
            # Add indicators if requested
            if plot_indicators and info_bus:
                self._add_chart_indicators(ax, prices, info_bus)
            
            # Enhance chart appearance
            ax.set_xlabel("Time Step", fontsize=12)
            ax.set_ylabel("Price", fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add regime backgrounds if available
            if info_bus:
                self._add_regime_backgrounds(ax, info_bus)
            
        except Exception as e:
            self.log_operator_warning(f"Main price chart plotting failed: {e}")

    def _plot_instrument_with_trades(self, ax, prices: List[float], trades: List[Dict], instrument: str) -> None:
        """Plot instrument with enhanced trade markers"""
        
        try:
            prices_np = np.asarray(prices)
            
            # Plot price line with enhanced styling
            line_color = 'darkblue' if instrument == "Main" else f'C{hash(instrument) % 10}'
            ax.plot(prices_np, label=f"{instrument} Price", linewidth=2, color=line_color, alpha=0.8)
            
            # Filter trades for this instrument
            inst_trades = [t for t in trades if t.get('instrument', '') == instrument or instrument == "Main"]
            
            # Plot trades with enhanced visualization
            for i, trade in enumerate(inst_trades):
                self._plot_single_trade(ax, trade, prices_np, i)
            
        except Exception as e:
            self.log_operator_warning(f"Instrument plotting failed for {instrument}: {e}")

    def _plot_single_trade(self, ax, trade: Dict, prices_np: np.ndarray, trade_idx: int) -> None:
        """Plot single trade with enhanced markers and annotations"""
        
        try:
            # Extract trade data
            entry_idx = int(trade.get('entry_idx', trade.get('entry_step', 0)))
            exit_idx = int(trade.get('exit_idx', trade.get('exit_step', entry_idx + trade.get('duration', 1))))
            pnl = trade.get('pnl', 0)
            size = abs(trade.get('size', 1))
            side = trade.get('side', '').lower()
            
            # Determine colors and markers
            pnl_color = self.color_schemes['profit_loss']['profit' if pnl >= 0 else 'loss']
            entry_marker = '^' if side in ['buy', 'long'] else 'v'
            exit_marker = 'v' if side in ['buy', 'long'] else '^'
            
            # Scale marker size
            marker_scale = np.clip(size, 0.5, 3.0)
            scaled_size = self.marker_size * marker_scale
            
            # Plot entry point
            if 0 <= entry_idx < len(prices_np):
                ax.scatter([entry_idx], [prices_np[entry_idx]], 
                          marker=entry_marker, c=pnl_color, s=scaled_size,
                          edgecolors='black', linewidth=2, alpha=0.9, zorder=5)
            
            # Plot exit point
            if 0 <= exit_idx < len(prices_np):
                ax.scatter([exit_idx], [prices_np[exit_idx]], 
                          marker=exit_marker, c=pnl_color, s=scaled_size,
                          edgecolors='black', linewidth=2, alpha=0.9, zorder=5)
            
            # Connect entry and exit with enhanced line
            if 0 <= entry_idx < len(prices_np) and 0 <= exit_idx < len(prices_np):
                line_style = '-' if pnl >= 0 else '--'
                line_alpha = 0.6 if pnl >= 0 else 0.4
                ax.plot([entry_idx, exit_idx], [prices_np[entry_idx], prices_np[exit_idx]],
                       color=pnl_color, alpha=line_alpha, linestyle=line_style,
                       linewidth=2 * marker_scale, zorder=3)
            
            # Add P&L annotation for significant trades
            pnl_threshold = 50 if abs(pnl) < 500 else 100
            if abs(pnl) > pnl_threshold:
                mid_idx = (entry_idx + exit_idx) // 2
                if 0 <= mid_idx < len(prices_np):
                    mid_price = (prices_np[entry_idx] + prices_np[exit_idx]) / 2
                    
                    # Enhanced annotation with background
                    bbox_props = dict(boxstyle="round,pad=0.3", facecolor=pnl_color, alpha=0.3)
                    ax.annotate(f'${pnl:+.0f}', xy=(mid_idx, mid_price),
                               xytext=(0, 15), textcoords='offset points',
                               fontsize=9, fontweight='bold', color=pnl_color,
                               ha='center', bbox=bbox_props, zorder=4)
            
        except Exception as e:
            self.log_operator_warning(f"Single trade plotting failed: {e}")

    def _plot_enhanced_pnl_curve(self, ax, trades: List[Dict]) -> None:
        """Plot enhanced cumulative P&L curve with additional metrics"""
        
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No Trades Available', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Sort trades by exit time
            sorted_trades = sorted(trades, key=lambda t: t.get('exit_idx', t.get('exit_step', 0)))
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            cum_sum = 0
            
            for trade in sorted_trades:
                cum_sum += trade.get('pnl', 0)
                cumulative_pnl.append(cum_sum)
            
            # Plot main P&L curve
            ax.plot(cumulative_pnl, color='darkblue', linewidth=3, label='Cumulative P&L')
            
            # Add fill areas for profit/loss
            ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                           where=[p >= 0 for p in cumulative_pnl],
                           color='green', alpha=0.3, label='Profit Periods')
            ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 0,
                           where=[p < 0 for p in cumulative_pnl],
                           color='red', alpha=0.3, label='Loss Periods')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.7, linewidth=1)
            
            # Add trend line
            if len(cumulative_pnl) > 2:
                x = np.arange(len(cumulative_pnl))
                trend_coef = np.polyfit(x, cumulative_pnl, 1)
                trend_line = np.polyval(trend_coef, x)
                ax.plot(x, trend_line, '--', color='orange', alpha=0.8, linewidth=2, label='Trend')
            
            # Enhance chart
            ax.set_title("Cumulative P&L Analysis", fontsize=12, fontweight='bold')
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Cumulative P&L ($)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Add performance metrics text
            if cumulative_pnl:
                final_pnl = cumulative_pnl[-1]
                max_pnl = max(cumulative_pnl)
                min_pnl = min(cumulative_pnl)
                
                metrics_text = f"Final: ${final_pnl:+.0f}\nMax: ${max_pnl:+.0f}\nMin: ${min_pnl:+.0f}"
                ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=8,
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
        except Exception as e:
            self.log_operator_warning(f"Enhanced P&L curve plotting failed: {e}")

    def _plot_enhanced_drawdown(self, ax, trades: List[Dict], info_bus: Optional[InfoBus]) -> None:
        """Plot enhanced drawdown analysis"""
        
        try:
            # Try to get drawdown history from InfoBus first
            if info_bus:
                viz_data = info_bus.get('module_data', {}).get('visualization_interface', {})
                dd_history = viz_data.get('performance_metrics', {}).get('drawdown', [])
                
                if dd_history:
                    drawdown_pct = [-dd * 100 for dd in dd_history]
                    ax.fill_between(range(len(drawdown_pct)), 0, drawdown_pct,
                                   color='red', alpha=0.5, label='Historical Drawdown')
                    ax.plot(drawdown_pct, color='darkred', linewidth=2)
                    
                    # Add max drawdown line
                    if drawdown_pct:
                        max_dd = min(drawdown_pct)
                        ax.axhline(y=max_dd, color='red', linestyle='--', alpha=0.8,
                                  label=f'Max DD: {max_dd:.1f}%')
            
            # Fallback: calculate from trades
            if not info_bus or not dd_history:
                self._calculate_drawdown_from_trades(ax, trades)
            
            # Enhance chart
            ax.set_title("Drawdown Analysis", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time/Trade")
            ax.set_ylabel("Drawdown (%)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.log_operator_warning(f"Enhanced drawdown plotting failed: {e}")

    def _calculate_drawdown_from_trades(self, ax, trades: List[Dict]) -> None:
        """Calculate and plot drawdown from trade data"""
        
        try:
            if not trades:
                return
            
            # Calculate cumulative P&L
            cumulative_pnl = []
            cum_sum = 0
            for trade in sorted(trades, key=lambda t: t.get('exit_idx', 0)):
                cum_sum += trade.get('pnl', 0)
                cumulative_pnl.append(cum_sum)
            
            # Calculate drawdown
            if cumulative_pnl:
                peak = cumulative_pnl[0]
                drawdowns = []
                
                for pnl in cumulative_pnl:
                    peak = max(peak, pnl)
                    dd = (peak - pnl) / abs(peak) if peak != 0 else 0
                    drawdowns.append(-dd * 100)
                
                # Plot drawdown
                ax.fill_between(range(len(drawdowns)), 0, drawdowns,
                               color='red', alpha=0.5)
                ax.plot(drawdowns, color='darkred', linewidth=2)
                
                # Add max drawdown annotation
                if drawdowns:
                    max_dd = min(drawdowns)
                    max_dd_idx = drawdowns.index(max_dd)
                    ax.axhline(y=max_dd, color='red', linestyle='--', alpha=0.8)
                    ax.annotate(f'Max DD: {max_dd:.1f}%', xy=(max_dd_idx, max_dd),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=8, color='red')
            
        except Exception as e:
            self.log_operator_warning(f"Drawdown calculation failed: {e}")

    def _plot_trade_metrics(self, ax, trades: List[Dict]) -> None:
        """Plot key trade metrics summary"""
        
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No Trade Data', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Calculate metrics
            pnls = [t.get('pnl', 0) for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            total_trades = len(trades)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_factor = sum(wins) / abs(sum(losses)) if losses and sum(losses) < 0 else float('inf')
            total_pnl = sum(pnls)
            
            # Create metrics display
            ax.axis('off')
            
            metrics_text = f"""TRADE METRICS
━━━━━━━━━━━━━━
Total Trades: {total_trades:,}
Win Rate: {win_rate:.1%}
Total P&L: ${total_pnl:+,.0f}

Avg Win: ${avg_win:+.0f}
Avg Loss: ${avg_loss:+.0f}
Profit Factor: {profit_factor:.2f}

Best Trade: ${max(pnls):+.0f}
Worst Trade: ${min(pnls):+.0f}"""
            
            # Color-coded background
            bg_color = 'lightgreen' if total_pnl > 0 else 'lightcoral' if total_pnl < 0 else 'lightgray'
            
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor=bg_color, alpha=0.3))
            
            ax.set_title("Performance Summary", fontsize=12, fontweight='bold')
            
        except Exception as e:
            self.log_operator_warning(f"Trade metrics plotting failed: {e}")

    def _plot_enhanced_trade_distribution(self, ax, trades: List[Dict]) -> None:
        """Plot enhanced trade P&L distribution with statistics"""
        
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No Trade Data', ha='center', va='center', transform=ax.transAxes)
                return
            
            pnls = [t.get('pnl', 0) for t in trades]
            
            # Create enhanced histogram
            n_bins = min(20, max(5, len(pnls) // 3))  # Adaptive bin count
            n, bins, patches = ax.hist(pnls, bins=n_bins, alpha=0.7, edgecolor='black')
            
            # Color bars based on profit/loss with gradient
            for i, (patch, left_edge, right_edge) in enumerate(zip(patches, bins[:-1], bins[1:])):
                mid_point = (left_edge + right_edge) / 2
                if mid_point >= 0:
                    # Green gradient for profits
                    intensity = min(1.0, mid_point / max(pnls) if max(pnls) > 0 else 1.0)
                    patch.set_facecolor((0, 0.8 * intensity + 0.2, 0))
                else:
                    # Red gradient for losses
                    intensity = min(1.0, abs(mid_point) / abs(min(pnls)) if min(pnls) < 0 else 1.0)
                    patch.set_facecolor((0.8 * intensity + 0.2, 0, 0))
            
            # Add statistical lines
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2, label='Break-even')
            ax.axvline(x=np.mean(pnls), color='blue', linestyle='-', linewidth=2,
                      label=f'Mean: ${np.mean(pnls):.0f}')
            ax.axvline(x=np.median(pnls), color='orange', linestyle='--', linewidth=2,
                      label=f'Median: ${np.median(pnls):.0f}')
            
            # Add standard deviation bands
            std_dev = np.std(pnls)
            mean_pnl = np.mean(pnls)
            ax.axvspan(mean_pnl - std_dev, mean_pnl + std_dev, alpha=0.2, color='gray', label='±1 StdDev')
            
            # Enhance chart
            ax.set_title("P&L Distribution Analysis", fontsize=12, fontweight='bold')
            ax.set_xlabel("P&L ($)")
            ax.set_ylabel("Frequency")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.log_operator_warning(f"Enhanced trade distribution plotting failed: {e}")

    def _plot_enhanced_win_loss_analysis(self, ax, trades: List[Dict]) -> None:
        """Plot enhanced win/loss analysis with detailed statistics"""
        
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No Trade Data', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Categorize trades
            wins = [t for t in trades if t.get('pnl', 0) > 0]
            losses = [t for t in trades if t.get('pnl', 0) < 0]
            breakevens = [t for t in trades if t.get('pnl', 0) == 0]
            
            # Prepare data for pie chart
            sizes = [len(wins), len(losses), len(breakevens)]
            labels = [f'Wins ({len(wins)})', f'Losses ({len(losses)})', f'B/E ({len(breakevens)})']
            colors = ['green', 'red', 'gray']
            explode = (0.1, 0.1, 0.05)  # Explode slices slightly
            
            # Create enhanced pie chart
            if sum(sizes) > 0:
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                                 autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                                                 startangle=90, explode=explode,
                                                 shadow=True, textprops={'fontsize': 9})
                
                # Enhance text appearance
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            
            # Add detailed statistics
            total_trades = len(trades)
            win_rate = len(wins) / total_trades if total_trades > 0 else 0
            avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
            avg_loss = np.mean([t['pnl'] for t in losses]) if losses else 0
            
            # Calculate additional metrics
            largest_win = max([t['pnl'] for t in wins]) if wins else 0
            largest_loss = min([t['pnl'] for t in losses]) if losses else 0
            win_streak = self._calculate_win_streak(trades)
            loss_streak = self._calculate_loss_streak(trades)
            
            stats_text = f"""DETAILED STATISTICS
Win Rate: {win_rate:.1%}
Avg Win: ${avg_win:+.0f}
Avg Loss: ${avg_loss:+.0f}

Largest Win: ${largest_win:+.0f}
Largest Loss: ${largest_loss:+.0f}

Max Win Streak: {win_streak}
Max Loss Streak: {loss_streak}"""
            
            ax.text(1.3, 0.5, stats_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
            
            ax.set_title("Win/Loss Analysis", fontsize=12, fontweight='bold')
            
        except Exception as e:
            self.log_operator_warning(f"Enhanced win/loss analysis failed: {e}")

    def _calculate_win_streak(self, trades: List[Dict]) -> int:
        """Calculate maximum winning streak"""
        try:
            max_streak = current_streak = 0
            for trade in sorted(trades, key=lambda t: t.get('exit_idx', 0)):
                if trade.get('pnl', 0) > 0:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            return max_streak
        except Exception:
            return 0

    def _calculate_loss_streak(self, trades: List[Dict]) -> int:
        """Calculate maximum losing streak"""
        try:
            max_streak = current_streak = 0
            for trade in sorted(trades, key=lambda t: t.get('exit_idx', 0)):
                if trade.get('pnl', 0) < 0:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            return max_streak
        except Exception:
            return 0

    def _plot_time_analysis(self, ax, trades: List[Dict]) -> None:
        """Plot time-based trading analysis"""
        
        try:
            if not trades:
                ax.text(0.5, 0.5, 'No Trade Data', ha='center', va='center', transform=ax.transAxes)
                return
            
            # Extract time-based data (using trade indices as time proxy)
            times = [t.get('exit_idx', t.get('exit_step', i)) for i, t in enumerate(trades)]
            pnls = [t.get('pnl', 0) for t in trades]
            
            # Create time-based P&L scatter
            colors = ['green' if pnl > 0 else 'red' if pnl < 0 else 'gray' for pnl in pnls]
            sizes = [abs(pnl) / 10 + 20 for pnl in pnls]  # Scale by P&L magnitude
            
            scatter = ax.scatter(times, pnls, c=colors, s=sizes, alpha=0.6, edgecolors='black')
            
            # Add trend line
            if len(times) > 1:
                z = np.polyfit(times, pnls, 1)
                p = np.poly1d(z)
                ax.plot(times, p(times), "r--", alpha=0.8, linewidth=2, label=f'Trend: {z[0]:+.2f}')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Enhance chart
            ax.set_title("Trade Performance Over Time", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time (Steps)")
            ax.set_ylabel("Trade P&L ($)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            self.log_operator_warning(f"Time analysis plotting failed: {e}")

    def _plot_system_status_panel(self, ax, info_bus: Optional[InfoBus]) -> None:
        """Plot comprehensive system status panel"""
        
        try:
            ax.axis('off')
            
            if not info_bus:
                ax.text(0.5, 0.5, 'No System Data Available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                return
            
            # Extract system information
            context = extract_standard_context(info_bus)
            risk_data = info_bus.get('risk', {})
            
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
💰 FINANCIAL STATUS:
   Balance: ${balance:,.2f}  |  Equity: ${equity:,.2f}  |  Drawdown: {drawdown:.1%}"""
            
            # Market status
            regime = context.get('regime', 'unknown').title()
            session = context.get('session', 'unknown').title()
            vol_level = context.get('volatility_level', 'medium').title()
            market_open = '🟢 OPEN' if context.get('market_open', True) else '🔴 CLOSED'
            
            market_text = f"""
📊 MARKET STATUS:
   Regime: {regime}  |  Session: {session}  |  Volatility: {vol_level}  |  Market: {market_open}"""
            
            # Trading activity
            positions = InfoBusExtractor.get_positions(info_bus)
            active_positions = len(positions)
            total_exposure = sum(abs(pos.get('size', 0)) for pos in positions)
            
            trading_text = f"""
⚡ TRADING ACTIVITY:
   Active Positions: {active_positions}  |  Total Exposure: {total_exposure:.2f}  |  Recent Trades: {len(info_bus.get('recent_trades', []))}"""
            
            # System modules status
            module_data = info_bus.get('module_data', {})
            active_modules = len(module_data)
            alerts = len(info_bus.get('alerts', []))
            
            system_text = f"""
🔧 SYSTEM MODULES:
   Active Modules: {active_modules}  |  Recent Alerts: {alerts}  |  System Health: {'✅ Good' if alerts < 5 else '⚠️ Attention'}"""
            
            # Combine all text
            full_text = status_text + financial_text + market_text + trading_text + system_text
            
            # Display with appropriate styling
            ax.text(0.02, 0.95, full_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.2))
            
        except Exception as e:
            self.log_operator_warning(f"System status panel failed: {e}")

    def create_performance_dashboard(
        self,
        info_bus: InfoBus,
        save: bool = True,
        title: str = "Performance Dashboard"
    ) -> Any:
        """Create comprehensive performance dashboard"""
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Apply style
            self._apply_chart_style()
            
            # Create large dashboard figure
            fig = plt.figure(figsize=(24, 16))
            gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
            
            # Extract data from InfoBus
            viz_data = info_bus.get('module_data', {}).get('visualization_interface', {})
            
            # 1. Balance evolution (large panel)
            ax1 = fig.add_subplot(gs[0, :2])
            self._plot_balance_evolution(ax1, viz_data)
            
            # 2. Current status panel
            ax2 = fig.add_subplot(gs[0, 2:])
            self._plot_dashboard_status(ax2, info_bus)
            
            # 3. P&L analysis
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_dashboard_pnl(ax3, viz_data)
            
            # 4. Drawdown tracking
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_dashboard_drawdown(ax4, viz_data)
            
            # 5. Risk metrics
            ax5 = fig.add_subplot(gs[1, 2])
            self._plot_dashboard_risk(ax5, viz_data)
            
            # 6. Module performance
            ax6 = fig.add_subplot(gs[1, 3])
            self._plot_module_performance_summary(ax6, info_bus)
            
            # 7. Regime analysis
            ax7 = fig.add_subplot(gs[2, :2])
            self._plot_regime_performance(ax7, viz_data)
            
            # 8. Trading activity
            ax8 = fig.add_subplot(gs[2, 2:])
            self._plot_trading_activity(ax8, info_bus)
            
            # 9. Alert summary
            ax9 = fig.add_subplot(gs[3, :])
            self._plot_alert_timeline(ax9, viz_data)
            
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
            self._record_chart_generation('performance_dashboard', [], info_bus)
            
            return fig
            
        except Exception as e:
            self.log_operator_error(f"Performance dashboard creation failed: {e}")
            return None

    def _apply_chart_style(self) -> None:
        """Apply consistent chart styling"""
        
        try:
            import matplotlib.pyplot as plt
            
            if self.style in self.available_styles:
                if self.style == "default":
                    plt.rcdefaults()
                else:
                    plt.style.use(self.style)
            else:
                plt.style.use('seaborn')
            
            # Apply custom enhancements
            plt.rcParams.update({
                'figure.facecolor': 'white',
                'axes.facecolor': 'white',
                'axes.edgecolor': 'black',
                'axes.linewidth': 1,
                'grid.alpha': 0.3,
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'legend.fontsize': 8
            })
            
        except Exception as e:
            self.log_operator_warning(f"Chart styling failed: {e}")

    def _save_chart(self, fig, filename: str) -> None:
        """Save chart with enhanced options"""
        
        try:
            if not self.save_path:
                return
            
            # Ensure save directory exists
            import os
            os.makedirs(self.save_path, exist_ok=True)
            
            # Save with multiple formats
            base_path = os.path.join(self.save_path, filename)
            
            # High-res PNG
            fig.savefig(f"{base_path}.png", dpi=self.chart_dpi, bbox_inches='tight')
            
            # PDF for scalability
            try:
                fig.savefig(f"{base_path}.pdf", bbox_inches='tight')
            except Exception:
                pass  # PDF backend might not be available
            
            self.viz_stats['charts_saved'] += 1
            
            self.log_operator_info(
                f"💾 Chart saved: {filename}",
                path=self.save_path,
                dpi=self.chart_dpi
            )
            
        except Exception as e:
            self.log_operator_warning(f"Chart saving failed: {e}")

    def _record_chart_generation(self, chart_type: str, trades: List[Dict], info_bus: Optional[InfoBus]) -> None:
        """Record chart generation for analytics"""
        
        try:
            chart_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'chart_type': chart_type,
                'trade_count': len(trades),
                'has_info_bus': info_bus is not None,
                'style': self.style,
                'marker_size': self.marker_size
            }
            
            self._chart_history.append(chart_record)
            
            # Limit history size
            if len(self._chart_history) > 100:
                self._chart_history = self._chart_history[-100:]
            
        except Exception as e:
            self.log_operator_warning(f"Chart generation recording failed: {e}")

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with visualizer status"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'trade_map_visualizer', {
            'charts_generated': self.viz_stats['charts_generated'],
            'last_chart_type': self._chart_history[-1]['chart_type'] if self._chart_history else None,
            'available_chart_types': list(self.chart_types.keys()),
            'style': self.style,
            'save_path': self.save_path,
            'auto_save': self.auto_save,
            'statistics': self.viz_stats.copy()
        })

    def get_visualization_report(self) -> str:
        """Generate operator-friendly visualization report"""
        
        chart_summary = ""
        if self._chart_history:
            recent_charts = self._chart_history[-5:]
            for chart in recent_charts:
                chart_summary += f"  • {chart['timestamp'][:19]}: {chart['chart_type']} ({chart['trade_count']} trades)\n"
        
        return f"""
🎨 TRADE MAP VISUALIZER REPORT
═══════════════════════════════════════
📊 Chart Generation Statistics:
• Total Charts: {self.viz_stats['charts_generated']}
• Trade Charts: {self.viz_stats['trade_charts']}
• Performance Charts: {self.viz_stats['performance_charts']}
• Dashboard Charts: {self.viz_stats['dashboard_charts']}
• Charts Saved: {self.viz_stats['charts_saved']}

⚙️ Configuration:
• Style: {self.style}
• Marker Size: {self.marker_size}
• Auto Save: {'✅ Enabled' if self.auto_save else '❌ Disabled'}
• Save Path: {self.save_path or 'Not set'}
• Chart DPI: {self.chart_dpi}

📈 Available Chart Types:
{chr(10).join([f'  • {name}: {desc}' for name, desc in self.chart_types.items()])}

🕒 Recent Charts Generated:
{chart_summary if chart_summary else '  📭 No recent charts'}

💾 Storage:
• Chart History: {len(self._chart_history)} entries
• Cache Size: {len(self._chart_cache)} items
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
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "debug": self.debug,
                "marker_size": self.marker_size,
                "style": self.style,
                "save_path": self.save_path,
                "auto_save": self.auto_save,
                "chart_dpi": self.chart_dpi
            },
            "statistics": self.viz_stats.copy(),
            "history": {
                "chart_history": self._chart_history[-20:],  # Recent only
                "cache_keys": list(self._chart_cache.keys())
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.debug = bool(config.get("debug", self.debug))
        self.marker_size = int(config.get("marker_size", self.marker_size))
        self.style = config.get("style", self.style)
        self.save_path = config.get("save_path", self.save_path)
        self.auto_save = bool(config.get("auto_save", self.auto_save))
        self.chart_dpi = int(config.get("chart_dpi", self.chart_dpi))
        
        # Load statistics
        self.viz_stats.update(state.get("statistics", {}))
        
        # Load history
        history = state.get("history", {})
        self._chart_history = history.get("chart_history", [])
        # Note: Don't restore cache as it contains matplotlib objects

    # ================== ADDITIONAL CHART METHODS ==================
    
    # Implementation stubs for dashboard-specific plotting methods
    def _plot_balance_evolution(self, ax, viz_data): pass
    def _plot_dashboard_status(self, ax, info_bus): pass  
    def _plot_dashboard_pnl(self, ax, viz_data): pass
    def _plot_dashboard_drawdown(self, ax, viz_data): pass
    def _plot_dashboard_risk(self, ax, viz_data): pass
    def _plot_module_performance_summary(self, ax, info_bus): pass
    def _plot_regime_performance(self, ax, viz_data): pass
    def _plot_trading_activity(self, ax, info_bus): pass
    def _plot_alert_timeline(self, ax, viz_data): pass
    def _add_chart_indicators(self, ax, prices, info_bus): pass
    def _add_regime_backgrounds(self, ax, info_bus): pass
    def _generate_auto_chart(self, viz_data, context): pass