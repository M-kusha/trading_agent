import numpy as np
from typing import List, Dict, Any, Optional, Union
from modules.core.core import Module
from modules.utils.info_bus import InfoBus
import copy
import random
import datetime
import json

class VisualizationInterface(Module):
    """
    Enhanced real-time visualization interface with InfoBus integration.
    Supports live streaming to dashboards and comprehensive data collection.
    """
    
    def __init__(self, debug: bool = True, max_steps: int = 500, 
                 stream_url: Optional[str] = None):
        super().__init__()
        self.debug = debug
        self.max_steps = max_steps
        self.stream_url = stream_url
        self.reset()
        
    def reset(self) -> None:
        """Clear all visualization data"""
        self.records: List[Dict[str, Any]] = []
        self._decision_trace: List[Dict[str, Any]] = []
        self._performance_metrics: Dict[str, List[float]] = {
            'pnl': [], 'drawdown': [], 'trades': [], 'balance': []
        }
        self._alert_history: List[Dict[str, Any]] = []
        
    def step(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Record visualization data from InfoBus or kwargs"""
        if info_bus:
            self._process_info_bus(info_bus)
        else:
            self.record_step(**kwargs)
            
    def _process_info_bus(self, info_bus: InfoBus) -> None:
        """Extract visualization data from InfoBus"""
        record = {
            '_time': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
            'step': info_bus.get('step_idx', 0),
            'episode': info_bus.get('episode_idx', 0),
            'balance': info_bus.get('risk', {}).get('balance', 0),
            'equity': info_bus.get('risk', {}).get('equity', 0),
            'drawdown': info_bus.get('risk', {}).get('current_drawdown', 0),
            'positions': len(info_bus.get('positions', [])),
            'pnl_today': info_bus.get('pnl_today', 0),
            'trade_count': info_bus.get('trade_count', 0),
            'win_rate': info_bus.get('win_rate', 0),
            'consensus': info_bus.get('consensus', 0),
            'regime': info_bus.get('market_context', {}).get('regime', 'unknown'),
            'alerts': info_bus.get('alerts', [])
        }
        
        # Update performance metrics
        self._performance_metrics['balance'].append(record['balance'])
        self._performance_metrics['drawdown'].append(record['drawdown'])
        self._performance_metrics['pnl'].append(record['pnl_today'])
        self._performance_metrics['trades'].append(record['trade_count'])
        
        # Process alerts
        for alert in record.get('alerts', []):
            self._alert_history.append({
                'time': record['_time'],
                'alert': alert,
                'step': record['step']
            })
            
        self._add_record(record)
        
    def record_step(self, **kwargs) -> None:
        """Legacy recording method"""
        entry = dict(kwargs)
        entry['_time'] = datetime.datetime.now().isoformat()
        self._add_record(entry)
        
    def _add_record(self, record: Dict[str, Any]) -> None:
        """Add record with size management"""
        self.records.append(record)
        if len(self.records) > self.max_steps:
            self.records = self.records[-self.max_steps:]
            
        self._decision_trace.append(record)
        if len(self._decision_trace) > self.max_steps:
            self._decision_trace = self._decision_trace[-self.max_steps:]
            
        if self.debug:
            self._print_summary(record)
            
        self.stream_update(record)
        
    def _print_summary(self, record: Dict[str, Any]) -> None:
        """Print formatted summary"""
        print(f"[Viz] Step {record.get('step', 0)} | "
              f"Balance: ${record.get('balance', 0):.2f} | "
              f"P&L: ${record.get('pnl_today', 0):.2f} | "
              f"DD: {record.get('drawdown', 0):.1%} | "
              f"Pos: {record.get('positions', 0)}")
              
    def stream_update(self, entry: Dict[str, Any]) -> None:
        """Stream to dashboard or external service"""
        if self.stream_url:
            try:
                # Implement actual streaming (WebSocket, HTTP POST, etc.)
                # This is a placeholder
                import requests
                requests.post(self.stream_url, json=entry, timeout=0.5)
            except:
                pass  # Don't let streaming errors affect trading
                
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data"""
        if not self.records:
            return {}
            
        recent = self.records[-100:]  # Last 100 steps
        
        return {
            'current': self.records[-1] if self.records else {},
            'performance': {
                'balance_history': self._performance_metrics['balance'][-100:],
                'pnl_history': self._performance_metrics['pnl'][-100:],
                'drawdown_history': self._performance_metrics['drawdown'][-100:],
                'cumulative_trades': self._performance_metrics['trades'][-100:]
            },
            'statistics': {
                'total_pnl': sum(self._performance_metrics['pnl']),
                'max_drawdown': max(self._performance_metrics['drawdown']) if self._performance_metrics['drawdown'] else 0,
                'total_trades': self._performance_metrics['trades'][-1] if self._performance_metrics['trades'] else 0,
                'recent_alerts': self._alert_history[-10:]
            },
            'regime_distribution': self._calculate_regime_distribution(recent)
        }
        
    def _calculate_regime_distribution(self, records: List[Dict]) -> Dict[str, float]:
        """Calculate time spent in each regime"""
        regime_counts = {}
        for record in records:
            regime = record.get('regime', 'unknown')
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
        total = len(records)
        return {k: v/total for k, v in regime_counts.items()}
        
    def generate_performance_report(self) -> str:
        """Generate text performance report"""
        data = self.get_dashboard_data()
        stats = data.get('statistics', {})
        
        report = f"""
Performance Report
==================
Generated: {datetime.datetime.now().isoformat()}

Summary:
- Total P&L: ${stats.get('total_pnl', 0):.2f}
- Max Drawdown: {stats.get('max_drawdown', 0):.1%}
- Total Trades: {stats.get('total_trades', 0)}

Recent Activity:
"""
        for alert in stats.get('recent_alerts', [])[-5:]:
            report += f"- [{alert['time']}] {alert['alert']}\n"
            
        return report
        
    def get_observation_components(self) -> np.ndarray:
        """Return visualization metrics"""
        if not self.records:
            return np.zeros(4, dtype=np.float32)
            
        recent = self.records[-10:]
        avg_balance = np.mean([r.get('balance', 0) for r in recent])
        avg_drawdown = np.mean([r.get('drawdown', 0) for r in recent])
        
        return np.array([
            float(len(self.records)),
            avg_balance / 10000,  # Normalized
            avg_drawdown,
            float(len(self._alert_history))
        ], dtype=np.float32)
        
    # Evolutionary methods
    def mutate(self, std: float = 1.0) -> None:
        """Mutate visualization parameters"""
        self.max_steps = max(50, int(self.max_steps + np.random.normal(0, std * 50)))
        if random.random() < 0.2:
            self.debug = not self.debug
            
    def crossover(self, other: 'VisualizationInterface') -> 'VisualizationInterface':
        """Create offspring visualizer"""
        child = VisualizationInterface(
            debug=self.debug if random.random() < 0.5 else other.debug,
            max_steps=self.max_steps if random.random() < 0.5 else other.max_steps,
            stream_url=self.stream_url if random.random() < 0.5 else other.stream_url
        )
        return child
        
    def get_state(self) -> Dict[str, Any]:
        """Save visualization state"""
        return {
            'debug': self.debug,
            'max_steps': self.max_steps,
            'stream_url': self.stream_url,
            'records': copy.deepcopy(self.records[-100:]),  # Save recent only
            '_performance_metrics': copy.deepcopy(self._performance_metrics),
            '_alert_history': copy.deepcopy(self._alert_history[-50:])
        }
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore visualization state"""
        self.debug = state.get('debug', self.debug)
        self.max_steps = state.get('max_steps', self.max_steps)
        self.stream_url = state.get('stream_url', self.stream_url)
        self.records = state.get('records', [])
        self._performance_metrics = state.get('_performance_metrics', 
                                            {'pnl': [], 'drawdown': [], 'trades': [], 'balance': []})
        self._alert_history = state.get('_alert_history', [])


class TradeMapVisualizer(Module):
    """
    Enhanced trade visualization with multiple chart types and InfoBus integration.
    """
    
    def __init__(self, debug: bool = False, marker_size: int = 60, 
                 style: str = "seaborn", save_path: Optional[str] = None):
        super().__init__()
        self.debug = debug
        self.marker_size = marker_size
        self.style = style
        self.save_path = save_path
        self._last_fig = None
        self._chart_cache: Dict[str, Any] = {}
        
    def reset(self) -> None:
        """Clear visualization cache"""
        self._last_fig = None
        self._chart_cache.clear()
        
    def step(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Optional real-time plotting"""
        if info_bus and self.debug:
            # Could create live plots during trading
            pass
            
    def plot_trades(
        self, 
        prices: Union[List[float], Dict[str, List[float]]], 
        trades: List[Dict], 
        info_bus: Optional[InfoBus] = None,
        show: bool = False, 
        title: str = "Trade Map",
        plot_indicators: bool = True
    ) -> Any:
        """Create comprehensive trade visualization"""
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.gridspec import GridSpec
        
        # Apply style
        if self.style != "default":
            try:
                plt.style.use(self.style)
            except:
                plt.style.use('seaborn')
                
        # Create figure with subplots
        fig = plt.figure(figsize=(15, 10))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1])
        
        # Main price chart
        ax_main = fig.add_subplot(gs[0, :])
        
        # Handle multi-instrument data
        if isinstance(prices, dict):
            for inst, price_series in prices.items():
                self._plot_instrument(ax_main, price_series, trades, inst)
        else:
            self._plot_instrument(ax_main, prices, trades, "Main")
            
        ax_main.set_title(title, fontsize=14, fontweight='bold')
        ax_main.set_xlabel("Time Step")
        ax_main.set_ylabel("Price")
        ax_main.legend(loc='best')
        ax_main.grid(True, alpha=0.3)
        
        # P&L chart
        ax_pnl = fig.add_subplot(gs[1, 0])
        self._plot_pnl_curve(ax_pnl, trades)
        
        # Drawdown chart
        ax_dd = fig.add_subplot(gs[1, 1])
        self._plot_drawdown(ax_dd, trades, info_bus)
        
        # Trade distribution
        ax_dist = fig.add_subplot(gs[2, 0])
        self._plot_trade_distribution(ax_dist, trades)
        
        # Win/Loss analysis
        ax_winloss = fig.add_subplot(gs[2, 1])
        self._plot_win_loss_analysis(ax_winloss, trades)
        
        fig.tight_layout()
        self._last_fig = fig
        
        if self.save_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(f"{self.save_path}/trade_map_{timestamp}.png", dpi=150)
            
        if show:
            plt.show()
            
        return fig
        
    def _plot_instrument(self, ax, prices: List[float], trades: List[Dict], 
                        instrument: str) -> None:
        """Plot single instrument with trades"""
        prices_np = np.asarray(prices)
        ax.plot(prices_np, label=f"{instrument} Price", linewidth=1.5)
        
        # Plot trades for this instrument
        inst_trades = [t for t in trades if t.get('instrument', '') == instrument or instrument == "Main"]
        
        for tr in inst_trades:
            ei = int(tr.get('entry_idx', tr.get('entry_step', 0)))
            xi = int(tr.get('exit_idx', tr.get('exit_step', ei + tr.get('duration', 1))))
            pnl = tr.get('pnl', 0)
            size = abs(tr.get('size', 1))
            
            # Color and marker based on P&L and side
            color = 'green' if pnl >= 0 else 'red'
            side = tr.get('side', '').lower()
            entry_marker = '^' if side in ['buy', 'long'] else 'v'
            exit_marker = 'v' if side in ['buy', 'long'] else '^'
            
            # Scale marker size by position size
            marker_scale = min(max(size, 0.5), 2.0)
            
            # Plot entry
            if 0 <= ei < len(prices_np):
                ax.scatter([ei], [prices_np[ei]], marker=entry_marker, 
                          c=color, s=self.marker_size * marker_scale, 
                          edgecolors='black', linewidth=1, alpha=0.8)
                          
            # Plot exit
            if 0 <= xi < len(prices_np):
                ax.scatter([xi], [prices_np[xi]], marker=exit_marker, 
                          c=color, s=self.marker_size * marker_scale, 
                          edgecolors='black', linewidth=1, alpha=0.8)
                          
            # Connect entry and exit
            if 0 <= ei < len(prices_np) and 0 <= xi < len(prices_np):
                ax.plot([ei, xi], [prices_np[ei], prices_np[xi]], 
                       c=color, alpha=0.4, linestyle='--' if pnl < 0 else '-',
                       linewidth=1.5 * marker_scale)
                       
                # Add P&L annotation for significant trades
                if abs(pnl) > 50:  # Threshold for annotation
                    mid_idx = (ei + xi) // 2
                    mid_price = (prices_np[ei] + prices_np[xi]) / 2
                    ax.annotate(f'${pnl:.0f}', xy=(mid_idx, mid_price),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, color=color)
                               
    def _plot_pnl_curve(self, ax, trades: List[Dict]) -> None:
        """Plot cumulative P&L"""
        if not trades:
            return
            
        cumulative_pnl = []
        cum_sum = 0
        
        for trade in sorted(trades, key=lambda t: t.get('exit_idx', t.get('exit_step', 0))):
            cum_sum += trade.get('pnl', 0)
            cumulative_pnl.append(cum_sum)
            
        ax.plot(cumulative_pnl, color='blue', linewidth=2)
        ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 
                       where=[p >= 0 for p in cumulative_pnl], 
                       color='green', alpha=0.3)
        ax.fill_between(range(len(cumulative_pnl)), cumulative_pnl, 
                       where=[p < 0 for p in cumulative_pnl], 
                       color='red', alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax.set_title("Cumulative P&L")
        ax.set_xlabel("Trade #")
        ax.set_ylabel("P&L ($)")
        ax.grid(True, alpha=0.3)
        
    def _plot_drawdown(self, ax, trades: List[Dict], info_bus: Optional[InfoBus]) -> None:
        """Plot drawdown chart"""
        if info_bus and 'risk' in info_bus:
            # Use historical drawdown from InfoBus if available
            dd_history = info_bus.get('module_data', {}).get('drawdown_history', [])
            if dd_history:
                ax.fill_between(range(len(dd_history)), 0, 
                               [-dd * 100 for dd in dd_history],
                               color='red', alpha=0.5)
                ax.plot([-dd * 100 for dd in dd_history], color='darkred', linewidth=2)
        else:
            # Calculate from trades
            cumulative_pnl = []
            cum_sum = 0
            for trade in sorted(trades, key=lambda t: t.get('exit_idx', 0)):
                cum_sum += trade.get('pnl', 0)
                cumulative_pnl.append(cum_sum)
                
            if cumulative_pnl:
                peak = 0
                drawdowns = []
                for pnl in cumulative_pnl:
                    peak = max(peak, pnl)
                    dd = (peak - pnl) / peak if peak > 0 else 0
                    drawdowns.append(-dd * 100)
                    
                ax.fill_between(range(len(drawdowns)), 0, drawdowns,
                               color='red', alpha=0.5)
                ax.plot(drawdowns, color='darkred', linewidth=2)
                
        ax.set_title("Drawdown %")
        ax.set_xlabel("Time")
        ax.set_ylabel("Drawdown (%)")
        ax.grid(True, alpha=0.3)
        
    def _plot_trade_distribution(self, ax, trades: List[Dict]) -> None:
        """Plot trade P&L distribution"""
        if not trades:
            return
            
        pnls = [t.get('pnl', 0) for t in trades]
        
        # Create histogram
        n, bins, patches = ax.hist(pnls, bins=20, alpha=0.7, edgecolor='black')
        
        # Color bars based on profit/loss
        for i, (patch, left_edge) in enumerate(zip(patches, bins[:-1])):
            if left_edge >= 0:
                patch.set_facecolor('green')
            else:
                patch.set_facecolor('red')
                
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.7)
        ax.axvline(x=np.mean(pnls), color='blue', linestyle='-', 
                  label=f'Mean: ${np.mean(pnls):.2f}')
        ax.set_title("P&L Distribution")
        ax.set_xlabel("P&L ($)")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    def _plot_win_loss_analysis(self, ax, trades: List[Dict]) -> None:
        """Plot win/loss analysis"""
        if not trades:
            return
            
        wins = [t['pnl'] for t in trades if t.get('pnl', 0) > 0]
        losses = [abs(t['pnl']) for t in trades if t.get('pnl', 0) < 0]
        
        # Summary statistics
        total_trades = len(trades)
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        profit_factor = sum(wins) / sum(losses) if losses and sum(losses) > 0 else float('inf')
        
        # Create pie chart
        sizes = [len(wins), len(losses)]
        labels = [f'Wins ({len(wins)})', f'Losses ({len(losses)})']
        colors = ['green', 'red']
        
        if sum(sizes) > 0:
            wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors,
                                              autopct='%1.1f%%', startangle=90)
                                              
        # Add statistics text
        stats_text = f"Win Rate: {win_rate:.1%}\n"
        stats_text += f"Avg Win: ${avg_win:.2f}\n"
        stats_text += f"Avg Loss: ${avg_loss:.2f}\n"
        stats_text += f"Profit Factor: {profit_factor:.2f}" if profit_factor != float('inf') else "Profit Factor: ∞"
        
        ax.text(0.5, -1.3, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title("Win/Loss Analysis")
        
    def create_performance_dashboard(
        self, 
        info_bus: InfoBus,
        save: bool = True
    ) -> Any:
        """Create comprehensive performance dashboard"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig)
        
        # Extract data from InfoBus
        viz_data = info_bus.get('module_data', {}).get('visualization', {})
        
        # 1. Balance chart
        ax1 = fig.add_subplot(gs[0, :2])
        if 'balance_history' in viz_data:
            ax1.plot(viz_data['balance_history'], linewidth=2)
            ax1.set_title("Account Balance", fontsize=14)
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Balance ($)")
            ax1.grid(True, alpha=0.3)
            
        # 2. Current status panel
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        status_text = self._create_status_text(info_bus)
        ax2.text(0.1, 0.9, status_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
                
        # Continue with other panels...
        
        fig.tight_layout()
        
        if save and self.save_path:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fig.savefig(f"{self.save_path}/dashboard_{timestamp}.png", dpi=150)
            
        return fig
        
    def _create_status_text(self, info_bus: InfoBus) -> str:
        """Create status summary text"""
        risk = info_bus.get('risk', {})
        
        text = "Current Status\n"
        text += "=" * 20 + "\n"
        text += f"Balance: ${risk.get('balance', 0):,.2f}\n"
        text += f"Equity: ${risk.get('equity', 0):,.2f}\n"
        text += f"Drawdown: {risk.get('current_drawdown', 0):.1%}\n"
        text += f"Positions: {len(info_bus.get('positions', []))}\n"
        text += f"Today P&L: ${info_bus.get('pnl_today', 0):,.2f}\n"
        text += f"Win Rate: {info_bus.get('win_rate', 0):.1%}\n"
        text += f"Regime: {info_bus.get('market_context', {}).get('regime', 'Unknown')}"
        
        return text
        
    def get_observation_components(self) -> np.ndarray:
        """Return visualization parameters"""
        styles = ["default", "ggplot", "seaborn", "bmh", "classic"]
        style_idx = styles.index(self.style) if self.style in styles else 0
        
        return np.array([
            float(self.marker_size),
            float(self.debug),
            float(style_idx),
            float(self._last_fig is not None)
        ], dtype=np.float32)
        
    # Evolutionary methods
    def mutate(self, std: float = 1.0) -> None:
        """Mutate visualization parameters"""
        self.marker_size = max(10, int(self.marker_size + np.random.normal(0, std * 10)))
        if random.random() < 0.2:
            self.debug = not self.debug
        if random.random() < 0.1:
            self.style = random.choice(["default", "ggplot", "seaborn", "bmh", "classic"])
            
    def crossover(self, other: 'TradeMapVisualizer') -> 'TradeMapVisualizer':
        """Create offspring visualizer"""
        return TradeMapVisualizer(
            debug=self.debug if random.random() < 0.5 else other.debug,
            marker_size=self.marker_size if random.random() < 0.5 else other.marker_size,
            style=self.style if random.random() < 0.5 else other.style,
            save_path=self.save_path if random.random() < 0.5 else other.save_path
        )
        
    def get_state(self) -> Dict[str, Any]:
        """Save visualizer state"""
        return {
            'debug': self.debug,
            'marker_size': self.marker_size,
            'style': self.style,
            'save_path': self.save_path,
            '_chart_cache': list(self._chart_cache.keys())  # Just save keys
        }
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore visualizer state"""
        self.debug = state.get('debug', self.debug)
        self.marker_size = state.get('marker_size', self.marker_size)
        self.style = state.get('style', self.style)
        self.save_path = state.get('save_path', self.save_path)