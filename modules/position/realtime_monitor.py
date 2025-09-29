"""
Real-time monitoring of Position Manager buy/sell decisions
Shows live trading activity with clear indicators
"""

import time
import datetime
import threading
from typing import Dict, Any, List, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TradeSignal:
    """Real-time trade signal"""
    timestamp: datetime.datetime
    instrument: str
    action: str  # BUY, SELL, HOLD
    size_eur: float
    confidence: float
    executed: bool
    price: float
    pnl: Optional[float] = None


class RealtimeMonitor:
    """
    Real-time monitoring of position decisions
    Tracks buying/selling with clear visual indicators
    """
    
    def __init__(self, 
                 position_manager: Any,
                 refresh_interval: float = 1.0,
                 display_mode: str = "console"):
        """
        Initialize real-time monitor
        
        Args:
            position_manager: The PositionManager instance
            refresh_interval: Update interval in seconds
            display_mode: Display mode (console, file, both)
        """
        
        self.pm = position_manager
        self.refresh_interval = refresh_interval
        self.display_mode = display_mode
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Signal tracking
        self.active_signals: Dict[str, TradeSignal] = {}
        self.signal_history: Deque[TradeSignal] = deque(maxlen=1000)
        
        # Statistics
        self.stats = defaultdict(lambda: {
            'total_buys': 0,
            'total_sells': 0,
            'total_holds': 0,
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'win_count': 0,
            'loss_count': 0,
            'total_pnl': 0.0
        })
        
        # Display state
        self._running = False
        self._monitor_thread = None
        
        # Output files
        self.output_dir = Path("logs/monitor")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.monitor_file = self.output_dir / f"monitor_{timestamp}.txt"
        self.signals_file = self.output_dir / f"signals_{timestamp}.json"
    
    def start(self):
        """Start real-time monitoring"""
        
        if self._running:
            print("Monitor already running")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        print(f"✅ Real-time Monitor Started")
        print(f"📊 Tracking instruments: {', '.join(self.pm.instruments)}")
        print(f"💾 Logging to: {self.output_dir}")
        print("="*80)
    
    def stop(self):
        """Stop monitoring"""
        
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        
        print("\n❌ Monitor Stopped")
        self.print_final_summary()
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        
        last_update = time.time()
        
        while self._running:
            try:
                current_time = time.time()
                
                if current_time - last_update >= self.refresh_interval:
                    self._update_signals()
                    self._display_status()
                    last_update = current_time
                
                time.sleep(0.1)
                
            except Exception as e:
                print(f"Monitor error: {e}")
    
    def _update_signals(self):
        """Update signal tracking from Position Manager"""
        
        try:
            # Get latest decisions
            if hasattr(self.pm, 'last_decisions'):
                for instrument, decision_result in self.pm.last_decisions.items():
                    self._process_decision(instrument, decision_result)
            
            # Get positions
            if hasattr(self.pm, 'open_positions'):
                self._update_positions(self.pm.open_positions)
            
            # Check order queue
            if hasattr(self.pm, 'smart_bus'):
                order_queue = self.pm.smart_bus.get('order_queue', 'RealtimeMonitor')
                if isinstance(order_queue, list):
                    self._process_order_queue(order_queue)
                    
        except Exception as e:
            print(f"Signal update error: {e}")
    
    def _process_decision(self, instrument: str, decision_result: Any):
        """Process a position decision"""
        
        try:
            decision = decision_result.decision.value
            intensity = decision_result.intensity
            size = decision_result.size
            confidence = decision_result.confidence
            
            # Determine action type
            if "open_long" in decision or (intensity > 0 and "scale_up" in decision):
                action = "BUY"
            elif "open_short" in decision or (intensity < 0 and "scale_up" in decision):
                action = "SELL"
            elif "close" in decision or "scale_down" in decision:
                action = "CLOSE"
            else:
                action = "HOLD"
            
            # Get current price
            price = self._get_current_price(instrument)
            
            # Create signal
            signal = TradeSignal(
                timestamp=datetime.datetime.now(),
                instrument=instrument,
                action=action,
                size_eur=size,
                confidence=confidence,
                executed=False,
                price=price
            )
            
            # Check if this is a new signal
            with self._lock:
                existing = self.active_signals.get(instrument)
                
                if not existing or self._is_new_signal(existing, signal):
                    self.active_signals[instrument] = signal
                    self.signal_history.append(signal)
                    
                    # Update stats
                    if action == "BUY":
                        self.stats[instrument]['total_buys'] += 1
                        self.stats[instrument]['buy_volume'] += size
                        self._alert_buy(instrument, signal)
                    elif action == "SELL":
                        self.stats[instrument]['total_sells'] += 1
                        self.stats[instrument]['sell_volume'] += size
                        self._alert_sell(instrument, signal)
                    elif action == "HOLD":
                        self.stats[instrument]['total_holds'] += 1
                        
        except Exception as e:
            print(f"Decision processing error: {e}")
    
    def _is_new_signal(self, existing: TradeSignal, new: TradeSignal) -> bool:
        """Check if signal is new"""
        
        # Different action
        if existing.action != new.action:
            return True
        
        # Significant size change
        if abs(existing.size_eur - new.size_eur) > existing.size_eur * 0.1:
            return True
        
        # Time gap
        time_diff = (new.timestamp - existing.timestamp).total_seconds()
        if time_diff > 60:  # More than 1 minute
            return True
        
        return False
    
    def _get_current_price(self, instrument: str) -> float:
        """Get current price for instrument"""
        
        try:
            if hasattr(self.pm, 'smart_bus'):
                price_data = self.pm.smart_bus.get('price_data', 'RealtimeMonitor')
                if isinstance(price_data, dict):
                    inst_data = price_data.get(instrument, {})
                    if isinstance(inst_data, dict):
                        return float(inst_data.get('last', 0.0))
        except Exception:
            pass
        
        return 0.0
    
    def _update_positions(self, positions: Dict[str, Any]):
        """Update position P&L"""
        
        for instrument, position in positions.items():
            if instrument in self.active_signals:
                signal = self.active_signals[instrument]
                
                # Get unrealized P&L
                pnl = self._get_position_pnl(instrument)
                if pnl is not None:
                    signal.pnl = pnl
                    
                    # Update stats
                    if pnl > 0:
                        self.stats[instrument]['win_count'] = 1
                    elif pnl < 0:
                        self.stats[instrument]['loss_count'] = 1
    
    def _get_position_pnl(self, instrument: str) -> Optional[float]:
        """Get position P&L"""
        
        try:
            if hasattr(self.pm, 'smart_bus'):
                positions = self.pm.smart_bus.get('positions', 'RealtimeMonitor')
                if isinstance(positions, dict):
                    pos = positions.get(instrument, {})
                    if isinstance(pos, dict):
                        return float(pos.get('unrealized_pnl_eur', 0.0))
        except Exception:
            pass
        
        return None
    
    def _process_order_queue(self, orders: List[Dict[str, Any]]):
        """Process order queue"""
        
        for order in orders:
            instrument = order.get('instrument')
            if instrument in self.active_signals:
                self.active_signals[instrument].executed = True
    
    def _alert_buy(self, instrument: str, signal: TradeSignal):
        """Alert for BUY signal"""
        
        if self.display_mode in ["console", "both"]:
            print(f"\n{'🟢'*20} BUY SIGNAL {'🟢'*20}")
            print(f"📍 {instrument}")
            print(f"💰 Size: €{signal.size_eur:,.2f}")
            print(f"📊 Confidence: {signal.confidence:.1%}")
            print(f"💵 Price: {signal.price:.5f}")
            print(f"⏰ Time: {signal.timestamp.strftime('%H:%M:%S')}")
            print(f"{'🟢'*50}\n")
    
    def _alert_sell(self, instrument: str, signal: TradeSignal):
        """Alert for SELL signal"""
        
        if self.display_mode in ["console", "both"]:
            print(f"\n{'🔴'*20} SELL SIGNAL {'🔴'*20}")
            print(f"📍 {instrument}")
            print(f"💰 Size: €{signal.size_eur:,.2f}")
            print(f"📊 Confidence: {signal.confidence:.1%}")
            print(f"💵 Price: {signal.price:.5f}")
            print(f"⏰ Time: {signal.timestamp.strftime('%H:%M:%S')}")
            print(f"{'🔴'*50}\n")
    
    def _display_status(self):
        """Display current status"""
        
        if self.display_mode not in ["console", "both"]:
            return
        
        # Clear screen (optional)
        # print("\033[2J\033[H")  # Uncomment for screen clear
        
        print("\n" + "="*80)
        print(f"📊 REAL-TIME MONITOR - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Portfolio overview
        balance, _ = self.pm._read_balance_and_drawdown()
        print(f"\n💰 Balance: €{balance:,.2f}")
        print(f"🏥 Health: {self.pm._portfolio_health_score:.1%}")
        print(f"📈 Exposure: {self.pm._total_exposure_ratio:.1%}")
        
        # Active signals
        print(f"\n🎯 ACTIVE SIGNALS:")
        print("-"*40)
        
        for instrument, signal in self.active_signals.items():
            status = "✅" if signal.executed else "⏳"
            
            if signal.action == "BUY":
                arrow = "↗️"
                color = "🟢"
            elif signal.action == "SELL":
                arrow = "↘️"
                color = "🔴"
            else:
                arrow = "→"
                color = "⚪"
            
            pnl_str = ""
            if signal.pnl is not None:
                if signal.pnl > 0:
                    pnl_str = f" | P&L: +€{signal.pnl:.2f} ✅"
                else:
                    pnl_str = f" | P&L: -€{abs(signal.pnl):.2f} ❌"
            
            print(f"{color} {instrument}: {signal.action} {arrow} €{signal.size_eur:,.0f} "
                  f"({signal.confidence:.0%}) {status}{pnl_str}")
        
        # Statistics
        print(f"\n📈 STATISTICS:")
        print("-"*40)
        
        total_buys = sum(s['total_buys'] for s in self.stats.values())
        total_sells = sum(s['total_sells'] for s in self.stats.values())
        total_holds = sum(s['total_holds'] for s in self.stats.values())
        total_buy_volume = sum(s['buy_volume'] for s in self.stats.values())
        total_sell_volume = sum(s['sell_volume'] for s in self.stats.values())
        
        print(f"Total Buys: {total_buys} (€{total_buy_volume:,.0f})")
        print(f"Total Sells: {total_sells} (€{total_sell_volume:,.0f})")
        print(f"Total Holds: {total_holds}")
        
        if total_buys + total_sells > 0:
            buy_ratio = total_buys / (total_buys + total_sells) * 100
            print(f"Buy/Sell Ratio: {buy_ratio:.1f}% / {100-buy_ratio:.1f}%")
        
        # Save to file
        if self.display_mode in ["file", "both"]:
            self._save_status_to_file()
    
    def _save_status_to_file(self):
        """Save current status to file"""
        
        try:
            # Save text status
            with open(self.monitor_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                f.write(f"Balance: €{self.pm._read_balance_and_drawdown()[0]:,.2f}\n")
                f.write(f"Health: {self.pm._portfolio_health_score:.1%}\n")
                f.write(f"Exposure: {self.pm._total_exposure_ratio:.1%}\n")
                
                for instrument, signal in self.active_signals.items():
                    f.write(f"{instrument}: {signal.action} - €{signal.size_eur:,.2f} "
                           f"({signal.confidence:.1%}) - "
                           f"{'Executed' if signal.executed else 'Pending'}\n")
            
            # Save JSON signals
            signals_data = []
            for signal in list(self.signal_history)[-100:]:  # Last 100 signals
                signals_data.append({
                    'timestamp': signal.timestamp.isoformat(),
                    'instrument': signal.instrument,
                    'action': signal.action,
                    'size_eur': signal.size_eur,
                    'confidence': signal.confidence,
                    'executed': signal.executed,
                    'price': signal.price,
                    'pnl': signal.pnl
                })
            
            with open(self.signals_file, 'w') as f:
                json.dump(signals_data, f, indent=2)
                
        except Exception as e:
            print(f"File save error: {e}")
    
    def print_final_summary(self):
        """Print final summary when stopping"""

        print("\n" + "="*80)
        print("📊 FINAL TRADING SUMMARY")
        print("="*80)

        for instrument, stats in self.stats.items():
            print(f"\n{instrument}:")
            print(f"  Buys: {stats['total_buys']} (€{stats['buy_volume']:,.2f})")
            print(f"  Sells: {stats['total_sells']} (€{stats['sell_volume']:,.2f})")
            print(f"  Holds: {stats['total_holds']}")

            if stats['win_count'] + stats['loss_count'] > 0:
                win_rate = stats['win_count'] / (stats['win_count'] + stats['loss_count']) * 100
                print(f"  Win Rate: {win_rate:.1f}%")

        print("\n" + "="*80)

    def get_health_status(self) -> Dict[str, Any]:
        """
        Returns health status compatible with HealthMonitor.
        Reports on monitor's operational health and signal tracking.
        """
        try:
            with self._lock:
                total_signals = len(self.signal_history)
                active_count = len(self.active_signals)

                # Calculate aggregate stats
                total_buys = sum(s['total_buys'] for s in self.stats.values())
                total_sells = sum(s['total_sells'] for s in self.stats.values())
                total_trades = total_buys + total_sells

                win_count = sum(s['win_count'] for s in self.stats.values())
                loss_count = sum(s['loss_count'] for s in self.stats.values())
                total_pnl = sum(s['total_pnl'] for s in self.stats.values())

            # Determine health status
            status = 'OK'
            is_healthy = True
            issues = []

            # Check if monitor is running
            if not self._running:
                status = 'DEGRADED'
                is_healthy = False
                issues.append("Monitor not running")

            # Check if monitoring thread is alive
            if self._running and (not self._monitor_thread or not self._monitor_thread.is_alive()):
                status = 'DEGRADED'
                is_healthy = False
                issues.append("Monitor thread not alive")

            # Check if any signals are being tracked
            if self._running and total_signals == 0:
                status = 'WARNING'
                issues.append("No signals tracked yet")

            # Calculate win rate if trades exist
            win_rate = None
            if win_count + loss_count > 0:
                win_rate = win_count / (win_count + loss_count)

            return {
                'status': status,
                'module': 'RealtimeMonitor',
                'version': '1.0',
                'is_healthy': is_healthy,
                'monitoring_active': self._running,
                'thread_alive': bool(self._monitor_thread and self._monitor_thread.is_alive()),
                'total_signals': total_signals,
                'active_signals': active_count,
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'instruments_tracked': len(self.stats),
                'issues': issues if issues else None,
                'performance': {
                    'buys': total_buys,
                    'sells': total_sells,
                    'wins': win_count,
                    'losses': loss_count,
                    'signals_in_history': total_signals
                }
            }
        except Exception as e:
            return {
                'status': 'ERROR',
                'module': 'RealtimeMonitor',
                'version': '1.0',
                'is_healthy': False,
                'error': str(e),
                'last_error': str(e)
            }