
# ─────────────────────────────────────────────────────────────
# File: modules/auditing/trade_thesis_tracker.py
# ENHANCED: Thesis tracker using SmartInfoBus architecture
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from collections import defaultdict, deque
from typing import Any, Dict, Optional, List

# ✅ FIXED: Proper imports for SmartInfoBus system
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import format_operator_message


@module(
    provides=['thesis_analysis', 'thesis_performance', 'thesis_alerts'],
    requires=['trading_signal', 'market_data', 'trades'],
    category='auditing',
    is_voting_member=False,
    hot_reload=True,
    explainable=True,
    timeout_ms=100,
    priority=3,
    version="2.0.0"
)
class TradeThesisTracker(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🧠 PRODUCTION-GRADE Trade Thesis Tracker
    
    Advanced thesis tracking with:
    - Real-time thesis evolution monitoring
    - Performance tracking per thesis
    - Thesis change detection and alerting
    - NASA/MILITARY GRADE reliability patterns
    """
    
    def _initialize(self):
        """Initialize thesis-specific state with SmartInfoBus integration"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        
        # Thesis tracking state
        self.current_thesis = "unknown"
        self.thesis_changes = 0
        self.thesis_performance = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "confidence": 0.0})
        self.thesis_history = deque(maxlen=500)
        
        # Thesis evolution tracking
        self.thesis_patterns = defaultdict(int)
        self.transition_matrix = defaultdict(lambda: defaultdict(int))
        
        # Performance metrics
        self.session_start = datetime.datetime.now()
        self.best_thesis = None
        self.worst_thesis = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'frequent_changes': 10,  # Alert if >10 changes per hour
            'poor_performance': -100,  # Alert if thesis loses >$100
            'low_confidence': 0.3  # Alert if thesis confidence <30%
        }
        
        self.logger.info(f"🧠 {self.__class__.__name__} initialized with thesis tracking")
    
    def reset(self) -> None:
        """Reset with automatic cleanup"""
        super().reset()
        
        # Reset thesis state
        self.current_thesis = "unknown"
        self.thesis_changes = 0
        self.thesis_performance.clear()
        self.thesis_history.clear()
        self.thesis_patterns.clear()
        self.transition_matrix.clear()
        
        self.session_start = datetime.datetime.now()
        self.best_thesis = None
        self.worst_thesis = None
        
        self.logger.info("🔄 Trade thesis tracker reset complete")
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        🧠 MAIN THESIS TRACKING PROCESS
        
        Tracks thesis evolution and performance:
        1. Extract current thesis from trading signals
        2. Detect and handle thesis changes
        3. Update performance metrics per thesis
        4. Generate alerts for significant changes
        """
        try:
            # Extract thesis context
            context = self._extract_thesis_context(inputs)
            
            # Detect thesis changes
            new_thesis = self._extract_current_thesis(inputs, context)
            thesis_changed = self._handle_potential_thesis_change(new_thesis)
            
            # Process trades with thesis context
            processed_trades = self._process_trades_with_thesis(inputs, context)
            
            # Update thesis performance
            self._update_thesis_performance(processed_trades)
            
            # Generate thesis analysis
            thesis_analysis = self._generate_thesis_analysis()
            
            # Check for alerts
            alerts = self._check_for_thesis_alerts(thesis_analysis, context)
            
            # Generate thesis for explainability
            thesis_explanation = self._generate_thesis_explanation(thesis_analysis, thesis_changed)
            
            # Store results in SmartInfoBus
            smart_bus = InfoBusManager.get_instance()
            smart_bus.set(
                'thesis_analysis',
                thesis_analysis,
                module=self.__class__.__name__,
                thesis=thesis_explanation,
                confidence=thesis_analysis.get('current_confidence', 0.7)
            )
            
            return {
                'thesis_analysis': thesis_analysis,
                'thesis_performance': dict(self.thesis_performance),
                'thesis_alerts': alerts,
                '_thesis': thesis_explanation
            }
            
        except Exception as e:
            self.logger.error(f"❌ Thesis tracking failed: {e}")
            return {
                'thesis_analysis': {'error': str(e)},
                'thesis_performance': dict(self.thesis_performance),
                'thesis_alerts': [{'type': 'tracking_failure', 'message': str(e)}],
                '_thesis': f"Thesis tracking encountered error: {str(e)}"
            }
    
    def _extract_thesis_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context for thesis analysis"""
        trading_signal = inputs.get('trading_signal', {})
        
        return {
            'trading_signal': trading_signal,
            'market_data': inputs.get('market_data', {}),
            'trades': inputs.get('trades', []),
            'timestamp': inputs.get('timestamp', datetime.datetime.now()),
            'step_idx': inputs.get('step_idx', 0),
            'signal_confidence': trading_signal.get('confidence', 0.5),
            'signal_action': trading_signal.get('action', 'unknown'),
            'signal_reason': trading_signal.get('reason', '')
        }
    
    def _extract_current_thesis(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract current trading thesis from inputs"""
        # Extract from trading signal
        signal = context['trading_signal']
        
        if not signal:
            return "no_signal"
        
        # Build thesis from components
        action = context['signal_action']
        confidence_level = "high" if context['signal_confidence'] > 0.7 else "medium" if context['signal_confidence'] > 0.4 else "low"
        
        # Try to extract market regime or sentiment
        market_regime = "unknown"
        if context['market_data']:
            # Simple market regime detection based on price movement
            prices = context['market_data'].get('close', [])
            if len(prices) >= 5:
                recent_trend = "trending_up" if prices[-1] > prices[-5] else "trending_down"
                market_regime = recent_trend
        
        # Construct thesis
        thesis = f"{action}_{market_regime}_{confidence_level}"
        
        # Include reason/thesis if available
        if context['signal_reason']:
            # Extract key words from reason to enhance thesis
            reason_words = context['signal_reason'].lower().split()
            key_indicators = ['bullish', 'bearish', 'breakout', 'reversal', 'momentum', 'support', 'resistance']
            found_indicators = [word for word in reason_words if word in key_indicators]
            if found_indicators:
                thesis += f"_{found_indicators[0]}"
        
        return thesis
    
    def _handle_potential_thesis_change(self, new_thesis: str) -> bool:
        """Handle potential thesis change with logging"""
        if new_thesis != self.current_thesis:
            self._handle_thesis_change(self.current_thesis, new_thesis)
            return True
        return False
    
    def _handle_thesis_change(self, old_thesis: str, new_thesis: str):
        """Handle thesis change with automatic logging and tracking"""
        self.thesis_changes += 1
        
        # Record transition
        self.transition_matrix[old_thesis][new_thesis] += 1
        
        # Record in history
        self.thesis_history.append({
            'timestamp': datetime.datetime.now(),
            'old_thesis': old_thesis,
            'new_thesis': new_thesis,
            'change_number': self.thesis_changes
        })
        
        # Update current thesis
        self.current_thesis = new_thesis
        
        # Log the change
        self.logger.info(
            format_operator_message(
                "🧠",
                f"Thesis change: {old_thesis} → {new_thesis}",
                change_count=self.thesis_changes,
                context="thesis_evolution"
            )
        )
    
    def _process_trades_with_thesis(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process trades with thesis context"""
        processed_trades = []
        trades = context.get('trades', [])
        
        for trade in trades:
            processed_trade = self._process_single_trade_with_thesis(trade, context)
            if processed_trade:
                processed_trades.append(processed_trade)
        
        return processed_trades
    
    def _process_single_trade_with_thesis(self, trade: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single trade with thesis context"""
        try:
            processed_trade = {
                'trade_id': f"{trade.get('symbol', 'UNK')}_{trade.get('timestamp', '')}",
                'symbol': trade.get('symbol'),
                'action': trade.get('action', 'unknown'),
                'pnl': trade.get('pnl', 0),
                'thesis': self.current_thesis,
                'confidence': context.get('signal_confidence', 0.5),
                'timestamp': context['timestamp'],
                'processed_at': datetime.datetime.now().isoformat()
            }
            
            return processed_trade
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process trade with thesis: {e}")
            return None
    
    def _update_thesis_performance(self, processed_trades: List[Dict[str, Any]]):
        """Update thesis performance metrics"""
        for trade in processed_trades:
            thesis = trade['thesis']
            pnl = trade['pnl']
            confidence = trade['confidence']
            
            # Update performance
            self.thesis_performance[thesis]['trades'] += 1
            self.thesis_performance[thesis]['pnl'] += pnl
            
            # Update confidence (exponential moving average)
            current_conf = self.thesis_performance[thesis]['confidence']
            if current_conf == 0:
                self.thesis_performance[thesis]['confidence'] = confidence
            else:
                alpha = 0.2
                self.thesis_performance[thesis]['confidence'] = (
                    alpha * confidence + (1 - alpha) * current_conf
                )
        
        # Update best/worst thesis
        self._update_best_worst_thesis()
    
    def _update_best_worst_thesis(self):
        """Update best and worst performing thesis"""
        if not self.thesis_performance:
            return
        
        performances = [(thesis, stats['pnl']) for thesis, stats in self.thesis_performance.items()]
        
        if performances:
            self.best_thesis = max(performances, key=lambda x: x[1])[0]
            self.worst_thesis = min(performances, key=lambda x: x[1])[0]
    
    def _generate_thesis_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive thesis analysis"""
        session_duration = (datetime.datetime.now() - self.session_start).total_seconds() / 3600
        
        # Calculate thesis diversity
        active_thesis_count = len(self.thesis_performance)
        
        # Calculate change frequency
        change_frequency = self.thesis_changes / max(session_duration, 0.1)  # changes per hour
        
        # Get current thesis performance
        current_performance = self.thesis_performance.get(self.current_thesis, {"pnl": 0, "trades": 0, "confidence": 0})
        
        # Calculate overall performance
        total_pnl = sum(stats['pnl'] for stats in self.thesis_performance.values())
        total_trades = sum(stats['trades'] for stats in self.thesis_performance.values())
        
        return {
            'current_thesis': self.current_thesis,
            'thesis_changes': self.thesis_changes,
            'change_frequency': change_frequency,
            'active_thesis_count': active_thesis_count,
            'current_confidence': current_performance['confidence'],
            'current_performance': current_performance,
            'best_thesis': self.best_thesis,
            'worst_thesis': self.worst_thesis,
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'session_duration_hours': session_duration,
            'thesis_diversity': active_thesis_count / max(self.thesis_changes, 1)
        }
    
    def _check_for_thesis_alerts(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for thesis-related alerts"""
        alerts = []
        
        # Frequent changes alert
        if analysis['change_frequency'] > self.alert_thresholds['frequent_changes']:
            alerts.append({
                'type': 'frequent_thesis_changes',
                'severity': 'medium',
                'message': f"High thesis change frequency: {analysis['change_frequency']:.1f}/hour",
                'frequency': analysis['change_frequency'],
                'threshold': self.alert_thresholds['frequent_changes'],
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        # Poor performance alert
        current_pnl = analysis['current_performance']['pnl']
        if current_pnl < self.alert_thresholds['poor_performance']:
            alerts.append({
                'type': 'poor_thesis_performance',
                'severity': 'high',
                'message': f"Current thesis showing poor performance: ${current_pnl:.2f}",
                'thesis': analysis['current_thesis'],
                'pnl': current_pnl,
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        # Low confidence alert
        if analysis['current_confidence'] < self.alert_thresholds['low_confidence']:
            alerts.append({
                'type': 'low_thesis_confidence',
                'severity': 'medium',
                'message': f"Low confidence in current thesis: {analysis['current_confidence']:.1%}",
                'thesis': analysis['current_thesis'],
                'confidence': analysis['current_confidence'],
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        return alerts
    
    def _generate_thesis_explanation(self, analysis: Dict[str, Any], thesis_changed: bool) -> str:
        """Generate thesis tracking explanation"""
        current_thesis = analysis['current_thesis']
        changes = analysis['thesis_changes']
        performance = analysis['current_performance']['pnl']
        
        if thesis_changed:
            explanation = f"Thesis evolved to '{current_thesis}' (change #{changes}). "
        else:
            explanation = f"Continuing with thesis '{current_thesis}'. "
        
        if performance > 0:
            explanation += f"Current thesis showing positive performance (+${performance:.2f}). "
        elif performance < 0:
            explanation += f"Current thesis showing negative performance (${performance:.2f}). "
        else:
            explanation += "Current thesis at break-even. "
        
        # Add diversity insight
        if analysis['active_thesis_count'] > 5:
            explanation += f"High thesis diversity ({analysis['active_thesis_count']} active) indicates adaptive strategy."
        elif analysis['active_thesis_count'] < 3:
            explanation += f"Low thesis diversity ({analysis['active_thesis_count']} active) indicates consistent strategy."
        
        return explanation
    
    def get_comprehensive_thesis_analysis(self) -> Dict[str, Any]:
        """Get comprehensive thesis analysis using trading state"""
        base_analysis = self._generate_thesis_analysis()
        
        # Add transition analysis
        transitions = {}
        for from_thesis, to_dict in self.transition_matrix.items():
            transitions[from_thesis] = dict(to_dict)
        
        # Add pattern analysis
        thesis_patterns = dict(self.thesis_patterns)
        
        # Add recent history
        recent_history = list(self.thesis_history)[-20:]  # Last 20 changes
        
        return {
            **base_analysis,
            'transitions': transitions,
            'patterns': thesis_patterns,
            'recent_history': recent_history,
            'trading_summary': self._get_trading_summary() if hasattr(self, '_get_trading_summary') else {}
        }
    
    def generate_thesis_report(self) -> str:
        """Generate comprehensive thesis report"""
        analysis = self.get_comprehensive_thesis_analysis()
        
        return f"""
🧠 TRADE THESIS ANALYSIS REPORT
═══════════════════════════════════════
📅 Session Duration: {analysis['session_duration_hours']:.1f} hours
🎯 Current Thesis: {analysis['current_thesis']}
🔄 Thesis Changes: {analysis['thesis_changes']}

📊 PERFORMANCE METRICS
• Current P&L: ${analysis['current_performance']['pnl']:.2f}
• Current Confidence: {analysis['current_confidence']:.1%}
• Total P&L: ${analysis['total_pnl']:.2f}
• Total Trades: {analysis['total_trades']}

🏆 BEST/WORST THESIS
• Best: {analysis['best_thesis']} (${self.thesis_performance.get(analysis['best_thesis'], {}).get('pnl', 0):.2f})
• Worst: {analysis['worst_thesis']} (${self.thesis_performance.get(analysis['worst_thesis'], {}).get('pnl', 0):.2f})

📈 ADAPTATION METRICS
• Change Frequency: {analysis['change_frequency']:.1f}/hour
• Thesis Diversity: {analysis['thesis_diversity']:.2f}
• Active Thesis Count: {analysis['active_thesis_count']}
        """


# ────────────────────────────────────────────────────────────────────────────────
# 🧠 MODULE REGISTRATION
# ────────────────────────────────────────────────────────────────────────────────
# Module is automatically discovered and registered via @module decorator
# No manual registration needed - SmartInfoBus handles everything!