# ─────────────────────────────────────────────────────────────
# File: modules/strategy/bias_auditor.py
# Enhanced with InfoBus integration & intelligent bias tracking
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class BiasAuditor(Module, AnalysisMixin, StateManagementMixin):
    """
    Enhanced bias auditor with InfoBus integration.
    Tracks, analyzes, and corrects psychological trading biases in real-time.
    Provides bias corrections to improve decision-making quality.
    """

    def __init__(
        self,
        history_len: int = 100,
        debug: bool = False,
        correction_threshold: int = 3,
        adaptation_rate: float = 0.1,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=history_len,
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        
        # Core parameters
        self.history_len = int(history_len)
        self.debug = bool(debug)
        self.correction_threshold = int(correction_threshold)
        self.adaptation_rate = float(adaptation_rate)
        
        # Bias tracking state
        self.bias_history = deque(maxlen=self.history_len)
        self.bias_corrections = defaultdict(int)
        self.bias_performance = defaultdict(list)
        self.bias_frequencies = defaultdict(int)
        
        # Enhanced bias categories with detailed tracking
        self.bias_categories = {
            'revenge': {
                'description': 'Trading to recover losses aggressively',
                'threshold': -50.0,  # Trigger after losing trades
                'weight_reduction': 0.3
            },
            'fear': {
                'description': 'Avoiding trades due to recent losses',
                'threshold': -100.0,  # Trigger after significant losses
                'weight_reduction': 0.2
            },
            'greed': {
                'description': 'Overconfident trading after wins',
                'threshold': 100.0,  # Trigger after winning streaks
                'weight_reduction': 0.25
            },
            'fomo': {
                'description': 'Fear of missing out on trends',
                'threshold': 50.0,  # Trigger during strong trends
                'weight_reduction': 0.35
            },
            'anchoring': {
                'description': 'Fixation on previous price levels',
                'threshold': 0.0,  # Always potential
                'weight_reduction': 0.15
            }
        }
        
        # Performance tracking
        self.session_stats = {
            'total_biases_detected': 0,
            'biases_corrected': 0,
            'correction_effectiveness': 0.0,
            'most_common_bias': 'none',
            'bias_impact_score': 0.0
        }
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "BiasAuditor",
            "logs/strategy/bias_auditor.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("BiasAuditor")
        
        self.log_operator_info(
            "🧠 Bias Auditor initialized",
            history_len=self.history_len,
            correction_threshold=self.correction_threshold,
            bias_categories=len(self.bias_categories)
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Clear bias tracking
        self.bias_history.clear()
        self.bias_corrections.clear()
        self.bias_performance.clear()
        self.bias_frequencies.clear()
        
        # Reset session stats
        self.session_stats = {
            'total_biases_detected': 0,
            'biases_corrected': 0,
            'correction_effectiveness': 0.0,
            'most_common_bias': 'none',
            'bias_impact_score': 0.0
        }
        
        self.log_operator_info("🔄 Bias Auditor reset - all bias tracking cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited bias analysis")
            return
        
        # Extract context and trading data
        context = extract_standard_context(info_bus)
        bias_signals = self._detect_biases_from_info_bus(info_bus, context)
        
        # Process detected biases
        for bias_type, strength in bias_signals.items():
            if strength > 0:
                self._record_bias_detection(bias_type, strength, context)
        
        # Update bias corrections
        self._update_bias_corrections(info_bus, context)
        
        # Update InfoBus with bias analysis
        self._update_info_bus_with_bias_data(info_bus)

    def _detect_biases_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]) -> Dict[str, float]:
        """Detect trading biases from InfoBus data"""
        
        bias_signals = {}
        
        try:
            # Get recent trading data
            recent_trades = info_bus.get('recent_trades', [])
            current_pnl = context.get('session_pnl', 0)
            positions = InfoBusExtractor.get_positions(info_bus)
            risk_data = info_bus.get('risk', {})
            
            # Detect revenge trading
            revenge_strength = self._detect_revenge_bias(recent_trades, current_pnl)
            if revenge_strength > 0:
                bias_signals['revenge'] = revenge_strength
            
            # Detect fear bias
            fear_strength = self._detect_fear_bias(recent_trades, risk_data)
            if fear_strength > 0:
                bias_signals['fear'] = fear_strength
            
            # Detect greed bias
            greed_strength = self._detect_greed_bias(recent_trades, positions)
            if greed_strength > 0:
                bias_signals['greed'] = greed_strength
            
            # Detect FOMO
            fomo_strength = self._detect_fomo_bias(info_bus, context)
            if fomo_strength > 0:
                bias_signals['fomo'] = fomo_strength
            
            # Detect anchoring bias
            anchoring_strength = self._detect_anchoring_bias(info_bus, recent_trades)
            if anchoring_strength > 0:
                bias_signals['anchoring'] = anchoring_strength
            
        except Exception as e:
            self.log_operator_warning(f"Bias detection failed: {e}")
            
        return bias_signals

    def _detect_revenge_bias(self, recent_trades: List[Dict], current_pnl: float) -> float:
        """Detect revenge trading patterns"""
        
        if not recent_trades or len(recent_trades) < 2:
            return 0.0
        
        # Look for increasing position sizes after losses
        recent_losses = [t for t in recent_trades[-5:] if t.get('pnl', 0) < 0]
        
        if len(recent_losses) >= 2:
            # Check if position sizes are increasing
            sizes = [abs(t.get('size', 0)) for t in recent_losses]
            if len(sizes) >= 2 and sizes[-1] > sizes[0] * 1.5:
                strength = min(1.0, (sizes[-1] / sizes[0] - 1.0) * 0.5)
                return strength
        
        # Check for rapid fire trading after losses
        if current_pnl < -100 and len(recent_trades) >= 3:
            last_hour_trades = len([t for t in recent_trades[-10:] 
                                   if self._is_recent_trade(t, minutes=60)])
            if last_hour_trades >= 5:
                return min(1.0, last_hour_trades / 10.0)
        
        return 0.0

    def _detect_fear_bias(self, recent_trades: List[Dict], risk_data: Dict) -> float:
        """Detect fear-based trading avoidance"""
        
        # Look for reduced trading after losses
        drawdown = risk_data.get('current_drawdown', 0)
        if drawdown > 0.05:  # 5% drawdown
            # Check if trading frequency has decreased
            recent_trade_count = len([t for t in recent_trades[-20:] 
                                    if self._is_recent_trade(t, hours=24)])
            expected_trades = 10  # Expected daily trades
            
            if recent_trade_count < expected_trades * 0.5:
                return min(1.0, drawdown * 2.0)
        
        return 0.0

    def _detect_greed_bias(self, recent_trades: List[Dict], positions: List[Dict]) -> float:
        """Detect greed-based overconfident trading"""
        
        if not recent_trades:
            return 0.0
        
        # Look for increasing position sizes after wins
        recent_wins = [t for t in recent_trades[-5:] if t.get('pnl', 0) > 0]
        
        if len(recent_wins) >= 3:
            # Check position size escalation
            total_exposure = sum(abs(p.get('size', 0)) for p in positions)
            
            if total_exposure > 2.0:  # Over-leveraged
                win_streak = len(recent_wins)
                return min(1.0, (win_streak / 5.0) * (total_exposure / 2.0))
        
        return 0.0

    def _detect_fomo_bias(self, info_bus: InfoBus, context: Dict) -> float:
        """Detect fear of missing out patterns"""
        
        # Look for chasing trending markets
        regime = context.get('regime', 'unknown')
        if regime == 'trending':
            recent_entries = len(info_bus.get('recent_trades', []))
            if recent_entries > 3:
                return min(1.0, recent_entries / 10.0)
        
        return 0.0

    def _detect_anchoring_bias(self, info_bus: InfoBus, recent_trades: List[Dict]) -> float:
        """Detect anchoring to previous price levels"""
        
        # This is a simplified version - in practice would need price level analysis
        if recent_trades:
            # Check for repeated trading at similar price levels
            price_levels = [t.get('entry_price', 0) for t in recent_trades[-5:] 
                          if t.get('entry_price', 0) > 0]
            
            if len(price_levels) >= 3:
                price_std = np.std(price_levels)
                price_mean = np.mean(price_levels)
                
                if price_std / price_mean < 0.01:  # Very tight price clustering
                    return 0.3
        
        return 0.0

    def _is_recent_trade(self, trade: Dict, minutes: int = 60, hours: int = 0) -> bool:
        """Check if trade is within specified time window"""
        
        try:
            trade_time = trade.get('timestamp', '')
            if not trade_time:
                return False
            
            # Simplified time check - in practice would parse timestamp
            return True  # Placeholder
            
        except Exception:
            return False

    def _record_bias_detection(self, bias_type: str, strength: float, context: Dict) -> None:
        """Record detected bias with context"""
        
        try:
            bias_record = {
                'type': bias_type,
                'strength': strength,
                'timestamp': datetime.datetime.now().isoformat(),
                'context': context.copy(),
                'pnl_impact': 0.0  # Will be updated later
            }
            
            self.bias_history.append(bias_record)
            self.bias_frequencies[bias_type] += 1
            self.session_stats['total_biases_detected'] += 1
            
            # Log significant bias detections
            if strength > 0.5:
                bias_desc = self.bias_categories.get(bias_type, {}).get('description', bias_type)
                self.log_operator_warning(
                    f"⚠️ Strong {bias_type.title()} bias detected",
                    strength=f"{strength:.1%}",
                    description=bias_desc,
                    session_total=self.session_stats['total_biases_detected']
                )
            
        except Exception as e:
            self.log_operator_error(f"Failed to record bias detection: {e}")

    def record_bias_outcome(self, bias_type: str, pnl: float) -> None:
        """Record the outcome of a biased decision for learning"""
        
        try:
            # Validate inputs
            if not isinstance(bias_type, str) or bias_type not in self.bias_categories:
                self.log_operator_warning(f"Invalid bias type: {bias_type}")
                return
            
            if np.isnan(pnl):
                self.log_operator_warning("NaN PnL in bias outcome, ignoring")
                return
            
            # Update performance tracking
            self.bias_performance[bias_type].append(pnl)
            
            # Update corrections if negative outcome
            if pnl < 0:
                self.bias_corrections[bias_type] += 1
                self.session_stats['biases_corrected'] += 1
                
                self.log_operator_info(
                    f"📚 Learning from {bias_type} bias",
                    pnl=f"€{pnl:.2f}",
                    total_corrections=self.bias_corrections[bias_type]
                )
            
            # Update bias records with actual outcome
            for record in reversed(self.bias_history):
                if record['type'] == bias_type and record.get('pnl_impact') == 0.0:
                    record['pnl_impact'] = pnl
                    break
            
            # Update session statistics
            self._update_session_stats()
            
        except Exception as e:
            self.log_operator_error(f"Failed to record bias outcome: {e}")

    def _update_bias_corrections(self, info_bus: InfoBus, context: Dict) -> None:
        """Update bias corrections based on recent performance"""
        
        # Calculate correction effectiveness
        if self.session_stats['biases_corrected'] > 0:
            recent_outcomes = []
            for bias_type, outcomes in self.bias_performance.items():
                if outcomes:
                    recent_outcomes.extend(outcomes[-5:])  # Last 5 outcomes
            
            if recent_outcomes:
                avg_outcome = np.mean(recent_outcomes)
                self.session_stats['correction_effectiveness'] = avg_outcome

    def _update_session_stats(self) -> None:
        """Update session-level bias statistics"""
        
        try:
            # Find most common bias
            if self.bias_frequencies:
                most_common = max(self.bias_frequencies.items(), key=lambda x: x[1])
                self.session_stats['most_common_bias'] = most_common[0]
            
            # Calculate bias impact score
            total_impact = 0.0
            for outcomes in self.bias_performance.values():
                if outcomes:
                    total_impact += sum(outcomes)
            
            self.session_stats['bias_impact_score'] = total_impact
            
        except Exception as e:
            self.log_operator_warning(f"Session stats update failed: {e}")

    def get_bias_adjustments(self) -> Dict[str, float]:
        """Get current bias adjustment weights"""
        
        adjustments = {}
        
        try:
            for bias_type, category in self.bias_categories.items():
                correction_count = self.bias_corrections.get(bias_type, 0)
                
                if correction_count >= self.correction_threshold:
                    # Apply weight reduction based on correction history
                    base_reduction = category['weight_reduction']
                    adaptive_reduction = min(0.8, correction_count * 0.1)
                    total_reduction = base_reduction + adaptive_reduction
                    
                    adjustments[bias_type] = 1.0 - total_reduction
                else:
                    adjustments[bias_type] = 1.0
            
            return adjustments
            
        except Exception as e:
            self.log_operator_error(f"Failed to calculate bias adjustments: {e}")
            return {bias_type: 1.0 for bias_type in self.bias_categories.keys()}

    def get_observation_components(self) -> np.ndarray:
        """Get bias frequencies and corrections for observation"""
        
        try:
            if not self.bias_history:
                # Return balanced defaults for cold start
                num_biases = len(self.bias_categories)
                defaults = np.full(num_biases * 2, 0.2, dtype=np.float32)  # frequencies + corrections
                self.log_operator_debug("Using default bias observations")
                return defaults
            
            # Calculate bias frequencies (normalized)
            total_biases = len(self.bias_history)
            frequencies = []
            for bias_type in self.bias_categories.keys():
                frequency = self.bias_frequencies.get(bias_type, 0) / max(1, total_biases)
                frequencies.append(frequency)
            
            # Calculate correction strengths (normalized)
            corrections = []
            max_corrections = max(self.bias_corrections.values()) if self.bias_corrections else 1
            for bias_type in self.bias_categories.keys():
                correction_strength = self.bias_corrections.get(bias_type, 0) / max(1, max_corrections)
                corrections.append(correction_strength)
            
            # Combine frequencies and corrections
            observation = np.array(frequencies + corrections, dtype=np.float32)
            
            # Validate for NaN/infinite values
            if np.any(~np.isfinite(observation)):
                self.log_operator_error(f"Invalid observation values: {observation}")
                observation = np.nan_to_num(observation, nan=0.2)
            
            self.log_operator_debug(f"Bias observation: {observation[:len(self.bias_categories)]}")
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            num_biases = len(self.bias_categories)
            return np.full(num_biases * 2, 0.2, dtype=np.float32)

    def _update_info_bus_with_bias_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with bias analysis results"""
        
        try:
            # Prepare bias data for InfoBus
            bias_data = {
                'session_stats': self.session_stats.copy(),
                'active_biases': self._get_active_biases(),
                'bias_adjustments': self.get_bias_adjustments(),
                'total_biases_tracked': len(self.bias_history),
                'bias_categories': list(self.bias_categories.keys()),
                'correction_effectiveness': self.session_stats.get('correction_effectiveness', 0.0)
            }
            
            # Add to InfoBus
            InfoBusUpdater.add_module_data(info_bus, 'bias_auditor', bias_data)
            
            # Add alerts for significant biases
            for bias_type, strength in self._get_active_biases().items():
                if strength > 0.7:  # High bias strength
                    InfoBusUpdater.add_alert(
                        info_bus,
                        f"High {bias_type} bias detected",
                        'bias_auditor',
                        'warning',
                        {'bias_type': bias_type, 'strength': strength}
                    )
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus update failed: {e}")

    def _get_active_biases(self) -> Dict[str, float]:
        """Get currently active biases with their strengths"""
        
        active_biases = {}
        
        # Look at recent bias detections (last 10 records)
        recent_biases = list(self.bias_history)[-10:] if self.bias_history else []
        
        for bias_type in self.bias_categories.keys():
            recent_strength = [r['strength'] for r in recent_biases 
                             if r['type'] == bias_type]
            
            if recent_strength:
                # Use maximum recent strength with decay
                max_strength = max(recent_strength)
                time_decay = 0.9  # Recent biases are stronger
                active_biases[bias_type] = max_strength * time_decay
            else:
                active_biases[bias_type] = 0.0
        
        return active_biases

    def log_operator_debug(self, message: str, **kwargs):
        """Log debug message with proper formatting"""
        if hasattr(self, 'debug') and self.debug and hasattr(self, 'logger'):
            # Convert kwargs to details string
            details = ""
            if kwargs:
                detail_parts = []
                for key, value in kwargs.items():
                    if isinstance(value, float):
                        if 'pct' in key or 'rate' in key or '%' in str(value):
                            detail_parts.append(f"{key}={value:.1%}")
                        else:
                            detail_parts.append(f"{key}={value:.3f}")
                    else:
                        detail_parts.append(f"{key}={value}")
                details = " | ".join(detail_parts)
            
            # Only log if we have a logger configured for debug
            if hasattr(self.logger, 'debug'):
                from modules.utils.audit_utils import format_operator_message
                formatted_message = format_operator_message(
                    emoji="🔧",
                    action=message,
                    details=details
                )
                self.logger.debug(formatted_message)

    def get_bias_report(self) -> str:
        """Generate operator-friendly bias analysis report"""
        
        active_biases = self._get_active_biases()
        adjustments = self.get_bias_adjustments()
        
        # Recent bias trend
        recent_trend = "Stable"
        if len(self.bias_history) >= 5:
            recent_count = len([b for b in self.bias_history[-5:]])
            if recent_count >= 4:
                recent_trend = "Increasing"
            elif recent_count <= 1:
                recent_trend = "Decreasing"
        
        # Most problematic bias
        if self.bias_corrections:
            worst_bias = max(self.bias_corrections.items(), key=lambda x: x[1])
            worst_bias_str = f"{worst_bias[0].title()} ({worst_bias[1]} corrections)"
        else:
            worst_bias_str = "None detected"
        
        return f"""
🧠 BIAS AUDITOR REPORT
═══════════════════════════════════════
📊 Session Overview:
• Total Biases Detected: {self.session_stats['total_biases_detected']}
• Biases Corrected: {self.session_stats['biases_corrected']}
• Most Common: {self.session_stats['most_common_bias'].title()}
• Recent Trend: {recent_trend}

⚠️ Active Biases:
{chr(10).join([f'  • {bias.title()}: {strength:.1%}' for bias, strength in active_biases.items() if strength > 0.1])}

🔧 Current Adjustments:
{chr(10).join([f'  • {bias.title()}: {adj:.1%} weight' for bias, adj in adjustments.items() if adj < 1.0])}

📈 Performance Impact:
• Correction Effectiveness: €{self.session_stats['correction_effectiveness']:.2f}
• Total Bias Impact: €{self.session_stats['bias_impact_score']:.2f}
• Most Problematic: {worst_bias_str}

🎯 Recommendations:
{self._generate_bias_recommendations()}
        """

    def _generate_bias_recommendations(self) -> str:
        """Generate actionable bias recommendations"""
        
        recommendations = []
        active_biases = self._get_active_biases()
        
        # Specific recommendations based on active biases
        if active_biases.get('revenge', 0) > 0.5:
            recommendations.append("• Take a 15-minute break after losses to reset mindset")
        
        if active_biases.get('greed', 0) > 0.5:
            recommendations.append("• Reduce position sizes after 3+ consecutive wins")
        
        if active_biases.get('fear', 0) > 0.5:
            recommendations.append("• Start with smaller positions to rebuild confidence")
        
        if active_biases.get('fomo', 0) > 0.5:
            recommendations.append("• Wait for pullbacks instead of chasing trends")
        
        if not recommendations:
            recommendations.append("• Continue current disciplined approach")
        
        return '\n'.join(recommendations)

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "history_len": self.history_len,
                "debug": self.debug,
                "correction_threshold": self.correction_threshold,
                "adaptation_rate": self.adaptation_rate
            },
            "bias_data": {
                "bias_history": list(self.bias_history),
                "bias_corrections": dict(self.bias_corrections),
                "bias_performance": {k: list(v) for k, v in self.bias_performance.items()},
                "bias_frequencies": dict(self.bias_frequencies),
                "session_stats": self.session_stats.copy()
            },
            "categories": self.bias_categories.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.history_len = int(config.get("history_len", self.history_len))
        self.debug = bool(config.get("debug", self.debug))
        self.correction_threshold = int(config.get("correction_threshold", self.correction_threshold))
        self.adaptation_rate = float(config.get("adaptation_rate", self.adaptation_rate))
        
        # Load bias data
        bias_data = state.get("bias_data", {})
        self.bias_history = deque(bias_data.get("bias_history", []), maxlen=self.history_len)
        self.bias_corrections = defaultdict(int, bias_data.get("bias_corrections", {}))
        
        # Restore performance data
        performance_data = bias_data.get("bias_performance", {})
        self.bias_performance = defaultdict(list)
        for k, v in performance_data.items():
            self.bias_performance[k] = list(v)
        
        self.bias_frequencies = defaultdict(int, bias_data.get("bias_frequencies", {}))
        self.session_stats = bias_data.get("session_stats", self.session_stats)
        
        # Load categories if provided
        self.bias_categories.update(state.get("categories", {}))