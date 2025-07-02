# ─────────────────────────────────────────────────────────────
# File: modules/risk/active_trade_monitor.py
# Enhanced with InfoBus integration & operator-centric design
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Union
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import RiskMixin, TradingMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import AuditTrailManager, format_operator_message


class ActiveTradeMonitor(Module, RiskMixin, TradingMixin, StateManagementMixin):
    """
    Enhanced active trade duration monitor with InfoBus integration.
    Provides real-time position duration tracking with intelligent alerts.
    """

    def __init__(
        self,
        max_duration: int = 200,
        warning_duration: int = 50,
        critical_duration: int = 150,
        enabled: bool = True,
        severity_weights: Optional[Dict[str, float]] = None,
        debug: bool = True,
        **kwargs
    ):
        # Initialize with enhanced config
        config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 100),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_risk_state()
        self._initialize_trading_state()
        
        # Enhanced configuration
        self.max_duration = max_duration
        self.warning_duration = warning_duration
        self.critical_duration = critical_duration
        self.enabled = enabled
        
        # Severity system with adaptive weights
        self.severity_weights = severity_weights or {
            "info": 0.0,
            "warning": 0.5,
            "critical": 1.0
        }
        
        # Enhanced state tracking
        self.position_durations: Dict[str, int] = {}
        self.position_first_seen: Dict[str, str] = {}  # Track when position was first detected
        self.alerts: Dict[str, str] = {}
        self.risk_score = 0.0
        self.step_count = 0
        self._last_alert_step = 0
        
        # Duration history for analytics
        self._duration_history = deque(maxlen=self.config.max_history)
        self._risk_score_history = deque(maxlen=50)
        self._alert_frequency = defaultdict(int)
        
        # Performance tracking
        self._positions_closed_normally = 0
        self._positions_closed_timeout = 0
        self._max_concurrent_positions = 0
        
        # Audit system
        self.audit_manager = AuditTrailManager("ActiveTradeMonitor")
        
        self.log_operator_info(
            "🔍 Active Trade Monitor initialized",
            max_duration=f"{max_duration} steps",
            warning_threshold=f"{warning_duration} steps",
            critical_threshold=f"{critical_duration} steps",
            enabled=enabled
        )

    def reset(self) -> None:
        """Enhanced reset with full state cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_trading_state()
        
        # Reset monitor state
        self.position_durations.clear()
        self.position_first_seen.clear()
        self.alerts.clear()
        self.risk_score = 0.0
        self.step_count = 0
        self._last_alert_step = 0
        
        # Reset history
        self._duration_history.clear()
        self._risk_score_history.clear()
        self._alert_frequency.clear()
        
        # Reset performance tracking
        self._positions_closed_normally = 0
        self._positions_closed_timeout = 0
        self._max_concurrent_positions = 0
        
        self.log_operator_info("🔄 Active Trade Monitor reset - all tracking cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - monitor inactive")
            return
            
        if not self.enabled:
            self.risk_score = 0.0
            return
        
        self.step_count += 1
        
        # Extract positions from InfoBus
        positions = InfoBusExtractor.get_positions(info_bus)
        current_step = info_bus.get('step_idx', self.step_count)
        
        # Process position monitoring
        self._monitor_positions(positions, current_step, info_bus)
        
        # Update risk assessment
        self._calculate_risk_score(info_bus)
        
        # Update InfoBus with monitoring results
        self._update_info_bus(info_bus)
        
        # Record performance metrics
        self._update_monitoring_metrics()

    def _monitor_positions(self, positions: List[Dict[str, Any]], 
                          current_step: int, info_bus: InfoBus) -> None:
        """Monitor active positions with enhanced duration tracking"""
        
        # Clear previous alerts
        self.alerts.clear()
        current_symbols = set()
        
        if not positions:
            self._handle_no_positions()
            return
        
        # Process each position
        for position in positions:
            symbol = position.get('symbol', position.get('instrument', 'UNKNOWN'))
            current_symbols.add(symbol)
            
            # Calculate duration
            duration = self._calculate_position_duration(position, symbol, current_step)
            self.position_durations[symbol] = duration
            
            # Track first seen
            if symbol not in self.position_first_seen:
                self.position_first_seen[symbol] = info_bus.get('timestamp', 
                    datetime.datetime.now().isoformat())
            
            # Assess severity and generate alerts
            severity = self._assess_position_severity(symbol, duration, position, info_bus)
            
            if severity != "info":
                self.alerts[symbol] = severity
                self._handle_position_alert(symbol, duration, severity, position)
        
        # Handle closed positions
        self._handle_closed_positions(current_symbols)
        
        # Update concurrent position tracking
        self._max_concurrent_positions = max(
            self._max_concurrent_positions, 
            len(current_symbols)
        )

    def _calculate_position_duration(self, position: Dict[str, Any], 
                                   symbol: str, current_step: int) -> int:
        """Enhanced position duration calculation with multiple fallback methods"""
        
        try:
            # Method 1: Direct duration field
            for duration_field in ["duration", "time_open", "bars_held", "steps_open"]:
                if duration_field in position and position[duration_field] is not None:
                    return max(0, int(position[duration_field]))
            
            # Method 2: Calculate from entry step
            for entry_field in ["entry_step", "open_step", "start_step"]:
                if entry_field in position and position[entry_field] is not None:
                    entry_step = int(position[entry_field])
                    return max(0, current_step - entry_step)
            
            # Method 3: Calculate from timestamps
            if "entry_time" in position:
                try:
                    entry_time = position["entry_time"]
                    if isinstance(entry_time, str):
                        entry_time = datetime.datetime.fromisoformat(
                            entry_time.replace('Z', '+00:00')
                        )
                    
                    now = datetime.datetime.now(datetime.timezone.utc)
                    duration_minutes = (now - entry_time).total_seconds() / 60
                    return max(0, int(duration_minutes))  # Convert to steps (assuming 1 step = 1 minute)
                except Exception as e:
                    self.log_operator_warning(f"Timestamp parsing failed for {symbol}: {e}")
            
            # Method 4: Increment from previous tracking
            previous_duration = self.position_durations.get(symbol, 0)
            return previous_duration + 1
            
        except Exception as e:
            self.log_operator_error(f"Duration calculation failed for {symbol}: {e}")
            return self.position_durations.get(symbol, 1)

    def _assess_position_severity(self, symbol: str, duration: int, 
                                position: Dict[str, Any], info_bus: InfoBus) -> str:
        """Enhanced severity assessment with context awareness"""
        
        # Base severity assessment
        if duration >= self.max_duration:
            base_severity = "critical"
        elif duration >= self.critical_duration:
            base_severity = "warning"
        elif duration >= self.warning_duration:
            base_severity = "warning"
        else:
            base_severity = "info"
        
        # Context-aware adjustments
        context = extract_standard_context(info_bus)
        
        # Adjust for market conditions
        market_regime = context.get('regime', 'unknown')
        volatility_level = context.get('volatility_level', 'medium')
        
        # More lenient in volatile markets
        if volatility_level in ['high', 'extreme'] and base_severity == "warning":
            # Allow longer durations in high volatility
            tolerance_multiplier = 1.3 if volatility_level == 'high' else 1.5
            adjusted_duration = duration / tolerance_multiplier
            
            if adjusted_duration < self.warning_duration:
                base_severity = "info"
            elif adjusted_duration < self.critical_duration:
                base_severity = "warning"
        
        # Consider position PnL
        position_pnl = position.get('unrealised_pnl', position.get('pnl', 0))
        if position_pnl > 0 and base_severity == "warning":
            # Profitable positions get slight tolerance
            base_severity = "info" if duration < self.critical_duration else "warning"
        
        return base_severity

    def _handle_position_alert(self, symbol: str, duration: int, 
                             severity: str, position: Dict[str, Any]) -> None:
        """Handle position alerts with structured logging and audit"""
        
        position_pnl = position.get('unrealised_pnl', position.get('pnl', 0))
        position_size = position.get('size', position.get('volume', 0))
        
        # Create structured alert
        alert_data = {
            'symbol': symbol,
            'duration': duration,
            'severity': severity,
            'pnl': position_pnl,
            'size': position_size,
            'thresholds': {
                'warning': self.warning_duration,
                'critical': self.critical_duration,
                'max': self.max_duration
            },
            'first_seen': self.position_first_seen.get(symbol),
            'step_count': self.step_count
        }
        
        # Log appropriate operator message
        if severity == "critical":
            self.log_operator_error(
                f"🚨 CRITICAL: {symbol} held {duration} steps",
                pnl=f"€{position_pnl:.2f}",
                size=f"{position_size:.4f}",
                limit=f"{self.max_duration} steps"
            )
        elif severity == "warning":
            self.log_operator_warning(
                f"⚠️ Long position: {symbol} held {duration} steps",
                pnl=f"€{position_pnl:.2f}",
                threshold=f"{self.warning_duration} steps"
            )
        
        # Record in audit trail
        self.audit_manager.record_event(
            event_type="position_duration_alert",
            module="ActiveTradeMonitor",
            details=alert_data,
            severity=severity
        )
        
        # Update alert frequency tracking
        self._alert_frequency[severity] += 1
        self._last_alert_step = self.step_count

    def _handle_no_positions(self) -> None:
        """Handle state when no positions are active"""
        
        # Mark any previously tracked positions as closed normally
        for symbol in list(self.position_durations.keys()):
            duration = self.position_durations[symbol]
            
            if duration < self.max_duration:
                self._positions_closed_normally += 1
            else:
                self._positions_closed_timeout += 1
            
            # Log position closure
            self.log_operator_info(
                f"📍 Position closed: {symbol}",
                duration=f"{duration} steps",
                status="normal" if duration < self.max_duration else "timeout"
            )
        
        # Clear tracking
        self.position_durations.clear()
        self.position_first_seen.clear()
        self.risk_score = 0.0

    def _handle_closed_positions(self, current_symbols: set) -> None:
        """Handle positions that are no longer active"""
        
        closed_symbols = set(self.position_durations.keys()) - current_symbols
        
        for symbol in closed_symbols:
            duration = self.position_durations[symbol]
            
            # Classify closure
            if duration < self.max_duration:
                self._positions_closed_normally += 1
                closure_type = "normal"
            else:
                self._positions_closed_timeout += 1
                closure_type = "timeout"
            
            # Log closure
            self.log_operator_info(
                f"📍 Position closed: {symbol}",
                duration=f"{duration} steps",
                status=closure_type
            )
            
            # Remove from tracking
            self.position_durations.pop(symbol, None)
            self.position_first_seen.pop(symbol, None)

    def _calculate_risk_score(self, info_bus: InfoBus) -> None:
        """Calculate comprehensive risk score"""
        
        if not self.position_durations:
            self.risk_score = 0.0
            return
        
        total_severity = 0.0
        position_count = len(self.position_durations)
        
        # Calculate severity-weighted score
        for symbol, severity in self.alerts.items():
            total_severity += self.severity_weights.get(severity, 0.0)
        
        # Base risk score
        base_score = total_severity / max(position_count, 1)
        
        # Context adjustments
        context = extract_standard_context(info_bus)
        
        # Increase risk in certain market conditions
        market_multiplier = 1.0
        if context.get('volatility_level') == 'extreme':
            market_multiplier = 1.2
        elif context.get('regime') == 'volatile':
            market_multiplier = 1.1
        
        # Adjust for position concentration
        if position_count > 5:  # High concentration
            concentration_multiplier = 1.0 + (position_count - 5) * 0.1
        else:
            concentration_multiplier = 1.0
        
        # Final risk score
        self.risk_score = min(1.0, base_score * market_multiplier * concentration_multiplier)
        
        # Track risk score history
        self._risk_score_history.append(self.risk_score)

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with monitoring results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'active_trade_monitor', {
            'risk_score': self.risk_score,
            'position_count': len(self.position_durations),
            'alerts': self.alerts.copy(),
            'durations': self.position_durations.copy(),
            'max_concurrent': self._max_concurrent_positions,
            'closure_stats': {
                'normal': self._positions_closed_normally,
                'timeout': self._positions_closed_timeout
            }
        })
        
        # Add risk data to InfoBus
        InfoBusUpdater.update_risk_snapshot(info_bus, {
            'duration_risk_score': self.risk_score,
            'long_positions_count': sum(1 for alerts in self.alerts.values() 
                                      if alerts in ['warning', 'critical']),
            'position_durations': self.position_durations.copy()
        })
        
        # Add alerts for critical situations
        if self.risk_score > 0.7:
            InfoBusUpdater.add_alert(
                info_bus,
                f"High duration risk: {self.risk_score:.1%} of positions held too long",
                severity="warning",
                module="ActiveTradeMonitor"
            )

    def _update_monitoring_metrics(self) -> None:
        """Update performance metrics"""
        
        self._update_performance_metric('risk_score', self.risk_score)
        self._update_performance_metric('position_count', len(self.position_durations))
        self._update_performance_metric('alert_count', len(self.alerts))
        
        # Update duration statistics
        if self.position_durations:
            avg_duration = np.mean(list(self.position_durations.values()))
            max_duration = max(self.position_durations.values())
            
            self._update_performance_metric('avg_duration', avg_duration)
            self._update_performance_metric('max_current_duration', max_duration)
            
            # Track in history
            self._duration_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'avg_duration': avg_duration,
                'max_duration': max_duration,
                'position_count': len(self.position_durations),
                'alert_count': len(self.alerts)
            })

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        
        try:
            # Basic risk metrics
            risk_score = float(self.risk_score)
            position_count_norm = min(len(self.position_durations) / 10.0, 1.0)
            alert_ratio = len(self.alerts) / max(len(self.position_durations), 1)
            
            # Duration statistics
            if self.position_durations:
                durations = list(self.position_durations.values())
                avg_duration_norm = np.mean(durations) / max(self.max_duration, 1)
                max_duration_norm = max(durations) / max(self.max_duration, 1)
            else:
                avg_duration_norm = 0.0
                max_duration_norm = 0.0
            
            # Alert frequency (recent activity)
            recent_alert_activity = min(
                sum(self._alert_frequency.values()) / max(self.step_count, 1),
                1.0
            )
            
            # Performance ratio
            total_closures = self._positions_closed_normally + self._positions_closed_timeout
            normal_closure_ratio = (
                self._positions_closed_normally / max(total_closures, 1)
            )
            
            return np.array([
                risk_score,                    # Current risk score
                position_count_norm,           # Normalized position count
                alert_ratio,                   # Alert ratio
                avg_duration_norm,             # Normalized average duration
                max_duration_norm,             # Normalized max duration
                recent_alert_activity,         # Recent alert frequency
                normal_closure_ratio           # Normal closure ratio
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(7, dtype=np.float32)

    def get_monitoring_report(self) -> str:
        """Generate operator-friendly monitoring report"""
        
        # Status indicators
        if self.risk_score > 0.7:
            risk_status = "🚨 Critical"
        elif self.risk_score > 0.3:
            risk_status = "⚠️ Elevated" 
        else:
            risk_status = "✅ Normal"
        
        # Performance status
        total_closures = self._positions_closed_normally + self._positions_closed_timeout
        if total_closures > 0:
            normal_rate = self._positions_closed_normally / total_closures
            if normal_rate > 0.8:
                performance_status = "🎯 Excellent"
            elif normal_rate > 0.6:
                performance_status = "✅ Good"
            elif normal_rate > 0.4:
                performance_status = "⚡ Fair"
            else:
                performance_status = "⚠️ Poor"
        else:
            performance_status = "📊 No Data"
        
        # Current positions summary
        position_lines = []
        for symbol, duration in self.position_durations.items():
            status = self.alerts.get(symbol, 'normal')
            status_emoji = "🚨" if status == "critical" else "⚠️" if status == "warning" else "✅"
            position_lines.append(f"  {status_emoji} {symbol}: {duration} steps ({status})")
        
        return f"""
🔍 ACTIVE TRADE MONITOR
═══════════════════════════════════════
🎯 Risk Status: {risk_status} ({self.risk_score:.1%})
📊 Active Positions: {len(self.position_durations)}
⚠️ Current Alerts: {len(self.alerts)}
🔄 Monitor Enabled: {'✅ Yes' if self.enabled else '❌ No'}

⏱️ DURATION THRESHOLDS
• Warning: {self.warning_duration} steps
• Critical: {self.critical_duration} steps
• Maximum: {self.max_duration} steps

📈 PERFORMANCE METRICS
• Performance: {performance_status} ({(normal_rate if total_closures > 0 else 0):.1%} normal closures)
• Normal Closures: {self._positions_closed_normally}
• Timeout Closures: {self._positions_closed_timeout}
• Max Concurrent: {self._max_concurrent_positions}
• Total Alerts: {sum(self._alert_frequency.values())}

📍 CURRENT POSITIONS
{chr(10).join(position_lines) if position_lines else "  📭 No active positions"}

🚨 ALERT BREAKDOWN
• Critical: {self._alert_frequency.get('critical', 0)}
• Warning: {self._alert_frequency.get('warning', 0)}
• Info: {self._alert_frequency.get('info', 0)}

💡 MONITOR STATUS
• Step Count: {self.step_count:,}
• Last Alert: Step {self._last_alert_step}
• Tracking Since: {min(self.position_first_seen.values()) if self.position_first_seen else 'N/A'}
        """

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health reporting"""
        base_details = super()._get_health_details()
        
        monitor_details = {
            'monitor_status': {
                'enabled': self.enabled,
                'step_count': self.step_count,
                'risk_score': self.risk_score,
                'positions_tracked': len(self.position_durations)
            },
            'alert_system': {
                'total_alerts': sum(self._alert_frequency.values()),
                'last_alert_step': self._last_alert_step,
                'alert_frequency': dict(self._alert_frequency)
            },
            'closure_performance': {
                'normal_closures': self._positions_closed_normally,
                'timeout_closures': self._positions_closed_timeout,
                'closure_rate': (
                    self._positions_closed_normally / 
                    max(self._positions_closed_normally + self._positions_closed_timeout, 1)
                )
            },
            'risk_trends': {
                'recent_risk_avg': (
                    np.mean(list(self._risk_score_history)[-10:]) 
                    if len(self._risk_score_history) >= 10 else self.risk_score
                ),
                'risk_volatility': (
                    np.std(list(self._risk_score_history)[-20:])
                    if len(self._risk_score_history) >= 20 else 0.0
                )
            }
        }
        
        if base_details:
            base_details.update(monitor_details)
            return base_details
        
        return monitor_details

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, open_positions: Optional[Union[List[Dict], Dict[str, Dict]]] = None,
            current_step: Optional[int] = None, **kwargs) -> float:
        """Legacy compatibility method"""
        
        # Convert legacy parameters to InfoBus format
        mock_info_bus = {
            'step_idx': current_step or self.step_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'positions': []
        }
        
        # Convert positions to standard format
        if open_positions:
            if isinstance(open_positions, dict):
                for symbol, pos_data in open_positions.items():
                    mock_info_bus['positions'].append({
                        'symbol': symbol,
                        'size': pos_data.get('size', pos_data.get('lots', 0)),
                        'entry_step': pos_data.get('entry_step', 0),
                        'duration': pos_data.get('duration', 0),
                        'pnl': pos_data.get('pnl', 0)
                    })
            elif isinstance(open_positions, list):
                mock_info_bus['positions'] = open_positions
        
        # Use enhanced step method
        self._step_impl(mock_info_bus)
        
        return self.risk_score