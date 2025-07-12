from modules.core.mixins import SmartInfoBusStateMixin, SmartInfoBusTradingMixin
# ─────────────────────────────────────────────────────────────
# File: modules/visualization/visualization_interface.py
# Enhanced Visualization Interface with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import copy
from typing import List, Dict, Any, Optional, Union
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class VisualizationInterface(Module, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Enhanced visualization interface with InfoBus integration.
    Serves as the central data aggregator for all visualization and dashboard needs.
    Captures trading decisions, performance metrics, and system state for analysis.
    """

    def __init__(
        self,
        debug: bool = True,
        max_steps: int = 1000,
        stream_url: Optional[str] = None,
        dashboard_update_freq: int = 5,
        performance_window: int = 100,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 500),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_trading_state()
        
        # Core parameters
        self.debug = debug
        self.max_steps = int(max_steps)
        self.stream_url = stream_url
        self.dashboard_update_freq = int(dashboard_update_freq)
        self.performance_window = int(performance_window)
        
        # Data storage
        self.records: List[Dict[str, Any]] = []
        self.decision_trace: List[Dict[str, Any]] = []
        self.alert_history: deque = deque(maxlen=200)
        
        # Performance metrics with enhanced tracking
        self.performance_metrics: Dict[str, deque] = {
            'balance': deque(maxlen=self.performance_window),
            'equity': deque(maxlen=self.performance_window),
            'pnl': deque(maxlen=self.performance_window),
            'drawdown': deque(maxlen=self.performance_window),
            'trades': deque(maxlen=self.performance_window),
            'win_rate': deque(maxlen=self.performance_window),
            'consensus': deque(maxlen=self.performance_window),
            'position_count': deque(maxlen=self.performance_window),
            'risk_score': deque(maxlen=self.performance_window)
        }
        
        # Advanced analytics
        self.regime_analytics = defaultdict(lambda: {
            'time_spent': 0,
            'performance': [],
            'trade_count': 0,
            'avg_pnl': 0.0
        })
        
        self.session_analytics = defaultdict(lambda: {
            'time_spent': 0,
            'performance': [],
            'trade_count': 0,
            'avg_pnl': 0.0
        })
        
        # Real-time dashboard data
        self.dashboard_data = {}
        self.last_dashboard_update = 0
        
        # Streaming and connectivity
        self.stream_clients = set()
        self.streaming_enabled = bool(stream_url)
        
        # Statistics
        self.viz_stats = {
            'total_records': 0,
            'alerts_generated': 0,
            'dashboard_updates': 0,
            'stream_updates': 0,
            'data_points_collected': 0
        }
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "VisualizationInterface",
            "logs/visualization/visualization_interface.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("VisualizationInterface")
        
        self.log_operator_info(
            "📊 Visualization Interface initialized",
            max_steps=self.max_steps,
            streaming=self.streaming_enabled,
            performance_window=self.performance_window
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Clear data storage
        self.records.clear()
        self.decision_trace.clear()
        self.alert_history.clear()
        
        # Clear performance metrics
        for metric_deque in self.performance_metrics.values():
            metric_deque.clear()
        
        # Clear analytics
        self.regime_analytics.clear()
        self.session_analytics.clear()
        
        # Reset dashboard
        self.dashboard_data.clear()
        self.last_dashboard_update = 0
        
        # Reset statistics
        self.viz_stats = {
            'total_records': 0,
            'alerts_generated': 0,
            'dashboard_updates': 0,
            'stream_updates': 0,
            'data_points_collected': 0
        }
        
        self.log_operator_info("🔄 Visualization Interface reset - all data cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if info_bus:
            self._process_info_bus_data(info_bus)
        else:
            # Legacy support
            self._process_legacy_data(**kwargs)
        
        # Update dashboard if needed
        self._update_dashboard_if_needed()
        
        # Stream updates to connected clients
        self._stream_updates()

    def _process_info_bus_data(self, info_bus: InfoBus) -> None:
        """Enhanced InfoBus data processing with comprehensive extraction"""
        
        try:
            # Extract standard context
            context = extract_standard_context(info_bus)
            
            # Extract comprehensive trading data
            trading_data = self._extract_comprehensive_trading_data(info_bus)
            
            # Create comprehensive record
            record = {
                # Timestamps
                '_time': datetime.datetime.now().isoformat(),
                'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
                'step': info_bus.get('step_idx', 0),
                'episode': info_bus.get('episode_idx', 0),
                
                # Financial metrics
                'balance': trading_data.get('balance', 0.0),
                'equity': trading_data.get('equity', 0.0),
                'pnl_today': trading_data.get('pnl_today', 0.0),
                'pnl_total': trading_data.get('pnl_total', 0.0),
                'drawdown': trading_data.get('drawdown', 0.0),
                'max_drawdown': trading_data.get('max_drawdown', 0.0),
                
                # Trading activity
                'positions': trading_data.get('position_count', 0),
                'active_trades': trading_data.get('active_trades', []),
                'completed_trades': trading_data.get('completed_trades', []),
                'trade_count': trading_data.get('trade_count', 0),
                'win_rate': trading_data.get('win_rate', 0.0),
                
                # Market context
                'regime': context.get('regime', 'unknown'),
                'session': context.get('session', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium'),
                'market_open': context.get('market_open', True),
                
                # System state
                'consensus': trading_data.get('consensus', 0.0),
                'risk_score': trading_data.get('risk_score', 0.0),
                'alerts': info_bus.get('alerts', []),
                'system_status': trading_data.get('system_status', 'active'),
                
                # Module performance
                'module_performance': trading_data.get('module_performance', {}),
                'voting_summary': trading_data.get('voting_summary', {}),
                
                # Advanced metrics
                'sharpe_ratio': trading_data.get('sharpe_ratio', 0.0),
                'total_exposure': trading_data.get('total_exposure', 0.0),
                'leverage': trading_data.get('leverage', 1.0)
            }
            
            # Process and store record
            self._process_record(record)
            
            # Update performance metrics
            self._update_performance_metrics(record)
            
            # Process alerts
            self._process_alerts(record.get('alerts', []), record)
            
            # Update analytics
            self._update_regime_analytics(record)
            self._update_session_analytics(record)
            
            # Update InfoBus with visualization data
            self._update_info_bus_with_viz_data(info_bus)
            
        except Exception as e:
            self.log_operator_error(f"InfoBus data processing failed: {e}")

    def _extract_comprehensive_trading_data(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract comprehensive trading data from InfoBus"""
        
        data = {}
        
        try:
            # Financial data
            risk_data = info_bus.get('risk', {})
            data['balance'] = float(risk_data.get('balance', risk_data.get('equity', 0)))
            data['equity'] = float(risk_data.get('equity', data['balance']))
            data['drawdown'] = float(risk_data.get('current_drawdown', 0))
            data['max_drawdown'] = float(risk_data.get('max_drawdown', 0))
            
            # Trading activity
            positions = InfoBusExtractor.get_positions(info_bus)
            data['position_count'] = len(positions)
            data['active_trades'] = positions
            
            recent_trades = info_bus.get('recent_trades', [])
            data['completed_trades'] = recent_trades
            data['trade_count'] = len(recent_trades)
            
            # Calculate win rate
            if recent_trades:
                winning_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
                data['win_rate'] = winning_trades / len(recent_trades)
                
                # Calculate total P&L
                data['pnl_total'] = sum(trade.get('pnl', 0) for trade in recent_trades)
                data['pnl_today'] = sum(trade.get('pnl', 0) for trade in recent_trades[-10:])  # Recent trades
                
                # Calculate Sharpe ratio if possible
                returns = [trade.get('pnl', 0) / max(data['balance'], 1000) for trade in recent_trades]
                if len(returns) > 5 and np.std(returns) > 0:
                    data['sharpe_ratio'] = np.sqrt(252) * np.mean(returns) / np.std(returns)
            else:
                data['win_rate'] = 0.0
                data['pnl_total'] = 0.0
                data['pnl_today'] = 0.0
                data['sharpe_ratio'] = 0.0
            
            # System state
            consensus_data = info_bus.get('consensus', {})
            data['consensus'] = float(consensus_data.get('score', 0.5))
            
            # Risk metrics
            data['risk_score'] = float(risk_data.get('risk_score', 0.0))
            
            # Calculate exposure
            total_exposure = sum(abs(pos.get('size', 0)) for pos in positions)
            data['total_exposure'] = total_exposure
            data['leverage'] = total_exposure / max(data['balance'], 1000)
            
            # Module performance
            module_data = info_bus.get('module_data', {})
            data['module_performance'] = self._extract_module_performance(module_data)
            
            # Voting summary
            data['voting_summary'] = InfoBusExtractor.get_voting_summary(info_bus)
            
            # System status
            data['system_status'] = info_bus.get('system_status', 'active')
            
        except Exception as e:
            self.log_operator_warning(f"Trading data extraction failed: {e}")
            # Provide safe defaults
            data = {
                'balance': 10000.0,
                'equity': 10000.0,
                'drawdown': 0.0,
                'max_drawdown': 0.0,
                'position_count': 0,
                'active_trades': [],
                'completed_trades': [],
                'trade_count': 0,
                'win_rate': 0.0,
                'pnl_total': 0.0,
                'pnl_today': 0.0,
                'consensus': 0.5,
                'risk_score': 0.0,
                'total_exposure': 0.0,
                'leverage': 1.0,
                'module_performance': {},
                'voting_summary': {},
                'system_status': 'active',
                'sharpe_ratio': 0.0
            }
        
        return data

    def _extract_module_performance(self, module_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance data from all modules"""
        
        performance = {}
        
        try:
            # Strategy Arbiter
            arbiter_data = module_data.get('strategy_arbiter', {})
            if arbiter_data:
                performance['arbiter'] = {
                    'gate_pass_rate': arbiter_data.get('gate_stats', {}).get('pass_rate', 0.5),
                    'decision_confidence': arbiter_data.get('voting_quality', {}).get('decision_confidence', 0.5),
                    'member_count': arbiter_data.get('member_count', 0)
                }
            
            # Trading Mode Manager
            mode_data = module_data.get('trading_mode_manager', {})
            if mode_data:
                performance['trading_mode'] = {
                    'current_mode': mode_data.get('current_mode', 'normal'),
                    'effectiveness': mode_data.get('effectiveness', 0.5),
                    'mode_persistence': mode_data.get('mode_persistence', 0)
                }
            
            # Risk modules
            risk_modules = ['dynamic_risk_controller', 'portfolio_risk_system', 'execution_quality_monitor']
            for module_name in risk_modules:
                module_info = module_data.get(module_name, {})
                if module_info:
                    performance[module_name] = {
                        'quality_score': module_info.get('quality_score', 0.5),
                        'effectiveness': module_info.get('effectiveness_score', 0.5),
                        'alerts_count': len(module_info.get('recent_alerts', []))
                    }
            
        except Exception as e:
            self.log_operator_warning(f"Module performance extraction failed: {e}")
        
        return performance

    def _process_record(self, record: Dict[str, Any]) -> None:
        """Process and store record with size management"""
        
        try:
            # Add to records with size management
            self.records.append(record)
            if len(self.records) > self.max_steps:
                self.records = self.records[-self.max_steps:]
            
            # Add to decision trace
            decision_summary = {
                'timestamp': record['_time'],
                'step': record['step'],
                'balance': record['balance'],
                'pnl': record['pnl_today'],
                'positions': record['positions'],
                'regime': record['regime'],
                'alerts': len(record.get('alerts', []))
            }
            self.decision_trace.append(decision_summary)
            if len(self.decision_trace) > self.max_steps:
                self.decision_trace = self.decision_trace[-self.max_steps:]
            
            # Update statistics
            self.viz_stats['total_records'] += 1
            self.viz_stats['data_points_collected'] += len(record)
            
            # Debug output
            if self.debug:
                self._print_summary(record)
            
        except Exception as e:
            self.log_operator_error(f"Record processing failed: {e}")

    def _update_performance_metrics(self, record: Dict[str, Any]) -> None:
        """Update performance metrics collections"""
        
        try:
            # Update all performance metrics
            for metric_name, metric_deque in self.performance_metrics.items():
                value = record.get(metric_name, 0.0)
                metric_deque.append(float(value))
            
            # Update performance tracking
            self._update_performance_metric('viz_records_processed', self.viz_stats['total_records'])
            self._update_performance_metric('current_balance', record.get('balance', 0))
            self._update_performance_metric('current_drawdown', record.get('drawdown', 0))
            
        except Exception as e:
            self.log_operator_warning(f"Performance metrics update failed: {e}")

    def _process_alerts(self, alerts: List[Dict], record: Dict[str, Any]) -> None:
        """Process and categorize alerts"""
        
        try:
            for alert in alerts:
                alert_record = {
                    'time': record['_time'],
                    'step': record['step'],
                    'alert': alert,
                    'severity': alert.get('severity', 'info'),
                    'module': alert.get('module', 'unknown'),
                    'context': {
                        'balance': record['balance'],
                        'regime': record['regime'],
                        'positions': record['positions']
                    }
                }
                self.alert_history.append(alert_record)
                self.viz_stats['alerts_generated'] += 1
                
                # Log critical alerts
                if alert.get('severity') in ['critical', 'error']:
                    self.log_operator_warning(
                        f"🚨 Critical alert: {alert.get('message', alert)}",
                        module=alert.get('module', 'unknown'),
                        step=record['step']
                    )
            
        except Exception as e:
            self.log_operator_warning(f"Alert processing failed: {e}")

    def _update_regime_analytics(self, record: Dict[str, Any]) -> None:
        """Update regime-based analytics"""
        
        try:
            regime = record.get('regime', 'unknown')
            analytics = self.regime_analytics[regime]
            
            analytics['time_spent'] += 1
            analytics['performance'].append(record.get('pnl_today', 0.0))
            
            if record.get('trade_count', 0) > 0:
                analytics['trade_count'] += 1
            
            # Update average P&L
            if analytics['performance']:
                analytics['avg_pnl'] = np.mean(analytics['performance'])
            
        except Exception as e:
            self.log_operator_warning(f"Regime analytics update failed: {e}")

    def _update_session_analytics(self, record: Dict[str, Any]) -> None:
        """Update session-based analytics"""
        
        try:
            session = record.get('session', 'unknown')
            analytics = self.session_analytics[session]
            
            analytics['time_spent'] += 1
            analytics['performance'].append(record.get('pnl_today', 0.0))
            
            if record.get('trade_count', 0) > 0:
                analytics['trade_count'] += 1
            
            # Update average P&L
            if analytics['performance']:
                analytics['avg_pnl'] = np.mean(analytics['performance'])
            
        except Exception as e:
            self.log_operator_warning(f"Session analytics update failed: {e}")

    def _update_dashboard_if_needed(self) -> None:
        """Update dashboard data if enough time has passed"""
        
        try:
            current_time = datetime.datetime.now().timestamp()
            
            if (current_time - self.last_dashboard_update) >= self.dashboard_update_freq:
                self.dashboard_data = self._generate_dashboard_data()
                self.last_dashboard_update = current_time
                self.viz_stats['dashboard_updates'] += 1
                
                self.log_operator_info(
                    f"📊 Dashboard updated",
                    records=len(self.records),
                    alerts=len(self.alert_history),
                    balance=self.dashboard_data.get('current', {}).get('balance', 0)
                )
            
        except Exception as e:
            self.log_operator_warning(f"Dashboard update failed: {e}")

    def _generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate comprehensive dashboard data"""
        
        try:
            if not self.records:
                return {}
            
            current_record = self.records[-1]
            recent_records = self.records[-min(100, len(self.records)):]
            
            return {
                'current': current_record,
                'performance': {
                    'balance_history': list(self.performance_metrics['balance']),
                    'pnl_history': list(self.performance_metrics['pnl']),
                    'drawdown_history': list(self.performance_metrics['drawdown']),
                    'position_history': list(self.performance_metrics['position_count']),
                    'consensus_history': list(self.performance_metrics['consensus']),
                    'risk_history': list(self.performance_metrics['risk_score'])
                },
                'statistics': {
                    'total_pnl': sum(self.performance_metrics['pnl']),
                    'max_drawdown': max(self.performance_metrics['drawdown']) if self.performance_metrics['drawdown'] else 0,
                    'current_win_rate': self.performance_metrics['win_rate'][-1] if self.performance_metrics['win_rate'] else 0,
                    'avg_consensus': np.mean(list(self.performance_metrics['consensus'])) if self.performance_metrics['consensus'] else 0,
                    'recent_alerts': list(self.alert_history)[-10:],
                    'trade_count': self.performance_metrics['trades'][-1] if self.performance_metrics['trades'] else 0
                },
                'analytics': {
                    'regime_distribution': self._calculate_regime_distribution(recent_records),
                    'session_performance': self._calculate_session_performance(),
                    'module_performance': current_record.get('module_performance', {}),
                    'voting_summary': current_record.get('voting_summary', {})
                },
                'metadata': {
                    'last_update': datetime.datetime.now().isoformat(),
                    'data_points': len(self.records),
                    'timespan_hours': self._calculate_timespan_hours(),
                    'update_frequency': self.dashboard_update_freq
                }
            }
            
        except Exception as e:
            self.log_operator_error(f"Dashboard data generation failed: {e}")
            return {}

    def _calculate_regime_distribution(self, records: List[Dict]) -> Dict[str, float]:
        """Calculate time spent in each regime"""
        
        try:
            regime_counts = defaultdict(int)
            for record in records:
                regime = record.get('regime', 'unknown')
                regime_counts[regime] += 1
            
            total = len(records)
            return {regime: count/total for regime, count in regime_counts.items()}
            
        except Exception as e:
            self.log_operator_warning(f"Regime distribution calculation failed: {e}")
            return {}

    def _calculate_session_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance by trading session"""
        
        try:
            session_perf = {}
            for session, analytics in self.session_analytics.items():
                if analytics['performance']:
                    session_perf[session] = {
                        'avg_pnl': analytics['avg_pnl'],
                        'total_pnl': sum(analytics['performance']),
                        'trade_count': analytics['trade_count'],
                        'time_spent': analytics['time_spent']
                    }
            return session_perf
            
        except Exception as e:
            self.log_operator_warning(f"Session performance calculation failed: {e}")
            return {}

    def _calculate_timespan_hours(self) -> float:
        """Calculate timespan of collected data in hours"""
        
        try:
            if len(self.records) < 2:
                return 0.0
            
            start_time = datetime.datetime.fromisoformat(self.records[0]['_time'])
            end_time = datetime.datetime.fromisoformat(self.records[-1]['_time'])
            
            return (end_time - start_time).total_seconds() / 3600.0
            
        except Exception:
            return 0.0

    def _stream_updates(self) -> None:
        """Stream updates to connected clients"""
        
        try:
            if not self.streaming_enabled or not self.stream_clients:
                return
            
            if self.records:
                stream_data = {
                    'type': 'trading_update',
                    'data': self.records[-1],
                    'timestamp': datetime.datetime.now().isoformat()
                }
                
                self._broadcast_to_clients(stream_data)
                self.viz_stats['stream_updates'] += 1
            
        except Exception as e:
            self.log_operator_warning(f"Streaming update failed: {e}")

    def _broadcast_to_clients(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected streaming clients"""
        
        # This would integrate with your WebSocket implementation
        # Placeholder for actual streaming logic
        if self.stream_url:
            try:
                import requests
                requests.post(self.stream_url, json=data, timeout=0.5)
            except Exception:
                pass  # Don't let streaming errors affect trading

    def _update_info_bus_with_viz_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with visualization data"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'visualization_interface', {
            'total_records': len(self.records),
            'performance_metrics': {k: list(v)[-10:] for k, v in self.performance_metrics.items()},
            'recent_alerts': list(self.alert_history)[-5:],
            'dashboard_ready': len(self.dashboard_data) > 0,
            'streaming_enabled': self.streaming_enabled,
            'statistics': self.viz_stats.copy(),
            'regime_analytics': dict(self.regime_analytics),
            'session_analytics': dict(self.session_analytics)
        })
        
        # Add visualization summary to InfoBus
        if 'visualization' not in info_bus:
            info_bus['visualization'] = {}
        
        info_bus['visualization'].update({
            'interface_ready': True,
            'data_points': len(self.records),
            'last_update': datetime.datetime.now().isoformat(),
            'dashboard_data': self.dashboard_data
        })

    def _print_summary(self, record: Dict[str, Any]) -> None:
        """Print formatted summary for debugging"""
        
        try:
            balance = record.get('balance', 0)
            pnl = record.get('pnl_today', 0)
            positions = record.get('positions', 0)
            drawdown = record.get('drawdown', 0)
            regime = record.get('regime', 'unknown')
            
            print(f"[VizInterface] Step {record.get('step', 0)} | "
                  f"Balance: ${balance:.2f} | "
                  f"P&L: ${pnl:+.2f} | "
                  f"DD: {drawdown:.1%} | "
                  f"Pos: {positions} | "
                  f"Regime: {regime}")
            
        except Exception as e:
            self.log_operator_warning(f"Summary printing failed: {e}")

    def _process_legacy_data(self, **kwargs) -> None:
        """Process legacy data format for backward compatibility"""
        
        try:
            record = dict(kwargs)
            record['_time'] = datetime.datetime.now().isoformat()
            record['timestamp'] = record['_time']
            
            # Add defaults for missing fields
            defaults = {
                'step': 0,
                'episode': 0,
                'balance': 10000.0,
                'equity': 10000.0,
                'pnl_today': 0.0,
                'drawdown': 0.0,
                'positions': 0,
                'regime': 'unknown',
                'session': 'unknown',
                'consensus': 0.5,
                'alerts': []
            }
            
            for key, default_value in defaults.items():
                if key not in record:
                    record[key] = default_value
            
            self._process_record(record)
            
        except Exception as e:
            self.log_operator_error(f"Legacy data processing failed: {e}")

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data or self._generate_dashboard_data()

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        if not self.performance_metrics['balance']:
            return {}
        
        return {
            'current_balance': self.performance_metrics['balance'][-1],
            'total_pnl': sum(self.performance_metrics['pnl']),
            'max_drawdown': max(self.performance_metrics['drawdown']) if self.performance_metrics['drawdown'] else 0,
            'win_rate': self.performance_metrics['win_rate'][-1] if self.performance_metrics['win_rate'] else 0,
            'trade_count': self.performance_metrics['trades'][-1] if self.performance_metrics['trades'] else 0,
            'avg_consensus': np.mean(list(self.performance_metrics['consensus'])) if self.performance_metrics['consensus'] else 0,
            'recent_alerts': len(list(self.alert_history)[-10:])
        }

    def generate_performance_report(self) -> str:
        """Generate comprehensive text performance report"""
        
        try:
            data = self.get_dashboard_data()
            stats = data.get('statistics', {})
            current = data.get('current', {})
            
            report = f"""
📊 TRADING PERFORMANCE REPORT
═══════════════════════════════════════
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💰 FINANCIAL SUMMARY:
• Current Balance: ${current.get('balance', 0):,.2f}
• Total P&L: ${stats.get('total_pnl', 0):+,.2f}
• Max Drawdown: {stats.get('max_drawdown', 0):.1%}
• Current Win Rate: {stats.get('current_win_rate', 0):.1%}

📈 TRADING ACTIVITY:
• Total Trades: {stats.get('trade_count', 0):,}
• Active Positions: {current.get('positions', 0)}
• Current Regime: {current.get('regime', 'unknown').title()}
• Average Consensus: {stats.get('avg_consensus', 0):.1%}

🚨 RECENT ALERTS:
"""
            
            for alert in stats.get('recent_alerts', [])[-5:]:
                alert_time = alert.get('time', '')[:19]  # Remove microseconds
                alert_msg = alert.get('alert', {}).get('message', str(alert.get('alert', 'Unknown alert')))
                report += f"• [{alert_time}] {alert_msg}\n"
            
            report += f"""
📊 SYSTEM STATISTICS:
• Data Points Collected: {self.viz_stats['data_points_collected']:,}
• Dashboard Updates: {self.viz_stats['dashboard_updates']:,}
• Streaming Enabled: {'✅ Yes' if self.streaming_enabled else '❌ No'}
• Data Timespan: {data.get('metadata', {}).get('timespan_hours', 0):.1f} hours
            """
            
            return report
            
        except Exception as e:
            self.log_operator_error(f"Performance report generation failed: {e}")
            return f"Error generating report: {e}"

    def add_stream_client(self, client_id: str) -> None:
        """Add streaming client"""
        self.stream_clients.add(client_id)
        self.log_operator_info(f"📡 Streaming client added: {client_id}")

    def remove_stream_client(self, client_id: str) -> None:
        """Remove streaming client"""
        self.stream_clients.discard(client_id)
        self.log_operator_info(f"📡 Streaming client removed: {client_id}")

    def get_observation_components(self) -> np.ndarray:
        """Return visualization metrics for observation"""
        
        try:
            if not self.records:
                return np.zeros(8, dtype=np.float32)
            
            recent_records = self.records[-10:] if len(self.records) >= 10 else self.records
            
            features = [
                float(len(self.records) / self.max_steps),  # Data fullness
                float(np.mean([r.get('balance', 0) for r in recent_records]) / 10000),  # Normalized balance
                float(np.mean([r.get('drawdown', 0) for r in recent_records])),  # Average drawdown
                float(len(self.alert_history) / 200),  # Alert fullness
                float(self.viz_stats['dashboard_updates']),  # Dashboard activity
                float(self.streaming_enabled),  # Streaming status
                float(len(self.stream_clients)),  # Connected clients
                float(bool(self.dashboard_data))  # Dashboard ready
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
                "max_steps": self.max_steps,
                "stream_url": self.stream_url,
                "dashboard_update_freq": self.dashboard_update_freq,
                "performance_window": self.performance_window
            },
            "data": {
                "records": self.records[-100:],  # Save recent only
                "decision_trace": list(self.decision_trace)[-50:],
                "alert_history": list(self.alert_history)[-50:]
            },
            "metrics": {
                "performance_metrics": {k: list(v) for k, v in self.performance_metrics.items()},
                "regime_analytics": dict(self.regime_analytics),
                "session_analytics": dict(self.session_analytics)
            },
            "statistics": self.viz_stats.copy(),
            "dashboard": {
                "last_update": self.last_dashboard_update,
                "streaming_enabled": self.streaming_enabled,
                "stream_clients": list(self.stream_clients)
            }
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.debug = bool(config.get("debug", self.debug))
        self.max_steps = int(config.get("max_steps", self.max_steps))
        self.stream_url = config.get("stream_url", self.stream_url)
        self.dashboard_update_freq = int(config.get("dashboard_update_freq", self.dashboard_update_freq))
        self.performance_window = int(config.get("performance_window", self.performance_window))
        
        # Load data
        data = state.get("data", {})
        self.records = data.get("records", [])
        self.decision_trace = data.get("decision_trace", [])
        
        alert_history = data.get("alert_history", [])
        self.alert_history.clear()
        for alert in alert_history:
            self.alert_history.append(alert)
        
        # Load metrics
        metrics = state.get("metrics", {})
        performance_metrics = metrics.get("performance_metrics", {})
        for metric_name, values in performance_metrics.items():
            if metric_name in self.performance_metrics:
                self.performance_metrics[metric_name].clear()
                for value in values:
                    self.performance_metrics[metric_name].append(value)
        
        self.regime_analytics.update(metrics.get("regime_analytics", {}))
        self.session_analytics.update(metrics.get("session_analytics", {}))
        
        # Load statistics
        self.viz_stats.update(state.get("statistics", {}))
        
        # Load dashboard state
        dashboard = state.get("dashboard", {})
        self.last_dashboard_update = dashboard.get("last_update", 0)
        self.streaming_enabled = bool(dashboard.get("streaming_enabled", self.streaming_enabled))
        self.stream_clients = set(dashboard.get("stream_clients", []))

    # ================== LEGACY COMPATIBILITY ==================

    def record_step(self, **kwargs) -> None:
        """Legacy recording method for backward compatibility"""
        self._process_legacy_data(**kwargs)