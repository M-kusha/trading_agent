# ─────────────────────────────────────────────────────────────
# File: modules/visualization/visualization_interface.py
# Enhanced Visualization Interface with Modern Architecture
# ─────────────────────────────────────────────────────────────

from modules.contracts import module_args
import numpy as np
import datetime
import time
import copy
from typing import List, Dict, Any, Optional, Union
from collections import deque, defaultdict

# Modern imports
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(**module_args(
    "VisualizationInterface",
    description="Modern visualization interface with comprehensive SmartInfoBus integration.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=3000,
))
class VisualizationInterface(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Modern visualization interface with comprehensive SmartInfoBus integration.
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
        # Initialize BaseModule
        super().__init__(**kwargs)
        
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
        self.regime_analytics: Dict[str, Dict[str, Union[int, float, List[float]]]] = defaultdict(lambda: {
            'time_spent': 0,
            'performance': [],
            'trade_count': 0,
            'avg_pnl': 0.0
        })
        
        self.session_analytics: Dict[str, Dict[str, Union[int, float, List[float]]]] = defaultdict(lambda: {
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
        
        # Circuit breaker and error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False

        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        self.logger.info(format_operator_message(
            icon="[STATS]",
            message="Visualization Interface initialized",
            max_steps=self.max_steps,
            streaming=self.streaming_enabled,
            performance_window=self.performance_window
        ))

    def _initialize(self):
        """Initialize module - required by BaseModule"""
        # Load default balance from risk_policy.yaml
        self._default_balance = 100000.0
        try:
            import yaml
            import os
            config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "risk_policy.yaml")
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    policy = yaml.safe_load(f) or {}
                self._default_balance = float(
                    policy.get("prop_firm", {}).get("account_size")
                    or policy.get("lot_sizing", {}).get("account_balance")
                    or 100000.0
                )
        except Exception:
            pass

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in visualization data quality and completeness."""
        try:
            # Base confidence from data availability
            base_confidence = 0.8  # High confidence in data aggregation capabilities
            
            # Adjust based on error count
            if self.is_disabled:
                return 0.1
            elif self.error_count > 0:
                base_confidence *= max(0.3, 1.0 - self.error_count * 0.2)
            
            # Boost confidence based on data quality
            if self.records:
                # Recent data availability boosts confidence
                recent_records = len(self.records) / max(self.max_steps, 1)
                base_confidence *= (0.5 + recent_records * 0.5)
            
            # Dashboard readiness factor
            if self.dashboard_data:
                base_confidence *= 1.1  # Small boost for active dashboard
            
            # Streaming capability factor
            if self.streaming_enabled and self.stream_clients:
                base_confidence *= 1.05  # Small boost for active streaming
            
            return max(0.0, min(1.0, base_confidence))
            
        except Exception as e:
            self.logger.warning(f"Error calculating confidence: {e}")
            return 0.5

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose visualization and data management actions."""
        try:
            confidence = await self.calculate_confidence({}, **inputs)
            
            action = {
                'type': 'data_management',
                'confidence': confidence,
                'recommendations': []
            }
            
            # Data collection recommendations
            if len(self.records) < self.max_steps * 0.1:
                action['recommendations'].append({
                    'action': 'increase_data_collection',
                    'reason': f'Low data volume: {len(self.records)} records',
                    'priority': 'medium'
                })
            
            # Dashboard update recommendations
            current_time = time.time()
            if (current_time - self.last_dashboard_update) > self.dashboard_update_freq * 2:
                action['recommendations'].append({
                    'action': 'update_dashboard',
                    'reason': f'Dashboard outdated by {current_time - self.last_dashboard_update:.1f}s',
                    'priority': 'medium'
                })
            
            # Alert management recommendations
            if len(self.alert_history) > 150:
                action['recommendations'].append({
                    'action': 'manage_alert_history',
                    'reason': f'Alert history size: {len(self.alert_history)}',
                    'priority': 'low'
                })
            
            # Performance monitoring recommendations
            if self.performance_metrics['balance'] and len(self.performance_metrics['balance']) > 0:
                recent_balance = list(self.performance_metrics['balance'])[-1]
                if recent_balance < 5000:  # Low balance warning
                    action['recommendations'].append({
                        'action': 'monitor_balance_risk',
                        'reason': f'Low balance detected: ${recent_balance:.2f}',
                        'priority': 'high'
                    })
            
            # Error handling recommendations
            if self.error_count > 2:
                action['recommendations'].append({
                    'action': 'investigate_errors',
                    'reason': f'High error count: {self.error_count}',
                    'priority': 'high'
                })
            
            # Streaming optimization recommendations
            if self.streaming_enabled and not self.stream_clients:
                action['recommendations'].append({
                    'action': 'optimize_streaming',
                    'reason': 'Streaming enabled but no clients connected',
                    'priority': 'low'
                })
            
            return action
            
        except Exception as e:
            self.logger.warning(f"Error proposing action: {e}")
            return {
                'type': 'data_management',
                'confidence': 0.5,
                'recommendations': [{'action': 'investigate_error', 'reason': str(e), 'priority': 'high'}]
            }

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="VisualizationInterface",
            log_path="logs/visualization/visualization_interface.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("VisualizationInterface", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        
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
        
        # Reset error state
        self.error_count = 0
        self.is_disabled = False
        
        self.logger.info(format_operator_message(
            icon="[RELOAD]",
            message="Visualization Interface reset - all data cleared"
        ))

    async def process(self, **inputs) -> Dict[str, Any]:
        """Modern async processing with comprehensive data aggregation"""
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                disabled = self._generate_disabled_response()
                # Ensure provides contract with thesis
                return {
                    'visualization_data': {
                        'total_records': 0,
                        'performance_metrics': {},
                        'recent_alerts': [],
                        'dashboard_ready': False,
                        'streaming_enabled': self.streaming_enabled,
                        'statistics': self.viz_stats.copy(),
                        'regime_analytics': {},
                        'session_analytics': {}
                    },
                    'performance_metrics': {},
                    'dashboard_data': {},
                    'alert_timeline': [],
                    'analytics_reports': {'status': 'disabled'},
                    'streaming_data': {'enabled': self.streaming_enabled, 'clients': len(self.stream_clients)},
                    'system_status': {'status': 'disabled', 'reason': disabled.get('reason', 'circuit_breaker_triggered')},
                    '_thesis': 'VisualizationInterface disabled due to circuit breaker'
                }
            
            # Process comprehensive data from SmartInfoBus
            data_results = await self._process_comprehensive_data()
            
            # Update dashboard if needed
            await self._update_dashboard_if_needed()
            
            # Stream updates to connected clients
            await self._stream_updates()
            
            # Update SmartInfoBus with aggregated visualization data
            await self._update_smartinfobus_comprehensive(data_results)

            # Also publish data synchronously as backup (critical for UI)
            self._publish_infobus_data_sync(data_results)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('VisualizationInterface', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            # Compose return payload to satisfy provides contract
            thesis = f"Processed {data_results.get('records_processed', 0)} records; dashboard_ready={bool(self.dashboard_data)}"
            return {
                'visualization_data': {
                    'total_records': len(self.records),
                    'performance_metrics': {k: list(v)[-10:] for k, v in self.performance_metrics.items()},
                    'recent_alerts': list(self.alert_history)[-5:],
                    'dashboard_ready': bool(self.dashboard_data),
                    'streaming_enabled': self.streaming_enabled,
                    'statistics': self.viz_stats.copy(),
                    'regime_analytics': dict(self.regime_analytics),
                    'session_analytics': dict(self.session_analytics)
                },
                'performance_metrics': self.get_performance_summary(),
                'dashboard_data': self.dashboard_data or {},
                'alert_timeline': list(self.alert_history),
                'analytics_reports': {
                    'performance_report': self.generate_performance_report()
                },
                'streaming_data': {
                    'enabled': self.streaming_enabled,
                    'clients': len(self.stream_clients)
                },
                'system_status': self.get_health_status(),
                '_thesis': thesis
            }
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _process_comprehensive_data(self) -> Dict[str, Any]:
        """Process comprehensive data from SmartInfoBus"""
        try:
            # Extract comprehensive trading data
            trading_data = await self._extract_comprehensive_trading_data()
            
            # Create comprehensive record
            record = await self._create_comprehensive_record(trading_data)
            
            # Process and store record
            self._process_record(record)
            
            # Update performance metrics
            self._update_performance_metrics(record)
            
            # Process alerts
            self._process_alerts(record.get('alerts', []), record)
            
            # Update analytics
            self._update_regime_analytics(record)
            self._update_session_analytics(record)
            
            return {
                'records_processed': 1,
                'total_records': len(self.records),
                'alerts_processed': len(record.get('alerts', [])),
                'dashboard_ready': bool(self.dashboard_data),
                'status': 'success'
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "data_processing")
            self.logger.error(f"Comprehensive data processing failed: {error_context}")
            return {'records_processed': 0, 'error': str(error_context), 'status': 'error'}

    async def _extract_comprehensive_trading_data(self) -> Dict[str, Any]:
        """Extract comprehensive trading data from SmartInfoBus"""
        
        data = {}
        
        try:
            # Financial data (robust fallbacks across publishers)
            risk_data = self.smart_bus.get('risk_metrics', 'VisualizationInterface') or {}
            portfolio_metrics = self.smart_bus.get('portfolio_metrics', 'VisualizationInterface') or {}
            market_state = self.smart_bus.get('market_state', 'VisualizationInterface') or {}
            # Top-level fallbacks written by PositionManager
            top_balance = self.smart_bus.get('balance', 'VisualizationInterface')
            top_equity = self.smart_bus.get('equity', 'VisualizationInterface')

            # Balance/equity (bus-first), then environment-config/performance anchors, else final default
            fallback_sources: Dict[str, Any] = {}
            balance_val = (risk_data.get('balance') if isinstance(risk_data.get('balance'), (int, float)) else None)
            if balance_val is None:
                pm_bal = portfolio_metrics.get('balance')
                if isinstance(pm_bal, (int, float)):
                    balance_val = pm_bal
                elif isinstance(top_balance, (int, float)):
                    balance_val = top_balance
                else:
                    ms_balance = market_state.get('balance')
                    if isinstance(ms_balance, (int, float)):
                        # Use environment-published market_state as a reliable early anchor
                        balance_val = float(ms_balance)
                        fallback_sources['balance_fallback'] = 'market_state.balance'
            # environment_config/performance_data fallback
            if balance_val is None:
                try:
                    env_cfg = self.smart_bus.get('environment_config', 'VisualizationInterface', default={}) or {}
                except Exception:
                    env_cfg = {}
                try:
                    perf = self.smart_bus.get('performance_data', 'VisualizationInterface', default={}) or {}
                except Exception:
                    perf = {}
                if isinstance(env_cfg.get('initial_balance'), (int, float)):
                    balance_val = float(env_cfg['initial_balance']); fallback_sources['balance_fallback'] = 'environment_config.initial_balance'
                elif isinstance(perf.get('balance'), (int, float)):
                    balance_val = float(perf['balance']); fallback_sources['balance_fallback'] = 'performance_data.balance'
                elif isinstance(perf.get('initial_balance'), (int, float)):
                    balance_val = float(perf['initial_balance']); fallback_sources['balance_fallback'] = 'performance_data.initial_balance'
            if balance_val is None:
                # Use default from risk_policy.yaml loaded in _initialize
                balance_val = getattr(self, '_default_balance', 100000.0)
                fallback_sources['balance_fallback'] = 'risk_policy_yaml_default'

            equity_val = (risk_data.get('equity') if isinstance(risk_data.get('equity'), (int, float)) else None)
            if equity_val is None and isinstance(top_equity, (int, float)):
                equity_val = top_equity
            if equity_val is None:
                equity_val = balance_val; fallback_sources['equity_fallback'] = 'use_balance'

            data['balance'] = float(balance_val)
            data['equity'] = float(equity_val)
            if fallback_sources:
                data['_fallback_info'] = fallback_sources

            # Drawdown
            dd = None
            if isinstance(risk_data.get('current_drawdown'), (int, float)):
                dd = risk_data.get('current_drawdown')
            elif isinstance(portfolio_metrics.get('drawdown'), (int, float)):
                dd = portfolio_metrics.get('drawdown')
            elif isinstance(market_state.get('current_drawdown'), (int, float)):
                dd = market_state.get('current_drawdown')
            data['drawdown'] = float(dd if dd is not None else 0.0)

            # Max drawdown (best-effort)
            mdd = risk_data.get('max_drawdown')
            if not isinstance(mdd, (int, float)):
                mdd = portfolio_metrics.get('max_drawdown')
            data['max_drawdown'] = float(mdd if isinstance(mdd, (int, float)) else 0.0)
            
            # Trading activity
            positions = self.smart_bus.get('positions', 'VisualizationInterface')
            if not positions:
                positions = self.smart_bus.get('current_positions', 'VisualizationInterface') or []
            if not isinstance(positions, list):
                # Some publishers use a dict keyed by instrument
                if isinstance(positions, dict):
                    positions = list(positions.values())
                else:
                    positions = []
            data['position_count'] = len(positions)
            data['active_trades'] = positions
            
            # Recent/completed trades
            recent_trades = self.smart_bus.get('recent_trades', 'VisualizationInterface')
            if not recent_trades:
                trades_all = self.smart_bus.get('trades', 'VisualizationInterface') or []
                if isinstance(trades_all, list):
                    recent_trades = trades_all[-20:]
            # Some modules nest under trade_data
            if not recent_trades:
                trade_data = self.smart_bus.get('trade_data', 'VisualizationInterface') or {}
                if isinstance(trade_data, dict):
                    recent_trades = trade_data.get('recent_trades', [])

            if not isinstance(recent_trades, list):
                recent_trades = []
            data['completed_trades'] = recent_trades
            data['trade_count'] = len(recent_trades)
            
            # Calculate win rate and P&L
            if recent_trades:
                winning_trades = sum(1 for trade in recent_trades if trade.get('pnl', 0) > 0)
                data['win_rate'] = winning_trades / len(recent_trades)
                data['pnl_total'] = sum(trade.get('pnl', 0) for trade in recent_trades)
                data['pnl_today'] = sum(trade.get('pnl', 0) for trade in recent_trades[-10:])
                
                # Calculate Sharpe ratio if possible
                returns = [trade.get('pnl', 0) / max(data['balance'], 1000) for trade in recent_trades]
                if len(returns) > 5 and np.std(returns) > 0:
                    data['sharpe_ratio'] = np.sqrt(252) * np.mean(returns) / np.std(returns)
                else:
                    data['sharpe_ratio'] = 0.0
            else:
                # Fall back to current unrealized PnL if available
                cur_pnl = self.smart_bus.get('current_pnl', 'VisualizationInterface')
                cur_pnl = float(cur_pnl) if isinstance(cur_pnl, (int, float)) else 0.0
                data['win_rate'] = 0.0
                data['pnl_total'] = cur_pnl
                data['pnl_today'] = cur_pnl
                data['sharpe_ratio'] = 0.0
            
            # System state
            consensus_data = self.smart_bus.get('consensus_data', 'VisualizationInterface') or {}
            data['consensus'] = float(consensus_data.get('score', 0.5))
            
            # Risk metrics
            risk_score = risk_data.get('risk_score')
            if not isinstance(risk_score, (int, float)):
                tra = self.smart_bus.get('time_risk_analysis', 'VisualizationInterface') or {}
                if isinstance(tra.get('risk_level'), (int, float)):
                    risk_score = tra.get('risk_level')
                elif isinstance(tra.get('risk_score'), (int, float)):
                    risk_score = tra.get('risk_score')
                else:
                    rs = self.smart_bus.get('risk_score', 'VisualizationInterface')
                    risk_score = rs if isinstance(rs, (int, float)) else 0.0
            try:
                data['risk_score'] = float(risk_score if isinstance(risk_score, (int, float)) else 0.0)
            except Exception:
                data['risk_score'] = 0.0
            
            # Calculate exposure
            total_exposure = sum(abs(pos.get('size', 0)) for pos in positions)
            data['total_exposure'] = total_exposure
            data['leverage'] = float(total_exposure) / max(float(data['balance']), 1.0)
            
            # Module performance
            module_data = self.smart_bus.get('module_performance', 'VisualizationInterface') or {}
            data['module_performance'] = self._extract_module_performance(module_data)
            
            # System status
            data['system_status'] = 'active'
            
            # Market context (harmonize across publishers)
            market_data = self.smart_bus.get('market_data', 'VisualizationInterface') or {}
            market_context = self.smart_bus.get('market_context', 'VisualizationInterface') or {}
            market_conditions = self.smart_bus.get('market_conditions', 'VisualizationInterface') or {}

            regime = market_data.get('regime')
            if not regime:
                regime = self.smart_bus.get('market_regime', 'VisualizationInterface')
            if not regime:
                regime = market_conditions.get('volatility_regime')
            data['regime'] = regime if isinstance(regime, str) else 'unknown'

            session = market_data.get('session')
            if not session:
                session = self.smart_bus.get('trading_session', 'VisualizationInterface')
            if not session:
                session = self.smart_bus.get('session_type', 'VisualizationInterface')
            if not session:
                session = market_context.get('session') or market_context.get('session_human')
            if not session:
                session = market_conditions.get('session')
            data['session'] = session if isinstance(session, str) else 'unknown'

            vol_level = market_data.get('volatility_level')
            if not vol_level:
                vol_level = self.smart_bus.get('volatility_level', 'VisualizationInterface')
            if not vol_level:
                vol_level = market_context.get('volatility_hint')
            data['volatility_level'] = vol_level if isinstance(vol_level, str) else 'medium'

            market_open = market_data.get('market_open')
            if market_open is None:
                market_open = self.smart_bus.get('market_open', 'VisualizationInterface')
            if market_open is None and isinstance(data['session'], str):
                market_open = data['session'].lower() != 'closed'
            data['market_open'] = bool(market_open) if isinstance(market_open, (bool, int)) else True
            
        except Exception as e:
            self.logger.warning(f"Trading data extraction failed: {e}")
            # Provide safe defaults
            data = self._get_safe_trading_defaults()
        
        return data

    def _get_safe_trading_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when trading data extraction fails.

        Prefer environment_config/performance_data anchors if present; else fall back to 3000.
        """
        try:
            env_cfg = self.smart_bus.get('environment_config', 'VisualizationInterface') or {}
        except Exception:
            env_cfg = {}
        try:
            perf = self.smart_bus.get('performance_data', 'VisualizationInterface') or {}
        except Exception:
            perf = {}

        base_bal = None
        if isinstance(env_cfg.get('initial_balance'), (int, float)):
            base_bal = float(env_cfg['initial_balance'])
        elif isinstance(perf.get('balance'), (int, float)):
            base_bal = float(perf['balance'])
        elif isinstance(perf.get('initial_balance'), (int, float)):
            base_bal = float(perf['initial_balance'])
        else:
            # Use default from risk_policy.yaml loaded in _initialize
            base_bal = getattr(self, '_default_balance', 100000.0)

        return {
            'balance': base_bal,
            'equity': base_bal,
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
            'system_status': 'active',
            'sharpe_ratio': 0.0,
            'regime': 'unknown',
            'session': 'unknown',
            'volatility_level': 'medium',
            'market_open': True
        }

    async def _create_comprehensive_record(self, trading_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive record with timestamps and context"""
        try:
            # Get step information
            step_data = self.smart_bus.get('step_data', 'VisualizationInterface') or {}
            
            out = {
                # Timestamps
                '_time': datetime.datetime.now().isoformat(),
                'timestamp': datetime.datetime.now().isoformat(),
                'step': step_data.get('step_idx', len(self.records)),
                'episode': step_data.get('episode_idx', 0),
                
                # Financial metrics
                'balance': trading_data['balance'],
                'equity': trading_data['equity'],
                'pnl_today': trading_data['pnl_today'],
                'pnl_total': trading_data['pnl_total'],
                'drawdown': trading_data['drawdown'],
                'max_drawdown': trading_data['max_drawdown'],
                
                # Trading activity
                'positions': trading_data['position_count'],
                'active_trades': trading_data['active_trades'],
                'completed_trades': trading_data['completed_trades'],
                'trade_count': trading_data['trade_count'],
                'win_rate': trading_data['win_rate'],
                
                # Market context
                'regime': trading_data['regime'],
                'session': trading_data['session'],
                'volatility_level': trading_data['volatility_level'],
                'market_open': trading_data['market_open'],
                
                # System state
                'consensus': trading_data['consensus'],
                'risk_score': trading_data['risk_score'],
                'alerts': self.smart_bus.get('alerts', 'VisualizationInterface') or [],
                'system_status': trading_data['system_status'],
                
                # Module performance
                'module_performance': trading_data['module_performance'],
                
                # Advanced metrics
                'sharpe_ratio': trading_data['sharpe_ratio'],
                'total_exposure': trading_data['total_exposure'],
                'leverage': trading_data['leverage']
            }
            # annotate fallback provenance if present
            if '_fallback_info' in trading_data:
                out['_fallback_info'] = trading_data.get('_fallback_info', {})
            return out
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "record_creation")
            self.logger.warning(f"Record creation failed: {error_context}")
            return self._get_minimal_record()

    def _get_minimal_record(self) -> Dict[str, Any]:
        """Get minimal record when full record creation fails (uses env/perf anchors)."""
        try:
            env_cfg = self.smart_bus.get('environment_config', 'VisualizationInterface') or {}
            perf = self.smart_bus.get('performance_data', 'VisualizationInterface') or {}
            if isinstance(env_cfg.get('initial_balance'), (int, float)):
                bal = float(env_cfg['initial_balance'])
            elif isinstance(perf.get('balance'), (int, float)):
                bal = float(perf['balance'])
            elif isinstance(perf.get('initial_balance'), (int, float)):
                bal = float(perf['initial_balance'])
            else:
                bal = 3000.0
        except Exception:
            bal = 3000.0

        return {
            '_time': datetime.datetime.now().isoformat(),
            'timestamp': datetime.datetime.now().isoformat(),
            'step': len(self.records),
            'balance': bal,
            'status': 'minimal'
        }

    def _extract_module_performance(self, module_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance data from all modules"""
        
        performance = {}
        
        try:
            # Extract performance from various modules
            for module_name, data in module_data.items():
                if isinstance(data, dict):
                    performance[module_name] = {
                        'quality_score': data.get('quality_score', 0.5),
                        'effectiveness': data.get('effectiveness_score', 0.5),
                        'alerts_count': len(data.get('recent_alerts', []))
                    }
            
        except Exception as e:
            self.logger.warning(f"Module performance extraction failed: {e}")
        
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
                'pnl': record.get('pnl_today', 0),
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
            
            # Debug output: wait until non-fallback data is available
            if self.debug and not record.get('_fallback_info'):
                self._print_summary(record)
            
        except Exception as e:
            self.logger.error(f"Record processing failed: {e}")

    def _update_performance_metrics(self, record: Dict[str, Any]) -> None:
        """Update performance metrics collections"""
        
        try:
            # Update all performance metrics
            for metric_name, metric_deque in self.performance_metrics.items():
                value = record.get(metric_name, 0.0)
                if isinstance(value, (int, float)):
                    metric_deque.append(float(value))
            
        except Exception as e:
            self.logger.warning(f"Performance metrics update failed: {e}")

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
                    self.logger.warning(format_operator_message(
                        icon="[ALERT]",
                        message=f"Critical alert: {alert.get('message', alert)}",
                        module=alert.get('module', 'unknown'),
                        step=record['step']
                    ))
            
        except Exception as e:
            self.logger.warning(f"Alert processing failed: {e}")

    def _update_regime_analytics(self, record: Dict[str, Any]) -> None:
        """Update regime-based analytics"""
        
        try:
            regime = record.get('regime', 'unknown')
            analytics = self.regime_analytics[regime]
            
            # Safely update time spent
            if isinstance(analytics['time_spent'], (int, float)):
                analytics['time_spent'] += 1
            else:
                analytics['time_spent'] = 1
            
            # Safely update performance list
            if isinstance(analytics['performance'], list):
                analytics['performance'].append(record.get('pnl_today', 0.0))
            else:
                analytics['performance'] = [record.get('pnl_today', 0.0)]
            
            # Safely update trade count
            if record.get('trade_count', 0) > 0:
                if isinstance(analytics['trade_count'], (int, float)):
                    analytics['trade_count'] += 1
                else:
                    analytics['trade_count'] = 1
            
            # Update average P&L
            if isinstance(analytics['performance'], list) and analytics['performance']:
                analytics['avg_pnl'] = float(np.mean(analytics['performance']))
            
        except Exception as e:
            self.logger.warning(f"Regime analytics update failed: {e}")

    def _update_session_analytics(self, record: Dict[str, Any]) -> None:
        """Update session-based analytics"""
        
        try:
            session = record.get('session', 'unknown')
            analytics = self.session_analytics[session]
            
            # Safely update time spent
            if isinstance(analytics['time_spent'], (int, float)):
                analytics['time_spent'] += 1
            else:
                analytics['time_spent'] = 1
            
            # Safely update performance list
            if isinstance(analytics['performance'], list):
                analytics['performance'].append(record.get('pnl_today', 0.0))
            else:
                analytics['performance'] = [record.get('pnl_today', 0.0)]
            
            # Safely update trade count
            if record.get('trade_count', 0) > 0:
                if isinstance(analytics['trade_count'], (int, float)):
                    analytics['trade_count'] += 1
                else:
                    analytics['trade_count'] = 1
            
            # Update average P&L
            if isinstance(analytics['performance'], list) and analytics['performance']:
                analytics['avg_pnl'] = float(np.mean(analytics['performance']))
            
        except Exception as e:
            self.logger.warning(f"Session analytics update failed: {e}")

    async def _update_dashboard_if_needed(self) -> None:
        """Update dashboard data if enough time has passed"""
        
        try:
            current_time = time.time()
            
            if (current_time - self.last_dashboard_update) >= self.dashboard_update_freq:
                self.dashboard_data = await self._generate_dashboard_data()
                self.last_dashboard_update = current_time
                self.viz_stats['dashboard_updates'] += 1
                
                self.logger.info(format_operator_message(
                    icon="[STATS]",
                    message="Dashboard updated",
                    records=len(self.records),
                    alerts=len(self.alert_history),
                    balance=self.dashboard_data.get('current', {}).get('balance', 0)
                ))
            
        except Exception as e:
            self.logger.warning(f"Dashboard update failed: {e}")

    async def _generate_dashboard_data(self) -> Dict[str, Any]:
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
                    'total_pnl': sum(list(self.performance_metrics['pnl'])) if self.performance_metrics['pnl'] else 0.0,
                    'max_drawdown': max(list(self.performance_metrics['drawdown'])) if self.performance_metrics['drawdown'] else 0.0,
                    'current_win_rate': self.performance_metrics['win_rate'][-1] if self.performance_metrics['win_rate'] else 0.0,
                    'avg_consensus': float(np.mean(list(self.performance_metrics['consensus']))) if self.performance_metrics['consensus'] else 0.0,
                    'recent_alerts': list(self.alert_history)[-10:],
                    'trade_count': self.performance_metrics['trades'][-1] if self.performance_metrics['trades'] else 0
                },
                'analytics': {
                    'regime_distribution': self._calculate_regime_distribution(recent_records),
                    'session_performance': self._calculate_session_performance(),
                    'module_performance': current_record.get('module_performance', {})
                },
                'metadata': {
                    'last_update': datetime.datetime.now().isoformat(),
                    'data_points': len(self.records),
                    'timespan_hours': self._calculate_timespan_hours(),
                    'update_frequency': self.dashboard_update_freq
                }
            }
            
        except Exception as e:
            self.logger.error(f"Dashboard data generation failed: {e}")
            return {}

    async def _stream_updates(self) -> None:
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
                
                await self._broadcast_to_clients(stream_data)
                self.viz_stats['stream_updates'] += 1
            
        except Exception as e:
            self.logger.warning(f"Streaming update failed: {e}")

    async def _broadcast_to_clients(self, data: Dict[str, Any]) -> None:
        """Broadcast data to all connected streaming clients"""
        
        # This would integrate with your WebSocket implementation
        # Placeholder for actual streaming logic
        if self.stream_url:
            try:
                # Import aiohttp only when needed for streaming
                try:
                    import aiohttp
                except ImportError:
                    self.logger.warning("aiohttp not available - streaming disabled")
                    return
                
                timeout = aiohttp.ClientTimeout(total=0.5)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    await session.post(self.stream_url, json=data)
            except Exception:
                pass  # Don't let streaming errors affect trading

    def _publish_infobus_data_sync(self, results: Dict[str, Any]):
        """Synchronously publish critical data to InfoBus (backup method)"""
        try:
            thesis = f"Processed {results.get('records_processed', 0)} records, dashboard ready: {results.get('dashboard_ready', False)}"

            # Always publish visualization data
            viz_data = {
                'total_records': len(self.records),
                'performance_metrics': {k: list(v)[-10:] for k, v in self.performance_metrics.items()},
                'recent_alerts': list(self.alert_history)[-5:],
                'dashboard_ready': bool(self.dashboard_data),
                'streaming_enabled': self.streaming_enabled,
                'statistics': self.viz_stats.copy(),
                'regime_analytics': dict(self.regime_analytics),
                'session_analytics': dict(self.session_analytics),
                'status': 'active'
            }

            self.smart_bus.set('visualization_data', viz_data,
                             module='VisualizationInterface', thesis=thesis)

            # Always publish dashboard data (even if empty)
            self.smart_bus.set('dashboard_data', self.dashboard_data or {},
                             module='VisualizationInterface', thesis=thesis)

            self.logger.info(f"[INFOBUS] Published viz_data and dashboard_data: {len(viz_data)} items")

        except Exception as e:
            self.logger.error(f"[INFOBUS] Sync publish failed: {e}")

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any]):
        """Update SmartInfoBus with comprehensive visualization data"""
        try:
            thesis = f"Processed {results.get('records_processed', 0)} records, dashboard ready: {results.get('dashboard_ready', False)}"

            # Update visualization data
            self.smart_bus.set('visualization_data', {
                'total_records': len(self.records),
                'performance_metrics': {k: list(v)[-10:] for k, v in self.performance_metrics.items()},
                'recent_alerts': list(self.alert_history)[-5:],
                'dashboard_ready': bool(self.dashboard_data),
                'streaming_enabled': self.streaming_enabled,
                'statistics': self.viz_stats.copy(),
                'regime_analytics': dict(self.regime_analytics),
                'session_analytics': dict(self.session_analytics)
            }, module='VisualizationInterface', thesis=thesis)

            # Update dashboard data (always publish, even if empty)
            self.smart_bus.set('dashboard_data', self.dashboard_data or {},
                             module='VisualizationInterface', thesis=thesis)

        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.warning(f"SmartInfoBus update failed: {error_context}")

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "VisualizationInterface")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="[ALERT]",
                message="VisualizationInterface disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        thesis = f"Visualization processing error: {error_context}"
        # Return minimal but complete provides payload with thesis
        return {
            'visualization_data': {
                'total_records': len(self.records),
                'performance_metrics': {k: list(v)[-10:] for k, v in self.performance_metrics.items()},
                'recent_alerts': list(self.alert_history)[-5:],
                'dashboard_ready': bool(self.dashboard_data),
                'streaming_enabled': self.streaming_enabled,
                'statistics': self.viz_stats.copy(),
                'regime_analytics': dict(self.regime_analytics),
                'session_analytics': dict(self.session_analytics),
                'status': 'error'
            },
            'performance_metrics': self.get_performance_summary(),
            'dashboard_data': self.dashboard_data or {},
            'alert_timeline': list(self.alert_history),
            'analytics_reports': {'error': str(error_context)},
            'streaming_data': {'enabled': self.streaming_enabled, 'clients': len(self.stream_clients)},
            'system_status': {'status': 'error', 'error_count': self.error_count},
            '_thesis': thesis
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'records_processed': 0,
            'status': 'disabled',
            'reason': 'circuit_breaker_triggered'
        }

    # ================== UTILITY METHODS ==================
    
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
            self.logger.warning(f"Regime distribution calculation failed: {e}")
            return {}

    def _calculate_session_performance(self) -> Dict[str, Dict[str, float]]:
        """Calculate performance by trading session"""
        
        try:
            session_perf = {}
            for session, analytics in self.session_analytics.items():
                if analytics['performance']:
                    session_perf[session] = {
                        'avg_pnl': analytics['avg_pnl'],
                        'total_pnl': sum(analytics['performance']) if isinstance(analytics['performance'], list) else 0.0,
                        'trade_count': analytics['trade_count'],
                        'time_spent': analytics['time_spent']
                    }
            return session_perf
            
        except Exception as e:
            self.logger.warning(f"Session performance calculation failed: {e}")
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

    def _print_summary(self, record: Dict[str, Any]) -> None:
        """Print formatted summary for debugging"""
        
        try:
            balance = record.get('balance', 0)
            pnl = record.get('pnl_today', 0)
            positions = record.get('positions', 0)
            drawdown = record.get('drawdown', 0)
            regime = record.get('regime', 'unknown')
            fb = record.get('_fallback_info', {})
            fb_tag = ''
            if isinstance(fb, dict) and fb:
                # compact tag when printing to indicate fallback used
                src = fb.get('balance_fallback') or fb.get('equity_fallback')
                if src:
                    fb_tag = f" [fallback:{src}]"

            # Disabled - using beautiful visualizer instead
            pass

        except Exception as e:
            self.logger.warning(f"Summary printing failed: {e}")

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self.dashboard_data or {}

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        
        if not self.performance_metrics['balance']:
            return {}
        
        # Safely get metrics, ensuring they are lists
        pnl_values = list(self.performance_metrics['pnl']) if self.performance_metrics['pnl'] else []
        drawdown_values = list(self.performance_metrics['drawdown']) if self.performance_metrics['drawdown'] else []
        win_rate_values = list(self.performance_metrics['win_rate']) if self.performance_metrics['win_rate'] else []
        trades_values = list(self.performance_metrics['trades']) if self.performance_metrics['trades'] else []
        consensus_values = list(self.performance_metrics['consensus']) if self.performance_metrics['consensus'] else []
        
        return {
            'current_balance': self.performance_metrics['balance'][-1],
            'total_pnl': sum(pnl_values) if pnl_values else 0.0,
            'max_drawdown': max(drawdown_values) if drawdown_values else 0.0,
            'win_rate': win_rate_values[-1] if win_rate_values else 0.0,
            'trade_count': trades_values[-1] if trades_values else 0,
            'avg_consensus': float(np.mean(consensus_values)) if consensus_values else 0.0,
            'recent_alerts': len(list(self.alert_history)[-10:])
        }

    def generate_performance_report(self) -> str:
        """Generate comprehensive text performance report"""
        
        try:
            data = self.get_dashboard_data()
            stats = data.get('statistics', {})
            current = data.get('current', {})
            
            report = f"""
[STATS] TRADING PERFORMANCE REPORT
═══════════════════════════════════════
Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

[MONEY] FINANCIAL SUMMARY:
• Current Balance: ${current.get('balance', 0):,.2f}
• Total P&L: ${stats.get('total_pnl', 0):+,.2f}
• Max Drawdown: {stats.get('max_drawdown', 0):.1%}
• Current Win Rate: {stats.get('current_win_rate', 0):.1%}

[CHART] TRADING ACTIVITY:
• Total Trades: {stats.get('trade_count', 0):,}
• Active Positions: {current.get('positions', 0)}
• Current Regime: {current.get('regime', 'unknown').title()}
• Average Consensus: {stats.get('avg_consensus', 0):.1%}

[ALERT] RECENT ALERTS:
"""
            
            for alert in stats.get('recent_alerts', [])[-5:]:
                alert_time = alert.get('time', '')[:19]
                alert_msg = alert.get('alert', {}).get('message', str(alert.get('alert', 'Unknown alert')))
                report += f"• [{alert_time}] {alert_msg}\n"
            
            report += f"""
[STATS] SYSTEM STATISTICS:
• Data Points Collected: {self.viz_stats['data_points_collected']:,}
• Dashboard Updates: {self.viz_stats['dashboard_updates']:,}
• Streaming Enabled: {'[OK] Yes' if self.streaming_enabled else '[FAIL] No'}
• Data Timespan: {data.get('metadata', {}).get('timespan_hours', 0):.1f} hours
• Error Count: {self.error_count}
• Status: {'[ALERT] Disabled' if self.is_disabled else '[OK] Healthy'}
            """
            
            return report
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return f"Error generating report: {e}"

    def add_stream_client(self, client_id: str) -> None:
        """Add streaming client"""
        self.stream_clients.add(client_id)
        self.logger.info(format_operator_message(
            icon="📡",
            message="Streaming client added",
            client_id=client_id
        ))

    def remove_stream_client(self, client_id: str) -> None:
        """Remove streaming client"""
        self.stream_clients.discard(client_id)
        self.logger.info(format_operator_message(
            icon="📡",
            message="Streaming client removed",
            client_id=client_id
        ))

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
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'VisualizationInterface',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'debug': self.debug,
                'max_steps': self.max_steps,
                'stream_url': self.stream_url,
                'dashboard_update_freq': self.dashboard_update_freq,
                'performance_window': self.performance_window
            },
            'data_state': {
                'records': self.records[-100:],  # Save recent only
                'decision_trace': list(self.decision_trace)[-50:],
                'alert_history': list(self.alert_history)[-50:]
            },
            'metrics_state': {
                'performance_metrics': {k: list(v) for k, v in self.performance_metrics.items()},
                'regime_analytics': dict(self.regime_analytics),
                'session_analytics': dict(self.session_analytics)
            },
            'system_state': {
                'statistics': self.viz_stats.copy(),
                'last_dashboard_update': self.last_dashboard_update,
                'streaming_enabled': self.streaming_enabled,
                'stream_clients': list(self.stream_clients),
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
            self.max_steps = int(config.get("max_steps", self.max_steps))
            self.stream_url = config.get("stream_url", self.stream_url)
            self.dashboard_update_freq = int(config.get("dashboard_update_freq", self.dashboard_update_freq))
            self.performance_window = int(config.get("performance_window", self.performance_window))
            
            # Load data state
            data_state = state.get("data_state", {})
            self.records = data_state.get("records", [])
            self.decision_trace = data_state.get("decision_trace", [])
            
            alert_history = data_state.get("alert_history", [])
            self.alert_history.clear()
            for alert in alert_history:
                self.alert_history.append(alert)
            
            # Load metrics state
            metrics_state = state.get("metrics_state", {})
            performance_metrics = metrics_state.get("performance_metrics", {})
            for metric_name, values in performance_metrics.items():
                if metric_name in self.performance_metrics:
                    self.performance_metrics[metric_name].clear()
                    for value in values:
                        self.performance_metrics[metric_name].append(value)
            
            self.regime_analytics.update(metrics_state.get("regime_analytics", {}))
            self.session_analytics.update(metrics_state.get("session_analytics", {}))
            
            # Load system state
            system_state = state.get("system_state", {})
            self.viz_stats.update(system_state.get("statistics", {}))
            self.last_dashboard_update = system_state.get("last_dashboard_update", 0)
            self.streaming_enabled = bool(system_state.get("streaming_enabled", self.streaming_enabled))
            self.stream_clients = set(system_state.get("stream_clients", []))
            self.error_count = system_state.get("error_count", 0)
            self.is_disabled = system_state.get("is_disabled", False)
            
            self.logger.info(format_operator_message(
                icon="[RELOAD]",
                message="VisualizationInterface state restored",
                records=len(self.records),
                alerts=len(self.alert_history),
                dashboard_updates=self.viz_stats.get('dashboard_updates', 0)
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for monitoring"""
        return {
            'module_name': 'VisualizationInterface',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'total_records': len(self.records),
            'dashboard_updates': self.viz_stats['dashboard_updates'],
            'streaming_enabled': self.streaming_enabled,
            'connected_clients': len(self.stream_clients),
            'last_dashboard_update': self.last_dashboard_update,
            'alerts_in_history': len(self.alert_history)
        }

    # ================== LEGACY COMPATIBILITY ==================

    def record_step(self, **kwargs) -> None:
        """Legacy recording method for backward compatibility"""
        try:
            record = dict(kwargs)
            record['_time'] = datetime.datetime.now().isoformat()
            record['timestamp'] = record['_time']
            
            # Add defaults for missing fields
            defaults = {
                'step': len(self.records),
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
            self.logger.error(f"Legacy data processing failed: {e}")
