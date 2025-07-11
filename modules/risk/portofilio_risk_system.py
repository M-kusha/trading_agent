# ─────────────────────────────────────────────────────────────
# File: modules/risk/portfolio_risk_system.py
# 🚀 PRODUCTION-READY Enhanced Portfolio Risk System
# Advanced portfolio risk management with SmartInfoBus integration and intelligent automation
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


class RiskMode(Enum):
    """Portfolio risk operational modes"""
    INITIALIZATION = "initialization"
    BOOTSTRAP = "bootstrap"
    NORMAL = "normal"
    ELEVATED = "elevated"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class PortfolioRiskConfig:
    """Configuration for Portfolio Risk System"""
    var_window: int = 20
    dd_limit: float = 0.20
    risk_mult: float = 2.0
    min_position_pct: float = 0.01
    max_position_pct: float = 0.25
    correlation_window: int = 50
    bootstrap_trades: int = 10
    var_confidence: float = 0.95
    max_portfolio_exposure: float = 1.0
    correlation_threshold: float = 0.8
    volatility_lookback: int = 30
    risk_budget_daily: float = 0.02
    
    # Performance thresholds
    max_processing_time_ms: float = 200
    circuit_breaker_threshold: int = 5
    min_risk_quality: float = 0.3
    
    # Adaptation parameters
    adaptive_learning_rate: float = 0.01
    risk_sensitivity: float = 1.0


@module(
    name="PortfolioRiskSystem",
    version="4.0.0",
    category="risk",
    provides=["portfolio_risk", "risk_metrics", "position_limits", "risk_analytics"],
    requires=["trade_data", "position_data", "market_data", "risk_signals"],
    description="Advanced portfolio risk management with comprehensive VaR analysis and dynamic position limits",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class PortfolioRiskSystem(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🚀 Advanced portfolio risk system with SmartInfoBus integration.
    Provides comprehensive portfolio-level risk management including VaR, 
    correlation analysis, and dynamic position limits.
    """

    def __init__(self, 
                 config: Optional[PortfolioRiskConfig] = None,
                 instruments: Optional[List[str]] = None,
                 **kwargs):
        
        self.config = config or PortfolioRiskConfig()
        self.instruments = instruments or ["EUR/USD", "XAU/USD"]
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize portfolio state
        self._initialize_portfolio_state()
        
        self.logger.info(
            format_operator_message(
                "💼", "PORTFOLIO_RISK_INITIALIZED",
                details=f"Instruments: {len(self.instruments)}, VaR window: {self.config.var_window}",
                result="Enhanced portfolio risk system ready",
                context="portfolio_initialization"
            )
        )

    def _initialize_advanced_systems(self):
        """Initialize advanced systems for portfolio risk"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="PortfolioRiskSystem", 
            log_path="logs/risk/portfolio_risk_system.log", 
            max_lines=5000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("PortfolioRiskSystem", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for risk operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        self._start_monitoring()

    def _initialize_portfolio_state(self):
        """Initialize portfolio risk state"""
        # Initialize mixin states
        self._initialize_risk_state()
        self._initialize_trading_state()
        self._initialize_state_management()
        
        # Current operational mode
        self.current_mode = RiskMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        
        # Enhanced state tracking
        self.returns_history: Dict[str, deque] = {
            inst: deque(maxlen=max(self.config.var_window, self.config.correlation_window))
            for inst in self.instruments
        }
        self.portfolio_returns = deque(maxlen=self.config.var_window)
        self.current_positions: Dict[str, float] = {}
        self.position_history = deque(maxlen=100)
        self.trade_count = 0
        
        # Portfolio performance metrics
        self.performance_metrics = {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.5,
            "recent_pnl": 0.0,
            "total_pnl": 0.0,
            "volatility": 0.0,
            "var_95": 0.0,
            "total_exposure": 0.0,
            "risk_quality": 0.5
        }
        
        # Risk factors
        self.risk_adjustment = 1.0
        self.min_risk_adjustment = 0.5
        self.max_risk_adjustment = 1.5
        
        # Position limits tracking
        self.position_limits: Dict[str, float] = {
            inst: self.config.max_position_pct for inst in self.instruments
        }
        
        # VaR and correlation tracking
        self.current_var = 0.0
        self.correlation_matrix: Optional[np.ndarray] = None
        self.max_correlation = 0.0
        
        # Market context awareness
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Bootstrap mode tracking
        self.bootstrap_mode = True
        
        # Risk budget tracking
        self.daily_risk_used = 0.0
        self.risk_budget_violations = 0
        
        # Performance analytics
        self.portfolio_analytics = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        
        # Risk events tracking
        self.risk_events: List[Dict[str, Any]] = []
        self.limit_violations = 0
        self.correlation_alerts = 0
        
        # Adaptive parameters
        self._adaptive_params = {
            'dynamic_limit_scaling': 1.0,
            'correlation_sensitivity': 1.0,
            'volatility_tolerance': 1.0,
            'risk_adaptation_confidence': 0.5
        }

    def _start_monitoring(self):
        """Start background monitoring for portfolio risk"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_portfolio_health()
                    self._analyze_risk_effectiveness()
                    self._adapt_risk_parameters()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Portfolio risk monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module with SmartInfoBus integration"""
        try:
            # Set initial portfolio risk status
            initial_status = {
                "current_mode": self.current_mode.value,
                "bootstrap_mode": self.bootstrap_mode,
                "var_95": self.current_var,
                "max_correlation": self.max_correlation,
                "risk_adjustment": self.risk_adjustment
            }
            
            self.smart_bus.set(
                'portfolio_risk',
                initial_status,
                module='PortfolioRiskSystem',
                thesis="Initial portfolio risk system status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Portfolio risk initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process portfolio risk assessment with enhanced analytics"""
        start_time = time.time()
        
        try:
            # Extract portfolio data from SmartInfoBus
            portfolio_data = await self._extract_portfolio_data(**inputs)
            
            if not portfolio_data:
                return await self._handle_no_data_fallback()
            
            # Update market context
            context_result = await self._update_market_context_async(portfolio_data)
            
            # Update positions and returns
            position_result = await self._update_positions_and_returns(portfolio_data)
            
            # Calculate comprehensive risk metrics
            risk_result = await self._calculate_comprehensive_risk_metrics(portfolio_data)
            
            # Update position limits dynamically
            limits_result = await self._update_dynamic_position_limits(portfolio_data)
            
            # Check risk violations
            violations_result = await self._check_portfolio_risk_violations(portfolio_data)
            
            # Update mode based on risk level
            mode_result = await self._update_operational_mode(portfolio_data)
            
            # Combine results
            result = {**context_result, **position_result, **risk_result, 
                     **limits_result, **violations_result, **mode_result}
            
            # Generate thesis
            thesis = await self._generate_portfolio_thesis(portfolio_data, result)
            
            # Update SmartInfoBus
            await self._update_portfolio_smart_bus(result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return result
            
        except Exception as e:
            return await self._handle_portfolio_error(e, start_time)

    async def _extract_portfolio_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract comprehensive portfolio data from SmartInfoBus"""
        try:
            # Get trade data from SmartInfoBus
            trade_data = self.smart_bus.get('trade_data', 'PortfolioRiskSystem') or {}
            recent_trades = trade_data.get('recent_trades', [])
            
            # Get position data
            position_data = self.smart_bus.get('position_data', 'PortfolioRiskSystem') or {}
            positions = position_data.get('positions', [])
            
            # Get market data
            market_data = self.smart_bus.get('market_data', 'PortfolioRiskSystem') or {}
            prices = market_data.get('prices', {})
            
            # Get risk signals
            risk_signals = self.smart_bus.get('risk_signals', 'PortfolioRiskSystem') or {}
            
            # Get direct inputs
            balance = inputs.get('balance', 0)
            trades = inputs.get('trades', recent_trades)
            portfolio_inputs = inputs.get('portfolio_data', {})
            
            return {
                'balance': balance,
                'trades': trades,
                'positions': positions,
                'prices': prices,
                'risk_signals': risk_signals,
                'market_data': market_data,
                'portfolio_inputs': portfolio_inputs,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract portfolio data: {e}")
            return None

    async def _update_market_context_async(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market context awareness asynchronously"""
        try:
            # Extract market context from SmartInfoBus
            market_context = self.smart_bus.get('market_context', 'PortfolioRiskSystem') or {}
            
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = market_context.get('regime', 'unknown')
            self.volatility_regime = market_context.get('volatility_level', 'medium')
            self.market_session = market_context.get('session', 'unknown')
            
            # Log regime changes
            if self.market_regime != old_regime:
                self.logger.info(
                    format_operator_message(
                        "📊", "MARKET_REGIME_CHANGE",
                        old_regime=old_regime,
                        new_regime=self.market_regime,
                        volatility=self.volatility_regime,
                        session=self.market_session,
                        context="market_context"
                    )
                )
                
                # Track regime-specific performance
                self.regime_performance[self.market_regime]['regime_changes'].append({
                    'timestamp': portfolio_data.get('timestamp', datetime.datetime.now().isoformat()),
                    'from_regime': old_regime,
                    'to_regime': self.market_regime
                })
            
            return {
                'market_context_updated': True,
                'current_regime': self.market_regime,
                'volatility_regime': self.volatility_regime,
                'market_session': self.market_session
            }
            
        except Exception as e:
            self.logger.error(f"Market context update failed: {e}")
            return {'market_context_updated': False, 'error': str(e)}

    async def _update_positions_and_returns(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update position tracking and returns history"""
        try:
            # Update positions
            positions = portfolio_data.get('positions', [])
            self.current_positions.clear()
            
            for pos in positions:
                instrument = pos.get('symbol', pos.get('instrument', 'UNKNOWN'))
                size = pos.get('size', pos.get('volume', 0))
                self.current_positions[instrument] = float(size)
            
            # Update trade count
            trades = portfolio_data.get('trades', [])
            if trades:
                self.trade_count += len(trades)
                
                # Check bootstrap mode
                if self.bootstrap_mode and self.trade_count >= self.config.bootstrap_trades:
                    self.bootstrap_mode = False
                    self.logger.info(
                        format_operator_message(
                            "📈", "BOOTSTRAP_COMPLETE",
                            trade_count=self.trade_count,
                            threshold=self.config.bootstrap_trades,
                            context="bootstrap"
                        )
                    )
            
            # Update returns history
            await self._update_returns_history_async(portfolio_data)
            
            # Record position history
            if positions:
                self.position_history.append({
                    'timestamp': portfolio_data.get('timestamp', datetime.datetime.now().isoformat()),
                    'positions': dict(self.current_positions),
                    'trade_count': self.trade_count
                })
            
            return {
                'positions_updated': True,
                'position_count': len(self.current_positions),
                'trade_count': self.trade_count,
                'bootstrap_mode': self.bootstrap_mode
            }
            
        except Exception as e:
            self.logger.error(f"Position update failed: {e}")
            return {'positions_updated': False, 'error': str(e)}

    async def _update_returns_history_async(self, portfolio_data: Dict[str, Any]):
        """Update returns history from market data"""
        try:
            prices = portfolio_data.get('prices', {})
            
            # Calculate returns for each instrument
            for instrument in self.instruments:
                if instrument in prices:
                    current_price = prices[instrument]
                    
                    # Get previous price
                    last_price_attr = f'_last_price_{instrument}'
                    if hasattr(self, last_price_attr):
                        last_price = getattr(self, last_price_attr)
                        if last_price > 0:
                            ret = (current_price - last_price) / last_price
                            self.returns_history[instrument].append(ret)
                    
                    # Store current price for next calculation
                    setattr(self, last_price_attr, current_price)
            
            # Calculate portfolio return if we have positions
            if self.current_positions:
                portfolio_return = 0.0
                total_weight = 0.0
                
                for instrument, position in self.current_positions.items():
                    if instrument in self.returns_history and len(self.returns_history[instrument]) > 0:
                        inst_return = self.returns_history[instrument][-1]
                        weight = abs(position)
                        portfolio_return += inst_return * weight
                        total_weight += weight
                
                if total_weight > 0:
                    portfolio_return /= total_weight
                    self.portfolio_returns.append(portfolio_return)
            
        except Exception as e:
            self.logger.error(f"Returns history update failed: {e}")

    async def _calculate_comprehensive_risk_metrics(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            # Calculate VaR
            var_result = await self._calculate_portfolio_var_async()
            
            # Calculate correlation matrix
            correlation_result = await self._calculate_correlation_matrix_async()
            
            # Calculate volatility metrics
            volatility_result = await self._calculate_portfolio_volatility_async()
            
            # Update performance metrics
            performance_result = await self._update_portfolio_performance_async(portfolio_data)
            
            # Update risk adjustment factor
            adjustment_result = await self._update_risk_adjustment_factor_async(portfolio_data)
            
            # Update risk budget usage
            budget_result = await self._update_risk_budget_usage_async(portfolio_data)
            
            return {
                'risk_metrics_calculated': True,
                'var_95': self.current_var,
                'max_correlation': self.max_correlation,
                'portfolio_volatility': self.performance_metrics["volatility"],
                'risk_adjustment': self.risk_adjustment,
                'risk_budget_used': self.daily_risk_used,
                **var_result, **correlation_result, **volatility_result,
                **performance_result, **adjustment_result, **budget_result
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return {'risk_metrics_calculated': False, 'error': str(e)}

    async def _calculate_portfolio_var_async(self) -> Dict[str, Any]:
        """Calculate portfolio Value at Risk asynchronously"""
        try:
            if len(self.portfolio_returns) < 10:
                self.current_var = 0.0
                return {'var_data_sufficient': False}
            
            returns = np.array(list(self.portfolio_returns))
            var_percentile = (1 - self.config.var_confidence) * 100
            self.current_var = abs(np.percentile(returns, var_percentile))
            
            # Update performance metrics
            self.performance_metrics["var_95"] = float(self.current_var)
            
            return {
                'var_calculated': True,
                'var_data_points': len(self.portfolio_returns),
                'var_confidence': self.config.var_confidence
            }
            
        except Exception as e:
            self.logger.warning(f"VaR calculation failed: {e}")
            self.current_var = 0.0
            return {'var_calculated': False, 'error': str(e)}

    async def _calculate_correlation_matrix_async(self) -> Dict[str, Any]:
        """Calculate correlation matrix for instruments asynchronously"""
        try:
            n_inst = len(self.instruments)
            self.correlation_matrix = np.eye(n_inst)
            
            # Need enough data for correlation
            min_len = min(
                len(self.returns_history[inst]) 
                for inst in self.instruments 
                if len(self.returns_history[inst]) > 0
            ) if any(len(self.returns_history[inst]) > 0 for inst in self.instruments) else 0
            
            if min_len < 10:
                self.max_correlation = 0.0
                return {'correlation_data_sufficient': False}
            
            # Build returns matrix
            returns_matrix = []
            valid_instruments = []
            
            for inst in self.instruments:
                if len(self.returns_history[inst]) >= min_len:
                    returns = list(self.returns_history[inst])[-min_len:]
                    returns_matrix.append(returns)
                    valid_instruments.append(inst)
            
            if len(returns_matrix) < 2:
                self.max_correlation = 0.0
                return {'correlation_pairs_insufficient': True}
            
            returns_matrix = np.array(returns_matrix)
            
            # Calculate correlations
            for i in range(len(valid_instruments)):
                for j in range(i+1, len(valid_instruments)):
                    try:
                        corr = np.corrcoef(returns_matrix[i], returns_matrix[j])[0, 1]
                        if np.isfinite(corr):
                            if i < n_inst and j < n_inst:
                                self.correlation_matrix[i, j] = corr
                                self.correlation_matrix[j, i] = corr
                    except Exception:
                        continue
            
            # Update max correlation
            off_diagonal = self.correlation_matrix[np.triu_indices(n_inst, k=1)]
            self.max_correlation = np.max(np.abs(off_diagonal)) if len(off_diagonal) > 0 else 0.0
            
            return {
                'correlation_calculated': True,
                'correlation_data_points': min_len,
                'valid_instruments': len(valid_instruments)
            }
            
        except Exception as e:
            self.logger.warning(f"Correlation calculation failed: {e}")
            self.max_correlation = 0.0
            return {'correlation_calculated': False, 'error': str(e)}

    async def _calculate_portfolio_volatility_async(self) -> Dict[str, Any]:
        """Calculate portfolio volatility asynchronously"""
        try:
            if len(self.portfolio_returns) < 5:
                self.performance_metrics["volatility"] = 0.0
                return {'volatility_data_sufficient': False}
            
            returns = np.array(list(self.portfolio_returns)[-self.config.volatility_lookback:])
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            self.performance_metrics["volatility"] = volatility
            
            return {
                'volatility_calculated': True,
                'volatility_data_points': len(returns),
                'annualized_volatility': volatility
            }
            
        except Exception as e:
            self.logger.warning(f"Volatility calculation failed: {e}")
            self.performance_metrics["volatility"] = 0.0
            return {'volatility_calculated': False, 'error': str(e)}

    async def _update_portfolio_performance_async(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update portfolio performance metrics asynchronously"""
        try:
            # Extract performance data
            recent_pnl = sum(trade.get('pnl', 0) for trade in portfolio_data.get('trades', []))
            balance = portfolio_data.get('balance', 0)
            
            # Update basic metrics
            self.performance_metrics["recent_pnl"] = recent_pnl
            self.performance_metrics["total_pnl"] += recent_pnl
            
            # Calculate total exposure
            total_exposure = 0.0
            for pos in portfolio_data.get('positions', []):
                size = abs(pos.get('size', 0))
                price = pos.get('current_price', pos.get('entry_price', 1.0))
                total_exposure += size * price
            
            self.performance_metrics["total_exposure"] = total_exposure / max(balance, 1.0) if balance > 0 else 0.0
            
            # Calculate Sharpe ratio if we have enough data
            if len(self.portfolio_returns) >= 20:
                returns = np.array(list(self.portfolio_returns)[-20:])
                if returns.std() > 0:
                    sharpe = np.sqrt(252) * returns.mean() / returns.std()
                    self.performance_metrics["sharpe"] = float(sharpe)
            
            # Update win rate from trade history
            if len(self.position_history) > 0:
                profitable_periods = sum(
                    1 for period in self.position_history 
                    if any(pos > 0 for pos in period.get('positions', {}).values())
                )
                self.performance_metrics["win_rate"] = profitable_periods / len(self.position_history)
            
            # Calculate risk quality score
            risk_quality = self._calculate_risk_quality()
            self.performance_metrics["risk_quality"] = risk_quality
            
            return {
                'performance_updated': True,
                'recent_pnl': recent_pnl,
                'total_exposure_pct': self.performance_metrics["total_exposure"],
                'sharpe_ratio': self.performance_metrics["sharpe"],
                'risk_quality': risk_quality
            }
            
        except Exception as e:
            self.logger.warning(f"Performance update failed: {e}")
            return {'performance_updated': False, 'error': str(e)}

    def _calculate_risk_quality(self) -> float:
        """Calculate comprehensive risk quality score"""
        try:
            quality_factors = []
            
            # VaR quality (lower is better)
            if self.current_var > 0:
                var_quality = max(0, 1.0 - (self.current_var / 0.05))  # 5% VaR threshold
                quality_factors.append(var_quality)
            
            # Correlation quality (lower correlation is better)
            corr_quality = max(0, 1.0 - (self.max_correlation / 0.8))  # 80% correlation threshold
            quality_factors.append(corr_quality)
            
            # Diversification quality
            position_count = len([p for p in self.current_positions.values() if abs(p) > 0.001])
            diversification_quality = min(1.0, position_count / 5.0)  # Optimal: 5+ positions
            quality_factors.append(diversification_quality)
            
            # Risk budget quality
            budget_quality = max(0, 1.0 - (self.daily_risk_used / self.config.risk_budget_daily))
            quality_factors.append(budget_quality)
            
            return float(np.mean(quality_factors)) if quality_factors else 0.5
            
        except Exception as e:
            self.logger.warning(f"Risk quality calculation failed: {e}")
            return 0.5

    async def _update_risk_adjustment_factor_async(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update dynamic risk adjustment factor asynchronously"""
        try:
            # Extract risk data
            balance = portfolio_data.get('balance', 0)
            drawdown = 0.0
            
            # Calculate drawdown from balance history
            if hasattr(self, '_balance_history'):
                if balance < max(self._balance_history, default=balance):
                    peak_balance = max(self._balance_history)
                    drawdown = (peak_balance - balance) / peak_balance if peak_balance > 0 else 0.0
            
            # Base adjustment from drawdown
            if drawdown <= 0.05:
                dd_factor = 1.0
            elif drawdown <= self.config.dd_limit:
                dd_factor = 1.0 - (drawdown - 0.05) / (self.config.dd_limit - 0.05) * 0.4
            else:
                dd_factor = 0.6 * np.exp(-(drawdown - self.config.dd_limit) * 8)
            
            # Volatility adjustment
            vol_factor = 1.0
            if self.performance_metrics["volatility"] > 0:
                if self.performance_metrics["volatility"] > 0.3:  # High volatility
                    vol_factor = 0.7
                elif self.performance_metrics["volatility"] > 0.2:  # Medium volatility
                    vol_factor = 0.85
            
            # Correlation adjustment
            corr_factor = 1.0
            if self.max_correlation > self.config.correlation_threshold:
                excess_corr = self.max_correlation - self.config.correlation_threshold
                corr_factor = 1.0 - excess_corr * 2.0
            
            # Regime adjustment
            regime_factor = 1.0
            if self.market_regime == 'volatile':
                regime_factor = 0.8
            elif self.volatility_regime == 'high':
                regime_factor = 0.85
            
            # Combine factors
            self.risk_adjustment = dd_factor * vol_factor * corr_factor * regime_factor
            self.risk_adjustment = np.clip(
                self.risk_adjustment, 
                self.min_risk_adjustment, 
                self.max_risk_adjustment
            )
            
            # Update balance history
            if not hasattr(self, '_balance_history'):
                self._balance_history = deque(maxlen=100)
            self._balance_history.append(balance)
            
            return {
                'risk_adjustment_updated': True,
                'drawdown': drawdown,
                'dd_factor': dd_factor,
                'vol_factor': vol_factor,
                'corr_factor': corr_factor,
                'regime_factor': regime_factor
            }
            
        except Exception as e:
            self.logger.warning(f"Risk adjustment calculation failed: {e}")
            self.risk_adjustment = 0.8  # Conservative fallback
            return {'risk_adjustment_updated': False, 'error': str(e)}

    async def _update_risk_budget_usage_async(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update daily risk budget usage asynchronously"""
        try:
            # Calculate risk used today (simplified)
            current_exposure = self.performance_metrics.get("total_exposure", 0.0)
            var_usage = self.current_var
            
            self.daily_risk_used = max(current_exposure * 0.5, var_usage)
            
            # Check for budget violations
            budget_violation = False
            if self.daily_risk_used > self.config.risk_budget_daily:
                self.risk_budget_violations += 1
                budget_violation = True
            
            return {
                'risk_budget_updated': True,
                'daily_risk_used': self.daily_risk_used,
                'risk_budget_daily': self.config.risk_budget_daily,
                'budget_violation': budget_violation,
                'total_violations': self.risk_budget_violations
            }
            
        except Exception as e:
            self.logger.warning(f"Risk budget update failed: {e}")
            return {'risk_budget_updated': False, 'error': str(e)}

    async def _update_dynamic_position_limits(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update dynamic position limits based on risk conditions"""
        try:
            base_limit = self.config.max_position_pct
            
            # Apply risk adjustment
            adjusted_limit = base_limit * self.risk_adjustment
            
            # Apply correlation penalty
            if self.max_correlation > 0.7:
                correlation_penalty = 1.0 - (self.max_correlation - 0.7) * 2
                adjusted_limit *= max(0.5, correlation_penalty)
            
            # Apply regime-specific adjustments
            if self.market_regime == 'volatile':
                adjusted_limit *= 0.8
            elif self.market_regime == 'trending':
                adjusted_limit *= 1.1
            
            # Apply volatility adjustments
            if self.volatility_regime == 'high':
                adjusted_limit *= 0.7
            elif self.volatility_regime == 'low':
                adjusted_limit *= 1.2
            
            # Bootstrap bonus
            if self.bootstrap_mode:
                adjusted_limit *= 1.3
            
            # Update limits for all instruments
            final_limit = np.clip(adjusted_limit, self.config.min_position_pct, self.config.max_position_pct)
            old_limits = self.position_limits.copy()
            
            for instrument in self.instruments:
                self.position_limits[instrument] = final_limit
            
            return {
                'position_limits_updated': True,
                'base_limit': base_limit,
                'adjusted_limit': adjusted_limit,
                'final_limit': final_limit,
                'limits_changed': old_limits != self.position_limits
            }
            
        except Exception as e:
            self.logger.warning(f"Position limits update failed: {e}")
            return {'position_limits_updated': False, 'error': str(e)}

    async def _check_portfolio_risk_violations(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for portfolio risk violations"""
        try:
            violations = []
            
            # Check total exposure
            exposure = self.performance_metrics.get("total_exposure", 0.0)
            if exposure > self.config.max_portfolio_exposure:
                violations.append(f"Portfolio exposure {exposure:.1%} > limit {self.config.max_portfolio_exposure:.1%}")
                self.limit_violations += 1
            
            # Check individual position limits
            for instrument, position in self.current_positions.items():
                limit = self.position_limits.get(instrument, self.config.max_position_pct)
                if abs(position) > limit:
                    violations.append(f"{instrument} position {abs(position):.1%} > limit {limit:.1%}")
                    self.limit_violations += 1
            
            # Check VaR limit
            if self.current_var > 0.05:  # 5% VaR limit
                violations.append(f"Portfolio VaR {self.current_var:.1%} > 5% limit")
            
            # Check correlation concentration
            if self.max_correlation > 0.9 and len(self.current_positions) > 1:
                violations.append(f"High correlation {self.max_correlation:.2f} with multiple positions")
                self.correlation_alerts += 1
            
            # Log violations
            if violations:
                self.logger.warning(
                    format_operator_message(
                        "🚨", "PORTFOLIO_RISK_VIOLATIONS",
                        violation_count=len(violations),
                        violations="; ".join(violations[:3]),
                        context="risk_violations"
                    )
                )
                
                # Record risk events
                for violation in violations:
                    self.risk_events.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'type': 'violation',
                        'description': violation,
                        'portfolio_data': portfolio_data.copy()
                    })
            
            # Trim risk events
            if len(self.risk_events) > 50:
                self.risk_events = self.risk_events[-50:]
            
            return {
                'violations_checked': True,
                'violations_found': len(violations),
                'violations': violations,
                'total_limit_violations': self.limit_violations,
                'correlation_alerts': self.correlation_alerts
            }
                
        except Exception as e:
            self.logger.warning(f"Risk violation check failed: {e}")
            return {'violations_checked': False, 'error': str(e)}

    async def _update_operational_mode(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update operational mode based on risk level"""
        try:
            old_mode = self.current_mode
            
            # Determine new mode based on risk conditions
            if self.current_var > 0.08 or self.daily_risk_used > self.config.risk_budget_daily * 1.5:
                new_mode = RiskMode.EMERGENCY
            elif self.current_var > 0.05 or self.limit_violations > 5:
                new_mode = RiskMode.CRITICAL
            elif self.current_var > 0.03 or self.max_correlation > 0.8:
                new_mode = RiskMode.ELEVATED
            elif self.bootstrap_mode:
                new_mode = RiskMode.BOOTSTRAP
            else:
                new_mode = RiskMode.NORMAL
            
            # Update mode if changed
            mode_changed = False
            if new_mode != old_mode:
                self.current_mode = new_mode
                self.mode_start_time = datetime.datetime.now()
                mode_changed = True
                
                self.logger.info(
                    format_operator_message(
                        "🔄", "RISK_MODE_CHANGE",
                        old_mode=old_mode.value,
                        new_mode=new_mode.value,
                        var=f"{self.current_var:.2%}",
                        correlation=f"{self.max_correlation:.2f}",
                        context="mode_transition"
                    )
                )
            
            return {
                'mode_updated': True,
                'current_mode': self.current_mode.value,
                'mode_changed': mode_changed,
                'old_mode': old_mode.value if mode_changed else None,
                'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds()
            }
            
        except Exception as e:
            self.logger.warning(f"Mode update failed: {e}")
            return {'mode_updated': False, 'error': str(e)}

    async def _generate_portfolio_thesis(self, portfolio_data: Dict[str, Any], 
                                        result: Dict[str, Any]) -> str:
        """Generate comprehensive portfolio thesis"""
        try:
            # Core metrics
            var = self.current_var
            correlation = self.max_correlation
            exposure = self.performance_metrics.get("total_exposure", 0.0)
            mode = self.current_mode.value
            
            thesis_parts = [
                f"Portfolio Risk: {mode.upper()} mode with {var:.2%} VaR and {exposure:.1%} exposure",
                f"Risk Quality: {self.performance_metrics['risk_quality']:.2f} quality score"
            ]
            
            # Risk assessment
            if var > 0.05:
                thesis_parts.append(f"HIGH RISK: VaR exceeds 5% threshold")
            elif correlation > 0.8:
                thesis_parts.append(f"CONCENTRATION RISK: {correlation:.1%} correlation detected")
            elif exposure > 0.8:
                thesis_parts.append(f"EXPOSURE RISK: {exposure:.1%} portfolio exposure")
            
            # Performance analysis
            sharpe = self.performance_metrics.get("sharpe", 0.0)
            if sharpe > 1.0:
                thesis_parts.append(f"Strong performance: {sharpe:.2f} Sharpe ratio")
            elif sharpe < 0:
                thesis_parts.append(f"Poor performance: {sharpe:.2f} Sharpe ratio")
            
            # Market context
            thesis_parts.append(f"Market context: {self.market_regime.upper()} regime, {self.volatility_regime.upper()} volatility")
            
            # Violations
            violations = result.get('violations_found', 0)
            if violations > 0:
                thesis_parts.append(f"VIOLATIONS: {violations} risk limit breaches detected")
            
            # Risk budget
            budget_used = (self.daily_risk_used / self.config.risk_budget_daily) * 100
            thesis_parts.append(f"Risk budget: {budget_used:.0f}% utilized")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Portfolio thesis generation failed: {str(e)} - Core risk monitoring functional"

    async def _update_portfolio_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with portfolio results"""
        try:
            # Portfolio risk
            portfolio_risk_data = {
                'current_mode': self.current_mode.value,
                'var_95': self.current_var,
                'max_correlation': self.max_correlation,
                'risk_adjustment': self.risk_adjustment,
                'bootstrap_mode': self.bootstrap_mode,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.smart_bus.set(
                'portfolio_risk',
                portfolio_risk_data,
                module='PortfolioRiskSystem',
                thesis=thesis
            )
            
            # Risk metrics
            risk_metrics_data = {
                'var_95': self.current_var,
                'correlation_matrix': self.correlation_matrix.tolist() if self.correlation_matrix is not None else None,
                'max_correlation': self.max_correlation,
                'portfolio_volatility': self.performance_metrics["volatility"],
                'risk_quality': self.performance_metrics["risk_quality"],
                'total_exposure': self.performance_metrics["total_exposure"]
            }
            
            self.smart_bus.set(
                'risk_metrics',
                risk_metrics_data,
                module='PortfolioRiskSystem',
                thesis="Comprehensive portfolio risk metrics and analysis"
            )
            
            # Position limits
            limits_data = {
                'position_limits': self.position_limits.copy(),
                'risk_adjustment': self.risk_adjustment,
                'base_limit': self.config.max_position_pct,
                'bootstrap_mode': self.bootstrap_mode
            }
            
            self.smart_bus.set(
                'position_limits',
                limits_data,
                module='PortfolioRiskSystem',
                thesis="Dynamic position limits based on current risk conditions"
            )
            
            # Risk analytics
            analytics_data = {
                'performance_metrics': self.performance_metrics.copy(),
                'risk_events': len(self.risk_events),
                'limit_violations': self.limit_violations,
                'correlation_alerts': self.correlation_alerts,
                'daily_risk_used': self.daily_risk_used,
                'risk_budget_violations': self.risk_budget_violations,
                'adaptive_params': self._adaptive_params.copy()
            }
            
            self.smart_bus.set(
                'risk_analytics',
                analytics_data,
                module='PortfolioRiskSystem',
                thesis="Portfolio risk analytics and performance tracking"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no portfolio data is available"""
        self.logger.warning("No portfolio data available - using fallback mode")
        
        return {
            'current_mode': self.current_mode.value,
            'var_95': self.current_var,
            'max_correlation': self.max_correlation,
            'risk_adjustment': self.risk_adjustment,
            'fallback_reason': 'no_portfolio_data'
        }

    async def _handle_portfolio_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle portfolio risk errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self._health_status = 'warning'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "PortfolioRiskSystem")
        explanation = self.english_explainer.explain_error(
            "PortfolioRiskSystem", str(error), "portfolio risk calculation"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "PORTFOLIO_RISK_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                circuit_breaker_state=self.circuit_breaker['state'],
                context="portfolio_error"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_error_fallback_response(f"error: {str(error)}")

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'current_mode': RiskMode.EMERGENCY.value,
            'var_95': 0.1,  # Conservative high VaR
            'max_correlation': 0.9,  # Conservative high correlation
            'risk_adjustment': self.min_risk_adjustment,  # Conservative low adjustment
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    def _update_portfolio_health(self):
        """Update portfolio health metrics"""
        try:
            # Check risk quality
            if self.performance_metrics["risk_quality"] < self.config.min_risk_quality:
                self._health_status = 'warning'
            else:
                self._health_status = 'healthy'
            
            # Check circuit breaker
            if self.circuit_breaker['state'] == 'OPEN':
                self._health_status = 'warning'
            
            # Check for excessive risk
            if self.current_var > 0.08 or self.max_correlation > 0.95:
                self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Portfolio health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_risk_effectiveness(self):
        """Analyze risk management effectiveness"""
        try:
            if len(self.position_history) >= 10:
                recent_performance = self.performance_metrics.get("risk_quality", 0.5)
                
                if recent_performance > 0.8:
                    self.logger.info(
                        format_operator_message(
                            "🎯", "HIGH_RISK_EFFECTIVENESS",
                            quality_score=f"{recent_performance:.2f}",
                            var=f"{self.current_var:.2%}",
                            context="risk_analysis"
                        )
                    )
                elif recent_performance < 0.3:
                    self.logger.warning(
                        format_operator_message(
                            "⚠️", "LOW_RISK_EFFECTIVENESS",
                            quality_score=f"{recent_performance:.2f}",
                            violations=self.limit_violations,
                            context="risk_analysis"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Risk effectiveness analysis failed: {e}")

    def _adapt_risk_parameters(self):
        """Continuous risk parameter adaptation"""
        try:
            # Adapt correlation sensitivity based on market conditions
            if self.market_regime == 'volatile':
                self._adaptive_params['correlation_sensitivity'] = min(
                    1.5, self._adaptive_params['correlation_sensitivity'] * 1.01
                )
            else:
                self._adaptive_params['correlation_sensitivity'] = max(
                    0.7, self._adaptive_params['correlation_sensitivity'] * 0.999
                )
            
            # Adapt volatility tolerance
            if self.performance_metrics["volatility"] > 0.3:
                self._adaptive_params['volatility_tolerance'] = max(
                    0.6, self._adaptive_params['volatility_tolerance'] * 0.99
                )
            elif self.performance_metrics["volatility"] < 0.1:
                self._adaptive_params['volatility_tolerance'] = min(
                    1.4, self._adaptive_params['volatility_tolerance'] * 1.005
                )
            
        except Exception as e:
            self.logger.warning(f"Risk parameter adaptation failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'PortfolioRiskSystem', 'portfolio_risk_calculation', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'PortfolioRiskSystem', 'portfolio_risk_calculation', 0, False
        )

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_position_limits(self) -> Dict[str, float]:
        """Get current position limits for each instrument"""
        return self.position_limits.copy()

    def check_risk_limits(self, proposed_positions: Dict[str, float]) -> Tuple[bool, str]:
        """Check if proposed positions violate risk limits"""
        try:
            # Calculate total exposure
            total_exposure = sum(abs(pos) for pos in proposed_positions.values())
            
            # Check total exposure limit
            if total_exposure > self.config.max_portfolio_exposure:
                return False, f"Total exposure {total_exposure:.1%} exceeds limit {self.config.max_portfolio_exposure:.1%}"
            
            # Check individual position limits
            for inst, pos in proposed_positions.items():
                limit = self.position_limits.get(inst, self.config.max_position_pct)
                if abs(pos) > limit:
                    return False, f"{inst} position {abs(pos):.1%} exceeds limit {limit:.1%}"
            
            # Check VaR limit (estimated)
            if self.current_var > 0.05:
                return False, f"Portfolio VaR {self.current_var:.1%} exceeds 5% limit"
            
            # Check correlation concentration
            if self.max_correlation > 0.9 and len(proposed_positions) > 1:
                return False, f"High correlation {self.max_correlation:.2f} with multiple positions"
            
            return True, "All risk checks passed"
            
        except Exception as e:
            self.logger.error(f"Risk limit check failed: {e}")
            return False, "Risk limit check failed"

    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics"""
        return {
            "var": float(self.current_var),
            "max_correlation": float(self.max_correlation),
            "risk_adjustment": float(self.risk_adjustment),
            "total_exposure": float(self.performance_metrics["total_exposure"]),
            "position_count": len([p for p in self.current_positions.values() if abs(p) > 0.001]),
            "bootstrap_mode": float(self.bootstrap_mode),
            "portfolio_volatility": float(self.performance_metrics["volatility"]),
            "sharpe_ratio": float(self.performance_metrics["sharpe"]),
            "max_drawdown": float(self.performance_metrics["max_dd"]),
            "risk_budget_used": float(self.daily_risk_used),
            "risk_budget_available": float(max(0, self.config.risk_budget_daily - self.daily_risk_used)),
            "risk_quality": float(self.performance_metrics["risk_quality"])
        }

    def get_observation_components(self) -> np.ndarray:
        """Get portfolio risk features for observation"""
        try:
            features = [
                float(self.current_var),
                float(self.max_correlation),
                float(self.risk_adjustment),
                float(self.bootstrap_mode),
                float(len(self.current_positions)),
                float(sum(abs(p) for p in self.current_positions.values())),
                float(self.performance_metrics["sharpe"]),
                float(self.performance_metrics["max_dd"]),
                float(self.performance_metrics["volatility"]),
                float(self.daily_risk_used / self.config.risk_budget_daily) if self.config.risk_budget_daily > 0 else 0.0,
                float(self.performance_metrics["risk_quality"]),
                float(1.0 if self.current_mode in [RiskMode.CRITICAL, RiskMode.EMERGENCY] else 0.0)
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Risk observation generation failed: {e}")
            return np.array([0.0] * 12, dtype=np.float32)

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'risk_quality': self.performance_metrics["risk_quality"],
            'var_95': self.current_var,
            'max_correlation': self.max_correlation,
            'bootstrap_mode': self.bootstrap_mode
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    def get_portfolio_risk_report(self) -> str:
        """Generate operator-friendly portfolio risk report"""
        
        # Risk status indicators
        if self.current_var > 0.05:
            var_status = "🚨 High Risk"
        elif self.current_var > 0.03:
            var_status = "⚠️ Elevated"
        else:
            var_status = "✅ Normal"
        
        # Correlation status
        if self.max_correlation > 0.8:
            corr_status = "🚨 High"
        elif self.max_correlation > 0.6:
            corr_status = "⚠️ Moderate"
        else:
            corr_status = "✅ Low"
        
        # Mode status
        mode_emoji = {
            RiskMode.INITIALIZATION: "🔄",
            RiskMode.BOOTSTRAP: "🏗️",
            RiskMode.NORMAL: "✅",
            RiskMode.ELEVATED: "⚠️",
            RiskMode.CRITICAL: "🚨",
            RiskMode.EMERGENCY: "🆘"
        }
        
        mode_status = f"{mode_emoji.get(self.current_mode, '❓')} {self.current_mode.value.upper()}"
        
        # Health status
        health_emoji = "✅" if self._health_status == 'healthy' else "⚠️"
        cb_status = "🔴 OPEN" if self.circuit_breaker['state'] == 'OPEN' else "🟢 CLOSED"
        
        return f"""
💼 ENHANCED PORTFOLIO RISK SYSTEM v4.0
═══════════════════════════════════════════════════
🎯 Risk Mode: {mode_status}
📊 VaR Status: {var_status} ({self.current_var:.2%})
🔗 Correlation: {corr_status} ({self.max_correlation:.2f})
🏗️ Bootstrap Mode: {'✅ Active' if self.bootstrap_mode else '❌ Inactive'}

🏥 SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Risk Quality: {self.performance_metrics['risk_quality']:.2f}

📊 PORTFOLIO METRICS
• Current VaR (95%): {self.current_var:.2%}
• Portfolio Volatility: {self.performance_metrics["volatility"]:.1%}
• Max Correlation: {self.max_correlation:.2f}
• Total Exposure: {self.performance_metrics["total_exposure"]:.1%}
• Sharpe Ratio: {self.performance_metrics["sharpe"]:.2f}
• Max Drawdown: {self.performance_metrics["max_dd"]:.1%}

💰 RISK BUDGET
• Daily Budget: {self.config.risk_budget_daily:.1%}
• Used Today: {self.daily_risk_used:.1%}
• Available: {max(0, self.config.risk_budget_daily - self.daily_risk_used):.1%}
• Budget Violations: {self.risk_budget_violations}

⚖️ RISK ADJUSTMENT
• Current Factor: {self.risk_adjustment:.1%}
• Base Position Limit: {self.config.max_position_pct:.1%}
• Adjusted Limit Range: {self.config.min_position_pct:.1%} - {self.config.max_position_pct:.1%}

🔧 SYSTEM PERFORMANCE
• Trade Count: {self.trade_count}
• Limit Violations: {self.limit_violations}
• Correlation Alerts: {self.correlation_alerts}
• Recent Risk Events: {len(self.risk_events)}

📈 PORTFOLIO PERFORMANCE
• Total PnL: {self.performance_metrics["total_pnl"]:.2f}
• Recent PnL: {self.performance_metrics["recent_pnl"]:.2f}
• Win Rate: {self.performance_metrics["win_rate"]:.1%}

💡 CONFIGURATION
• Instruments: {len(self.instruments)} tracked
• VaR Window: {self.config.var_window} periods
• Correlation Window: {self.config.correlation_window} periods
• DD Limit: {self.config.dd_limit:.1%}
        """

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> Dict[str, Any]:
        """Legacy step interface for backward compatibility"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(self.process(**kwargs))
            return result
        finally:
            loop.close()

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        # Reset mixin states
        # Note: Mixin reset methods will be implemented as needed
        
        # Reset returns history
        for inst in self.instruments:
            self.returns_history[inst].clear()
        self.portfolio_returns.clear()
        
        # Reset positions
        self.current_positions.clear()
        self.position_history.clear()
        self.trade_count = 0
        self.bootstrap_mode = True
        
        # Reset performance metrics
        self.performance_metrics = {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.5,
            "recent_pnl": 0.0,
            "total_pnl": 0.0,
            "volatility": 0.0,
            "var_95": 0.0,
            "total_exposure": 0.0,
            "risk_quality": 0.5
        }
        
        # Reset risk factors
        self.risk_adjustment = 1.0
        self.current_var = 0.0
        self.correlation_matrix = None
        self.max_correlation = 0.0
        
        # Reset position limits
        for inst in self.instruments:
            self.position_limits[inst] = self.config.max_position_pct
        
        # Reset market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Reset risk budget
        self.daily_risk_used = 0.0
        self.risk_budget_violations = 0
        
        # Reset analytics
        self.portfolio_analytics.clear()
        self.regime_performance.clear()
        self.risk_events.clear()
        self.limit_violations = 0
        self.correlation_alerts = 0
        
        # Reset mode
        self.current_mode = RiskMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        
        # Reset circuit breaker
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['state'] = 'CLOSED'
        self._health_status = 'healthy'
        
        # Reset adaptive parameters
        self._adaptive_params = {
            'dynamic_limit_scaling': 1.0,
            'correlation_sensitivity': 1.0,
            'volatility_tolerance': 1.0,
            'risk_adaptation_confidence': 0.5
        }
        
        self.logger.info("🔄 Enhanced Portfolio Risk System reset - all state cleared")

# End of enhanced PortfolioRiskSystem class