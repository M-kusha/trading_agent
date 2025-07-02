# ─────────────────────────────────────────────────────────────
# File: modules/risk/portfolio_risk_system.py
# Enhanced Portfolio Risk System with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import copy
from typing import Dict, List, Optional, Any, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import RiskMixin, AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class PortfolioRiskSystem(Module, RiskMixin, AnalysisMixin, StateManagementMixin):
    """
    Enhanced portfolio risk system with InfoBus integration.
    Provides comprehensive portfolio-level risk management including VaR, 
    correlation analysis, and dynamic position limits.
    """

    # Enhanced default configuration
    ENHANCED_DEFAULTS = {
        "var_window": 20,
        "dd_limit": 0.20,
        "risk_mult": 2.0,
        "min_position_pct": 0.01,
        "max_position_pct": 0.25,
        "correlation_window": 50,
        "bootstrap_trades": 10,
        "var_confidence": 0.95,
        "max_portfolio_exposure": 1.0,
        "correlation_threshold": 0.8,
        "volatility_lookback": 30,
        "risk_budget_daily": 0.02
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        instruments: Optional[List[str]] = None,
        debug: bool = False,
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
        self._initialize_risk_state()
        self._initialize_analysis_state()
        
        # Merge configuration with enhanced defaults
        self.portfolio_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if config:
            self.portfolio_config.update(config)
        
        # Core parameters
        self.var_window = int(self.portfolio_config["var_window"])
        self.dd_limit = float(self.portfolio_config["dd_limit"])
        self.risk_mult = float(self.portfolio_config["risk_mult"])
        self.min_position_pct = float(self.portfolio_config["min_position_pct"])
        self.max_position_pct = float(self.portfolio_config["max_position_pct"])
        self.correlation_window = int(self.portfolio_config["correlation_window"])
        self.bootstrap_trades = int(self.portfolio_config["bootstrap_trades"])
        self.var_confidence = float(self.portfolio_config["var_confidence"])
        self.max_portfolio_exposure = float(self.portfolio_config["max_portfolio_exposure"])
        self.correlation_threshold = float(self.portfolio_config["correlation_threshold"])
        self.volatility_lookback = int(self.portfolio_config["volatility_lookback"])
        self.risk_budget_daily = float(self.portfolio_config["risk_budget_daily"])
        
        # Instruments
        self.instruments = instruments or ["EUR/USD", "XAU/USD"]
        
        # Enhanced state tracking
        self.returns_history: Dict[str, deque] = {
            inst: deque(maxlen=max(self.var_window, self.correlation_window))
            for inst in self.instruments
        }
        self.portfolio_returns = deque(maxlen=self.var_window)
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
            "total_exposure": 0.0
        }
        
        # Risk factors
        self.risk_adjustment = 1.0
        self.min_risk_adjustment = 0.5
        self.max_risk_adjustment = 1.5
        
        # Position limits tracking
        self.position_limits: Dict[str, float] = {
            inst: self.max_position_pct for inst in self.instruments
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
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "PortfolioRiskSystem",
            "logs/risk/portfolio_risk_system.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("PortfolioRiskSystem")
        
        self.log_operator_info(
            "💼 Enhanced Portfolio Risk System initialized",
            instruments=len(self.instruments),
            var_window=self.var_window,
            dd_limit=f"{self.dd_limit:.1%}",
            max_exposure=f"{self.max_portfolio_exposure:.1%}",
            bootstrap_trades=self.bootstrap_trades
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_analysis_state()
        
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
            "total_exposure": 0.0
        }
        
        # Reset risk factors
        self.risk_adjustment = 1.0
        self.current_var = 0.0
        self.correlation_matrix = None
        self.max_correlation = 0.0
        
        # Reset position limits
        for inst in self.instruments:
            self.position_limits[inst] = self.max_position_pct
        
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
        
        self.log_operator_info("🔄 Portfolio Risk System reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using fallback mode")
            self._process_legacy_step(**kwargs)
            return
        
        # Extract comprehensive context
        context = extract_standard_context(info_bus)
        
        # Update market context awareness
        self._update_market_context(context, info_bus)
        
        # Extract portfolio data from InfoBus
        portfolio_data = self._extract_portfolio_data_from_info_bus(info_bus)
        
        # Update positions from trades
        self._update_positions_from_info_bus(info_bus)
        
        # Update returns history
        self._update_returns_history(info_bus)
        
        # Calculate comprehensive risk metrics
        self._calculate_comprehensive_risk_metrics(portfolio_data, context)
        
        # Update position limits
        self._update_dynamic_position_limits(context, portfolio_data)
        
        # Check risk violations
        self._check_portfolio_risk_violations(portfolio_data, context)
        
        # Update InfoBus with results
        self._update_info_bus(info_bus)
        
        # Record audit for significant events
        self._record_portfolio_audit(info_bus, context, portfolio_data)
        
        # Update performance metrics
        self._update_portfolio_performance_metrics()

    def _extract_portfolio_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract comprehensive portfolio data from InfoBus"""
        
        data = {}
        
        try:
            # Risk snapshot
            risk_data = info_bus.get('risk', {})
            data['balance'] = float(risk_data.get('balance', risk_data.get('equity', 0)))
            data['drawdown'] = max(0.0, float(risk_data.get('current_drawdown', 0.0)))
            data['max_drawdown'] = max(0.0, float(risk_data.get('max_drawdown', 0.0)))
            
            # Position data
            positions = InfoBusExtractor.get_positions(info_bus)
            data['positions'] = positions
            data['position_count'] = len(positions)
            
            # Calculate total exposure
            total_exposure = 0.0
            for pos in positions:
                size = abs(pos.get('size', 0))
                price = pos.get('current_price', pos.get('entry_price', 1.0))
                total_exposure += size * price
            
            data['total_exposure'] = total_exposure
            data['exposure_pct'] = total_exposure / max(data['balance'], 1.0)
            
            # Recent trades for PnL tracking
            recent_trades = info_bus.get('recent_trades', [])
            data['recent_pnl'] = sum(trade.get('pnl', 0) for trade in recent_trades)
            data['trade_count'] = len(recent_trades)
            
            # Market data
            data['prices'] = info_bus.get('prices', {})
            
            # Module risk scores
            module_data = info_bus.get('module_data', {})
            data['correlation_risk'] = module_data.get('correlated_risk_controller', {}).get('risk_score', 0.0)
            data['anomaly_risk'] = module_data.get('anomaly_detector', {}).get('anomaly_score', 0.0)
            
        except Exception as e:
            self.log_operator_warning(f"Portfolio data extraction failed: {e}")
            # Provide safe defaults
            data = {
                'balance': 10000.0,
                'drawdown': 0.0,
                'max_drawdown': 0.0,
                'positions': [],
                'position_count': 0,
                'total_exposure': 0.0,
                'exposure_pct': 0.0,
                'recent_pnl': 0.0,
                'trade_count': 0,
                'prices': {},
                'correlation_risk': 0.0,
                'anomaly_risk': 0.0
            }
        
        return data

    def _update_market_context(self, context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Update market context awareness"""
        
        try:
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = context.get('regime', 'unknown')
            self.volatility_regime = context.get('volatility_level', 'medium')
            self.market_session = context.get('session', 'unknown')
            
            # Log regime changes
            if self.market_regime != old_regime:
                self.log_operator_info(
                    f"📊 Market regime change: {old_regime} → {self.market_regime}",
                    volatility=self.volatility_regime,
                    session=self.market_session
                )
                
                # Track regime-specific performance
                self.regime_performance[self.market_regime]['regime_changes'].append({
                    'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
                    'from_regime': old_regime,
                    'to_regime': self.market_regime,
                    'step': info_bus.get('step_idx', 0)
                })
            
        except Exception as e:
            self.log_operator_warning(f"Market context update failed: {e}")

    def _update_positions_from_info_bus(self, info_bus: InfoBus) -> None:
        """Update position tracking from InfoBus"""
        
        try:
            # Get current positions
            positions = InfoBusExtractor.get_positions(info_bus)
            
            # Update current positions dictionary
            self.current_positions.clear()
            for pos in positions:
                instrument = pos.get('symbol', 'UNKNOWN')
                size = pos.get('size', 0)
                self.current_positions[instrument] = float(size)
            
            # Track recent trades to update trade count
            recent_trades = info_bus.get('recent_trades', [])
            new_trades = len(recent_trades)
            
            if new_trades > 0:
                self.trade_count += new_trades
                
                # Check bootstrap mode
                if self.bootstrap_mode and self.trade_count >= self.bootstrap_trades:
                    self.bootstrap_mode = False
                    self.log_operator_info(f"📈 Exiting bootstrap mode after {self.trade_count} trades")
                
                # Record position history
                self.position_history.append({
                    'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
                    'positions': dict(self.current_positions),
                    'trade_count': self.trade_count,
                    'step': info_bus.get('step_idx', 0)
                })
            
        except Exception as e:
            self.log_operator_warning(f"Position update failed: {e}")

    def _update_returns_history(self, info_bus: InfoBus) -> None:
        """Update returns history from market data"""
        
        try:
            prices = info_bus.get('prices', {})
            
            # Calculate returns for each instrument
            for instrument in self.instruments:
                if instrument in prices:
                    current_price = prices[instrument]
                    
                    # Get previous price
                    if hasattr(self, f'_last_price_{instrument}'):
                        last_price = getattr(self, f'_last_price_{instrument}')
                        if last_price > 0:
                            ret = (current_price - last_price) / last_price
                            self.returns_history[instrument].append(ret)
                    
                    # Store current price for next calculation
                    setattr(self, f'_last_price_{instrument}', current_price)
            
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
            self.log_operator_warning(f"Returns history update failed: {e}")

    def _calculate_comprehensive_risk_metrics(self, portfolio_data: Dict[str, Any], 
                                            context: Dict[str, Any]) -> None:
        """Calculate comprehensive portfolio risk metrics"""
        
        try:
            # Update VaR calculation
            self._calculate_portfolio_var()
            
            # Update correlation matrix
            self._calculate_correlation_matrix()
            
            # Update volatility metrics
            self._calculate_portfolio_volatility()
            
            # Update performance metrics
            self._update_portfolio_performance(portfolio_data)
            
            # Update risk adjustment factor
            self._update_risk_adjustment_factor(portfolio_data, context)
            
            # Update risk budget usage
            self._update_risk_budget_usage(portfolio_data)
            
        except Exception as e:
            self.log_operator_error(f"Risk metrics calculation failed: {e}")

    def _calculate_portfolio_var(self) -> None:
        """Calculate portfolio Value at Risk"""
        
        try:
            if len(self.portfolio_returns) < 10:
                self.current_var = 0.0
                return
            
            returns = np.array(list(self.portfolio_returns))
            var_percentile = (1 - self.var_confidence) * 100
            self.current_var = abs(np.percentile(returns, var_percentile))
            
            # Update performance metrics
            self.performance_metrics["var_95"] = self.current_var
            
        except Exception as e:
            self.log_operator_warning(f"VaR calculation failed: {e}")
            self.current_var = 0.0

    def _calculate_correlation_matrix(self) -> None:
        """Calculate correlation matrix for instruments"""
        
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
                return
            
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
                return
            
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
            
        except Exception as e:
            self.log_operator_warning(f"Correlation calculation failed: {e}")
            self.max_correlation = 0.0

    def _calculate_portfolio_volatility(self) -> None:
        """Calculate portfolio volatility"""
        
        try:
            if len(self.portfolio_returns) < 5:
                self.performance_metrics["volatility"] = 0.0
                return
            
            returns = np.array(list(self.portfolio_returns)[-self.volatility_lookback:])
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            self.performance_metrics["volatility"] = volatility
            
        except Exception as e:
            self.log_operator_warning(f"Volatility calculation failed: {e}")
            self.performance_metrics["volatility"] = 0.0

    def _update_portfolio_performance(self, portfolio_data: Dict[str, Any]) -> None:
        """Update portfolio performance metrics"""
        
        try:
            # Update basic metrics
            self.performance_metrics["recent_pnl"] = portfolio_data.get("recent_pnl", 0.0)
            self.performance_metrics["total_pnl"] += self.performance_metrics["recent_pnl"]
            self.performance_metrics["max_dd"] = portfolio_data.get("max_drawdown", 0.0)
            self.performance_metrics["total_exposure"] = portfolio_data.get("exposure_pct", 0.0)
            
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
            
        except Exception as e:
            self.log_operator_warning(f"Performance update failed: {e}")

    def _update_risk_adjustment_factor(self, portfolio_data: Dict[str, Any], 
                                     context: Dict[str, Any]) -> None:
        """Update dynamic risk adjustment factor"""
        
        try:
            # Base adjustment from drawdown
            drawdown = portfolio_data.get("drawdown", 0.0)
            if drawdown <= 0.05:
                dd_factor = 1.0
            elif drawdown <= self.dd_limit:
                dd_factor = 1.0 - (drawdown - 0.05) / (self.dd_limit - 0.05) * 0.4
            else:
                dd_factor = 0.6 * np.exp(-(drawdown - self.dd_limit) * 8)
            
            # Volatility adjustment
            vol_factor = 1.0
            if self.performance_metrics["volatility"] > 0:
                if self.performance_metrics["volatility"] > 0.3:  # High volatility
                    vol_factor = 0.7
                elif self.performance_metrics["volatility"] > 0.2:  # Medium volatility
                    vol_factor = 0.85
            
            # Correlation adjustment
            corr_factor = 1.0
            if self.max_correlation > self.correlation_threshold:
                excess_corr = self.max_correlation - self.correlation_threshold
                corr_factor = 1.0 - excess_corr * 2.0
            
            # Regime adjustment
            regime_factor = 1.0
            if context.get('regime') == 'volatile':
                regime_factor = 0.8
            elif context.get('volatility_level') == 'high':
                regime_factor = 0.85
            
            # Combine factors
            self.risk_adjustment = dd_factor * vol_factor * corr_factor * regime_factor
            self.risk_adjustment = np.clip(
                self.risk_adjustment, 
                self.min_risk_adjustment, 
                self.max_risk_adjustment
            )
            
        except Exception as e:
            self.log_operator_warning(f"Risk adjustment calculation failed: {e}")
            self.risk_adjustment = 0.8  # Conservative fallback

    def _update_risk_budget_usage(self, portfolio_data: Dict[str, Any]) -> None:
        """Update daily risk budget usage"""
        
        try:
            # Calculate risk used today (simplified)
            current_exposure = portfolio_data.get("exposure_pct", 0.0)
            var_usage = self.current_var
            
            self.daily_risk_used = max(current_exposure * 0.5, var_usage)
            
            # Check for budget violations
            if self.daily_risk_used > self.risk_budget_daily:
                self.risk_budget_violations += 1
                
        except Exception as e:
            self.log_operator_warning(f"Risk budget update failed: {e}")

    def _update_dynamic_position_limits(self, context: Dict[str, Any], 
                                       portfolio_data: Dict[str, Any]) -> None:
        """Update dynamic position limits based on risk conditions"""
        
        try:
            base_limit = self.max_position_pct
            
            # Apply risk adjustment
            adjusted_limit = base_limit * self.risk_adjustment
            
            # Apply correlation penalty
            if self.max_correlation > 0.7:
                correlation_penalty = 1.0 - (self.max_correlation - 0.7) * 2
                adjusted_limit *= max(0.5, correlation_penalty)
            
            # Apply regime-specific adjustments
            regime = context.get('regime', 'unknown')
            if regime == 'volatile':
                adjusted_limit *= 0.8
            elif regime == 'trending':
                adjusted_limit *= 1.1
            
            # Apply volatility adjustments
            vol_level = context.get('volatility_level', 'medium')
            if vol_level == 'high':
                adjusted_limit *= 0.7
            elif vol_level == 'low':
                adjusted_limit *= 1.2
            
            # Bootstrap bonus
            if self.bootstrap_mode:
                adjusted_limit *= 1.3
            
            # Update limits for all instruments
            final_limit = np.clip(adjusted_limit, self.min_position_pct, self.max_position_pct)
            for instrument in self.instruments:
                self.position_limits[instrument] = final_limit
                
        except Exception as e:
            self.log_operator_warning(f"Position limits update failed: {e}")

    def _check_portfolio_risk_violations(self, portfolio_data: Dict[str, Any], 
                                        context: Dict[str, Any]) -> None:
        """Check for portfolio risk violations"""
        
        try:
            violations = []
            
            # Check total exposure
            exposure = portfolio_data.get("exposure_pct", 0.0)
            if exposure > self.max_portfolio_exposure:
                violations.append(f"Portfolio exposure {exposure:.1%} > limit {self.max_portfolio_exposure:.1%}")
                self.limit_violations += 1
            
            # Check individual position limits
            for instrument, position in self.current_positions.items():
                limit = self.position_limits.get(instrument, self.max_position_pct)
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
            
            # Check drawdown
            drawdown = portfolio_data.get("drawdown", 0.0)
            if drawdown > self.dd_limit:
                violations.append(f"Drawdown {drawdown:.1%} > limit {self.dd_limit:.1%}")
            
            # Log violations
            if violations:
                self.log_operator_warning(
                    f"🚨 Portfolio risk violations detected ({len(violations)})",
                    violations="; ".join(violations[:3])  # Show first 3
                )
                
                # Record risk events
                for violation in violations:
                    self.risk_events.append({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'type': 'violation',
                        'description': violation,
                        'context': context.copy(),
                        'portfolio_data': portfolio_data.copy()
                    })
            
            # Trim risk events
            if len(self.risk_events) > 50:
                self.risk_events = self.risk_events[-50:]
                
        except Exception as e:
            self.log_operator_warning(f"Risk violation check failed: {e}")

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with portfolio risk results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'portfolio_risk_system', {
            'current_var': self.current_var,
            'max_correlation': self.max_correlation,
            'risk_adjustment': self.risk_adjustment,
            'position_limits': self.position_limits.copy(),
            'bootstrap_mode': self.bootstrap_mode,
            'portfolio_exposure': self.performance_metrics["total_exposure"],
            'risk_budget_used': self.daily_risk_used,
            'performance_metrics': self.performance_metrics.copy(),
            'active_violations': len([e for e in self.risk_events if e.get('type') == 'violation']),
            'correlation_alerts': self.correlation_alerts,
            'limit_violations': self.limit_violations
        })
        
        # Update risk snapshot
        InfoBusUpdater.update_risk_snapshot(info_bus, {
            'portfolio_var': self.current_var,
            'correlation_risk': self.max_correlation,
            'position_risk_scale': self.risk_adjustment,
            'portfolio_volatility': self.performance_metrics["volatility"],
            'risk_budget_used': self.daily_risk_used,
            'risk_budget_available': max(0, self.risk_budget_daily - self.daily_risk_used)
        })
        
        # Add alerts for significant risks
        if self.current_var > 0.04:  # 4% VaR threshold
            InfoBusUpdater.add_alert(
                info_bus,
                f"High portfolio VaR: {self.current_var:.1%}",
                severity="warning",
                module="PortfolioRiskSystem"
            )
        
        if self.max_correlation > 0.8:
            InfoBusUpdater.add_alert(
                info_bus,
                f"High correlation risk: {self.max_correlation:.2f}",
                severity="warning",
                module="PortfolioRiskSystem"
            )
        
        if self.risk_adjustment < 0.7:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Significant risk reduction: {self.risk_adjustment:.1%} scale",
                severity="critical",
                module="PortfolioRiskSystem"
            )

    def _record_portfolio_audit(self, info_bus: InfoBus, context: Dict[str, Any], 
                               portfolio_data: Dict[str, Any]) -> None:
        """Record comprehensive audit trail"""
        
        # Only audit significant changes or periodically
        should_audit = (
            len(self.risk_events) > 0 or
            self.current_var > 0.03 or
            self.max_correlation > 0.8 or
            info_bus.get('step_idx', 0) % 100 == 0
        )
        
        if should_audit:
            audit_data = {
                'portfolio_metrics': {
                    'var': self.current_var,
                    'correlation': self.max_correlation,
                    'risk_adjustment': self.risk_adjustment,
                    'exposure': portfolio_data.get('exposure_pct', 0),
                    'drawdown': portfolio_data.get('drawdown', 0),
                    'volatility': self.performance_metrics["volatility"]
                },
                'position_limits': self.position_limits.copy(),
                'market_context': context.copy(),
                'performance_metrics': self.performance_metrics.copy(),
                'risk_events': len(self.risk_events),
                'violations': {
                    'limit_violations': self.limit_violations,
                    'correlation_alerts': self.correlation_alerts,
                    'risk_budget_violations': self.risk_budget_violations
                },
                'bootstrap_mode': self.bootstrap_mode
            }
            
            severity = "critical" if self.current_var > 0.05 or self.max_correlation > 0.9 else "warning"
            
            self.audit_tracker.record_event(
                event_type="portfolio_risk_assessment",
                module="PortfolioRiskSystem",
                details=audit_data,
                severity=severity
            )

    def _update_portfolio_performance_metrics(self) -> None:
        """Update performance metrics"""
        
        # Update performance metrics
        self._update_performance_metric('current_var', self.current_var)
        self._update_performance_metric('max_correlation', self.max_correlation)
        self._update_performance_metric('risk_adjustment', self.risk_adjustment)
        self._update_performance_metric('portfolio_volatility', self.performance_metrics["volatility"])
        self._update_performance_metric('limit_violations', self.limit_violations)
        self._update_performance_metric('correlation_alerts', self.correlation_alerts)
        
        # Update regime performance tracking
        if self.market_regime != "unknown":
            regime_data = self.regime_performance[self.market_regime]
            regime_data['var_history'].append(self.current_var)
            regime_data['correlation_history'].append(self.max_correlation)
            regime_data['timestamps'].append(datetime.datetime.now().isoformat())

    def _process_legacy_step(self, **kwargs) -> None:
        """Process legacy step parameters for backward compatibility"""
        
        try:
            # Extract legacy data
            if 'returns' in kwargs:
                for instrument, ret in kwargs['returns'].items():
                    if instrument in self.instruments:
                        self.returns_history[instrument].append(float(ret))
            
            # Update positions if provided
            if 'proposed_positions' in kwargs:
                self.current_positions.update(kwargs['proposed_positions'])
            
            # Update trade count
            if 'trade_count' in kwargs:
                self.trade_count = kwargs['trade_count']
                if self.bootstrap_mode and self.trade_count >= self.bootstrap_trades:
                    self.bootstrap_mode = False
            
            # Recalculate metrics
            if not self.bootstrap_mode:
                self._calculate_portfolio_var()
                self._calculate_correlation_matrix()
            
        except Exception as e:
            self.log_operator_error(f"Legacy step processing failed: {e}")

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
            if total_exposure > self.max_portfolio_exposure:
                return False, f"Total exposure {total_exposure:.1%} exceeds limit {self.max_portfolio_exposure:.1%}"
            
            # Check individual position limits
            for inst, pos in proposed_positions.items():
                limit = self.position_limits.get(inst, self.max_position_pct)
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
            self.log_operator_error(f"Risk limit check failed: {e}")
            return False, "Risk limit check failed"

    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics"""
        
        return {
            "var": float(self.current_var),
            "max_correlation": float(self.max_correlation),
            "risk_adjustment": float(self.risk_adjustment),
            "total_exposure": self.performance_metrics["total_exposure"],
            "position_count": len([p for p in self.current_positions.values() if abs(p) > 0.001]),
            "bootstrap_mode": float(self.bootstrap_mode),
            "portfolio_volatility": self.performance_metrics["volatility"],
            "sharpe_ratio": self.performance_metrics["sharpe"],
            "max_drawdown": self.performance_metrics["max_dd"],
            "risk_budget_used": self.daily_risk_used,
            "risk_budget_available": max(0, self.risk_budget_daily - self.daily_risk_used)
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
                float(self.daily_risk_used / self.risk_budget_daily) if self.risk_budget_daily > 0 else 0.0,
            ]
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Risk observation generation failed: {e}")
            return np.array([0.0] * 10, dtype=np.float32)

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
        
        # Risk adjustment status
        if self.risk_adjustment < 0.6:
            risk_adj_status = "🚨 Severe Reduction"
        elif self.risk_adjustment < 0.8:
            risk_adj_status = "⚠️ Moderate Reduction"
        else:
            risk_adj_status = "✅ Normal"
        
        # Position limits summary
        limit_lines = []
        for inst, limit in list(self.position_limits.items())[:5]:  # Show first 5
            current_pos = abs(self.current_positions.get(inst, 0))
            utilization = current_pos / limit if limit > 0 else 0
            
            if utilization > 0.8:
                emoji = "🔴"
            elif utilization > 0.6:
                emoji = "🟡"
            else:
                emoji = "🟢"
            
            limit_lines.append(f"  {emoji} {inst}: {current_pos:.1%}/{limit:.1%} ({utilization:.0%} used)")
        
        # Recent risk events
        event_lines = []
        for event in self.risk_events[-3:]:  # Show last 3 events
            timestamp = event['timestamp'][:19]
            event_type = event.get('type', 'unknown')
            description = event.get('description', 'Unknown event')[:50]
            event_lines.append(f"  ⚠️ {timestamp}: {description}")
        
        return f"""
💼 PORTFOLIO RISK SYSTEM
═══════════════════════════════════════
🎯 VaR Status: {var_status} ({self.current_var:.2%})
🔗 Correlation: {corr_status} ({self.max_correlation:.2f})
⚖️ Risk Adjustment: {risk_adj_status} ({self.risk_adjustment:.1%})
🏗️ Bootstrap Mode: {'✅ Active' if self.bootstrap_mode else '❌ Inactive'}

📊 PORTFOLIO METRICS
• Current VaR (95%): {self.current_var:.2%}
• Portfolio Volatility: {self.performance_metrics["volatility"]:.1%}
• Max Correlation: {self.max_correlation:.2f}
• Total Exposure: {self.performance_metrics["total_exposure"]:.1%}
• Sharpe Ratio: {self.performance_metrics["sharpe"]:.2f}
• Max Drawdown: {self.performance_metrics["max_dd"]:.1%}

💰 RISK BUDGET
• Daily Budget: {self.risk_budget_daily:.1%}
• Used Today: {self.daily_risk_used:.1%}
• Available: {max(0, self.risk_budget_daily - self.daily_risk_used):.1%}
• Budget Violations: {self.risk_budget_violations}

📋 POSITION LIMITS
{chr(10).join(limit_lines) if limit_lines else "  📭 No active positions"}

🔧 SYSTEM PERFORMANCE
• Trade Count: {self.trade_count}
• Limit Violations: {self.limit_violations}
• Correlation Alerts: {self.correlation_alerts}
• Recent Risk Events: {len(self.risk_events)}

📈 PORTFOLIO PERFORMANCE
• Total PnL: {self.performance_metrics["total_pnl"]:.2f}
• Recent PnL: {self.performance_metrics["recent_pnl"]:.2f}
• Win Rate: {self.performance_metrics["win_rate"]:.1%}

📜 RECENT RISK EVENTS
{chr(10).join(event_lines) if event_lines else "  📭 No recent risk events"}

💡 CONFIGURATION
• Instruments: {len(self.instruments)} tracked
• VaR Window: {self.var_window} periods
• Correlation Window: {self.correlation_window} periods
• Max Position Size: {self.max_position_pct:.1%}
• DD Limit: {self.dd_limit:.1%}
        """

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        self._process_legacy_step(**kwargs)

    def calculate_var(self, returns: Optional[Dict[str, np.ndarray]] = None) -> float:
        """Legacy VaR calculation interface"""
        if returns:
            # Use provided returns
            all_returns = []
            for ret_array in returns.values():
                all_returns.extend(ret_array)
            
            if len(all_returns) >= 10:
                var_percentile = (1 - self.var_confidence) * 100
                return abs(np.percentile(all_returns, var_percentile))
        
        return self.current_var

    def calculate_correlations(self) -> np.ndarray:
        """Legacy correlation calculation interface"""
        if self.correlation_matrix is not None:
            return self.correlation_matrix
        
        # Return identity matrix as fallback
        return np.eye(len(self.instruments))