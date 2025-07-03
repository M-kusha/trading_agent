# ─────────────────────────────────────────────────────────────
# File: modules/risk/dynamic_risk_controller.py
# Enhanced with InfoBus integration & intelligent risk scaling
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import copy
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import RiskMixin, AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import AuditTracker, format_operator_message


class DynamicRiskController(Module, RiskMixin, AnalysisMixin, StateManagementMixin):
    """
    Enhanced dynamic risk controller with InfoBus integration.
    Provides intelligent risk scaling based on comprehensive market analysis.
    """

    # Enhanced default configuration
    ENHANCED_DEFAULTS = {
        "base_risk_scale": 1.0,
        "min_risk_scale": 0.1,
        "max_risk_scale": 1.5,
        "vol_history_len": 30,
        "dd_threshold": 0.15,
        "vol_ratio_threshold": 2.0,
        "recovery_speed": 0.15,
        "risk_decay": 0.95,
        "adaptive_scaling": True,
        "regime_sensitivity": 1.0,
        "correlation_sensitivity": 0.8
    }

    def __init__(
        self,
        config: Optional[Dict[str, float]] = None,
        action_dim: int = 1,
        debug: bool = False,
        adaptive_scaling: bool = True,
        regime_aware: bool = True,
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
        self.risk_config = copy.deepcopy(self.ENHANCED_DEFAULTS)
        if config:
            if isinstance(config, ModuleConfig):
                self.risk_config.update(vars(config))
            else:
                self.risk_config.update(config)
        
        # Core parameters
        self.base_risk_scale = float(self.risk_config["base_risk_scale"])
        self.min_risk_scale = float(self.risk_config["min_risk_scale"])
        self.max_risk_scale = float(self.risk_config["max_risk_scale"])
        self.vol_history_len = int(self.risk_config["vol_history_len"])
        self.dd_threshold = float(self.risk_config["dd_threshold"])
        self.vol_ratio_threshold = float(self.risk_config["vol_ratio_threshold"])
        self.recovery_speed = float(self.risk_config["recovery_speed"])
        self.risk_decay = float(self.risk_config["risk_decay"])
        
        # Enhanced features
        self.adaptive_scaling = adaptive_scaling
        self.regime_aware = regime_aware
        self.regime_sensitivity = float(self.risk_config["regime_sensitivity"])
        self.correlation_sensitivity = float(self.risk_config["correlation_sensitivity"])
        
        self.action_dim = int(action_dim)
        
        # Enhanced state tracking
        self.current_risk_scale = self.base_risk_scale
        self.risk_factors: Dict[str, float] = {
            "drawdown": 1.0,
            "volatility": 1.0,
            "correlation": 1.0,
            "losing_streak": 1.0,
            "market_stress": 1.0,
            "liquidity": 1.0,
            "news_sentiment": 1.0
        }
        
        # Enhanced history tracking
        self.vol_history = deque(maxlen=self.vol_history_len)
        self.dd_history = deque(maxlen=50)
        self.risk_scale_history = deque(maxlen=100)
        self.consecutive_losses = 0
        self.last_pnl = 0.0
        
        # Market context tracking
        self.market_regime = "normal"
        self.market_regime_history = deque(maxlen=20)
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Enhanced risk event tracking
        self.risk_events: List[Dict[str, Any]] = []
        self.risk_adjustments_made = 0
        self.emergency_interventions = 0
        
        # Performance analytics
        self.risk_analytics = defaultdict(list)
        self.regime_performance = defaultdict(lambda: defaultdict(list))
        self._last_significant_change = 0
        
        # External integrations
        self.external_risk_scale = 1.0
        self.external_signals = {}
        
        # Audit system
        self.audit_manager = AuditTracker("DynamicRiskController")
        
        self.log_operator_info(
            "⚙️ Enhanced Dynamic Risk Controller initialized",
            base_scale=f"{self.base_risk_scale:.2f}",
            dd_threshold=f"{self.dd_threshold:.1%}",
            vol_threshold=f"{self.vol_ratio_threshold:.1f}x",
            adaptive_scaling=adaptive_scaling,
            regime_aware=regime_aware
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_analysis_state()
        
        # Reset core state
        self.current_risk_scale = self.base_risk_scale
        self.risk_factors = {
            "drawdown": 1.0,
            "volatility": 1.0,
            "correlation": 1.0,
            "losing_streak": 1.0,
            "market_stress": 1.0,
            "liquidity": 1.0,
            "news_sentiment": 1.0
        }
        
        # Reset history
        self.vol_history.clear()
        self.dd_history.clear()
        self.risk_scale_history.clear()
        self.market_regime_history.clear()
        
        # Reset tracking
        self.consecutive_losses = 0
        self.last_pnl = 0.0
        self.risk_adjustments_made = 0
        self.emergency_interventions = 0
        
        # Reset market context
        self.market_regime = "normal"
        self.volatility_regime = "medium"
        self.market_session = "unknown"
        
        # Reset analytics
        self.risk_analytics.clear()
        self.regime_performance.clear()
        self.risk_events.clear()
        
        # Reset external integrations
        self.external_risk_scale = 1.0
        self.external_signals.clear()
        
        self.log_operator_info("🔄 Dynamic Risk Controller reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - risk controller using fallback")
            # Use kwargs for backward compatibility
            self._process_legacy_step(**kwargs)
            return
        
        # Extract comprehensive context
        context = extract_standard_context(info_bus)
        
        # Extract risk metrics from InfoBus
        risk_stats = self._extract_risk_stats_from_info_bus(info_bus)
        
        # Update market context awareness
        self._update_market_context(context, info_bus)
        
        # Perform comprehensive risk adjustment
        self._adjust_risk_comprehensive(risk_stats, context, info_bus)
        
        # Update external integrations
        self._update_external_integrations(info_bus)
        
        # Calculate final risk scale
        self._calculate_final_risk_scale(context)
        
        # Update InfoBus with results
        self._update_info_bus(info_bus)
        
        # Record audit for significant changes
        self._record_risk_audit(info_bus, context, risk_stats)
        
        # Update performance metrics
        self._update_risk_performance_metrics()

    def _extract_risk_stats_from_info_bus(self, info_bus: InfoBus) -> Dict[str, float]:
        """Extract comprehensive risk statistics from InfoBus"""
        
        stats = {}
        
        try:
            # Extract from risk snapshot
            risk_data = info_bus.get('risk', {})
            
            # Drawdown
            stats["drawdown"] = max(0.0, float(risk_data.get('current_drawdown', 0.0)))
            
            # Balance for PnL calculation
            current_balance = risk_data.get('balance', risk_data.get('equity', 0))
            if current_balance is not None:
                if hasattr(self, '_last_balance') and self._last_balance is not None:
                    stats["pnl"] = float(current_balance) - self._last_balance
                else:
                    stats["pnl"] = 0.0
                self._last_balance = float(current_balance)
            else:
                stats["pnl"] = 0.0
            
            # Volatility from market data or positions
            stats["volatility"] = self._calculate_current_volatility(info_bus)
            
            # Position metrics
            positions = InfoBusExtractor.get_positions(info_bus)
            stats["position_count"] = len(positions)
            stats["total_exposure"] = sum(
                abs(pos.get('size', 0)) * pos.get('current_price', 1.0) 
                for pos in positions
            )
            
            # Risk scores from other modules
            module_data = info_bus.get('module_data', {})
            
            # Correlation risk
            correlation_data = module_data.get('correlated_risk_controller', {})
            stats["correlation_risk"] = correlation_data.get('risk_score', 0.0)
            
            # Anomaly risk
            anomaly_data = module_data.get('anomaly_detector', {})
            stats["anomaly_risk"] = anomaly_data.get('anomaly_score', 0.0)
            
            # Compliance risk
            compliance_data = module_data.get('compliance', {})
            stats["compliance_risk"] = 1.0 - compliance_data.get('approval_rate', 1.0)
            
        except Exception as e:
            self.log_operator_warning(f"Risk stats extraction failed: {e}")
            # Provide safe defaults
            stats = {
                "drawdown": 0.0,
                "volatility": 0.01,
                "pnl": 0.0,
                "position_count": 0,
                "total_exposure": 0.0,
                "correlation_risk": 0.0,
                "anomaly_risk": 0.0,
                "compliance_risk": 0.0
            }
        
        return stats

    def _calculate_current_volatility(self, info_bus: InfoBus) -> float:
        """Calculate current market volatility from various sources"""
        
        try:
            # Method 1: From market context
            market_context = info_bus.get('market_context', {})
            if 'volatility' in market_context:
                return float(market_context['volatility'])
            
            # Method 2: From price movements
            prices = info_bus.get('prices', {})
            if prices and len(self.vol_history) > 0:
                # Calculate volatility from price changes
                total_vol = 0.0
                count = 0
                
                for instrument, current_price in prices.items():
                    if hasattr(self, f'_last_price_{instrument}'):
                        last_price = getattr(self, f'_last_price_{instrument}')
                        if last_price > 0:
                            return_pct = abs((current_price - last_price) / last_price)
                            total_vol += return_pct
                            count += 1
                    
                    setattr(self, f'_last_price_{instrument}', current_price)
                
                if count > 0:
                    return total_vol / count
            
            # Method 3: From position volatility
            positions = InfoBusExtractor.get_positions(info_bus)
            if positions:
                pnl_volatility = np.std([pos.get('unrealised_pnl', 0) for pos in positions])
                return max(0.001, pnl_volatility / 1000.0)  # Normalize
            
            # Fallback
            return 0.01
            
        except Exception as e:
            self.log_operator_warning(f"Volatility calculation failed: {e}")
            return 0.01

    def _update_market_context(self, context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Update market context awareness"""
        
        try:
            # Update regime tracking
            old_regime = self.market_regime
            self.market_regime = context.get('regime', 'unknown')
            self.volatility_regime = context.get('volatility_level', 'medium')
            self.market_session = context.get('session', 'unknown')
            
            # Track regime changes
            if self.market_regime != old_regime:
                self.market_regime_history.append({
                    'regime': self.market_regime,
                    'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
                    'step': info_bus.get('step_idx', 0)
                })
                
                self.log_operator_info(
                    f"📊 Market regime change: {old_regime} → {self.market_regime}",
                    volatility=self.volatility_regime,
                    session=self.market_session
                )
            
            # Update regime-specific risk factors
            if self.regime_aware:
                self._update_regime_risk_factors(context)
                
        except Exception as e:
            self.log_operator_warning(f"Market context update failed: {e}")

    def _update_regime_risk_factors(self, context: Dict[str, Any]) -> None:
        """Update risk factors based on market regime"""
        
        try:
            regime = context.get('regime', 'unknown')
            volatility_level = context.get('volatility_level', 'medium')
            
            # Base regime adjustments
            regime_adjustments = {
                "trending": {"market_stress": 0.9, "volatility": 1.1},
                "volatile": {"market_stress": 0.7, "volatility": 0.8},
                "ranging": {"market_stress": 1.1, "volatility": 1.2},
                "unknown": {"market_stress": 1.0, "volatility": 1.0}
            }
            
            # Volatility level adjustments
            vol_adjustments = {
                "low": {"volatility": 1.2, "market_stress": 1.1},
                "medium": {"volatility": 1.0, "market_stress": 1.0},
                "high": {"volatility": 0.8, "market_stress": 0.8},
                "extreme": {"volatility": 0.6, "market_stress": 0.6}
            }
            
            # Apply regime adjustments
            if regime in regime_adjustments:
                for factor, multiplier in regime_adjustments[regime].items():
                    if factor in self.risk_factors:
                        self.risk_factors[factor] *= multiplier * self.regime_sensitivity
            
            # Apply volatility adjustments
            if volatility_level in vol_adjustments:
                for factor, multiplier in vol_adjustments[volatility_level].items():
                    if factor in self.risk_factors:
                        self.risk_factors[factor] *= multiplier
                        
        except Exception as e:
            self.log_operator_warning(f"Regime risk factor update failed: {e}")

    def _adjust_risk_comprehensive(self, stats: Dict[str, float], 
                                  context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Comprehensive risk adjustment based on all available factors"""
        
        try:
            old_scale = self.current_risk_scale
            
            # Update individual risk factors
            self._update_drawdown_factor(stats.get("drawdown", 0.0))
            self._update_volatility_factor(stats.get("volatility", 0.01))
            self._update_correlation_factor(stats.get("correlation_risk", 0.0))
            self._update_losing_streak_factor(stats.get("pnl", 0.0))
            self._update_liquidity_factor(info_bus)
            self._update_news_sentiment_factor(info_bus)
            
            # Calculate preliminary risk scale
            preliminary_scale = self._calculate_preliminary_risk_scale()
            
            # Apply adaptive scaling if enabled
            if self.adaptive_scaling:
                preliminary_scale = self._apply_adaptive_scaling(preliminary_scale, context, stats)
            
            # Apply emergency interventions if needed
            emergency_scale = self._apply_emergency_interventions(preliminary_scale, stats, context)
            
            # Smooth transitions to avoid sudden changes
            self.current_risk_scale = self._smooth_risk_transitions(
                old_scale, emergency_scale, context
            )
            
            # Track significant changes
            if abs(self.current_risk_scale - old_scale) > 0.1:
                self.risk_adjustments_made += 1
                self._record_risk_adjustment_event(old_scale, self.current_risk_scale, stats, context)
            
            # Update history
            self.risk_scale_history.append(self.current_risk_scale)
            
        except Exception as e:
            self.log_operator_error(f"Comprehensive risk adjustment failed: {e}")
            # Conservative fallback
            self.current_risk_scale = max(self.min_risk_scale, self.current_risk_scale * 0.9)

    def _update_drawdown_factor(self, drawdown: float) -> None:
        """Update drawdown risk factor with enhanced logic"""
        
        if drawdown <= 0.05:  # <5% drawdown
            self.risk_factors["drawdown"] = 1.0
        elif drawdown <= self.dd_threshold:  # Normal range
            reduction = (drawdown - 0.05) / (self.dd_threshold - 0.05) * 0.4
            self.risk_factors["drawdown"] = 1.0 - reduction
        else:  # Excessive drawdown
            excess = drawdown - self.dd_threshold
            self.risk_factors["drawdown"] = 0.6 * np.exp(-excess * 8)  # Exponential reduction

    def _update_volatility_factor(self, volatility: float) -> None:
        """Update volatility risk factor with enhanced logic"""
        
        # Update volatility history
        self.vol_history.append(volatility)
        
        if len(self.vol_history) >= 5:
            avg_vol = np.mean(list(self.vol_history)[-10:])  # Recent average
            vol_ratio = volatility / (avg_vol + 1e-8)
            
            if vol_ratio <= 1.2:  # Normal volatility
                self.risk_factors["volatility"] = 1.0
            elif vol_ratio <= self.vol_ratio_threshold:  # Elevated
                reduction = (vol_ratio - 1.2) / (self.vol_ratio_threshold - 1.2) * 0.3
                self.risk_factors["volatility"] = 1.0 - reduction
            else:  # Extreme volatility
                excess = vol_ratio - self.vol_ratio_threshold
                self.risk_factors["volatility"] = 0.7 * np.exp(-excess * 3)

    def _update_correlation_factor(self, correlation_risk: float) -> None:
        """Update correlation risk factor"""
        
        if correlation_risk <= 0.3:
            self.risk_factors["correlation"] = 1.0
        elif correlation_risk <= 0.6:
            reduction = (correlation_risk - 0.3) / 0.3 * 0.2
            self.risk_factors["correlation"] = 1.0 - reduction
        else:
            excess = correlation_risk - 0.6
            self.risk_factors["correlation"] = 0.8 * (1.0 - excess * self.correlation_sensitivity)

    def _update_losing_streak_factor(self, pnl: float) -> None:
        """Update losing streak risk factor"""
        
        # Track consecutive losses
        if pnl < 0 and self.last_pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = max(0, self.consecutive_losses - 1)
        
        self.last_pnl = pnl
        
        # Calculate factor
        if self.consecutive_losses <= 2:
            self.risk_factors["losing_streak"] = 1.0
        elif self.consecutive_losses <= 5:
            reduction = (self.consecutive_losses - 2) * 0.15
            self.risk_factors["losing_streak"] = 1.0 - reduction
        else:
            self.risk_factors["losing_streak"] = 0.4

    def _update_liquidity_factor(self, info_bus: InfoBus) -> None:
        """Update liquidity risk factor"""
        
        try:
            market_status = info_bus.get('market_status', {})
            liquidity_score = market_status.get('liquidity_score', 1.0)
            
            if liquidity_score >= 0.8:
                self.risk_factors["liquidity"] = 1.0
            elif liquidity_score >= 0.5:
                self.risk_factors["liquidity"] = 0.8 + liquidity_score * 0.2
            else:
                self.risk_factors["liquidity"] = 0.5 + liquidity_score * 0.3
                
        except Exception:
            self.risk_factors["liquidity"] = 1.0  # Default to normal

    def _update_news_sentiment_factor(self, info_bus: InfoBus) -> None:
        """Update news sentiment risk factor"""
        
        try:
            market_context = info_bus.get('market_context', {})
            news_sentiment = market_context.get('news_sentiment', 0.0)
            
            if news_sentiment >= -0.2:  # Positive or neutral sentiment
                self.risk_factors["news_sentiment"] = 1.0
            elif news_sentiment >= -0.5:  # Moderately negative
                self.risk_factors["news_sentiment"] = 0.9
            else:  # Very negative sentiment
                self.risk_factors["news_sentiment"] = 0.7
                
        except Exception:
            self.risk_factors["news_sentiment"] = 1.0  # Default to neutral

    def _calculate_preliminary_risk_scale(self) -> float:
        """Calculate preliminary risk scale from all factors"""
        
        scale = self.base_risk_scale
        
        # Apply all risk factors
        for factor_name, factor_value in self.risk_factors.items():
            scale *= factor_value
        
        # Apply external risk scale
        scale *= self.external_risk_scale
        
        # Apply bounds
        return float(np.clip(scale, self.min_risk_scale, self.max_risk_scale))

    def _apply_adaptive_scaling(self, preliminary_scale: float, 
                               context: Dict[str, Any], stats: Dict[str, float]) -> float:
        """Apply adaptive scaling based on recent performance"""
        
        try:
            if len(self.risk_scale_history) < 10:
                return preliminary_scale
            
            # Analyze recent risk scale effectiveness
            recent_scales = list(self.risk_scale_history)[-10:]
            recent_performance = self._calculate_recent_performance(stats)
            
            # If recent performance is poor despite conservative scaling, be more aggressive
            if np.mean(recent_scales) < 0.7 and recent_performance < -0.1:
                adaptation_factor = 1.2  # Increase risk slightly
            # If recent performance is good with conservative scaling, maintain conservatism
            elif np.mean(recent_scales) < 0.7 and recent_performance > 0.1:
                adaptation_factor = 0.9  # Be more conservative
            # If recent performance is poor with aggressive scaling, be more conservative
            elif np.mean(recent_scales) > 0.8 and recent_performance < -0.1:
                adaptation_factor = 0.8  # Reduce risk significantly
            else:
                adaptation_factor = 1.0  # No adjustment
            
            adapted_scale = preliminary_scale * adaptation_factor
            return float(np.clip(adapted_scale, self.min_risk_scale, self.max_risk_scale))
            
        except Exception as e:
            self.log_operator_warning(f"Adaptive scaling failed: {e}")
            return preliminary_scale

    def _calculate_recent_performance(self, stats: Dict[str, float]) -> float:
        """Calculate recent performance score"""
        
        try:
            # Simple performance based on drawdown and PnL trends
            drawdown_score = max(0, 1.0 - stats.get("drawdown", 0) * 5)  # Penalize drawdown
            pnl_score = np.tanh(stats.get("pnl", 0) / 100.0)  # Normalize PnL
            
            return (drawdown_score + pnl_score) / 2.0
            
        except Exception:
            return 0.0

    def _apply_emergency_interventions(self, scale: float, stats: Dict[str, float], 
                                      context: Dict[str, Any]) -> float:
        """Apply emergency interventions for extreme risk situations"""
        
        try:
            emergency_scale = scale
            intervention_applied = False
            
            # Emergency intervention for extreme drawdown
            if stats.get("drawdown", 0) > 0.2:  # >20% drawdown
                emergency_scale = min(emergency_scale, 0.3)
                intervention_applied = True
                
            # Emergency intervention for extreme volatility
            if (len(self.vol_history) > 5 and 
                stats.get("volatility", 0) > np.mean(self.vol_history) * 3):
                emergency_scale = min(emergency_scale, 0.4)
                intervention_applied = True
                
            # Emergency intervention for multiple risk factors
            active_risk_factors = sum(1 for factor in self.risk_factors.values() if factor < 0.8)
            if active_risk_factors >= 4:
                emergency_scale = min(emergency_scale, 0.5)
                intervention_applied = True
                
            # Emergency intervention for extreme correlation
            if stats.get("correlation_risk", 0) > 0.8:
                emergency_scale = min(emergency_scale, 0.6)
                intervention_applied = True
            
            if intervention_applied:
                self.emergency_interventions += 1
                self.log_operator_warning(
                    f"🚨 Emergency risk intervention applied",
                    old_scale=f"{scale:.2f}",
                    new_scale=f"{emergency_scale:.2f}",
                    active_risks=active_risk_factors,
                    drawdown=f"{stats.get('drawdown', 0):.1%}"
                )
            
            return emergency_scale
            
        except Exception as e:
            self.log_operator_warning(f"Emergency intervention failed: {e}")
            return scale

    def _smooth_risk_transitions(self, old_scale: float, new_scale: float, 
                                context: Dict[str, Any]) -> float:
        """Smooth risk scale transitions to avoid abrupt changes"""
        
        try:
            # Calculate maximum allowed change per step
            max_change = 0.15  # 15% max change per step
            
            # Adjust max change based on market conditions
            if context.get('volatility_level') == 'extreme':
                max_change = 0.25  # Allow faster changes in extreme conditions
            elif context.get('regime') == 'ranging':
                max_change = 0.1   # Slower changes in ranging markets
            
            # Apply smoothing
            change = new_scale - old_scale
            
            if abs(change) <= max_change:
                return new_scale
            else:
                # Limit the change
                limited_change = np.sign(change) * max_change
                smoothed_scale = old_scale + limited_change
                
                return float(np.clip(smoothed_scale, self.min_risk_scale, self.max_risk_scale))
                
        except Exception as e:
            self.log_operator_warning(f"Risk transition smoothing failed: {e}")
            return new_scale

    def _record_risk_adjustment_event(self, old_scale: float, new_scale: float,
                                     stats: Dict[str, float], context: Dict[str, Any]) -> None:
        """Record significant risk adjustment events"""
        
        try:
            event = {
                "timestamp": datetime.datetime.now().isoformat(),
                "old_scale": old_scale,
                "new_scale": new_scale,
                "change": new_scale - old_scale,
                "stats": stats.copy(),
                "context": context.copy(),
                "risk_factors": self.risk_factors.copy(),
                "reason": self._determine_adjustment_reason(old_scale, new_scale, stats)
            }
            
            self.risk_events.append(event)
            
            # Keep only recent events
            if len(self.risk_events) > 50:
                self.risk_events = self.risk_events[-50:]
                
            # Log significant adjustments
            if abs(new_scale - old_scale) > 0.2:
                self.log_operator_warning(
                    f"⚙️ Significant risk adjustment: {old_scale:.2f} → {new_scale:.2f}",
                    reason=event["reason"],
                    regime=context.get('regime', 'unknown')
                )
                
        except Exception as e:
            self.log_operator_warning(f"Risk event recording failed: {e}")

    def _determine_adjustment_reason(self, old_scale: float, new_scale: float, 
                                   stats: Dict[str, float]) -> str:
        """Determine the primary reason for risk adjustment"""
        
        try:
            if new_scale < old_scale:  # Risk reduction
                if stats.get("drawdown", 0) > 0.1:
                    return "drawdown_protection"
                elif stats.get("correlation_risk", 0) > 0.6:
                    return "correlation_risk"
                elif self.consecutive_losses > 3:
                    return "losing_streak"
                else:
                    return "volatility_protection"
            else:  # Risk increase
                if self.recovery_speed > 0.1:
                    return "recovery_mode"
                else:
                    return "favorable_conditions"
                    
        except Exception:
            return "unknown"

    def _update_external_integrations(self, info_bus: InfoBus) -> None:
        """Update external integrations from other modules"""
        
        try:
            module_data = info_bus.get('module_data', {})
            
            # Update from drawdown rescue
            drawdown_data = module_data.get('drawdown_rescue', {})
            if 'risk_adjustment' in drawdown_data:
                self.external_signals['drawdown_rescue'] = drawdown_data['risk_adjustment']
            
            # Update from compliance module
            compliance_data = module_data.get('compliance', {})
            if 'risk_budget_used' in compliance_data:
                compliance_factor = 1.0 - compliance_data['risk_budget_used']
                self.external_signals['compliance'] = compliance_factor
            
            # Aggregate external signals
            if self.external_signals:
                self.external_risk_scale = np.mean(list(self.external_signals.values()))
            else:
                self.external_risk_scale = 1.0
                
        except Exception as e:
            self.log_operator_warning(f"External integration update failed: {e}")

    def _calculate_final_risk_scale(self, context: Dict[str, Any]) -> None:
        """Calculate final risk scale with all adjustments"""
        
        try:
            # The risk scale has already been calculated in _adjust_risk_comprehensive
            # This method applies any final adjustments
            
            # Apply decay factor to gradually return to base scale
            if self.current_risk_scale < self.base_risk_scale:
                self.current_risk_scale = min(
                    self.base_risk_scale,
                    self.current_risk_scale + (self.base_risk_scale - self.current_risk_scale) * (1 - self.risk_decay)
                )
            
            # Final bounds check
            self.current_risk_scale = float(np.clip(
                self.current_risk_scale, 
                self.min_risk_scale, 
                self.max_risk_scale
            ))
            
        except Exception as e:
            self.log_operator_warning(f"Final risk scale calculation failed: {e}")

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with risk controller results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'dynamic_risk_controller', {
            'current_risk_scale': self.current_risk_scale,
            'risk_factors': self.risk_factors.copy(),
            'market_regime': self.market_regime,
            'volatility_regime': self.volatility_regime,
            'consecutive_losses': self.consecutive_losses,
            'risk_adjustments_made': self.risk_adjustments_made,
            'emergency_interventions': self.emergency_interventions,
            'external_risk_scale': self.external_risk_scale,
            'active_risk_factors': sum(1 for factor in self.risk_factors.values() if factor < 0.9)
        })
        
        # Update risk snapshot
        InfoBusUpdater.update_risk_snapshot(info_bus, {
            'dynamic_risk_scale': self.current_risk_scale,
            'risk_reduction_active': self.current_risk_scale < 0.8,
            'emergency_mode': self.current_risk_scale < 0.5
        })
        
        # Add alerts for significant risk reductions
        if self.current_risk_scale < 0.5:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Emergency risk reduction: {self.current_risk_scale:.1%} scale",
                severity="critical",
                module="DynamicRiskController"
            )
        elif self.current_risk_scale < 0.7:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Significant risk reduction: {self.current_risk_scale:.1%} scale",
                severity="warning",
                module="DynamicRiskController"
            )

    def _record_risk_audit(self, info_bus: InfoBus, context: Dict[str, Any], 
                          stats: Dict[str, float]) -> None:
        """Record comprehensive audit trail"""
        
        # Only audit significant changes or periodically
        should_audit = (
            abs(self.current_risk_scale - self.base_risk_scale) > 0.15 or
            len(self.risk_events) > 0 or
            info_bus.get('step_idx', 0) % 100 == 0
        )
        
        if should_audit:
            audit_data = {
                'risk_scale': self.current_risk_scale,
                'risk_factors': self.risk_factors.copy(),
                'market_context': {
                    'regime': self.market_regime,
                    'volatility_regime': self.volatility_regime,
                    'session': self.market_session
                },
                'stats': stats,
                'external_signals': self.external_signals.copy(),
                'performance_metrics': {
                    'adjustments_made': self.risk_adjustments_made,
                    'emergency_interventions': self.emergency_interventions,
                    'consecutive_losses': self.consecutive_losses
                },
                'recent_events': len(self.risk_events)
            }
            
            self.audit_manager.record_event(
                event_type="risk_scaling",
                module="DynamicRiskController",
                details=audit_data,
                severity="warning" if self.current_risk_scale < 0.5 else "info"
            )

    def _update_risk_performance_metrics(self) -> None:
        """Update performance and risk metrics"""
        
        # Update performance metrics
        self._update_performance_metric('current_risk_scale', self.current_risk_scale)
        self._update_performance_metric('risk_adjustments_made', self.risk_adjustments_made)
        self._update_performance_metric('emergency_interventions', self.emergency_interventions)
        
        # Update risk factor metrics
        active_risk_factors = sum(1 for factor in self.risk_factors.values() if factor < 0.9)
        self._update_performance_metric('active_risk_factors', active_risk_factors)
        
        # Update regime performance tracking
        if self.market_regime != "unknown":
            self.regime_performance[self.market_regime]['risk_scales'].append(self.current_risk_scale)
            self.regime_performance[self.market_regime]['timestamps'].append(
                datetime.datetime.now().isoformat()
            )

    def _process_legacy_step(self, **kwargs) -> None:
        """Process legacy step parameters for backward compatibility"""
        
        try:
            # Extract legacy stats
            stats = {
                "drawdown": kwargs.get("drawdown", kwargs.get("current_drawdown", 0.0)),
                "volatility": kwargs.get("volatility", 0.01),
                "pnl": kwargs.get("pnl", 0.0)
            }
            
            # Create minimal context
            context = {
                "regime": kwargs.get("market_regime", "unknown"),
                "volatility_level": "medium",
                "session": "unknown"
            }
            
            # Update market context
            if "market_regime" in kwargs:
                self.market_regime = kwargs["market_regime"]
            
            # Update correlation if provided
            if "correlation" in kwargs:
                self.risk_factors["correlation"] = self._calculate_correlation_factor(kwargs["correlation"])
            
            # Perform basic risk adjustment
            self._adjust_risk_comprehensive(stats, context, {})
            
        except Exception as e:
            self.log_operator_error(f"Legacy step processing failed: {e}")

    def _calculate_correlation_factor(self, max_correlation: float) -> float:
        """Calculate correlation factor from maximum correlation"""
        
        if max_correlation > 0.8:
            return 0.6
        elif max_correlation > 0.6:
            return 0.8
        else:
            return 1.0

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        
        try:
            return np.array([
                float(self.current_risk_scale),
                float(self.risk_factors["drawdown"]),
                float(self.risk_factors["volatility"]),
                float(self.risk_factors["correlation"]),
                float(self.risk_factors["losing_streak"]),
                float(min(self.consecutive_losses / 10.0, 1.0)),
                float(self.external_risk_scale),
                float(1.0 if self.market_regime in ["volatile", "extreme"] else 0.0)
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Risk observation generation failed: {e}")
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def get_risk_control_report(self) -> str:
        """Generate operator-friendly risk control report"""
        
        # Status indicators
        if self.current_risk_scale < 0.3:
            risk_status = "🚨 Emergency"
        elif self.current_risk_scale < 0.6:
            risk_status = "⚠️ High Reduction"
        elif self.current_risk_scale < 0.8:
            risk_status = "⚡ Moderate Reduction"
        else:
            risk_status = "✅ Normal"
        
        # Market status
        regime_emoji = {"trending": "📈", "volatile": "💥", "ranging": "↔️", "unknown": "❓"}
        market_status = f"{regime_emoji.get(self.market_regime, '❓')} {self.market_regime.title()}"
        
        # Risk factor status
        risk_factor_lines = []
        for factor_name, factor_value in self.risk_factors.items():
            if factor_value < 0.9:
                emoji = "🚨" if factor_value < 0.5 else "⚠️" if factor_value < 0.7 else "⚡"
                risk_factor_lines.append(f"  {emoji} {factor_name.replace('_', ' ').title()}: {factor_value:.1%}")
        
        # Recent risk events
        event_lines = []
        for event in self.risk_events[-3:]:  # Show last 3 events
            timestamp = event['timestamp'][:19]  # Remove microseconds
            event_lines.append(
                f"  🔧 {timestamp}: {event['old_scale']:.2f} → {event['new_scale']:.2f} "
                f"({event['reason'].replace('_', ' ')})"
            )
        
        return f"""
⚙️ DYNAMIC RISK CONTROLLER
═══════════════════════════════════════
🎯 Risk Status: {risk_status} ({self.current_risk_scale:.1%} scale)
📊 Market Regime: {market_status}
💥 Volatility Level: {self.volatility_regime.title()}
🕐 Market Session: {self.market_session.title()}

⚖️ RISK SCALE CONFIGURATION
• Base Scale: {self.base_risk_scale:.1%}
• Current Scale: {self.current_risk_scale:.1%}
• Minimum Scale: {self.min_risk_scale:.1%}
• Maximum Scale: {self.max_risk_scale:.1%}
• External Scale: {self.external_risk_scale:.1%}

📊 ACTIVE RISK FACTORS
{chr(10).join(risk_factor_lines) if risk_factor_lines else "  ✅ All risk factors normal"}

🔧 CONTROLLER PERFORMANCE
• Risk Adjustments Made: {self.risk_adjustments_made}
• Emergency Interventions: {self.emergency_interventions}
• Consecutive Losses: {self.consecutive_losses}
• Adaptive Scaling: {'✅ Enabled' if self.adaptive_scaling else '❌ Disabled'}
• Regime Awareness: {'✅ Enabled' if self.regime_aware else '❌ Disabled'}

📈 MARKET CONTEXT
• Current Regime: {self.market_regime.title()}
• Volatility Regime: {self.volatility_regime.title()}
• Session: {self.market_session.title()}
• Regime Sensitivity: {self.regime_sensitivity:.1f}x
• Correlation Sensitivity: {self.correlation_sensitivity:.1f}x

📜 RECENT RISK EVENTS
{chr(10).join(event_lines) if event_lines else "  📭 No recent risk adjustments"}

💡 SYSTEM STATUS
• History Tracking: {len(self.risk_scale_history)} records
• Volatility History: {len(self.vol_history)} records
• Regime History: {len(self.market_regime_history)} changes
• Recovery Speed: {self.recovery_speed:.1%}
        """

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_current_risk_scale(self) -> float:
        """Get current risk scale for external use"""
        return self.current_risk_scale

    def set_external_risk_scale(self, scale: float) -> None:
        """Set external risk scale override"""
        self.external_risk_scale = float(np.clip(scale, 0.1, 2.0))

    def get_risk_factors(self) -> Dict[str, float]:
        """Get current risk factors"""
        return self.risk_factors.copy()

    def force_emergency_mode(self, reason: str = "manual_override") -> None:
        """Force emergency risk reduction"""
        old_scale = self.current_risk_scale
        self.current_risk_scale = self.min_risk_scale
        self.emergency_interventions += 1
        
        self.log_operator_error(
            f"🚨 Emergency mode FORCED",
            reason=reason,
            old_scale=f"{old_scale:.2f}",
            new_scale=f"{self.current_risk_scale:.2f}"
        )

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        
        # Use enhanced step if no InfoBus
        self._process_legacy_step(**kwargs)

    def adjust_risk(self, stats: Dict[str, float]) -> None:
        """Legacy risk adjustment interface"""
        
        context = {"regime": "unknown", "volatility_level": "medium", "session": "unknown"}
        self._adjust_risk_comprehensive(stats, context, {})

    def calculate_risk_scale(self) -> float:
        """Legacy interface to get risk scale"""
        return self.current_risk_scale