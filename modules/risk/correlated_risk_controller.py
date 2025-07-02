# ─────────────────────────────────────────────────────────────
# File: modules/risk/correlated_risk_controller.py
# Enhanced with InfoBus integration & intelligent correlation analysis
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import RiskMixin, AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import AuditTracker, format_operator_message


class CorrelatedRiskController(Module, RiskMixin, AnalysisMixin, StateManagementMixin):
    """
    Enhanced correlation risk controller with InfoBus integration.
    Monitors and controls risk from correlated positions with intelligent analysis.
    """

    def __init__(
        self,
        max_corr: float = 0.9,
        warning_corr: float = 0.7,
        info_corr: float = 0.5,
        enabled: bool = True,
        history_size: int = 20,
        correlation_window: int = 50,
        dynamic_thresholds: bool = True,
        debug: bool = True,
        **kwargs
    ):
        # Initialize with enhanced config
        config = ModuleConfig(
            debug=debug,
            max_history=max(history_size, 50),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_risk_state()
        self._initialize_analysis_state()
        
        # Core configuration
        self.enabled = enabled
        self.dynamic_thresholds = dynamic_thresholds
        
        # Correlation thresholds
        self.base_thresholds = {
            'max_corr': float(max_corr),
            'warning_corr': float(warning_corr),
            'info_corr': float(info_corr)
        }
        
        # Current thresholds (may be adjusted dynamically)
        self.current_thresholds = self.base_thresholds.copy()
        
        # Enhanced state tracking
        self.correlation_history = deque(maxlen=history_size)
        self.current_correlations: Dict[Tuple[str, str], float] = {}
        self.correlation_matrix_history = deque(maxlen=10)  # Store recent matrices
        self.risk_score = 0.0
        self.step_count = 0
        
        # Alert system with enhanced categorization
        self.alerts: Dict[str, List[Dict[str, Any]]] = {
            "info": [],
            "warning": [],
            "critical": []
        }
        
        # Performance tracking
        self.correlation_analytics = defaultdict(list)
        self.regime_correlations = defaultdict(lambda: defaultdict(list))
        self.correlation_trends = {}
        
        # Risk assessment
        self._high_correlation_pairs = set()
        self._correlation_concentration = 0.0
        self._diversification_score = 1.0
        
        # Audit system
        self.audit_manager = AuditTracker("CorrelatedRiskController")
        self._last_significant_event = None
        
        self.log_operator_info(
            "🔗 Enhanced Correlation Risk Controller initialized",
            max_correlation=f"{max_corr:.1%}",
            warning_threshold=f"{warning_corr:.1%}",
            info_threshold=f"{info_corr:.1%}",
            dynamic_thresholds=dynamic_thresholds,
            enabled=enabled
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_analysis_state()
        
        # Reset correlation tracking
        self.correlation_history.clear()
        self.current_correlations.clear()
        self.correlation_matrix_history.clear()
        self.risk_score = 0.0
        self.step_count = 0
        
        # Reset alerts
        for severity in self.alerts:
            self.alerts[severity].clear()
        
        # Reset analytics
        self.correlation_analytics.clear()
        self.regime_correlations.clear()
        self.correlation_trends.clear()
        
        # Reset risk assessment
        self._high_correlation_pairs.clear()
        self._correlation_concentration = 0.0
        self._diversification_score = 1.0
        
        # Reset thresholds to base values
        self.current_thresholds = self.base_thresholds.copy()
        
        self.log_operator_info("🔄 Correlation Risk Controller reset - all tracking cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - correlation controller inactive")
            return
        
        if not self.enabled:
            self.risk_score = 0.0
            return
        
        self.step_count += 1
        
        # Extract context for intelligent analysis
        context = extract_standard_context(info_bus)
        
        # Extract correlation data from InfoBus
        correlations = self._extract_correlations_from_info_bus(info_bus)
        
        if not correlations:
            self._handle_no_correlation_data(context)
            return
        
        # Process correlation analysis
        critical_found = self._analyze_correlations_enhanced(correlations, info_bus, context)
        
        # Update dynamic thresholds if enabled
        if self.dynamic_thresholds:
            self._update_dynamic_thresholds(context)
        
        # Calculate comprehensive risk score
        self._calculate_comprehensive_risk_score(context)
        
        # Update InfoBus with results
        self._update_info_bus(info_bus, critical_found)
        
        # Record audit for significant events
        if critical_found or self.risk_score > 0.5:
            self._record_comprehensive_audit(info_bus, context, correlations)
        
        # Update performance metrics
        self._update_correlation_metrics()

    def _extract_correlations_from_info_bus(self, info_bus: InfoBus) -> Dict[Tuple[str, str], float]:
        """Extract correlation data from InfoBus with multiple fallback methods"""
        
        correlations = {}
        
        try:
            # Method 1: Direct correlation data from module data
            module_data = info_bus.get('module_data', {})
            
            # Check for correlation data from various modules
            for module_name in ['correlation_engine', 'market_analyzer', 'risk_analyzer']:
                if module_name in module_data:
                    corr_data = module_data[module_name].get('correlations', {})
                    if corr_data:
                        correlations.update(self._process_correlation_data(corr_data))
            
            # Method 2: Calculate from position data
            if not correlations:
                correlations = self._calculate_correlations_from_positions(info_bus)
            
            # Method 3: Calculate from price data
            if not correlations and len(self.correlation_matrix_history) >= 2:
                correlations = self._estimate_correlations_from_history(info_bus)
            
            # Method 4: Generate synthetic for training (if absolutely no data)
            if not correlations and self.step_count < 50:  # Only during bootstrap
                correlations = self._generate_bootstrap_correlations(info_bus)
            
        except Exception as e:
            self.log_operator_warning(f"Correlation extraction failed: {e}")
        
        return correlations

    def _process_correlation_data(self, corr_data: Any) -> Dict[Tuple[str, str], float]:
        """Safely process correlation data from various formats"""
        
        processed = {}
        
        try:
            if isinstance(corr_data, dict):
                # Handle different key formats
                for key, value in corr_data.items():
                    try:
                        if isinstance(key, (tuple, list)) and len(key) >= 2:
                            clean_key = (str(key[0]), str(key[1]))
                            processed[clean_key] = float(value)
                        elif isinstance(key, str):
                            # Handle string formats
                            if '_' in key:
                                parts = key.split('_', 1)
                                if len(parts) == 2:
                                    processed[(parts[0], parts[1])] = float(value)
                            elif '-' in key:
                                parts = key.split('-', 1)
                                if len(parts) == 2:
                                    processed[(parts[0], parts[1])] = float(value)
                    except (ValueError, IndexError, TypeError):
                        continue
                        
            elif isinstance(corr_data, (list, np.ndarray)):
                # Handle matrix format
                corr_array = np.array(corr_data)
                if corr_array.ndim == 2:
                    n = min(corr_array.shape[0], corr_array.shape[1])
                    for i in range(n):
                        for j in range(i + 1, n):
                            if (i < corr_array.shape[0] and j < corr_array.shape[1] and
                                not np.isnan(corr_array[i, j]) and not np.isinf(corr_array[i, j])):
                                processed[(f"INST_{i}", f"INST_{j}")] = float(corr_array[i, j])
                                
        except Exception as e:
            self.log_operator_warning(f"Correlation data processing failed: {e}")
        
        return processed

    def _calculate_correlations_from_positions(self, info_bus: InfoBus) -> Dict[Tuple[str, str], float]:
        """Calculate correlations from position price movements"""
        
        positions = InfoBusExtractor.get_positions(info_bus)
        prices = info_bus.get('prices', {})
        
        if len(positions) < 2 or not prices:
            return {}
        
        correlations = {}
        
        try:
            # Get instruments from positions
            instruments = list(set(pos.get('symbol', '') for pos in positions))
            instruments = [inst for inst in instruments if inst in prices]
            
            if len(instruments) < 2:
                return {}
            
            # Use price history to estimate correlations
            if len(self.correlation_matrix_history) >= 10:
                # Calculate rolling correlation from recent price data
                recent_prices = {}
                for inst in instruments:
                    recent_prices[inst] = [prices[inst]]  # Current price
                    
                    # Add historical prices if available
                    for hist_data in list(self.correlation_matrix_history)[-10:]:
                        if inst in hist_data.get('prices', {}):
                            recent_prices[inst].append(hist_data['prices'][inst])
                
                # Calculate correlations
                for i, inst1 in enumerate(instruments):
                    for j, inst2 in enumerate(instruments[i+1:], i+1):
                        if (len(recent_prices.get(inst1, [])) >= 5 and 
                            len(recent_prices.get(inst2, [])) >= 5):
                            
                            prices1 = np.array(recent_prices[inst1][-5:])
                            prices2 = np.array(recent_prices[inst2][-5:])
                            
                            # Calculate returns
                            returns1 = np.diff(prices1) / (prices1[:-1] + 1e-8)
                            returns2 = np.diff(prices2) / (prices2[:-1] + 1e-8)
                            
                            # Calculate correlation
                            if len(returns1) >= 3 and len(returns2) >= 3:
                                corr = np.corrcoef(returns1, returns2)[0, 1]
                                if not np.isnan(corr):
                                    correlations[(inst1, inst2)] = float(corr)
                                    
        except Exception as e:
            self.log_operator_warning(f"Position correlation calculation failed: {e}")
        
        return correlations

    def _estimate_correlations_from_history(self, info_bus: InfoBus) -> Dict[Tuple[str, str], float]:
        """Estimate current correlations from historical patterns"""
        
        if not self.correlation_history:
            return {}
        
        # Get recent average correlations
        recent_correlations = defaultdict(list)
        
        for hist_data in list(self.correlation_history)[-5:]:
            correlations = hist_data.get('correlations', {})
            for pair, corr in correlations.items():
                recent_correlations[pair].append(corr)
        
        # Calculate averages
        estimated = {}
        for pair, corr_list in recent_correlations.items():
            if len(corr_list) >= 2:
                estimated[pair] = float(np.mean(corr_list))
        
        return estimated

    def _generate_bootstrap_correlations(self, info_bus: InfoBus) -> Dict[Tuple[str, str], float]:
        """Generate realistic bootstrap correlations for early training"""
        
        positions = InfoBusExtractor.get_positions(info_bus)
        if len(positions) < 2:
            return {}
        
        instruments = list(set(pos.get('symbol', '') for pos in positions))
        if len(instruments) < 2:
            return {}
        
        correlations = {}
        
        try:
            # Generate realistic correlations based on instrument types
            for i, inst1 in enumerate(instruments):
                for j, inst2 in enumerate(instruments[i+1:], i+1):
                    # Generate correlation based on instrument similarity
                    if self._are_instruments_similar(inst1, inst2):
                        # Similar instruments have higher correlation
                        base_corr = 0.6 + np.random.normal(0, 0.2)
                    else:
                        # Different instruments have lower correlation
                        base_corr = 0.2 + np.random.normal(0, 0.3)
                    
                    # Clamp to valid range
                    corr = np.clip(base_corr, -0.95, 0.95)
                    correlations[(inst1, inst2)] = float(corr)
                    
        except Exception as e:
            self.log_operator_warning(f"Bootstrap correlation generation failed: {e}")
        
        return correlations

    def _are_instruments_similar(self, inst1: str, inst2: str) -> bool:
        """Determine if two instruments are similar (for correlation estimation)"""
        
        # Major currency pairs
        major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'NZD/USD']
        major_pairs_no_slash = [p.replace('/', '') for p in major_pairs]
        
        # Gold instruments
        gold_instruments = ['XAU/USD', 'XAUUSD', 'GOLD']
        
        # Check if both are major pairs
        if ((inst1 in major_pairs or inst1 in major_pairs_no_slash) and
            (inst2 in major_pairs or inst2 in major_pairs_no_slash)):
            return True
        
        # Check if both are gold instruments
        if inst1 in gold_instruments and inst2 in gold_instruments:
            return True
        
        # Check if they share a common currency
        for inst in [inst1, inst2]:
            inst_clean = inst.replace('/', '')
            if len(inst_clean) >= 6:
                base1, quote1 = inst_clean[:3], inst_clean[3:6]
                
                for other_inst in [inst1, inst2]:
                    if other_inst == inst:
                        continue
                    other_clean = other_inst.replace('/', '')
                    if len(other_clean) >= 6:
                        base2, quote2 = other_clean[:3], other_clean[3:6]
                        if base1 in [base2, quote2] or quote1 in [base2, quote2]:
                            return True
        
        return False

    def _handle_no_correlation_data(self, context: Dict[str, Any]) -> None:
        """Handle case when no correlation data is available"""
        
        self.risk_score = 0.0
        self.current_correlations.clear()
        
        # Clear alerts
        for severity in self.alerts:
            self.alerts[severity].clear()
        
        # Log occasionally
        if self.step_count % 100 == 0:
            self.log_operator_info(
                "📊 No correlation data available",
                step=self.step_count,
                regime=context.get('regime', 'unknown')
            )

    def _analyze_correlations_enhanced(self, correlations: Dict[Tuple[str, str], float],
                                     info_bus: InfoBus, context: Dict[str, Any]) -> bool:
        """Enhanced correlation analysis with context awareness"""
        
        # Update current correlations
        self.current_correlations = correlations.copy()
        
        # Clear previous alerts
        for severity in self.alerts:
            self.alerts[severity].clear()
        
        critical_found = False
        correlation_stats = []
        
        # Analyze each correlation pair
        for (inst1, inst2), corr in correlations.items():
            try:
                abs_corr = abs(float(corr))
                correlation_stats.append(abs_corr)
                
                # Determine severity with context adjustment
                severity = self._get_correlation_severity_enhanced(abs_corr, context)
                
                if severity != "none":
                    alert_data = {
                        'pair': (inst1, inst2),
                        'correlation': corr,
                        'abs_correlation': abs_corr,
                        'severity': severity,
                        'context': context.copy(),
                        'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat())
                    }
                    
                    self.alerts[severity].append(alert_data)
                    
                    # Log significant correlations
                    if severity in ["warning", "critical"]:
                        if severity == "critical":
                            self.log_operator_error(
                                f"🚨 CRITICAL correlation: {inst1}/{inst2}",
                                correlation=f"{corr:.3f}",
                                threshold=f"{self.current_thresholds['max_corr']:.3f}",
                                regime=context.get('regime', 'unknown')
                            )
                            critical_found = True
                        else:
                            self.log_operator_warning(
                                f"⚠️ High correlation: {inst1}/{inst2}",
                                correlation=f"{corr:.3f}",
                                threshold=f"{self.current_thresholds['warning_corr']:.3f}"
                            )
                    
                    # Track high correlation pairs
                    if severity in ["warning", "critical"]:
                        self._high_correlation_pairs.add((inst1, inst2))
                    
            except (ValueError, TypeError) as e:
                self.log_operator_warning(f"Invalid correlation data for {inst1}/{inst2}: {e}")
                continue
        
        # Calculate portfolio-level metrics
        self._calculate_portfolio_correlation_metrics(correlation_stats, context)
        
        # Update correlation trends
        self._update_correlation_trends(correlations, context)
        
        # Store correlation history
        self._store_correlation_snapshot(correlations, context, info_bus)
        
        return critical_found

    def _get_correlation_severity_enhanced(self, abs_corr: float, context: Dict[str, Any]) -> str:
        """Enhanced severity assessment with context awareness"""
        
        # Get current thresholds
        thresholds = self.current_thresholds
        
        # Base severity assessment
        if abs_corr >= thresholds['max_corr']:
            base_severity = "critical"
        elif abs_corr >= thresholds['warning_corr']:
            base_severity = "warning"
        elif abs_corr >= thresholds['info_corr']:
            base_severity = "info"
        else:
            base_severity = "none"
        
        # Context-based adjustments
        regime = context.get('regime', 'unknown')
        volatility_level = context.get('volatility_level', 'medium')
        
        # More tolerant in volatile markets (correlations naturally increase)
        if volatility_level in ['high', 'extreme'] and base_severity == "warning":
            if abs_corr < thresholds['max_corr'] * 0.95:  # 5% tolerance
                base_severity = "info"
        
        # More strict in ranging markets (should maintain diversification)
        elif regime == 'ranging' and base_severity == "info":
            if abs_corr > thresholds['info_corr'] * 1.2:  # 20% stricter
                base_severity = "warning"
        
        return base_severity

    def _calculate_portfolio_correlation_metrics(self, correlation_stats: List[float], 
                                                context: Dict[str, Any]) -> None:
        """Calculate portfolio-level correlation metrics"""
        
        if not correlation_stats:
            self._correlation_concentration = 0.0
            self._diversification_score = 1.0
            return
        
        try:
            # Correlation concentration (how clustered the correlations are)
            high_corr_count = sum(1 for corr in correlation_stats if corr > 0.7)
            self._correlation_concentration = high_corr_count / max(len(correlation_stats), 1)
            
            # Diversification score (inverse of average correlation)
            avg_correlation = np.mean(correlation_stats)
            self._diversification_score = max(0.0, 1.0 - avg_correlation)
            
            # Update analytics
            regime = context.get('regime', 'unknown')
            self.correlation_analytics['avg_correlation'].append(avg_correlation)
            self.correlation_analytics['max_correlation'].append(max(correlation_stats))
            self.correlation_analytics['concentration'].append(self._correlation_concentration)
            
            # Store by regime
            self.regime_correlations[regime]['avg'].append(avg_correlation)
            self.regime_correlations[regime]['max'].append(max(correlation_stats))
            
        except Exception as e:
            self.log_operator_warning(f"Portfolio correlation metrics calculation failed: {e}")

    def _update_correlation_trends(self, correlations: Dict[Tuple[str, str], float],
                                  context: Dict[str, Any]) -> None:
        """Update correlation trend analysis"""
        
        try:
            for pair, current_corr in correlations.items():
                # Initialize trend tracking
                if pair not in self.correlation_trends:
                    self.correlation_trends[pair] = deque(maxlen=10)
                
                # Add current correlation
                self.correlation_trends[pair].append({
                    'correlation': current_corr,
                    'timestamp': datetime.datetime.now().isoformat(),
                    'regime': context.get('regime', 'unknown')
                })
                
                # Analyze trend if enough data
                if len(self.correlation_trends[pair]) >= 5:
                    recent_corrs = [item['correlation'] for item in list(self.correlation_trends[pair])[-5:]]
                    trend_slope = np.polyfit(range(len(recent_corrs)), recent_corrs, 1)[0]
                    
                    # Alert on rapidly increasing correlations
                    if trend_slope > 0.1:  # Correlation increasing by >0.1 over 5 periods
                        self.log_operator_warning(
                            f"📈 Rising correlation trend: {pair[0]}/{pair[1]}",
                            trend_slope=f"{trend_slope:.3f}",
                            current_corr=f"{current_corr:.3f}"
                        )
                        
        except Exception as e:
            self.log_operator_warning(f"Correlation trend update failed: {e}")

    def _store_correlation_snapshot(self, correlations: Dict[Tuple[str, str], float],
                                   context: Dict[str, Any], info_bus: InfoBus) -> None:
        """Store comprehensive correlation snapshot"""
        
        snapshot = {
            'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
            'step_idx': info_bus.get('step_idx', self.step_count),
            'correlations': correlations.copy(),
            'context': context.copy(),
            'prices': info_bus.get('prices', {}),
            'portfolio_metrics': {
                'concentration': self._correlation_concentration,
                'diversification_score': self._diversification_score,
                'high_corr_pairs': len(self._high_correlation_pairs)
            },
            'alert_counts': {severity: len(alerts) for severity, alerts in self.alerts.items()}
        }
        
        self.correlation_history.append(snapshot)
        self.correlation_matrix_history.append(snapshot)

    def _update_dynamic_thresholds(self, context: Dict[str, Any]) -> None:
        """Update thresholds dynamically based on market context"""
        
        if not self.dynamic_thresholds:
            return
        
        try:
            regime = context.get('regime', 'unknown')
            volatility_level = context.get('volatility_level', 'medium')
            
            # Base threshold adjustments
            adjustments = {'max_corr': 1.0, 'warning_corr': 1.0, 'info_corr': 1.0}
            
            # Regime-based adjustments
            if regime == 'volatile':
                # More tolerant during volatile periods
                adjustments = {'max_corr': 1.1, 'warning_corr': 1.15, 'info_corr': 1.2}
            elif regime == 'ranging':
                # More strict during ranging periods
                adjustments = {'max_corr': 0.95, 'warning_corr': 0.9, 'info_corr': 0.85}
            
            # Volatility-based adjustments
            if volatility_level == 'extreme':
                # Additional tolerance in extreme volatility
                for key in adjustments:
                    adjustments[key] *= 1.1
            elif volatility_level == 'low':
                # Stricter in low volatility
                for key in adjustments:
                    adjustments[key] *= 0.95
            
            # Apply adjustments
            for threshold_name, multiplier in adjustments.items():
                self.current_thresholds[threshold_name] = min(
                    self.base_thresholds[threshold_name] * multiplier,
                    0.98  # Never exceed 98% correlation threshold
                )
                
        except Exception as e:
            self.log_operator_warning(f"Dynamic threshold update failed: {e}")

    def _calculate_comprehensive_risk_score(self, context: Dict[str, Any]) -> None:
        """Calculate comprehensive correlation risk score"""
        
        try:
            base_score = 0.0
            
            # Weight by alert severity
            base_score += len(self.alerts["critical"]) * 0.5
            base_score += len(self.alerts["warning"]) * 0.3
            base_score += len(self.alerts["info"]) * 0.1
            
            # Factor in concentration risk
            base_score += self._correlation_concentration * 0.4
            
            # Factor in diversification
            base_score += (1.0 - self._diversification_score) * 0.3
            
            # Context adjustments
            regime = context.get('regime', 'unknown')
            if regime == 'volatile':
                base_score *= 0.8  # More tolerant in volatile markets
            elif regime == 'ranging':
                base_score *= 1.2  # More concerned in ranging markets
            
            # Normalize
            self.risk_score = min(base_score, 1.0)
            
            # Update risk tracking
            self._update_risk_metrics({
                'correlation_risk_score': self.risk_score,
                'concentration': self._correlation_concentration,
                'diversification': self._diversification_score
            })
            
        except Exception as e:
            self.log_operator_warning(f"Risk score calculation failed: {e}")
            self.risk_score = 0.0

    def _update_info_bus(self, info_bus: InfoBus, critical_found: bool) -> None:
        """Update InfoBus with correlation analysis results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'correlated_risk_controller', {
            'risk_score': self.risk_score,
            'correlation_count': len(self.current_correlations),
            'high_corr_pairs': len(self._high_correlation_pairs),
            'alerts': {severity: len(alerts) for severity, alerts in self.alerts.items()},
            'thresholds': self.current_thresholds.copy(),
            'portfolio_metrics': {
                'concentration': self._correlation_concentration,
                'diversification_score': self._diversification_score
            },
            'critical_found': critical_found
        })
        
        # Update risk snapshot
        InfoBusUpdater.update_risk_snapshot(info_bus, {
            'correlation_risk_score': self.risk_score,
            'high_correlations': len([
                alert for alert in self.alerts['critical'] + self.alerts['warning']
            ]),
            'diversification_score': self._diversification_score
        })
        
        # Add alerts for critical situations
        if critical_found:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Critical correlations detected: {len(self.alerts['critical'])} pairs",
                severity="critical",
                module="CorrelatedRiskController"
            )
        elif self.risk_score > 0.6:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Elevated correlation risk: {self.risk_score:.1%}",
                severity="warning",
                module="CorrelatedRiskController"
            )

    def _record_comprehensive_audit(self, info_bus: InfoBus, context: Dict[str, Any],
                                   correlations: Dict[Tuple[str, str], float]) -> None:
        """Record comprehensive audit trail"""
        
        audit_data = {
            'risk_score': self.risk_score,
            'correlation_count': len(correlations),
            'context': context,
            'alerts': {
                severity: [
                    {
                        'pair': alert['pair'],
                        'correlation': alert['correlation'],
                        'severity': alert['severity']
                    }
                    for alert in alerts
                ]
                for severity, alerts in self.alerts.items()
                if alerts
            },
            'portfolio_metrics': {
                'concentration': self._correlation_concentration,
                'diversification_score': self._diversification_score,
                'high_corr_pairs_count': len(self._high_correlation_pairs)
            },
            'thresholds': self.current_thresholds.copy(),
            'step_count': self.step_count
        }
        
        self.audit_manager.record_event(
            event_type="correlation_analysis",
            module="CorrelatedRiskController",
            details=audit_data,
            severity="critical" if any(
                alert.get('severity') == 'critical'
                for alerts in self.alerts.values()
                for alert in alerts
            ) else "warning" if self.risk_score > 0.5 else "info"
        )

    def _update_correlation_metrics(self) -> None:
        """Update performance and correlation metrics"""
        
        # Update performance metrics
        self._update_performance_metric('risk_score', self.risk_score)
        self._update_performance_metric('correlation_count', len(self.current_correlations))
        self._update_performance_metric('high_corr_pairs', len(self._high_correlation_pairs))
        self._update_performance_metric('diversification_score', self._diversification_score)
        
        # Update alert metrics
        total_alerts = sum(len(alerts) for alerts in self.alerts.values())
        self._update_performance_metric('total_alerts', total_alerts)

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        
        try:
            # Basic risk metrics
            risk_score = float(self.risk_score)
            correlation_count_norm = min(len(self.current_correlations) / 20.0, 1.0)
            
            # Correlation statistics
            if self.current_correlations:
                corr_values = list(self.current_correlations.values())
                avg_corr = np.mean(np.abs(corr_values))
                max_corr = np.max(np.abs(corr_values))
            else:
                avg_corr = 0.0
                max_corr = 0.0
            
            # Alert indicators
            critical_alerts = len(self.alerts["critical"]) / 10.0  # Normalized
            warning_alerts = len(self.alerts["warning"]) / 10.0
            
            # Portfolio metrics
            concentration = self._correlation_concentration
            diversification = self._diversification_score
            
            return np.array([
                risk_score,                    # Overall correlation risk
                correlation_count_norm,        # Number of correlations tracked
                avg_corr,                      # Average absolute correlation
                max_corr,                      # Maximum absolute correlation
                min(critical_alerts, 1.0),    # Critical alerts (normalized)
                min(warning_alerts, 1.0),     # Warning alerts (normalized)
                concentration,                 # Correlation concentration
                diversification                # Diversification score
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Correlation observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)

    def get_correlation_report(self) -> str:
        """Generate operator-friendly correlation report"""
        
        # Status indicators
        if self.risk_score > 0.7:
            risk_status = "🚨 Critical"
        elif self.risk_score > 0.4:
            risk_status = "⚠️ Elevated"
        else:
            risk_status = "✅ Normal"
        
        # Diversification status
        if self._diversification_score > 0.8:
            diversification_status = "🎯 Excellent"
        elif self._diversification_score > 0.6:
            diversification_status = "✅ Good"
        elif self._diversification_score > 0.4:
            diversification_status = "⚡ Fair"
        else:
            diversification_status = "⚠️ Poor"
        
        # Current high correlations
        high_corr_lines = []
        for severity in ['critical', 'warning']:
            for alert in self.alerts[severity]:
                pair = alert['pair']
                corr = alert['correlation']
                emoji = "🚨" if severity == 'critical' else "⚠️"
                high_corr_lines.append(f"  {emoji} {pair[0]}/{pair[1]}: {corr:.3f}")
        
        # Recent correlation trends
        trend_lines = []
        for pair, trend_data in list(self.correlation_trends.items())[:5]:  # Show top 5
            if len(trend_data) >= 3:
                recent_corrs = [item['correlation'] for item in list(trend_data)[-3:]]
                trend_direction = "📈" if recent_corrs[-1] > recent_corrs[0] else "📉" if recent_corrs[-1] < recent_corrs[0] else "➡️"
                trend_lines.append(f"  {trend_direction} {pair[0]}/{pair[1]}: {recent_corrs[-1]:.3f}")
        
        return f"""
🔗 CORRELATED RISK CONTROLLER
═══════════════════════════════════════
🎯 Risk Status: {risk_status} ({self.risk_score:.1%})
📊 Diversification: {diversification_status} ({self._diversification_score:.1%})
🔗 Tracked Pairs: {len(self.current_correlations)}
🔄 Controller Enabled: {'✅ Yes' if self.enabled else '❌ No'}

⚖️ CORRELATION THRESHOLDS
• Info Level: {self.current_thresholds['info_corr']:.1%}
• Warning Level: {self.current_thresholds['warning_corr']:.1%}
• Critical Level: {self.current_thresholds['max_corr']:.1%}
• Dynamic Adjustments: {'✅ Enabled' if self.dynamic_thresholds else '❌ Disabled'}

📊 PORTFOLIO ANALYSIS
• Concentration Risk: {self._correlation_concentration:.1%}
• High Correlation Pairs: {len(self._high_correlation_pairs)}
• Average Correlation: {(np.mean(np.abs(list(self.current_correlations.values()))) if self.current_correlations else 0):.1%}
• Maximum Correlation: {(np.max(np.abs(list(self.current_correlations.values()))) if self.current_correlations else 0):.1%}

🚨 CURRENT HIGH CORRELATIONS
{chr(10).join(high_corr_lines) if high_corr_lines else "  ✅ No high correlations detected"}

📈 CORRELATION TRENDS
{chr(10).join(trend_lines[:5]) if trend_lines else "  📊 Insufficient data for trends"}

🚨 ALERT SUMMARY
• Critical: {len(self.alerts['critical'])}
• Warning: {len(self.alerts['warning'])}
• Info: {len(self.alerts['info'])}

💡 MONITORING STATUS
• Step Count: {self.step_count:,}
• Analysis History: {len(self.correlation_history)} snapshots
• Trend Tracking: {len(self.correlation_trends)} pairs
• Last Update: {(self.correlation_history[-1]['timestamp'] if self.correlation_history else 'Never')}
        """

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, correlations: Optional[Union[Dict, List, np.ndarray]] = None,
            positions: Optional[Dict[str, Any]] = None,
            correlation_matrix: Optional[np.ndarray] = None,
            instruments: Optional[List[str]] = None, **kwargs) -> bool:
        """Legacy compatibility method"""
        
        # Create mock InfoBus from legacy parameters
        mock_info_bus = {
            'step_idx': self.step_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'positions': [],
            'module_data': {}
        }
        
        # Add correlation data to mock InfoBus
        if correlations is not None:
            mock_info_bus['module_data']['correlation_engine'] = {
                'correlations': correlations
            }
        
        # Add positions
        if positions:
            for symbol, pos_data in positions.items():
                mock_info_bus['positions'].append({
                    'symbol': symbol,
                    'size': pos_data.get('size', 0),
                    'current_price': pos_data.get('current_price', 1.0)
                })
        
        # Use enhanced step method
        self._step_impl(mock_info_bus)
        
        # Return True if critical correlations detected
        return len(self.alerts['critical']) > 0