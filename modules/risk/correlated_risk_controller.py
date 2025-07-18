"""
Enhanced Correlated Risk Controller with SmartInfoBus Integration
Monitors correlation risk between positions and instruments
"""

import numpy as np
import datetime
import time
from typing import Dict, Any, List, Optional, Union, Tuple
from collections import deque, defaultdict
from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, fcluster

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="CorrelatedRiskController",
    version="3.0.0",
    category="risk",
    provides=["correlation_risk", "diversification_score", "correlation_clusters"],
    requires=["positions", "prices", "market_context"],
    description="Enhanced correlation risk monitoring with intelligent clustering and diversification analysis",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class CorrelatedRiskController(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Enhanced Correlated Risk Controller with SmartInfoBus Integration
    
    Monitors correlation risk between positions with advanced analytics including
    hierarchical clustering, dynamic correlation analysis, and regime-aware adjustments.
    """
    
    def __init__(self, config=None, **kwargs):
        self.config = config or {}
        super().__init__()
        self._initialize_advanced_systems()
        
        # Configuration with intelligent defaults
        self.max_correlation = self.config.get('max_correlation', 0.8)
        self.warning_correlation = self.config.get('warning_correlation', 0.6)
        self.min_diversification = self.config.get('min_diversification', 0.3)
        self.lookback_window = self.config.get('lookback_window', 50)
        self.enabled = self.config.get('enabled', True)
        
        # Enhanced correlation tracking
        self.correlation_matrix = {}
        self.correlation_history = deque(maxlen=100)
        self.price_history = defaultdict(lambda: deque(maxlen=self.lookback_window))
        self.return_history = defaultdict(lambda: deque(maxlen=self.lookback_window))
        
        # Risk assessment
        self.correlation_risk_score = 0.0
        self.diversification_score = 1.0
        self.cluster_risk_score = 0.0
        self.severity_level = "normal"
        
        # Advanced analytics
        self.correlation_clusters = {}
        self.regime_correlations = defaultdict(lambda: defaultdict(list))
        self.dynamic_correlations = {}
        self.volatility_adjusted_correlations = {}
        
        # Performance tracking
        self.step_count = 0
        self.correlation_violations = 0
        self.diversification_violations = 0
        
        self.logger.info(format_operator_message(
            message="Enhanced Correlated Risk Controller initialized",
            icon="🔗",
            max_correlation=f"{self.max_correlation:.2f}",
            warning_threshold=f"{self.warning_correlation:.2f}",
            min_diversification=f"{self.min_diversification:.2f}",
            enabled=self.enabled
        ))
    
    def _initialize_advanced_systems(self):
        """Initialize advanced monitoring and error handling systems"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="CorrelatedRiskController",
            log_path="logs/risk/correlated_risk_controller.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("CorrelatedRiskController", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
    
    def _initialize(self) -> None:
        """Initialize the correlated risk controller (required by BaseModule)"""
        try:
            # Validate configuration
            if not self.enabled:
                self.logger.warning("Correlated Risk Controller is disabled")
                return
            
            # Initialize correlation tracking
            self.correlation_matrix = {}
            self.correlation_history = deque(maxlen=100)
            self.price_history = defaultdict(lambda: deque(maxlen=self.lookback_window))
            self.return_history = defaultdict(lambda: deque(maxlen=self.lookback_window))
            
            # Initialize risk metrics
            self.correlation_risk_score = 0.0
            self.diversification_score = 1.0
            self.cluster_risk_score = 0.0
            self.severity_level = "normal"
            
            # Initialize performance tracking
            self.step_count = 0
            self.correlation_violations = 0
            self.diversification_violations = 0
            
            self.logger.info("Correlated Risk Controller initialization completed successfully")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "controller_initialization")
            self.logger.error(f"Controller initialization failed: {error_context}")
    
    async def calculate_confidence(self, action: Dict[str, Any], **kwargs) -> float:
        """Calculate confidence score for correlation risk assessment"""
        try:
            # Base confidence starts high for correlation analysis
            confidence = 0.9
            
            # Factors that affect confidence
            factors = {}
            
            # Check diversification score
            factors['diversification_score'] = self.diversification_score
            confidence *= self.diversification_score
            
            # Check correlation risk score (inverse relationship)
            if self.correlation_risk_score > 0:
                risk_penalty = max(0.3, 1.0 - self.correlation_risk_score)
                factors['risk_penalty'] = risk_penalty
                confidence *= risk_penalty
            
            # Check data availability - more instruments = higher confidence
            instruments_count = len(self.price_history)
            if instruments_count >= 5:
                data_factor = min(1.0, instruments_count / 10.0)
            else:
                data_factor = max(0.3, instruments_count / 5.0)
            factors['data_availability'] = data_factor
            confidence *= data_factor
            
            # Check severity level
            severity_penalties = {
                'normal': 1.0,
                'warning': 0.8,
                'critical': 0.6,
                'error': 0.4,
                'disabled': 0.1
            }
            severity_factor = severity_penalties.get(self.severity_level, 0.5)
            factors['severity_factor'] = severity_factor
            confidence *= severity_factor
            
            # Ensure confidence is in valid range
            confidence = max(0.0, min(1.0, confidence))
            
            # Log confidence calculation for debugging
            self.logger.debug(f"Correlation confidence: {confidence:.3f}, factors: {factors}")
            
            return float(confidence)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "confidence_calculation")
            self.logger.error(f"Confidence calculation failed: {error_context}")
            return 0.5  # Default medium confidence on error
    
    async def propose_action(self, **kwargs) -> Dict[str, Any]:
        """Propose correlation risk management actions"""
        try:
            action_proposal = {
                'action_type': 'correlation_risk_management',
                'timestamp': time.time(),
                'correlation_risk_score': self.correlation_risk_score,
                'diversification_score': self.diversification_score,
                'severity_level': self.severity_level,
                'recommendations': [],
                'warnings': [],
                'adjustments': {}
            }
            
            # Generate recommendations based on correlation state
            if self.correlation_risk_score > 0.7:
                action_proposal['recommendations'].append({
                    'type': 'reduce_correlation',
                    'reason': 'High correlation risk detected',
                    'suggested_action': 'Reduce position sizes in highly correlated instruments',
                    'priority': 'high'
                })
            
            if self.diversification_score < self.min_diversification:
                action_proposal['recommendations'].append({
                    'type': 'improve_diversification',
                    'reason': f'Diversification score {self.diversification_score:.2f} below threshold {self.min_diversification}',
                    'suggested_action': 'Add positions in uncorrelated instruments',
                    'priority': 'medium'
                })
            
            # Check cluster concentrations
            if hasattr(self, 'correlation_clusters') and self.correlation_clusters:
                max_cluster_size = max(len(instruments) for instruments in self.correlation_clusters.values())
                if max_cluster_size > 5:
                    action_proposal['warnings'].append({
                        'type': 'cluster_concentration',
                        'cluster_size': max_cluster_size,
                        'threshold': 5,
                        'risk_level': 'high'
                    })
            
            # Check recent violations
            if self.correlation_violations > 3:
                action_proposal['warnings'].append({
                    'type': 'frequent_violations',
                    'violation_count': self.correlation_violations,
                    'risk_level': 'medium'
                })
            
            # Suggest position adjustments based on correlation matrix
            if hasattr(self, 'correlation_matrix') and self.correlation_matrix:
                high_corr_pairs = [(pair, corr) for pair, corr in self.correlation_matrix.items() 
                                 if abs(corr) > self.max_correlation]
                if high_corr_pairs:
                    action_proposal['adjustments']['high_correlation_pairs'] = [
                        {'instruments': pair, 'correlation': corr, 'action': 'reduce_exposure'}
                        for pair, corr in high_corr_pairs[:3]  # Top 3 most problematic
                    ]
            
            self.logger.debug(f"Correlation action proposed: {len(action_proposal['recommendations'])} recommendations, "
                            f"{len(action_proposal['warnings'])} warnings")
            
            return action_proposal
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "action_proposal")
            self.logger.error(f"Action proposal failed: {error_context}")
            return {
                'action_type': 'correlation_risk_management',
                'timestamp': time.time(),
                'error': str(e),
                'recommendations': [],
                'warnings': [],
                'adjustments': {}
            }
    
    async def process(self, **kwargs) -> Dict[str, Any]:
        """
        Enhanced correlation risk analysis with comprehensive monitoring
        
        Returns:
            Dict containing correlation risk assessment, clusters, and diversification metrics
        """
        try:
            if not self.enabled:
                return self._generate_disabled_response()
            
            self.step_count += 1
            
            # Extract comprehensive market data
            positions = self.smart_bus.get('positions', 'CorrelatedRiskController') or []
            prices = self.smart_bus.get('prices', 'CorrelatedRiskController') or {}
            market_context = self.smart_bus.get('market_context', 'CorrelatedRiskController') or {}
            
            # Update price and return histories
            self._update_price_histories(prices)
            
            # Perform comprehensive correlation analysis
            correlation_results = await self._analyze_correlations_comprehensive(positions, market_context)
            
            # Generate intelligent thesis
            thesis = await self._generate_correlation_thesis(correlation_results, market_context)
            
            # Calculate comprehensive risk metrics
            risk_metrics = self._calculate_correlation_risk_metrics(correlation_results)
            
            # Update SmartInfoBus
            self._update_smart_info_bus(correlation_results, risk_metrics, thesis)
            
            # Record performance metrics
            self.performance_tracker.record_metric(
                'CorrelatedRiskController', 'correlation_analysis', 
                correlation_results.get('processing_time_ms', 0), True
            )
            
            return {
                'correlation_risk_score': self.correlation_risk_score,
                'diversification_score': self.diversification_score,
                'severity_level': self.severity_level,
                'correlation_results': correlation_results,
                'risk_metrics': risk_metrics,
                'thesis': thesis,
                'recommendations': self._generate_recommendations(correlation_results)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "CorrelatedRiskController")
            self.logger.error(f"Correlation analysis failed: {error_context}")
            return self._generate_error_response(str(error_context))
    
    def _update_price_histories(self, prices: Dict[str, float]):
        """Update price and return histories for correlation calculation"""
        try:
            current_time = datetime.datetime.now()
            
            for instrument, price in prices.items():
                if price and price > 0:
                    # Update price history
                    self.price_history[instrument].append({
                        'price': float(price),
                        'timestamp': current_time
                    })
                    
                    # Calculate returns if we have previous price
                    if len(self.price_history[instrument]) >= 2:
                        prev_price = self.price_history[instrument][-2]['price']
                        return_value = (price - prev_price) / prev_price if prev_price > 0 else 0.0
                        
                        self.return_history[instrument].append({
                            'return': return_value,
                            'timestamp': current_time
                        })
                        
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "price_history_update")
            self.logger.warning(f"Price history update failed: {error_context}")
    
    async def _analyze_correlations_comprehensive(self, positions: List[Dict], 
                                                market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive correlation analysis with advanced features"""
        start_time = datetime.datetime.now()
        
        try:
            # Extract instruments from positions
            instruments = list(set(pos.get('symbol', pos.get('instrument', '')) for pos in positions))
            instruments = [inst for inst in instruments if inst and inst in self.return_history]
            
            if len(instruments) < 2:
                return self._generate_insufficient_data_response()
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_enhanced_correlation_matrix(instruments, market_context)
            
            # Perform clustering analysis
            cluster_analysis = self._perform_correlation_clustering(correlation_matrix, instruments)
            
            # Calculate diversification metrics
            diversification_metrics = self._calculate_diversification_metrics(correlation_matrix, positions)
            
            # Analyze correlation violations
            violation_analysis = self._analyze_correlation_violations(correlation_matrix, instruments)
            
            # Regime-specific analysis
            regime_analysis = self._analyze_regime_correlations(correlation_matrix, market_context)
            
            # Calculate processing time
            processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            return {
                'correlation_matrix': correlation_matrix,
                'cluster_analysis': cluster_analysis,
                'diversification_metrics': diversification_metrics,
                'violation_analysis': violation_analysis,
                'regime_analysis': regime_analysis,
                'instruments_analyzed': instruments,
                'processing_time_ms': processing_time,
                'market_context': market_context
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "correlation_analysis")
            self.logger.error(f"Correlation analysis failed: {error_context}")
            return self._generate_analysis_error_response(str(error_context))
    
    def _calculate_enhanced_correlation_matrix(self, instruments: List[str], 
                                             market_context: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
        """Calculate enhanced correlation matrix with regime and volatility adjustments"""
        correlation_matrix = {}
        regime = market_context.get('regime', 'unknown')
        volatility_regime = market_context.get('volatility_level', 'medium')
        
        try:
            for i, inst1 in enumerate(instruments):
                for j, inst2 in enumerate(instruments[i+1:], i+1):
                    # Get return data
                    returns1 = [r['return'] for r in self.return_history[inst1] if r['return'] is not None]
                    returns2 = [r['return'] for r in self.return_history[inst2] if r['return'] is not None]
                    
                    if len(returns1) >= 10 and len(returns2) >= 10:
                        # Calculate basic correlation
                        min_length = min(len(returns1), len(returns2))
                        returns1_aligned = returns1[-min_length:]
                        returns2_aligned = returns2[-min_length:]
                        
                        try:
                            pearson_result = pearsonr(returns1_aligned, returns2_aligned)
                            correlation: float = pearson_result[0]  # type: ignore
                            
                            # Apply regime adjustments
                            adjusted_correlation = self._apply_regime_adjustments(
                                correlation, inst1, inst2, regime, volatility_regime
                            )
                            
                            correlation_matrix[(inst1, inst2)] = adjusted_correlation
                            
                            # Store in regime history
                            self.regime_correlations[regime][(inst1, inst2)].append(adjusted_correlation)
                            
                        except Exception:
                            # Fallback to basic calculation
                            correlation_matrix[(inst1, inst2)] = 0.0
                    else:
                        # Insufficient data - use instrument similarity heuristic
                        correlation_matrix[(inst1, inst2)] = self._estimate_correlation_heuristic(inst1, inst2)
            
            return correlation_matrix
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "correlation_matrix")
            self.logger.error(f"Correlation matrix calculation failed: {error_context}")
            return {}
    
    def _apply_regime_adjustments(self, correlation: float, inst1: str, inst2: str, 
                                 regime: str, volatility_regime: str) -> float:
        """Apply regime-specific adjustments to correlation"""
        try:
            adjusted_correlation = correlation
            
            # Volatility regime adjustments
            if volatility_regime == 'high':
                # Correlations tend to increase during high volatility
                adjusted_correlation = min(0.95, correlation * 1.2)
            elif volatility_regime == 'extreme':
                # Extreme volatility often leads to correlation convergence
                adjusted_correlation = min(0.95, correlation * 1.4)
            elif volatility_regime == 'low':
                # Low volatility allows for better diversification
                adjusted_correlation = max(-0.95, correlation * 0.8)
            
            # Market regime adjustments
            if regime == 'crisis':
                # Crisis periods show correlation convergence
                adjusted_correlation = min(0.95, adjusted_correlation * 1.3)
            elif regime == 'trending':
                # Trending markets may reduce correlations
                adjusted_correlation = max(-0.95, adjusted_correlation * 0.9)
            
            return float(np.clip(adjusted_correlation, -0.95, 0.95))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_adjustment")
            self.logger.warning(f"Regime adjustment failed: {error_context}")
            return correlation
    
    def _estimate_correlation_heuristic(self, inst1: str, inst2: str) -> float:
        """Estimate correlation using instrument type heuristics"""
        try:
            # Currency pair correlations
            if all('USD' in inst for inst in [inst1, inst2]):
                return 0.4  # USD pairs tend to be moderately correlated
            elif all(any(curr in inst for curr in ['EUR', 'GBP']) for inst in [inst1, inst2]):
                return 0.5  # European currencies moderate correlation
            elif all('JPY' in inst for inst in [inst1, inst2]):
                return 0.6  # JPY pairs tend to be more correlated
            
            # Gold and currency correlations
            elif all('XAU' in inst or 'GOLD' in inst for inst in [inst1, inst2]):
                return 0.8  # Gold instruments highly correlated
            elif any('XAU' in inst or 'GOLD' in inst for inst in [inst1, inst2]):
                return -0.2  # Gold vs currencies often negative correlation
            
            # Different asset classes
            else:
                return 0.1  # Low correlation for different asset types
                
        except Exception:
            return 0.0
    
    def _perform_correlation_clustering(self, correlation_matrix: Dict[Tuple[str, str], float], 
                                      instruments: List[str]) -> Dict[str, Any]:
        """Perform hierarchical clustering based on correlations"""
        try:
            if len(instruments) < 3:
                return {'clusters': {0: instruments}, 'cluster_count': 1, 'silhouette_score': 1.0}
            
            # Build distance matrix (1 - |correlation|)
            n = len(instruments)
            distance_matrix = np.ones((n, n))
            
            for i, inst1 in enumerate(instruments):
                for j, inst2 in enumerate(instruments):
                    if i != j:
                        correlation = correlation_matrix.get((inst1, inst2), 
                                                           correlation_matrix.get((inst2, inst1), 0.0))
                        distance_matrix[i, j] = 1 - abs(correlation)
            
            # Perform hierarchical clustering
            linkage_matrix = linkage(distance_matrix[np.triu_indices(n, k=1)], method='ward')
            
            # Determine optimal number of clusters
            optimal_clusters = min(max(2, len(instruments) // 3), 5)
            cluster_labels = fcluster(linkage_matrix, optimal_clusters, criterion='maxclust')
            
            # Organize clusters
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(instruments[i])
            
            # Calculate cluster risk scores
            cluster_risks = {}
            for cluster_id, cluster_instruments in clusters.items():
                cluster_risk = self._calculate_cluster_risk(cluster_instruments, correlation_matrix)
                cluster_risks[cluster_id] = cluster_risk
            
            # Store clusters for external access
            self.correlation_clusters = dict(clusters)
            
            return {
                'clusters': dict(clusters),
                'cluster_count': len(clusters),
                'cluster_risks': cluster_risks,
                'max_cluster_risk': max(cluster_risks.values()) if cluster_risks else 0.0,
                'clustering_quality': self._assess_clustering_quality(clusters, correlation_matrix)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "correlation_clustering")
            self.logger.warning(f"Correlation clustering failed: {error_context}")
            return {'clusters': {0: instruments}, 'cluster_count': 1, 'error': error_context}
    
    def _calculate_cluster_risk(self, cluster_instruments: List[str], 
                               correlation_matrix: Dict[Tuple[str, str], float]) -> float:
        """Calculate risk score for a correlation cluster"""
        try:
            if len(cluster_instruments) < 2:
                return 0.0
            
            # Calculate average intra-cluster correlation
            correlations = []
            for i, inst1 in enumerate(cluster_instruments):
                for j, inst2 in enumerate(cluster_instruments[i+1:], i+1):
                    corr = correlation_matrix.get((inst1, inst2), 
                                                correlation_matrix.get((inst2, inst1), 0.0))
                    correlations.append(abs(corr))
            
            if correlations:
                avg_correlation = np.mean(correlations)
                # Risk increases exponentially with correlation and cluster size
                size_factor = len(cluster_instruments) / 10.0  # Normalize by typical portfolio size
                return float(avg_correlation * (1 + size_factor))
            
            return 0.0
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "cluster_risk")
            self.logger.warning(f"Cluster risk calculation failed: {error_context}")
            return 0.0
    
    def _assess_clustering_quality(self, clusters: Dict[int, List[str]], 
                                  correlation_matrix: Dict[Tuple[str, str], float]) -> float:
        """Assess the quality of correlation clustering"""
        try:
            if len(clusters) <= 1:
                return 0.0
            
            # Calculate silhouette-like score
            total_score = 0.0
            total_instruments = 0
            
            for cluster_id, cluster_instruments in clusters.items():
                for instrument in cluster_instruments:
                    # Intra-cluster cohesion
                    intra_cluster_corr = []
                    for other_inst in cluster_instruments:
                        if other_inst != instrument:
                            corr = correlation_matrix.get((instrument, other_inst),
                                                        correlation_matrix.get((other_inst, instrument), 0.0))
                            intra_cluster_corr.append(abs(corr))
                    
                    # Inter-cluster separation
                    inter_cluster_corr = []
                    for other_cluster_id, other_cluster in clusters.items():
                        if other_cluster_id != cluster_id:
                            for other_inst in other_cluster:
                                corr = correlation_matrix.get((instrument, other_inst),
                                                            correlation_matrix.get((other_inst, instrument), 0.0))
                                inter_cluster_corr.append(abs(corr))
                    
                    if intra_cluster_corr and inter_cluster_corr:
                        cohesion = np.mean(intra_cluster_corr)
                        separation = np.mean(inter_cluster_corr)
                        score = cohesion - separation  # Higher is better clustering
                        total_score += score
                        total_instruments += 1
            
            return float(total_score / max(total_instruments, 1))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "clustering_quality")
            self.logger.warning(f"Clustering quality assessment failed: {error_context}")
            return 0.0
    
    def _calculate_diversification_metrics(self, correlation_matrix: Dict[Tuple[str, str], float], 
                                         positions: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive diversification metrics"""
        try:
            if not correlation_matrix:
                return {'diversification_score': 1.0, 'effective_positions': len(positions)}
            
            # Calculate average correlation
            correlations = list(correlation_matrix.values())
            avg_correlation = np.mean([abs(corr) for corr in correlations]) if correlations else 0.0
            
            # Calculate diversification ratio (1 - average correlation)
            basic_diversification = 1.0 - avg_correlation
            
            # Calculate effective number of positions (Herfindahl-like index)
            position_weights = []
            total_exposure = 0.0
            
            for position in positions:
                size = abs(position.get('size', position.get('volume', 0)))
                price = position.get('current_price', position.get('price', 1.0))
                exposure = size * price
                total_exposure += exposure
                position_weights.append(exposure)
            
            if total_exposure > 0:
                # Normalize weights
                weights = [w / total_exposure for w in position_weights]
                # Calculate concentration (Herfindahl index)
                concentration = sum(w**2 for w in weights)
                effective_positions = 1.0 / concentration if concentration > 0 else len(positions)
            else:
                effective_positions = len(positions)
            
            # Combine diversification measures
            position_diversification = min(1.0, effective_positions / len(positions)) if positions else 1.0
            self.diversification_score = (basic_diversification + position_diversification) / 2.0
            
            return {
                'diversification_score': self.diversification_score,
                'avg_correlation': avg_correlation,
                'effective_positions': effective_positions,
                'concentration_index': 1.0 - position_diversification,
                'position_count': len(positions),
                'correlation_pairs': len(correlation_matrix)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "diversification_metrics")
            self.logger.error(f"Diversification metrics calculation failed: {error_context}")
            return {'diversification_score': 0.5, 'effective_positions': 1}
    
    def _analyze_correlation_violations(self, correlation_matrix: Dict[Tuple[str, str], float], 
                                      instruments: List[str]) -> Dict[str, Any]:
        """Analyze correlation violations and risk concentrations"""
        try:
            violations = {'critical': [], 'warning': [], 'info': []}
            
            for (inst1, inst2), correlation in correlation_matrix.items():
                abs_corr = abs(correlation)
                
                if abs_corr >= self.max_correlation:
                    violations['critical'].append({
                        'instruments': (inst1, inst2),
                        'correlation': correlation,
                        'threshold': self.max_correlation,
                        'severity': 'critical'
                    })
                    self.correlation_violations += 1
                elif abs_corr >= self.warning_correlation:
                    violations['warning'].append({
                        'instruments': (inst1, inst2),
                        'correlation': correlation,
                        'threshold': self.warning_correlation,
                        'severity': 'warning'
                    })
            
            # Check diversification violations
            if self.diversification_score < self.min_diversification:
                violations['critical'].append({
                    'type': 'diversification',
                    'score': self.diversification_score,
                    'threshold': self.min_diversification,
                    'severity': 'critical'
                })
                self.diversification_violations += 1
            
            return {
                'violations': violations,
                'total_violations': sum(len(v) for v in violations.values()),
                'critical_pairs': len(violations['critical']),
                'warning_pairs': len(violations['warning']),
                'max_correlation': max([abs(corr) for corr in correlation_matrix.values()]) if correlation_matrix else 0.0
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "violation_analysis")
            self.logger.error(f"Violation analysis failed: {error_context}")
            return {'violations': {'critical': [], 'warning': [], 'info': []}, 'total_violations': 0}
    
    def _analyze_regime_correlations(self, correlation_matrix: Dict[Tuple[str, str], float], 
                                   market_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze regime-specific correlation patterns"""
        try:
            regime = market_context.get('regime', 'unknown')
            volatility_regime = market_context.get('volatility_level', 'medium')
            
            # Store current correlations in regime history
            for pair, correlation in correlation_matrix.items():
                self.regime_correlations[regime][pair].append(correlation)
            
            # Calculate regime-specific statistics
            regime_stats = {}
            for regime_name, regime_data in self.regime_correlations.items():
                if regime_data:
                    all_correlations = []
                    for pair_correlations in regime_data.values():
                        all_correlations.extend(pair_correlations)
                    
                    if all_correlations:
                        regime_stats[regime_name] = {
                            'avg_correlation': float(np.mean([abs(c) for c in all_correlations])),
                            'max_correlation': float(np.max([abs(c) for c in all_correlations])),
                            'correlation_volatility': float(np.std(all_correlations)),
                            'sample_count': len(all_correlations)
                        }
            
            return {
                'current_regime': regime,
                'current_volatility': volatility_regime,
                'regime_stats': regime_stats,
                'regime_shift_impact': self._assess_regime_shift_impact(regime_stats, regime)
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_analysis")
            self.logger.warning(f"Regime analysis failed: {error_context}")
            return {'current_regime': 'unknown', 'regime_stats': {}}
    
    def _assess_regime_shift_impact(self, regime_stats: Dict[str, Dict], current_regime: str) -> str:
        """Assess the impact of potential regime shifts on correlations"""
        try:
            if current_regime not in regime_stats or len(regime_stats) < 2:
                return "insufficient_data"
            
            current_avg = regime_stats[current_regime]['avg_correlation']
            
            # Compare with other regimes
            other_regimes = {k: v for k, v in regime_stats.items() if k != current_regime}
            if not other_regimes:
                return "no_comparison_data"
            
            max_other_avg = max(stats['avg_correlation'] for stats in other_regimes.values())
            min_other_avg = min(stats['avg_correlation'] for stats in other_regimes.values())
            
            if current_avg > max_other_avg * 1.2:
                return "high_correlation_regime"
            elif current_avg < min_other_avg * 0.8:
                return "low_correlation_regime"
            else:
                return "normal_correlation_regime"
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_shift_assessment")
            return "assessment_error"
    
    def _calculate_correlation_risk_metrics(self, correlation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive correlation risk metrics"""
        try:
            violation_analysis = correlation_results['violation_analysis']
            diversification_metrics = correlation_results['diversification_metrics']
            cluster_analysis = correlation_results['cluster_analysis']
            
            # Violation-based risk
            violation_risk = (
                len(violation_analysis['violations']['critical']) * 1.0 +
                len(violation_analysis['violations']['warning']) * 0.6
            ) / max(len(correlation_results['instruments_analyzed']), 1)
            
            # Diversification risk
            diversification_risk = max(0.0, (self.min_diversification - self.diversification_score) / self.min_diversification)
            
            # Cluster concentration risk
            self.cluster_risk_score = cluster_analysis.get('max_cluster_risk', 0.0)
            
            # Combined correlation risk score
            self.correlation_risk_score = min(1.0, 
                violation_risk * 0.4 + 
                diversification_risk * 0.4 + 
                self.cluster_risk_score * 0.2
            )
            
            # Determine severity level
            if self.correlation_risk_score > 0.7 or len(violation_analysis['violations']['critical']) > 0:
                self.severity_level = 'critical'
            elif self.correlation_risk_score > 0.4 or len(violation_analysis['violations']['warning']) > 0:
                self.severity_level = 'warning'
            elif self.correlation_risk_score > 0.1:
                self.severity_level = 'elevated'
            else:
                self.severity_level = 'normal'
            
            return {
                'correlation_risk_score': self.correlation_risk_score,
                'diversification_score': self.diversification_score,
                'cluster_risk_score': self.cluster_risk_score,
                'severity_level': self.severity_level,
                'violation_risk': violation_risk,
                'diversification_risk': diversification_risk
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "risk_metrics")
            self.logger.error(f"Risk metrics calculation failed: {error_context}")
            return {'correlation_risk_score': 0.5, 'severity_level': 'unknown'}
    
    async def _generate_correlation_thesis(self, correlation_results: Dict[str, Any], 
                                         market_context: Dict[str, Any]) -> str:
        """Generate intelligent thesis explaining correlation analysis"""
        try:
            instruments = correlation_results['instruments_analyzed']
            violation_analysis = correlation_results['violation_analysis']
            diversification_metrics = correlation_results['diversification_metrics']
            cluster_analysis = correlation_results['cluster_analysis']
            regime_analysis = correlation_results['regime_analysis']
            
            thesis_parts = []
            
            # Portfolio overview
            thesis_parts.append(
                f"Analyzed {len(instruments)} instruments with {len(correlation_results['correlation_matrix'])} correlation pairs"
            )
            
            # Diversification assessment
            div_score = diversification_metrics['diversification_score']
            if div_score >= 0.7:
                thesis_parts.append(f"EXCELLENT diversification achieved ({div_score:.1%})")
            elif div_score >= 0.5:
                thesis_parts.append(f"ADEQUATE diversification maintained ({div_score:.1%})")
            else:
                thesis_parts.append(f"POOR diversification detected ({div_score:.1%}) - concentration risk elevated")
            
            # Violation analysis
            critical_violations = len(violation_analysis['violations']['critical'])
            warning_violations = len(violation_analysis['violations']['warning'])
            
            if critical_violations > 0:
                thesis_parts.append(f"CRITICAL: {critical_violations} correlation violations exceed {self.max_correlation:.1%} threshold")
            elif warning_violations > 0:
                thesis_parts.append(f"WARNING: {warning_violations} correlations approaching limits")
            else:
                thesis_parts.append("All correlations within acceptable ranges")
            
            # Cluster analysis
            cluster_count = cluster_analysis['cluster_count']
            max_cluster_risk = cluster_analysis.get('max_cluster_risk', 0.0)
            
            if max_cluster_risk > 0.7:
                thesis_parts.append(f"HIGH cluster concentration risk detected in {cluster_count} correlation clusters")
            elif cluster_count > 1:
                thesis_parts.append(f"Portfolio organized into {cluster_count} correlation clusters with manageable risk")
            
            # Regime analysis
            regime = regime_analysis['current_regime']
            regime_shift_impact = regime_analysis.get('regime_shift_impact', 'unknown')
            
            if regime != 'unknown':
                if regime_shift_impact == 'high_correlation_regime':
                    thesis_parts.append(f"Current {regime} regime shows elevated correlation levels - monitor for regime shifts")
                elif regime_shift_impact == 'low_correlation_regime':
                    thesis_parts.append(f"Current {regime} regime provides favorable diversification environment")
            
            # Risk assessment conclusion
            thesis_parts.append(
                f"Overall correlation risk: {self.severity_level.upper()} "
                f"(score: {self.correlation_risk_score:.2f})"
            )
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Thesis generation failed: {error_context}"
    
    def _generate_recommendations(self, correlation_results: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on correlation analysis"""
        recommendations = []
        
        try:
            violation_analysis = correlation_results['violation_analysis']
            diversification_metrics = correlation_results['diversification_metrics']
            cluster_analysis = correlation_results['cluster_analysis']
            
            # Critical violation recommendations
            critical_violations = violation_analysis['violations']['critical']
            if critical_violations:
                recommendations.append("IMMEDIATE: Reduce positions in highly correlated instruments")
                for violation in critical_violations[:3]:  # Show top 3
                    if violation.get('instruments'):
                        inst1, inst2 = violation['instruments']
                        recommendations.append(f"Consider closing or reducing {inst1} or {inst2} positions")
            
            # Diversification recommendations
            if diversification_metrics['diversification_score'] < self.min_diversification:
                recommendations.append("Improve portfolio diversification by adding uncorrelated instruments")
                recommendations.append("Consider reducing position sizes in concentrated areas")
            
            # Cluster-based recommendations
            max_cluster_risk = cluster_analysis.get('max_cluster_risk', 0.0)
            if max_cluster_risk > 0.6:
                recommendations.append("High cluster concentration detected - rebalance across clusters")
                
                # Identify problematic clusters
                cluster_risks = cluster_analysis.get('cluster_risks', {})
                high_risk_clusters = [cid for cid, risk in cluster_risks.items() if risk > 0.6]
                if high_risk_clusters:
                    recommendations.append(f"Focus on rebalancing clusters: {', '.join(map(str, high_risk_clusters))}")
            
            # Position-specific recommendations
            effective_positions = diversification_metrics['effective_positions']
            position_count = diversification_metrics['position_count']
            
            if effective_positions < position_count * 0.6:
                recommendations.append("Concentration detected - consider equal weighting or risk parity approach")
            
            # Proactive recommendations
            if not recommendations:
                recommendations.append("Correlation risk well managed - maintain current diversification strategy")
                recommendations.append("Continue monitoring for regime shifts that may affect correlations")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendations")
            recommendations.append(f"Recommendation generation failed: {error_context}")
        
        return recommendations
    
    def _update_smart_info_bus(self, correlation_results: Dict[str, Any], 
                              risk_metrics: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with correlation analysis results"""
        try:
            # Core correlation risk data
            self.smart_bus.set('correlation_risk', {
                'risk_score': self.correlation_risk_score,
                'severity_level': self.severity_level,
                'correlation_results': correlation_results,
                'risk_metrics': risk_metrics,
                'thesis': thesis
            }, module='CorrelatedRiskController', thesis=thesis)
            
            # Diversification score for other modules
            self.smart_bus.set('diversification_score', self.diversification_score, 
                             module='CorrelatedRiskController', 
                             thesis=f"Portfolio diversification: {self.diversification_score:.1%}")
            
            # Correlation clusters for portfolio management
            self.smart_bus.set('correlation_clusters', self.correlation_clusters, 
                             module='CorrelatedRiskController', 
                             thesis=f"Identified {len(self.correlation_clusters)} correlation clusters")
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smart_info_bus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")
    
    def _generate_insufficient_data_response(self) -> Dict[str, Any]:
        """Generate response when insufficient data is available"""
        return {
            'correlation_matrix': {},
            'cluster_analysis': {'clusters': {}, 'cluster_count': 0},
            'diversification_metrics': {'diversification_score': 1.0, 'effective_positions': 0},
            'violation_analysis': {'violations': {'critical': [], 'warning': [], 'info': []}, 'total_violations': 0},
            'regime_analysis': {'current_regime': 'unknown'},
            'instruments_analyzed': [],
            'processing_time_ms': 0,
            'status': 'insufficient_data'
        }
    
    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'correlation_risk_score': 0.0,
            'diversification_score': 1.0,
            'severity_level': 'disabled',
            'correlation_results': self._generate_insufficient_data_response(),
            'risk_metrics': {'correlation_risk_score': 0.0, 'severity_level': 'disabled'},
            'thesis': "Correlated Risk Controller is disabled",
            'recommendations': ["Enable Correlated Risk Controller for correlation monitoring"]
        }
    
    def _generate_error_response(self, error_context: str) -> Dict[str, Any]:
        """Generate response when processing fails"""
        return {
            'correlation_risk_score': 0.5,
            'diversification_score': 0.5,
            'severity_level': 'error',
            'correlation_results': self._generate_insufficient_data_response(),
            'risk_metrics': {'correlation_risk_score': 0.5, 'severity_level': 'error'},
            'thesis': f"Correlation analysis failed: {error_context}",
            'recommendations': ["Investigate correlation analysis system errors"]
        }
    
    def _generate_analysis_error_response(self, error_context: str) -> Dict[str, Any]:
        """Generate response when analysis fails"""
        return {
            'correlation_matrix': {},
            'cluster_analysis': {'error': error_context},
            'diversification_metrics': {'diversification_score': 0.5},
            'violation_analysis': {'violations': {'critical': [], 'warning': [], 'info': []}, 'total_violations': 0},
            'regime_analysis': {'current_regime': 'error'},
            'instruments_analyzed': [],
            'processing_time_ms': 0,
            'status': 'analysis_error',
            'error': error_context
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete module state for hot-reload"""
        return {
            'correlation_matrix': self.correlation_matrix,
            'correlation_risk_score': self.correlation_risk_score,
            'diversification_score': self.diversification_score,
            'severity_level': self.severity_level,
            'correlation_clusters': self.correlation_clusters,
            'step_count': self.step_count,
            'correlation_violations': self.correlation_violations,
            'diversification_violations': self.diversification_violations,
            'config': self.config.copy()
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set module state for hot-reload"""
        self.correlation_matrix = state.get('correlation_matrix', {})
        self.correlation_risk_score = state.get('correlation_risk_score', 0.0)
        self.diversification_score = state.get('diversification_score', 1.0)
        self.severity_level = state.get('severity_level', 'normal')
        self.correlation_clusters = state.get('correlation_clusters', {})
        self.step_count = state.get('step_count', 0)
        self.correlation_violations = state.get('correlation_violations', 0)
        self.diversification_violations = state.get('diversification_violations', 0)
        self.config.update(state.get('config', {}))
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get health metrics for monitoring"""
        return {
            'correlation_risk_score': self.correlation_risk_score,
            'diversification_score': self.diversification_score,
            'severity_level': self.severity_level,
            'instruments_tracked': len(self.price_history),
            'correlation_violations': self.correlation_violations,
            'diversification_violations': self.diversification_violations,
            'cluster_count': len(self.correlation_clusters),
            'enabled': self.enabled
        }