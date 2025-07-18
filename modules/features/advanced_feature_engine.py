# ─────────────────────────────────────────────────────────────
# File: modules/features/advanced_feature_engine.py  
# [ROCKET] PRODUCTION-GRADE Advanced Feature Engine
# NASA/MILITARY GRADE - ZERO ERROR TOLERANCE
# ENHANCED: Complete SmartInfoBus integration with all advanced features
# ─────────────────────────────────────────────────────────────

import time
import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Union
from collections import deque
from dataclasses import dataclass

# Core infrastructure
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class FeatureEngineConfig:
    """Configuration for Advanced Feature Engine"""
    window_sizes: Optional[List[int]] = None
    max_buffer_size: int = 1000
    enable_neural_processing: bool = True
    enable_health_monitoring: bool = True
    enable_performance_tracking: bool = True
    enable_error_pinpointing: bool = True
    enable_english_explanations: bool = True
    circuit_breaker_threshold: int = 5
    
    def __post_init__(self):
        if self.window_sizes is None:
            self.window_sizes = [7, 14, 28, 56]


@module(
    name="AdvancedFeatureEngine",
    version="3.0.0",
    category="features",
    provides=[
        "advanced_features", "feature_analysis", "feature_health", "feature_thesis",
        "features", "technical_indicators", "market_features", "price_features"
    ],
    requires=["market_data", "price_data"],
    description="Advanced feature extraction with comprehensive SmartInfoBus integration",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class AdvancedFeatureEngine(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    [ROCKET] PRODUCTION-GRADE Advanced Feature Engine
    
    FEATURES:
    - Complete SmartInfoBus integration with thesis generation
    - ErrorPinpointer for advanced error analysis
    - English explanations for all decisions
    - Health monitoring with circuit breakers
    - Performance tracking and optimization
    - State management for hot-reload
    - Comprehensive validation and auditing
    """
    
    def __init__(self, config: Optional[Union[FeatureEngineConfig, Dict[str, Any]]] = None, **kwargs):
        # Store config first 
        if isinstance(config, dict):
            # Convert dict config to FeatureEngineConfig
            self.feature_config = FeatureEngineConfig(**{k: v for k, v in config.items() if k in FeatureEngineConfig.__dataclass_fields__})
        else:
            self.feature_config = config or FeatureEngineConfig()
        
        self.config = self.feature_config  # Set config early for init methods
        
        # Initialize all systems before super().__init__() 
        # because BaseModule calls _initialize() which needs these attributes
        self._initialize_advanced_systems()
        
        # Feature-specific initialization
        self.window_sizes = sorted(self.config.window_sizes or [7, 14, 28, 56])
        self.out_dim = len(self.window_sizes) * 6 + 8  # Enhanced feature set
        self.max_buffer_size = self.config.max_buffer_size
        
        # Initialize state
        self._initialize_feature_state()
        
        # Start monitoring
        self._start_monitoring()
        
        super().__init__(**kwargs)  # Don't pass config to BaseModule
        
        # Ensure our config is preserved after BaseModule initialization
        self.config = self.feature_config
        
        self.logger.info(
            format_operator_message(
                "[ROCKET]", "ADVANCED_FEATURE_ENGINE_INITIALIZED",
                details=f"Windows: {self.window_sizes}, Output dim: {self.out_dim}",
                result="Production-grade feature engine active",
                context="feature_engine_startup"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize all advanced systems"""
        # Core systems
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="AdvancedFeatureEngine",
            log_path="logs/features/advanced_engine.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        
        # Advanced systems
        if self.config.enable_error_pinpointing:
            self.error_pinpointer = ErrorPinpointer()
            self.error_handler = create_error_handler("AdvancedFeatureEngine", self.error_pinpointer)
        
        if self.config.enable_english_explanations:
            self.english_explainer = EnglishExplainer()
            self.system_utilities = SystemUtilities()
        
        if self.config.enable_performance_tracking:
            self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker state
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',  # CLOSED, OPEN, HALF_OPEN
            'threshold': self.config.circuit_breaker_threshold
        }
    
    def _initialize_feature_state(self):
        """Initialize feature-specific state"""
        # Feature buffers
        self.price_buffer = deque(maxlen=self.max_buffer_size)
        self.feature_buffer = deque(maxlen=1000)
        
        # Feature outputs
        self.last_features = np.zeros(self.out_dim, dtype=np.float32)
        self.feature_quality_score = 100.0
        
        # Statistics
        self.feature_stats = {
            'total_extractions': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'avg_extraction_time_ms': 0.0,
            'avg_feature_quality': 0.0,
            'price_points_processed': 0
        }
        
        # Health tracking
        self.health_metrics = {
            'last_health_check': time.time(),
            'health_score': 100.0,
            'issues_detected': [],
            'performance_trend': 'stable'
        }
    
    def _start_monitoring(self):
        """Start background monitoring tasks"""
        # Only start monitoring tasks if we're in an async context
        try:
            loop = asyncio.get_running_loop()
            if self.config.enable_health_monitoring:
                loop.create_task(self._health_monitoring_loop())
            
            if self.config.enable_performance_tracking:
                loop.create_task(self._performance_monitoring_loop())
        except RuntimeError:
            # No event loop running, monitoring will start when module is initialized
            pass
    
    def _initialize(self):
        """Initialize module - called by orchestrator"""
        super()._initialize()
        
        # Store module capabilities in SmartInfoBus
        self.smart_bus.set(
            'feature_engine_capabilities',
            {
                'window_sizes': self.window_sizes,
                'output_dimensions': self.out_dim,
                'max_buffer_size': self.max_buffer_size,
                'supports_neural_processing': self.feature_config.enable_neural_processing,
                'features_available': ['price_momentum', 'volatility', 'trend_strength', 'volume_profile']
            },
            module='AdvancedFeatureEngine',
            thesis="Advanced feature engine capabilities for system optimization"
        )
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """Main processing function with full error handling and monitoring"""
        
        process_start_time = time.time()
        
        # Check circuit breaker
        if not self._check_circuit_breaker():
            return self._create_fallback_response("Circuit breaker open")
        
        try:
            # Extract market data
            market_data = await self._extract_market_data(**inputs)
            
            # Process features with monitoring
            features = await self._process_features_with_monitoring(market_data)
            
            # Generate thesis
            thesis = await self._generate_feature_thesis(features, market_data)
            
            # Update SmartInfoBus
            await self._update_smart_bus(features, thesis)
            
            # Record success
            self._record_success(time.time() - process_start_time)
            
            return {
                'success': True,
                'advanced_features': features,  # Fixed: Use 'advanced_features' to match provides declaration
                'features': features,  # Keep for backward compatibility
                'thesis': thesis,
                'quality_score': self.feature_quality_score,
                'processing_time_ms': (time.time() - process_start_time) * 1000
            }
            
        except Exception as e:
            return await self._handle_processing_error(e, process_start_time)
    
    async def _extract_market_data(self, **inputs) -> Dict[str, Any]:
        """Extract market data from multiple sources"""
        
        market_data = {
            'prices': [],
            'volumes': [],
            'timestamps': [],
            'instruments': []
        }
        
        # Extract from SmartInfoBus
        bus_data = self.smart_bus.get('market_data', self.__class__.__name__)
        if bus_data and isinstance(bus_data, dict):
            for key, value in bus_data.items():
                if 'price' in key.lower() and isinstance(value, (list, np.ndarray)):
                    market_data['prices'].extend(np.asarray(value).flatten())
                elif 'volume' in key.lower() and isinstance(value, (list, np.ndarray)):
                    market_data['volumes'].extend(np.asarray(value).flatten())
        
        # Extract from direct inputs
        for key, value in inputs.items():
            if key in ['price', 'prices', 'close', 'price_series'] and value is not None:
                if isinstance(value, (list, np.ndarray)):
                    market_data['prices'].extend(np.asarray(value).flatten())
                elif isinstance(value, (int, float)):
                    market_data['prices'].append(float(value))
        
        # Validate and clean
        market_data['prices'] = self._validate_prices(market_data['prices'])
        
        if not market_data['prices']:
            raise ValueError("No valid price data available")
        
        return market_data
    
    def _validate_prices(self, prices: List[float]) -> List[float]:
        """Validate and clean price data"""
        valid_prices = []
        
        for price in prices:
            if isinstance(price, (int, float)) and np.isfinite(price) and price > 0:
                valid_prices.append(float(price))
        
        # Remove outliers (beyond 3 standard deviations)
        if len(valid_prices) > 10:
            prices_array = np.array(valid_prices)
            mean_price = np.mean(prices_array)
            std_price = np.std(prices_array)
            
            valid_prices = [
                p for p in valid_prices 
                if abs(p - mean_price) <= 3 * std_price
            ]
        
        return valid_prices
    
    async def _process_features_with_monitoring(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process features with comprehensive monitoring"""
        
        extraction_start = time.time()
        
        # Update price buffer
        self.price_buffer.extend(market_data['prices'])
        
        # Extract features
        features = self._extract_comprehensive_features(market_data['prices'])
        
        # Calculate quality score
        quality_score = self._calculate_feature_quality(features)
        self.feature_quality_score = quality_score
        
        # Store features
        self.last_features = features
        self.feature_buffer.append({
            'features': features.copy(),
            'quality_score': quality_score,
            'timestamp': time.time()
        })
        
        # Update statistics
        extraction_time = (time.time() - extraction_start) * 1000
        self._update_feature_stats(extraction_time, quality_score)
        
        # Generate English explanation
        explanation = self._generate_feature_explanation(features, quality_score)
        
        return {
            'raw_features': features,
            'quality_score': quality_score,
            'explanation': explanation,
            'extraction_time_ms': extraction_time,
            'buffer_size': len(self.price_buffer),
            'feature_count': len(features)
        }
    
    def _extract_comprehensive_features(self, prices: List[float]) -> np.ndarray:
        """Extract comprehensive feature set"""
        
        if len(prices) < max(self.window_sizes):
            return self._get_fallback_features()
        
        prices_array = np.array(prices[-max(self.window_sizes):])
        features = []
        
        # Multi-window features
        for window in self.window_sizes:
            if len(prices_array) >= window:
                window_prices = prices_array[-window:]
                
                # Basic features
                features.extend([
                    np.mean(window_prices),  # Mean price
                    np.std(window_prices),   # Volatility
                    (window_prices[-1] - window_prices[0]) / window_prices[0],  # Return
                    np.max(window_prices) - np.min(window_prices),  # Range
                    np.mean(np.diff(window_prices) > 0),  # Trend strength
                    len(window_prices)  # Window size
                ])
            else:
                features.extend([0.0] * 6)
        
        # Global features
        features.extend([
            prices_array[-1],  # Current price
            np.mean(prices_array),  # Overall mean
            np.std(prices_array),   # Overall volatility
            np.max(prices_array),   # Max price
            np.min(prices_array),   # Min price
            len(prices_array),      # Data points
            time.time(),            # Timestamp
            self.feature_quality_score / 100.0  # Quality indicator
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _calculate_feature_quality(self, features: np.ndarray) -> float:
        """Calculate feature quality score"""
        
        try:
            # Check for invalid values
            if np.any(~np.isfinite(features)):
                return 0.0
            
            # Check for zero variance
            if np.std(features) == 0:
                return 30.0
            
            # Check for reasonable ranges
            if np.max(features) - np.min(features) < 1e-6:
                return 40.0
            
            # Quality indicators
            quality_score = 100.0
            
            # Penalize extreme values
            if np.max(np.abs(features)) > 1e6:
                quality_score -= 20
            
            # Reward good variance
            feature_std = np.std(features)
            if 0.1 < feature_std < 100:
                quality_score += 10
            
            return max(0.0, min(100.0, quality_score))
            
        except Exception:
            return 0.0
    
    def _generate_feature_explanation(self, features: np.ndarray, quality_score: float) -> str:
        """Generate English explanation of features"""
        
        if not self.config.enable_english_explanations:
            return "Feature extraction completed"
        
        try:
            # Analyze features
            feature_analysis = {
                'feature_count': len(features),
                'quality_score': quality_score,
                'max_value': float(np.max(features)),
                'min_value': float(np.min(features)),
                'mean_value': float(np.mean(features)),
                'std_value': float(np.std(features)),
                'window_sizes': self.window_sizes,
                'buffer_size': len(self.price_buffer)
            }
            
            # Generate explanation
            explanation = self.english_explainer.explain_module_decision(
                module_name="AdvancedFeatureEngine",
                decision="feature_extraction",
                context=feature_analysis,
                confidence=quality_score / 100.0
            )
            
            return explanation
            
        except Exception as e:
            return f"Feature extraction completed (explanation generation failed: {str(e)})"
    
    async def _generate_feature_thesis(self, features: Dict[str, Any], market_data: Dict[str, Any]) -> str:
        """Generate thesis for feature extraction"""
        
        try:
            # Analyze market conditions
            prices = market_data['prices']
            latest_price = prices[-1] if prices else 0
            price_change = (prices[-1] - prices[0]) / prices[0] if len(prices) > 1 else 0
            
            # Feature analysis
            feature_quality = features['quality_score']
            feature_count = features['feature_count']
            
            # Generate thesis
            thesis = f"""
Advanced Feature Analysis:

Market Assessment:
- Current price: ${latest_price:.2f}
- Price change: {price_change:.2%}
- Data points processed: {len(prices)}

Feature Quality:
- Quality score: {feature_quality:.1f}/100
- Features extracted: {feature_count}
- Buffer utilization: {len(self.price_buffer)}/{self.max_buffer_size}

Technical Indicators:
- Multi-timeframe analysis across {len(self.window_sizes)} windows
- Window sizes: {self.window_sizes}
- Comprehensive feature set including momentum, volatility, and trend strength

Confidence Assessment:
- Feature extraction: {'High' if feature_quality > 80 else 'Medium' if feature_quality > 60 else 'Low'}
- Data quality: {'Good' if len(prices) > 50 else 'Adequate' if len(prices) > 20 else 'Limited'}
- System health: {'Optimal' if self.health_metrics['health_score'] > 90 else 'Good' if self.health_metrics['health_score'] > 70 else 'Needs attention'}

Recommendation: {'Continue processing' if feature_quality > 60 else 'Review data quality'}
            """.strip()
            
            return thesis
            
        except Exception as e:
            return f"Feature extraction completed. Thesis generation encountered error: {str(e)}"
    
    async def _update_smart_bus(self, features: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with results"""
        
        # Main feature data
        self.smart_bus.set(
            'advanced_features',
            {
                'features': features['raw_features'].tolist(),
                'quality_score': features['quality_score'],
                'extraction_time_ms': features['extraction_time_ms'],
                'timestamp': time.time()
            },
            module='AdvancedFeatureEngine',
            thesis=thesis
        )
        
        # Feature analysis
        self.smart_bus.set(
            'feature_analysis',
            {
                'explanation': features['explanation'],
                'buffer_status': {
                    'current_size': len(self.price_buffer),
                    'max_size': self.max_buffer_size,
                    'utilization': len(self.price_buffer) / self.max_buffer_size
                },
                'statistics': self.feature_stats,
                'health_metrics': self.health_metrics
            },
            module='AdvancedFeatureEngine',
            thesis=f"Feature analysis summary: {features['quality_score']:.1f}% quality"
        )
        
        # Health status
        self.smart_bus.set(
            'feature_health',
            {
                'health_score': self.health_metrics['health_score'],
                'circuit_breaker_state': self.circuit_breaker['state'],
                'issues_detected': self.health_metrics['issues_detected'],
                'performance_trend': self.health_metrics['performance_trend']
            },
            module='AdvancedFeatureEngine',
            thesis=f"Feature engine health: {self.health_metrics['health_score']:.1f}%"
        )
    
    def _update_feature_stats(self, extraction_time_ms: float, quality_score: float):
        """Update feature extraction statistics"""
        
        self.feature_stats['total_extractions'] += 1
        self.feature_stats['successful_extractions'] += 1
        
        # Update averages
        total = self.feature_stats['total_extractions']
        self.feature_stats['avg_extraction_time_ms'] = (
            (self.feature_stats['avg_extraction_time_ms'] * (total - 1) + extraction_time_ms) / total
        )
        self.feature_stats['avg_feature_quality'] = (
            (self.feature_stats['avg_feature_quality'] * (total - 1) + quality_score) / total
        )
    
    def _check_circuit_breaker(self) -> bool:
        """Check circuit breaker state"""
        
        if self.circuit_breaker['state'] == 'OPEN':
            # Check if we should try half-open
            if time.time() - self.circuit_breaker['last_failure'] > 60:  # 1 minute recovery
                self.circuit_breaker['state'] = 'HALF_OPEN'
                return True
            return False
        
        return True
    
    def _record_success(self, processing_time: float):
        """Record successful operation"""
        
        if self.circuit_breaker['state'] == 'HALF_OPEN':
            self.circuit_breaker['state'] = 'CLOSED'
            self.circuit_breaker['failures'] = 0
        
        # Update health metrics
        self.health_metrics['health_score'] = min(100.0, self.health_metrics['health_score'] + 1)
        self.health_metrics['performance_trend'] = 'improving'
        
        # Performance tracking
        if hasattr(self, 'performance_tracker'):
            self.performance_tracker.record_metric(
                'AdvancedFeatureEngine',
                'feature_extraction',
                processing_time * 1000,
                True
            )
    
    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with comprehensive analysis"""
        
        processing_time = time.time() - start_time
        
        # Record failure
        self._record_failure(error)
        
        # Error analysis
        if hasattr(self, 'error_pinpointer'):
            error_context = self.error_pinpointer.analyze_error(error, "AdvancedFeatureEngine")
            
            # Generate debugging guide
            debug_guide = self.error_pinpointer.create_debugging_guide(error_context)
            
            self.logger.error(
                format_operator_message(
                    "[CRASH]", "FEATURE_EXTRACTION_ERROR",
                    details=str(error),
                    context="feature_processing",
                    recovery_actions=len(error_context.recovery_actions)
                )
            )
        
        # Generate fallback response
        fallback_features = self._get_fallback_features()
        fallback_thesis = f"Feature extraction failed: {str(error)}. Using fallback features."
        
        # Update SmartInfoBus with error info
        self.smart_bus.set(
            'feature_error',
            {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'fallback_used': True,
                'timestamp': time.time()
            },
            module='AdvancedFeatureEngine',
            thesis=fallback_thesis
        )
        
        return {
            'success': False,
            'error': str(error),
            'fallback_features': fallback_features,
            'thesis': fallback_thesis,
            'processing_time_ms': processing_time * 1000
        }
    
    def _record_failure(self, error: Exception):
        """Record failure for circuit breaker"""
        
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            
            self.logger.error(
                format_operator_message(
                    "[ALERT]", "CIRCUIT_BREAKER_OPEN",
                    details=f"Too many failures ({self.circuit_breaker['failures']})",
                    context="circuit_breaker"
                )
            )
        
        # Update health metrics
        self.health_metrics['health_score'] = max(0.0, self.health_metrics['health_score'] - 10)
        self.health_metrics['issues_detected'].append(f"{type(error).__name__}: {str(error)}")
        self.health_metrics['performance_trend'] = 'degrading'
        
        # Update stats
        self.feature_stats['failed_extractions'] += 1
    
    def _get_fallback_features(self) -> np.ndarray:
        """Get fallback features when processing fails"""
        
        # Try to use last successful features
        if len(self.feature_buffer) > 0:
            return self.feature_buffer[-1]['features']
        
        # Generate synthetic features
        return np.zeros(self.out_dim, dtype=np.float32)
    
    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response"""
        
        return {
            'success': False,
            'reason': reason,
            'fallback_features': self._get_fallback_features(),
            'thesis': f"Feature extraction unavailable: {reason}",
            'quality_score': 0.0,
            'processing_time_ms': 0.0
        }
    
    async def _health_monitoring_loop(self):
        """Background health monitoring"""
        
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Update health metrics
                self._update_health_metrics()
                
                # Check for issues
                self._check_health_issues()
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def _update_health_metrics(self):
        """Update health metrics"""
        
        current_time = time.time()
        
        # Calculate success rate
        total_ops = self.feature_stats['total_extractions']
        success_rate = (
            self.feature_stats['successful_extractions'] / max(total_ops, 1)
        )
        
        # Update health score based on success rate
        if success_rate > 0.95:
            self.health_metrics['health_score'] = min(100.0, self.health_metrics['health_score'] + 0.5)
        elif success_rate < 0.8:
            self.health_metrics['health_score'] = max(0.0, self.health_metrics['health_score'] - 1.0)
        
        # Update timestamp
        self.health_metrics['last_health_check'] = current_time
    
    def _check_health_issues(self):
        """Check for health issues"""
        
        issues = []
        
        # Check circuit breaker
        if self.circuit_breaker['state'] == 'OPEN':
            issues.append("Circuit breaker is open")
        
        # Check buffer utilization
        buffer_utilization = len(self.price_buffer) / self.max_buffer_size
        if buffer_utilization > 0.9:
            issues.append("Price buffer nearly full")
        
        # Check processing time
        if self.feature_stats['avg_extraction_time_ms'] > 100:
            issues.append("Processing time is high")
        
        # Check feature quality
        if self.feature_stats['avg_feature_quality'] < 60:
            issues.append("Feature quality is low")
        
        # Update issues
        self.health_metrics['issues_detected'] = issues
        
        # Log critical issues
        if issues:
            self.logger.warning(
                format_operator_message(
                    "[WARN]", "HEALTH_ISSUES_DETECTED",
                    details=f"{len(issues)} issues found",
                    context="health_monitoring"
                )
            )
    
    async def _performance_monitoring_loop(self):
        """Background performance monitoring"""
        
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Record performance metrics
                if hasattr(self, 'performance_tracker'):
                    self.performance_tracker.record_metric(
                        'AdvancedFeatureEngine',
                        'periodic_metrics',
                        1.0,  # Dummy duration for metric collection
                        True
                    )
                
            except Exception as e:
                self.logger.error(f"Performance monitoring error: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get complete module state"""
        
        base_state = super().get_state()
        
        feature_state = {
            'config': {
                'window_sizes': self.window_sizes,
                'max_buffer_size': self.max_buffer_size,
                'out_dim': self.out_dim
            },
            'buffers': {
                'price_buffer': list(self.price_buffer),
                'feature_buffer': [fb for fb in self.feature_buffer]
            },
            'features': {
                'last_features': self.last_features.tolist(),
                'quality_score': self.feature_quality_score
            },
            'statistics': self.feature_stats,
            'health_metrics': self.health_metrics,
            'circuit_breaker': self.circuit_breaker
        }
        
        return {**base_state, **feature_state}
    
    def set_state(self, state: Dict[str, Any]):
        """Restore module state"""
        
        super().set_state(state)
        
        # Restore buffers
        if 'buffers' in state:
            if 'price_buffer' in state['buffers']:
                self.price_buffer = deque(state['buffers']['price_buffer'], maxlen=self.max_buffer_size)
            if 'feature_buffer' in state['buffers']:
                self.feature_buffer = deque(state['buffers']['feature_buffer'], maxlen=1000)
        
        # Restore features
        if 'features' in state:
            if 'last_features' in state['features']:
                self.last_features = np.array(state['features']['last_features'], dtype=np.float32)
            if 'quality_score' in state['features']:
                self.feature_quality_score = state['features']['quality_score']
        
        # Restore statistics
        if 'statistics' in state:
            self.feature_stats.update(state['statistics'])
        
        # Restore health metrics
        if 'health_metrics' in state:
            self.health_metrics.update(state['health_metrics'])
        
        # Restore circuit breaker
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        
        return {
            'health_score': self.health_metrics['health_score'],
            'circuit_breaker_state': self.circuit_breaker['state'],
            'issues_detected': self.health_metrics['issues_detected'],
            'performance_trend': self.health_metrics['performance_trend'],
            'statistics': self.feature_stats,
            'buffer_status': {
                'price_buffer_size': len(self.price_buffer),
                'price_buffer_utilization': len(self.price_buffer) / self.max_buffer_size,
                'feature_buffer_size': len(self.feature_buffer)
            }
        }
    
    def get_performance_report(self) -> str:
        """Get comprehensive performance report in plain English"""
        
        if not self.config.enable_english_explanations:
            return "Performance reporting disabled"
        
        try:
            return self.english_explainer.explain_performance(
                module_name="AdvancedFeatureEngine",
                metrics={
                    'total_extractions': self.feature_stats['total_extractions'],
                    'success_rate': self.feature_stats['successful_extractions'] / max(self.feature_stats['total_extractions'], 1),
                    'avg_extraction_time_ms': self.feature_stats['avg_extraction_time_ms'],
                    'avg_feature_quality': self.feature_stats['avg_feature_quality'],
                    'health_score': self.health_metrics['health_score'],
                    'buffer_utilization': len(self.price_buffer) / self.max_buffer_size
                }
            )
        except Exception as e:
            return f"Performance report generation failed: {str(e)}"
    
    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose feature-based action"""
        try:
            # Process features first
            result = await self.process(**inputs)
            
            if not result.get('success', False):
                return {
                    'action_type': 'no_action',
                    'confidence': 0.0,
                    'reasoning': 'Feature extraction failed',
                    'features_available': False
                }
            
            # Analyze features for action proposal
            features = result['features']['raw_features']
            quality_score = result['quality_score']
            
            # Simple feature-based action logic
            if len(features) > 0:
                # Look at recent momentum and volatility
                momentum_indicators = features[:len(self.window_sizes)]  # First window features
                avg_momentum = float(np.mean(momentum_indicators)) if len(momentum_indicators) > 0 else 0.0
                
                # Propose action based on feature analysis
                if quality_score > 80:
                    if avg_momentum > 0.01:  # Positive momentum
                        action_type = "increase_position"
                        magnitude = min(abs(avg_momentum) * 10, 1.0)
                    elif avg_momentum < -0.01:  # Negative momentum  
                        action_type = "decrease_position"
                        magnitude = min(abs(avg_momentum) * 10, 1.0)
                    else:
                        action_type = "hold_position"
                        magnitude = 0.0
                else:
                    action_type = "reduce_risk"
                    magnitude = 0.5
                
                return {
                    'action_type': action_type,
                    'magnitude': magnitude,
                    'confidence': quality_score / 100.0,
                    'reasoning': f"Feature analysis: {len(features)} features, {quality_score:.1f}% quality, momentum: {avg_momentum:.4f}",
                    'features_used': len(features),
                    'quality_score': quality_score,
                    'momentum_signal': avg_momentum
                }
            else:
                return {
                    'action_type': 'no_action',
                    'confidence': 0.0,
                    'reasoning': 'No features available',
                    'features_available': False
                }
                
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                'action_type': 'no_action',
                'confidence': 0.0,
                'reasoning': f'Action proposal error: {str(e)}',
                'error': str(e)
            }
    
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in the proposed action"""
        try:
            if not isinstance(action, dict):
                return 0.0
            
            # Base confidence from feature quality
            base_confidence = self.feature_quality_score / 100.0
            
            # Adjust based on action characteristics
            action_type = action.get('action_type', 'no_action')
            magnitude = action.get('magnitude', 0.0)
            
            # Higher confidence for well-supported actions
            if action_type in ['increase_position', 'decrease_position']:
                # Check if we have sufficient features
                features_used = action.get('features_used', 0)
                if features_used >= len(self.window_sizes):
                    feature_confidence = 1.0
                elif features_used > 0:
                    feature_confidence = features_used / len(self.window_sizes)
                else:
                    feature_confidence = 0.0
                
                # Confidence based on magnitude (higher magnitude needs higher confidence)
                magnitude_confidence = 1.0 - min(abs(magnitude), 0.5)
                
                # Combine confidences
                combined_confidence = (base_confidence * 0.5 + 
                                     feature_confidence * 0.3 + 
                                     magnitude_confidence * 0.2)
            
            elif action_type == 'hold_position':
                # Holding is usually lower risk
                combined_confidence = base_confidence * 0.8
            
            elif action_type == 'reduce_risk':
                # Risk reduction is conservative
                combined_confidence = max(base_confidence, 0.6)
            
            else:  # no_action
                combined_confidence = 0.1
            
            # Adjust based on health metrics
            health_adjustment = self.health_metrics['health_score'] / 100.0
            final_confidence = combined_confidence * health_adjustment
            
            # Adjust based on circuit breaker state
            if self.circuit_breaker['state'] == 'OPEN':
                final_confidence *= 0.1
            elif self.circuit_breaker['state'] == 'HALF_OPEN':
                final_confidence *= 0.5
            
            return float(np.clip(final_confidence, 0.0, 1.0))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.0