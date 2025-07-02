# ─────────────────────────────────────────────────────────────
# File: modules/features/advanced_feature_engine.py
# Enhanced with new infrastructure - 75% less boilerplate!
# ─────────────────────────────────────────────────────────────

import numpy as np
from typing import Union, List, Dict, Any, Optional
from collections import deque

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class AdvancedFeatureEngine(Module, AnalysisMixin):

    def __init__(self, window_sizes: List[int] = [7, 14, 28], debug: bool = True, **kwargs):
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=200,
            **kwargs
        )
        super().__init__(config)
        
        # Module-specific configuration
        self.windows = sorted(window_sizes)
        self.out_dim = len(self.windows) * 4 + 4
        self.max_buffer_size = max(self.windows) + 10
        
        # Enhanced state initialization
        self._initialize_module_state()
        
        self.log_operator_info(
            "Advanced feature engine initialized",
            windows=self.windows,
            output_dim=self.out_dim,
            buffer_size=self.max_buffer_size
        )

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        
        # Feature engine specific state
        self.last_feats = np.zeros(self.out_dim, dtype=np.float32)
        self.price_buffer = deque(maxlen=self.max_buffer_size)
        
        # Enhanced tracking
        self._feature_history = deque(maxlen=100)
        self._price_stats = {'count': 0, 'avg': 0.0, 'std': 0.0}
        self._feature_quality_score = 100.0
        self._invalid_price_count = 0

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Module-specific reset
        self.last_feats.fill(0.0)
        self.price_buffer.clear()
        self._feature_history.clear()
        self._price_stats = {'count': 0, 'avg': 0.0, 'std': 0.0}
        self._feature_quality_score = 100.0
        self._invalid_price_count = 0

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract prices from InfoBus or kwargs
        prices = self._extract_prices_from_sources(info_bus, kwargs)
        
        if prices is not None and len(prices) > 0:
            self._update_buffer_enhanced(prices)
            # Update feature extraction automatically happens in transform()
        else:
            self.log_operator_warning("No valid prices available for feature extraction")

    def _extract_prices_from_sources(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Optional[List[float]]:
        """Extract prices from InfoBus or kwargs with comprehensive fallback"""
        
        prices = []
        
        # Try InfoBus first
        if info_bus:
            # Get from current prices
            current_prices = info_bus.get('prices', {})
            if current_prices:
                prices.extend(current_prices.values())
            
            # Get from recent trades
            recent_trades = info_bus.get('recent_trades', [])
            for trade in recent_trades:
                if 'price' in trade:
                    prices.append(trade['price'])
            
            # Get from features if available
            features = info_bus.get('features', {})
            for key, feature_data in features.items():
                if 'price' in key.lower() and isinstance(feature_data, (list, np.ndarray)):
                    try:
                        prices.extend(np.asarray(feature_data).flatten())
                    except:
                        pass
        
        # Try kwargs (backward compatibility)
        for key in ("price", "prices", "close", "price_series"):
            if key in kwargs:
                src = kwargs[key]
                if isinstance(src, (float, int)):
                    prices.append(src)
                elif isinstance(src, (list, np.ndarray)):
                    try:
                        prices.extend(np.asarray(src).flatten())
                    except:
                        pass
        
        # Filter valid prices
        valid_prices = []
        for p in prices:
            if isinstance(p, (int, float)) and np.isfinite(p) and p > 0:
                valid_prices.append(float(p))
            else:
                self._invalid_price_count += 1
        
        return valid_prices if valid_prices else None

    def step(self, **kwargs):
        """Backward compatibility wrapper"""
        self._step_impl(None, **kwargs)

    def transform(self, price_series: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Enhanced transform with comprehensive validation and monitoring"""
        
        try:
            # Input validation and preprocessing
            prices = self._preprocess_price_series(price_series)
            
            # Update buffer and statistics
            if len(prices) > 0:
                self._update_buffer_enhanced(prices)
                self._update_price_statistics(prices)
            
            # Generate features with enhanced error handling
            features = self._extract_features_robust(prices)
            
            # Quality assessment and monitoring
            quality_score = self._assess_feature_quality(features)
            self._update_performance_metric('feature_quality', quality_score)
            
            # Store results
            self.last_feats = features
            self._feature_history.append({
                'timestamp': np.datetime64('now').astype(str),
                'features': features.copy(),
                'input_size': len(prices),
                'quality_score': quality_score
            })
            
            return self.last_feats
            
        except Exception as e:
            self.log_operator_error(f"Feature extraction failed: {e}")
            # Return safe fallback
            return self._get_fallback_features()

    def _preprocess_price_series(self, price_series: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Enhanced preprocessing with validation"""
        
        prices = np.asarray(price_series, dtype=np.float32)
        
        # Remove invalid values
        mask = np.isfinite(prices) & (prices > 0)
        valid_prices = prices[mask]
        
        if valid_prices.size == 0:
            # Fallback strategies
            if len(self.price_buffer) > 0:
                valid_prices = np.array(list(self.price_buffer)[-28:], dtype=np.float32)
                self.log_operator_info("Using buffered prices for feature extraction")
            else:
                valid_prices = self._generate_synthetic_prices_enhanced(30)
                self.log_operator_warning("Using synthetic prices - check data feed")
        
        return valid_prices

    def _extract_features_robust(self, prices: np.ndarray) -> np.ndarray:
        """Enhanced feature extraction with robust error handling"""
        
        features = []
        
        try:
            # Per-window features with enhanced error handling
            for window_size in self.windows:
                window_features = self._extract_window_features(prices, window_size)
                features.extend(window_features)
            
            # Global features with enhanced calculations
            global_features = self._extract_global_features(prices)
            features.extend(global_features)
            
        except Exception as e:
            self.log_operator_error(f"Feature extraction error: {e}")
            # Return zeros for safety
            features = [0.0] * self.out_dim
        
        # Ensure correct dimensions and clean values
        features = np.asarray(features, dtype=np.float32)
        if features.size < self.out_dim:
            features = np.pad(features, (0, self.out_dim - features.size))
        else:
            features = features[:self.out_dim]
        
        # Clean up invalid values
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return features

    def _extract_window_features(self, prices: np.ndarray, window_size: int) -> List[float]:
        """Extract features for a specific window with enhanced calculations"""
        
        # Get window data with padding if needed
        if prices.size >= window_size:
            window = prices[-window_size:]
        else:
            # Pad with mean if insufficient data
            mean_price = prices.mean() if prices.size > 0 else 1.0
            padding = np.full(window_size - prices.size, mean_price)
            window = np.concatenate([padding, prices])
        
        features = []
        
        try:
            # 1. Enhanced volatility calculation
            if window.size > 4:
                returns = np.diff(np.log(window + 1e-8))
                vol = np.std(returns) * np.sqrt(252)  # Annualized
            else:
                vol = 0.01
            features.append(float(vol))
            
            # 2. Enhanced return calculation
            if window[0] > 0:
                ret = (window[-1] - window[0]) / window[0]
            else:
                ret = 0.0
            features.append(float(ret))
            
            # 3. Enhanced mean-reversion signal
            mean_price = window.mean()
            if mean_price > 0:
                mean_rev = (window[-1] - mean_price) / mean_price
                # Add momentum component
                if window.size >= 3:
                    recent_trend = (window[-1] - window[-3]) / window[-3] if window[-3] > 0 else 0
                    mean_rev = mean_rev - 0.3 * recent_trend
            else:
                mean_rev = 0.0
            features.append(float(mean_rev))
            
            # 4. Enhanced trend strength (slope with confidence)
            if window.size > 1:
                x = np.arange(window.size, dtype=np.float32)
                if window[0] > 0:
                    normalized_prices = window / window[0]
                    if np.std(normalized_prices) > 1e-6:
                        slope, residuals = np.polyfit(x, normalized_prices, 1, full=True)[:2]
                        # Weight slope by R-squared approximation
                        r_squared = 1 - (residuals[0] / np.var(normalized_prices)) if len(residuals) > 0 and np.var(normalized_prices) > 0 else 0
                        slope = slope[0] * r_squared
                    else:
                        slope = 0.0
                else:
                    slope = 0.0
            else:
                slope = 0.0
            features.append(float(slope))
            
        except Exception as e:
            self.log_operator_warning(f"Window feature extraction error for size {window_size}: {e}")
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        return features

    def _extract_global_features(self, prices: np.ndarray) -> List[float]:
        """Extract global features with enhanced calculations"""
        
        features = []
        
        try:
            if prices.size > 1:
                # 1. Enhanced spread calculation
                recent_prices = prices[-10:] if prices.size >= 10 else prices
                if recent_prices.size > 1:
                    log_returns = np.diff(np.log(recent_prices + 1e-8))
                    spread = np.mean(np.abs(log_returns))
                else:
                    spread = 0.0
                features.append(float(spread))
                
                # 2. Enhanced momentum with multiple timeframes
                if prices.size >= 5:
                    short_momentum = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] > 0 else 0.0
                    medium_momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0.0
                    # Combine with recency weighting
                    momentum = 0.7 * short_momentum + 0.3 * medium_momentum
                else:
                    momentum = 0.0
                features.append(float(momentum))
                
                # 3. Enhanced volatility ratio with regime detection
                short_window = min(7, prices.size)
                long_window = min(21, prices.size)
                
                if prices.size >= short_window:
                    short_returns = np.diff(np.log(prices[-short_window:] + 1e-8))
                    short_vol = np.std(short_returns) if short_returns.size > 1 else 0.01
                else:
                    short_vol = 0.01
                
                if prices.size >= long_window:
                    long_returns = np.diff(np.log(prices[-long_window:] + 1e-8))
                    long_vol = np.std(long_returns) if long_returns.size > 1 else 0.01
                else:
                    long_vol = short_vol
                
                vol_ratio = short_vol / (long_vol + 1e-8)
                features.append(float(vol_ratio))
                
                # 4. Enhanced price position with support/resistance
                lookback = min(20, prices.size)
                recent_prices = prices[-lookback:]
                high = recent_prices.max()
                low = recent_prices.min()
                
                if high > low:
                    # Basic price position
                    price_pos = (prices[-1] - low) / (high - low)
                    
                    # Add support/resistance bias
                    mid_point = (high + low) / 2
                    if prices[-1] > mid_point:
                        # Near resistance - adjust upward
                        price_pos = min(1.0, price_pos * 1.1)
                    else:
                        # Near support - adjust downward  
                        price_pos = max(0.0, price_pos * 0.9)
                else:
                    price_pos = 0.5
                
                features.append(float(price_pos))
                
            else:
                # Fallback for insufficient data
                features.extend([0.0, 0.0, 1.0, 0.5])
                
        except Exception as e:
            self.log_operator_warning(f"Global feature extraction error: {e}")
            features.extend([0.0, 0.0, 1.0, 0.5])
        
        return features

    def _update_buffer_enhanced(self, prices: Union[List[float], np.ndarray]):
        """Enhanced buffer management with statistics tracking"""
        
        valid_count = 0
        for p in prices:
            if np.isfinite(p) and p > 0:
                self.price_buffer.append(float(p))
                valid_count += 1
        
        if valid_count > 0:
            self._update_performance_metric('prices_processed', valid_count)
        
        if len(prices) - valid_count > 0:
            self._update_performance_metric('invalid_prices', len(prices) - valid_count)

    def _update_price_statistics(self, prices: np.ndarray):
        """Update rolling price statistics"""
        if prices.size > 0:
            self._price_stats['count'] += prices.size
            # Running average update
            alpha = 0.1  # Learning rate
            current_avg = prices.mean()
            self._price_stats['avg'] = (1 - alpha) * self._price_stats['avg'] + alpha * current_avg
            self._price_stats['std'] = (1 - alpha) * self._price_stats['std'] + alpha * prices.std()

    def _assess_feature_quality(self, features: np.ndarray) -> float:
        """Assess the quality of extracted features"""
        
        quality_score = 100.0
        
        # Check for invalid values
        invalid_count = np.sum(~np.isfinite(features))
        if invalid_count > 0:
            quality_score -= invalid_count * 10
        
        # Check for constant features (low information)
        if features.size > 1:
            non_zero_std = np.std(features[features != 0])
            if non_zero_std < 1e-6:
                quality_score -= 20
        
        # Check buffer utilization
        buffer_utilization = len(self.price_buffer) / self.max_buffer_size
        if buffer_utilization < 0.3:
            quality_score -= 15
        
        # Update running quality score
        alpha = 0.1
        self._feature_quality_score = (1 - alpha) * self._feature_quality_score + alpha * quality_score
        
        return quality_score

    def _get_fallback_features(self) -> np.ndarray:
        """Get safe fallback features when extraction fails"""
        
        if len(self._feature_history) > 0:
            # Use last known good features
            return self._feature_history[-1]['features'].copy()
        else:
            # Generate neutral features
            return np.zeros(self.out_dim, dtype=np.float32)

    @staticmethod
    def _generate_synthetic_prices_enhanced(n: int = 30) -> np.ndarray:
        """Enhanced synthetic price generation with realistic patterns"""
        
        base = 1.0
        # More realistic returns with regime changes
        regime_length = max(5, n // 3)
        returns = []
        
        for i in range(n):
            if i % regime_length == 0:
                # Change regime
                volatility = np.random.choice([0.005, 0.01, 0.02], p=[0.6, 0.3, 0.1])
                drift = np.random.choice([-0.001, 0.0, 0.001], p=[0.2, 0.6, 0.2])
            
            ret = np.random.normal(drift, volatility)
            returns.append(ret)
        
        returns = np.array(returns, dtype=np.float32)
        prices = base * np.exp(np.cumsum(returns))
        
        return prices

    def _get_observation_impl(self) -> np.ndarray:
        """Enhanced observation with feature quality metrics"""
        
        # Ensure features are valid
        obs = np.nan_to_num(self.last_feats, nan=0.0, posinf=1.0, neginf=-1.0).copy()
        
        return obs

    def _check_state_integrity(self) -> bool:
        """Enhanced health check for feature engine"""
        try:
            # Check feature dimensions
            if self.last_feats.size != self.out_dim:
                return False
            
            # Check buffer is reasonable
            if len(self.price_buffer) > self.max_buffer_size * 1.1:
                return False
            
            # Check feature quality
            if self._feature_quality_score < 30:
                return False
            
            # Check for excessive invalid prices
            total_processed = self._get_performance_metric('prices_processed', 1)
            invalid_ratio = self._invalid_price_count / max(total_processed, 1)
            if invalid_ratio > 0.5:
                return False
            
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details with feature-specific metrics"""
        base_details = super()._get_health_details()
        
        feature_details = {
            'feature_stats': {
                'output_dim': self.out_dim,
                'buffer_size': len(self.price_buffer),
                'buffer_utilization': len(self.price_buffer) / self.max_buffer_size,
                'feature_quality_score': self._feature_quality_score,
                'invalid_price_ratio': self._invalid_price_count / max(self._get_performance_metric('prices_processed', 1), 1)
            },
            'price_stats': self._price_stats.copy(),
            'feature_history_size': len(self._feature_history),
            'window_config': self.windows
        }
        
        if base_details:
            base_details.update(feature_details)
            return base_details
        
        return feature_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        return {
            'last_feats': self.last_feats.tolist(),
            'price_buffer': list(self.price_buffer),
            'price_stats': self._price_stats.copy(),
            'feature_quality_score': self._feature_quality_score,
            'invalid_price_count': self._invalid_price_count,
            'feature_history': list(self._feature_history)[-20:]  # Keep recent only
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        self.last_feats = np.array(module_state.get('last_feats', self.last_feats), dtype=np.float32)
        
        buffer_data = module_state.get('price_buffer', [])
        self.price_buffer.clear()
        self.price_buffer.extend(buffer_data)
        
        self._price_stats = module_state.get('price_stats', {'count': 0, 'avg': 0.0, 'std': 0.0})
        self._feature_quality_score = module_state.get('feature_quality_score', 100.0)
        self._invalid_price_count = module_state.get('invalid_price_count', 0)
        
        history_data = module_state.get('feature_history', [])
        self._feature_history.clear()
        self._feature_history.extend(history_data)

    def get_feature_analysis_report(self) -> str:
        """Generate operator-friendly feature analysis report"""
        
        buffer_util = len(self.price_buffer) / self.max_buffer_size
        
        return f"""
📊 FEATURE ENGINE ANALYSIS
═══════════════════════════════════════
🔧 Configuration: {len(self.windows)} windows, {self.out_dim}D output
📈 Buffer Status: {len(self.price_buffer)}/{self.max_buffer_size} ({buffer_util:.1%} utilized)
⭐ Quality Score: {self._feature_quality_score:.1f}/100

📋 PRICE STATISTICS
• Processed Count: {self._price_stats['count']}
• Average Price: {self._price_stats['avg']:.5f}
• Price Volatility: {self._price_stats['std']:.5f}
• Invalid Price Ratio: {(self._invalid_price_count / max(self._price_stats['count'], 1)):.1%}

🔍 FEATURE METRICS
• Feature History: {len(self._feature_history)} snapshots
• Last Feature Norm: {np.linalg.norm(self.last_feats):.4f}
• Window Sizes: {self.windows}

🎯 RECENT PERFORMANCE
• Quality Trend: {'📈 Improving' if self._feature_quality_score > 80 else '📉 Degrading' if self._feature_quality_score < 60 else '➡️ Stable'}
• Data Availability: {'✅ Good' if buffer_util > 0.5 else '⚠️ Limited' if buffer_util > 0.2 else '❌ Poor'}
        """
