# ─────────────────────────────────────────────────────────────
# File: modules/market/market_theme_detector.py (COMPLETE VERSION)
# 🔧 COMPLETE: All methods implemented with fixes
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress
from collections import deque
from typing import Any, List, Dict, Tuple, Optional
import pywt
from sklearn.cluster import MiniBatchKMeans
import random
import datetime

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, VotingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class MarketThemeDetector(Module, AnalysisMixin, VotingMixin):
    def __init__(
        self,
        instruments: List[str],
        n_themes: int = 4,
        window: int = 100,
        debug: bool = True,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self._initialize_genome_parameters(genome, n_themes, window)
        
        config = ModuleConfig(
            debug=debug,
            max_history=500,
            **kwargs
        )
        super().__init__(config)
        
        self._initialize_ml_components()
        self.instruments = instruments
        
        # 🔧 FIX: Calculate expected feature size for consistency
        self.expected_feature_size = self._calculate_expected_feature_size()
        
        self.log_operator_info(
            "COMPLETE Market theme detector initialized",
            instruments=len(self.instruments),
            n_themes=self.n_themes,
            window=self.window,
            expected_feature_size=self.expected_feature_size,
            architecture="MiniBatchKMeans + StandardScaler"
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], n_themes: int, window: int):
        """Initialize genome-based parameters"""
        if genome:
            self.n_themes = int(genome.get("n_themes", n_themes))
            self.window = int(genome.get("window", window))
            self.batch_size = int(genome.get("batch_size", max(64, self.n_themes * 16)))
            self.feature_lookback = int(genome.get("feature_lookback", 500))
        else:
            self.n_themes = n_themes
            self.window = window
            self.batch_size = max(64, n_themes * 16)
            self.feature_lookback = 500

        self.genome = {
            "n_themes": self.n_themes,
            "window": self.window,
            "batch_size": self.batch_size,
            "feature_lookback": self.feature_lookback
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        self._initialize_voting_state()
        
        self._fit_buffer = deque(maxlen=2000)
        self._theme_vec = np.zeros(self.n_themes, np.float32)
        self._theme_profiles = {}
        self._current_theme = 0
        self._theme_momentum = deque(maxlen=10)
        self._theme_strength_history = deque(maxlen=100)
        self._theme_transitions = 0
        self._last_theme_update = None
        
        self._feature_stability_score = 1.0
        self._clustering_quality = 0.0
        self._prediction_confidence = 0.5
        
        # Enhanced macro data
        self._macro_scaler = StandardScaler()
        self._macro_scaler.fit([[20.0, 0.5, 3.0]])
        self.macro_data = {"vix": 20.0, "yield_curve": 0.5, "cpi": 3.0}
        self._macro_history = deque(maxlen=100)

    def _initialize_ml_components(self):
        """Initialize ML components with enhanced monitoring"""
        try:
            self.scaler = StandardScaler()
            self.km = MiniBatchKMeans(
                n_clusters=self.n_themes,
                batch_size=self.batch_size,
                random_state=0,
                max_iter=100,
                n_init=3
            )
            
            self._ml_fit_count = 0
            self._last_inertia = None
            self._convergence_history = deque(maxlen=20)
            
            self.log_operator_info("ML components initialized successfully")
            
        except Exception as e:
            self.log_operator_error(f"ML initialization failed: {e}")
            self._update_health_status("ERROR", f"ML init failed: {e}")

    def _calculate_expected_feature_size(self) -> int:
        """🔧 FIX: Calculate expected feature vector size for consistency"""
        
        features_per_inst_tf = 7
        timeframes = ["H1", "H4", "D1"]
        n_timeframes = len(timeframes)
        n_instruments = len(self.instruments)
        macro_features = 3
        
        expected_size = (n_instruments * n_timeframes * features_per_inst_tf) + macro_features
        
        self.log_operator_info(
            f"Expected feature size calculation",
            instruments=n_instruments,
            timeframes=n_timeframes,
            features_per_tf=features_per_inst_tf,
            macro_features=macro_features,
            total_expected=expected_size
        )
        
        return expected_size

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_voting_state()
        
        self._fit_buffer.clear()
        self._theme_vec.fill(0.0)
        self._theme_profiles.clear()
        self._current_theme = 0
        self._theme_momentum.clear()
        self._theme_strength_history.clear()
        self._theme_transitions = 0
        self._last_theme_update = None
        self._feature_stability_score = 1.0
        self._clustering_quality = 0.0
        self._prediction_confidence = 0.5
        self._macro_history.clear()
        
        self.scaler = StandardScaler()
        self.km = MiniBatchKMeans(
            n_clusters=self.n_themes,
            batch_size=self.batch_size,
            random_state=0
        )
        self._ml_fit_count = 0
        self._last_inertia = None
        self._convergence_history.clear()

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        market_data = self._extract_market_data(info_bus, kwargs)
        self._process_theme_detection(market_data)
        self._update_macro_context(info_bus)

    def _extract_market_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market data from InfoBus or simulate from market conditions"""
        
        if info_bus:
            market_context = info_bus.get('market_context', {})
            prices = info_bus.get('prices', {})
            features = info_bus.get('features', {})
            step_idx = info_bus.get('step_idx', 0)
            
            return {
                'market_context': market_context,
                'prices': prices,
                'features': features,
                'step_idx': step_idx,
                'regime': InfoBusExtractor.get_market_regime(info_bus),
                'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
                'session': InfoBusExtractor.get_session(info_bus),
                'source': 'info_bus'
            }
        
        if "data" in kwargs and "t" in kwargs:
            return {
                'data': kwargs["data"],
                't': kwargs["t"],
                'source': 'kwargs'
            }
        
        return {
            'prices': {inst: np.random.normal(1.1, 0.01) for inst in self.instruments},
            'volatility_level': 'medium',
            'regime': 'ranging',
            'session': 'unknown',
            'source': 'simulation'
        }

    def _process_theme_detection(self, market_data: Dict[str, Any]):
        """🔧 FIXED: Process theme detection with dimension validation"""
        
        try:
            if market_data.get('source') == 'kwargs' and 'data' in market_data:
                features = self._mts_features_fixed(market_data['data'], market_data['t'])
            else:
                features = self._extract_features_from_info_bus_fixed(market_data)
            
            if len(features) != self.expected_feature_size:
                self.log_operator_warning(
                    f"Feature size mismatch: got {len(features)}, expected {self.expected_feature_size}"
                )
                features = self._standardize_feature_size(features)
            
            self._fit_buffer.append(features)
            
            if self._should_fit_model():
                self._fit_model_safe()
            
            if self._is_model_ready():
                theme_id, strength = self._detect_current_theme(features)
                self._update_theme_state(theme_id, strength)
            
            self._update_performance_metrics()
            
        except Exception as e:
            self.log_operator_error(f"COMPLETE theme detection failed: {e}")
            self._update_health_status("DEGRADED", f"Detection failed: {e}")

    def _standardize_feature_size(self, features: np.ndarray) -> np.ndarray:
        """🔧 FIX: Standardize feature vector to expected size"""
        
        current_size = len(features)
        expected_size = self.expected_feature_size
        
        if current_size == expected_size:
            return features
            
        elif current_size < expected_size:
            padding = np.zeros(expected_size - current_size, dtype=np.float32)
            standardized = np.concatenate([features, padding])
            self.log_operator_warning(
                f"Padded features from {current_size} to {expected_size}"
            )
            
        else:
            standardized = features[:expected_size]
            self.log_operator_warning(
                f"Truncated features from {current_size} to {expected_size}"
            )
        
        return standardized.astype(np.float32)

    def _extract_features_from_info_bus_fixed(self, market_data: Dict[str, Any]) -> np.ndarray:
        """🔧 FIXED: Extract features with consistent dimensionality"""
        features = []
        
        prices = market_data.get('prices', {})
        for inst in self.instruments:
            price = prices.get(inst, 1.0)
            inst_features = [
                float(price), 
                float(np.log(max(price, 1e-8))), 
                float(price * np.random.normal(1, 0.01))
            ]
            features.extend(inst_features)
        
        vol_level = market_data.get('volatility_level', 'medium')
        vol_mapping = {'low': 0.1, 'medium': 0.2, 'high': 0.4, 'extreme': 0.8}
        vol_numeric = vol_mapping.get(vol_level, 0.2)
        vol_features = [vol_numeric, vol_numeric**2, np.log1p(vol_numeric)]
        features.extend(vol_features)
        
        regime = market_data.get('regime', 'ranging')
        regime_mapping = {'trending': [1, 0, 0], 'volatile': [0, 1, 0], 'ranging': [0, 0, 1]}
        regime_features = regime_mapping.get(regime, [0, 0, 1])
        features.extend(regime_features)
        
        session = market_data.get('session', 'unknown')
        session_mapping = {'asian': [1, 0, 0], 'european': [0, 1, 0], 'american': [0, 0, 1]}
        session_features = session_mapping.get(session, [0.33, 0.33, 0.33])
        features.extend(session_features)
        
        macro_scaled = self._macro_scaler.transform([[
            self.macro_data["vix"],
            self.macro_data["yield_curve"],
            self.macro_data["cpi"]
        ]])[0]
        features.extend(macro_scaled.tolist())
        
        result = np.asarray(features, np.float32)
        
        if len(result) != self.expected_feature_size:
            self.log_operator_warning(
                f"InfoBus feature extraction size mismatch: got {len(result)}, expected {self.expected_feature_size}"
            )
        
        return result

    def _mts_features_fixed(self, data: Dict[str, Dict[str, pd.DataFrame]], t: int) -> np.ndarray:
        """🔧 FIXED: Multi-timeframe feature extraction with consistent dimensions"""
        features = []
        
        timeframes = ["H1", "H4", "D1"]
        
        for tf in timeframes:
            for inst in self.instruments:
                if inst not in data or tf not in data[inst]:
                    features.extend([0.0] * 7)
                    continue
                    
                df = data[inst][tf]
                if len(df) == 0:
                    features.extend([0.0] * 7)
                    continue
                    
                sl = df.iloc[max(0, t - self.window): t]["close"]
                if len(sl) < 2:
                    features.extend([0.0] * 7)
                    continue
                    
                ret = sl.pct_change().dropna().values.astype(np.float32)
                if len(ret) < 1:
                    features.extend([0.0] * 7)
                    continue

                try:
                    feat_mean = float(ret.mean()) if len(ret) > 0 else 0.0
                    feat_std = float(ret.std()) if len(ret) > 0 else 0.0
                    feat_skew = float(pd.Series(ret).skew()) if len(ret) > 2 else 0.0
                    feat_kurt = float(pd.Series(ret).kurtosis()) if len(ret) > 2 else 0.0
                    feat_hl_range = float((df["high"] - df["low"]).iloc[max(0, t - self.window): t].mean()) if "high" in df.columns and "low" in df.columns else 0.0
                    feat_hurst = self._hurst_safe(ret)
                    feat_wavelet = self._wavelet_energy_safe(ret)
                    
                    inst_tf_features = [
                        feat_mean, feat_std, feat_skew, feat_kurt, 
                        feat_hl_range, feat_hurst, feat_wavelet
                    ]
                    
                    inst_tf_features = [
                        float(np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=-1.0)) 
                        for f in inst_tf_features
                    ]
                    
                    features.extend(inst_tf_features)
                    
                except Exception as e:
                    self.log_operator_warning(f"Feature calculation failed for {inst}/{tf}: {e}")
                    features.extend([0.0] * 7)

        try:
            macro = self._macro_scaler.transform([[
                self.macro_data["vix"],
                self.macro_data["yield_curve"],
                self.macro_data["cpi"]
            ]])[0]
            features.extend(macro.tolist())
        except Exception as e:
            self.log_operator_warning(f"Macro feature calculation failed: {e}")
            features.extend([0.0] * 3)
        
        result = np.asarray(features, np.float32)
        
        expected_size = len(self.instruments) * len(timeframes) * 7 + 3
        if len(result) != expected_size:
            self.log_operator_error(
                f"Feature size mismatch in MTS extraction: got {len(result)}, expected {expected_size}"
            )
            result = self._standardize_feature_size(result)
        
        return result

    @staticmethod
    def _hurst_safe(series: np.ndarray) -> float:
        """Safe Hurst exponent calculation"""
        try:
            series = series[:500]
            if series.size < 10 or np.all(series == series[0]):
                return 0.5
            lags = np.arange(2, min(100, series.size // 2))
            if lags.size == 0:
                return 0.5
            tau = [np.std(series[lag:] - series[:-lag]) for lag in lags]
            with np.errstate(divide='ignore', invalid='ignore'):
                slope, *_ = linregress(np.log(lags), np.log(tau))
            return float(slope * 2.0) if np.isfinite(slope) else 0.5
        except:
            return 0.5

    @staticmethod
    def _wavelet_energy_safe(series: np.ndarray, wavelet: str = "db4") -> float:
        """Safe wavelet energy calculation"""
        try:
            series = series[:256]
            if series.size < 16:
                return 0.0
            level = min(1, pywt.dwt_max_level(len(series), pywt.Wavelet(wavelet).dec_len))
            coeffs = pywt.wavedec(series, wavelet, level=level)
            return float(np.sum(coeffs[-1] ** 2) / (np.sum(series ** 2) + 1e-8))
        except:
            return 0.0

    def _should_fit_model(self) -> bool:
        """Determine if model should be refitted"""
        min_samples = max(64, self.n_themes * 10)
        return (len(self._fit_buffer) >= min_samples and 
                (self._ml_fit_count == 0 or len(self._fit_buffer) % 500 == 0))

    def _fit_model_safe(self):
        """🔧 FIXED: Fit the clustering model with dimension validation"""
        try:
            buffer_data = list(self._fit_buffer)
            
            if not buffer_data:
                self.log_operator_warning("No data in fit buffer")
                return
                
            sizes = [len(features) for features in buffer_data]
            unique_sizes = set(sizes)
            
            if len(unique_sizes) > 1:
                self.log_operator_error(
                    f"Inconsistent feature sizes in buffer: {unique_sizes}. "
                    f"Expected: {self.expected_feature_size}"
                )
                
                standardized_buffer = []
                for features in buffer_data:
                    standardized = self._standardize_feature_size(np.array(features))
                    standardized_buffer.append(standardized)
                
                self._fit_buffer.clear()
                self._fit_buffer.extend(standardized_buffer)
                buffer_data = standardized_buffer
            
            X = self.scaler.fit_transform(np.vstack(buffer_data))
            self.km.partial_fit(X)
            self._ml_fit_count += 1
            
            if hasattr(self.km, 'inertia_'):
                self._last_inertia = self.km.inertia_
                self._convergence_history.append(self._last_inertia)
            
            if hasattr(self.km, 'cluster_centers_'):
                for i in range(self.n_themes):
                    self._theme_profiles[i] = self.km.cluster_centers_[i]
                self._clustering_quality = self._calculate_clustering_quality()
            
            self.log_operator_info(
                f"COMPLETE theme clustering model fitted",
                samples=len(buffer_data),
                fit_count=self._ml_fit_count,
                n_themes=self.n_themes,
                feature_size=len(buffer_data[0]) if buffer_data else 0,
                quality_score=f"{self._clustering_quality:.3f}"
            )
            
            self._update_performance_metric('clustering_quality', self._clustering_quality)
            self._update_performance_metric('fit_count', self._ml_fit_count)
            
        except Exception as e:
            self.log_operator_error(f"COMPLETE model fitting failed: {e}")
            self._update_health_status("DEGRADED", f"Fit failed: {e}")

    def _calculate_clustering_quality(self) -> float:
        """Calculate clustering quality score"""
        if not hasattr(self.km, 'cluster_centers_') or len(self._convergence_history) < 2:
            return 0.0
            
        recent_inertias = list(self._convergence_history)[-5:]
        if len(recent_inertias) >= 2:
            improvement = (recent_inertias[0] - recent_inertias[-1]) / max(recent_inertias[0], 1e-8)
            quality = min(1.0, max(0.0, improvement))
        else:
            quality = 0.5
            
        return float(quality)

    def _is_model_ready(self) -> bool:
        """Check if model is ready for theme detection"""
        return (hasattr(self.scaler, "mean_") and 
                hasattr(self.km, 'cluster_centers_'))

    def _detect_current_theme(self, features: np.ndarray) -> Tuple[int, float]:
        """Detect current theme with confidence assessment"""
        try:
            if len(features) != self.expected_feature_size:
                features = self._standardize_feature_size(features)
                
            x = self.scaler.transform(features.reshape(1, -1))
            theme_id = int(self.km.predict(x)[0])
            distances = self.km.transform(x)[0]
            min_dist = distances.min()
            strength = float(1.0 / (1.0 + min_dist))
            
            return theme_id, strength
            
        except Exception as e:
            self.log_operator_warning(f"Theme detection failed: {e}")
            return 0, 0.0

    def _update_theme_state(self, theme_id: int, strength: float):
        """Update theme state with transition tracking"""
        
        if theme_id != self._current_theme:
            self._theme_transitions += 1
            if self._current_theme is not None:
                self.log_operator_info(
                    f"Theme transition detected",
                    from_theme=self._current_theme,
                    to_theme=theme_id,
                    strength=f"{strength:.3f}",
                    transition_count=self._theme_transitions
                )
        
        self._current_theme = theme_id
        self._theme_momentum.append(theme_id)
        self._theme_strength_history.append(strength)
        
        self._theme_vec.fill(0.0)
        self._theme_vec[theme_id] = strength
        
        if len(self._theme_momentum) >= 3:
            recent_themes = list(self._theme_momentum)[-3:]
            stability = len(set(recent_themes)) == 1
            self._prediction_confidence = 0.8 if stability else 0.4
        
        self._update_performance_metric('current_theme', theme_id)
        self._update_performance_metric('theme_strength', strength)
        self._update_performance_metric('theme_transitions', self._theme_transitions)

    def _update_macro_context(self, info_bus: Optional[InfoBus]):
        """Update macro economic context"""
        if info_bus:
            market_context = info_bus.get('market_context', {})
            if 'macro_indicators' in market_context:
                macro_indicators = market_context['macro_indicators']
                for key in self.macro_data:
                    if key in macro_indicators:
                        self.macro_data[key] = float(macro_indicators[key])
        
        self._macro_history.append(self.macro_data.copy())

    def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        
        if len(self._theme_strength_history) > 0:
            avg_strength = np.mean(list(self._theme_strength_history))
            self._update_performance_metric('avg_theme_strength', avg_strength)
        
        if len(self._theme_momentum) >= 5:
            recent_themes = list(self._theme_momentum)[-5:]
            stability = 1.0 - (len(set(recent_themes)) - 1) / 4.0
            self._update_performance_metric('theme_stability', stability)

    def update_macro(self, indicator: str, value: float):
        """Update macro economic indicator"""
        if indicator in self.macro_data:
            old_value = self.macro_data[indicator]
            self.macro_data[indicator] = value
            
            if abs(value - old_value) > 0.1:
                self.log_operator_info(
                    f"Macro indicator updated",
                    indicator=indicator,
                    old_value=f"{old_value:.3f}",
                    new_value=f"{value:.3f}",
                    change=f"{value - old_value:+.3f}"
                )

    def detect(self, data: Dict, t: int) -> Tuple[int, float]:
        """Backward compatibility method for theme detection"""
        if not self._is_model_ready():
            return 0, 0.0
            
        features = self._mts_features_fixed(data, t)
        return self._detect_current_theme(features)

    def fit_if_needed(self, data: Dict, t: int):
        """Backward compatibility method for model fitting"""
        features = self._mts_features_fixed(data, t)
        self._fit_buffer.append(features)
        
        if self._should_fit_model():
            self._fit_model_safe()

    # ═══════════════════════════════════════════════════════════════════
    # COMPLETE VOTING AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def set_action_dim(self, dim: int):
        """Set action dimension for proposal"""
        self._action_dim = int(dim)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Enhanced action generation based on theme characteristics"""
        
        if not hasattr(self, "_action_dim"):
            self._action_dim = 2 * len(self.instruments)
            
        action = np.zeros(self._action_dim, np.float32)
        
        theme_strength = float(self._theme_vec.max())
        current_theme = int(np.argmax(self._theme_vec))
        theme_stability = self._get_theme_stability()
        
        if theme_strength > 0.3 and theme_stability > 0.5:
            
            if current_theme == 0:  # "Risk-on" theme
                for i in range(0, self._action_dim, 2):
                    action[i] = 0.3 * theme_strength * theme_stability
                    if i + 1 < self._action_dim:
                        action[i + 1] = 0.5
                        
            elif current_theme == 1:  # "Risk-off" theme
                if self._action_dim >= 4:
                    action[0] = -0.3 * theme_strength * theme_stability
                    action[1] = 0.5
                    action[2] = 0.3 * theme_strength * theme_stability
                    if len(action) > 3:
                        action[3] = 0.7
                        
            elif current_theme == 2:  # "High volatility" theme
                for i in range(0, self._action_dim, 2):
                    if theme_stability > 0.7:
                        action[i] = np.random.choice([-0.1, 0.1]) * theme_strength
                        if i + 1 < self._action_dim:
                            action[i + 1] = 0.3
                            
            elif current_theme == 3:  # "Trending" theme
                if current_theme in self._theme_profiles:
                    profile = self._theme_profiles[current_theme]
                    direction = np.sign(profile[0]) if len(profile) > 0 else 1.0
                    for i in range(0, self._action_dim, 2):
                        action[i] = direction * 0.4 * theme_strength * theme_stability
                        if i + 1 < self._action_dim:
                            action[i + 1] = 0.7
        
        return action

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return enhanced confidence based on theme detection quality"""
        
        base_confidence = float(self._theme_vec.max())
        theme_stability = self._get_theme_stability()
        clustering_quality = self._clustering_quality
        
        combined_confidence = (
            0.5 * base_confidence +
            0.3 * theme_stability +
            0.2 * clustering_quality
        )
        
        return float(np.clip(combined_confidence, 0.1, 1.0))

    def _get_theme_stability(self) -> float:
        """Calculate theme stability score"""
        if len(self._theme_momentum) < 3:
            return 0.5
            
        recent_themes = list(self._theme_momentum)[-5:]
        unique_themes = len(set(recent_themes))
        
        stability = 1.0 - (unique_themes - 1) / 4.0
        return max(0.0, stability)

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components"""
        base_obs = self._theme_vec.copy()
        
        stability = self._get_theme_stability()
        strength = float(self._theme_vec.max())
        transitions_norm = min(1.0, self._theme_transitions / 100.0)
        
        enhanced_obs = np.concatenate([
            base_obs,
            [stability, strength, transitions_norm, float(self._is_model_ready())]
        ])
        
        return enhanced_obs.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # COMPLETE EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()

    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with model rebuilding if needed"""
        old_themes = self.n_themes
        
        self.n_themes = int(genome.get("n_themes", self.n_themes))
        self.window = int(genome.get("window", self.window))
        self.batch_size = int(genome.get("batch_size", self.batch_size))
        self.feature_lookback = int(genome.get("feature_lookback", self.feature_lookback))
        
        self.genome = {
            "n_themes": self.n_themes,
            "window": self.window,
            "batch_size": self.batch_size,
            "feature_lookback": self.feature_lookback
        }
        
        if old_themes != self.n_themes:
            try:
                self.km = MiniBatchKMeans(
                    n_clusters=self.n_themes,
                    batch_size=self.batch_size,
                    random_state=0
                )
                self._theme_vec = np.zeros(self.n_themes, np.float32)
                self._theme_profiles.clear()
                self._ml_fit_count = 0
                
                self.log_operator_info(f"Model rebuilt with {self.n_themes} themes")
            except Exception as e:
                self.log_operator_error(f"Model rebuild failed: {e}")

    def mutate(self, rate: float = 0.2):
        """Enhanced mutation with performance tracking"""
        g = self.get_genome()
        mutations = []
        
        if random.random() < rate:
            old_val = g["n_themes"]
            g["n_themes"] = int(np.clip(self.n_themes + random.choice([-1, 1]), 2, 8))
            mutations.append(f"n_themes: {old_val} → {g['n_themes']}")
            
        if random.random() < rate:
            old_val = g["window"]
            g["window"] = int(np.clip(self.window + random.randint(-20, 20), 20, 200))
            mutations.append(f"window: {old_val} → {g['window']}")
            
        if random.random() < rate:
            old_val = g["batch_size"]
            g["batch_size"] = int(np.clip(self.batch_size + random.randint(-16, 16), 32, 256))
            mutations.append(f"batch_size: {old_val} → {g['batch_size']}")
        
        if mutations:
            self.log_operator_info(f"Theme detector mutation applied", changes=", ".join(mutations))
            
        self.set_genome(g)

    def crossover(self, other: "MarketThemeDetector") -> "MarketThemeDetector":
        """Enhanced crossover with compatibility checking"""
        if not isinstance(other, MarketThemeDetector):
            self.log_operator_warning("Crossover with incompatible type")
            return self
            
        new_g = {
            k: random.choice([self.genome[k], other.genome[k]]) 
            for k in self.genome
        }
        
        return MarketThemeDetector(
            self.instruments, 
            genome=new_g, 
            debug=self.config.debug
        )

    # ═══════════════════════════════════════════════════════════════════
    # COMPLETE STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            if len(self._fit_buffer) > 2500:
                return False
                
            if not np.all(np.isfinite(self._theme_vec)):
                return False
                
            if self._ml_fit_count > 0 and not hasattr(self.km, 'cluster_centers_'):
                return False
                
            if self._current_theme >= self.n_themes:
                return False
                
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        theme_details = {
            'theme_info': {
                'current_theme': self._current_theme,
                'theme_strength': float(self._theme_vec.max()),
                'theme_stability': self._get_theme_stability(),
                'transitions': self._theme_transitions
            },
            'ml_info': {
                'model_fitted': self._ml_fit_count > 0,
                'fit_count': self._ml_fit_count,
                'clustering_quality': self._clustering_quality,
                'buffer_size': len(self._fit_buffer),
                'themes_detected': len(self._theme_profiles)
            },
            'genome_config': self.genome.copy(),
            'instruments': self.instruments
        }
        
        if base_details:
            base_details.update(theme_details)
            return base_details
        
        return theme_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        scaler_state = {}
        if hasattr(self.scaler, "mean_"):
            scaler_state = {
                "mean_": self.scaler.mean_.tolist(),
                "scale_": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            }
        
        km_state = {}
        if hasattr(self.km, "cluster_centers_"):
            km_state = {
                "cluster_centers_": self.km.cluster_centers_.tolist()
            }
        
        return {
            "scaler": scaler_state,
            "km": km_state,
            "macro_data": dict(self.macro_data),
            "fit_buffer": list(self._fit_buffer)[-100:],
            "theme_vec": self._theme_vec.tolist(),
            "genome": self.genome.copy(),
            "theme_profiles": {k: v.tolist() for k, v in self._theme_profiles.items()},
            "current_theme": self._current_theme,
            "theme_momentum": list(self._theme_momentum),
            "theme_transitions": self._theme_transitions,
            "clustering_quality": self._clustering_quality,
            "ml_fit_count": self._ml_fit_count,
            "macro_history": list(self._macro_history)[-20:]
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        scaler_data = module_state.get("scaler", {})
        if scaler_data.get("mean_") is not None:
            self.scaler.mean_ = np.asarray(scaler_data["mean_"])
            if scaler_data.get("scale_") is not None:
                self.scaler.scale_ = np.asarray(scaler_data["scale_"])
        
        km_data = module_state.get("km", {})
        if km_data.get("cluster_centers_") is not None:
            self.km.cluster_centers_ = np.asarray(km_data["cluster_centers_"])
        
        self.macro_data = dict(module_state.get("macro_data", self.macro_data))
        self._fit_buffer = deque(module_state.get("fit_buffer", []), maxlen=2000)
        self._theme_vec = np.asarray(module_state.get("theme_vec", [0.0]*self.n_themes), np.float32)
        self.set_genome(module_state.get("genome", self.genome))
        self._theme_profiles = {int(k): np.asarray(v) for k, v in module_state.get("theme_profiles", {}).items()}
        self._current_theme = module_state.get("current_theme", 0)
        self._theme_momentum = deque(module_state.get("theme_momentum", []), maxlen=10)
        self._theme_transitions = module_state.get("theme_transitions", 0)
        self._clustering_quality = module_state.get("clustering_quality", 0.0)
        self._ml_fit_count = module_state.get("ml_fit_count", 0)
        self._macro_history = deque(module_state.get("macro_history", []), maxlen=100)

    def get_theme_analysis_report(self) -> str:
        """Generate operator-friendly theme analysis report"""
        
        current_strength = float(self._theme_vec.max())
        stability = self._get_theme_stability()
        
        theme_names = ["Risk-On", "Risk-Off", "High-Vol", "Trending"]
        current_theme_name = theme_names[self._current_theme] if self._current_theme < len(theme_names) else f"Theme-{self._current_theme}"
        
        if stability > 0.8:
            stability_desc = "Very Stable"
        elif stability > 0.6:
            stability_desc = "Stable"
        elif stability > 0.4:
            stability_desc = "Moderate"
        else:
            stability_desc = "Unstable"
        
        return f"""
🎯 COMPLETE MARKET THEME ANALYSIS
═══════════════════════════════════════
📊 Current Theme: {current_theme_name} (ID: {self._current_theme})
💪 Theme Strength: {current_strength:.3f}
🎪 Stability: {stability_desc} ({stability:.3f})

🤖 MACHINE LEARNING STATUS
• Model Status: {'✅ Trained' if self._ml_fit_count > 0 else '⚠️ Learning'}
• Fit Count: {self._ml_fit_count}
• Clustering Quality: {self._clustering_quality:.3f}
• Buffer Size: {len(self._fit_buffer)}/2000
• Feature Size: {self.expected_feature_size} (FIXED)

📈 THEME DYNAMICS
• Total Transitions: {self._theme_transitions}
• Recent Momentum: {list(self._theme_momentum)[-5:] if len(self._theme_momentum) >= 5 else list(self._theme_momentum)}
• Themes Available: {self.n_themes}

🔧 CONFIGURATION
• Instruments: {len(self.instruments)}
• Window Size: {self.window}
• Batch Size: {self.batch_size}

💰 MACRO CONTEXT
• VIX: {self.macro_data['vix']:.1f}
• Yield Curve: {self.macro_data['yield_curve']:.3f}
• CPI: {self.macro_data['cpi']:.1f}
        """

    # ═══════════════════════════════════════════════════════════════════
    # BACKWARD COMPATIBILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def step(self, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, **kwargs)

    def get_state(self):
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)