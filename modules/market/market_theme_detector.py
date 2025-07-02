# ─────────────────────────────────────────────────────────────
# File: modules/market/market_theme_detector.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
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

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, VotingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class MarketThemeDetector(Module, AnalysisMixin, VotingMixin):
    """
    Enhanced market theme detector with infrastructure integration.
    Detects market themes and patterns using ML clustering.
    """
    
    def __init__(
        self,
        instruments: List[str],
        n_themes: int = 4,
        window: int = 100,
        debug: bool = True,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=500,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome, n_themes, window)
        
        # Enhanced state initialization
        self._initialize_module_state()
        
        # Initialize ML components
        self._initialize_ml_components()
        
        self.instruments = instruments
        
        self.log_operator_info(
            "Market theme detector initialized",
            instruments=len(self.instruments),
            n_themes=self.n_themes,
            window=self.window,
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

        # Store genome for evolution
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
        
        # Theme-specific state
        self._fit_buffer = deque(maxlen=2000)
        self._theme_vec = np.zeros(self.n_themes, np.float32)
        self._theme_profiles = {}
        self._current_theme = 0
        self._theme_momentum = deque(maxlen=10)
        self._theme_strength_history = deque(maxlen=100)
        self._theme_transitions = 0
        self._last_theme_update = None
        
        # Enhanced tracking
        self._feature_stability_score = 1.0
        self._clustering_quality = 0.0
        self._prediction_confidence = 0.5
        
        # Macro data with enhanced tracking
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
            
            # Track ML performance
            self._ml_fit_count = 0
            self._last_inertia = None
            self._convergence_history = deque(maxlen=20)
            
            self.log_operator_info("ML components initialized successfully")
            
        except Exception as e:
            self.log_operator_error(f"ML initialization failed: {e}")
            self._update_health_status("ERROR", f"ML init failed: {e}")

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_voting_state()
        
        # Module-specific reset
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
        
        # Reset ML components
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
        
        # Extract market data for theme analysis
        market_data = self._extract_market_data(info_bus, kwargs)
        
        # Process theme detection with enhanced analytics
        self._process_theme_detection(market_data)
        
        # Update macro economic context
        self._update_macro_context(info_bus)

    def _extract_market_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market data from InfoBus or simulate from market conditions"""
        
        # Try InfoBus first
        if info_bus:
            # Extract market context
            market_context = info_bus.get('market_context', {})
            prices = info_bus.get('prices', {})
            features = info_bus.get('features', {})
            
            # Get current step for data access
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
        
        # Try kwargs (backward compatibility)
        if "data" in kwargs and "t" in kwargs:
            return {
                'data': kwargs["data"],
                't': kwargs["t"],
                'source': 'kwargs'
            }
        
        # Fallback to default simulation
        return {
            'prices': {inst: np.random.normal(1.1, 0.01) for inst in self.instruments},
            'volatility_level': 'medium',
            'regime': 'ranging',
            'session': 'unknown',
            'source': 'simulation'
        }

    def _process_theme_detection(self, market_data: Dict[str, Any]):
        """Process theme detection with enhanced analytics"""
        
        try:
            # Extract features based on data source
            if market_data.get('source') == 'kwargs' and 'data' in market_data:
                features = self._mts_features(market_data['data'], market_data['t'])
            else:
                features = self._extract_features_from_info_bus(market_data)
            
            # Update fit buffer
            self._fit_buffer.append(features)
            
            # Fit model if needed and detect themes
            if self._should_fit_model():
                self._fit_model()
            
            # Detect current theme
            if self._is_model_ready():
                theme_id, strength = self._detect_current_theme(features)
                self._update_theme_state(theme_id, strength)
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            self.log_operator_error(f"Theme detection failed: {e}")
            self._update_health_status("DEGRADED", f"Detection failed: {e}")

    def _extract_features_from_info_bus(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Extract features from InfoBus data"""
        features = []
        
        # Price-based features
        prices = market_data.get('prices', {})
        for inst in self.instruments:
            price = prices.get(inst, 1.0)
            features.extend([price, np.log(price), price * np.random.normal(1, 0.01)])
        
        # Volatility features
        vol_level = market_data.get('volatility_level', 'medium')
        vol_mapping = {'low': 0.1, 'medium': 0.2, 'high': 0.4, 'extreme': 0.8}
        vol_numeric = vol_mapping.get(vol_level, 0.2)
        features.extend([vol_numeric, vol_numeric**2, np.log1p(vol_numeric)])
        
        # Regime features
        regime = market_data.get('regime', 'ranging')
        regime_mapping = {'trending': [1, 0, 0], 'volatile': [0, 1, 0], 'ranging': [0, 0, 1]}
        regime_features = regime_mapping.get(regime, [0, 0, 1])
        features.extend(regime_features)
        
        # Session features
        session = market_data.get('session', 'unknown')
        session_mapping = {'asian': [1, 0, 0], 'european': [0, 1, 0], 'american': [0, 0, 1]}
        session_features = session_mapping.get(session, [0.33, 0.33, 0.33])
        features.extend(session_features)
        
        # Add macro data
        macro_scaled = self._macro_scaler.transform([[
            self.macro_data["vix"],
            self.macro_data["yield_curve"],
            self.macro_data["cpi"]
        ]])[0]
        features.extend(macro_scaled.tolist())
        
        return np.asarray(features, np.float32)

    def _mts_features(self, data: Dict[str, Dict[str, pd.DataFrame]], t: int) -> np.ndarray:
        """Multi-timeframe feature extraction (backward compatibility)"""
        features = []
        
        for tf in ("H1", "H4", "D1"):
            for inst in self.instruments:
                if inst not in data:
                    features.extend([0.0] * 7)
                    continue
                    
                df = data[inst][tf]
                sl = df.iloc[max(0, t - self.window): t]["close"]
                if len(sl) < 2:
                    features.extend([0.0] * 7)
                    continue
                    
                ret = sl.pct_change().dropna().values.astype(np.float32)
                if len(ret) < 1:
                    features.extend([0.0] * 7)
                    continue

                features += [
                    float(ret.mean()), 
                    float(ret.std()),
                    float(pd.Series(ret).skew()) if len(ret) > 2 else 0.0, 
                    float(pd.Series(ret).kurtosis()) if len(ret) > 2 else 0.0,
                    float((df["high"] - df["low"]).iloc[max(0, t - self.window): t].mean()),
                    self._hurst(ret), 
                    self._wavelet_energy(ret),
                ]

        # Add macro features
        macro = self._macro_scaler.transform([[
            self.macro_data["vix"],
            self.macro_data["yield_curve"],
            self.macro_data["cpi"]
        ]])[0]
        features.extend(macro.tolist())
        
        return np.asarray(features, np.float32)

    @staticmethod
    def _hurst(series: np.ndarray) -> float:
        """Calculate Hurst exponent with enhanced stability"""
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

    @staticmethod
    def _wavelet_energy(series: np.ndarray, wavelet: str = "db4") -> float:
        """Calculate wavelet energy with enhanced error handling"""
        series = series[:256]
        if series.size < 16:
            return 0.0
        try:
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

    def _fit_model(self):
        """Fit the clustering model with enhanced monitoring"""
        try:
            X = self.scaler.fit_transform(np.vstack(self._fit_buffer))
            self.km.partial_fit(X)
            self._ml_fit_count += 1
            
            # Track clustering quality
            if hasattr(self.km, 'inertia_'):
                self._last_inertia = self.km.inertia_
                self._convergence_history.append(self._last_inertia)
            
            # Update theme profiles
            if hasattr(self.km, 'cluster_centers_'):
                for i in range(self.n_themes):
                    self._theme_profiles[i] = self.km.cluster_centers_[i]
                    
                # Calculate clustering quality
                self._clustering_quality = self._calculate_clustering_quality()
            
            self.log_operator_info(
                f"Theme clustering model fitted",
                samples=len(self._fit_buffer),
                fit_count=self._ml_fit_count,
                n_themes=self.n_themes,
                quality_score=f"{self._clustering_quality:.3f}"
            )
            
            # Update performance metrics
            self._update_performance_metric('clustering_quality', self._clustering_quality)
            self._update_performance_metric('fit_count', self._ml_fit_count)
            
        except Exception as e:
            self.log_operator_error(f"Model fitting failed: {e}")
            self._update_health_status("DEGRADED", f"Fit failed: {e}")

    def _calculate_clustering_quality(self) -> float:
        """Calculate clustering quality score"""
        if not hasattr(self.km, 'cluster_centers_') or len(self._convergence_history) < 2:
            return 0.0
            
        # Quality based on inertia improvement
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
        
        # Check for theme transition
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
        
        # Update state
        self._current_theme = theme_id
        self._theme_momentum.append(theme_id)
        self._theme_strength_history.append(strength)
        
        # Update theme vector
        self._theme_vec.fill(0.0)
        self._theme_vec[theme_id] = strength
        
        # Update prediction confidence based on theme stability
        if len(self._theme_momentum) >= 3:
            recent_themes = list(self._theme_momentum)[-3:]
            stability = len(set(recent_themes)) == 1
            self._prediction_confidence = 0.8 if stability else 0.4
        
        # Update metrics
        self._update_performance_metric('current_theme', theme_id)
        self._update_performance_metric('theme_strength', strength)
        self._update_performance_metric('theme_transitions', self._theme_transitions)

    def _update_macro_context(self, info_bus: Optional[InfoBus]):
        """Update macro economic context"""
        if info_bus:
            # Extract any macro indicators from InfoBus
            market_context = info_bus.get('market_context', {})
            if 'macro_indicators' in market_context:
                macro_indicators = market_context['macro_indicators']
                for key in self.macro_data:
                    if key in macro_indicators:
                        self.macro_data[key] = float(macro_indicators[key])
        
        # Store macro history
        self._macro_history.append(self.macro_data.copy())

    def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        
        # Theme consistency metrics
        if len(self._theme_strength_history) > 0:
            avg_strength = np.mean(list(self._theme_strength_history))
            self._update_performance_metric('avg_theme_strength', avg_strength)
        
        # Theme stability
        if len(self._theme_momentum) >= 5:
            recent_themes = list(self._theme_momentum)[-5:]
            stability = 1.0 - (len(set(recent_themes)) - 1) / 4.0  # Normalized stability
            self._update_performance_metric('theme_stability', stability)

    def update_macro(self, indicator: str, value: float):
        """Update macro economic indicator"""
        if indicator in self.macro_data:
            old_value = self.macro_data[indicator]
            self.macro_data[indicator] = value
            
            if abs(value - old_value) > 0.1:  # Significant change
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
            
        features = self._mts_features(data, t)
        return self._detect_current_theme(features)

    def fit_if_needed(self, data: Dict, t: int):
        """Backward compatibility method for model fitting"""
        features = self._mts_features(data, t)
        self._fit_buffer.append(features)
        
        if self._should_fit_model():
            self._fit_model()

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED VOTING AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def set_action_dim(self, dim: int):
        """Set action dimension for proposal"""
        self._action_dim = int(dim)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Enhanced action generation based on theme characteristics"""
        
        if not hasattr(self, "_action_dim"):
            self._action_dim = 2 * len(self.instruments)
            
        action = np.zeros(self._action_dim, np.float32)
        
        # Get current theme strength and stability
        theme_strength = float(self._theme_vec.max())
        current_theme = int(np.argmax(self._theme_vec))
        
        # Analyze theme momentum for stability
        theme_stability = self._get_theme_stability()
        
        # Only trade when we have sufficient confidence
        if theme_strength > 0.3 and theme_stability > 0.5:
            
            # Theme-based strategy mapping
            if current_theme == 0:  # "Risk-on" theme
                for i in range(0, self._action_dim, 2):
                    action[i] = 0.3 * theme_strength * theme_stability
                    if i + 1 < self._action_dim:
                        action[i + 1] = 0.5  # Medium duration
                        
            elif current_theme == 1:  # "Risk-off" theme
                if self._action_dim >= 4:
                    action[0] = -0.3 * theme_strength * theme_stability
                    action[1] = 0.5
                    action[2] = 0.3 * theme_strength * theme_stability
                    if len(action) > 3:
                        action[3] = 0.7  # Longer duration for safe haven
                        
            elif current_theme == 2:  # "High volatility" theme
                for i in range(0, self._action_dim, 2):
                    if theme_stability > 0.7:
                        action[i] = np.random.choice([-0.1, 0.1]) * theme_strength
                        if i + 1 < self._action_dim:
                            action[i + 1] = 0.3  # Short duration
                            
            elif current_theme == 3:  # "Trending" theme
                if current_theme in self._theme_profiles:
                    profile = self._theme_profiles[current_theme]
                    direction = np.sign(profile[0]) if len(profile) > 0 else 1.0
                    for i in range(0, self._action_dim, 2):
                        action[i] = direction * 0.4 * theme_strength * theme_stability
                        if i + 1 < self._action_dim:
                            action[i + 1] = 0.7  # Longer duration for trends
        
        return action

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return enhanced confidence based on theme detection quality"""
        
        base_confidence = float(self._theme_vec.max())
        theme_stability = self._get_theme_stability()
        clustering_quality = self._clustering_quality
        
        # Combine multiple confidence factors
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
        
        # More stable = fewer unique themes in recent history
        stability = 1.0 - (unique_themes - 1) / 4.0
        return max(0.0, stability)

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components"""
        base_obs = self._theme_vec.copy()
        
        # Add additional context
        stability = self._get_theme_stability()
        strength = float(self._theme_vec.max())
        transitions_norm = min(1.0, self._theme_transitions / 100.0)
        
        enhanced_obs = np.concatenate([
            base_obs,
            [stability, strength, transitions_norm, float(self._is_model_ready())]
        ])
        
        return enhanced_obs.astype(np.float32)

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
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
        
        # Rebuild model if architecture changed
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
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check buffer sizes
            if len(self._fit_buffer) > 2500:
                return False
                
            # Check theme vector validity
            if not np.all(np.isfinite(self._theme_vec)):
                return False
                
            # Check ML components
            if self._ml_fit_count > 0 and not hasattr(self.km, 'cluster_centers_'):
                return False
                
            # Check theme consistency
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
        
        # Safely extract scaler state
        scaler_state = {}
        if hasattr(self.scaler, "mean_"):
            scaler_state = {
                "mean_": self.scaler.mean_.tolist(),
                "scale_": self.scaler.scale_.tolist() if hasattr(self.scaler, "scale_") else None,
            }
        
        # Safely extract clustering state
        km_state = {}
        if hasattr(self.km, "cluster_centers_"):
            km_state = {
                "cluster_centers_": self.km.cluster_centers_.tolist()
            }
        
        return {
            "scaler": scaler_state,
            "km": km_state,
            "macro_data": dict(self.macro_data),
            "fit_buffer": list(self._fit_buffer)[-100:],  # Keep recent only
            "theme_vec": self._theme_vec.tolist(),
            "genome": self.genome.copy(),
            "theme_profiles": {k: v.tolist() for k, v in self._theme_profiles.items()},
            "current_theme": self._current_theme,
            "theme_momentum": list(self._theme_momentum),
            "theme_transitions": self._theme_transitions,
            "clustering_quality": self._clustering_quality,
            "ml_fit_count": self._ml_fit_count,
            "macro_history": list(self._macro_history)[-20:]  # Keep recent only
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore scaler
        scaler_data = module_state.get("scaler", {})
        if scaler_data.get("mean_") is not None:
            self.scaler.mean_ = np.asarray(scaler_data["mean_"])
            if scaler_data.get("scale_") is not None:
                self.scaler.scale_ = np.asarray(scaler_data["scale_"])
        
        # Restore clustering model
        km_data = module_state.get("km", {})
        if km_data.get("cluster_centers_") is not None:
            self.km.cluster_centers_ = np.asarray(km_data["cluster_centers_"])
        
        # Restore other state
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
        
        # Theme description
        theme_names = ["Risk-On", "Risk-Off", "High-Vol", "Trending"]
        current_theme_name = theme_names[self._current_theme] if self._current_theme < len(theme_names) else f"Theme-{self._current_theme}"
        
        # Stability trend
        if stability > 0.8:
            stability_desc = "Very Stable"
        elif stability > 0.6:
            stability_desc = "Stable"
        elif stability > 0.4:
            stability_desc = "Moderate"
        else:
            stability_desc = "Unstable"
        
        return f"""
🎯 MARKET THEME ANALYSIS
═══════════════════════════════════════
📊 Current Theme: {current_theme_name} (ID: {self._current_theme})
💪 Theme Strength: {current_strength:.3f}
🎪 Stability: {stability_desc} ({stability:.3f})

🤖 MACHINE LEARNING STATUS
• Model Status: {'✅ Trained' if self._ml_fit_count > 0 else '⚠️ Learning'}
• Fit Count: {self._ml_fit_count}
• Clustering Quality: {self._clustering_quality:.3f}
• Buffer Size: {len(self._fit_buffer)}/2000

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

    # Maintain backward compatibility
    def step(self, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, **kwargs)

    def get_state(self):
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)