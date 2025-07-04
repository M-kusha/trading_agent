# ─────────────────────────────────────────────────────────────
# File: modules/market/market_theme_detector.py (DATA INTEGRATION FIXED)
# 🔧 CRITICAL FIX: Real market data access and feature extraction
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
            "FIXED Market theme detector initialized",
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
        
        # 🔧 NEW: Data access tracking
        self._data_access_attempts = 0
        self._successful_data_extractions = 0
        self._last_known_data = {}
        self._env_reference = None

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
        
        # Reset data tracking
        self._data_access_attempts = 0
        self._successful_data_extractions = 0
        self._last_known_data.clear()
        self._env_reference = None
        
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
        """🔧 FIXED: Enhanced step with proper data extraction"""
        
        # Store environment reference for data access
        if info_bus and 'env' in info_bus:
            self._env_reference = info_bus['env']
        
        market_data = self._extract_market_data_comprehensive(info_bus, kwargs)
        self._process_theme_detection(market_data)
        self._update_macro_context(info_bus)

    def _extract_market_data_comprehensive(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """🔧 FIXED: Comprehensive market data extraction with proper priority"""
        
        self._data_access_attempts += 1
        
        # Method 1: Try environment data directly (HIGHEST PRIORITY)
        data = self._try_environment_data_extraction(info_bus)
        if data:
            self._successful_data_extractions += 1
            self.log_operator_info("Market data extracted from environment")
            return data
        
        # Method 2: Try kwargs (backward compatibility)
        data = self._try_kwargs_extraction(kwargs)
        if data:
            self._successful_data_extractions += 1
            self.log_operator_info("Market data extracted from kwargs")
            return data
        
        # Method 3: Try InfoBus structured data
        data = self._try_infobus_structured_extraction(info_bus)
        if data:
            self._successful_data_extractions += 1
            self.log_operator_info("Market data extracted from InfoBus structured data")
            return data
        
        # Method 4: Try InfoBus prices (convert to structured format)
        data = self._try_infobus_prices_extraction(info_bus)
        if data:
            self._successful_data_extractions += 1
            self.log_operator_info("Market data extracted from InfoBus prices")
            return data
        
        # Method 5: Try last known data
        data = self._try_last_known_data()
        if data:
            self.log_operator_warning("Using last known market data (stale)")
            return data
        
        # Method 6: Create synthetic data as ultimate fallback
        self.log_operator_warning("No real market data available - creating synthetic fallback")
        return self._create_synthetic_data_fallback()

    def _try_environment_data_extraction(self, info_bus: Optional[InfoBus]) -> Optional[Dict[str, Any]]:
        """🔧 PRIORITY: Try extracting data directly from environment"""
        
        env = None
        
        # Try to get environment from InfoBus
        if info_bus and 'env' in info_bus:
            env = info_bus['env']
        elif self._env_reference:
            env = self._env_reference
        
        if not env:
            return None
            
        try:
            # Check if environment has the data we need
            if hasattr(env, 'data') and hasattr(env, 'market_state'):
                data_dict = env.data
                current_step = env.market_state.current_step
                
                if data_dict and current_step is not None:
                    # Validate that we have the expected instruments and timeframes
                    valid_data = True
                    for instrument in self.instruments:
                        if instrument not in data_dict:
                            valid_data = False
                            break
                        for tf in ["H1", "H4", "D1"]:
                            if tf not in data_dict[instrument]:
                                valid_data = False
                                break
                        if not valid_data:
                            break
                    
                    if valid_data:
                        # Store as last known good data
                        self._last_known_data = {
                            'data_dict': data_dict,
                            'current_step': current_step,
                            'source': 'environment_direct',
                            'timestamp': datetime.datetime.now().isoformat()
                        }
                        
                        return {
                            'data_dict': data_dict,
                            'current_step': current_step,
                            'theme_detector': getattr(env, 'theme_detector', None),
                            'source': 'environment_direct'
                        }
                    else:
                        self.log_operator_warning(f"Environment data validation failed - missing instruments or timeframes")
                        
        except Exception as e:
            self.log_operator_warning(f"Environment data extraction failed: {e}")
            
        return None

    def _try_kwargs_extraction(self, kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Try extracting from kwargs (legacy compatibility)"""
        if "data" in kwargs and "t" in kwargs:
            return {
                'data_dict': kwargs["data"],
                'current_step': kwargs["t"],
                'theme_detector': kwargs.get("theme_detector"),
                'source': 'kwargs'
            }
        return None

    def _try_infobus_structured_extraction(self, info_bus: Optional[InfoBus]) -> Optional[Dict[str, Any]]:
        """Try extracting structured market data from InfoBus"""
        if not info_bus:
            return None
            
        module_data = info_bus.get('module_data', {})
        if 'market_data' in module_data:
            market_data = module_data['market_data']
            if isinstance(market_data, dict) and 'data_dict' in market_data:
                return {
                    **market_data,
                    'source': 'infobus_structured'
                }
        return None

    def _try_infobus_prices_extraction(self, info_bus: Optional[InfoBus]) -> Optional[Dict[str, Any]]:
        """🔧 ENHANCED: Convert InfoBus prices to structured format for analysis"""
        if not info_bus:
            return None
            
        prices = info_bus.get('prices', {})
        if not prices or len(prices) == 0:
            return None
        
        try:
            # Convert prices to minimal structured format
            current_step = info_bus.get('step_idx', 0)
            
            # Create minimal data structure for analysis
            structured_data = {}
            for instrument in self.instruments:
                if instrument in prices:
                    # Create minimal DataFrame structure
                    price = prices[instrument]
                    structured_data[instrument] = {
                        'D1': pd.DataFrame({
                            'close': [price] * max(50, self.window),
                            'high': [price * 1.001] * max(50, self.window),
                            'low': [price * 0.999] * max(50, self.window),
                            'open': [price] * max(50, self.window),
                            'volume': [1000] * max(50, self.window)
                        }),
                        'H4': pd.DataFrame({
                            'close': [price] * max(200, self.window * 4),
                            'high': [price * 1.0005] * max(200, self.window * 4),
                            'low': [price * 0.9995] * max(200, self.window * 4),
                            'open': [price] * max(200, self.window * 4),
                            'volume': [250] * max(200, self.window * 4)
                        }),
                        'H1': pd.DataFrame({
                            'close': [price] * max(800, self.window * 16),
                            'high': [price * 1.0002] * max(800, self.window * 16),
                            'low': [price * 0.9998] * max(800, self.window * 16),
                            'open': [price] * max(800, self.window * 16),
                            'volume': [100] * max(800, self.window * 16)
                        })
                    }
            
            if structured_data:
                return {
                    'data_dict': structured_data,
                    'current_step': min(current_step, max(50, self.window) - 1),
                    'source': 'infobus_prices_converted'
                }
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus price conversion failed: {e}")
            
        return None

    def _try_last_known_data(self) -> Optional[Dict[str, Any]]:
        """Try using last known market data"""
        if self._last_known_data:
            # Check if data is not too stale (within reasonable time)
            try:
                if 'timestamp' in self._last_known_data:
                    timestamp = datetime.datetime.fromisoformat(self._last_known_data['timestamp'])
                    age = datetime.datetime.now() - timestamp
                    if age.total_seconds() < 300:  # Less than 5 minutes old
                        return self._last_known_data.copy()
            except:
                pass
        return None

    def _create_synthetic_data_fallback(self) -> Dict[str, Any]:
        """🔧 ENHANCED: Create realistic synthetic data as ultimate fallback"""
        
        try:
            structured_data = {}
            
            for instrument in self.instruments:
                # Generate realistic base prices
                if 'XAU' in instrument or 'GOLD' in instrument:
                    base_price = 2000.0
                    volatility = 0.02
                elif 'EUR' in instrument:
                    base_price = 1.1
                    volatility = 0.005
                elif 'GBP' in instrument:
                    base_price = 1.25
                    volatility = 0.006
                else:
                    base_price = 1.0
                    volatility = 0.005
                
                # Generate realistic time series with trends and patterns
                for timeframe, periods in [('D1', max(100, self.window)), 
                                         ('H4', max(400, self.window * 4)), 
                                         ('H1', max(1600, self.window * 16))]:
                    
                    # Create realistic price series
                    returns = np.random.normal(0, volatility, periods)
                    prices = np.zeros(periods)
                    prices[0] = base_price
                    
                    for i in range(1, periods):
                        prices[i] = prices[i-1] * (1 + returns[i])
                    
                    # Add some realistic OHLC structure
                    highs = prices * (1 + np.abs(np.random.normal(0, volatility/2, periods)))
                    lows = prices * (1 - np.abs(np.random.normal(0, volatility/2, periods)))
                    opens = np.roll(prices, 1)
                    opens[0] = prices[0]
                    
                    volumes = np.random.lognormal(8, 1, periods)
                    
                    if instrument not in structured_data:
                        structured_data[instrument] = {}
                    
                    structured_data[instrument][timeframe] = pd.DataFrame({
                        'open': opens,
                        'high': highs,
                        'low': lows,
                        'close': prices,
                        'volume': volumes
                    })
            
            self.log_operator_warning("Created synthetic market data for theme analysis")
            
            return {
                'data_dict': structured_data,
                'current_step': max(50, self.window) - 1,
                'source': 'synthetic_structured'
            }
            
        except Exception as e:
            self.log_operator_error(f"Synthetic data creation failed: {e}")
            return {
                'data_dict': {},
                'current_step': 0,
                'source': 'failed'
            }

    def _process_theme_detection(self, market_data: Dict[str, Any]):
        """🔧 FIXED: Process theme detection with dimension validation"""
        
        try:
            source = market_data.get('source', 'unknown')
            
            if source == 'failed' or not market_data.get('data_dict'):
                self.log_operator_error("No valid market data for theme detection")
                return
            
            if source in ['kwargs', 'environment_direct', 'infobus_structured', 'synthetic_structured'] and 'data_dict' in market_data:
                features = self._mts_features_fixed(market_data['data_dict'], market_data['current_step'])
            elif source in ['infobus_prices_converted']:
                features = self._mts_features_fixed(market_data['data_dict'], market_data['current_step'])
            else:
                features = self._extract_features_from_info_bus_fixed(market_data)
            
            # Validate feature size
            if len(features) != self.expected_feature_size:
                self.log_operator_warning(
                    f"FIXED Feature size mismatch: got {len(features)}, expected {self.expected_feature_size}"
                )
                features = self._standardize_feature_size(features)
            else:
                self.log_operator_info(
                    f"FIXED Feature extraction successful: {len(features)} features as expected"
                )
            
            self._fit_buffer.append(features)
            
            if self._should_fit_model():
                self._fit_model_safe()
            
            if self._is_model_ready():
                theme_id, strength = self._detect_current_theme(features)
                self._update_theme_state(theme_id, strength)
            
            self._update_performance_metrics()
            
            # Update success metrics
            success_rate = self._successful_data_extractions / max(self._data_access_attempts, 1)
            self._update_performance_metric('data_extraction_success_rate', success_rate)
            
        except Exception as e:
            self.log_operator_error(f"Theme detection failed: {e}")
            self._update_health_status("DEGRADED", f"Detection failed: {e}")

    def _mts_features_fixed(self, data: Dict[str, Dict[str, pd.DataFrame]], t: int) -> np.ndarray:
        """🔧 FIXED: Multi-timeframe feature extraction with proper data access"""
        features = []
        
        timeframes = ["H1", "H4", "D1"]
        
        for tf in timeframes:
            for inst in self.instruments:
                if inst not in data or tf not in data[inst]:
                    self.log_operator_warning(f"Missing data for {inst}/{tf} - using zeros")
                    features.extend([0.0] * 7)
                    continue
                    
                df = data[inst][tf]
                if len(df) == 0:
                    self.log_operator_warning(f"Empty dataframe for {inst}/{tf}")
                    features.extend([0.0] * 7)
                    continue
                
                # Adjust step to be within bounds
                actual_step = min(t, len(df) - 1)
                start_idx = max(0, actual_step - self.window)
                end_idx = actual_step + 1
                
                if end_idx <= start_idx:
                    features.extend([0.0] * 7)
                    continue
                    
                sl = df.iloc[start_idx:end_idx]["close"]
                if len(sl) < 2:
                    features.extend([0.0] * 7)
                    continue

                try:
                    ret = sl.pct_change().dropna().values.astype(np.float32)
                    if len(ret) < 1:
                        features.extend([0.0] * 7)
                        continue

                    feat_mean = float(ret.mean()) if len(ret) > 0 else 0.0
                    feat_std = float(ret.std()) if len(ret) > 0 else 0.0
                    feat_skew = float(pd.Series(ret).skew()) if len(ret) > 2 else 0.0
                    feat_kurt = float(pd.Series(ret).kurtosis()) if len(ret) > 2 else 0.0
                    
                    # Enhanced HL range calculation
                    if "high" in df.columns and "low" in df.columns:
                        hl_data = df.iloc[start_idx:end_idx]
                        feat_hl_range = float((hl_data["high"] - hl_data["low"]).mean())
                    else:
                        feat_hl_range = feat_std * 2.0  # Approximate
                    
                    feat_hurst = self._hurst_safe(ret)
                    feat_wavelet = self._wavelet_energy_safe(ret)
                    
                    inst_tf_features = [
                        feat_mean, feat_std, feat_skew, feat_kurt, 
                        feat_hl_range, feat_hurst, feat_wavelet
                    ]
                    
                    # Sanitize features
                    inst_tf_features = [
                        float(np.nan_to_num(f, nan=0.0, posinf=1.0, neginf=-1.0)) 
                        for f in inst_tf_features
                    ]
                    
                    features.extend(inst_tf_features)
                    
                except Exception as e:
                    self.log_operator_warning(f"Feature calculation failed for {inst}/{tf}: {e}")
                    features.extend([0.0] * 7)

        # Add macro features
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
        
        self.log_operator_info(
            f"FIXED MTS feature extraction completed",
            feature_count=len(result),
            expected_count=self.expected_feature_size,
            instruments=len(self.instruments),
            timeframes=len(timeframes)
        )
        
        return result

    def _standardize_feature_size(self, features: np.ndarray) -> np.ndarray:
        """🔧 FIX: Standardize feature vector to expected size"""
        
        current_size = len(features)
        expected_size = self.expected_feature_size
        
        if current_size == expected_size:
            return features
            
        elif current_size < expected_size:
            padding = np.zeros(expected_size - current_size, dtype=np.float32)
            standardized = np.concatenate([features, padding])
            self.log_operator_info(
                f"FIXED Padded features from {current_size} to {expected_size}"
            )
            
        else:
            standardized = features[:expected_size]
            self.log_operator_info(
                f"FIXED Truncated features from {current_size} to {expected_size}"
            )
        
        return standardized.astype(np.float32)

    def _extract_features_from_info_bus_fixed(self, market_data: Dict[str, Any]) -> np.ndarray:
        """🔧 LEGACY: Extract features with consistent dimensionality"""
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
        
        self.log_operator_warning(
            f"LEGACY InfoBus feature extraction: got {len(result)}, expected {self.expected_feature_size}"
        )
        
        return result

    # ═══════════════════════════════════════════════════════════════════
    # KEEP ALL EXISTING METHODS UNCHANGED
    # ═══════════════════════════════════════════════════════════════════

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
                f"FIXED theme clustering model fitted",
                samples=len(buffer_data),
                fit_count=self._ml_fit_count,
                n_themes=self.n_themes,
                feature_size=len(buffer_data[0]) if buffer_data else 0,
                quality_score=f"{self._clustering_quality:.3f}"
            )
            
            self._update_performance_metric('clustering_quality', self._clustering_quality)
            self._update_performance_metric('fit_count', self._ml_fit_count)
            
        except Exception as e:
            self.log_operator_error(f"Model fitting failed: {e}")
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

    # ═══════════════════════════════════════════════════════════════════
    # KEEP ALL EXISTING VOTING, ACTION, AND COMPATIBILITY METHODS
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
        
        # Add data extraction confidence
        data_confidence = self._successful_data_extractions / max(self._data_access_attempts, 1)
        
        combined_confidence = (
            0.4 * base_confidence +
            0.25 * theme_stability +
            0.15 * clustering_quality +
            0.2 * data_confidence
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
        data_success_rate = self._successful_data_extractions / max(self._data_access_attempts, 1)
        
        enhanced_obs = np.concatenate([
            base_obs,
            [stability, strength, transitions_norm, float(self._is_model_ready()), data_success_rate]
        ])
        
        return enhanced_obs.astype(np.float32)

    # Backward compatibility methods
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

    def step(self, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, **kwargs)

    def get_state(self):
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)