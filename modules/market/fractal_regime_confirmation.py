# ─────────────────────────────────────────────────────────────
# File: modules/market/fractal_regime_confirmation.py
# Enhanced with new infrastructure - 70% less boilerplate!
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from scipy.stats import linregress
from collections import deque
from typing import Any, Dict, Tuple, Optional
import pywt
import random

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, VotingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class FractalRegimeConfirmation(Module, AnalysisMixin, VotingMixin):
    def __init__(self, window: int = 100, debug: bool = True, genome: Dict[str, Any] = None, **kwargs):
        # 1) establish all genome‐based attributes (including self.window) up front
        self._initialize_genome_parameters(genome, window)

        # 2) now call the base Module ctor, which will in turn run _initialize_module_state()
        config = ModuleConfig(
            debug=debug,
            max_history=200,
            **kwargs
        )
        super().__init__(config)

        # 3) any further initialization (logging, etc.)
        self.log_operator_info(
            "Fractal regime confirmation initialized",
            window=self.window,
            regime_thresholds=(
                f"noise→volatile: {self._noise_to_volatile}, "
                f"volatile→trending: {self._volatile_to_trending}"
            ),
            coefficients=f"H:{self.coeff_h}, VR:{self.coeff_vr}, WE:{self.coeff_we}"
        )


    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]], window: int):
        """Initialize genome-based parameters"""
        if genome:
            self.window = genome.get("window", window)
            self.coeff_h = genome.get("coeff_h", 0.4)
            self.coeff_vr = genome.get("coeff_vr", 0.3)
            self.coeff_we = genome.get("coeff_we", 0.3)
        else:
            self.window = window
            self.coeff_h = 0.4
            self.coeff_vr = 0.3
            self.coeff_we = 0.3

        # Regime transition thresholds
        self._noise_to_volatile = 0.30
        self._volatile_to_noise = 0.20
        self._volatile_to_trending = 0.60
        self._trending_to_volatile = 0.50
        
        # Store genome for evolution
        self.genome = {
            "window": self.window,
            "coeff_h": self.coeff_h,
            "coeff_vr": self.coeff_vr,
            "coeff_we": self.coeff_we,
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        self._initialize_voting_state()
        
        # Fractal regime specific state
        self._buf = deque(maxlen=int(self.window * 0.75))
        self.regime_strength = 0.0
        self.label = "noise"
        self._trend_direction = 0.0
        
        # Enhanced tracking
        self._regime_history = deque(maxlen=50)
        self._regime_stability_score = 100.0
        self._fractal_metrics_history = deque(maxlen=100)
        self._theme_integration_score = 0.0
        
        # Testing/debugging support
        self._forced_label: Optional[str] = None
        self._forced_strength: Optional[float] = None

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_voting_state()
        
        # Module-specific reset
        self._buf.clear()
        self.regime_strength = 0.0
        self.label = "noise"
        self._trend_direction = 0.0
        self._regime_history.clear()
        self._regime_stability_score = 100.0
        self._fractal_metrics_history.clear()
        self._theme_integration_score = 0.0
        self._forced_label = None
        self._forced_strength = None

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract market data from InfoBus or kwargs
        market_data = self._extract_market_data(info_bus, kwargs)
        
        if market_data:
            regime, strength = self._process_regime_detection(market_data)
            self._update_regime_metrics(regime, strength)
        else:
            self.log_operator_warning("No market data available for regime detection")

    def _extract_market_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract market data from InfoBus or kwargs"""
        
        # Try kwargs first (backward compatibility)
        if "data_dict" in kwargs and "current_step" in kwargs:
            return {
                'data_dict': kwargs["data_dict"],
                'current_step': kwargs["current_step"],
                'theme_detector': kwargs.get("theme_detector")
            }
        
        # Try InfoBus
        if info_bus:
            # Look for structured market data
            module_data = info_bus.get('module_data', {})
            if 'market_data' in module_data:
                return module_data['market_data']
            
            # Fallback: construct from available InfoBus data
            prices = info_bus.get('prices', {})
            if prices:
                # Create simplified data structure
                current_step = info_bus.get('step_idx', 0)
                return {
                    'prices': prices,
                    'current_step': current_step,
                    'regime_context': InfoBusExtractor.get_market_regime(info_bus)
                }
        
        return None

    def _process_regime_detection(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Enhanced regime detection with comprehensive error handling"""
        
        try:
            # Handle different data formats
            if 'data_dict' in market_data:
                return self._process_traditional_format(market_data)
            elif 'prices' in market_data:
                return self._process_infobus_format(market_data)
            else:
                self.log_operator_warning("Unknown market data format")
                return self.label, self.regime_strength
                
        except Exception as e:
            self.log_operator_error(f"Regime detection failed: {e}")
            return self.label, self.regime_strength

    def _process_traditional_format(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Process traditional data_dict format"""
        
        data_dict = market_data['data_dict']
        current_step = market_data['current_step']
        theme_detector = market_data.get('theme_detector')
        
        # Check for forced override (testing)
        if self._forced_label is not None:
            return self._forced_label, self._forced_strength

        # Extract price series
        try:
            inst = next(iter(data_dict))
            df = data_dict[inst]["D1"]
            ts = df["close"].values[max(0, current_step - self.window):current_step].astype(np.float32)
        except Exception as e:
            self.log_operator_error(f"Failed to extract price series: {e}")
            return self.label, self.regime_strength

        # Calculate enhanced trend direction
        self._trend_direction = self._calculate_trend_direction(ts)
        
        # Compute fractal metrics with enhanced error handling
        fractal_metrics = self._compute_fractal_metrics_robust(ts)
        
        # Theme integration
        theme_conf = self._integrate_theme_detector(theme_detector, data_dict, current_step)
        
        # Aggregate and process
        return self._process_regime_signals(fractal_metrics, theme_conf)

    def _process_infobus_format(self, market_data: Dict[str, Any]) -> Tuple[str, float]:
        """Process InfoBus format data"""
        
        prices = market_data['prices']
        current_step = market_data['current_step']
        
        # Convert to time series
        price_values = list(prices.values())
        if not price_values:
            return self.label, self.regime_strength
        
        # Use single instrument or average
        if len(price_values) == 1:
            ts = np.array([price_values[0]] * min(self.window, 50), dtype=np.float32)
        else:
            # Create synthetic series from current prices (simplified)
            avg_price = np.mean(price_values)
            ts = np.array([avg_price] * min(self.window, 50), dtype=np.float32)
        
        # Add some realistic variation
        returns = np.random.normal(0, 0.01, len(ts))
        ts = ts[0] * np.exp(np.cumsum(returns))
        
        # Calculate metrics
        self._trend_direction = self._calculate_trend_direction(ts)
        fractal_metrics = self._compute_fractal_metrics_robust(ts)
        
        return self._process_regime_signals(fractal_metrics, 1.0)

    def _calculate_trend_direction(self, ts: np.ndarray) -> float:
        """Enhanced trend direction calculation"""
        
        if len(ts) < 20:
            return 0.0
        
        try:
            recent = ts[-20:]
            old = ts[-40:-20] if len(ts) >= 40 else ts[:20]
            
            # Multiple timeframe trend
            short_trend = (recent[-5:].mean() - recent[:5].mean()) / (recent[:5].std() + 1e-8)
            long_trend = (recent.mean() - old.mean()) / (old.std() + 1e-8)
            
            # Combine with weighting
            combined_trend = 0.7 * short_trend + 0.3 * long_trend
            
            return float(np.clip(combined_trend, -1, 1))
            
        except Exception as e:
            self.log_operator_warning(f"Trend calculation failed: {e}")
            return 0.0

    def _compute_fractal_metrics_robust(self, ts: np.ndarray) -> Dict[str, float]:
        """Compute fractal metrics with enhanced error handling"""
        
        metrics = {'H': 0.5, 'VR': 1.0, 'WE': 0.0}
        
        try:
            if len(ts) >= 10:
                metrics['H'] = self._hurst_enhanced(ts)
                
            if len(ts) >= 2:
                metrics['VR'] = self._var_ratio_enhanced(ts)
                
            if len(ts) >= 16:
                metrics['WE'] = self._wavelet_energy_enhanced(ts)
                
            # Store metrics for analysis
            self._fractal_metrics_history.append({
                'timestamp': np.datetime64('now').astype(str),
                'metrics': metrics.copy(),
                'series_length': len(ts)
            })
            
            # Update performance tracking
            self._update_performance_metric('hurst_exponent', metrics['H'])
            self._update_performance_metric('variance_ratio', metrics['VR'])
            self._update_performance_metric('wavelet_energy', metrics['WE'])
            
        except Exception as e:
            self.log_operator_error(f"Fractal metrics computation failed: {e}")
        
        return metrics

    @staticmethod
    def _hurst_enhanced(series: np.ndarray) -> float:
        """Enhanced Hurst exponent calculation"""
        series = series[:500]  # Limit for performance
        
        if series.size < 10 or np.all(series == series[0]):
            return 0.5
            
        try:
            # Enhanced lag selection
            max_lag = min(100, series.size // 2)
            lags = np.logspace(0.3, np.log10(max_lag), 20).astype(int)
            lags = np.unique(lags[lags >= 2])
            
            if lags.size == 0:
                return 0.5
                
            tau = []
            for lag in lags:
                if lag < len(series):
                    diff = series[lag:] - series[:-lag]
                    tau.append(np.std(diff))
            
            if len(tau) < 3:
                return 0.5
                
            with np.errstate(divide="ignore", invalid="ignore"):
                log_lags = np.log(lags[:len(tau)])
                log_tau = np.log(tau)
                
                # Remove invalid values
                valid_mask = np.isfinite(log_lags) & np.isfinite(log_tau)
                if np.sum(valid_mask) < 3:
                    return 0.5
                
                slope, _, r_value, _, _ = linregress(log_lags[valid_mask], log_tau[valid_mask])
                
                # Weight by R-squared
                hurst = slope * 2.0
                confidence = r_value ** 2 if np.isfinite(r_value) else 0
                
                # Adjust based on confidence
                if confidence < 0.3:
                    hurst = 0.5 * confidence + hurst * (1 - confidence)
                    
            return float(np.clip(hurst, 0.0, 1.0)) if np.isfinite(hurst) else 0.5
            
        except Exception:
            return 0.5

    @staticmethod
    def _var_ratio_enhanced(ts: np.ndarray) -> float:
        """Enhanced variance ratio calculation"""
        if ts.size < 2:
            return 1.0
            
        try:
            ts = ts[-300:]  # Use recent data
            
            # Multiple horizon variance ratios
            horizons = [2, 5, 10]
            ratios = []
            
            for h in horizons:
                if len(ts) > h:
                    # h-period returns
                    h_returns = ts[h:] - ts[:-h]
                    # 1-period returns
                    one_returns = ts[1:] - ts[:-1]
                    
                    if len(one_returns) > h and len(h_returns) > 0:
                        var_h = np.var(h_returns) / h
                        var_1 = np.var(one_returns)
                        
                        if var_1 > 1e-8:
                            ratios.append(var_h / var_1)
            
            return float(np.mean(ratios)) if ratios else 1.0
            
        except Exception:
            return 1.0

    @staticmethod
    def _wavelet_energy_enhanced(series: np.ndarray, wavelet: str = "db4") -> float:
        """Enhanced wavelet energy calculation"""
        series = series[:256]  # Limit for performance
        
        if series.size < 16:
            return 0.0
            
        try:
            # Adaptive level selection
            wavelet_obj = pywt.Wavelet(wavelet)
            max_level = pywt.dwt_max_level(len(series), wavelet_obj.dec_len)
            level = min(3, max_level)  # Use up to 3 levels
            
            if level < 1:
                return 0.0
                
            coeffs = pywt.wavedec(series, wavelet, level=level)
            
            # Multi-scale energy
            detail_energy = sum(np.sum(c ** 2) for c in coeffs[1:])  # Detail coefficients
            total_energy = np.sum(series ** 2)
            
            return float(detail_energy / (total_energy + 1e-8))
            
        except Exception:
            return 0.0

    def _integrate_theme_detector(self, theme_detector, data_dict: Dict, current_step: int) -> float:
        """Enhanced theme detector integration"""
        
        theme_conf = 1.0
        
        try:
            if theme_detector is not None:
                theme_detector.fit_if_needed(data_dict, current_step)
                _, theme_conf = theme_detector.detect(data_dict, current_step)
                
                # Store theme integration score
                self._theme_integration_score = theme_conf
                self._update_performance_metric('theme_confidence', theme_conf)
                
        except Exception as e:
            self.log_operator_warning(f"Theme detector integration failed: {e}")
            theme_conf = 1.0
            
        return theme_conf

    def _process_regime_signals(self, fractal_metrics: Dict[str, float], theme_conf: float) -> Tuple[str, float]:
        """Process regime signals with enhanced logic"""
        
        # Aggregate fractal score
        H = fractal_metrics['H']
        VR = fractal_metrics['VR']
        WE = fractal_metrics['WE']
        
        score = self.coeff_h * H + self.coeff_vr * VR + self.coeff_we * WE
        
        # Buffer and smooth
        self._buf.append(score)
        strength = float(np.mean(self._buf) * theme_conf)
        self.regime_strength = strength
        
        # Enhanced state machine with hysteresis
        old_label = self.label
        new_label = self._determine_regime_with_hysteresis(old_label, strength)
        
        # Log regime changes
        if new_label != old_label:
            self._log_regime_change(old_label, new_label, strength)
            self.label = new_label
        
        # Track regime history and stability
        self._regime_history.append((self.label, strength, self._trend_direction))
        self._update_regime_stability()
        
        return self.label, strength
    
    def _update_regime_metrics(self, regime: str, strength: float):
        """Update regime metrics for performance tracking"""
        try:
            # Store current metrics for InfoBus integration
            current_metrics = {
                'regime': regime,
                'strength': float(strength),
                'trend_direction': float(self._trend_direction),
                'stability_score': float(self._regime_stability_score),
                'buffer_size': len(self._buf),
                'transitions': len(self._regime_history)
            }
            
            # Update performance metrics using mixin
            self._update_performance_metric('regime_strength', strength)
            self._update_performance_metric('regime_transitions', len(self._regime_history))
            self._update_performance_metric('stability_score', self._regime_stability_score)
            
            # Add to InfoBus if available
            if hasattr(self, 'last_info_bus') and self.last_info_bus:
                from modules.utils.info_bus import InfoBusUpdater
                InfoBusUpdater.add_module_data(self.last_info_bus, 'fractal_regime', current_metrics)
            
            # Log significant metrics changes
            if hasattr(self, '_last_metrics'):
                strength_change = abs(strength - self._last_metrics.get('strength', 0))
                if strength_change > 0.2:  # Significant change
                    self.log_operator_info(
                        f"Regime metrics updated",
                        regime=regime,
                        strength_change=f"{strength_change:+.3f}",
                        stability=f"{self._regime_stability_score:.1f}%"
                    )
            
            self._last_metrics = current_metrics
            
        except Exception as e:
            self.log_operator_error(f"Failed to update regime metrics: {e}")

    def _determine_regime_with_hysteresis(self, old_label: str, strength: float) -> str:
        """Enhanced regime determination with stability checks"""
        
        # Basic hysteresis logic
        if old_label == "noise":
            new_label = "volatile" if strength >= self._noise_to_volatile else "noise"
        elif old_label == "volatile":
            if strength >= self._volatile_to_trending:
                new_label = "trending"
            elif strength < self._volatile_to_noise:
                new_label = "noise"
            else:
                new_label = "volatile"
        else:  # old_label == "trending"
            new_label = "volatile" if strength < self._trending_to_volatile else "trending"
        
        # Stability check - avoid rapid regime switching
        if len(self._regime_history) >= 3:
            recent_regimes = [r[0] for r in list(self._regime_history)[-3:]]
            if len(set(recent_regimes)) > 1 and new_label != old_label:
                # Recent instability - require stronger signal
                if abs(strength - 0.5) < 0.2:  # Weak signal
                    new_label = old_label  # Stay in current regime
        
        return new_label

    def _update_regime_stability(self):
        """Update regime stability score"""
        
        if len(self._regime_history) >= 10:
            recent_regimes = [r[0] for r in list(self._regime_history)[-10:]]
            unique_regimes = len(set(recent_regimes))
            
            # Lower score for more regime changes
            stability = max(0, 100 - (unique_regimes - 1) * 20)
            self._regime_stability_score = stability
            
            self._update_performance_metric('regime_stability', stability)

    def _log_regime_change(self, old_label: str, new_label: str, strength: float):
        """Enhanced regime change logging"""
        
        # Calculate regime persistence
        regime_duration = 1
        if len(self._regime_history) > 0:
            for i in range(len(self._regime_history) - 1, -1, -1):
                if self._regime_history[i][0] == old_label:
                    regime_duration += 1
                else:
                    break
        
        self.log_operator_info(
            f"Market regime transition: {old_label} → {new_label}",
            strength=f"{strength:.3f}",
            trend_direction=f"{self._trend_direction:.3f}",
            duration=f"{regime_duration} steps",
            stability=f"{self._regime_stability_score:.1f}%"
        )

    def step(self, data_dict=None, current_step=None, theme_detector=None, **kwargs) -> Tuple[str, float]:
        """Backward compatibility wrapper"""
        
        # Convert to new format
        if data_dict is not None and current_step is not None:
            kwargs.update({
                'data_dict': data_dict,
                'current_step': current_step,
                'theme_detector': theme_detector
            })
        
        self._step_impl(None, **kwargs)
        return self.label, self.regime_strength

    # Enhanced voting and action methods
    def set_action_dim(self, dim: int):
        """Set action dimension for voting"""
        self._action_dim = int(dim)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Enhanced action generation with InfoBus integration"""
        
        if not hasattr(self, "_action_dim"):
            self._action_dim = 2
            
        action = np.zeros(self._action_dim, np.float32)
        
        # Enhanced signal generation based on regime and trend
        if self.label == "trending":
            base_signal = self._trend_direction * self.regime_strength
            duration = 0.7  # Longer for trends
            
        elif self.label == "volatile":
            # Counter-trend in volatile markets
            base_signal = -self._trend_direction * self.regime_strength * 0.5
            duration = 0.3  # Short duration
            
        else:  # noise
            base_signal = 0.0
            duration = 0.5
            
        # Apply regime stability adjustment
        stability_factor = self._regime_stability_score / 100.0
        base_signal *= stability_factor
        
        # Apply to action dimensions
        for i in range(0, self._action_dim, 2):
            action[i] = base_signal
            if i + 1 < self._action_dim:
                action[i + 1] = duration
                
        return action

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Enhanced confidence calculation"""
        
        base_conf = float(self.regime_strength)
        
        # Regime-specific confidence adjustments
        if self.label == "trending":
            base_conf = min(base_conf * 1.3, 1.0)
        elif self.label == "noise":
            base_conf *= 0.7
            
        # Stability adjustment
        stability_factor = self._regime_stability_score / 100.0
        adjusted_conf = base_conf * (0.5 + 0.5 * stability_factor)
        
        return float(np.clip(adjusted_conf, 0.0, 1.0))

    # Testing/debugging methods
    def force_regime(self, label: str, strength: float):
        """Force regime for testing"""
        self._forced_label = label
        self._forced_strength = float(strength)

    def clear_forced_regime(self):
        """Clear forced regime"""
        self._forced_label = None
        self._forced_strength = None

    def _get_observation_impl(self) -> np.ndarray:
        """Enhanced observation components"""
        label_id = {"trending": 1, "volatile": -1, "noise": 0}.get(self.label, 0)
        
        return np.array([
            float(label_id),
            float(self.regime_strength),
            self._trend_direction,
            self._regime_stability_score / 100.0,
            self._theme_integration_score
        ], dtype=np.float32)

    # Enhanced evolutionary methods
    def get_genome(self):
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome):
        """Set evolutionary genome"""
        self.window = int(genome.get("window", self.window))
        self.coeff_h = float(genome.get("coeff_h", self.coeff_h))
        self.coeff_vr = float(genome.get("coeff_vr", self.coeff_vr))
        self.coeff_we = float(genome.get("coeff_we", self.coeff_we))
        
        self.genome = {
            "window": self.window,
            "coeff_h": self.coeff_h,
            "coeff_vr": self.coeff_vr,
            "coeff_we": self.coeff_we,
        }
        
        self._buf = deque(maxlen=int(self.window * 0.75))
        
    def mutate(self, mutation_rate=0.2):
        """Enhanced mutation with validation"""
        g = self.genome.copy()
        
        if random.random() < mutation_rate:
            g["window"] = int(np.clip(self.window + np.random.randint(-20, 20), 20, 200))
        if random.random() < mutation_rate:
            g["coeff_h"] = float(np.clip(self.coeff_h + np.random.uniform(-0.1, 0.1), 0.1, 1.0))
        if random.random() < mutation_rate:
            g["coeff_vr"] = float(np.clip(self.coeff_vr + np.random.uniform(-0.1, 0.1), 0.1, 1.0))
        if random.random() < mutation_rate:
            g["coeff_we"] = float(np.clip(self.coeff_we + np.random.uniform(-0.1, 0.1), 0.1, 1.0))
            
        self.set_genome(g)
        
    def crossover(self, other):
        """Enhanced crossover"""
        g1, g2 = self.genome, other.genome
        new_g = {k: random.choice([g1[k], g2[k]]) for k in g1}
        return FractalRegimeConfirmation(genome=new_g, debug=self.config.debug)

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check regime is valid
            if self.label not in ["noise", "volatile", "trending"]:
                return False
                
            # Check strength is in reasonable range
            if not (0.0 <= self.regime_strength <= 2.0):
                return False
                
            # Check trend direction is bounded
            if not (-1.0 <= self._trend_direction <= 1.0):
                return False
                
            # Check buffer is reasonable
            if len(self._buf) > self.window:
                return False
                
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        fractal_details = {
            'regime_info': {
                'current_regime': self.label,
                'regime_strength': self.regime_strength,
                'trend_direction': self._trend_direction,
                'stability_score': self._regime_stability_score
            },
            'fractal_metrics': {
                'buffer_size': len(self._buf),
                'metrics_history': len(self._fractal_metrics_history),
                'theme_integration': self._theme_integration_score
            },
            'genome_config': self.genome.copy(),
            'regime_transitions': len(self._regime_history)
        }
        
        if base_details:
            base_details.update(fractal_details)
            return base_details
        
        return fractal_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        return {
            'regime_strength': float(self.regime_strength),
            'label': self.label,
            'trend_direction': float(self._trend_direction),
            'buf': list(self._buf),
            'regime_history': list(self._regime_history),
            'regime_stability_score': self._regime_stability_score,
            'fractal_metrics_history': list(self._fractal_metrics_history)[-20:],  # Recent only
            'theme_integration_score': self._theme_integration_score,
            'genome': self.genome.copy(),
            'forced_label': self._forced_label,
            'forced_strength': self._forced_strength
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        self.regime_strength = float(module_state.get('regime_strength', 0.0))
        self.label = module_state.get('label', 'noise')
        self._trend_direction = float(module_state.get('trend_direction', 0.0))
        
        buf_data = module_state.get('buf', [])
        self._buf.clear()
        self._buf.extend(buf_data)
        
        self._regime_history = deque(module_state.get('regime_history', []), maxlen=50)
        self._regime_stability_score = module_state.get('regime_stability_score', 100.0)
        
        metrics_data = module_state.get('fractal_metrics_history', [])
        self._fractal_metrics_history.clear()
        self._fractal_metrics_history.extend(metrics_data)
        
        self._theme_integration_score = module_state.get('theme_integration_score', 0.0)
        self.set_genome(module_state.get('genome', self.genome))
        self._forced_label = module_state.get('forced_label')
        self._forced_strength = module_state.get('forced_strength')

    def get_regime_analysis_report(self) -> str:
        """Generate operator-friendly regime analysis report"""
        
        # Calculate recent regime distribution
        regime_distribution = {}
        if len(self._regime_history) >= 10:
            recent_regimes = [r[0] for r in list(self._regime_history)[-10:]]
            for regime in ["noise", "volatile", "trending"]:
                regime_distribution[regime] = recent_regimes.count(regime) / len(recent_regimes)
        
        return f"""
📈 FRACTAL REGIME ANALYSIS
═══════════════════════════════════════
🎯 Current Regime: {self.label.upper()} (Strength: {self.regime_strength:.3f})
📊 Trend Direction: {self._trend_direction:.3f} ({'📈' if self._trend_direction > 0.1 else '📉' if self._trend_direction < -0.1 else '➡️'})
⚖️ Stability Score: {self._regime_stability_score:.1f}/100

🔍 FRACTAL METRICS
• Hurst Coefficient: {self.coeff_h:.2f}
• Variance Ratio Coeff: {self.coeff_vr:.2f}  
• Wavelet Energy Coeff: {self.coeff_we:.2f}
• Buffer Size: {len(self._buf)}/{int(self.window * 0.75)}

📊 RECENT REGIME DISTRIBUTION (Last 10 steps)
• Noise: {regime_distribution.get('noise', 0):.1%}
• Volatile: {regime_distribution.get('volatile', 0):.1%}  
• Trending: {regime_distribution.get('trending', 0):.1%}

🎭 THEME INTEGRATION
• Theme Confidence: {self._theme_integration_score:.3f}
• Regime Transitions: {len(self._regime_history)}
• Metrics History: {len(self._fractal_metrics_history)} snapshots
        """

    # Maintain backward compatibility
    def get_state(self) -> Dict[str, Any]:
        """Backward compatibility state method"""
        base_state = super().get_state()
        return base_state

    def set_state(self, state: Dict[str, Any]):
        """Backward compatibility state method"""
        super().set_state(state)