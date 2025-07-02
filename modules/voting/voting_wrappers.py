# ─────────────────────────────────────────────────────────────
# File: modules/voting/voting_wrappers.py
# Enhanced Voting Wrappers with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import math
import torch
import inspect
from typing import Any, Dict, List, Optional
from collections import deque

from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, format_operator_message


class ThemeExpert:
    """
    Enhanced theme-based trading expert with InfoBus integration.
    Provides trading signals based on detected market themes and regimes.
    """

    def __init__(self, detector, env_ref, trend_label: str = "trending", 
                 max_size: float = 0.7, debug: bool = True):
        self.det = detector
        self.env = env_ref
        self.trend_label = trend_label
        self.max_size = max_size
        self.debug = debug
        
        # Enhanced state tracking
        self._last_strength = 0.0
        self._last_theme = 0
        self._theme_history = deque(maxlen=50)
        self._performance_history = deque(maxlen=30)
        self._zero = np.zeros(self.env.action_dim, dtype=np.float32)
        
        # Theme performance tracking
        self.theme_performance = {
            0: {'signals': 0, 'success': 0, 'avg_strength': 0.0},  # Risk-on
            1: {'signals': 0, 'success': 0, 'avg_strength': 0.0},  # Risk-off
            2: {'signals': 0, 'success': 0, 'avg_strength': 0.0},  # High volatility
            3: {'signals': 0, 'success': 0, 'avg_strength': 0.0},  # Trending
        }
        
        # Market context awareness
        self.market_context = {
            'regime': 'unknown',
            'session': 'unknown',
            'volatility_level': 'medium'
        }
        
        # Setup logging
        self.logger = RotatingLogger(
            "ThemeExpert",
            "logs/voting/theme_expert.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        self.log_operator_info(
            "🎭 Theme Expert initialized",
            max_size=self.max_size,
            action_dim=self.env.action_dim
        )

    def reset(self):
        """Enhanced reset with comprehensive state cleanup"""
        self._last_strength = 0.0
        self._last_theme = 0
        self._theme_history.clear()
        self._performance_history.clear()
        
        # Reset performance tracking
        for theme_data in self.theme_performance.values():
            theme_data.update({'signals': 0, 'success': 0, 'avg_strength': 0.0})
        
        self.log_operator_info("🔄 Theme Expert reset")

    def propose_action(self, obs: Any, extras: Dict = None, info_bus: InfoBus = None) -> np.ndarray:
        """Enhanced action proposal with InfoBus integration"""
        
        try:
            # Extract market context if InfoBus available
            if info_bus:
                context = extract_standard_context(info_bus)
                self.market_context.update({
                    'regime': context.get('regime', 'unknown'),
                    'session': context.get('session', 'unknown'),
                    'volatility_level': context.get('volatility_level', 'medium')
                })
                
                # Check market open status
                market_open = context.get('market_open', True)
                if not market_open:
                    self.log_operator_info("📴 Market closed - abstaining from theme signals")
                    return self._zero.copy()
            
            # Market-closed check (legacy)
            if extras and not extras.get("market_open", True):
                self.log_operator_info("📴 Market closed - abstaining from theme signals")
                return self._zero.copy()

            # Detect current theme
            try:
                theme, strength = self.det.detect(self.env.data, self.env.current_step)
                self._last_theme = theme
                self._last_strength = float(np.clip(strength, 0.0, 1.0))
            except Exception as e:
                self.log_operator_warning(f"Theme detection failed: {e}")
                theme, strength = 0, 0.0
                self._last_theme = 0
                self._last_strength = 0.0

            # Get enhanced theme characteristics
            current_theme = getattr(self.det, '_current_theme', theme)
            
            # Generate action based on theme
            vec = self._generate_theme_action(current_theme, self._last_strength)
            
            # Apply market context adjustments
            vec = self._apply_market_context_adjustments(vec)
            
            # Record theme signal
            self._record_theme_signal(current_theme, self._last_strength, vec)
            
            return vec
            
        except Exception as e:
            self.log_operator_error(f"Theme action proposal failed: {e}")
            return self._zero.copy()

    def _generate_theme_action(self, theme: int, strength: float) -> np.ndarray:
        """Generate action based on current theme"""
        
        vec = self._zero.copy()
        
        try:
            if theme == 0:  # Risk-on theme
                # Tend to be long risk assets
                for i in range(0, self.env.action_dim, 2):
                    vec[i] = strength * self.max_size * 0.8
                    vec[i+1] = 0.5  # Medium duration
                
                self.log_operator_info(
                    f"🔥 Risk-on theme detected",
                    strength=f"{strength:.1%}",
                    signal="Long risk assets"
                )
                
            elif theme == 1:  # Risk-off theme
                # EUR/USD short, XAU/USD long (safe haven)
                if self.env.action_dim >= 4:
                    vec[0] = -strength * self.max_size * 0.7  # Short EUR
                    vec[1] = 0.5
                    vec[2] = strength * self.max_size * 0.9   # Long XAU
                    vec[3] = 0.7  # Longer duration for safe haven
                
                self.log_operator_info(
                    f"🛡️ Risk-off theme detected",
                    strength=f"{strength:.1%}",
                    signal="EUR short, XAU long"
                )
                
            elif theme == 2:  # High volatility theme
                # Smaller positions, shorter durations
                for i in range(0, self.env.action_dim, 2):
                    # Alternate directions for diversification
                    direction = 1 if (i//2) % 2 == 0 else -1
                    vec[i] = direction * strength * self.max_size * 0.4
                    vec[i+1] = 0.3  # Short duration
                
                self.log_operator_info(
                    f"💥 High volatility theme detected",
                    strength=f"{strength:.1%}",
                    signal="Small diversified positions"
                )
                
            elif theme == 3:  # Trending theme
                # Follow trend with higher confidence
                for i in range(0, self.env.action_dim, 2):
                    vec[i] = strength * self.max_size
                    vec[i+1] = 0.7  # Longer duration for trends
                
                self.log_operator_info(
                    f"📈 Trending theme detected",
                    strength=f"{strength:.1%}",
                    signal="Trend following"
                )
            
            else:
                self.log_operator_info(
                    f"❓ Unknown theme {theme}",
                    strength=f"{strength:.1%}",
                    signal="Neutral position"
                )
                
        except Exception as e:
            self.log_operator_warning(f"Theme action generation failed: {e}")
        
        return vec

    def _apply_market_context_adjustments(self, vec: np.ndarray) -> np.ndarray:
        """Apply adjustments based on market context"""
        
        try:
            regime = self.market_context.get('regime', 'unknown')
            session = self.market_context.get('session', 'unknown')
            vol_level = self.market_context.get('volatility_level', 'medium')
            
            # Regime-based adjustments
            if regime == 'volatile':
                vec *= 0.7  # Reduce position sizes in volatile regimes
            elif regime == 'trending':
                vec *= 1.2  # Increase in trending regimes
            elif regime == 'noise':
                vec *= 0.5  # Very conservative in noisy markets
            
            # Session-based adjustments
            if session == 'rollover':
                vec *= 0.3  # Very small positions during rollover
            elif session == 'american':
                vec *= 1.1  # Slightly larger during high liquidity
            
            # Volatility-based adjustments
            if vol_level == 'extreme':
                vec *= 0.4
            elif vol_level == 'high':
                vec *= 0.7
            elif vol_level == 'low':
                vec *= 1.2
            
        except Exception as e:
            self.log_operator_warning(f"Market context adjustment failed: {e}")
        
        return vec

    def _record_theme_signal(self, theme: int, strength: float, action: np.ndarray) -> None:
        """Record theme signal for performance tracking"""
        
        try:
            # Update theme performance tracking
            if theme in self.theme_performance:
                perf_data = self.theme_performance[theme]
                perf_data['signals'] += 1
                
                # Update average strength
                count = perf_data['signals']
                old_avg = perf_data['avg_strength']
                perf_data['avg_strength'] = (old_avg * (count - 1) + strength) / count
            
            # Record in history
            signal_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'theme': theme,
                'strength': strength,
                'action_magnitude': float(np.linalg.norm(action)),
                'market_context': self.market_context.copy()
            }
            self._theme_history.append(signal_record)
            
        except Exception as e:
            self.log_operator_warning(f"Theme signal recording failed: {e}")

    def confidence(self, obs: Any = None, extras: Dict = None, info_bus: InfoBus = None) -> float:
        """Enhanced confidence calculation with performance feedback"""
        
        try:
            # Base confidence from theme mapping
            theme_confidence_map = {0: 0.7, 1: 0.8, 2: 0.5, 3: 0.9}
            base_conf = theme_confidence_map.get(self._last_theme, 0.5)
            
            # Adjust for strength
            strength_adjusted = base_conf * self._last_strength
            
            # Adjust based on historical performance
            if self._last_theme in self.theme_performance:
                perf_data = self.theme_performance[self._last_theme]
                if perf_data['signals'] > 5:
                    success_rate = perf_data['success'] / perf_data['signals']
                    performance_multiplier = 0.5 + success_rate  # 0.5 to 1.5 range
                    strength_adjusted *= performance_multiplier
            
            # Market context adjustments
            regime = self.market_context.get('regime', 'unknown')
            if regime == 'volatile' and self._last_theme == 2:
                strength_adjusted *= 1.2  # Higher confidence in vol theme during volatile regime
            elif regime == 'trending' and self._last_theme == 3:
                strength_adjusted *= 1.3  # Higher confidence in trend theme during trending regime
            
            return float(np.clip(strength_adjusted, 0.1, 1.0))
            
        except Exception as e:
            self.log_operator_warning(f"Confidence calculation failed: {e}")
            return 0.5

    def update_performance(self, outcome: Dict[str, Any]) -> None:
        """Update performance tracking based on trade outcomes"""
        
        try:
            pnl = outcome.get('pnl', 0.0)
            theme = outcome.get('theme', self._last_theme)
            
            if theme in self.theme_performance:
                if pnl > 0:
                    self.theme_performance[theme]['success'] += 1
                
                # Record performance
                self._performance_history.append({
                    'theme': theme,
                    'pnl': pnl,
                    'timestamp': datetime.datetime.now().isoformat()
                })
            
        except Exception as e:
            self.log_operator_warning(f"Performance update failed: {e}")

    def get_state(self) -> Dict[str, Any]:
        """Get theme expert state for serialization"""
        return {
            "_last_strength": self._last_strength,
            "_last_theme": self._last_theme,
            "theme_performance": self.theme_performance.copy(),
            "market_context": self.market_context.copy(),
            "theme_history": list(self._theme_history)[-20:],
            "performance_history": list(self._performance_history)[-20:]
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load theme expert state from serialization"""
        self._last_strength = state.get("_last_strength", 0.0)
        self._last_theme = state.get("_last_theme", 0)
        self.theme_performance.update(state.get("theme_performance", {}))
        self.market_context.update(state.get("market_context", {}))
        
        # Restore history
        theme_history = state.get("theme_history", [])
        self._theme_history.clear()
        for entry in theme_history:
            self._theme_history.append(entry)
        
        performance_history = state.get("performance_history", [])
        self._performance_history.clear()
        for entry in performance_history:
            self._performance_history.append(entry)

    def log_operator_info(self, message: str, **kwargs):
        """Log operator message"""
        if self.debug and hasattr(self, 'logger'):
            self.logger.info(format_operator_message(message, **kwargs))

    def log_operator_warning(self, message: str, **kwargs):
        """Log operator warning"""
        if hasattr(self, 'logger'):
            self.logger.warning(format_operator_message(message, **kwargs))

    def log_operator_error(self, message: str, **kwargs):
        """Log operator error"""
        if hasattr(self, 'logger'):
            self.logger.error(format_operator_message(message, **kwargs))


class SeasonalityRiskExpert:
    """
    Enhanced seasonality-based risk expert with InfoBus integration.
    Provides position signals based on market seasonality patterns.
    """

    def __init__(self, tars_module, env_ref, base_signal: float = 0.3, debug: bool = True):
        self.tars = tars_module
        self.env = env_ref
        self.base_signal = base_signal
        self.debug = debug
        
        # Enhanced tracking
        self._zero = np.zeros(self.env.action_dim, dtype=np.float32)
        self._seasonality_history = deque(maxlen=100)
        self._performance_by_session = {
            'american': {'signals': 0, 'success': 0},
            'european': {'signals': 0, 'success': 0},
            'asian': {'signals': 0, 'success': 0},
            'rollover': {'signals': 0, 'success': 0}
        }
        
        # Setup logging
        self.logger = RotatingLogger(
            "SeasonalityRiskExpert",
            "logs/voting/seasonality_expert.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        self.log_operator_info(
            "🕐 Seasonality Risk Expert initialized",
            base_signal=self.base_signal,
            action_dim=self.env.action_dim
        )

    def reset(self):
        """Enhanced reset"""
        self._seasonality_history.clear()
        for session_data in self._performance_by_session.values():
            session_data.update({'signals': 0, 'success': 0})
        
        self.log_operator_info("🔄 Seasonality Expert reset")

    def propose_action(self, obs: Any, extras: Dict = None, info_bus: InfoBus = None) -> np.ndarray:
        """Enhanced seasonality-based action proposal"""
        
        try:
            # Get seasonality factor
            factor = getattr(self.tars, "seasonality_factor", 1.0)
            try:
                factor = float(factor)
                if not math.isfinite(factor):
                    factor = 1.0
            except (ValueError, TypeError):
                factor = 1.0
            
            # Get current session from InfoBus if available
            current_session = 'unknown'
            if info_bus:
                context = extract_standard_context(info_bus)
                current_session = context.get('session', 'unknown')
            
            vec = self._zero.copy()
            
            # Generate trading signals scaled by seasonality
            for i in range(0, self.env.action_dim, 2):
                if i == 0:  # EUR/USD - trade with session flow
                    direction = 1 if factor > 1.0 else -1
                    signal_strength = self.base_signal * factor
                else:  # XAU/USD - counter-trade weak sessions
                    direction = -1 if factor < 1.0 else 1
                    signal_strength = self.base_signal * factor * 0.8  # Slightly smaller for gold
                
                vec[i] = direction * signal_strength
                vec[i+1] = 0.5  # Standard duration
            
            # Apply session-specific adjustments
            vec = self._apply_session_adjustments(vec, current_session, factor)
            
            # Record seasonality signal
            self._record_seasonality_signal(factor, current_session, vec)
            
            self.log_operator_info(
                f"🕐 Seasonality signal generated",
                factor=f"{factor:.2f}",
                session=current_session,
                signal_strength=f"{np.linalg.norm(vec):.3f}"
            )
            
            return vec
            
        except Exception as e:
            self.log_operator_error(f"Seasonality action proposal failed: {e}")
            return self._zero.copy()

    def _apply_session_adjustments(self, vec: np.ndarray, session: str, factor: float) -> np.ndarray:
        """Apply session-specific adjustments"""
        
        try:
            session_multipliers = {
                'american': 1.0,    # Base multiplier
                'european': 0.9,    # Slightly more conservative
                'asian': 0.8,       # Lower volatility
                'rollover': 0.3,    # Very conservative
                'unknown': 0.7      # Conservative default
            }
            
            multiplier = session_multipliers.get(session, 0.7)
            
            # Additional factor-based adjustments
            if factor > 1.5:
                multiplier *= 1.2  # Boost strong seasonality
            elif factor < 0.5:
                multiplier *= 0.8  # Reduce weak seasonality
            
            return vec * multiplier
            
        except Exception as e:
            self.log_operator_warning(f"Session adjustment failed: {e}")
            return vec

    def _record_seasonality_signal(self, factor: float, session: str, action: np.ndarray) -> None:
        """Record seasonality signal for tracking"""
        
        try:
            signal_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'factor': factor,
                'session': session,
                'action_magnitude': float(np.linalg.norm(action))
            }
            self._seasonality_history.append(signal_record)
            
            # Update session performance tracking
            if session in self._performance_by_session:
                self._performance_by_session[session]['signals'] += 1
            
        except Exception as e:
            self.log_operator_warning(f"Seasonality signal recording failed: {e}")

    def confidence(self, obs: Any = None, extras: Dict = None, info_bus: InfoBus = None) -> float:
        """Enhanced confidence based on seasonality strength and session performance"""
        
        try:
            factor = float(getattr(self.tars, "seasonality_factor", 1.0))
            
            # Base confidence from factor deviation
            deviation = abs(factor - 1.0)
            base_confidence = 0.5 + deviation * 0.5
            
            # Get current session
            current_session = 'unknown'
            if info_bus:
                context = extract_standard_context(info_bus)
                current_session = context.get('session', 'unknown')
            
            # Adjust for session performance
            if current_session in self._performance_by_session:
                session_data = self._performance_by_session[current_session]
                if session_data['signals'] > 5:
                    success_rate = session_data['success'] / session_data['signals']
                    performance_multiplier = 0.7 + success_rate * 0.6  # 0.7 to 1.3 range
                    base_confidence *= performance_multiplier
            
            return float(np.clip(base_confidence, 0.3, 0.9))
            
        except Exception as e:
            self.log_operator_warning(f"Confidence calculation failed: {e}")
            return 0.5

    def log_operator_info(self, message: str, **kwargs):
        """Log operator message"""
        if self.debug and hasattr(self, 'logger'):
            self.logger.info(format_operator_message(message, **kwargs))

    def log_operator_warning(self, message: str, **kwargs):
        """Log operator warning"""
        if hasattr(self, 'logger'):
            self.logger.warning(format_operator_message(message, **kwargs))

    def log_operator_error(self, message: str, **kwargs):
        """Log operator error"""
        if hasattr(self, 'logger'):
            self.logger.error(format_operator_message(message, **kwargs))


class MetaRLExpert:
    """
    Enhanced Meta-RL expert wrapper with InfoBus integration.
    Wraps MetaRLController for committee voting.
    """

    def __init__(self, meta_rl, env_ref, debug: bool = True):
        self.mrl = meta_rl
        self.env = env_ref
        self.debug = debug
        
        # Enhanced state tracking
        self.last_action = np.zeros(self.env.action_dim, dtype=np.float32)
        self.last_entropy = 0.5
        self.action_history = deque(maxlen=50)
        self.performance_metrics = {
            'actions_generated': 0,
            'avg_entropy': 0.5,
            'confidence_trend': deque(maxlen=20)
        }
        
        # Setup logging
        self.logger = RotatingLogger(
            "MetaRLExpert",
            "logs/voting/meta_rl_expert.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        self.log_operator_info(
            "🧠 Meta-RL Expert initialized",
            action_dim=self.env.action_dim
        )

    def reset(self):
        """Enhanced reset"""
        self.last_action.fill(0.0)
        self.last_entropy = 0.5
        self.action_history.clear()
        
        # Reset performance metrics
        self.performance_metrics = {
            'actions_generated': 0,
            'avg_entropy': 0.5,
            'confidence_trend': deque(maxlen=20)
        }
        
        self.log_operator_info("🔄 Meta-RL Expert reset")

    def propose_action(self, obs: np.ndarray, extras: dict = None, info_bus: InfoBus = None) -> np.ndarray:
        """Enhanced action proposal with comprehensive error handling"""
        
        try:
            if isinstance(obs, np.ndarray):
                action = self._call_controller(obs)
                
                # Record action for analysis
                self._record_action(action)
                
                return action
            else:
                self.log_operator_warning("Invalid observation type - using last action")
                return self.last_action.copy()
                
        except Exception as e:
            self.log_operator_error(f"Meta-RL action proposal failed: {e}")
            return self.last_action.copy()

    def _call_controller(self, obs_vec: np.ndarray) -> np.ndarray:
        """Enhanced controller call with comprehensive error handling"""
        
        try:
            # Ensure obs length matches controller requirements
            obs_dim = getattr(self.mrl, "obs_size", obs_vec.size)
            if callable(obs_dim):
                obs_dim = obs_dim()
            
            flat = np.asarray(obs_vec, dtype=np.float32).ravel()
            if flat.size < obs_dim:
                flat = np.pad(flat, (0, obs_dim - flat.size))
            elif flat.size > obs_dim:
                flat = flat[:obs_dim]

            # Build torch tensor
            device = getattr(self.mrl, "device", "cpu")
            obs_t = torch.tensor(flat, dtype=torch.float32, device=device).unsqueeze(0)

            # Check for market_lags parameter
            kwargs = {}
            sig = inspect.signature(self.mrl.act)
            if "market_lags" in sig.parameters:
                lags = getattr(self.env, "market_lags", None)
                if not isinstance(lags, np.ndarray):
                    lags = np.zeros_like(flat, dtype=np.float32)
                ml_t = torch.tensor(lags, dtype=torch.float32, device=device).unsqueeze(0)
                kwargs["market_lags"] = ml_t

            # Call controller
            with torch.no_grad():
                action = self.mrl.act(obs_t, **kwargs)

            # Convert to numpy
            if torch.is_tensor(action):
                act = action.squeeze(0).cpu().numpy()
            else:
                act = np.asarray(action, dtype=np.float32).ravel()

            # Capture entropy if available
            ent = getattr(self.mrl, "last_entropy", None)
            if isinstance(ent, float):
                self.last_entropy = ent

            # Ensure correct size
            if act.size < self.env.action_dim:
                act = np.pad(act, (0, self.env.action_dim - act.size))
            elif act.size > self.env.action_dim:
                act = act[:self.env.action_dim]

            self.last_action = act.copy()
            
            self.log_operator_info(
                f"🧠 Meta-RL action generated",
                entropy=f"{self.last_entropy:.3f}",
                action_magnitude=f"{np.linalg.norm(act):.3f}"
            )
            
            return act

        except Exception as e:
            self.log_operator_error(f"Controller call failed: {e}")
            return np.zeros(self.env.action_dim, dtype=np.float32)

    def _record_action(self, action: np.ndarray) -> None:
        """Record action for performance tracking"""
        
        try:
            action_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'action': action.tolist(),
                'entropy': self.last_entropy,
                'magnitude': float(np.linalg.norm(action))
            }
            self.action_history.append(action_record)
            
            # Update performance metrics
            self.performance_metrics['actions_generated'] += 1
            
            # Update average entropy
            count = self.performance_metrics['actions_generated']
            old_avg = self.performance_metrics['avg_entropy']
            self.performance_metrics['avg_entropy'] = (old_avg * (count - 1) + self.last_entropy) / count
            
            # Track confidence trend
            confidence = 1.0 - self.last_entropy
            self.performance_metrics['confidence_trend'].append(confidence)
            
        except Exception as e:
            self.log_operator_warning(f"Action recording failed: {e}")

    def confidence(self, obs: np.ndarray = None, extras: dict = None, info_bus: InfoBus = None) -> float:
        """Enhanced confidence calculation with trend analysis"""
        
        try:
            # Base confidence from entropy
            base_confidence = np.clip(1.0 - self.last_entropy, 0.0, 1.0)
            
            # Adjust based on recent confidence trend
            if len(self.performance_metrics['confidence_trend']) > 5:
                recent_confidences = list(self.performance_metrics['confidence_trend'])[-5:]
                trend_slope = np.polyfit(range(len(recent_confidences)), recent_confidences, 1)[0]
                
                # Boost confidence if improving trend
                if trend_slope > 0.01:
                    base_confidence *= 1.1
                elif trend_slope < -0.01:
                    base_confidence *= 0.9
            
            return float(np.clip(base_confidence, 0.0, 1.0))
            
        except Exception as e:
            self.log_operator_warning(f"Confidence calculation failed: {e}")
            return 0.5

    def log_operator_info(self, message: str, **kwargs):
        """Log operator message"""
        if self.debug and hasattr(self, 'logger'):
            self.logger.info(format_operator_message(message, **kwargs))

    def log_operator_warning(self, message: str, **kwargs):
        """Log operator warning"""
        if hasattr(self, 'logger'):
            self.logger.warning(format_operator_message(message, **kwargs))

    def log_operator_error(self, message: str, **kwargs):
        """Log operator error"""
        if hasattr(self, 'logger'):
            self.logger.error(format_operator_message(message, **kwargs))


class TradeMonitorVetoExpert:
    """
    Enhanced trade monitor expert with InfoBus integration.
    Provides risk-adjusted signals based on active trade monitoring.
    """

    def __init__(self, monitor, env_ref, max_signal: float = 0.5, debug: bool = True):
        self.mon = monitor
        self.env = env_ref
        self.max_signal = max_signal
        self.debug = debug
        
        # Enhanced tracking
        self._zero = np.zeros(self.env.action_dim, dtype=np.float32)
        self.signal_history = deque(maxlen=50)
        self.risk_episodes = deque(maxlen=30)
        
        # Setup logging
        self.logger = RotatingLogger(
            "TradeMonitorVetoExpert",
            "logs/voting/trade_monitor_expert.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        self.log_operator_info(
            "🚨 Trade Monitor Veto Expert initialized",
            max_signal=self.max_signal,
            action_dim=self.env.action_dim
        )

    def reset(self):
        """Enhanced reset"""
        self.signal_history.clear()
        self.risk_episodes.clear()
        self.log_operator_info("🔄 Trade Monitor Expert reset")

    def propose_action(self, obs: Any, extras: Dict = None, info_bus: InfoBus = None) -> np.ndarray:
        """Enhanced risk-adjusted action proposal"""
        
        try:
            # Get monitor state
            alerted = getattr(self.mon, "alerted", False)
            risk_score = getattr(self.mon, "risk_score", 0.0)
            
            vec = self._zero.copy()
            
            if alerted:
                # When alerted, suggest position management
                vec = self._generate_risk_management_signal(risk_score, info_bus)
                
                self.log_operator_warning(
                    f"🚨 Risk alert active - risk management mode",
                    risk_score=f"{risk_score:.1%}",
                    signal="Position adjustment"
                )
                
                # Record risk episode
                self.risk_episodes.append({
                    'timestamp': datetime.datetime.now().isoformat(),
                    'risk_score': risk_score,
                    'action_taken': 'risk_management'
                })
                
            else:
                # Normal mode - provide risk-adjusted signals
                vec = self._generate_normal_trading_signal(risk_score, info_bus)
                
                self.log_operator_info(
                    f"✅ Normal trading mode",
                    risk_score=f"{risk_score:.1%}",
                    signal="Risk-adjusted trading"
                )
            
            # Record signal
            self._record_signal(alerted, risk_score, vec)
            
            return vec
            
        except Exception as e:
            self.log_operator_error(f"Trade monitor action proposal failed: {e}")
            return self._zero.copy()

    def _generate_risk_management_signal(self, risk_score: float, info_bus: InfoBus = None) -> np.ndarray:
        """Generate risk management signals when alerted"""
        
        vec = self._zero.copy()
        
        try:
            # Get current positions from InfoBus if available
            if info_bus:
                positions = InfoBusExtractor.get_positions(info_bus)
                
                # Generate closing signals for risky positions
                for i, pos in enumerate(positions):
                    if i * 2 < self.env.action_dim:
                        current_size = pos.get('size', 0.0)
                        if abs(current_size) > 0.1:  # Significant position
                            # Suggest partial or full closing
                            close_ratio = min(1.0, risk_score * 2)  # Higher risk = more closing
                            vec[i * 2] = -np.sign(current_size) * close_ratio * self.max_signal
                            vec[i * 2 + 1] = 0.2  # Quick exit
            else:
                # Fallback: general risk reduction
                for i in range(0, self.env.action_dim, 2):
                    vec[i] = -self.max_signal * risk_score * 0.5  # Small closing bias
                    vec[i + 1] = 0.3
                    
        except Exception as e:
            self.log_operator_warning(f"Risk management signal generation failed: {e}")
        
        return vec

    def _generate_normal_trading_signal(self, risk_score: float, info_bus: InfoBus = None) -> np.ndarray:
        """Generate normal trading signals adjusted for risk"""
        
        vec = self._zero.copy()
        
        try:
            # Risk-adjusted signal strength
            risk_multiplier = 1.0 - risk_score * 0.7  # Higher risk = lower signal
            
            # Get market momentum if available
            momentum_signals = []
            if hasattr(self.env, 'last_actions') and len(self.env.last_actions) > 0:
                momentum_signals = self.env.last_actions
            
            for i in range(0, self.env.action_dim, 2):
                if i // 2 < len(momentum_signals):
                    # Follow recent momentum but scale by risk
                    momentum = momentum_signals[i // 2]
                    vec[i] = momentum * risk_multiplier * 0.7
                else:
                    # Default small exploratory position
                    direction = 1 if (i // 2) % 2 == 0 else -1
                    vec[i] = direction * self.max_signal * risk_multiplier * 0.3
                
                # Duration based on risk
                vec[i + 1] = 0.5 + risk_score * 0.3  # Longer duration when risky (patience)
                
        except Exception as e:
            self.log_operator_warning(f"Normal trading signal generation failed: {e}")
        
        return vec

    def _record_signal(self, alerted: bool, risk_score: float, action: np.ndarray) -> None:
        """Record signal for tracking"""
        
        try:
            signal_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'alerted': alerted,
                'risk_score': risk_score,
                'action_magnitude': float(np.linalg.norm(action)),
                'mode': 'risk_management' if alerted else 'normal'
            }
            self.signal_history.append(signal_record)
            
        except Exception as e:
            self.log_operator_warning(f"Signal recording failed: {e}")

    def confidence(self, obs: Any = None, extras: Dict = None, info_bus: InfoBus = None) -> float:
        """Enhanced confidence based on risk state and historical effectiveness"""
        
        try:
            alerted = getattr(self.mon, "alerted", False)
            risk_score = getattr(self.mon, "risk_score", 0.0)
            
            if alerted:
                # High confidence when managing risk
                base_confidence = 0.8 + risk_score * 0.2
            else:
                # Confidence inversely related to risk
                base_confidence = 0.6 * (1.0 - risk_score * 0.5)
            
            # Adjust based on recent effectiveness
            if len(self.risk_episodes) > 3:
                # Could implement effectiveness tracking here
                pass
            
            return float(np.clip(base_confidence, 0.2, 0.9))
            
        except Exception as e:
            self.log_operator_warning(f"Confidence calculation failed: {e}")
            return 0.5

    def log_operator_info(self, message: str, **kwargs):
        """Log operator message"""
        if self.debug and hasattr(self, 'logger'):
            self.logger.info(format_operator_message(message, **kwargs))

    def log_operator_warning(self, message: str, **kwargs):
        """Log operator warning"""
        if hasattr(self, 'logger'):
            self.logger.warning(format_operator_message(message, **kwargs))

    def log_operator_error(self, message: str, **kwargs):
        """Log operator error"""
        if hasattr(self, 'logger'):
            self.logger.error(format_operator_message(message, **kwargs))


class RegimeBiasExpert:
    """
    Enhanced regime bias expert with InfoBus integration.
    Provides directional bias based on market regime analysis.
    """

    def __init__(self, frc_module, env_ref, max_size: float = 0.6, debug: bool = True):
        self.frc = frc_module
        self.env = env_ref
        self.max_size = max_size
        self.debug = debug
        
        # Enhanced tracking
        self._zero = np.zeros(self.env.action_dim, dtype=np.float32)
        self.regime_history = deque(maxlen=100)
        self.regime_performance = {
            'trending': {'signals': 0, 'success': 0},
            'volatile': {'signals': 0, 'success': 0},
            'noise': {'signals': 0, 'success': 0}
        }
        
        # Setup logging
        self.logger = RotatingLogger(
            "RegimeBiasExpert",
            "logs/voting/regime_bias_expert.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        self.log_operator_info(
            "📊 Regime Bias Expert initialized",
            max_size=self.max_size,
            action_dim=self.env.action_dim
        )

    def reset(self):
        """Enhanced reset"""
        self.regime_history.clear()
        for regime_data in self.regime_performance.values():
            regime_data.update({'signals': 0, 'success': 0})
        
        self.log_operator_info("🔄 Regime Bias Expert reset")

    def propose_action(self, obs: Any, extras: Dict = None, info_bus: InfoBus = None) -> np.ndarray:
        """Enhanced regime-based action proposal"""
        
        try:
            # Get regime information
            regime_label = getattr(self.frc, "label", "noise")
            regime_strength = float(getattr(self.frc, "regime_strength", 0.5))
            trend_direction = float(getattr(self.frc, "_trend_direction", 0.0))
            
            vec = self._zero.copy()
            
            if regime_label == "trending":
                vec = self._generate_trending_signal(regime_strength, trend_direction)
                
                self.log_operator_info(
                    f"📈 Trending regime detected",
                    strength=f"{regime_strength:.1%}",
                    direction=f"{trend_direction:+.2f}"
                )
                
            elif regime_label == "volatile":
                vec = self._generate_volatile_signal(regime_strength, trend_direction)
                
                self.log_operator_info(
                    f"💥 Volatile regime detected",
                    strength=f"{regime_strength:.1%}",
                    strategy="Mean reversion"
                )
                
            else:  # noise regime
                vec = self._generate_noise_signal(regime_strength)
                
                self.log_operator_info(
                    f"🌫️ Noise regime detected",
                    strength=f"{regime_strength:.1%}",
                    strategy="Conservative exploration"
                )
            
            # Record regime signal
            self._record_regime_signal(regime_label, regime_strength, trend_direction, vec)
            
            return vec
            
        except Exception as e:
            self.log_operator_error(f"Regime bias action proposal failed: {e}")
            return self._zero.copy()

    def _generate_trending_signal(self, strength: float, direction: float) -> np.ndarray:
        """Generate signals for trending regime"""
        
        vec = self._zero.copy()
        
        try:
            # Strong directional bias based on trend
            for i in range(0, self.env.action_dim, 2):
                signal_direction = np.sign(direction) if direction != 0 else 1
                vec[i] = signal_direction * strength * self.max_size
                vec[i + 1] = 0.7  # Longer holds in trends
                
        except Exception as e:
            self.log_operator_warning(f"Trending signal generation failed: {e}")
        
        return vec

    def _generate_volatile_signal(self, strength: float, direction: float) -> np.ndarray:
        """Generate signals for volatile regime"""
        
        vec = self._zero.copy()
        
        try:
            # Mean reversion strategy
            for i in range(0, self.env.action_dim, 2):
                if i == 0:  # EUR/USD - fade moves
                    signal_direction = -np.sign(direction) if direction != 0 else 1
                    vec[i] = signal_direction * strength * self.max_size * 0.5
                else:  # XAU/USD - momentum
                    signal_direction = np.sign(direction) if direction != 0 else -1
                    vec[i] = signal_direction * strength * self.max_size * 0.5
                
                vec[i + 1] = 0.3  # Shorter holds in volatile regime
                
        except Exception as e:
            self.log_operator_warning(f"Volatile signal generation failed: {e}")
        
        return vec

    def _generate_noise_signal(self, strength: float) -> np.ndarray:
        """Generate signals for noise regime"""
        
        vec = self._zero.copy()
        
        try:
            # Very conservative, small exploratory positions
            for i in range(0, self.env.action_dim, 2):
                direction = np.random.choice([-1, 1])
                vec[i] = direction * strength * self.max_size * 0.2
                vec[i + 1] = 0.5  # Medium duration
                
        except Exception as e:
            self.log_operator_warning(f"Noise signal generation failed: {e}")
        
        return vec

    def _record_regime_signal(self, regime: str, strength: float, direction: float, action: np.ndarray) -> None:
        """Record regime signal for tracking"""
        
        try:
            signal_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'regime': regime,
                'strength': strength,
                'direction': direction,
                'action_magnitude': float(np.linalg.norm(action))
            }
            self.regime_history.append(signal_record)
            
            # Update regime performance tracking
            if regime in self.regime_performance:
                self.regime_performance[regime]['signals'] += 1
            
        except Exception as e:
            self.log_operator_warning(f"Regime signal recording failed: {e}")

    def confidence(self, obs: Any = None, extras: Dict = None, info_bus: InfoBus = None) -> float:
        """Enhanced confidence based on regime clarity and historical performance"""
        
        try:
            regime_label = getattr(self.frc, "label", "noise")
            regime_strength = float(getattr(self.frc, "regime_strength", 0.5))
            
            # Base confidence mapping
            confidence_map = {
                "trending": 0.8,
                "volatile": 0.6,
                "noise": 0.3
            }
            
            base_conf = confidence_map.get(regime_label, 0.5)
            strength_adjusted = base_conf * regime_strength
            
            # Adjust based on historical performance
            if regime_label in self.regime_performance:
                perf_data = self.regime_performance[regime_label]
                if perf_data['signals'] > 5:
                    success_rate = perf_data['success'] / perf_data['signals']
                    performance_multiplier = 0.5 + success_rate  # 0.5 to 1.5 range
                    strength_adjusted *= performance_multiplier
            
            return float(np.clip(strength_adjusted, 0.1, 0.9))
            
        except Exception as e:
            self.log_operator_warning(f"Confidence calculation failed: {e}")
            return 0.5

    def log_operator_info(self, message: str, **kwargs):
        """Log operator message"""
        if self.debug and hasattr(self, 'logger'):
            self.logger.info(format_operator_message(message, **kwargs))

    def log_operator_warning(self, message: str, **kwargs):
        """Log operator warning"""
        if hasattr(self, 'logger'):
            self.logger.warning(format_operator_message(message, **kwargs))

    def log_operator_error(self, message: str, **kwargs):
        """Log operator error"""
        if hasattr(self, 'logger'):
            self.logger.error(format_operator_message(message, **kwargs))


def create_all_experts(env, modules_dict: Dict[str, Any]) -> List[Any]:
    """
    Create all expert wrappers for the voting committee with enhanced integration.
    
    Args:
        env: The trading environment
        modules_dict: Dictionary of initialized modules
        
    Returns:
        List of expert wrapper instances
    """
    experts = []
    
    try:
        # Theme expert
        if "theme_detector" in modules_dict:
            experts.append(ThemeExpert(
                modules_dict["theme_detector"],
                env,
                debug=env.config.debug
            ))
        
        # Seasonality expert
        if "time_risk_scaler" in modules_dict:
            experts.append(SeasonalityRiskExpert(
                modules_dict["time_risk_scaler"],
                env,
                debug=env.config.debug
            ))
        
        # Meta-RL expert
        if "meta_rl" in modules_dict:
            experts.append(MetaRLExpert(
                modules_dict["meta_rl"],
                env,
                debug=env.config.debug
            ))
        
        # Trade monitor expert
        if "active_monitor" in modules_dict:
            experts.append(TradeMonitorVetoExpert(
                modules_dict["active_monitor"],
                env,
                debug=env.config.debug
            ))
        
        # Regime bias expert
        if "fractal_confirm" in modules_dict:
            experts.append(RegimeBiasExpert(
                modules_dict["fractal_confirm"],
                env,
                debug=env.config.debug
            ))
        
        # Log successful creation
        expert_names = [type(expert).__name__ for expert in experts]
        if hasattr(env, 'logger'):
            env.logger.info(f"Created {len(experts)} voting experts: {', '.join(expert_names)}")
        
    except Exception as e:
        if hasattr(env, 'logger'):
            env.logger.error(f"Expert creation failed: {e}")
    
    return experts