# modules/strategy/voting_wrappers.py

import numpy as np
from typing import Any, List, Dict
import math
import torch
import logging

from modules.risk.risk_monitor import ActiveTradeMonitor
from modules.strategy.strategy import MetaRLController

# Utility for action vector construction
def _dir_size_to_vec(angle: float, magnitude: float, n_instruments: int, action_dim: int) -> np.ndarray:
    vec = np.zeros(action_dim, np.float32)
    if magnitude <= 0:
        return vec
    idx = int((angle % 360) / 360 * n_instruments) % n_instruments
    vec[2 * idx] = magnitude
    vec[2 * idx + 1] = 0.5
    return vec

# ========== 1. MarketThemeDetector → ThemeExpert ==========
class ThemeExpert:
    """
    Converts (label,strength) from MarketThemeDetector into an action.
    FIXED: Now trades on all instruments, not just the first one.
    """
    def __init__(self, detector, env_ref, trend_label: str = "trending", max_size: float = 0.7, debug=True):
        self.det         = detector
        self.env         = env_ref
        self.trend_label = trend_label
        self.max_size    = max_size
        self._last_strength = 0.0
        self._last_theme = 0
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug
        self.logger = logging.getLogger("ThemeExpert")
        if self.debug:
            has_stream = any(isinstance(h, logging.StreamHandler) for h in self.logger.handlers)
            if not has_stream:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
                self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    def reset(self):
        self._last_strength = 0.0
        self._last_theme = 0

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        # Market-closed check
        if extras and not extras.get("market_open", True):
            if self.debug:
                self.logger.debug("Market closed: abstaining from action.")
            return self._zero.copy()

        try:
            theme, stren = self.det.detect(self.env.data, self.env.current_step)
            self._last_theme = theme
        except Exception as e:
            theme, stren = 0, 0.0
            if self.debug:
                self.logger.debug(f"Detector error: {e}")

        self._last_strength = float(np.clip(stren, 0.0, 1.0))
        
        # Get theme characteristics from detector if available
        if hasattr(self.det, '_current_theme'):
            current_theme = self.det._current_theme
        else:
            current_theme = theme
            
        vec = self._zero.copy()
        
        # Apply theme-based strategy to ALL instruments
        if current_theme == 0:  # Risk-on theme
            # Tend to be long risk assets
            for i in range(0, self.env.action_dim, 2):
                vec[i] = self._last_strength * self.max_size * 0.8
                vec[i+1] = 0.5  # Medium duration
                
        elif current_theme == 1:  # Risk-off theme
            # EUR/USD short, XAU/USD long (assuming standard order)
            if self.env.action_dim >= 4:
                vec[0] = -self._last_strength * self.max_size * 0.7  # Short EUR
                vec[1] = 0.5
                vec[2] = self._last_strength * self.max_size * 0.9   # Long XAU
                vec[3] = 0.7  # Longer duration for safe haven
                
        elif current_theme == 2:  # High volatility theme
            # Smaller positions, shorter durations
            for i in range(0, self.env.action_dim, 2):
                # Alternate directions for diversification
                direction = 1 if (i//2) % 2 == 0 else -1
                vec[i] = direction * self._last_strength * self.max_size * 0.4
                vec[i+1] = 0.3  # Short duration
                
        elif current_theme == 3:  # Trending theme
            # Follow trend with higher confidence
            for i in range(0, self.env.action_dim, 2):
                vec[i] = self._last_strength * self.max_size
                vec[i+1] = 0.7  # Longer duration for trends
                
        return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        # Higher confidence for certain themes
        theme_confidence_map = {0: 0.7, 1: 0.8, 2: 0.5, 3: 0.9}
        base_conf = theme_confidence_map.get(self._last_theme, 0.5)
        return base_conf * self._last_strength

    def get_state(self):
        return {
            "_last_strength": self._last_strength,
            "_last_theme": self._last_theme
        }

    def set_state(self, state):
        self._last_strength = state.get("_last_strength", 0.0)
        self._last_theme = state.get("_last_theme", 0)

# ========== 2. TimeAwareRiskScaling → SeasonalityRiskExpert ==========
class SeasonalityRiskExpert:
    """
    Scales risk: outputs position size adjustments based on seasonality.
    FIXED: Now provides actual position signals, not just scaling factors.
    """
    def __init__(self, tars_module, env_ref, base_signal: float = 0.3, debug=True):
        self.tars = tars_module
        self.env  = env_ref
        self.base_signal = base_signal
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug

    def reset(self):
        pass

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        f = getattr(self.tars, "seasonality_factor", 1.0)
        try:
            f = float(f)
            if not math.isfinite(f):
                f = 1.0
        except (ValueError, TypeError):
            f = 1.0
            
        vec = self._zero.copy()
        
        # Generate actual trading signals scaled by seasonality
        # During good sessions (f > 1), increase positions
        # During bad sessions (f < 1), reduce positions
        for i in range(0, self.env.action_dim, 2):
            # Use different base directions for diversification
            if i == 0:  # EUR/USD - trade with session flow
                direction = 1 if f > 1.0 else -1
            else:  # XAU/USD - counter-trade weak sessions
                direction = -1 if f < 1.0 else 1
                
            vec[i] = direction * self.base_signal * f
            vec[i+1] = 0.5  # Standard duration
            
        return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        f = float(getattr(self.tars, "seasonality_factor", 1.0))
        # Higher confidence when seasonality factor is extreme
        deviation = abs(f - 1.0)
        return float(np.clip(0.5 + deviation, 0.3, 0.9))

# ========== 3. MetaRLController → MetaRLExpert ==========
class MetaRLExpert:
    """
    Uses the Meta-RL policy's own action; confidence ~ (1 − entropy).
    FIXED: Proper implementation with error handling.
    """
    def __init__(self, meta_rl: "MetaRLController", env_ref, debug=True):
        self.mrl  = meta_rl
        self.env  = env_ref
        self.last_action = np.zeros(self.env.action_dim, np.float32)
        self.last_entropy = 0.5
        self.debug = debug
        self.logger = logging.getLogger("MetaRLExpert")

    def reset(self):
        self.last_action[:] = 0.0
        self.last_entropy = 0.5

    def _call_policy(self, obs_vec: np.ndarray) -> np.ndarray:
        """Safely call the Meta-RL policy"""
        try:
            # Ensure correct shape
            if obs_vec.size != self.mrl.obs_dim:
                if self.debug:
                    self.logger.debug(f"Obs size mismatch: {obs_vec.size} vs {self.mrl.obs_dim}")
                # Pad or truncate as needed
                if obs_vec.size < self.mrl.obs_dim:
                    obs_vec = np.pad(obs_vec, (0, self.mrl.obs_dim - obs_vec.size))
                else:
                    obs_vec = obs_vec[:self.mrl.obs_dim]
                    
            # Convert to tensor
            obs_t = torch.tensor(obs_vec, dtype=torch.float32, device=self.mrl.device).unsqueeze(0)
            
            # Get action from policy
            with torch.no_grad():
                if hasattr(self.mrl.agent, 'act'):
                    raw = self.mrl.agent.act(obs_t)
                else:
                    # Fallback to direct call
                    raw = self.mrl.agent(obs_t)
                    
            # Extract action array
            if isinstance(raw, dict):
                act = raw.get("action", np.zeros(self.env.action_dim))
                # Store entropy if available
                if "entropy" in raw:
                    self.last_entropy = float(raw["entropy"])
            else:
                act = raw
                
            # Convert to numpy
            if torch.is_tensor(act):
                act = act.cpu().numpy()
            act = np.asarray(act, dtype=np.float32).reshape(-1)
            
            # Ensure correct size
            if act.size < self.env.action_dim:
                act = np.pad(act, (0, self.env.action_dim - act.size))
            elif act.size > self.env.action_dim:
                act = act[:self.env.action_dim]
                
            return np.clip(act, -1.0, 1.0)
            
        except Exception as e:
            if self.debug:
                self.logger.error(f"Policy call failed: {e}")
            return self._zero.copy()

    def propose_action(self, obs: np.ndarray, extras: Dict = None) -> np.ndarray:
        self.last_action = self._call_policy(obs)
        return self.last_action.copy()

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        # Use stored entropy or estimate from action diversity
        if hasattr(self.mrl.agent, "last_entropy"):
            ent = self.mrl.agent.last_entropy
        else:
            ent = self.last_entropy
            
        if not math.isfinite(ent):
            return 0.6
            
        # Normalize entropy to confidence (lower entropy = higher confidence)
        ent = float(np.clip(ent, 0.0, 2.0))
        return 0.9 - ent / 2.5

# ========== 4. ActiveTradeMonitor → TradeMonitorVetoExpert ==========
class TradeMonitorVetoExpert:
    """
    FIXED: Now provides risk-adjusted signals instead of always returning zeros.
    Acts as a risk modifier rather than pure veto.
    """
    def __init__(self, monitor: "ActiveTradeMonitor", env_ref, max_signal: float = 0.5, debug=True):
        self.mon  = monitor
        self.env  = env_ref
        self.max_signal = max_signal
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug

    def reset(self):
        pass

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        alerted = getattr(self.mon, "alerted", False)
        
        if alerted:
            # When alerted, suggest closing positions
            vec = self._zero.copy()
            # Get current positions from env if available
            if hasattr(self.env, 'position_manager'):
                positions = self.env.position_manager.open_positions
                for i, inst in enumerate(self.env.instruments):
                    if inst in positions:
                        # Suggest closing position
                        current_side = positions[inst].get('side', 0)
                        vec[2*i] = -current_side * self.max_signal
                        vec[2*i+1] = 0.3  # Quick exit
            return vec
        else:
            # When not alerted, provide modest trend-following signals
            vec = self._zero.copy()
            for i in range(0, self.env.action_dim, 2):
                # Small position sizes when monitor is calm
                vec[i] = self.max_signal * 0.5
                vec[i+1] = 0.5
            return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        alerted = getattr(self.mon, "alerted", False)
        # Higher confidence when alerted (for closing positions)
        # Lower confidence when calm (for new positions)
        return 0.8 if alerted else 0.4

# ========== 5. FractalRegimeConfirmation → RegimeBiasExpert ==========
class RegimeBiasExpert:
    """
    Long-bias in trending regime, short-bias in volatile, careful in noise.
    FIXED: Syntax errors and now trades all instruments properly.
    """
    def __init__(self, frc_module, env_ref, max_size=0.6, debug=True):
        self.frc  = frc_module
        self.env  = env_ref
        self.max_size = max_size
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug

    def reset(self):
        pass

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        label = getattr(self.frc, "label", "noise")
        regime_strength = getattr(self.frc, "regime_strength", 0.0)
        trend_direction = getattr(self.frc, "_trend_direction", 0.0)
        
        try:
            strength = float(np.clip(regime_strength, 0.0, 1.0))
        except (ValueError, TypeError):
            strength = 0.0
            
        vec = self._zero.copy()
        
        if label == "trending":
            # Follow the trend on all instruments
            for i in range(0, self.env.action_dim, 2):
                # Use trend direction if available, otherwise default to long
                direction = np.sign(trend_direction) if trend_direction != 0 else 1
                vec[i] = direction * strength * self.max_size
                vec[i+1] = 0.7  # Longer duration for trends
                
        elif label == "volatile":
            # Counter-trend scalping in volatile markets
            for i in range(0, self.env.action_dim, 2):
                # Opposite of recent trend direction
                direction = -np.sign(trend_direction) if trend_direction != 0 else -1
                vec[i] = direction * strength * self.max_size * 0.5  # Smaller size
                vec[i+1] = 0.3  # Short duration
                
        elif label == "noise":
            # Mean reversion in noise regime
            for i in range(0, self.env.action_dim, 2):
                # Fade extreme moves
                direction = -np.sign(trend_direction) if abs(trend_direction) > 0.5 else 0
                vec[i] = direction * strength * self.max_size * 0.3  # Very small size
                vec[i+1] = 0.4  # Medium duration
                
        return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        label = getattr(self.frc, "label", "noise")
        strength = float(abs(getattr(self.frc, "regime_strength", 0.0)))
        
        # Confidence based on regime type and strength
        confidence_map = {
            "trending": 0.8,
            "volatile": 0.5,
            "noise": 0.3
        }
        base_conf = confidence_map.get(label, 0.4)
        return np.clip(base_conf * strength, 0.2, 0.9)

    def get_state(self):
        return {}

    def set_state(self, state):
        pass