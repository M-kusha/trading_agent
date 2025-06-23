# modules/strategy/voting_wrappers.py
"""
FIXED: Expert wrappers that provide actual trading signals instead of just zeros or scaling factors.
Each expert now contributes meaningful trading decisions to the committee.
"""

import numpy as np
from typing import Any, List, Dict
import math
import torch
import logging
import inspect


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
    FIXED: Now trades on all instruments based on detected market themes.
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
    Wraps your MetaRLController: calls controller.act(),
    inspects for extra kwargs (e.g. market_lags), and returns
    a fixed-size action vector. Confidence is 1 - last_entropy.
    """
    def __init__(self, meta_rl: "MetaRLController", env_ref, debug: bool = True):
        self.mrl          = meta_rl
        self.env          = env_ref
        self.last_action  = np.zeros(self.env.action_dim, dtype=np.float32)
        self.last_entropy = 0.5
        self.debug        = debug

        self.logger = logging.getLogger("MetaRLExpert")
        if self.debug and not any(isinstance(h, logging.StreamHandler)
                                  for h in self.logger.handlers):
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(h)
            self.logger.setLevel(logging.DEBUG)

    def reset(self):
        """Clear history at episode start."""
        self.last_action.fill(0.0)
        self.last_entropy = 0.5

    def _call_controller(self, obs_vec: np.ndarray) -> np.ndarray:
        try:
            # --- ensure obs length matches controller.obs_size ---
            od = getattr(self.mrl, "obs_size", obs_vec.size)
            obs_dim = od() if callable(od) else od
            flat = np.asarray(obs_vec, dtype=np.float32).ravel()
            if flat.size < obs_dim:
                flat = np.pad(flat, (0, obs_dim - flat.size))
            elif flat.size > obs_dim:
                flat = flat[:obs_dim]

            # --- build torch tensor ---
            device = getattr(self.mrl, "device", "cpu")
            obs_t = torch.tensor(flat, dtype=torch.float32, device=device).unsqueeze(0)

            # --- inspect act() signature for market_lags ---
            sig = inspect.signature(self.mrl.act)
            kwargs = {}
            if "market_lags" in sig.parameters:
                lags = getattr(self.env, "market_lags", None)
                if not isinstance(lags, np.ndarray):
                    lags = np.zeros_like(flat, dtype=np.float32)
                ml_t = torch.tensor(lags, dtype=torch.float32, device=device).unsqueeze(0)
                kwargs["market_lags"] = ml_t

            # --- call your controller ---
            with torch.no_grad():
                action = self.mrl.act(obs_t, **kwargs)

            # --- convert to numpy 1D ---
            if torch.is_tensor(action):
                act = action.squeeze(0).cpu().numpy()
            else:
                act = np.asarray(action, dtype=np.float32).ravel()

            # --- capture entropy if your controller exposed it ---
            ent = getattr(self.mrl, "last_entropy", None)
            if isinstance(ent, float):
                self.last_entropy = ent

            # --- fix final length to env.action_dim ---
            if act.size < self.env.action_dim:
                act = np.pad(act, (0, self.env.action_dim - act.size))
            elif act.size > self.env.action_dim:
                act = act[: self.env.action_dim]

            self.last_action = act.copy()
            return act

        except Exception as e:
            if self.debug:
                self.logger.warning(f"MetaRLExpert call failed: {e}")
            return np.zeros(self.env.action_dim, dtype=np.float32)

    def propose_action(self, obs: np.ndarray, extras: dict = None) -> np.ndarray:
        """
        Entry point from the arbiter: if we have an ndarray obs,
        actually call the controller; otherwise return last_action.
        """
        if isinstance(obs, np.ndarray):
            return self._call_controller(obs)
        return self.last_action.copy()

    def confidence(self, obs: np.ndarray = None, extras: dict = None) -> float:
        """Simple confidence = 1 − entropy, clipped to [0,1]."""
        return float(np.clip(1.0 - self.last_entropy, 0.0, 1.0))

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
        risk_score = getattr(self.mon, "risk_score", 0.0)
        
        vec = self._zero.copy()
        
        if alerted:
            # When alerted, suggest closing positions
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
            # When not alerted, provide risk-adjusted signals
            # Lower risk score = more aggressive trading
            risk_multiplier = 1.0 - risk_score
            
            for i in range(0, self.env.action_dim, 2):
                # Momentum-based signals
                if hasattr(self.env, 'last_actions') and i < len(self.env._last_actions):
                    # Follow recent momentum but scale by risk
                    momentum = self.env._last_actions[i]
                    vec[i] = momentum * risk_multiplier * 0.7
                else:
                    # Default small position
                    vec[i] = self.max_signal * risk_multiplier * 0.3
                    
                vec[i+1] = 0.5 + risk_score * 0.3  # Longer duration when risky
                
            return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        alerted = getattr(self.mon, "alerted", False)
        risk_score = getattr(self.mon, "risk_score", 0.0)
        
        if alerted:
            # High confidence when closing risky positions
            return 0.8 + risk_score * 0.2
        else:
            # Lower confidence in normal trading, inversely proportional to risk
            return 0.4 * (1.0 - risk_score * 0.5)

# ========== 5. FractalRegimeConfirmation → RegimeBiasExpert ==========
class RegimeBiasExpert:
    """
    Long-bias in trending regime, short-bias in volatile, careful in noise.
    FIXED: Syntax errors and now trades all instruments properly.
    """
    def __init__(self, frc_module, env_ref, max_size: float = 0.6, debug=True):
        self.frc = frc_module
        self.env = env_ref
        self.max_size = max_size
        self._zero = np.zeros(self.env.action_dim, np.float32)
        self.debug = debug
        self.logger = logging.getLogger("RegimeBiasExpert")

    def reset(self):
        pass

    def propose_action(self, obs: Any, extras: Dict = None) -> np.ndarray:
        # Get regime info
        regime_label = getattr(self.frc, "label", "noise")
        regime_strength = float(getattr(self.frc, "regime_strength", 0.5))
        trend_direction = float(getattr(self.frc, "_trend_direction", 0.0))
        
        vec = self._zero.copy()
        
        if regime_label == "trending":
            # Strong directional bias based on trend
            for i in range(0, self.env.action_dim, 2):
                # All instruments follow the trend
                direction = np.sign(trend_direction) if trend_direction != 0 else 1
                vec[i] = direction * regime_strength * self.max_size
                vec[i+1] = 0.7  # Longer holds in trends
                
        elif regime_label == "volatile":
            # Mean reversion strategy in volatile markets
            for i in range(0, self.env.action_dim, 2):
                # Alternate directions for hedging
                if i == 0:  # EUR/USD - fade moves
                    vec[i] = -np.sign(trend_direction) * regime_strength * self.max_size * 0.5
                else:  # XAU/USD - momentum
                    vec[i] = np.sign(trend_direction) * regime_strength * self.max_size * 0.5
                vec[i+1] = 0.3  # Shorter holds in volatile regime
                
        else:  # noise regime
            # Very conservative, small exploratory positions
            for i in range(0, self.env.action_dim, 2):
                # Random small positions for price discovery
                vec[i] = np.random.choice([-1, 1]) * regime_strength * self.max_size * 0.2
                vec[i+1] = 0.5  # Medium duration
                
        return vec

    def confidence(self, obs: Any, extras: Dict = None) -> float:
        regime_label = getattr(self.frc, "label", "noise")
        regime_strength = float(getattr(self.frc, "regime_strength", 0.5))
        
        # Confidence based on regime clarity
        confidence_map = {
            "trending": 0.8,
            "volatile": 0.6,
            "noise": 0.3
        }
        
        base_conf = confidence_map.get(regime_label, 0.5)
        return base_conf * regime_strength

# ========== Helper function to create all experts ==========
def create_all_experts(env, modules_dict: Dict[str, Any]) -> List[Any]:
    """
    Create all expert wrappers for the voting committee.
    
    Args:
        env: The trading environment
        modules_dict: Dictionary of initialized modules
        
    Returns:
        List of expert wrapper instances
    """
    experts = []
    
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
        
    return experts