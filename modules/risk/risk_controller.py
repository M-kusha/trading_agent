import numpy as np
import logging
import json
import os
import datetime
from collections import deque
from typing import Dict, Any, List, Optional, Tuple
from modules.core.core import Module
import copy


class DynamicRiskController(Module):
    """
    FIXED: Graduated risk controller with smooth scaling and better integration.
    
    Key improvements:
    - Graduated risk scaling instead of binary freeze
    - Multiple risk factors with weighted combination
    - Integration hooks for other risk systems
    - Adaptive thresholds based on market conditions
    - Better bootstrapping behavior
    """

    DEFAULTS = {
        "base_risk_scale": 1.0,
        "min_risk_scale": 0.1,      # Never go below 10% risk
        "vol_history_len": 50,      # Reduced for faster adaptation
        "dd_threshold": 0.15,       # More reasonable drawdown threshold
        "vol_ratio_threshold": 2.0, # Less strict volatility threshold
        "recovery_speed": 0.1,      # How fast to recover from risk reduction
        "risk_decay": 0.95,         # Exponential decay for risk events
    }

    def __init__(
        self,
        config: Optional[Dict[str, float]] = None,
        action_dim: int = 1,
        debug: bool = True,
        audit_log_size: int = 100,
        audit_log_path: str = "logs/risk/dynamic_risk_audit.jsonl",
        log_path: str = "logs/risk/dynamic_risk.log",
    ):
        super().__init__()
        
        # Merge config with defaults
        self.config = copy.deepcopy(self.DEFAULTS)
        if config:
            self.config.update(config)
            
        # Core parameters
        self.base_risk_scale = float(self.config["base_risk_scale"])
        self.min_risk_scale = float(self.config["min_risk_scale"])
        self.vol_history_len = int(self.config["vol_history_len"])
        self.dd_threshold = float(self.config["dd_threshold"])
        self.vol_ratio_threshold = float(self.config["vol_ratio_threshold"])
        self.recovery_speed = float(self.config["recovery_speed"])
        self.risk_decay = float(self.config["risk_decay"])
        
        self.action_dim = int(action_dim)
        self.debug = debug
        
        # State tracking
        self.current_risk_scale = self.base_risk_scale
        self.risk_factors: Dict[str, float] = {
            "drawdown": 1.0,
            "volatility": 1.0,
            "correlation": 1.0,
            "losing_streak": 1.0,
            "market_stress": 1.0,
        }
        self.vol_history = deque(maxlen=self.vol_history_len)
        self.dd_history = deque(maxlen=20)
        self.consecutive_losses = 0
        self.last_pnl = 0.0
        
        # Risk event tracking
        self.risk_events: List[Dict[str, Any]] = []
        self.max_risk_events = 50
        
        # Integration hooks
        self.external_risk_scale = 1.0  # From other modules
        self.market_regime = "normal"    # From regime detector
        
        # Audit trail
        self._audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size = audit_log_size
        self.audit_log_path = audit_log_path
        self.log_path = log_path
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info(
            f"Initialized DynamicRiskController | "
            f"dd_threshold={self.dd_threshold:.2f} "
            f"vol_ratio_threshold={self.vol_ratio_threshold:.1f}"
        )

    def _setup_logging(self):
        """Setup file logging"""
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        self.logger = logging.getLogger(f"DynamicRiskController_{id(self)}")
        self.logger.handlers.clear()
        
        fh = logging.FileHandler(self.log_path)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self.logger.propagate = False

    def set_market_regime(self, regime: str):
        """Set market regime from external regime detector"""
        self.market_regime = regime
        
        # Adjust thresholds based on regime
        if regime == "volatile":
            self.risk_factors["market_stress"] = 0.7
        elif regime == "trending":
            self.risk_factors["market_stress"] = 1.1  # Slightly increase risk in trends
        else:
            self.risk_factors["market_stress"] = 1.0

    def set_external_risk_scale(self, scale: float):
        """Set risk scale from other risk modules (e.g., PortfolioRiskSystem)"""
        self.external_risk_scale = np.clip(float(scale), 0.1, 2.0)

    def update_correlation_risk(self, max_correlation: float):
        """Update correlation risk factor"""
        # High correlation reduces risk appetite
        if max_correlation > 0.8:
            self.risk_factors["correlation"] = 0.6
        elif max_correlation > 0.6:
            self.risk_factors["correlation"] = 0.8
        else:
            self.risk_factors["correlation"] = 1.0

    def calculate_risk_scale(self) -> float:
        """
        Calculate current risk scale based on all factors.
        Returns a value between min_risk_scale and base_risk_scale.
        """
        # Start with base scale
        scale = self.base_risk_scale
        
        # Apply all risk factors
        for factor, value in self.risk_factors.items():
            scale *= value
            
        # Apply external risk scale
        scale *= self.external_risk_scale
        
        # Apply bounds
        scale = np.clip(scale, self.min_risk_scale, self.base_risk_scale)
        
        return float(scale)

    def _calculate_drawdown_factor(self, dd: float) -> float:
        """Calculate risk factor based on drawdown"""
        if dd <= 0.05:  # Small drawdown is ok
            return 1.0
        elif dd <= self.dd_threshold:
            # Linear scaling
            return 1.0 - (dd - 0.05) / (self.dd_threshold - 0.05) * 0.5
        else:
            # More aggressive reduction
            excess = dd - self.dd_threshold
            return 0.5 * np.exp(-excess * 5)  # Exponential decay

    def _calculate_volatility_factor(self, vol: float, vol_ratio: float) -> float:
        """Calculate risk factor based on volatility"""
        if vol_ratio <= 1.2:  # Normal volatility
            return 1.0
        elif vol_ratio <= self.vol_ratio_threshold:
            # Linear scaling
            return 1.0 - (vol_ratio - 1.2) / (self.vol_ratio_threshold - 1.2) * 0.3
        else:
            # More aggressive reduction
            excess = vol_ratio - self.vol_ratio_threshold
            return 0.7 * np.exp(-excess * 2)

    def _calculate_streak_factor(self) -> float:
        """Calculate risk factor based on losing streak"""
        if self.consecutive_losses <= 2:
            return 1.0
        elif self.consecutive_losses <= 5:
            return 1.0 - (self.consecutive_losses - 2) * 0.1
        else:
            return 0.5  # Cap at 50% reduction

    def adjust_risk(self, stats: Dict[str, float]) -> None:
        """
        Main method to adjust risk based on current market conditions.
        More nuanced than binary freeze/unfreeze.
        """
        # Extract stats
        dd = float(stats.get("drawdown", 0.0))
        vol = float(stats.get("volatility", 0.01))
        pnl = float(stats.get("pnl", 0.0))
        
        # Update histories
        self.vol_history.append(vol)
        self.dd_history.append(dd)
        
        # Track consecutive losses
        if pnl < 0 and self.last_pnl < 0:
            self.consecutive_losses += 1
        elif pnl > 0:
            self.consecutive_losses = max(0, self.consecutive_losses - 1)
        self.last_pnl = pnl
        
        # Bootstrap volatility history if needed
        if len(self.vol_history) < 5:
            for _ in range(5 - len(self.vol_history)):
                self.vol_history.append(vol * np.random.uniform(0.8, 1.2))
                
        # Calculate volatility ratio
        avg_vol = np.mean(self.vol_history)
        vol_ratio = vol / (avg_vol + 1e-8)
        
        # Update risk factors
        old_scale = self.current_risk_scale
        
        self.risk_factors["drawdown"] = self._calculate_drawdown_factor(dd)
        self.risk_factors["volatility"] = self._calculate_volatility_factor(vol, vol_ratio)
        self.risk_factors["losing_streak"] = self._calculate_streak_factor()
        
        # Calculate new risk scale
        new_scale = self.calculate_risk_scale()
        
        # Smooth transitions to avoid abrupt changes
        if new_scale < old_scale:
            # Quick reduction when risk increases
            self.current_risk_scale = new_scale
        else:
            # Gradual recovery when risk decreases
            self.current_risk_scale = old_scale + (new_scale - old_scale) * self.recovery_speed
            
        # Record risk event if significant change
        if abs(new_scale - old_scale) > 0.1:
            self._record_risk_event(
                reason="risk_adjustment",
                old_scale=old_scale,
                new_scale=new_scale,
                factors=self.risk_factors.copy(),
                stats=stats.copy()
            )
            
        # Create audit record
        self._log_audit(
            drawdown=dd,
            volatility=vol,
            vol_ratio=vol_ratio,
            old_scale=old_scale,
            new_scale=self.current_risk_scale,
            risk_factors=self.risk_factors.copy()
        )
        
        if self.debug:
            self.logger.info(
                f"Risk adjusted: {old_scale:.3f} -> {self.current_risk_scale:.3f} | "
                f"DD={dd:.3f} Vol={vol:.4f} VR={vol_ratio:.2f}"
            )

    def _record_risk_event(self, **kwargs):
        """Record significant risk events"""
        event = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            **kwargs
        }
        self.risk_events.append(event)
        if len(self.risk_events) > self.max_risk_events:
            self.risk_events.pop(0)

    def _log_audit(self, **kwargs):
        """Log audit trail"""
        audit = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            **kwargs
        }
        
        self._audit_trail.append(audit)
        if len(self._audit_trail) > self._audit_log_size:
            self._audit_trail.pop(0)
            
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(audit) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit: {e}")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics for monitoring"""
        return {
            "current_scale": self.current_risk_scale,
            "risk_factors": self.risk_factors.copy(),
            "consecutive_losses": self.consecutive_losses,
            "avg_volatility": float(np.mean(self.vol_history)) if self.vol_history else 0.0,
            "max_drawdown": float(np.max(self.dd_history)) if self.dd_history else 0.0,
            "market_regime": self.market_regime,
            "external_scale": self.external_risk_scale,
        }

    # ================= Module Interface ==================== #

    def reset(self):
        """Reset controller state"""
        self.current_risk_scale = self.base_risk_scale
        self.risk_factors = {
            "drawdown": 1.0,
            "volatility": 1.0,
            "correlation": 1.0,
            "losing_streak": 1.0,
            "market_stress": 1.0,
        }
        self.vol_history.clear()
        self.dd_history.clear()
        self.consecutive_losses = 0
        self.last_pnl = 0.0
        self.risk_events.clear()
        self._audit_trail.clear()
        self.external_risk_scale = 1.0
        self.market_regime = "normal"

    def step(self, **kwargs):
        """Process step data and adjust risk"""
        # Build stats dict from kwargs
        stats = {
            "drawdown": kwargs.get("drawdown", kwargs.get("current_drawdown", 0.0)),
            "volatility": kwargs.get("volatility", 0.01),
            "pnl": kwargs.get("pnl", 0.0),
        }
        
        # Adjust risk
        self.adjust_risk(stats)
        
        # Update from external sources if provided
        if "market_regime" in kwargs:
            self.set_market_regime(kwargs["market_regime"])
            
        if "correlation" in kwargs:
            self.update_correlation_risk(kwargs["correlation"])

    def get_observation_components(self) -> np.ndarray:
        """Return risk state as observation"""
        return np.array([
            self.current_risk_scale,
            self.risk_factors["drawdown"],
            self.risk_factors["volatility"],
            self.risk_factors["losing_streak"],
            self.consecutive_losses / 10.0,  # Normalized
        ], dtype=np.float32)

    def propose_action(self, obs: Any = None) -> np.ndarray:
        """
        Propose risk-scaled action.
        Unlike binary freeze, this scales the action smoothly.
        """
        # Return risk scale for all action dimensions
        return np.full(self.action_dim, self.current_risk_scale, dtype=np.float32)

    def confidence(self, obs: Any = None) -> float:
        """
        Return confidence based on risk scale.
        More nuanced than binary 0.3/1.0.
        """
        # Base confidence on risk scale
        base_conf = 0.5 + 0.5 * self.current_risk_scale
        
        # Reduce confidence if multiple risk factors are triggered
        active_factors = sum(1 for v in self.risk_factors.values() if v < 0.8)
        if active_factors >= 3:
            base_conf *= 0.7
        elif active_factors >= 2:
            base_conf *= 0.85
            
        return float(np.clip(base_conf, 0.3, 1.0))

    def get_state(self) -> Dict[str, Any]:
        """Get state for serialization"""
        return {
            "config": self.config.copy(),
            "current_risk_scale": self.current_risk_scale,
            "risk_factors": self.risk_factors.copy(),
            "vol_history": list(self.vol_history),
            "dd_history": list(self.dd_history),
            "consecutive_losses": self.consecutive_losses,
            "last_pnl": self.last_pnl,
            "risk_events": self.risk_events.copy(),
            "external_risk_scale": self.external_risk_scale,
            "market_regime": self.market_regime,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialization"""
        # Restore config
        if "config" in state:
            self.config.update(state["config"])
            self._update_from_config()
            
        # Restore state
        self.current_risk_scale = state.get("current_risk_scale", self.base_risk_scale)
        self.risk_factors = state.get("risk_factors", self.risk_factors).copy()
        
        # Restore histories
        self.vol_history = deque(
            state.get("vol_history", []),
            maxlen=self.vol_history_len
        )
        self.dd_history = deque(
            state.get("dd_history", []),
            maxlen=20
        )
        
        # Restore other state
        self.consecutive_losses = state.get("consecutive_losses", 0)
        self.last_pnl = state.get("last_pnl", 0.0)
        self.risk_events = state.get("risk_events", [])[:self.max_risk_events]
        self.external_risk_scale = state.get("external_risk_scale", 1.0)
        self.market_regime = state.get("market_regime", "normal")

    def _update_from_config(self):
        """Update parameters from config"""
        self.base_risk_scale = float(self.config.get("base_risk_scale", 1.0))
        self.min_risk_scale = float(self.config.get("min_risk_scale", 0.1))
        self.dd_threshold = float(self.config.get("dd_threshold", 0.15))
        self.vol_ratio_threshold = float(self.config.get("vol_ratio_threshold", 2.0))
        self.recovery_speed = float(self.config.get("recovery_speed", 0.1))
        self.risk_decay = float(self.config.get("risk_decay", 0.95))

    # ================= Evolutionary Methods ==================== #

    def mutate(self, std: float = 0.1):
        """Mutate risk parameters"""
        # Mutate thresholds
        self.dd_threshold = float(np.clip(
            self.dd_threshold + np.random.normal(0, std * 0.05),
            0.05, 0.5
        ))
        
        self.vol_ratio_threshold = float(np.clip(
            self.vol_ratio_threshold + np.random.normal(0, std * 0.3),
            1.2, 4.0
        ))
        
        # Mutate recovery parameters
        self.recovery_speed = float(np.clip(
            self.recovery_speed + np.random.normal(0, std * 0.05),
            0.05, 0.5
        ))
        
        self.min_risk_scale = float(np.clip(
            self.min_risk_scale + np.random.normal(0, std * 0.1),
            0.05, 0.5
        ))
        
        # Update config
        self.config.update({
            "dd_threshold": self.dd_threshold,
            "vol_ratio_threshold": self.vol_ratio_threshold,
            "recovery_speed": self.recovery_speed,
            "min_risk_scale": self.min_risk_scale,
        })
        
        self.logger.info(f"Mutated: dd_th={self.dd_threshold:.3f}, vol_th={self.vol_ratio_threshold:.2f}")

    def crossover(self, other: "DynamicRiskController") -> "DynamicRiskController":
        """Create offspring via crossover"""
        def blend(a, b):
            return a if np.random.rand() > 0.5 else b
            
        child_config = {}
        for key in self.config:
            child_config[key] = blend(self.config[key], other.config.get(key, self.config[key]))
            
        return DynamicRiskController(
            config=child_config,
            action_dim=self.action_dim,
            debug=self.debug or other.debug,
            audit_log_size=self._audit_log_size,
            audit_log_path=self.audit_log_path,
        )

    def get_last_audit(self) -> Dict[str, Any]:
        """Get most recent audit record"""
        return self._audit_trail[-1] if self._audit_trail else {}

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent audit records"""
        return self._audit_trail[-n:]

    def get_risk_events(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get recent risk events"""
        if n is None:
            return self.risk_events.copy()
        return self.risk_events[-n:]