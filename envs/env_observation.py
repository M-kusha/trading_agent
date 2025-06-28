# envs/env_observation.py
"""
Observation building methods for the trading environment
"""
from typing import List, Dict, Any
import numpy as np

from .shared_utils import profile_method


def _sanitize_observation(self, obs: np.ndarray) -> np.ndarray:
    """Ensure observation contains no invalid values"""
    # Replace NaN/Inf with zeros
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip extreme values
    obs = np.clip(obs, -1e6, 1e6)
    
    # Validate
    assert np.all(np.isfinite(obs)), "Non-finite values in observation"
    
    return obs


@profile_method
def _get_full_observation(self, data: Dict[str, Any]) -> np.ndarray:
    """
    Build a unified observation by passing `standardized_data` through
    the pipeline, then guard against NaNs before sanitizing/caching.
    """
    cache_key = (self.market_state.current_step, id(data))
    if cache_key in self._obs_cache:
        return self._obs_cache[cache_key]

    # augment inputs for modules
    standardized_data = data.copy()
    standardized_data.update({
        "env": self,
        "open_positions": list(self.position_manager.open_positions.values()),
        "current_step": self.market_state.current_step,
        "data_dict": self.data,
        "price": standardized_data.get("price", None),
        "balance": self.market_state.balance,
        "drawdown": self.market_state.current_drawdown,
        "instruments": self.instruments,
    })

    obs = self.pipeline.step(standardized_data)

    # ←— New sanity check right after pipeline
    if not np.all(np.isfinite(obs)):
        self.logger.error(f"🛑 NaN/Inf in obs at step {self.market_state.current_step}: {obs}")
        self.logger.error(f"  input keys: {list(standardized_data.keys())}")
        for k, v in standardized_data.items():
            if isinstance(v, np.ndarray) and not np.all(np.isfinite(v)):
                self.logger.error(f"    '{k}' has bad values: {v}")
        import pdb; pdb.set_trace()
    # ————————————————————————————————————————

    # ensure matches declared observation_space size
    if hasattr(self, "observation_space"):
        target = self.observation_space.shape[0]
        if obs.size < target:
            obs = np.concatenate([obs, np.zeros(target - obs.size, dtype=np.float32)])
        elif obs.size > target:
            obs = obs[:target]

    # sanitize infinities, extreme outliers, and cache
    obs = self._sanitize_observation(obs)
    self._obs_cache[cache_key] = obs

    # trim cache if it grows too big
    if len(self._obs_cache) > 100:
        for key in list(self._obs_cache)[:-100]:
            del self._obs_cache[key]

    return obs


def _get_next_observation(
    self,
    trades: List[Dict],
    actions: np.ndarray
) -> np.ndarray:
    """Get observation for next step"""
    # Get price histories
    hist_h1 = self._get_price_history(self.instruments[0], "H1")
    hist_h4 = self._get_price_history(self.instruments[0], "H4")
    hist_d1 = self._get_price_history(self.instruments[0], "D1")
    
    # Calculate PnL
    pnl = sum(t.get("pnl", 0.0) for t in trades)
    
    # Get memory embedding
    memory = (
        self.feature_engine.last_embedding
        if hasattr(self.feature_engine, 'last_embedding')
        else None
    )
    
    # FIXED: Create observation data with standardized naming
    obs_data = {
        "env": self,
        "price_h1": hist_h1,
        "price_h4": hist_h4,
        "price_d1": hist_d1,
        "actions": actions,
        "trades": trades,
        "open_positions": list(self.position_manager.open_positions.values()),  # FIXED
        "drawdown": self.market_state.current_drawdown,
        "memory": memory,
        "pnl": pnl,
        "correlations": self.get_instrument_correlations(),
        "current_step": self.market_state.current_step,  # FIXED: Added
    }
    
    # Get full observation
    obs = self._get_full_observation(obs_data)
    
    # Sanitize
    obs = self._sanitize_observation(obs)
    
    # Update meta-RL
    if obs.size == self.meta_rl.obs_dim:
        self.meta_rl.record_step(obs, self._last_reward if hasattr(self, '_last_reward') else 0.0)
        
    # Store for next step
    self._last_actions = actions.copy()
    self._last_reward = self._last_reward if hasattr(self, '_last_reward') else 0.0
    
    return obs