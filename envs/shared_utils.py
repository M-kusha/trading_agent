# envs/shared_utils.py
"""
Shared utilities and classes used across the trading environment
"""
import time
import logging
from functools import wraps
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from modules.core.core import Module


class DummyExplanationGenerator:
    """Dummy explanation generator for fallback"""
    
    def __init__(self):
        pass
        
    def reset(self):
        pass
        
    def step(self, **kwargs):
        pass
        
    def get_observation_components(self):
        return np.zeros(0, dtype=np.float32)


def profile_method(func):
    """Performance profiling decorator"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        
        if elapsed > 0.1 and hasattr(self, 'logger'):  # Log slow operations
            self.logger.warning(f"{func.__name__} took {elapsed:.3f}s")
            
        return result
    return wrapper


class TradingPipeline:
    """Manages the sequential processing of trading modules"""

    def __init__(self, modules: List[Module]):
        self.modules = modules
        self._module_map = {m.__class__.__name__: m for m in modules}
        self.expected_size: Optional[int] = None  # length the model already saw
        
        # FIXED: Add logger
        import logging
        self.logger = logging.getLogger(f"TradingPipeline_{id(self)}")

    def reset(self):
        for module in self.modules:
            try:
                module.reset()
            except Exception as e:
                logging.warning(f"Failed to reset {module.__class__.__name__}: {e}")

    def step(self, data: Dict[str, Any]) -> np.ndarray:
        """FIXED: Pipeline step with proper MemoryCompressor handling"""
        env = data.get("env")
        obs_parts: List[np.ndarray] = []

        for module in self.modules:
            if env and not env.module_enabled.get(module.__class__.__name__, True):
                continue

            try:
                module_name = module.__class__.__name__
                
                # FIXED: Only these memory modules are handled separately
                if module_name in ['MistakeMemory', 'HistoricalReplayAnalyzer', 
                                'PlaybookMemory', 'MemoryBudgetOptimizer']:
                    # These are fed by environment directly
                    part = module.get_observation_components()
                    
                else:
                    # FIXED: MemoryCompressor and other modules get normal step() calls
                    sig = module.step.__code__.co_varnames[1:module.step.__code__.co_argcount]
                    kwargs = {k: data[k] for k in sig if k in data}
                    
                    # Call module step
                    module.step(**kwargs)
                    part = module.get_observation_components()

            except Exception as e:
                if hasattr(env, 'logger'):
                    env.logger.error(f"Error in {module.__class__.__name__}.step(): {e}")
                part = np.zeros(0, dtype=np.float32)

            # Ensure 1D float32 and handle NaN/Inf
            part = np.asarray(part, dtype=np.float32).ravel()
            if not np.all(np.isfinite(part)):
                if hasattr(env, 'logger'):
                    env.logger.warning(f"Set {module.__class__.__name__} output to zeros due to NaN/Inf")
                part = np.zeros_like(part)

            obs_parts.append(part)

        # Concatenate and manage size
        obs = np.concatenate(obs_parts) if obs_parts else np.zeros(0, np.float32)

        if self.expected_size is None:
            self.expected_size = obs.size
            if hasattr(env, 'logger'):
                env.logger.info(f"Pipeline observation size locked at {self.expected_size}")
        else:
            if obs.size != self.expected_size:
                if obs.size < self.expected_size:
                    pad = np.zeros(self.expected_size - obs.size, dtype=np.float32)
                    obs = np.concatenate([obs, pad])
                else:
                    obs = obs[: self.expected_size]

        return obs

    def get_module(self, name: str) -> Optional[Module]:
        return self._module_map.get(name)


class UnifiedRiskManager:
    """FIXED: Centralized risk management system"""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        self.dd_limit = config.get('dd_limit', 0.3)
        self.correlation_limit = config.get('correlation_limit', 0.8)
        self.var_limit = config.get('var_limit', 0.1)
        self.max_positions = config.get('max_positions', 10)
        self.logger = logger or logging.getLogger("UnifiedRiskManager")
        
    def pre_trade_check(self, context: Dict[str, Any]) -> Tuple[bool, str]:
        """Centralized pre-trade risk checks"""
        # Check drawdown
        if context['drawdown'] > self.dd_limit:
            return False, f"Drawdown {context['drawdown']:.1%} exceeds limit {self.dd_limit:.1%}"
            
        # Check correlations
        max_corr = max(context.get('correlations', {}).values()) if context.get('correlations') else 0
        if max_corr > self.correlation_limit:
            return False, f"Correlation {max_corr:.2f} exceeds limit {self.correlation_limit:.2f}"
            
        # Check position count
        if len(context.get('open_positions', {})) >= self.max_positions:
            return False, f"Already have {len(context['open_positions'])} positions (max: {self.max_positions})"
            
        return True, "All risk checks passed"
        
    def post_trade_update(self, trade: Dict[str, Any]):
        """Update risk systems after trade"""
        # This would update internal risk tracking
        pass