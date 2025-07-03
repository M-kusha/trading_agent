# envs/env_observation.py
"""
Enhanced observation building with InfoBus integration
Maintains backward compatibility while adding InfoBus infrastructure
"""
from typing import Dict, Any, List, Optional
import numpy as np
from gymnasium import spaces

from modules.utils.info_bus import InfoBus, create_info_bus, validate_info_bus
from modules.utils.audit_utils import format_operator_message
from .shared_utils import profile_method


def _sanitize_observation(self, obs: np.ndarray) -> np.ndarray:
    """Enhanced observation sanitization with InfoBus logging"""
    original_size = obs.size
    
    # Replace NaN/Inf with zeros
    obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Clip extreme values
    obs = np.clip(obs, -1e6, 1e6)
    
    # Count issues for logging
    nan_count = np.sum(np.isnan(obs)) if not np.all(np.isfinite(obs)) else 0
    
    # Validate and log if issues found
    if nan_count > 0 or original_size == 0:
        self.logger.warning(
            format_operator_message(
                "🧹", "OBSERVATION_SANITIZED",
                details=f"Fixed {nan_count} invalid values, size={original_size}",
                context="data_quality"
            )
        )
    
    # Final validation
    assert np.all(np.isfinite(obs)), "Non-finite values remain in observation"
    
    return obs


@profile_method
def _get_full_observation(self, info_bus: InfoBus) -> np.ndarray:
    """Enhanced observation building with InfoBus pipeline integration"""
    
    # Create cache key based on step and InfoBus content signature
    step = info_bus.get('step_idx', self.market_state.current_step)
    cache_key = (step, hash(str(sorted(info_bus.keys()))))
    
    if cache_key in self._obs_cache:
        return self._obs_cache[cache_key]

    # Validate InfoBus quality - FIXED VERSION
    if self.config.info_bus_validation:
        from modules.utils.info_bus import safe_quality_check
        quality_check = safe_quality_check(info_bus)
        
        if not quality_check['is_valid']:
            self.logger.warning(
                format_operator_message(
                    "⚠️", "INFOBUS_QUALITY_ISSUES",
                    details=f"Issues: {quality_check['issue_count']}, Score: {quality_check['score']:.1f}%",
                    context="data_validation"
                )
            )

    try:
        # Process through enhanced pipeline
        obs = self.pipeline_processor.step(info_bus)
        
        # Sanity check right after pipeline
        if not np.all(np.isfinite(obs)):
            self.logger.error(
                format_operator_message(
                    "🛑", "INVALID_OBSERVATION",
                    details=f"NaN/Inf detected at step {step}",
                    context="data_integrity"
                )
            )
            # Log InfoBus keys for debugging
            self.logger.error(f"InfoBus keys: {list(info_bus.keys())}")
            
            # Emergency handling - replace with zeros
            obs = np.zeros_like(obs)
        
        # Ensure matches declared observation_space size
        if hasattr(self, "observation_space"):
            target = self.observation_space.shape[0]
            if obs.size < target:
                obs = np.concatenate([obs, np.zeros(target - obs.size, dtype=np.float32)])
            elif obs.size > target:
                obs = obs[:target]

        # Sanitize and cache
        obs = self._sanitize_observation(obs)
        self._obs_cache[cache_key] = obs

        # Trim cache if it grows too large
        if len(self._obs_cache) > 100:
            for key in list(self._obs_cache)[:-100]:
                del self._obs_cache[key]
                
        # Log observation quality for debugging
        if self.config.debug and step % 50 == 0:
            self.logger.info(
                format_operator_message(
                    "📊", "OBSERVATION_STATS",
                    details=f"Size={obs.size}, Range=[{obs.min():.3f}, {obs.max():.3f}]",
                    context="observation_monitoring"
                )
            )

        return obs
        
    except Exception as e:
        self.logger.error(
            format_operator_message(
                "💥", "OBSERVATION_ERROR",
                details=f"Pipeline failed: {e}",
                result="Using fallback observation",
                context="error_recovery"
            )
        )
        
        # Return safe fallback observation
        fallback_size = getattr(self, 'observation_space', spaces.Box(low=0, high=1, shape=(366,))).shape[0]
        return np.zeros(fallback_size, dtype=np.float32)


def _get_next_observation(
    self,
    trades: List[Dict],
    actions: np.ndarray
) -> np.ndarray:
    """Enhanced observation generation with InfoBus integration"""
    
    try:
        # Create comprehensive InfoBus for next observation
        info_bus = create_info_bus(self, step=self.market_state.current_step)
        
        # Add recent trading activity
        info_bus['recent_trades'] = trades
        info_bus['raw_actions'] = actions
        
        # Add calculated PnL
        total_pnl = sum(t.get("pnl", 0.0) for t in trades)
        info_bus['step_pnl'] = total_pnl
        
        # Add memory embedding if available
        if hasattr(self.feature_engine, 'last_embedding') and self.feature_engine.last_embedding is not None:
            info_bus['memory_embedding'] = self.feature_engine.last_embedding
        
        # Add correlations
        info_bus['correlations'] = self.get_instrument_correlations()
        
        # Get enhanced observation through InfoBus pipeline
        obs = self._get_full_observation(info_bus)
        
        # Update meta-RL with new observation
        if hasattr(self, 'meta_rl') and self.meta_rl and obs.size == self.meta_rl.obs_dim:
            reward = getattr(self, '_last_reward', 0.0)
            self.meta_rl.record_step(obs, reward)
        
        # Store for next step
        self._last_actions = actions.copy()
        self._last_reward = getattr(self, '_last_reward', 0.0)
        
        # Log significant events
        if len(trades) > 0:
            self.logger.info(
                format_operator_message(
                    "📈", "OBSERVATION_WITH_TRADES",
                    details=f"{len(trades)} trades, PnL={total_pnl:+.2f}",
                    context="trading_activity"
                )
            )
        
        return obs
        
    except Exception as e:
        self.logger.error(
            format_operator_message(
                "💥", "NEXT_OBSERVATION_ERROR",
                details=f"Failed to create next observation: {e}",
                result="Using fallback",
                context="error_recovery"
            )
        )
        
        # Fallback to simple observation
        return self._get_fallback_observation()


def _get_fallback_observation(self) -> np.ndarray:
    """Create fallback observation when InfoBus pipeline fails"""
    
    try:
        # Create minimal observation components
        obs_parts = []
        
        # Market data
        for inst in self.instruments[:1]:  # Just primary instrument
            hist = self._get_price_history(inst, "D1")
            obs_parts.append(hist)
        
        # Position data
        pos_count = len(getattr(self.position_manager, 'open_positions', {}))
        obs_parts.append(np.array([pos_count, self.market_state.balance / 10000], dtype=np.float32))
        
        # Risk data
        obs_parts.append(np.array([
            self.market_state.current_drawdown,
            self._get_current_volatility()
        ], dtype=np.float32))
        
        # Combine and ensure proper size
        obs = np.concatenate(obs_parts)
        
        # Pad to expected size if needed
        if hasattr(self, 'observation_space'):
            target_size = self.observation_space.shape[0]
            if obs.size < target_size:
                obs = np.concatenate([obs, np.zeros(target_size - obs.size, dtype=np.float32)])
            elif obs.size > target_size:
                obs = obs[:target_size]
        
        return self._sanitize_observation(obs)
        
    except Exception as e:
        self.logger.error(f"Fallback observation failed: {e}")
        # Ultimate fallback - zeros
        fallback_size = getattr(self, 'observation_space', spaces.Box(low=0, high=1, shape=(100,))).shape[0]
        return np.zeros(fallback_size, dtype=np.float32)


def _create_info_bus_for_step(self, trades: List[Dict], actions: np.ndarray) -> InfoBus:
    """Create comprehensive InfoBus for current step"""
    
    # Start with base InfoBus
    info_bus = create_info_bus(self, step=self.market_state.current_step)
    
    # Add step-specific data
    info_bus.update({
        'recent_trades': trades,
        'raw_actions': actions,
        'step_pnl': sum(t.get("pnl", 0.0) for t in trades),
        'correlations': self.get_instrument_correlations(),
    })
    
    # Add performance metrics
    if hasattr(self, 'episode_metrics'):
        info_bus['episode_metrics'] = {
            'total_trades': len(self.episode_metrics.trades),
            'total_pnl': sum(self.episode_metrics.pnls),
            'current_drawdown': self.market_state.current_drawdown,
        }
    
    # Add module performance if available
    if hasattr(self.pipeline, 'module_performance'):
        info_bus['module_performance'] = dict(self.pipeline.module_performance)
    
    return info_bus


# Backward compatibility methods for legacy interface
def get_observation_legacy(self, 
                          price_data: Dict[str, np.ndarray], 
                          actions: np.ndarray = None,
                          trades: List[Dict] = None) -> np.ndarray:
    """Legacy observation method - converts to InfoBus format"""
    
    # Create InfoBus from legacy data
    info_bus = create_info_bus(self, step=self.market_state.current_step)
    
    # Add legacy price data
    for inst, prices in price_data.items():
        if inst in self.instruments:
            info_bus[f'price_history_{inst}'] = prices
    
    # Add other legacy data
    if actions is not None:
        info_bus['raw_actions'] = actions
    if trades is not None:
        info_bus['recent_trades'] = trades
    
    # Use enhanced observation method
    return self._get_full_observation(info_bus)