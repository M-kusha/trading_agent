# envs/env_utils.py
"""
Utility methods for the trading environment
"""
import random
import os
import pickle
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import MetaTrader5 as mt5


def _get_initial_balance(self) -> float:
    """Get initial balance for episode"""
    if self.config.live_mode:
        info = mt5.account_info()
        if info and hasattr(info, "balance"):
            return float(info.balance)
    return self.config.initial_balance


def _select_starting_step(self) -> int:
    """Select random starting step for episode"""
    # Find minimum available bars across instruments
    min_bars = min(
        len(self.data[inst]["D1"])
        for inst in self.instruments
        if "D1" in self.data[inst]
    )
    
    # Ensure we have enough data
    min_start = 100  # Need history for indicators
    max_start = max(min_bars - self.config.max_steps - 1, min_start)
    
    if max_start <= min_start:
        self.logger.warning(
            f"Insufficient data: {min_bars} bars, using step 0"
        )
        return 0
        
    return random.randint(min_start, max_start)


def _reset_all_modules(self):
    """Reset all modules to initial state"""
    # Reset pipeline modules
    self.pipeline.reset()
    
    # Reset other modules
    modules_to_reset = [
        self.reward_shaper, self.meta_rl, self.arbiter,
        self.consensus, self.collusion, self.haligner,
        self.strategy_pool, self.meta_agent, self.meta_planner,
        self.long_term_memory, self.world_model
    ]
    
    for module in modules_to_reset:
        if hasattr(module, 'reset'):
            try:
                module.reset()
            except Exception as e:
                self.logger.warning(f"Failed to reset {module.__class__.__name__}: {e}")


def _prime_risk_system(self):
    """Prime risk system with historical price data"""
    try:
        price_dict = {}
        for inst in self.instruments:
            df = self.data[inst]["D1"]
            start_idx = max(0, self.market_state.current_step - self.risk_system.var_window)
            end_idx = self.market_state.current_step + 1
            prices = df["close"].iloc[start_idx:end_idx].values
            price_dict[inst] = prices
            
        if all(len(p) > 10 for p in price_dict.values()):
            self.risk_system.prime_returns_with_history(price_dict)
        else:
            self.risk_system.prime_returns_with_random()
    except Exception as e:
        self.logger.warning(f"Failed to prime risk system: {e}")
        self.risk_system.prime_returns_with_random()


def _select_strategy_genome(self):
    """Select strategy genome for episode"""
    self.strategy_pool.select_genome("random")
    self.current_genome = self.strategy_pool.active_genome.copy()
    self.logger.info(f"Selected genome: {self.current_genome}")


def _get_current_volatility(self) -> float:
    """Get current volatility for primary instrument"""
    try:
        df = self.data[self.instruments[0]]["D1"]
        if self.market_state.current_step < len(df):
            return float(df.iloc[self.market_state.current_step]["volatility"])
    except Exception:
        pass
    return 0.01  # Safe default


def _get_instrument_volatility(self, instrument: str) -> float:
    """Get volatility for specific instrument"""
    try:
        df = self.data[instrument]["D1"]
        if self.market_state.current_step < len(df):
            vol = df.iloc[self.market_state.current_step].get("volatility", 0.01)
            vol = float(np.nan_to_num(vol, nan=0.01))
            return max(vol, self.position_manager.min_volatility)
    except Exception:
        pass
    return self.position_manager.min_volatility


def _get_price_history(self, instrument: str, timeframe: str) -> np.ndarray:
    """Get recent price history for instrument/timeframe"""
    try:
        df = self.data[instrument][timeframe]
        end_idx = min(self.market_state.current_step + 1, len(df))
        start_idx = max(0, end_idx - 7)
        
        if end_idx > start_idx:
            prices = df["close"].iloc[start_idx:end_idx].values
            # Pad if needed
            if len(prices) < 7:
                prices = np.pad(prices, (7 - len(prices), 0), mode='edge')
            return prices[-7:].astype(np.float32)
    except Exception as e:
        self.logger.warning(f"Failed to get price history: {e}")
        
    return np.zeros(7, dtype=np.float32)


def _get_recent_returns(self) -> Dict[str, np.ndarray]:
    """Get recent returns for risk calculations"""
    returns = {}
    for inst in self.instruments:
        try:
            df = self.data[inst]["D1"]
            end_idx = self.market_state.current_step + 1
            start_idx = max(0, end_idx - 20)
            
            if end_idx > start_idx + 1:
                prices = df["close"].iloc[start_idx:end_idx].values
                ret = np.diff(np.log(prices))
                returns[inst] = ret
            else:
                returns[inst] = np.array([])
        except Exception:
            returns[inst] = np.array([])
            
    return returns


def get_instrument_correlations(self) -> Dict[str, float]:
    """Calculate pairwise correlations between instruments"""
    correlations = {}
    
    try:
        # Get returns for all instruments
        returns_dict = self._get_recent_returns()
        
        # Calculate pairwise correlations
        for i, inst1 in enumerate(self.instruments):
            for j, inst2 in enumerate(self.instruments):
                if i >= j:
                    continue
                    
                ret1 = returns_dict.get(inst1, np.array([]))
                ret2 = returns_dict.get(inst2, np.array([]))
                
                if len(ret1) > 5 and len(ret2) > 5:
                    # Align lengths
                    min_len = min(len(ret1), len(ret2))
                    ret1 = ret1[-min_len:]
                    ret2 = ret2[-min_len:]
                    
                    # Calculate correlation
                    corr = np.corrcoef(ret1, ret2)[0, 1]
                    correlations[f"{inst1}-{inst2}"] = float(np.nan_to_num(corr))
                else:
                    correlations[f"{inst1}-{inst2}"] = 0.0
                    
    except Exception as e:
        self.logger.warning(f"Failed to calculate correlations: {e}")
        
    return correlations


def _calculate_reward(
    self,
    trades: List[Dict],
    actions: np.ndarray,
    consensus: float
) -> float:
    """Calculate step reward"""
    # Get reward from reward shaper
    reward = self.reward_shaper.shape_reward(
        trades=trades,
        balance=self.market_state.balance,
        drawdown=self.market_state.current_drawdown,
        consensus=consensus,
        actions=actions,
    )
    
    # Store for next step
    self._last_reward = reward
    
    return reward


def _check_termination(self) -> bool:
    """Check if episode should terminate"""
    # Max steps reached
    if self.market_state.current_step >= self.config.max_steps - 1:
        return True
        
    # Data exhausted
    for inst in self.instruments:
        if self.market_state.current_step >= len(self.data[inst]["D1"]) - 1:
            return True
            
    # Catastrophic loss
    if self.market_state.balance < self.config.initial_balance * 0.5:
        self.logger.warning("Episode terminated: 50% loss")
        return True
        
    # Extreme drawdown
    if self.market_state.current_drawdown > 0.5:
        self.logger.warning("Episode terminated: 50% drawdown")
        return True
        
    return False


def _finalize_step(
    self,
    trades: List[Dict],
    actions: np.ndarray,
    consensus: float
) -> Tuple[np.ndarray, float, bool, bool, Dict]:
    """Finalize step with proper memory module integration"""
    # 1) Store trades and calculate PnL
    self.trades = trades
    self.episode_metrics.trades.extend(trades)
    step_pnl = sum(t.get("pnl", 0.0) for t in trades)
    self.episode_metrics.pnls.append(step_pnl)

    # 2) Update regime performance matrix
    vol = self._get_current_volatility()
    try:
        reg_label, _ = self.fractal_confirm.step(
            data_dict=self.data,
            current_step=self.market_state.current_step,
            theme_detector=self.theme_detector
        )
        regime_map = {"noise": 0, "volatile": 1, "trending": 2}
        pred_idx = regime_map.get(reg_label, 0)
        self.regime_matrix.step(
            pnl=step_pnl,
            volatility=vol,
            predicted_regime=pred_idx
        )
    except Exception as e:
        self.logger.error(f"Error updating regime matrix: {e}")

    # 3) Calculate reward
    reward = self._calculate_reward(trades, actions, consensus)

    # 4) Build next observation BEFORE feeding memory
    obs = self._get_next_observation(trades, actions)

    # 5) FIXED: Feed memory modules with proper data
    self._feed_memory_modules(trades, actions, obs)

    # 6) Check termination
    terminated = self._check_termination()

    # 7) Pack info dict
    info = self._create_step_info(trades, step_pnl, consensus)

    # 8) Update trading mode
    self._update_mode_manager(trades, step_pnl, consensus)

    # 9) Advance step counter
    self.market_state.current_step += 1
    self.current_step = self.market_state.current_step

    # 10) Handle end of episode with memory updates
    if terminated:
        self._handle_episode_end(step_pnl)
        # FIXED: Update memory modules at episode end
        self._update_memory_compressor(self.episode_metrics.trades)
        self._record_episode_in_replay_analyzer()
        
        # Update memory budget optimizer
        try:
            self.memory_budget.optimize_allocation(self.episode_count)
        except Exception as e:
            self.logger.error(f"Error optimizing memory allocation: {e}")

    return obs, float(reward), terminated, False, info


def _create_reset_info(self) -> Dict[str, Any]:
    """Create info dict for reset"""
    return {
        "episode": self.episode_count,
        "balance": self.market_state.balance,
        "genome": self.current_genome,
        "start_step": self.market_state.current_step,
    }


def _create_step_info(
    self,
    trades: List[Dict],
    pnl: float,
    consensus: float
) -> Dict[str, Any]:
    """Create info dict for step"""
    return {
        "balance": self.market_state.balance,
        "pnl": pnl,
        "drawdown": self.market_state.current_drawdown,
        "trades": len(trades),
        "consensus": consensus,
        "mode": self.mode_manager.get_mode(),
        "step": self.market_state.current_step,
        "positions": len(self.position_manager.open_positions),
    }


def _update_mode_manager(
    self,
    trades: List[Dict],
    pnl: float,
    consensus: float
):
    """Update trading mode based on performance"""
    self.mode_manager.update(
        pnl=pnl,
        drawdown=self.market_state.current_drawdown,
        consensus=consensus,
        trade_count=len(trades),
    )


def _handle_episode_end(self, final_pnl: float):
    """Handle episode termination"""
    # Log episode summary
    total_pnl = sum(self.episode_metrics.pnls)
    total_trades = len(self.episode_metrics.trades)
    max_dd = max(self.episode_metrics.drawdowns) if self.episode_metrics.drawdowns else 0
    
    self.logger.info(
        f"Episode {self.episode_count} ended: "
        f"PnL={total_pnl:.2f}, Trades={total_trades}, MaxDD={max_dd:.2%}"
    )
    
    # Save checkpoints
    if self.episode_count % 100 == 0:
        try:
            self._save_checkpoints()
        except Exception as e:
            self.logger.error(f"Failed to save checkpoints: {e}")


def _save_checkpoints(self):
    """Save all module checkpoints"""
    try:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        
        # Save environment state
        state_path = os.path.join(
            self.config.checkpoint_dir,
            f"env_state_ep{self.episode_count}.pkl"
        )
        with open(state_path, "wb") as f:
            pickle.dump(self.get_state(), f)
            
        # Save module states
        modules_to_save = [
            self.position_manager,
            self.risk_controller,
            self.risk_system,
            self.strategy_pool,
            self.mistake_memory,
            self.meta_rl,
        ]
        
        for module in modules_to_save:
            if hasattr(module, 'save_checkpoint'):
                try:
                    module.save_checkpoint(
                        os.path.join(
                            self.config.checkpoint_dir,
                            f"{module.__class__.__name__}_ep{self.episode_count}.pkl"
                        )
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to save {module.__class__.__name__}: {e}"
                    )
                    
        self.logger.info(f"Saved checkpoints for episode {self.episode_count}")
        
    except Exception as e:
        self.logger.error(f"Checkpoint save failed: {e}")


def set_module_enabled(self, name: str, enabled: bool):
    """Enable or disable a module"""
    if name not in self.module_enabled:
        raise KeyError(f"Unknown module: {name}")
    self.module_enabled[name] = enabled
    self.logger.info(f"Module {name} {'enabled' if enabled else 'disabled'}")


def get_state(self) -> Dict[str, Any]:
    """Get complete environment state for serialization"""
    return {
        "market_state": {
            "balance": self.market_state.balance,
            "peak_balance": self.market_state.peak_balance,
            "current_step": self.market_state.current_step,
            "current_drawdown": self.market_state.current_drawdown,
            "last_trade_step": self.market_state.last_trade_step,
        },
        "episode_metrics": {
            "pnls": self.episode_metrics.pnls,
            "durations": self.episode_metrics.durations,
            "drawdowns": self.episode_metrics.drawdowns,
            "trades": self.episode_metrics.trades,
            "votes_log": self.episode_metrics.votes_log,
            "reasoning_trace": self.episode_metrics.reasoning_trace,
        },
        "episode_count": self.episode_count,
        "position_manager": self.position_manager.get_state(),
    }


def set_state(self, state: Dict[str, Any]):
    """Restore environment state from serialization"""
    # Restore market state
    ms = state.get("market_state", {})
    self.market_state.balance = ms.get("balance", self.config.initial_balance)
    self.market_state.peak_balance = ms.get("peak_balance", self.market_state.balance)
    self.market_state.current_step = ms.get("current_step", 0)
    self.market_state.current_drawdown = ms.get("current_drawdown", 0.0)
    self.market_state.last_trade_step = ms.get("last_trade_step", {})
    
    # Restore episode metrics
    em = state.get("episode_metrics", {})
    self.episode_metrics.pnls = em.get("pnls", [])
    self.episode_metrics.durations = em.get("durations", [])
    self.episode_metrics.drawdowns = em.get("drawdowns", [])
    self.episode_metrics.trades = em.get("trades", [])
    self.episode_metrics.votes_log = em.get("votes_log", [])
    self.episode_metrics.reasoning_trace = em.get("reasoning_trace", [])
    
    # Restore other state
    self.episode_count = state.get("episode_count", 0)
    self.current_step = self.market_state.current_step  # FIXED: Sync step counters
    
    # Restore position manager
    if "position_manager" in state:
        self.position_manager.set_state(state["position_manager"])


def render(self, mode: str = "human"):
    """Render the environment"""
    if mode == "human":
        # Text output
        print(
            f"Step {self.market_state.current_step} | "
            f"Mode: {self.mode_manager.get_mode().upper()} | "
            f"Balance: ${self.market_state.balance:.2f} | "
            f"Drawdown: {self.market_state.current_drawdown:.2%} | "
            f"Trades: {len(self.episode_metrics.trades)}"
        )
    elif mode == "rgb_array":
        # Could implement visual rendering here
        return None
        
    return None


def close(self):
    """Clean up resources"""
    # Save final checkpoints
    try:
        self._save_checkpoints()
    except Exception as e:
        self.logger.error(f"Failed to save final checkpoints: {e}")
        
    # Close loggers
    for handler in self.logger.handlers:
        handler.close()
        
    self.logger.info("Environment closed")