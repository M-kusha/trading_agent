# ─────────────────────────────────────────────────────────────
# modules/meta/metar_rl_controller.py

from __future__ import annotations
import logging
import numpy as np
import torch
from typing import Dict, Any
from modules.core.core import Module
from modules.meta.ppo_agent import PPOAgent
from modules.meta.ppo_lag_agent import PPOLagAgent


class MetaRLController(Module):
    def __init__(self, obs_size: int, act_size: int=2, method="ppo-lag", 
                 device="cpu", debug=True, profit_target=150.0):
        self.device = device
        self.obs_size = obs_size
        self.act_size = act_size
        self.debug = debug
        self.profit_target = profit_target
        self._step_count = 0

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"MetaRLController_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/controller/metarl_controller.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        try:
            # Initialize only PPO variants
            self._agents = {
                "ppo": PPOAgent(obs_size, act_size=act_size, device=device, debug=debug),
                "ppo-lag": PPOLagAgent(obs_size, act_size=act_size, device=device, debug=debug),
            }
            
            # FIX: Default to ppo-lag for profit generation
            self.mode = method if method in self._agents else "ppo-lag"
            self.agent = self._agents[self.mode]
            
            # Performance tracking
            self.episode_count = 0
            self.total_profit = 0.0
            self.best_daily_profit = 0.0
            self.last_profit_check = 0

            self.logger.info(f"MetaRLController initialized with {self.mode} agent for €{profit_target}/day target")
            
        except Exception as e:
            self.logger.error(f"Error initializing MetaRLController: {e}")
            raise

    def set_mode(self, method: str):
        """Switch between PPO variants with validation"""
        try:
            if method not in self._agents:
                self.logger.warning(f"Unknown method: {method}, keeping {self.mode}")
                return
                
            old_mode = self.mode
            self.mode = method
            self.agent = self._agents[method]
            self.logger.info(f"Switched from {old_mode} to {self.mode}")
            
        except Exception as e:
            self.logger.error(f"Error setting mode: {e}")

    def record_step(self, obs_vec, reward, **market_data):
        """
        FIX: Enhanced recording with market data support and comprehensive validation
        """
        self._step_count += 1
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.logger.error(f"NaN in observation vector: {obs_vec}")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.logger.error("NaN reward, setting to 0")
                reward = 0.0
                
            self.total_profit += reward
            
            # Validate market data
            validated_market_data = {}
            for key, value in market_data.items():
                if isinstance(value, (int, float)) and np.isnan(value):
                    self.logger.warning(f"NaN in market data {key}, setting to 0")
                    validated_market_data[key] = 0.0
                else:
                    validated_market_data[key] = value
            
            if self.mode == "ppo-lag":
                # PPO-Lag needs full market data
                self.agent.record_step(obs_vec, reward, **validated_market_data)
            else:
                # Standard PPO
                self.agent.record_step(obs_vec, reward)
                
            # Log significant profits
            if reward > 10:  # €10+ single trade
                self.logger.info(f"Profitable trade: €{reward:.2f} using {self.mode}")
            elif reward < -10:  # €10+ loss
                self.logger.warning(f"Large loss: €{reward:.2f} using {self.mode}")
                
            # Log progress periodically
            if self._step_count % 50 == 0:
                progress = (self.total_profit / self.profit_target) * 100
                self.logger.info(f"Step {self._step_count}: Progress {progress:.1f}% (€{self.total_profit:.2f}/€{self.profit_target})")
                
        except Exception as e:
            self.logger.error(f"Error in record_step: {e}")

    def end_episode(self, *args, **kwargs):
        """End episode with performance tracking and comprehensive logging"""
        try:
            self.episode_count += 1
            
            # Check daily profit progress
            if self.total_profit > self.best_daily_profit:
                improvement = self.total_profit - self.best_daily_profit
                self.best_daily_profit = self.total_profit
                self.logger.info(f"New best daily profit: €{self.best_daily_profit:.2f} (improvement: €{improvement:.2f})")
                
            # Log episode summary
            progress_pct = (self.total_profit / self.profit_target) * 100
            self.logger.info(f"Episode {self.episode_count} complete: total_profit=€{self.total_profit:.2f} ({progress_pct:.1f}% of target)")
                
            # Switch algorithms if underperforming
            if self.episode_count % 50 == 0:
                if self.total_profit < self.profit_target * 0.3:  # Less than 30% of target
                    new_mode = "ppo-lag" if self.mode == "ppo" else "ppo"
                    self.logger.warning(f"Underperforming after {self.episode_count} episodes - switching to {new_mode}")
                    self.set_mode(new_mode)
                else:
                    self.logger.info(f"Performance satisfactory after {self.episode_count} episodes")
                    
            return self.agent.end_episode(*args, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error in end_episode: {e}")

    def get_observation_components(self):
        """Always return 4-component vector with comprehensive validation"""
        try:
            obs = self.agent.get_observation_components()
            
            # Ensure exactly 4 components
            if len(obs) < 4:
                obs = np.pad(obs, (0, 4 - len(obs)))
            elif len(obs) > 4:
                obs = obs[:4]
                
            # Check for NaN
            if np.any(np.isnan(obs)):
                self.logger.error(f"NaN in observation: {obs}")
                obs = np.nan_to_num(obs)
                
            # Validate range
            obs = np.clip(obs, -100, 100)  # Reasonable bounds
            
            return obs.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_state(self):
        """Save controller state with validation"""
        try:
            return {
                "mode": self.mode,
                "agents": {k: v.get_state() for k, v in self._agents.items()},
                "episode_count": self.episode_count,
                "total_profit": self.total_profit,
                "best_daily_profit": self.best_daily_profit,
                "step_count": self._step_count,
            }
        except Exception as e:
            self.logger.error(f"Error getting state: {e}")
            return {}

    def set_state(self, state, strict=False):
        """Load controller state with validation"""
        try:
            self.mode = state.get("mode", self.mode)
            self.episode_count = state.get("episode_count", 0)
            self.total_profit = state.get("total_profit", 0.0)
            self.best_daily_profit = state.get("best_daily_profit", 0.0)
            self._step_count = state.get("step_count", 0)
            
            # Validate loaded values
            if np.isnan(self.total_profit):
                self.logger.error("NaN total_profit in loaded state")
                self.total_profit = 0.0
            if np.isnan(self.best_daily_profit):
                self.logger.error("NaN best_daily_profit in loaded state")
                self.best_daily_profit = 0.0
            
            agents_state = state.get("agents", {})
            for agent_name, agent_state in agents_state.items():
                if agent_name in self._agents:
                    self._agents[agent_name].set_state(agent_state, strict=strict)
                    
            self.agent = self._agents[self.mode]
            self.logger.info(f"State loaded: mode={self.mode}, episodes={self.episode_count}, profit=€{self.total_profit:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error setting state: {e}")

    def reset(self):
        """Reset all agents with comprehensive cleanup"""
        try:
            for agent in self._agents.values():
                agent.reset()
            self.episode_count = 0
            self.total_profit = 0.0
            self.best_daily_profit = 0.0
            self._step_count = 0
            self.logger.info("All agents reset")
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")

    def act(self, obs_tensor):
        """Get action from active agent with validation"""
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.logger.error("NaN in action input tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
                
            action = self.agent.select_action(obs_tensor)
            
            # Ensure valid numpy array
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            action = np.asarray(action)
            
            # Check for NaN
            if np.isnan(action).any():
                self.logger.error(f"NaN in action output: {action}")
                action = np.nan_to_num(action)
                
            # Validate action range
            action = np.clip(action, -1.0, 1.0)
                
            return action
            
        except Exception as e:
            self.logger.error(f"Error in act: {e}")
            return np.zeros(self.act_size, dtype=np.float32)

    def step(self, *args, **kwargs):
        """Compatibility method"""
        pass
        
    def obs_dim(self):
        return self.obs_size

    def save_checkpoint(self, filepath: str):
        """Save full checkpoint with validation"""
        try:
            checkpoint = {
                "controller_state": self.get_state(),
                "timestamp": str(np.datetime64('now')),
                "profit_achieved": self.total_profit,
                "target": self.profit_target
            }
            torch.save(checkpoint, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, filepath: str):
        """Load checkpoint with validation"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.set_state(checkpoint["controller_state"])
            self.logger.info(f"Loaded checkpoint from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def get_weights(self) -> Dict[str, Any]:
        """Get active agent weights"""
        try:
            if hasattr(self.agent, "get_weights"):
                return self.agent.get_weights()
            return {}
        except Exception as e:
            self.logger.error(f"Error getting weights: {e}")
            return {}

    def get_gradients(self) -> Dict[str, Any]:
        """Get active agent gradients"""
        try:
            if hasattr(self.agent, "get_gradients"):
                return self.agent.get_gradients()
            return {}
        except Exception as e:
            self.logger.error(f"Error getting gradients: {e}")
            return {}

