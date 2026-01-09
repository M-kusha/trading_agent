"""
train/callbacks/episode_callback.py

VecEpisodeTradingCallback - Correct per-env episode tracking for VecEnv.
Logs trading metrics from info dict at episode end.

Extracted from train_prop_firm.py for modularity.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

logger = logging.getLogger(__name__)


class VecEpisodeTradingCallback(BaseCallback):
    """
    Correct per-env episode tracking for VecEnv.
    Logs trading metrics from info dict at episode end.
    
    Also integrates with UnifiedMemory for online learning during training.
    
    AUDIT FIX (CRIT-5): Uses bounded deques instead of unbounded lists to prevent
    memory leaks during long training runs. Cumulative stats tracked separately.
    """
    
    # Maximum episodes to keep in rolling window
    MAX_EPISODE_HISTORY = 1000

    def __init__(self, total_timesteps: int, log_interval_steps: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.log_interval_steps = log_interval_steps
        self._last_log = 0
        self._n_envs = 1

        # AUDIT FIX (CRIT-5): Bounded deques prevent memory leaks
        self._ep_rewards: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_lens: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_pnls: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_trades: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_wrs: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_dds: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        
        # New trading metrics from prop_firm_env (bounded)
        self._ep_r_multiples: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_profit_factors: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_avg_maes: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_avg_mfes: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_avg_bars_held: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_avg_entry_quality: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_consecutive_wins: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._ep_consecutive_losses: deque = deque(maxlen=self.MAX_EPISODE_HISTORY)
        self._exit_reason_counts: Dict[str, int] = {}  # Aggregate across all episodes
        
        # Reward component tracking for dashboard
        self._reward_component_totals: Dict[str, float] = {}  # Aggregate totals
        self._reward_component_counts: Dict[str, int] = {}  # Count of occurrences
        
        # AUDIT FIX (CRIT-5): Cumulative stats for all-time metrics (O(1) update)
        self._cumulative_pnl: float = 0.0
        self._cumulative_trades: int = 0
        self._cumulative_wins: int = 0
        self._cumulative_episodes: int = 0

        self._cur_rewards: List[float] = []
        self._cur_lens: List[int] = []
        
        # PPO training diagnostics (updated after each rollout)
        self._diagnostics: Dict[str, float] = {}
        
        # Timing for FPS calculation
        self._start_time: Optional[float] = None
        self._last_metrics_save: float = 0.0  # For metrics save throttling
        
        # Memory system integration (lazy initialization)
        self._unified_memory: Optional[Any] = None
        self._memory_init_attempted: bool = False

    def _on_training_start(self) -> None:
        env = self.training_env
        self._n_envs = int(getattr(env, "num_envs", 1))
        self._cur_rewards = [0.0 for _ in range(self._n_envs)]
        self._cur_lens = [0 for _ in range(self._n_envs)]
        self._start_time = time.time()  # Initialize FPS tracking
        self._last_metrics_save = self._start_time  # Initialize metrics save throttle

    def _on_rollout_end(self) -> None:
        """Capture PPO diagnostics after each rollout (before update)."""
        # SB3 stores training stats in model.logger.name_to_value
        try:
            if hasattr(self.model, 'logger') and self.model.logger is not None:
                logger_dict = getattr(self.model.logger, 'name_to_value', {})
                
                # Map SB3 logger keys to our diagnostics
                # F-5 FIX: Use 'train/entropy' (true entropy) not 'train/entropy_loss' (scaled, signed)
                key_mapping = {
                    'train/approx_kl': 'kl_divergence',  # F-10 FIX: Use consistent key name
                    'train/clip_fraction': 'clip_fraction',
                    'train/entropy': 'entropy',  # F-5 FIX: true entropy, not entropy_loss
                    'train/explained_variance': 'explained_variance',
                    'train/value_loss': 'value_loss',
                    'train/policy_gradient_loss': 'policy_loss',
                    'train/loss': 'total_loss',
                    'train/learning_rate': 'learning_rate',
                    'time/fps': 'fps',
                    'train/clip_range': 'clip_range',
                    'train/n_updates': 'n_updates',
                }
                
                for sb3_key, our_key in key_mapping.items():
                    if sb3_key in logger_dict:
                        self._diagnostics[our_key] = float(logger_dict[sb3_key])
                
                # Also try rollout keys
                rollout_keys = ['rollout/ep_rew_mean', 'rollout/ep_len_mean']
                for key in rollout_keys:
                    if key in logger_dict:
                        short_key = key.split('/')[-1]
                        self._diagnostics[short_key] = float(logger_dict[key])
            
            # Try to get FPS from model's num_timesteps and elapsed time
            if 'fps' not in self._diagnostics or self._diagnostics.get('fps', 0) == 0:
                if self._start_time is not None:
                    elapsed = time.time() - self._start_time
                    if elapsed > 0:
                        self._diagnostics['fps'] = self.num_timesteps / elapsed
                else:
                    self._start_time = time.time()
            
            # Get n_updates from model if not in logger
            if 'n_updates' not in self._diagnostics or self._diagnostics.get('n_updates', 0) == 0:
                if hasattr(self.model, '_n_updates'):
                    self._diagnostics['n_updates'] = int(self.model._n_updates)
            
            # Get learning rate from model
            if 'learning_rate' not in self._diagnostics or self._diagnostics.get('learning_rate', 0) == 0:
                if hasattr(self.model, 'learning_rate'):
                    lr = self.model.learning_rate
                    if callable(lr):
                        # Learning rate schedule
                        progress = self.num_timesteps / max(self.model._total_timesteps, 1) if hasattr(self.model, '_total_timesteps') else 1.0
                        lr = lr(1.0 - progress)
                    self._diagnostics['learning_rate'] = float(lr)
            
            # Get clip_range from model
            if 'clip_range' not in self._diagnostics:
                if hasattr(self.model, 'clip_range'):
                    cr: Any = getattr(self.model, 'clip_range', 0.2)
                    if callable(cr):
                        progress = self.num_timesteps / max(getattr(self.model, '_total_timesteps', 1), 1)
                        cr = cr(1.0 - progress)
                    self._diagnostics['clip_range'] = float(cr) if cr is not None else 0.2
                    
        except Exception as e:
            logger.debug(f"Diagnostics collection failed: {e}")

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)

        if rewards is None or dones is None or infos is None:
            return True

        rewards_arr = np.array(rewards, dtype=np.float64).reshape(-1)
        dones_arr = np.array(dones, dtype=np.bool_).reshape(-1)

        # Accumulate per-env running reward/length
        for i in range(min(self._n_envs, len(rewards_arr))):
            self._cur_rewards[i] += float(rewards_arr[i])
            self._cur_lens[i] += 1

        # Handle episode termination per env
        for i, done in enumerate(dones_arr[:self._n_envs]):
            if not done:
                continue

            info = infos[i] if i < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            # Capture episode reward/length BEFORE resetting
            ep_reward = self._cur_rewards[i]
            ep_len = self._cur_lens[i]

            self._ep_rewards.append(ep_reward)
            self._ep_lens.append(ep_len)

            # SB3 DummyVecEnv stores terminal info under 'terminal_observation' and 'terminal_info'
            # For VecFrameStack, terminal info is preserved in the info dict
            finfo = info.get("terminal_info", info)

            # Extract metrics - try both direct and nested
            pnl = float(finfo.get("total_pnl", info.get("total_pnl", 0.0)))
            trades = int(finfo.get("trade_count", info.get("trade_count", 0)))
            wr = float(finfo.get("win_rate", info.get("win_rate", 0.0)))
            dd = float(finfo.get("drawdown", info.get("drawdown", 0.0)))

            self._ep_pnls.append(pnl)
            self._ep_trades.append(trades)
            self._ep_wrs.append(wr)
            self._ep_dds.append(dd)
            
            # AUDIT FIX (CRIT-5): Update cumulative stats (O(1) instead of O(n) sum)
            self._cumulative_pnl += pnl
            self._cumulative_trades += trades
            self._cumulative_wins += int(trades * wr) if trades > 0 else 0
            self._cumulative_episodes += 1

            # Get episode_stats from env (contains aggregated trade metrics)
            ep_stats = finfo.get("episode_stats", info.get("episode_stats", {})) or {}

            # Use episode_stats for accurate metrics (falls back to per-step if not available)
            ep_r_mult = float(ep_stats.get("avg_r_multiple", 0.0))
            ep_mae = float(ep_stats.get("avg_mae", 0.0))
            ep_mfe = float(ep_stats.get("avg_mfe", 0.0))
            ep_bars = float(ep_stats.get("avg_bars_held", 0.0))
            ep_entry_q = float(ep_stats.get("avg_entry_quality", 0.5))
            ep_pf = float(ep_stats.get("profit_factor", 0.0))
            if ep_pf == float('inf'):
                ep_pf = 10.0  # Cap infinite profit factor

            self._ep_r_multiples.append(ep_r_mult)
            self._ep_avg_maes.append(ep_mae)
            self._ep_avg_mfes.append(ep_mfe)
            self._ep_avg_bars_held.append(ep_bars)
            self._ep_avg_entry_quality.append(ep_entry_q)
            self._ep_profit_factors.append(ep_pf)

            # Consecutive wins/losses - prefer max_consecutive_losses_reached (peak during episode)
            # over consecutive_losses (current at episode end, which may be 0 if last trade was win)
            cons_wins = int(finfo.get("consecutive_wins", info.get("consecutive_wins", 0)))
            # Use max_consecutive_losses_reached from episode_stats if available (peak during episode)
            max_cons_losses = int(ep_stats.get("max_consecutive_losses_reached", 
                                               finfo.get("consecutive_losses", info.get("consecutive_losses", 0))))
            self._ep_consecutive_wins.append(cons_wins)
            self._ep_consecutive_losses.append(max_cons_losses)

            # Exit reason tracking from episode_stats (contains all trade close reasons)
            exit_dist = ep_stats.get("exit_quality_distribution", {}) or {}
            for reason, count in exit_dist.items():
                self._exit_reason_counts[reason] = self._exit_reason_counts.get(reason, 0) + count

            # Reward component tracking from episode_stats
            reward_components = ep_stats.get("reward_components", {}) or {}
            for comp_name, comp_data in reward_components.items():
                if isinstance(comp_data, dict):
                    total = float(comp_data.get("total", 0.0))
                    count = int(comp_data.get("count", 0))
                else:
                    total = float(comp_data)
                    count = 1
                self._reward_component_totals[comp_name] = self._reward_component_totals.get(comp_name, 0.0) + total
                self._reward_component_counts[comp_name] = self._reward_component_counts.get(comp_name, 0) + count

            # Also track termination reason
            term_reason = str(finfo.get("termination_reason", info.get("termination_reason", "")))
            if term_reason:
                key = f"term:{term_reason}"
                self._exit_reason_counts[key] = self._exit_reason_counts.get(key, 0) + 1

            # Extract trades from episode info for memory learning
            episode_trades = finfo.get("trades", info.get("trades", []))
            if not episode_trades:
                # Construct minimal trade record from summary
                episode_trades = [{
                    "pnl": pnl,
                    "trade_count": trades,
                    "win_rate": wr,
                    "drawdown": dd,
                    "instrument": finfo.get("instrument", info.get("instrument", "UNKNOWN")),
                }]

            # Update memory system with episode data (now using correct ep_reward)
            self._update_memory_from_episode(
                episode_reward=ep_reward,
                episode_trades=episode_trades,
                market_context={
                    "regime": finfo.get("regime", info.get("regime", "unknown")),
                    "volatility": finfo.get("volatility", info.get("volatility", 0.5)),
                    "session": finfo.get("session", info.get("session", "unknown")),
                },
            )

            # Reset per-env running counters AFTER using them
            self._cur_rewards[i] = 0.0
            self._cur_lens[i] = 0

            # Save metrics with throttling (max 1Hz to avoid disk thrashing)
            now = time.time()
            if now - self._last_metrics_save >= 1.0:
                self._save_live_metrics()
                self._last_metrics_save = now

        # Periodic logging based on timesteps
        if self.num_timesteps - self._last_log >= self.log_interval_steps:
            self._log()
            self._last_log = self.num_timesteps

        return True

    def _log(self) -> None:
        if not self._ep_rewards:
            return

        n = min(50, len(self._ep_rewards))
        r = list(self._ep_rewards)[-n:]
        pnls = list(self._ep_pnls)[-n:]
        trades = list(self._ep_trades)[-n:]
        wrs = list(self._ep_wrs)[-n:]
        dds = list(self._ep_dds)[-n:]
        lens = list(self._ep_lens)[-n:]

        mean_reward = float(np.mean(r))
        mean_pnl = float(np.mean(pnls))
        total_trades = int(np.sum(trades))
        mean_wr = float(np.mean(wrs)) if wrs else 0.0  # Include ALL episodes
        max_dd = float(np.max(dds)) if dds else 0.0

        total_steps = int(np.sum(lens)) if lens else 1
        trades_per_1k = (total_trades / max(total_steps, 1)) * 1000.0
        progress = 100.0 * (self.num_timesteps / max(self.total_timesteps, 1))

        logger.info(
            f"Step {self.num_timesteps:,}/{self.total_timesteps:,} ({progress:.1f}%) | "
            f"Reward {mean_reward:+.3f} | PnL €{mean_pnl:+.0f} | "
            f"WR {mean_wr:.1%} | Trades/1k {trades_per_1k:.1f} | MaxDD {max_dd:.1%}"
        )
        
        # Save live metrics for dashboard
        self._save_live_metrics()
    
    def _save_live_metrics(self) -> None:
        """Save current training metrics to JSON for dashboard."""
        try:
            metrics_file = Path("logs/training/live_metrics.json")
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Get recent episodes for charts
            n_recent = min(100, len(self._ep_rewards))
            
            # Ensure FPS is calculated if not yet available from SB3 logger
            if 'fps' not in self._diagnostics or self._diagnostics.get('fps', 0) <= 0:
                if self._start_time is not None and self.num_timesteps > 0:
                    elapsed = time.time() - self._start_time
                    if elapsed > 0:
                        self._diagnostics['fps'] = self.num_timesteps / elapsed
            
            # Calculate ETA
            eta_seconds = 0.0
            if self._start_time and self.num_timesteps > 0:
                elapsed = time.time() - self._start_time
                remaining = self.total_timesteps - self.num_timesteps
                rate = self.num_timesteps / max(elapsed, 1)
                eta_seconds = remaining / max(rate, 1)
            
            # Calculate summary stats
            # AUDIT FIX (CRIT-5): Use cumulative stats (O(1)) instead of sum() (O(n))
            mean_reward = float(np.mean(list(self._ep_rewards)[-50:])) if self._ep_rewards else 0.0
            mean_pnl = float(np.mean(list(self._ep_pnls)[-50:])) if self._ep_pnls else 0.0
            total_pnl = self._cumulative_pnl  # O(1) cumulative instead of O(n) sum
            mean_win_rate = float(np.mean(list(self._ep_wrs)[-50:])) if self._ep_wrs else 0.0
            max_drawdown = float(np.max(list(self._ep_dds)[-50:])) if self._ep_dds else 0.0
            mean_trades = float(np.mean(list(self._ep_trades)[-50:])) if self._ep_trades else 0.0
            total_trades = self._cumulative_trades  # O(1) cumulative instead of O(n) sum
            mean_r_multiple = float(np.mean(list(self._ep_r_multiples)[-50:])) if self._ep_r_multiples else 0.0
            mean_profit_factor = float(np.mean(list(self._ep_profit_factors)[-50:])) if self._ep_profit_factors else 0.0
            mean_entry_quality = float(np.mean(list(self._ep_avg_entry_quality)[-50:])) if self._ep_avg_entry_quality else 0.5
            
            # Status calculations for dashboard
            def status_for_win_rate(wr):
                if wr >= 0.55: return "good"
                if wr >= 0.45: return "ok"
                return "bad"
            
            def status_for_drawdown(dd):
                if dd <= 0.03: return "good"
                if dd <= 0.06: return "ok"
                return "bad"
            
            def status_for_profit_factor(pf):
                if pf >= 1.5: return "good"
                if pf >= 1.0: return "ok"
                return "bad"
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                
                # Progress section (for dashboard progress display)
                "progress": {
                    "timesteps": self.num_timesteps,
                    "total_timesteps": self.total_timesteps,
                    "progress_pct": 100.0 * (self.num_timesteps / max(self.total_timesteps, 1)),
                    "total_episodes": len(self._ep_rewards),
                    "eta_seconds": eta_seconds,
                },
                
                # Learning section (PPO diagnostics)
                "learning": {
                    "fps": self._diagnostics.get('fps', 0),
                    "n_updates": self._diagnostics.get('n_updates', 0),
                    "mean_reward": mean_reward,
                    "mean_reward_status": "good" if mean_reward > 0 else "ok" if mean_reward > -5 else "bad",
                    "total_pnl": total_pnl,
                    "total_pnl_status": "good" if total_pnl > 0 else "ok" if total_pnl > -1000 else "bad",
                    "policy_loss": self._diagnostics.get('policy_loss', 0),
                    "value_loss": self._diagnostics.get('value_loss', 0),
                    "entropy": self._diagnostics.get('entropy', 0),
                    "kl_divergence": self._diagnostics.get('kl_divergence', 0),
                    "clip_fraction": self._diagnostics.get('clip_fraction', 0),
                    "explained_variance": self._diagnostics.get('explained_variance', 0),
                    "learning_rate": self._diagnostics.get('learning_rate', 0),
                },
                
                # Trading section
                "trading": {
                    "total_trades": total_trades,
                    "mean_trades": mean_trades,
                    "mean_trades_status": "good" if mean_trades >= 5 else "ok" if mean_trades >= 1 else "bad",
                    "mean_win_rate": mean_win_rate * 100,  # As percentage
                    "mean_win_rate_status": status_for_win_rate(mean_win_rate),
                    "max_drawdown": max_drawdown * 100,  # As percentage
                    "max_drawdown_status": status_for_drawdown(max_drawdown),
                },
                
                # Quality section
                "quality": {
                    "mean_profit_factor": mean_profit_factor,
                    "mean_profit_factor_status": status_for_profit_factor(mean_profit_factor),
                    "mean_r_multiple": mean_r_multiple,
                    "mean_r_multiple_status": "good" if mean_r_multiple > 1 else "ok" if mean_r_multiple > 0 else "bad",
                    "mean_entry_quality": mean_entry_quality,
                    "mean_entry_quality_status": "good" if mean_entry_quality > 0.6 else "ok" if mean_entry_quality > 0.4 else "bad",
                    "mean_mae": float(np.mean(list(self._ep_avg_maes)[-50:])) if self._ep_avg_maes else 0.0,
                    "mean_mfe": float(np.mean(list(self._ep_avg_mfes)[-50:])) if self._ep_avg_mfes else 0.0,
                    "mean_bars_held": float(np.mean(list(self._ep_avg_bars_held)[-50:])) if self._ep_avg_bars_held else 0.0,
                    "max_consecutive_wins": int(max(list(self._ep_consecutive_wins)[-50:])) if self._ep_consecutive_wins else 0,
                    "max_consecutive_losses": int(max(list(self._ep_consecutive_losses)[-50:])) if self._ep_consecutive_losses else 0,
                    # Average and rate metrics for consecutive losses (more informative than just max)
                    "avg_consecutive_losses": float(np.mean(list(self._ep_consecutive_losses)[-50:])) if self._ep_consecutive_losses else 0.0,
                    # Rate of episodes with 3+ consecutive losses (catches problematic streaks early)
                    "consecutive_loss_streak_rate": float(sum(1 for x in list(self._ep_consecutive_losses)[-50:] if x >= 3) / max(len(list(self._ep_consecutive_losses)[-50:]), 1)) if self._ep_consecutive_losses else 0.0,
                },
                
                # Exit stats
                "exit_stats": {
                    "distribution": dict(self._exit_reason_counts),
                },
                
                # Reward component breakdown for dashboard
                "reward_components": {
                    name: {
                        "total": float(total),
                        "count": self._reward_component_counts.get(name, 0),
                        "avg": float(total / self._reward_component_counts.get(name, 1)) if self._reward_component_counts.get(name, 0) > 0 else 0.0,
                    }
                    for name, total in self._reward_component_totals.items()
                },
                
                # Recent history for charts (convert deque to list for slicing)
                "recent_rewards": [float(x) for x in list(self._ep_rewards)[-n_recent:]],
                "recent_pnls": [float(x) for x in list(self._ep_pnls)[-n_recent:]],
                "recent_win_rates": [float(x) * 100 for x in list(self._ep_wrs)[-n_recent:]],  # As percentage
                "recent_drawdowns": [float(x) * 100 for x in list(self._ep_dds)[-n_recent:]],  # As percentage
                "recent_trades": [int(x) for x in list(self._ep_trades)[-n_recent:]],
                "recent_lengths": [int(x) for x in list(self._ep_lens)[-n_recent:]],
                "recent_r_multiples": [float(x) for x in list(self._ep_r_multiples)[-n_recent:]],
                "recent_entry_quality": [float(x) for x in list(self._ep_avg_entry_quality)[-n_recent:]],
                "recent_bars_held": [float(x) for x in list(self._ep_avg_bars_held)[-n_recent:]],
                
                # Legacy flat fields for backward compatibility
                "timesteps": self.num_timesteps,
                "total_timesteps": self.total_timesteps,
                "progress_pct": 100.0 * (self.num_timesteps / max(self.total_timesteps, 1)),
                "total_episodes": len(self._ep_rewards),
                "mean_reward": mean_reward,
                "mean_pnl": mean_pnl,
                "total_pnl": total_pnl,
                "mean_win_rate": mean_win_rate,
                "max_drawdown": max_drawdown,
                "mean_trades": mean_trades,
                "total_trades": total_trades,
                "mean_r_multiple": mean_r_multiple,
                "mean_profit_factor": mean_profit_factor,
                "mean_entry_quality": mean_entry_quality,
            }
            
            # AUDIT FIX: Atomic write to avoid partial JSON when process is interrupted
            tmp_file = metrics_file.with_suffix('.json.tmp')
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f)  # No indent for faster writes
            tmp_file.replace(metrics_file)  # Atomic on POSIX, near-atomic on Windows
                
            # Debug: confirm file was written
            if len(self._ep_rewards) <= 5:
                logger.debug(f"[Dashboard] Saved metrics: ep={len(self._ep_rewards)}, pnl={metrics['mean_pnl']:.4f}, wr={metrics['mean_win_rate']:.2%}, trades={metrics['mean_trades']:.1f}")
        except Exception as e:
            logger.warning(f"[Dashboard] Error saving metrics: {e}")
    
    def _get_unified_memory(self) -> Optional[Any]:
        """Lazy-load UnifiedMemory for training integration."""
        if self._memory_init_attempted:
            return self._unified_memory
        
        self._memory_init_attempted = True
        
        try:
            from modules.memory.unified_memory import UnifiedMemory, UnifiedMemoryConfig
            
            # Create memory with minimal config for training
            # UnifiedMemory.__init__ expects UnifiedMemoryConfig dataclass
            memory_config = UnifiedMemoryConfig(
                debug=False,  # Disable debug during training for performance
                enable_replay=False,  # Skip replay component (not needed for learning)
                enable_budget=False,  # Skip budget optimization
                enable_neural=True,  # Enable for encoder training
                enable_mistakes=True,  # Enable for danger zone learning
                enable_playbook=True,  # Enable for KNN learning
                enable_loss_risk_head=True,  # Enable for loss prediction
                enable_interventions=True,  # Enable for intervention learning
                enable_compression=True,  # Enable for intuition vector
            )
            
            self._unified_memory = UnifiedMemory(config=memory_config)  # type: ignore[arg-type]
            logger.info("[Memory] UnifiedMemory initialized for training integration")
            
        except Exception as e:
            logger.warning(f"[Memory] Could not initialize UnifiedMemory: {e}")
            self._unified_memory = None
        
        return self._unified_memory
    
    def _update_memory_from_episode(
        self,
        episode_reward: float,
        episode_trades: List[Dict[str, Any]],
        market_context: Dict[str, Any],
    ) -> None:
        """
        Update UnifiedMemory's learning components at episode end.
        
        This enables online learning of:
        - LossRiskHead: P(loss > τ) from trade outcomes
        - SharedEncoder: Contrastive learning (winners vs losers)
        - Interventions: Anti-relapse patterns
        """
        memory = self._get_unified_memory()
        if memory is None:
            return
        
        try:
            # Call the update_from_episode method we added to UnifiedMemory
            if hasattr(memory, 'update_from_episode'):
                stats = memory.update_from_episode(
                    episode_trades=episode_trades,
                    episode_reward=episode_reward,
                    market_context=market_context,
                )
                
                # Log occasionally (every 100 episodes)
                if len(self._ep_rewards) % 100 == 0 and stats.get("trades_processed", 0) > 0:
                    logger.info(
                        f"[Memory] Episode {len(self._ep_rewards)}: "
                        f"trades={stats.get('trades_processed', 0)}, "
                        f"loss_head={stats.get('loss_head_updates', 0)}, "
                        f"encoder={stats.get('encoder_updates', 0)}"
                    )
        except Exception as e:
            # Non-fatal: memory update failure shouldn't crash training
            if len(self._ep_rewards) <= 5:
                logger.debug(f"[Memory] Episode update failed (non-fatal): {e}")
