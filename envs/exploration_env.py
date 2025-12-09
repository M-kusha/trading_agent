# envs/exploration_env.py
"""
Lightweight Exploration Environment for Autonomous PPO Training
===============================================================

This is a STANDALONE environment for training PPO without modules.
It provides simple simulated trading with direct reward signals.

Key Features:
- No SmartInfoBus/Orchestrator dependency
- Built-in trade simulation (open/close positions)
- Direct PnL-based rewards
- Same 64-dim observation space as ModernTradingEnv (for transfer learning)
- Simple risk management (SL/TP/time decay)

Use this for initial exploration training, then transfer the model to
ModernTradingEnv with full modules for fine-tuning.

Usage:
    python train/train_exploration.py --timesteps 100000
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────
@dataclass
class ExplorationConfig:
    """Configuration for exploration training environment."""
    
    # Account
    initial_balance: float = 100_000.0
    
    # Trading
    instruments: List[str] = field(default_factory=lambda: ["EURUSD", "XAUUSD"])
    primary_timeframe: str = "M15"
    
    # Position sizing
    base_lot_size: float = 0.1      # Base lot size
    max_lot_size: float = 1.0       # Maximum lot size
    
    # Risk Management (simplified)
    take_profit_eur: float = 500.0   # €500 TP
    stop_loss_eur: float = 200.0     # €200 SL
    max_position_age: int = 100      # Close after 100 steps (~25h at M15)
    max_drawdown_pct: float = 0.20   # 20% max drawdown
    
    # PPO hyperparameters
    gamma: float = 0.95              # Short-term horizon (~5h for M15)
    learning_rate: float = 3e-4
    
    # Direction thresholds (v4.1)
    direction_long_threshold: float = 0.3
    direction_short_threshold: float = -0.3
    
    # Environment
    max_steps_per_episode: int = 2000
    observation_size: int = 64       # Match PPO_OBS_SIZE
    
    # Data
    data_dir: str = "data/processed"


# ─────────────────────────────────────────────────────────────────
# Position tracking
# ─────────────────────────────────────────────────────────────────
@dataclass
class Position:
    """Simple position tracking."""
    instrument: str
    direction: str          # "long" or "short"
    entry_price: float
    entry_step: int
    lot_size: float
    peak_pnl: float = 0.0   # For trailing


# ─────────────────────────────────────────────────────────────────
# Exploration Environment
# ─────────────────────────────────────────────────────────────────
class ExplorationTradingEnv(gym.Env):
    """
    Lightweight trading environment for autonomous PPO exploration.
    
    No modules, no SmartInfoBus - just pure PPO learning on price data.
    Produces same 64-dim observations as ModernTradingEnv for transfer.
    """
    
    metadata = {"render_modes": ["human"]}
    
    def __init__(
        self,
        data_dict: Dict[str, Dict[str, pd.DataFrame]],
        config: Optional[ExplorationConfig] = None,
    ):
        super().__init__()
        
        self.config = config or ExplorationConfig()
        self.data = data_dict
        
        # Validate data
        self.instruments = [inst for inst in self.config.instruments if inst in data_dict]
        if not self.instruments:
            # Try to use whatever instruments are available
            self.instruments = list(data_dict.keys())[:2]
        
        if not self.instruments:
            raise ValueError("No valid instruments found in data")
        
        # Action space: [direction_score, size_score] per instrument
        # direction_score: [-1, 1] -> short/flat/long
        # size_score: [-1, 1] -> position size (mapped to [0, 1])
        n_instruments = len(self.instruments)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(n_instruments * 2,),
            dtype=np.float32,
        )
        
        # Observation space: 64 dims (matches PPO_OBS_SIZE)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.config.observation_size,),
            dtype=np.float32,
        )
        
        # State
        self.balance = self.config.initial_balance
        self.initial_balance = self.config.initial_balance  # Keep reference for callbacks
        self.peak_balance = self.config.initial_balance
        self.current_step = 0
        self.episode_step = 0
        self.positions: Dict[str, Position] = {}
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        
        # Episode tracking
        self._episode_start_balance = self.config.initial_balance
        self._last_balance = self.config.initial_balance
        
        # Anti-overfit: action diversity tracking
        self._episode_actions: List[float] = []
        
        # Metrics for callback
        self.total_trades = 0
        self.winning_trades = 0
        
        # Price caches
        self._prices: Dict[str, float] = {}
        self._returns: Dict[str, np.ndarray] = {}
        
        # Data info
        self._min_data_len = self._get_min_data_length()
        
    def _get_min_data_length(self) -> int:
        """Get minimum data length across all instruments."""
        min_len = float('inf')
        for inst in self.instruments:
            for tf, df in self.data.get(inst, {}).items():
                min_len = min(min_len, len(df))
        return int(min_len) if min_len != float('inf') else 0
    
    def _get_price(self, instrument: str) -> float:
        """Get current close price for instrument."""
        tf = self.config.primary_timeframe
        if tf not in self.data.get(instrument, {}):
            # Fallback to any available timeframe
            available = list(self.data.get(instrument, {}).keys())
            if not available:
                return 0.0
            tf = available[0]
        
        df = self.data[instrument][tf]
        if self.current_step >= len(df):
            return float(df["close"].iloc[-1])
        return float(df["close"].iloc[self.current_step])
    
    def _get_pip_value(self, instrument: str) -> Tuple[float, float]:
        """Get pip value and multiplier for instrument.
        
        Returns: (pip_value_per_lot, price_to_pip_multiplier)
        """
        inst = instrument.upper().replace("_", "").replace("/", "")
        
        if "XAU" in inst or "GOLD" in inst:
            # Gold: $100 per point per standard lot
            return 100.0, 1.0
        elif "XAG" in inst or "SILVER" in inst:
            # Silver: $50 per point per standard lot
            return 50.0, 1.0
        else:
            # FX pairs: $10 per pip per standard lot (for USD quote)
            return 10.0, 10000.0  # Convert price diff to pips
    
    def _calculate_pnl(self, position: Position, current_price: float) -> float:
        """Calculate unrealized PnL for a position."""
        pip_value, multiplier = self._get_pip_value(position.instrument)
        price_diff = current_price - position.entry_price
        pips = price_diff * multiplier
        
        if position.direction == "short":
            pips = -pips
        
        return pips * pip_value * position.lot_size
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode.
        
        Anti-overfitting measures:
        1. Random start position in data (different sequence each episode)
        2. Random instrument order (if multiple instruments)
        3. Small noise injection in observations (optional)
        """
        super().reset(seed=seed)
        
        # Reset account state
        self.balance = self.config.initial_balance
        self.peak_balance = self.config.initial_balance
        self._episode_start_balance = self.config.initial_balance
        self._last_balance = self.config.initial_balance
        self.positions = {}
        self.total_pnl = 0.0
        self.trade_count = 0
        self.win_count = 0
        
        # ANTI-OVERFIT: Random start position with buffer
        # Ensures each episode sees different market conditions
        buffer = 100  # Need lookback for features
        max_start = max(buffer, self._min_data_len - self.config.max_steps_per_episode - buffer)
        if max_start > buffer:
            self.current_step = self.np_random.integers(buffer, max_start)
        else:
            self.current_step = buffer
        
        # ANTI-OVERFIT: Shuffle instrument order occasionally
        # Forces model to generalize across instruments, not memorize order
        if self.np_random.random() < 0.3 and len(self.instruments) > 1:
            self.np_random.shuffle(self.instruments)
        
        self.episode_step = 0
        
        # Track episode diversity metrics
        self._episode_actions: List[float] = []  # For action diversity tracking
        
        obs = self._get_observation()
        info = {
            "balance": self.balance, 
            "step": self.current_step,
            "start_position": self.current_step,  # Track for diversity analysis
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        self.current_step += 1
        self.episode_step += 1
        
        # ANTI-OVERFIT: Track action diversity
        if hasattr(self, '_episode_actions'):
            self._episode_actions.append(float(action[0]))  # Track direction score
        
        # Parse action for each instrument
        for i, inst in enumerate(self.instruments):
            if i * 2 + 1 >= len(action):
                break
            
            direction_score = float(action[i * 2])
            size_score = float(action[i * 2 + 1])
            
            # Interpret direction
            if direction_score > self.config.direction_long_threshold:
                target_direction = "long"
            elif direction_score < self.config.direction_short_threshold:
                target_direction = "short"
            else:
                target_direction = "flat"
            
            # Calculate lot size from size_score
            raw_size = (size_score + 1.0) / 2.0  # Map [-1,1] to [0,1]
            lot_size = self.config.base_lot_size + raw_size * (
                self.config.max_lot_size - self.config.base_lot_size
            )
            lot_size = max(0.01, min(self.config.max_lot_size, lot_size))
            
            # Get current price
            current_price = self._get_price(inst)
            if current_price <= 0:
                continue
            
            # Process existing position
            if inst in self.positions:
                pos = self.positions[inst]
                pnl = self._calculate_pnl(pos, current_price)
                
                # Update peak PnL for trailing
                if pnl > pos.peak_pnl:
                    pos.peak_pnl = pnl
                
                # Check exit conditions
                should_close = False
                close_reason = ""
                
                position_age = self.episode_step - pos.entry_step
                
                # Direction change
                if target_direction == "flat" or target_direction != pos.direction:
                    should_close = True
                    close_reason = "direction_change"
                # Time decay
                elif position_age > self.config.max_position_age:
                    should_close = True
                    close_reason = "time_decay"
                # Take profit
                elif pnl >= self.config.take_profit_eur:
                    should_close = True
                    close_reason = "take_profit"
                # Stop loss
                elif pnl <= -self.config.stop_loss_eur:
                    should_close = True
                    close_reason = "stop_loss"
                # Trailing stop (50% retracement from peak)
                elif pos.peak_pnl > 100 and pnl < pos.peak_pnl * 0.5:
                    should_close = True
                    close_reason = "trailing_stop"
                
                if should_close:
                    # Close position
                    self.balance += pnl
                    self.total_pnl += pnl
                    self.trade_count += 1
                    self.total_trades += 1  # Lifetime counter for callback
                    if pnl > 0:
                        self.win_count += 1
                        self.winning_trades += 1  # Lifetime counter for callback
                    del self.positions[inst]
            
            # Open new position if no position and direction is clear
            if inst not in self.positions and target_direction in ("long", "short"):
                # Check if we have margin (simple check)
                if self.balance > self.config.initial_balance * 0.5:
                    self.positions[inst] = Position(
                        instrument=inst,
                        direction=target_direction,
                        entry_price=current_price,
                        entry_step=self.episode_step,
                        lot_size=lot_size,
                    )
        
        # Update peak balance
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check termination
        terminated = False
        truncated = False
        
        # Drawdown termination
        drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1.0)
        if drawdown >= self.config.max_drawdown_pct:
            terminated = True
        
        # Bankruptcy
        if self.balance <= 0:
            terminated = True
        
        # Episode length
        if self.episode_step >= self.config.max_steps_per_episode:
            truncated = True
        
        # Data exhaustion
        if self.current_step >= self._min_data_len - 1:
            truncated = True
        
        # Get observation
        obs = self._get_observation()
        
        # ANTI-OVERFIT: Calculate action diversity metric
        action_diversity = 0.0
        if hasattr(self, '_episode_actions') and len(self._episode_actions) > 10:
            actions = np.array(self._episode_actions)
            action_diversity = float(np.std(actions))  # Higher = more diverse actions
        
        # Build info
        info = {
            "balance": self.balance,
            "drawdown": drawdown,
            "total_pnl": self.total_pnl,
            "trade_count": self.trade_count,
            "win_rate": self.win_count / max(self.trade_count, 1),
            "open_positions": len(self.positions),
            "step": self.current_step,
            "episode_step": self.episode_step,
            # Anti-overfit metrics
            "action_diversity": action_diversity,  # Monitor for policy collapse
        }
        
        self._last_balance = self.balance
        
        return obs, reward, terminated, truncated, info
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on risk-adjusted PnL.
        
        Anti-overfitting measures:
        1. Risk-adjusted returns (Sharpe-like) instead of raw PnL
        2. Penalty for excessive trading (transaction costs)
        3. Bonus for consistent profitability vs lucky wins
        4. No penalty for staying flat (don't force bad trades)
        """
        pnl_change = self.balance - self._last_balance
        
        # Base reward: PnL change normalized by initial balance
        base_reward = pnl_change / self.config.initial_balance * 100.0
        
        # Risk adjustment: penalize high variance in returns
        # This discourages "swing for the fences" behavior
        drawdown = (self.peak_balance - self.balance) / max(self.peak_balance, 1.0)
        risk_penalty = drawdown * 0.1  # Penalize being in drawdown
        
        # Transaction cost awareness: slight penalty for each trade
        # Prevents overtrading and encourages holding good positions
        trade_penalty = 0.0
        if pnl_change != 0 and self.trade_count > 0:
            # Only penalize if we just closed a trade
            if abs(pnl_change) > 10:  # Meaningful trade closure
                trade_penalty = 0.005  # Small cost per trade
        
        # Consistency bonus: reward maintaining a good win rate
        consistency_bonus = 0.0
        if self.trade_count >= 5:  # After enough trades to measure
            win_rate = self.win_count / self.trade_count
            if win_rate > 0.5:
                consistency_bonus = (win_rate - 0.5) * 0.02  # Small bonus for good win rate
        
        # Combine rewards
        reward = base_reward - risk_penalty - trade_penalty + consistency_bonus
        
        # Clip to prevent extreme values (anti-overfitting)
        reward = float(np.clip(reward, -1.0, 1.0))
        
        # NO penalty for holding no positions!
        # Forcing trades creates bad habits. Better to wait for good setups.
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Build 64-dim observation vector.
        
        Structure (matches PPO_OBS_SIZE v4.0):
        - [0-15]:  Market features (prices, returns, volatility)
        - [16-23]: Account features (balance, equity, drawdown)
        - [24-31]: Risk features (position exposure, etc.)
        - [32-39]: Committee/consensus features (zeros in exploration)
        - [40-47]: Regime features (trend, volatility regime)
        - [48-55]: World model predictions (zeros in exploration)
        - [56-63]: Trading mode state (zeros in exploration)
        """
        obs = np.zeros(self.config.observation_size, dtype=np.float32)
        
        # Market features [0-15]
        for i, inst in enumerate(self.instruments[:2]):
            base_idx = i * 8
            
            price = self._get_price(inst)
            if price <= 0:
                continue
            
            # Get OHLC data
            tf = self.config.primary_timeframe
            if tf not in self.data.get(inst, {}):
                available = list(self.data.get(inst, {}).keys())
                if available:
                    tf = available[0]
                else:
                    continue
            
            df = self.data[inst][tf]
            if self.current_step >= len(df):
                continue
            
            # Current bar
            idx = min(self.current_step, len(df) - 1)
            
            # Price features (normalized)
            obs[base_idx + 0] = float(df["close"].iloc[idx]) / price - 1.0
            obs[base_idx + 1] = float(df["high"].iloc[idx]) / price - 1.0
            obs[base_idx + 2] = float(df["low"].iloc[idx]) / price - 1.0
            
            # Returns
            if idx > 0:
                ret_1 = (df["close"].iloc[idx] - df["close"].iloc[idx-1]) / df["close"].iloc[idx-1]
                obs[base_idx + 3] = float(np.clip(ret_1 * 100, -5, 5))
            
            # Volatility (rolling std of returns)
            if idx >= 20:
                closes = df["close"].iloc[idx-20:idx+1].to_numpy(dtype=np.float32, copy=False)
                returns = np.diff(closes) / np.maximum(closes[:-1], 1e-8)
                vol = float(np.std(returns))
                obs[base_idx + 4] = float(np.clip(vol * 100, 0, 5))
            
            # Trend (SMA crossover signal)
            if idx >= 20:
                sma_fast = float(df["close"].iloc[idx-5:idx+1].mean())
                sma_slow = float(df["close"].iloc[idx-20:idx+1].mean())
                trend = (sma_fast - sma_slow) / max(abs(sma_slow), 1e-8)
                obs[base_idx + 5] = float(np.clip(trend * 100, -2, 2))
            
            # Volume (if available)
            if "volume" in df.columns:
                vol = float(df["volume"].iloc[idx])
                avg_vol = float(df["volume"].iloc[max(0,idx-20):idx+1].mean())
                if avg_vol > 0:
                    obs[base_idx + 6] = float(np.clip(vol / avg_vol - 1.0, -2, 2))
            
            # RSI-like momentum
            if idx >= 14:
                changes = np.diff(df["close"].iloc[idx-14:idx+1].to_numpy(dtype=np.float32, copy=False))
                gains = np.sum(changes[changes > 0])
                losses = -np.sum(changes[changes < 0])
                if gains + losses > 0:
                    rsi = gains / (gains + losses)
                    obs[base_idx + 7] = float(rsi * 2 - 1)  # Map [0,1] to [-1,1]
        
        # Account features [16-23]
        obs[16] = (self.balance - self.config.initial_balance) / self.config.initial_balance
        obs[17] = self.balance / self.config.initial_balance
        obs[18] = (self.peak_balance - self.balance) / max(self.peak_balance, 1.0)  # Drawdown
        obs[19] = len(self.positions) / max(len(self.instruments), 1)  # Position ratio
        
        # Risk features [24-31]
        total_exposure = 0.0
        for pos in self.positions.values():
            price = self._get_price(pos.instrument)
            pnl = self._calculate_pnl(pos, price) if price > 0 else 0.0
            total_exposure += abs(pnl)
        
        obs[24] = total_exposure / self.config.initial_balance
        obs[25] = self.trade_count / 100.0  # Normalized trade count
        obs[26] = self.win_count / max(self.trade_count, 1)  # Win rate
        obs[27] = self.total_pnl / self.config.initial_balance
        
        # Position-specific features [28-31]
        for i, inst in enumerate(self.instruments[:2]):
            if inst in self.positions:
                pos = self.positions[inst]
                price = self._get_price(inst)
                pnl = self._calculate_pnl(pos, price) if price > 0 else 0.0
                obs[28 + i] = 1.0 if pos.direction == "long" else -1.0
                obs[30 + i] = pnl / self.config.take_profit_eur  # Normalized PnL
        
        # Regime features [40-47]
        # Simple regime detection based on first instrument
        if self.instruments:
            inst = self.instruments[0]
            tf = self.config.primary_timeframe
            if tf in self.data.get(inst, {}):
                df = self.data[inst][tf]
                idx = min(self.current_step, len(df) - 1)
                if idx >= 20:
                    closes = df["close"].iloc[idx-20:idx+1].to_numpy(dtype=np.float32, copy=False)
                    returns = np.diff(closes) / np.maximum(closes[:-1], 1e-8)
                    vol = float(np.std(returns))
                    slope = float(np.polyfit(np.arange(len(closes)), closes, 1)[0])
                    
                    # Volatility regime
                    if vol < 0.003:
                        obs[40] = -1.0  # Low vol
                    elif vol < 0.01:
                        obs[40] = 0.0   # Normal vol
                    else:
                        obs[40] = 1.0   # High vol
                    
                    # Trend regime
                    obs[41] = float(np.clip(slope * 10000, -1, 1))
        
        # Ensure no NaN/Inf
        obs = np.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return obs.astype(np.float32)
    
    def render(self, mode: str = "human") -> None:
        """Render current state."""
        print(f"Step {self.episode_step}: Balance=${self.balance:.2f}, "
              f"PnL=${self.total_pnl:.2f}, Positions={len(self.positions)}")
    
    def close(self) -> None:
        """Clean up resources."""
        pass
