import os
import numpy as np
import logging
from collections import deque
from typing import List, Dict, Any, Optional
from ..core.core import Module
import datetime

def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()

class RiskAdjustedReward(Module):
    """
    FIXED: Balanced risk-adjusted reward shaper that encourages profitable trading
    while managing risk appropriately.
    
    Key improvements:
    - Positive reinforcement for successful trades
    - Progressive penalties that scale with account state
    - Better bootstrapping behavior
    - Sharpe ratio bonus for consistent performance
    - PROPER LOGGING IN ALL METHODS
    """

    def __init__(
        self,
        initial_balance: float,
        env=None,
        debug: bool = True,
        history: int = 50,
        min_trade_bonus: float = 0.5,  # Minimum bonus for taking trades
    ):
        self.initial_balance = initial_balance
        self.env = env
        self.debug = debug
        self.history_size = history
        self.min_trade_bonus = min_trade_bonus

        # More balanced reward weights (all evolvable)
        self.regime_weights = np.array([0.3, 0.4, 0.3], np.float32)  # noise, normal, trending
        
        # REDUCED penalties to encourage trading
        self.dd_pen_weight = 2.0          # Reduced from 10.0
        self.risk_pen_weight = 0.1        # Reduced from 0.3
        self.tail_pen_weight = 0.5        # Reduced from 1.0
        self.mistake_pen_weight = 0.3     # Reduced from 0.5
        self.no_trade_penalty_weight = 0.05  # Reduced from 0.1
        
        # NEW: Positive reinforcement weights
        self.win_bonus_weight = 1.0       # Bonus for profitable trades
        self.consistency_bonus_weight = 0.5  # Bonus for consistent performance
        self.sharpe_bonus_weight = 0.3    # Bonus for good risk-adjusted returns
        self.trade_frequency_bonus = 0.2  # Bonus for maintaining activity
        
        # Performance tracking
        self._reward_history = deque(maxlen=history)
        self._pnl_history = deque(maxlen=history)
        self._trade_count_history = deque(maxlen=20)
        self._last_reward = 0.0
        self._last_reason = ""
        self._total_trades = 0
        self._winning_trades = 0
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size = history

        # FIXED: Enhanced logging setup
        self._setup_logging()

    def _setup_logging(self):
        """FIXED: Proper logging configuration"""
        # Setup logging
        log_dir = os.path.join("logs", "reward")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "risk_adjusted_reward.log")

        # Create unique logger name to avoid conflicts
        logger_name = f"RiskAdjustedReward_{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.handlers.clear()  # Clear existing handlers
        
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for debug mode
        if self.debug:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter("[REWARD] %(message)s")
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Set levels
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self.logger.propagate = False  # Prevent conflicts
        
        # FIXED: Test logging to verify it works
        self.logger.info("RiskAdjustedReward initialized with enhanced logging")
        if self.debug:
            self.logger.debug("Debug logging enabled")
        
        print(f"[REWARD INIT] Logging to: {log_path}")  # Console confirmation

    def _record_audit(self, details: Dict[str, Any]) -> None:
        """FIXED: Enhanced audit recording with logging"""
        details["timestamp"] = utcnow()
        self.audit_trail.append(details)
        
        if len(self.audit_trail) > self._audit_log_size:
            self.audit_trail = self.audit_trail[-self._audit_log_size:]
        
        # FIXED: Always log audit in debug mode
        if self.debug:
            self.logger.debug(f"[AUDIT] {details}")
        
        # FIXED: Log important events even in non-debug mode
        if abs(details.get("final_reward", 0)) > 0.1:
            self.logger.info(f"Significant reward: {details.get('final_reward', 0):.4f}")

    def get_last_audit(self) -> Dict[str, Any]:
        return self.audit_trail[-1] if self.audit_trail else {}

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.audit_trail[-n:]

    # ================= Evolution methods =================

    def mutate(self, std: float = 0.1) -> None:
        """Mutate reward weights for evolutionary optimization"""
        # Mutate regime weights
        self.regime_weights += np.random.normal(0, std, size=3)
        self.regime_weights = np.clip(self.regime_weights, 0.0, 1.0)
        self.regime_weights /= (self.regime_weights.sum() + 1e-8)  # Normalize
        
        # Mutate penalty weights (keep them reasonable)
        self.dd_pen_weight = np.clip(
            self.dd_pen_weight + np.random.normal(0, std * 2), 0.5, 5.0
        )
        self.risk_pen_weight = np.clip(
            self.risk_pen_weight + np.random.normal(0, std), 0.0, 0.5
        )
        self.tail_pen_weight = np.clip(
            self.tail_pen_weight + np.random.normal(0, std), 0.0, 2.0
        )
        self.mistake_pen_weight = np.clip(
            self.mistake_pen_weight + np.random.normal(0, std), 0.0, 1.0
        )
        self.no_trade_penalty_weight = np.clip(
            self.no_trade_penalty_weight + np.random.normal(0, std * 0.5), 0.0, 0.2
        )
        
        # Mutate bonus weights
        self.win_bonus_weight = np.clip(
            self.win_bonus_weight + np.random.normal(0, std), 0.5, 2.0
        )
        self.consistency_bonus_weight = np.clip(
            self.consistency_bonus_weight + np.random.normal(0, std), 0.0, 1.0
        )
        self.sharpe_bonus_weight = np.clip(
            self.sharpe_bonus_weight + np.random.normal(0, std), 0.0, 1.0
        )
        
        if self.debug:
            self.logger.debug(f"Mutated weights: {self.get_weights()}")

    def crossover(self, other: "RiskAdjustedReward") -> "RiskAdjustedReward":
        """Create offspring via crossover"""
        child = self.__class__(
            self.initial_balance,
            env=self.env,
            debug=self.debug,
            history=self.history_size
        )
        
        # Crossover all weight attributes
        weight_attrs = [
            "regime_weights", "dd_pen_weight", "risk_pen_weight",
            "tail_pen_weight", "mistake_pen_weight", "no_trade_penalty_weight",
            "win_bonus_weight", "consistency_bonus_weight", "sharpe_bonus_weight",
            "trade_frequency_bonus"
        ]
        
        for attr in weight_attrs:
            if np.random.rand() > 0.5:
                value = getattr(other, attr)
                if isinstance(value, np.ndarray):
                    value = value.copy()
                setattr(child, attr, value)
            else:
                value = getattr(self, attr)
                if isinstance(value, np.ndarray):
                    value = value.copy()
                setattr(child, attr, value)
                
        if self.debug:
            self.logger.debug("Crossover complete.")
        return child

    def get_weights(self) -> Dict[str, Any]:
        """Get all weight parameters"""
        return {
            "regime_weights": self.regime_weights.copy(),
            "dd_pen_weight": self.dd_pen_weight,
            "risk_pen_weight": self.risk_pen_weight,
            "tail_pen_weight": self.tail_pen_weight,
            "mistake_pen_weight": self.mistake_pen_weight,
            "no_trade_penalty_weight": self.no_trade_penalty_weight,
            "win_bonus_weight": self.win_bonus_weight,
            "consistency_bonus_weight": self.consistency_bonus_weight,
            "sharpe_bonus_weight": self.sharpe_bonus_weight,
            "trade_frequency_bonus": self.trade_frequency_bonus,
        }

    # ================= Core Methods =================

    def reset(self) -> None:
        """Reset internal state"""
        self._reward_history.clear()
        self._pnl_history.clear()
        self._trade_count_history.clear()
        self._last_reward = 0.0
        self._last_reason = ""
        self._total_trades = 0
        self._winning_trades = 0
        
        if self.debug:
            self.logger.debug("Reward shaper reset")

    def _calculate_sharpe_bonus(self) -> float:
        """Calculate bonus based on Sharpe ratio of recent rewards"""
        if len(self._reward_history) < 5:
            return 0.0
            
        rewards = np.array(self._reward_history)
        if rewards.std() < 1e-8:
            return 0.0
            
        sharpe = rewards.mean() / (rewards.std() + 1e-8) * np.sqrt(len(rewards))
        # Normalize to 0-1 range
        normalized_sharpe = np.tanh(sharpe / 2.0)
        return float(normalized_sharpe * self.sharpe_bonus_weight)

    def _calculate_consistency_bonus(self) -> float:
        """Calculate bonus for consistent positive performance"""
        if len(self._pnl_history) < 3:
            return 0.0
            
        recent_pnls = list(self._pnl_history)[-10:]
        positive_ratio = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
        
        # Bonus for steady positive performance
        consistency_score = positive_ratio ** 2  # Quadratic to reward high consistency
        return float(consistency_score * self.consistency_bonus_weight)

    def _calculate_activity_bonus(self) -> float:
        """Calculate bonus for maintaining trading activity"""
        if not self._trade_count_history:
            return 0.0
            
        recent_activity = np.mean(self._trade_count_history)
        # Target 0.5-2 trades per step on average
        if recent_activity < 0.5:
            activity_score = recent_activity / 0.5
        elif recent_activity > 2.0:
            activity_score = 1.0 - (recent_activity - 2.0) / 5.0
        else:
            activity_score = 1.0
            
        return float(max(0, activity_score) * self.trade_frequency_bonus)

    def step(
        self,
        balance: float,
        trades: List[dict],
        drawdown: float,
        regime_onehot: np.ndarray,
        actions: np.ndarray,
        info: Optional[Dict[str, Any]] = None,
        **kwargs  # Accept old parameter names
    ) -> float:
        """
        Calculate risk-adjusted reward with balanced incentives.
        
        FIXED:
        - Rewards successful trades beyond just PnL
        - Progressive penalties based on account state
        - Bonuses for consistency and good risk management
        - Minimal penalty for exploration
        - COMPREHENSIVE LOGGING
        """
        # Handle legacy parameters
        if 'current_balance' in kwargs:
            balance = kwargs['current_balance']
        if 'current_drawdown' in kwargs:
            drawdown = kwargs['current_drawdown']
            
        # Prepare audit record
        rec: Dict[str, Any] = {
            "reason": None,
            "pnl": 0.0,
            "drawdown_penalty": 0.0,
            "risk_penalty": 0.0,
            "tail_penalty": 0.0,
            "regime_bonus": 0.0,
            "mistake_penalty": 0.0,
            "no_trade_penalty": 0.0,
            "win_bonus": 0.0,
            "consistency_bonus": 0.0,
            "sharpe_bonus": 0.0,
            "activity_bonus": 0.0,
            "raw_actions_norm": float(np.linalg.norm(actions)),
        }

        # Calculate PnL and track trades
        pnl = sum(t["pnl"] for t in trades) if trades else 0.0
        self._pnl_history.append(pnl)
        self._trade_count_history.append(len(trades))
        
        if trades:
            self._total_trades += len(trades)
            self._winning_trades += sum(1 for t in trades if t["pnl"] > 0)
            
        # Base reward is PnL
        reward = pnl
        rec["pnl"] = pnl
        
        # ========== PENALTIES (kept reasonable) ==========
        
        # 1. Drawdown penalty (progressive)
        if drawdown > 0.05:  # Only penalize significant drawdown
            dd_penalty = drawdown ** 2 * self.dd_pen_weight  # Quadratic scaling
            reward -= dd_penalty
            rec["drawdown_penalty"] = dd_penalty
            
        # 2. Risk penalty (minimal to avoid discouraging trades)
        risk_penalty = min(rec["raw_actions_norm"] * self.risk_pen_weight, 0.1)
        reward -= risk_penalty
        rec["risk_penalty"] = risk_penalty
        
        # 3. Tail risk penalty (only for actual losses)
        if trades:
            losses = [t["pnl"] for t in trades if t["pnl"] < 0]
            if losses:
                tail_penalty = abs(np.mean(losses)) * self.tail_pen_weight * 0.1
                reward -= tail_penalty
                rec["tail_penalty"] = tail_penalty
                
        # 4. Mistake memory penalty (if available)
        if self.env and hasattr(self.env, "mistake_memory"):
            try:
                mm_score = float(self.env.mistake_memory.get_observation_components()[0])
                mm_penalty = mm_score * self.mistake_pen_weight
                reward -= mm_penalty
                rec["mistake_penalty"] = mm_penalty
            except:
                pass
                
        # ========== BONUSES (encourage good behavior) ==========
        
        # 1. Regime bonus
        regime_bonus = float(np.dot(regime_onehot, self.regime_weights))
        reward += regime_bonus
        rec["regime_bonus"] = regime_bonus
        
        # 2. Win bonus (reward profitable trades)
        if trades:
            win_ratio = sum(1 for t in trades if t["pnl"] > 0) / len(trades)
            win_bonus = win_ratio * self.win_bonus_weight
            reward += win_bonus
            rec["win_bonus"] = win_bonus
            rec["reason"] = "trade"
        else:
            rec["reason"] = "no-trade"
            
        # 3. Consistency bonus
        consistency_bonus = self._calculate_consistency_bonus()
        reward += consistency_bonus
        rec["consistency_bonus"] = consistency_bonus
        
        # 4. Sharpe bonus
        sharpe_bonus = self._calculate_sharpe_bonus()
        reward += sharpe_bonus
        rec["sharpe_bonus"] = sharpe_bonus
        
        # 5. Activity bonus
        activity_bonus = self._calculate_activity_bonus()
        reward += activity_bonus
        rec["activity_bonus"] = activity_bonus
        
        # ========== NO-TRADE HANDLING ==========
        if not trades:
            # Very light penalty, mainly to encourage exploration
            no_trade_penalty = self.no_trade_penalty_weight
            
            # Reduce penalty if we're in drawdown (defensive is ok)
            if drawdown > 0.1:
                no_trade_penalty *= 0.5
                
            # Reduce penalty if action was small (genuinely no opportunity)
            if rec["raw_actions_norm"] < 0.1:
                no_trade_penalty *= 0.3
                
            reward -= no_trade_penalty
            rec["no_trade_penalty"] = no_trade_penalty
            
            # But still give minimum exploration bonus
            reward += self.min_trade_bonus * 0.1
            
        # ========== FINAL ADJUSTMENTS ==========
        
        # Ensure reward is bounded to prevent instability
        reward = np.clip(reward, -10.0, 10.0)
        
        # Bootstrap bonus: Be more forgiving early on
        if self._total_trades < 10:
            reward += 0.1  # Small exploration bonus
            
        # Update history and audit
        self._last_reward = float(reward)
        self._last_reason = rec["reason"]
        self._reward_history.append(reward)
        
        # Add final reward to record
        rec["final_reward"] = reward
        rec["method"] = "step"
        self._record_audit(rec)
        
        # Populate info dict if provided
        if info is not None:
            info["shaped_reward"] = reward
            info["reward_components"] = rec.copy()
            info["win_rate"] = (
                self._winning_trades / self._total_trades 
                if self._total_trades > 0 else 0.0
            )
        
        # FIXED: Always log when step() is called
        self.logger.info(
            f"step: reward={reward:.4f}, pnl={pnl:.4f}, "
            f"trades={len(trades)}, drawdown={drawdown:.3f}"
        )
        
        # Log detailed breakdown
        if self.debug or abs(reward) > 0.1:
            self.logger.debug(f"Detailed components: {rec}")
        
        # Log performance metrics
        if self._total_trades > 0 and self._total_trades % 10 == 0:
            win_rate = self._winning_trades / self._total_trades
            avg_reward = np.mean(self._reward_history) if self._reward_history else 0
            self.logger.info(
                f"Performance update: trades={self._total_trades}, "
                f"win_rate={win_rate:.3f}, avg_reward={avg_reward:.4f}"
            )
            
        return float(reward)

    def shape_reward(
        self,
        trades: List[dict],
        balance: float,
        drawdown: float,
        consensus: float,
        actions: Optional[np.ndarray] = None,
    ) -> float:
        """
        FIXED: Enhanced shape_reward with comprehensive logging and better calculation
        This is the method your environment calls!
        """
        # Calculate components
        realised_pnl = sum(t.get("pnl", 0.0) for t in trades)
        base_component = realised_pnl / (self.initial_balance + 1e-12)
        drawdown_penalty = 0.5 * drawdown
        consensus_factor = 0.5 + consensus
        
        # FIXED: Add bonuses and penalties like in step() method
        win_bonus = 0.0
        activity_bonus = 0.0
        no_trade_penalty = 0.0
        consistency_bonus = 0.0
        sharpe_bonus = 0.0
        
        if trades:
            # Win bonus
            win_ratio = sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades)
            win_bonus = win_ratio * 0.5  # Reduced compared to step() method
            
            # Activity bonus (encourage some trading)
            activity_bonus = min(len(trades) * 0.1, 0.3)
            
            # Update tracking
            self._total_trades += len(trades)
            self._winning_trades += sum(1 for t in trades if t.get("pnl", 0) > 0)
        else:
            # Light no-trade penalty
            no_trade_penalty = 0.05
            
            # Reduce penalty if in drawdown (defensive trading is ok)
            if drawdown > 0.1:
                no_trade_penalty *= 0.5
        
        # Calculate bonuses
        consistency_bonus = self._calculate_consistency_bonus()
        sharpe_bonus = self._calculate_sharpe_bonus()
        
        # Combine components
        raw_reward = (
            base_component + win_bonus + activity_bonus + consistency_bonus + sharpe_bonus - 
            drawdown_penalty - no_trade_penalty
        ) * consensus_factor
        
        final_reward = float(np.clip(raw_reward, -10.0, 10.0))
        
        # FIXED: Add comprehensive logging
        components = {
            "pnl": realised_pnl,
            "base_component": base_component,
            "drawdown_penalty": drawdown_penalty,
            "consensus_factor": consensus_factor,
            "win_bonus": win_bonus,
            "activity_bonus": activity_bonus,
            "consistency_bonus": consistency_bonus,
            "sharpe_bonus": sharpe_bonus,
            "no_trade_penalty": no_trade_penalty,
            "trades_count": len(trades),
            "balance": balance,
            "drawdown": drawdown,
            "consensus": consensus,
            "final_reward": final_reward,
            "method": "shape_reward"
        }
        
        # Update history for consistency
        self._reward_history.append(final_reward)
        self._pnl_history.append(realised_pnl)
        
        # FIXED: Always log every call (this was the missing piece!)
        self.logger.info(
            f"shape_reward: reward={final_reward:.4f}, "
            f"pnl={realised_pnl:.4f}, trades={len(trades)}, "
            f"drawdown={drawdown:.3f}, consensus={consensus:.3f}"
        )
        
        # Detailed logging in debug mode
        if self.debug:
            self.logger.debug(f"Components: {components}")
            
            if self._total_trades > 0:
                win_rate = self._winning_trades / self._total_trades
                self.logger.debug(
                    f"Performance: total_trades={self._total_trades}, "
                    f"win_rate={win_rate:.3f}"
                )
        
        # Log significant rewards even in non-debug mode
        if abs(final_reward) > 0.1:
            self.logger.info(f"Significant reward calculated: {final_reward:.4f}")
        
        # Record audit trail
        self._record_audit(components)
        
        return final_reward

    def get_observation_components(self) -> np.ndarray:
        """Return observation components for the agent"""
        if not self._reward_history:
            return np.zeros(5, np.float32)
            
        rewards = np.array(self._reward_history, np.float32)
        
        # Recent statistics
        recent_mean = rewards[-10:].mean() if len(rewards) >= 10 else rewards.mean()
        recent_std = rewards[-10:].std() if len(rewards) >= 10 else 0.1
        
        # Win rate
        win_rate = (
            self._winning_trades / max(self._total_trades, 1)
            if self._total_trades > 0 else 0.5
        )
        
        # Activity level
        activity = (
            np.mean(self._trade_count_history) 
            if self._trade_count_history else 0.0
        )
        
        return np.array([
            self._last_reward,
            recent_mean,
            recent_std,
            win_rate,
            activity
        ], np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Get state for serialization"""
        return {
            "reward_history": list(self._reward_history),
            "pnl_history": list(self._pnl_history),
            "trade_count_history": list(self._trade_count_history),
            "_last_reward": float(self._last_reward),
            "_last_reason": self._last_reason,
            "_total_trades": self._total_trades,
            "_winning_trades": self._winning_trades,
            "weights": self.get_weights(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state from serialization"""
        self._reward_history = deque(
            state.get("reward_history", []), 
            maxlen=self.history_size
        )
        self._pnl_history = deque(
            state.get("pnl_history", []), 
            maxlen=self.history_size
        )
        self._trade_count_history = deque(
            state.get("trade_count_history", []), 
            maxlen=20
        )
        self._last_reward = float(state.get("_last_reward", 0.0))
        self._last_reason = state.get("_last_reason", "")
        self._total_trades = state.get("_total_trades", 0)
        self._winning_trades = state.get("_winning_trades", 0)
        
        # Restore weights
        weights = state.get("weights", {})
        if weights:
            for key, value in weights.items():
                if hasattr(self, key):
                    if isinstance(value, list):
                        value = np.array(value, np.float32)
                    setattr(self, key, value)

    # ======================================================================
    # Debug and Testing Methods
    # ======================================================================
    
    def debug_reward_usage(self):
        """Debug utility to check reward calculation usage"""
        print("\n" + "="*50)
        print("REWARD SYSTEM DEBUG")
        print("="*50)
        
        print(f"Logger name: {self.logger.name}")
        print(f"Logger level: {self.logger.level}")
        print(f"Logger handlers: {len(self.logger.handlers)}")
        
        for i, handler in enumerate(self.logger.handlers):
            print(f"  Handler {i}: {type(handler).__name__}")
            if hasattr(handler, 'baseFilename'):
                print(f"    File: {handler.baseFilename}")
                print(f"    File exists: {os.path.exists(handler.baseFilename)}")
                if os.path.exists(handler.baseFilename):
                    size = os.path.getsize(handler.baseFilename)
                    print(f"    File size: {size} bytes")
        
        print(f"Total trades recorded: {self._total_trades}")
        print(f"Winning trades: {self._winning_trades}")
        print(f"Reward history length: {len(self._reward_history)}")
        print(f"Last reward: {self._last_reward}")
        print(f"Last reason: {self._last_reason}")
        print(f"Audit trail length: {len(self.audit_trail)}")
        
        # Test logging
        print("\nTesting logging...")
        self.logger.info("TEST INFO MESSAGE FROM DEBUG")
        self.logger.debug("TEST DEBUG MESSAGE FROM DEBUG")
        self.logger.warning("TEST WARNING MESSAGE FROM DEBUG")
        
        print("="*50)

    def test_reward_calculation(self):
        """Test reward calculation with logging"""
        print("Testing reward calculation...")
        
        # Test with trades
        test_trades = [{"pnl": 10.0}, {"pnl": -5.0}]
        reward1 = self.shape_reward(
            trades=test_trades,
            balance=3000.0,
            drawdown=0.05,
            consensus=0.6
        )
        print(f"Test reward with trades: {reward1:.4f}")
        
        # Test without trades
        reward2 = self.shape_reward(
            trades=[],
            balance=3000.0,
            drawdown=0.05,
            consensus=0.6
        )
        print(f"Test reward without trades: {reward2:.4f}")
        
        return reward1, reward2