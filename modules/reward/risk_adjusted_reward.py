# ─────────────────────────────────────────────────────────────
# File: modules/reward/risk_adjusted_reward.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque, defaultdict
from typing import List, Dict, Any, Optional
import datetime
import random

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import TradingMixin, AnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, extract_standard_context


def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()


class RiskAdjustedReward(Module, TradingMixin, AnalysisMixin):
    """
    Enhanced risk-adjusted reward system with infrastructure integration.
    Provides sophisticated multi-component reward calculation with InfoBus integration.
    """
    
    def __init__(self, initial_balance: float, env=None, debug: bool = True,
                 history: int = 50, min_trade_bonus: float = 0.5,
                 genome: Optional[Dict[str, Any]] = None, **kwargs):
        
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=max(history, 100),
            **kwargs
        )
        super().__init__(config)
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome, initial_balance, history, min_trade_bonus)
        
        # Enhanced state initialization
        self._initialize_module_state()
        
        # Set environment reference
        self.env = env
        
        self.log_operator_info(
            "Enhanced risk-adjusted reward initialized",
            initial_balance=f"€{initial_balance:,.0f}",
            history_size=self.history_size,
            regime_weights=f"[{', '.join(f'{w:.2f}' for w in self.regime_weights)}]",
            dd_penalty_weight=f"{self.dd_pen_weight:.2f}",
            win_bonus_weight=f"{self.win_bonus_weight:.2f}"
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], initial_balance: float,
                                    history: int, min_trade_bonus: float):
        """Initialize genome-based parameters"""
        if genome:
            self.initial_balance = float(genome.get("initial_balance", initial_balance))
            self.history_size = int(genome.get("history_size", history))
            self.min_trade_bonus = float(genome.get("min_trade_bonus", min_trade_bonus))
            
            # Regime weights
            self.regime_weights = np.array(
                genome.get("regime_weights", [0.3, 0.4, 0.3]), dtype=np.float32
            )
            
            # Penalty weights
            self.dd_pen_weight = float(genome.get("dd_pen_weight", 2.0))
            self.risk_pen_weight = float(genome.get("risk_pen_weight", 0.1))
            self.tail_pen_weight = float(genome.get("tail_pen_weight", 0.5))
            self.mistake_pen_weight = float(genome.get("mistake_pen_weight", 0.3))
            self.no_trade_penalty_weight = float(genome.get("no_trade_penalty_weight", 0.05))
            
            # Bonus weights
            self.win_bonus_weight = float(genome.get("win_bonus_weight", 1.0))
            self.consistency_bonus_weight = float(genome.get("consistency_bonus_weight", 0.5))
            self.sharpe_bonus_weight = float(genome.get("sharpe_bonus_weight", 0.3))
            self.trade_frequency_bonus = float(genome.get("trade_frequency_bonus", 0.2))
            
            # Advanced parameters
            self.volatility_adjustment = float(genome.get("volatility_adjustment", 1.0))
            self.regime_bonus_weight = float(genome.get("regime_bonus_weight", 0.2))
            self.momentum_bonus_weight = float(genome.get("momentum_bonus_weight", 0.1))
        else:
            self.initial_balance = float(initial_balance)
            self.history_size = int(history)
            self.min_trade_bonus = float(min_trade_bonus)
            
            # Default regime weights
            self.regime_weights = np.array([0.3, 0.4, 0.3], dtype=np.float32)
            
            # Default penalty weights
            self.dd_pen_weight = 2.0
            self.risk_pen_weight = 0.1
            self.tail_pen_weight = 0.5
            self.mistake_pen_weight = 0.3
            self.no_trade_penalty_weight = 0.05
            
            # Default bonus weights
            self.win_bonus_weight = 1.0
            self.consistency_bonus_weight = 0.5
            self.sharpe_bonus_weight = 0.3
            self.trade_frequency_bonus = 0.2
            
            # Default advanced parameters
            self.volatility_adjustment = 1.0
            self.regime_bonus_weight = 0.2
            self.momentum_bonus_weight = 0.1

        # Store genome for evolution
        self.genome = {
            "initial_balance": self.initial_balance,
            "history_size": self.history_size,
            "min_trade_bonus": self.min_trade_bonus,
            "regime_weights": self.regime_weights.tolist(),
            "dd_pen_weight": self.dd_pen_weight,
            "risk_pen_weight": self.risk_pen_weight,
            "tail_pen_weight": self.tail_pen_weight,
            "mistake_pen_weight": self.mistake_pen_weight,
            "no_trade_penalty_weight": self.no_trade_penalty_weight,
            "win_bonus_weight": self.win_bonus_weight,
            "consistency_bonus_weight": self.consistency_bonus_weight,
            "sharpe_bonus_weight": self.sharpe_bonus_weight,
            "trade_frequency_bonus": self.trade_frequency_bonus,
            "volatility_adjustment": self.volatility_adjustment,
            "regime_bonus_weight": self.regime_bonus_weight,
            "momentum_bonus_weight": self.momentum_bonus_weight
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_trading_state()
        self._initialize_analysis_state()
        
        # Reward calculation state
        self._reward_history = deque(maxlen=self.history_size)
        self._pnl_history = deque(maxlen=self.history_size)
        self._trade_count_history = deque(maxlen=20)
        self._last_reward = 0.0
        self._last_reason = ""
        self._call_count = 0
        
        # Enhanced tracking
        self._reward_components_history = deque(maxlen=50)
        self._performance_analytics = defaultdict(list)
        self._regime_performance = defaultdict(lambda: {'rewards': [], 'trades': [], 'pnl': []})
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size = self.history_size
        
        # Performance metrics
        self._sharpe_ratio = 0.0
        self._consistency_score = 0.0
        self._win_rate = 0.0
        self._avg_reward = 0.0
        self._reward_volatility = 0.0
        
        # Adaptive parameters
        self._adaptive_params = {
            'dynamic_penalty_scaling': 1.0,
            'regime_sensitivity': 1.0,
            'activity_threshold': 1.0,
            'risk_tolerance': 1.0
        }

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_trading_state()
        self._reset_analysis_state()
        
        # Reset reward state
        self._reward_history.clear()
        self._pnl_history.clear()
        self._trade_count_history.clear()
        self._last_reward = 0.0
        self._last_reason = ""
        self._call_count = 0
        
        # Reset tracking
        self._reward_components_history.clear()
        self._performance_analytics.clear()
        self._regime_performance.clear()
        self.audit_trail.clear()
        
        # Reset performance metrics
        self._sharpe_ratio = 0.0
        self._consistency_score = 0.0
        self._win_rate = 0.0
        self._avg_reward = 0.0
        self._reward_volatility = 0.0
        
        # Reset adaptive parameters
        self._adaptive_params = {
            'dynamic_penalty_scaling': 1.0,
            'regime_sensitivity': 1.0,
            'activity_threshold': 1.0,
            'risk_tolerance': 1.0
        }

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Process reward calculation if InfoBus available
        if info_bus:
            reward = self._calculate_reward_from_info_bus(info_bus)
            self._track_reward_performance(reward, info_bus)
        
        # Update adaptive parameters
        self._update_adaptive_parameters()
        
        # Update performance metrics
        self._update_reward_performance()

    def _calculate_reward_from_info_bus(self, info_bus: InfoBus) -> float:
        """Calculate reward from InfoBus data"""
        
        # Extract data from InfoBus
        recent_trades = info_bus.get('recent_trades', [])
        risk_snapshot = info_bus.get('risk', {})
        balance = risk_snapshot.get('balance', self.initial_balance)
        drawdown = risk_snapshot.get('current_drawdown', 0.0)
        
        # Extract market context
        context = extract_standard_context(info_bus)
        consensus = context.get('consensus', 0.5)
        
        # Extract actions if available
        actions = info_bus.get('raw_actions', None)
        
        # Calculate reward using enhanced method
        reward = self.shape_reward_from_info_bus(
            info_bus=info_bus,
            trades=recent_trades,
            balance=balance,
            drawdown=drawdown,
            consensus=consensus,
            actions=actions
        )
        
        return reward

    def shape_reward_from_info_bus(self, info_bus: InfoBus, trades: List[dict],
                                  balance: float, drawdown: float, consensus: float,
                                  actions: Optional[np.ndarray] = None) -> float:
        """Enhanced reward calculation with InfoBus integration"""
        
        # Calculate base components
        realised_pnl = sum(t.get("pnl", 0.0) for t in trades)
        base_component = realised_pnl / (self.initial_balance + 1e-12)
        
        # Initialize reward with base PnL
        reward = realised_pnl
        
        # Extract enhanced context from InfoBus
        context = extract_standard_context(info_bus)
        market_context = info_bus.get('market_context', {})
        
        # Prepare detailed components for audit
        components = {
            "pnl": realised_pnl,
            "base_component": base_component,
            "balance": balance,
            "drawdown": drawdown,
            "consensus": consensus,
            "trades_count": len(trades),
            "timestamp": info_bus.get('timestamp', utcnow()),
            "step_idx": info_bus.get('step_idx', 0),
            "market_regime": context.get('regime', 'unknown'),
            "volatility_level": context.get('volatility_level', 'medium'),
            "session": context.get('session', 'unknown'),
            
            # Component scores (initialized)
            "drawdown_penalty": 0.0,
            "risk_penalty": 0.0,
            "tail_penalty": 0.0,
            "mistake_penalty": 0.0,
            "no_trade_penalty": 0.0,
            "win_bonus": 0.0,
            "activity_bonus": 0.0,
            "consistency_bonus": 0.0,
            "sharpe_bonus": 0.0,
            "regime_bonus": 0.0,
            "volatility_adjustment": 0.0,
            "consensus_factor": 0.5 + consensus
        }
        
        # ========== ENHANCED PENALTIES ==========
        
        # 1. Drawdown penalty (progressive with regime awareness)
        if drawdown > 0.05:
            dd_penalty = drawdown ** 2 * self.dd_pen_weight
            
            # Adjust penalty based on market regime
            regime = context.get('regime', 'unknown')
            if regime == 'volatile':
                dd_penalty *= 0.8  # More tolerant in volatile markets
            elif regime == 'trending':
                dd_penalty *= 1.2  # Less tolerant in trending markets
                
            reward -= dd_penalty * self._adaptive_params['dynamic_penalty_scaling']
            components["drawdown_penalty"] = dd_penalty
            
        # 2. Enhanced risk penalty
        if actions is not None:
            risk_penalty = min(np.linalg.norm(actions) * self.risk_pen_weight, 0.2)
            
            # Adjust based on volatility
            vol_level = context.get('volatility_level', 'medium')
            vol_multiplier = {'low': 1.2, 'medium': 1.0, 'high': 0.8, 'extreme': 0.6}
            risk_penalty *= vol_multiplier.get(vol_level, 1.0)
            
            reward -= risk_penalty
            components["risk_penalty"] = risk_penalty
            
        # 3. Enhanced tail risk penalty
        if trades:
            losses = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0]
            if losses:
                tail_penalty = abs(np.mean(losses)) * self.tail_pen_weight * 0.1
                
                # Increase penalty for extreme losses
                extreme_losses = [l for l in losses if l < -100]  # Losses > €100
                if extreme_losses:
                    tail_penalty *= 1.5
                    
                reward -= tail_penalty
                components["tail_penalty"] = tail_penalty
                
        # 4. Enhanced mistake memory penalty
        mistake_penalty = self._calculate_mistake_penalty(info_bus)
        if mistake_penalty > 0:
            reward -= mistake_penalty
            components["mistake_penalty"] = mistake_penalty
        
        # ========== ENHANCED BONUSES ==========
        
        if trades:
            # Enhanced win bonus with streak detection
            win_ratio = sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades)
            win_bonus = win_ratio * self.win_bonus_weight
            
            # Bonus for winning streaks
            if len(self._pnl_history) >= 3:
                recent_wins = [pnl > 0 for pnl in list(self._pnl_history)[-3:]]
                if all(recent_wins):
                    win_bonus *= 1.3  # Streak bonus
                    
            reward += win_bonus
            components["win_bonus"] = win_bonus
            
            # Enhanced activity bonus with quality consideration
            activity_bonus = min(len(trades) * 0.1, 0.3)
            
            # Quality adjustment - higher bonus for profitable activity
            if realised_pnl > 0:
                activity_bonus *= 1.2
                
            reward += activity_bonus
            components["activity_bonus"] = activity_bonus
            
        else:
            # Enhanced no-trade penalty with context awareness
            no_trade_penalty = self.no_trade_penalty_weight * self._adaptive_params['activity_threshold']
            
            # Reduce penalty in high-risk conditions
            if drawdown > 0.1:
                no_trade_penalty *= 0.3  # Defensive trading is acceptable
            elif context.get('volatility_level') == 'extreme':
                no_trade_penalty *= 0.5  # Caution in extreme volatility
                
            reward -= no_trade_penalty
            components["no_trade_penalty"] = no_trade_penalty
        
        # ========== SOPHISTICATED BONUSES ==========
        
        # Calculate advanced bonuses
        consistency_bonus = self._calculate_consistency_bonus()
        sharpe_bonus = self._calculate_enhanced_sharpe_bonus()
        regime_bonus = self._calculate_regime_bonus(context, realised_pnl)
        volatility_adjustment = self._calculate_volatility_adjustment(context, realised_pnl)
        
        reward += consistency_bonus + sharpe_bonus + regime_bonus + volatility_adjustment
        
        components["consistency_bonus"] = consistency_bonus
        components["sharpe_bonus"] = sharpe_bonus
        components["regime_bonus"] = regime_bonus
        components["volatility_adjustment"] = volatility_adjustment
        
        # Apply consensus factor
        reward *= components["consensus_factor"]
        
        # Apply final bounds to prevent training instability
        final_reward = float(np.clip(reward, -10.0, 10.0))
        components["final_reward"] = final_reward
        components["method"] = "shape_reward_from_info_bus"
        
        # Update state
        self._update_state_once(trades, realised_pnl, final_reward)
        
        # Enhanced logging
        if self._call_count % 10 == 1 or abs(final_reward) > 0.2:
            self.log_operator_info(
                f"Reward calculated",
                reward=f"{final_reward:.4f}",
                pnl=f"€{realised_pnl:.2f}",
                trades=len(trades),
                drawdown=f"{drawdown:.1%}",
                consensus=f"{consensus:.3f}",
                regime=context.get('regime', 'unknown'),
                volatility=context.get('volatility_level', 'unknown')
            )
        
        # Record audit trail
        self._record_audit(components)
        
        return final_reward

    def _calculate_mistake_penalty(self, info_bus: InfoBus) -> float:
        """Calculate penalty from mistake memory with InfoBus integration"""
        
        try:
            # Try to get mistake data from InfoBus module data
            module_data = info_bus.get('module_data', {})
            mistake_data = module_data.get('mistake_memory', {})
            
            if mistake_data:
                mistake_score = mistake_data.get('current_score', 0.0)
                return mistake_score * self.mistake_pen_weight
            
            # Fallback to environment
            if self.env and hasattr(self.env, "mistake_memory"):
                mm_score = float(self.env.mistake_memory.get_observation_components()[0])
                return mm_score * self.mistake_pen_weight
                
        except Exception as e:
            self.log_operator_warning(f"Mistake penalty calculation failed: {e}")
            
        return 0.0

    def _calculate_enhanced_sharpe_bonus(self) -> float:
        """Enhanced Sharpe ratio calculation with stability improvements"""
        if len(self._reward_history) < 5:
            return 0.0
            
        rewards = np.array(self._reward_history)
        
        # Calculate rolling Sharpe with improved stability
        mean_reward = rewards.mean()
        std_reward = rewards.std()
        
        # Ensure minimum std to prevent explosion
        min_std = max(0.1, abs(mean_reward) * 0.1)
        std_reward = max(std_reward, min_std)
        
        # Calculate Sharpe with time adjustment
        sharpe = mean_reward / std_reward * np.sqrt(min(len(rewards), 252))
        
        # Apply regime-aware adjustment
        if hasattr(self, '_last_regime'):
            regime_multiplier = {'trending': 1.2, 'ranging': 1.0, 'volatile': 0.8, 'unknown': 0.9}
            sharpe *= regime_multiplier.get(self._last_regime, 1.0)
        
        # Conservative normalization
        normalized_sharpe = np.tanh(sharpe / 6.0)  # Further reduced sensitivity
        bonus = float(np.clip(normalized_sharpe * self.sharpe_bonus_weight, -0.5, 0.5))
        
        # Update performance metric
        self._sharpe_ratio = sharpe
        
        return bonus

    def _calculate_consistency_bonus(self) -> float:
        """Enhanced consistency bonus with streak detection"""
        if len(self._pnl_history) < 3:
            return 0.0
            
        recent_pnls = list(self._pnl_history)[-10:]
        positive_ratio = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
        
        # Base consistency score
        consistency_score = positive_ratio ** 2
        
        # Streak bonus
        if len(recent_pnls) >= 5:
            streak_length = 0
            for pnl in reversed(recent_pnls):
                if pnl > 0:
                    streak_length += 1
                else:
                    break
            
            if streak_length >= 3:
                consistency_score *= (1.0 + streak_length * 0.1)
        
        # Update performance metric
        self._consistency_score = consistency_score
        
        return float(consistency_score * self.consistency_bonus_weight)

    def _calculate_regime_bonus(self, context: Dict[str, Any], pnl: float) -> float:
        """Calculate bonus based on regime-appropriate trading"""
        
        regime = context.get('regime', 'unknown')
        self._last_regime = regime  # Store for Sharpe calculation
        
        # Track regime performance
        self._regime_performance[regime]['pnl'].append(pnl)
        
        # Regime-specific bonuses
        regime_bonus = 0.0
        
        if regime == 'trending' and pnl > 0:
            # Bonus for profitable trend following
            regime_bonus = min(pnl / 100.0, 0.2) * self.regime_bonus_weight
        elif regime == 'ranging' and abs(pnl) < 20:
            # Bonus for controlled trading in ranging markets
            regime_bonus = 0.1 * self.regime_bonus_weight
        elif regime == 'volatile':
            # Penalty for large losses in volatile markets, bonus for small profits
            if pnl < -50:
                regime_bonus = -0.15 * self.regime_bonus_weight
            elif 0 < pnl < 30:
                regime_bonus = 0.1 * self.regime_bonus_weight
        
        return regime_bonus

    def _calculate_volatility_adjustment(self, context: Dict[str, Any], pnl: float) -> float:
        """Calculate volatility-adjusted reward component"""
        
        vol_level = context.get('volatility_level', 'medium')
        
        # Volatility-based adjustments
        vol_multipliers = {
            'low': 1.1,      # Slight bonus in low vol (easier to trade)
            'medium': 1.0,   # Neutral
            'high': 0.9,     # Slight penalty in high vol
            'extreme': 0.8   # Higher penalty in extreme vol
        }
        
        vol_adjustment = (vol_multipliers.get(vol_level, 1.0) - 1.0) * abs(pnl) * 0.1
        
        # Additional bonus for profitable trading in difficult conditions
        if vol_level in ['high', 'extreme'] and pnl > 0:
            vol_adjustment += pnl * 0.05  # Extra bonus for high-vol profits
        
        return vol_adjustment * self.volatility_adjustment

    def _calculate_activity_bonus(self) -> float:
        """Enhanced activity bonus calculation"""
        if not self._trade_count_history:
            return 0.0
            
        recent_activity = np.mean(self._trade_count_history)
        
        # Optimal activity range
        optimal_range = (0.5, 2.0)
        
        if recent_activity < optimal_range[0]:
            activity_score = recent_activity / optimal_range[0]
        elif recent_activity > optimal_range[1]:
            # Penalize overtrading
            activity_score = max(0.3, 1.0 - (recent_activity - optimal_range[1]) / 3.0)
        else:
            activity_score = 1.0
            
        # Quality adjustment
        if len(self._pnl_history) >= 5:
            recent_profit_ratio = sum(1 for p in list(self._pnl_history)[-5:] if p > 0) / 5
            activity_score *= (0.5 + recent_profit_ratio)  # Scale by recent success
            
        return float(max(0, activity_score) * self.trade_frequency_bonus)

    def _update_state_once(self, trades: List[dict], pnl: float, reward: float) -> None:
        """Single state update method to prevent duplicates"""
        # Update histories
        self._pnl_history.append(pnl)
        self._trade_count_history.append(len(trades))
        self._reward_history.append(reward)
        
        # Update trade tracking via mixin
        if trades:
            for trade in trades:
                self._update_trading_metrics(trade)
            
        # Update last reward
        self._last_reward = float(reward)
        self._last_reason = "trade" if trades else "no-trade"
        self._call_count += 1

    def _record_audit(self, details: Dict[str, Any]) -> None:
        """Enhanced audit recording with controlled frequency"""
        details["timestamp"] = utcnow()
        details["call_count"] = self._call_count
        
        self.audit_trail.append(details)
        
        # Maintain audit trail size
        if len(self.audit_trail) > self._audit_log_size:
            self.audit_trail = self.audit_trail[-self._audit_log_size:]
        
        # Store components for analysis
        self._reward_components_history.append({
            'timestamp': details["timestamp"],
            'final_reward': details.get("final_reward", 0.0),
            'pnl': details.get("pnl", 0.0),
            'regime': details.get("market_regime", "unknown"),
            'trades_count': details.get("trades_count", 0)
        })

    def _track_reward_performance(self, reward: float, info_bus: InfoBus):
        """Track reward performance for continuous improvement"""
        
        context = extract_standard_context(info_bus)
        
        performance_record = {
            'timestamp': info_bus.get('timestamp', utcnow()),
            'step_idx': info_bus.get('step_idx', 0),
            'reward': reward,
            'regime': context.get('regime', 'unknown'),
            'volatility_level': context.get('volatility_level', 'medium'),
            'session': context.get('session', 'unknown')
        }
        
        # Track performance by regime
        regime = context.get('regime', 'unknown')
        self._regime_performance[regime]['rewards'].append(reward)
        
        # Update performance analytics
        self._performance_analytics['reward_by_regime'][regime].append(reward)
        self._performance_analytics['reward_by_session'][context.get('session', 'unknown')].append(reward)

    def _update_adaptive_parameters(self):
        """Update adaptive parameters based on performance"""
        
        try:
            # Adapt penalty scaling based on recent performance
            if len(self._reward_history) >= 20:
                recent_rewards = list(self._reward_history)[-20:]
                avg_reward = np.mean(recent_rewards)
                
                if avg_reward < -0.5:  # Poor performance
                    self._adaptive_params['dynamic_penalty_scaling'] = min(1.5, self._adaptive_params['dynamic_penalty_scaling'] * 1.05)
                elif avg_reward > 0.5:  # Good performance
                    self._adaptive_params['dynamic_penalty_scaling'] = max(0.5, self._adaptive_params['dynamic_penalty_scaling'] * 0.98)
            
            # Adapt activity threshold based on trading frequency
            if len(self._trade_count_history) >= 10:
                avg_activity = np.mean(self._trade_count_history)
                if avg_activity < 0.3:  # Low activity
                    self._adaptive_params['activity_threshold'] = min(2.0, self._adaptive_params['activity_threshold'] * 1.1)
                elif avg_activity > 3.0:  # High activity
                    self._adaptive_params['activity_threshold'] = max(0.5, self._adaptive_params['activity_threshold'] * 0.9)
            
            # Adapt regime sensitivity based on regime performance
            if len(self._regime_performance) >= 2:
                regime_rewards = {k: np.mean(v['rewards'][-10:]) if v['rewards'] else 0.0 
                                for k, v in self._regime_performance.items()}
                reward_variance = np.var(list(regime_rewards.values()))
                
                if reward_variance > 0.5:  # High variance suggests regime matters
                    self._adaptive_params['regime_sensitivity'] = min(1.5, self._adaptive_params['regime_sensitivity'] * 1.02)
                else:
                    self._adaptive_params['regime_sensitivity'] = max(0.7, self._adaptive_params['regime_sensitivity'] * 0.99)
                    
        except Exception as e:
            self.log_operator_warning(f"Adaptive parameter update failed: {e}")

    def _update_reward_performance(self):
        """Update comprehensive reward performance metrics"""
        
        try:
            # Update win rate
            if self._trades_processed > 0:
                self._win_rate = self._winning_trades / self._trades_processed
            
            # Update average reward
            if self._reward_history:
                self._avg_reward = np.mean(self._reward_history)
                self._reward_volatility = np.std(self._reward_history)
            
            # Update performance metrics
            self._update_performance_metric('sharpe_ratio', self._sharpe_ratio)
            self._update_performance_metric('consistency_score', self._consistency_score)
            self._update_performance_metric('win_rate', self._win_rate)
            self._update_performance_metric('avg_reward', self._avg_reward)
            
        except Exception as e:
            self.log_operator_warning(f"Reward performance update failed: {e}")

    # ================== BACKWARD COMPATIBILITY METHODS ==================

    def shape_reward(self, trades: List[dict], balance: float, drawdown: float,
                    consensus: float, actions: Optional[np.ndarray] = None) -> float:
        """
        Backward compatibility method - delegates to enhanced calculation
        """
        # Create a minimal InfoBus-like structure for compatibility
        mock_info_bus = {
            'recent_trades': trades,
            'risk': {'balance': balance, 'current_drawdown': drawdown},
            'consensus': consensus,
            'raw_actions': actions,
            'timestamp': utcnow(),
            'step_idx': self._call_count,
            'market_context': {},
            'module_data': {}
        }
        
        # Extract standard context (will be mostly defaults)
        context = {
            'regime': 'unknown',
            'session': 'unknown',
            'volatility_level': 'medium',
            'consensus': consensus
        }
        
        # Use enhanced calculation
        return self.shape_reward_from_info_bus(
            info_bus=mock_info_bus,
            trades=trades,
            balance=balance,
            drawdown=drawdown,
            consensus=consensus,
            actions=actions
        )

    def step(self, balance: float, trades: List[dict], drawdown: float,
            regime_onehot: np.ndarray, actions: np.ndarray,
            info: Optional[Dict[str, Any]] = None, **kwargs) -> float:
        """
        Backward compatibility step method
        """
        # Handle legacy parameters
        if 'current_balance' in kwargs:
            balance = kwargs['current_balance']
        if 'current_drawdown' in kwargs:
            drawdown = kwargs['current_drawdown']
            
        # Calculate consensus from regime
        consensus = float(np.dot(regime_onehot, self.regime_weights))
        
        # Use enhanced shape_reward method
        reward = self.shape_reward(
            trades=trades,
            balance=balance,
            drawdown=drawdown,
            consensus=consensus,
            actions=actions
        )
        
        # Populate info dict if provided
        if info is not None:
            info["shaped_reward"] = reward
            info["reward_components"] = self.get_last_audit()
            info["win_rate"] = self._win_rate
        
        return reward

    # ================== ENHANCED OBSERVATION AND INTERFACE METHODS ==================

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with reward metrics"""
        
        try:
            if not self._reward_history:
                return np.zeros(8, np.float32)
                
            rewards = np.array(self._reward_history, np.float32)
            
            # Recent statistics
            recent_mean = rewards[-10:].mean() if len(rewards) >= 10 else rewards.mean()
            recent_std = rewards[-10:].std() if len(rewards) >= 10 else 0.1
            
            # Performance metrics
            win_rate = self._win_rate
            activity = np.mean(self._trade_count_history) if self._trade_count_history else 0.0
            
            # Advanced metrics
            reward_trend = 0.0
            if len(rewards) >= 5:
                reward_trend = np.polyfit(range(len(rewards[-5:])), rewards[-5:], 1)[0]
            
            consistency = self._consistency_score
            sharpe = np.tanh(self._sharpe_ratio / 3.0)  # Normalized
            
            return np.array([
                self._last_reward,      # Last reward
                recent_mean,            # Recent average
                recent_std,             # Recent volatility
                win_rate,               # Win rate
                activity,               # Activity level
                reward_trend,           # Trend
                consistency,            # Consistency score
                sharpe                  # Normalized Sharpe
            ], np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)

    def get_last_audit(self) -> Dict[str, Any]:
        """Get last audit record"""
        return self.audit_trail[-1] if self.audit_trail else {}

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:
        """Get recent audit trail"""
        return self.audit_trail[-n:]

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
            "volatility_adjustment": self.volatility_adjustment,
            "regime_bonus_weight": self.regime_bonus_weight,
            "momentum_bonus_weight": self.momentum_bonus_weight,
        }

    # ================== EVOLUTIONARY METHODS ==================

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome"""
        for key, value in genome.items():
            if hasattr(self, key):
                if key == "regime_weights":
                    self.regime_weights = np.array(value, dtype=np.float32)
                else:
                    setattr(self, key, value)
        
        # Update genome
        self.genome.update(genome)
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance-based adjustments"""
        g = self.genome.copy()
        mutations = []
        
        # Mutate regime weights
        if np.random.rand() < mutation_rate:
            old_weights = g["regime_weights"].copy()
            g["regime_weights"] = np.array(g["regime_weights"]) + np.random.normal(0, 0.1, size=3)
            g["regime_weights"] = np.clip(g["regime_weights"], 0.0, 1.0)
            g["regime_weights"] = g["regime_weights"] / (g["regime_weights"].sum() + 1e-8)
            mutations.append(f"regime_weights: {old_weights} → {g['regime_weights']}")
        
        # Mutate penalty weights
        penalty_weights = ["dd_pen_weight", "risk_pen_weight", "tail_pen_weight", "mistake_pen_weight"]
        for weight_name in penalty_weights:
            if np.random.rand() < mutation_rate:
                old_val = g[weight_name]
                std = 0.2 if weight_name == "dd_pen_weight" else 0.1
                g[weight_name] = float(np.clip(old_val + np.random.normal(0, std), 0.0, 5.0))
                mutations.append(f"{weight_name}: {old_val:.3f} → {g[weight_name]:.3f}")
        
        # Mutate bonus weights
        bonus_weights = ["win_bonus_weight", "consistency_bonus_weight", "sharpe_bonus_weight"]
        for weight_name in bonus_weights:
            if np.random.rand() < mutation_rate:
                old_val = g[weight_name]
                g[weight_name] = float(np.clip(old_val + np.random.normal(0, 0.1), 0.0, 2.0))
                mutations.append(f"{weight_name}: {old_val:.3f} → {g[weight_name]:.3f}")
        
        if mutations:
            self.log_operator_info(f"Reward system mutation applied", changes=", ".join(mutations))
            
        self.set_genome(g)
        
    def crossover(self, other: "RiskAdjustedReward") -> "RiskAdjustedReward":
        """Enhanced crossover with performance-based selection"""
        if not isinstance(other, RiskAdjustedReward):
            self.log_operator_warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_performance = self._avg_reward
        other_performance = other._avg_reward
        
        # Favor higher performance parent
        if self_performance > other_performance:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {}
        for key in self.genome:
            if np.random.rand() < bias:
                new_g[key] = self.genome[key]
            else:
                new_g[key] = other.genome[key]
        
        child = RiskAdjustedReward(
            initial_balance=self.initial_balance,
            env=self.env,
            debug=self.config.debug,
            genome=new_g
        )
        
        return child

    # ================== ENHANCED STATE MANAGEMENT ==================

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check basic data consistency
            if not isinstance(self._reward_history, deque):
                return False
            if not isinstance(self.regime_weights, np.ndarray):
                return False
            if len(self.regime_weights) != 3:
                return False
                
            # Check parameter bounds
            if not (0.0 <= self.dd_pen_weight <= 10.0):
                return False
            if not (0.0 <= self.win_bonus_weight <= 5.0):
                return False
            if not np.allclose(self.regime_weights.sum(), 1.0, atol=0.01):
                return False
                
            # Check adaptive parameters
            for param_name, param_value in self._adaptive_params.items():
                if not np.isfinite(param_value):
                    return False
                    
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        reward_details = {
            'reward_info': {
                'avg_reward': self._avg_reward,
                'reward_volatility': self._reward_volatility,
                'sharpe_ratio': self._sharpe_ratio,
                'consistency_score': self._consistency_score,
                'win_rate': self._win_rate
            },
            'history_info': {
                'reward_history_size': len(self._reward_history),
                'pnl_history_size': len(self._pnl_history),
                'audit_trail_size': len(self.audit_trail),
                'call_count': self._call_count
            },
            'weight_info': {
                'regime_weights': self.regime_weights.tolist(),
                'dd_penalty_weight': self.dd_pen_weight,
                'win_bonus_weight': self.win_bonus_weight,
                'sharpe_bonus_weight': self.sharpe_bonus_weight
            },
            'adaptive_params': self._adaptive_params.copy(),
            'regime_performance': {
                regime: {
                    'reward_count': len(data['rewards']),
                    'avg_reward': np.mean(data['rewards'][-10:]) if data['rewards'] else 0.0
                }
                for regime, data in self._regime_performance.items()
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(reward_details)
            return base_details
        
        return reward_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        return {
            "reward_history": list(self._reward_history),
            "pnl_history": list(self._pnl_history),
            "trade_count_history": list(self._trade_count_history),
            "_last_reward": float(self._last_reward),
            "_last_reason": self._last_reason,
            "_call_count": self._call_count,
            "genome": self.genome.copy(),
            "adaptive_params": self._adaptive_params.copy(),
            "performance_metrics": {
                'sharpe_ratio': self._sharpe_ratio,
                'consistency_score': self._consistency_score,
                'win_rate': self._win_rate,
                'avg_reward': self._avg_reward,
                'reward_volatility': self._reward_volatility
            },
            "regime_performance": {
                regime: {
                    'rewards': data['rewards'][-20:],  # Keep recent only
                    'pnl': data['pnl'][-20:]
                }
                for regime, data in self._regime_performance.items()
            },
            "reward_components_history": list(self._reward_components_history)[-30:],
            "audit_trail": self.audit_trail[-50:]  # Keep recent audit trail
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore core state
        self._reward_history = deque(
            module_state.get("reward_history", []), 
            maxlen=self.history_size
        )
        self._pnl_history = deque(
            module_state.get("pnl_history", []), 
            maxlen=self.history_size
        )
        self._trade_count_history = deque(
            module_state.get("trade_count_history", []), 
            maxlen=20
        )
        self._last_reward = float(module_state.get("_last_reward", 0.0))
        self._last_reason = module_state.get("_last_reason", "")
        self._call_count = module_state.get("_call_count", 0)
        
        # Restore genome and parameters
        self.set_genome(module_state.get("genome", self.genome))
        self._adaptive_params = module_state.get("adaptive_params", self._adaptive_params)
        
        # Restore performance metrics
        performance_metrics = module_state.get("performance_metrics", {})
        self._sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
        self._consistency_score = performance_metrics.get('consistency_score', 0.0)
        self._win_rate = performance_metrics.get('win_rate', 0.0)
        self._avg_reward = performance_metrics.get('avg_reward', 0.0)
        self._reward_volatility = performance_metrics.get('reward_volatility', 0.0)
        
        # Restore regime performance
        regime_performance_data = module_state.get("regime_performance", {})
        self._regime_performance.clear()
        for regime, data in regime_performance_data.items():
            self._regime_performance[regime] = {
                'rewards': data.get('rewards', []),
                'trades': [],
                'pnl': data.get('pnl', [])
            }
        
        # Restore tracking data
        self._reward_components_history = deque(
            module_state.get("reward_components_history", []), 
            maxlen=50
        )
        self.audit_trail = module_state.get("audit_trail", [])

    def get_reward_system_report(self) -> str:
        """Generate operator-friendly reward system report"""
        
        # Performance status
        if self._avg_reward > 0.5:
            performance_status = "🚀 Excellent"
        elif self._avg_reward > 0.0:
            performance_status = "✅ Good"
        elif self._avg_reward > -0.5:
            performance_status = "⚡ Fair"
        else:
            performance_status = "⚠️ Poor"
        
        # Consistency status
        if self._consistency_score > 0.8:
            consistency_status = "🎯 High"
        elif self._consistency_score > 0.6:
            consistency_status = "✅ Good"
        elif self._consistency_score > 0.4:
            consistency_status = "⚡ Fair"
        else:
            consistency_status = "❌ Low"
        
        # Regime performance summary
        regime_summary = []
        for regime, data in self._regime_performance.items():
            if data['rewards']:
                avg_reward = np.mean(data['rewards'][-10:])
                regime_summary.append(f"{regime}: {avg_reward:.3f}")
        
        return f"""
🎯 ENHANCED RISK-ADJUSTED REWARD
═══════════════════════════════════════
📊 Performance: {performance_status} ({self._avg_reward:.4f})
🎯 Consistency: {consistency_status} ({self._consistency_score:.3f})
📈 Sharpe Ratio: {self._sharpe_ratio:.3f}
💰 Win Rate: {self._win_rate:.1%}

⚖️ WEIGHT CONFIGURATION
• Regime Weights: [{', '.join(f'{w:.2f}' for w in self.regime_weights)}]
• Drawdown Penalty: {self.dd_pen_weight:.2f}
• Win Bonus: {self.win_bonus_weight:.2f}
• Consistency Bonus: {self.consistency_bonus_weight:.2f}
• Sharpe Bonus: {self.sharpe_bonus_weight:.2f}

📊 PERFORMANCE METRICS
• Average Reward: {self._avg_reward:.4f}
• Reward Volatility: {self._reward_volatility:.4f}
• Total Calls: {self._call_count:,}
• Recent Activity: {np.mean(self._trade_count_history) if self._trade_count_history else 0:.1f} trades/step

🔧 ADAPTIVE PARAMETERS
• Penalty Scaling: {self._adaptive_params['dynamic_penalty_scaling']:.2f}
• Regime Sensitivity: {self._adaptive_params['regime_sensitivity']:.2f}
• Activity Threshold: {self._adaptive_params['activity_threshold']:.2f}
• Risk Tolerance: {self._adaptive_params['risk_tolerance']:.2f}

📈 REGIME PERFORMANCE
{chr(10).join([f"• {summary}" for summary in regime_summary]) if regime_summary else "• No regime data yet"}

💡 RECENT ACTIVITY
• Reward History: {len(self._reward_history)} records
• Audit Trail: {len(self.audit_trail)} entries
• Last Reward: {self._last_reward:.4f} ({self._last_reason})
• Components Tracked: {len(self._reward_components_history)} records
        """
    # ================== DEBUG AND TESTING METHODS ==================
    
    def debug_reward_usage(self):
        """Enhanced debug utility to check reward calculation usage"""
        print("\n" + "="*60)
        print("ENHANCED REWARD SYSTEM DEBUG")
        print("="*60)
        
        health_details = self._get_health_details()
        print(f"Module Health: {'✅ HEALTHY' if self._check_state_integrity() else '❌ UNHEALTHY'}")
        print(f"Logger: {self.logger.name} (Level: {self.logger.level})")
        print(f"Handlers: {len(self.logger.handlers)}")
        
        print("\n📊 PERFORMANCE METRICS:")
        print(f"  • Average Reward: {self._avg_reward:.4f}")
        print(f"  • Sharpe Ratio: {self._sharpe_ratio:.3f}")
        print(f"  • Consistency: {self._consistency_score:.3f}")
        print(f"  • Win Rate: {self._win_rate:.1%}")
        
        print("\n📈 STATE INFORMATION:")
        print(f"  • Total Calls: {self._call_count}")
        print(f"  • Trades Processed: {self._trades_processed}")
        print(f"  • Reward History: {len(self._reward_history)} records")
        print(f"  • Audit Trail: {len(self.audit_trail)} entries")
        
        print("\n🧬 GENOME CONFIGURATION:")
        for key, value in self.genome.items():
            if isinstance(value, (list, np.ndarray)):
                print(f"  • {key}: {value}")
            else:
                print(f"  • {key}: {value:.3f}")
        
        print("="*60)

    def test_reward_calculation(self):
        """Enhanced test reward calculation"""
        print("Testing enhanced reward calculation...")
        
        # Test with profitable trades
        test_trades = [{"pnl": 15.0}, {"pnl": -3.0}, {"pnl": 8.0}]
        reward1 = self.shape_reward(
            trades=test_trades,
            balance=12000.0,
            drawdown=0.03,
            consensus=0.7
        )
        print(f"✅ Test reward with profitable trades: {reward1:.4f}")
        
        # Test with losing trades
        test_trades = [{"pnl": -10.0}, {"pnl": -5.0}]
        reward2 = self.shape_reward(
            trades=test_trades,
            balance=9500.0,
            drawdown=0.08,
            consensus=0.4
        )
        print(f"❌ Test reward with losing trades: {reward2:.4f}")
        
        # Test without trades
        reward3 = self.shape_reward(
            trades=[],
            balance=10000.0,
            drawdown=0.05,
            consensus=0.5
        )
        print(f"⏸️ Test reward without trades: {reward3:.4f}")
        
        print(f"\n📊 System State After Tests:")
        print(f"  • Total Calls: {self._call_count}")
        print(f"  • Trades Processed: {self._trades_processed}")
        print(f"  • Win Rate: {self._win_rate:.1%}")
        
        return reward1, reward2, reward3

    # ================== ABSTRACT METHOD FIX FOR COMPATIBILITY ==================
    def _get_observation_impl(self, *args, **kwargs):
        """
        Satisfy abstract method requirement from Module.
        Delegates to get_observation_components for compatibility.
        """
        return self.get_observation_components()

# End of class