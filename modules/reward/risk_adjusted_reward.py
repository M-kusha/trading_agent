# ─────────────────────────────────────────────────────────────
# File: modules/reward/risk_adjusted_reward.py
# 🚀 PRODUCTION-READY Enhanced Risk-Adjusted Reward System
# Advanced reward calculation with SmartInfoBus integration and intelligent adaptation
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
import random

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


def utcnow() -> str:
    return datetime.datetime.utcnow().isoformat()


class RewardMode(Enum):
    """Reward calculation modes"""
    TRAINING = "training"
    VALIDATION = "validation"
    LIVE_TRADING = "live_trading"
    EMERGENCY = "emergency"
    OPTIMIZATION = "optimization"


@dataclass
class RewardConfig:
    """Configuration for Risk-Adjusted Reward System"""
    initial_balance: float = 10000.0
    history_size: int = 50
    min_trade_bonus: float = 0.5
    
    # Regime weights
    regime_weights: List[float] = field(default_factory=lambda: [0.3, 0.4, 0.3])
    
    # Penalty weights
    dd_pen_weight: float = 2.0
    risk_pen_weight: float = 0.1
    tail_pen_weight: float = 0.5
    mistake_pen_weight: float = 0.3
    no_trade_penalty_weight: float = 0.05
    
    # Bonus weights
    win_bonus_weight: float = 1.0
    consistency_bonus_weight: float = 0.5
    sharpe_bonus_weight: float = 0.3
    trade_frequency_bonus: float = 0.2
    
    # Advanced parameters
    volatility_adjustment: float = 1.0
    regime_bonus_weight: float = 0.2
    momentum_bonus_weight: float = 0.1
    
    # Performance thresholds
    max_processing_time_ms: float = 100
    circuit_breaker_threshold: int = 5
    min_reward_quality: float = 0.3
    
    # Adaptation parameters
    confidence_decay: float = 0.98
    performance_smoothing: float = 0.95
    adaptive_learning_rate: float = 0.01


@module(
    name="RiskAdjustedReward",
    version="4.0.0",
    category="reward",
    provides=["shaped_reward", "reward_components", "reward_analytics", "reward_performance"],
    requires=["trade_data", "risk_metrics", "market_context", "performance_data"],
    description="Advanced risk-adjusted reward system with intelligent adaptation and comprehensive analytics",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class RiskAdjustedReward(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    🚀 Advanced risk-adjusted reward system with SmartInfoBus integration.
    Provides sophisticated multi-component reward calculation with intelligent adaptation.
    """
    
    def __init__(self, 
                 config: Optional[RewardConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 env=None,
                 **kwargs):
        
        self.config = config or RewardConfig()
        self.env = env
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize reward state
        self._initialize_reward_state()
        
        self.logger.info(
            format_operator_message(
                "🎯", "REWARD_SYSTEM_INITIALIZED",
                details=f"Balance: €{self.config.initial_balance:,.0f}, History: {self.config.history_size}",
                result="Enhanced reward system ready for calculation",
                context="reward_initialization"
            )
        )

    def _initialize_advanced_systems(self):
        """Initialize advanced systems for reward calculation"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="RiskAdjustedReward", 
            log_path="logs/reward/risk_adjusted_reward.log", 
            max_lines=5000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("RiskAdjustedReward", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for reward operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        self._start_monitoring()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters with validation"""
        if genome:
            self.genome = {
                "initial_balance": float(genome.get("initial_balance", self.config.initial_balance)),
                "history_size": int(genome.get("history_size", self.config.history_size)),
                "min_trade_bonus": float(genome.get("min_trade_bonus", self.config.min_trade_bonus)),
                "regime_weights": list(genome.get("regime_weights", self.config.regime_weights)),
                "dd_pen_weight": float(genome.get("dd_pen_weight", self.config.dd_pen_weight)),
                "risk_pen_weight": float(genome.get("risk_pen_weight", self.config.risk_pen_weight)),
                "tail_pen_weight": float(genome.get("tail_pen_weight", self.config.tail_pen_weight)),
                "mistake_pen_weight": float(genome.get("mistake_pen_weight", self.config.mistake_pen_weight)),
                "no_trade_penalty_weight": float(genome.get("no_trade_penalty_weight", self.config.no_trade_penalty_weight)),
                "win_bonus_weight": float(genome.get("win_bonus_weight", self.config.win_bonus_weight)),
                "consistency_bonus_weight": float(genome.get("consistency_bonus_weight", self.config.consistency_bonus_weight)),
                "sharpe_bonus_weight": float(genome.get("sharpe_bonus_weight", self.config.sharpe_bonus_weight)),
                "trade_frequency_bonus": float(genome.get("trade_frequency_bonus", self.config.trade_frequency_bonus)),
                "volatility_adjustment": float(genome.get("volatility_adjustment", self.config.volatility_adjustment)),
                "regime_bonus_weight": float(genome.get("regime_bonus_weight", self.config.regime_bonus_weight)),
                "momentum_bonus_weight": float(genome.get("momentum_bonus_weight", self.config.momentum_bonus_weight)),
                "confidence_decay": float(genome.get("confidence_decay", self.config.confidence_decay)),
                "performance_smoothing": float(genome.get("performance_smoothing", self.config.performance_smoothing))
            }
            
            # Update config with genome values
            for key, value in self.genome.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
                    
            # Validate and normalize regime weights
            self.regime_weights = np.array(self.genome["regime_weights"], dtype=np.float32)
            if self.regime_weights.sum() > 0:
                self.regime_weights = self.regime_weights / self.regime_weights.sum()
            else:
                self.regime_weights = np.array([0.3, 0.4, 0.3], dtype=np.float32)
                
        else:
            # Default genome
            self.genome = {
                "initial_balance": self.config.initial_balance,
                "history_size": self.config.history_size,
                "min_trade_bonus": self.config.min_trade_bonus,
                "regime_weights": self.config.regime_weights,
                "dd_pen_weight": self.config.dd_pen_weight,
                "risk_pen_weight": self.config.risk_pen_weight,
                "tail_pen_weight": self.config.tail_pen_weight,
                "mistake_pen_weight": self.config.mistake_pen_weight,
                "no_trade_penalty_weight": self.config.no_trade_penalty_weight,
                "win_bonus_weight": self.config.win_bonus_weight,
                "consistency_bonus_weight": self.config.consistency_bonus_weight,
                "sharpe_bonus_weight": self.config.sharpe_bonus_weight,
                "trade_frequency_bonus": self.config.trade_frequency_bonus,
                "volatility_adjustment": self.config.volatility_adjustment,
                "regime_bonus_weight": self.config.regime_bonus_weight,
                "momentum_bonus_weight": self.config.momentum_bonus_weight,
                "confidence_decay": self.config.confidence_decay,
                "performance_smoothing": self.config.performance_smoothing
            }
            
            self.regime_weights = np.array(self.config.regime_weights, dtype=np.float32)

    def _initialize_reward_state(self):
        """Initialize reward calculation state"""
        # Initialize mixin states
        self._initialize_trading_state()
        self._initialize_risk_state()
        self._initialize_state_management()
        
        # Current operational mode
        self.current_mode = RewardMode.TRAINING
        self.mode_start_time = datetime.datetime.now()
        
        # Core reward state
        self._reward_history = deque(maxlen=self.config.history_size)
        self._pnl_history = deque(maxlen=self.config.history_size)
        self._trade_count_history = deque(maxlen=20)
        self._last_reward = 0.0
        self._last_reason = ""
        self._call_count = 0
        
        # Enhanced tracking
        self._reward_components_history = deque(maxlen=50)
        self._performance_analytics = defaultdict(list)
        self._regime_performance = defaultdict(lambda: {'rewards': [], 'trades': [], 'pnl': []})
        
        # Audit trail with enhanced metadata
        self.audit_trail: List[Dict[str, Any]] = []
        self._audit_log_size = self.config.history_size
        
        # Performance metrics
        self._sharpe_ratio = 0.0
        self._consistency_score = 0.0
        self._win_rate = 0.0
        self._avg_reward = 0.0
        self._reward_volatility = 0.0
        self._reward_quality = 0.5
        
        # Adaptive parameters with learning
        self._adaptive_params = {
            'dynamic_penalty_scaling': 1.0,
            'regime_sensitivity': 1.0,
            'activity_threshold': 1.0,
            'risk_tolerance': 1.0,
            'learning_momentum': 0.0,
            'adaptation_confidence': 0.5
        }
        
        # Advanced analytics
        self._regime_transition_rewards = defaultdict(list)
        self._volatility_performance = defaultdict(list)
        self._session_analytics = defaultdict(list)
        self._error_recovery_metrics = []

    def _start_monitoring(self):
        """Start background monitoring for reward system"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_reward_health()
                    self._analyze_reward_effectiveness()
                    self._adapt_parameters()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Reward monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module with SmartInfoBus integration"""
        try:
            # Set initial reward status
            initial_status = {
                "current_mode": self.current_mode.value,
                "reward_quality": self._reward_quality,
                "avg_reward": self._avg_reward,
                "sharpe_ratio": self._sharpe_ratio,
                "win_rate": self._win_rate
            }
            
            self.smart_bus.set(
                'reward_performance',
                initial_status,
                module='RiskAdjustedReward',
                thesis="Initial reward system performance metrics"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Reward system initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process reward calculation with enhanced analytics"""
        start_time = time.time()
        
        try:
            # Extract reward data from SmartInfoBus
            reward_data = await self._extract_reward_data(**inputs)
            
            if not reward_data:
                return await self._handle_no_data_fallback()
            
            # Calculate reward with enhanced components
            reward_result = await self._calculate_enhanced_reward(reward_data)
            
            # Update performance analytics
            analytics_result = await self._update_reward_analytics(reward_result, reward_data)
            
            # Update adaptive parameters
            adaptation_result = await self._update_adaptive_learning(reward_result)
            
            # Combine results
            result = {**reward_result, **analytics_result, **adaptation_result}
            
            # Generate thesis
            thesis = await self._generate_reward_thesis(reward_data, result)
            
            # Update SmartInfoBus
            await self._update_reward_smart_bus(result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return result
            
        except Exception as e:
            return await self._handle_reward_error(e, start_time)

    async def _extract_reward_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract reward data from SmartInfoBus and inputs"""
        try:
            # Get trade data from SmartInfoBus
            trade_data = self.smart_bus.get('trade_data', 'RiskAdjustedReward') or {}
            recent_trades = trade_data.get('recent_trades', [])
            
            # Get risk metrics
            risk_metrics = self.smart_bus.get('risk_metrics', 'RiskAdjustedReward') or {}
            balance = risk_metrics.get('balance', self.config.initial_balance)
            drawdown = risk_metrics.get('current_drawdown', 0.0)
            
            # Get market context
            market_context = self.smart_bus.get('market_context', 'RiskAdjustedReward') or {}
            regime = market_context.get('regime', 'unknown')
            volatility_level = market_context.get('volatility_level', 'medium')
            consensus = market_context.get('consensus', 0.5)
            
            # Get performance data
            performance_data = self.smart_bus.get('performance_data', 'RiskAdjustedReward') or {}
            
            # Get direct inputs
            trades = inputs.get('trades', recent_trades)
            actions = inputs.get('actions')
            raw_reward_inputs = inputs.get('reward_inputs', {})
            
            return {
                'trades': trades,
                'balance': balance,
                'drawdown': drawdown,
                'regime': regime,
                'volatility_level': volatility_level,
                'consensus': consensus,
                'actions': actions,
                'market_context': market_context,
                'risk_metrics': risk_metrics,
                'performance_data': performance_data,
                'raw_inputs': raw_reward_inputs,
                'timestamp': datetime.datetime.now().isoformat(),
                'step_idx': inputs.get('step_idx', self._call_count)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract reward data: {e}")
            return None

    async def _calculate_enhanced_reward(self, reward_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced reward with comprehensive components"""
        try:
            # Extract core data
            trades = reward_data.get('trades', [])
            balance = reward_data.get('balance', self.config.initial_balance)
            drawdown = reward_data.get('drawdown', 0.0)
            consensus = reward_data.get('consensus', 0.5)
            actions = reward_data.get('actions')
            regime = reward_data.get('regime', 'unknown')
            volatility_level = reward_data.get('volatility_level', 'medium')
            
            # Calculate base components
            realised_pnl = sum(t.get("pnl", 0.0) for t in trades)
            base_component = realised_pnl / (self.config.initial_balance + 1e-12)
            
            # Initialize reward with base PnL
            reward = realised_pnl
            
            # Prepare detailed components for audit
            components = {
                "pnl": realised_pnl,
                "base_component": base_component,
                "balance": balance,
                "drawdown": drawdown,
                "consensus": consensus,
                "trades_count": len(trades),
                "timestamp": reward_data.get('timestamp', utcnow()),
                "step_idx": reward_data.get('step_idx', 0),
                "market_regime": regime,
                "volatility_level": volatility_level,
                
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
            
            # 1. Progressive drawdown penalty with regime awareness
            if drawdown > 0.05:
                dd_penalty = drawdown ** 2 * self.config.dd_pen_weight
                
                # Adjust penalty based on market regime
                if regime == 'volatile':
                    dd_penalty *= 0.8  # More tolerant in volatile markets
                elif regime == 'trending':
                    dd_penalty *= 1.2  # Less tolerant in trending markets
                    
                reward -= dd_penalty * self._adaptive_params['dynamic_penalty_scaling']
                components["drawdown_penalty"] = dd_penalty
                
            # 2. Enhanced risk penalty with volatility adjustment
            if actions is not None:
                risk_penalty = min(np.linalg.norm(actions) * self.config.risk_pen_weight, 0.2)
                
                # Adjust based on volatility
                vol_multiplier = {'low': 1.2, 'medium': 1.0, 'high': 0.8, 'extreme': 0.6}
                risk_penalty *= vol_multiplier.get(volatility_level, 1.0)
                
                reward -= risk_penalty
                components["risk_penalty"] = risk_penalty
                
            # 3. Enhanced tail risk penalty with extreme loss detection
            if trades:
                losses = [t.get("pnl", 0) for t in trades if t.get("pnl", 0) < 0]
                if losses:
                    tail_penalty = abs(np.mean(losses)) * self.config.tail_pen_weight * 0.1
                    
                    # Increase penalty for extreme losses
                    extreme_losses = [l for l in losses if l < -100]  # Losses > €100
                    if extreme_losses:
                        tail_penalty *= 1.5
                        
                    reward -= tail_penalty
                    components["tail_penalty"] = tail_penalty
                    
            # 4. Enhanced mistake memory penalty
            mistake_penalty = await self._calculate_mistake_penalty(reward_data)
            if mistake_penalty > 0:
                reward -= mistake_penalty
                components["mistake_penalty"] = mistake_penalty
            
            # ========== ENHANCED BONUSES ==========
            
            if trades:
                # Enhanced win bonus with streak detection
                win_ratio = sum(1 for t in trades if t.get("pnl", 0) > 0) / len(trades)
                win_bonus = win_ratio * self.config.win_bonus_weight
                
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
                no_trade_penalty = self.config.no_trade_penalty_weight * self._adaptive_params['activity_threshold']
                
                # Reduce penalty in high-risk conditions
                if drawdown > 0.1:
                    no_trade_penalty *= 0.3  # Defensive trading is acceptable
                elif volatility_level == 'extreme':
                    no_trade_penalty *= 0.5  # Caution in extreme volatility
                    
                reward -= no_trade_penalty
                components["no_trade_penalty"] = no_trade_penalty
            
            # ========== SOPHISTICATED BONUSES ==========
            
            # Calculate advanced bonuses
            consistency_bonus = await self._calculate_consistency_bonus()
            sharpe_bonus = await self._calculate_enhanced_sharpe_bonus()
            regime_bonus = await self._calculate_regime_bonus(regime, realised_pnl)
            volatility_adjustment = await self._calculate_volatility_adjustment(volatility_level, realised_pnl)
            
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
            components["method"] = "enhanced_async_calculation"
            
            # Update state
            await self._update_reward_state(trades, realised_pnl, final_reward)
            
            # Enhanced logging with circuit breaker awareness
            if (self._call_count % 10 == 1 or abs(final_reward) > 0.2 or 
                self.circuit_breaker['state'] == 'OPEN'):
                self.logger.info(
                    format_operator_message(
                        "🎯", "REWARD_CALCULATED",
                        reward=f"{final_reward:.4f}",
                        pnl=f"€{realised_pnl:.2f}",
                        trades=len(trades),
                        drawdown=f"{drawdown:.1%}",
                        regime=regime,
                        volatility=volatility_level,
                        context="reward_calculation"
                    )
                )
            
            # Record audit trail
            await self._record_audit(components)
            
            return {
                'shaped_reward': final_reward,
                'reward_components': components,
                'calculation_method': 'enhanced_async'
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced reward calculation failed: {e}")
            raise

    async def _calculate_mistake_penalty(self, reward_data: Dict[str, Any]) -> float:
        """Calculate penalty from mistake memory with enhanced integration"""
        try:
            # Try to get mistake data from SmartInfoBus
            mistake_data = self.smart_bus.get('mistake_memory', 'RiskAdjustedReward')
            
            if mistake_data:
                mistake_score = mistake_data.get('current_score', 0.0)
                penalty = mistake_score * self.config.mistake_pen_weight
                
                # Apply adaptive scaling
                penalty *= self._adaptive_params.get('dynamic_penalty_scaling', 1.0)
                
                return penalty
            
            # Fallback to environment
            if self.env and hasattr(self.env, "mistake_memory"):
                mm_score = float(self.env.mistake_memory.get_observation_components()[0])
                return mm_score * self.config.mistake_pen_weight
                
        except Exception as e:
            self.logger.warning(f"Mistake penalty calculation failed: {e}")
            
        return 0.0

    async def _calculate_enhanced_sharpe_bonus(self) -> float:
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
        
        # Conservative normalization with adaptive sensitivity
        sensitivity = self._adaptive_params.get('regime_sensitivity', 1.0)
        normalized_sharpe = np.tanh(sharpe / (6.0 / sensitivity))
        bonus = float(np.clip(normalized_sharpe * self.config.sharpe_bonus_weight, -0.5, 0.5))
        
        # Update performance metric
        self._sharpe_ratio = sharpe
        
        return bonus

    async def _calculate_consistency_bonus(self) -> float:
        """Enhanced consistency bonus with streak detection and momentum"""
        if len(self._pnl_history) < 3:
            return 0.0
            
        recent_pnls = list(self._pnl_history)[-10:]
        positive_ratio = sum(1 for p in recent_pnls if p > 0) / len(recent_pnls)
        
        # Base consistency score with momentum consideration
        consistency_score = positive_ratio ** 2
        
        # Add momentum factor
        if len(recent_pnls) >= 5:
            early_half = recent_pnls[:len(recent_pnls)//2]
            late_half = recent_pnls[len(recent_pnls)//2:]
            
            early_ratio = sum(1 for p in early_half if p > 0) / len(early_half)
            late_ratio = sum(1 for p in late_half if p > 0) / len(late_half)
            
            momentum = late_ratio - early_ratio
            consistency_score *= (1.0 + momentum * 0.2)  # Bonus for improving consistency
        
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
        
        return float(consistency_score * self.config.consistency_bonus_weight)

    async def _calculate_regime_bonus(self, regime: str, pnl: float) -> float:
        """Calculate bonus based on regime-appropriate trading with learning"""
        self._last_regime = regime  # Store for Sharpe calculation
        
        # Track regime performance
        self._regime_performance[regime]['pnl'].append(pnl)
        
        # Regime-specific bonuses with adaptive learning
        regime_bonus = 0.0
        sensitivity = self._adaptive_params.get('regime_sensitivity', 1.0)
        
        if regime == 'trending' and pnl > 0:
            # Bonus for profitable trend following
            regime_bonus = min(pnl / 100.0, 0.2) * self.config.regime_bonus_weight * sensitivity
        elif regime == 'ranging' and abs(pnl) < 20:
            # Bonus for controlled trading in ranging markets
            regime_bonus = 0.1 * self.config.regime_bonus_weight * sensitivity
        elif regime == 'volatile':
            # Penalty for large losses in volatile markets, bonus for small profits
            if pnl < -50:
                regime_bonus = -0.15 * self.config.regime_bonus_weight * sensitivity
            elif 0 < pnl < 30:
                regime_bonus = 0.1 * self.config.regime_bonus_weight * sensitivity
        
        # Store regime transition data for analysis
        if len(self._regime_performance) > 1:
            self._regime_transition_rewards[regime].append(pnl)
        
        return regime_bonus

    async def _calculate_volatility_adjustment(self, volatility_level: str, pnl: float) -> float:
        """Calculate volatility-adjusted reward component with adaptive learning"""
        
        # Track volatility performance
        self._volatility_performance[volatility_level].append(pnl)
        
        # Volatility-based adjustments with adaptive scaling
        vol_multipliers = {
            'low': 1.1,      # Slight bonus in low vol (easier to trade)
            'medium': 1.0,   # Neutral
            'high': 0.9,     # Slight penalty in high vol
            'extreme': 0.8   # Higher penalty in extreme vol
        }
        
        base_adjustment = (vol_multipliers.get(volatility_level, 1.0) - 1.0) * abs(pnl) * 0.1
        
        # Additional bonus for profitable trading in difficult conditions
        if volatility_level in ['high', 'extreme'] and pnl > 0:
            base_adjustment += pnl * 0.05  # Extra bonus for high-vol profits
        
        # Apply adaptive volatility adjustment
        adaptive_factor = self._adaptive_params.get('risk_tolerance', 1.0)
        
        return base_adjustment * self.config.volatility_adjustment * adaptive_factor

    async def _update_reward_state(self, trades: List[dict], pnl: float, reward: float) -> None:
        """Update reward state with enhanced tracking"""
        # Update histories
        self._pnl_history.append(pnl)
        self._trade_count_history.append(len(trades))
        self._reward_history.append(reward)
        
        # Update trade tracking via mixin
        if trades:
            for trade in trades:
                await self._update_trading_metrics_async(trade)
            
        # Update session analytics
        session = datetime.datetime.now().strftime("%Y-%m-%d")
        self._session_analytics[session].append({
            'timestamp': time.time(),
            'pnl': pnl,
            'reward': reward,
            'trades': len(trades)
        })
        
        # Update last reward
        self._last_reward = float(reward)
        self._last_reason = "trade" if trades else "no-trade"
        self._call_count += 1

    async def _update_trading_metrics_async(self, trade: dict):
        """Async wrapper for trading metrics update"""
        # Use mixin method but in async context
        self._update_trading_metrics(trade)

    async def _record_audit(self, details: Dict[str, Any]) -> None:
        """Enhanced audit recording with metadata and analytics"""
        details["timestamp"] = utcnow()
        details["call_count"] = self._call_count
        details["mode"] = self.current_mode.value
        details["circuit_breaker_state"] = self.circuit_breaker['state']
        
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
            'trades_count': details.get("trades_count", 0),
            'volatility_level': details.get("volatility_level", "medium")
        })

    async def _update_reward_analytics(self, reward_result: Dict[str, Any], 
                                     reward_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update comprehensive reward analytics"""
        try:
            # Update performance metrics
            await self._update_performance_metrics()
            
            # Analyze component effectiveness
            component_analysis = await self._analyze_component_effectiveness(reward_result)
            
            # Update regime performance tracking
            regime_analysis = await self._update_regime_analysis(reward_data)
            
            return {
                'reward_analytics': {
                    'performance_metrics': {
                        'sharpe_ratio': self._sharpe_ratio,
                        'consistency_score': self._consistency_score,
                        'win_rate': self._win_rate,
                        'avg_reward': self._avg_reward,
                        'reward_volatility': self._reward_volatility,
                        'reward_quality': self._reward_quality
                    },
                    'component_analysis': component_analysis,
                    'regime_analysis': regime_analysis
                }
            }
            
        except Exception as e:
            self.logger.error(f"Reward analytics update failed: {e}")
            return {'reward_analytics': {'error': str(e)}}

    async def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        try:
            # Update win rate
            if self._trades_processed > 0:
                self._win_rate = self._winning_trades / self._trades_processed
            
            # Update average reward and volatility
            if self._reward_history:
                self._avg_reward = np.mean(self._reward_history)
                self._reward_volatility = np.std(self._reward_history)
            
            # Calculate reward quality score
            if len(self._reward_history) >= 10:
                recent_rewards = list(self._reward_history)[-10:]
                positive_ratio = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                stability = 1.0 - (np.std(recent_rewards) / (abs(np.mean(recent_rewards)) + 1e-8))
                self._reward_quality = (positive_ratio + max(0, stability)) / 2
            
        except Exception as e:
            self.logger.warning(f"Performance metrics update failed: {e}")

    async def _analyze_component_effectiveness(self, reward_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze effectiveness of reward components"""
        try:
            components = reward_result.get('reward_components', {})
            
            # Calculate component contributions
            component_magnitudes = {}
            total_magnitude = 0
            
            for key, value in components.items():
                if key.endswith('_penalty') or key.endswith('_bonus') or key == 'volatility_adjustment':
                    magnitude = abs(float(value))
                    component_magnitudes[key] = magnitude
                    total_magnitude += magnitude
            
            # Calculate relative contributions
            component_contributions = {}
            if total_magnitude > 0:
                for key, magnitude in component_magnitudes.items():
                    component_contributions[key] = magnitude / total_magnitude
            
            return {
                'component_magnitudes': component_magnitudes,
                'component_contributions': component_contributions,
                'total_component_magnitude': total_magnitude
            }
            
        except Exception as e:
            self.logger.warning(f"Component effectiveness analysis failed: {e}")
            return {}

    async def _update_regime_analysis(self, reward_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update regime-specific performance analysis"""
        try:
            regime = reward_data.get('regime', 'unknown')
            
            # Calculate regime performance statistics
            regime_stats = {}
            for reg, data in self._regime_performance.items():
                if data['rewards']:
                    regime_stats[reg] = {
                        'avg_reward': np.mean(data['rewards'][-20:]),
                        'reward_count': len(data['rewards']),
                        'win_rate': sum(1 for r in data['rewards'][-20:] if r > 0) / min(len(data['rewards']), 20)
                    }
            
            return {
                'current_regime': regime,
                'regime_stats': regime_stats
            }
            
        except Exception as e:
            self.logger.warning(f"Regime analysis update failed: {e}")
            return {}

    async def _update_adaptive_learning(self, reward_result: Dict[str, Any]) -> Dict[str, Any]:
        """Update adaptive learning parameters"""
        try:
            reward = reward_result.get('shaped_reward', 0.0)
            
            # Adapt penalty scaling based on recent performance
            if len(self._reward_history) >= 20:
                recent_rewards = list(self._reward_history)[-20:]
                avg_reward = np.mean(recent_rewards)
                
                learning_rate = self.config.adaptive_learning_rate
                
                if avg_reward < -0.5:  # Poor performance
                    self._adaptive_params['dynamic_penalty_scaling'] = min(
                        1.5, 
                        self._adaptive_params['dynamic_penalty_scaling'] * (1 + learning_rate)
                    )
                elif avg_reward > 0.5:  # Good performance
                    self._adaptive_params['dynamic_penalty_scaling'] = max(
                        0.5, 
                        self._adaptive_params['dynamic_penalty_scaling'] * (1 - learning_rate * 0.5)
                    )
            
            # Adapt activity threshold based on trading frequency
            if len(self._trade_count_history) >= 10:
                avg_activity = np.mean(self._trade_count_history)
                
                if avg_activity < 0.3:  # Low activity
                    self._adaptive_params['activity_threshold'] = min(
                        2.0, 
                        self._adaptive_params['activity_threshold'] * 1.05
                    )
                elif avg_activity > 3.0:  # High activity
                    self._adaptive_params['activity_threshold'] = max(
                        0.5, 
                        self._adaptive_params['activity_threshold'] * 0.98
                    )
            
            # Update adaptation confidence
            if self._reward_quality > 0.7:
                self._adaptive_params['adaptation_confidence'] = min(
                    1.0, 
                    self._adaptive_params['adaptation_confidence'] * 1.01
                )
            elif self._reward_quality < 0.3:
                self._adaptive_params['adaptation_confidence'] = max(
                    0.1, 
                    self._adaptive_params['adaptation_confidence'] * 0.99
                )
            
            return {
                'adaptive_learning': {
                    'adaptive_params': self._adaptive_params.copy(),
                    'reward_quality': self._reward_quality,
                    'learning_effectiveness': self._adaptive_params['adaptation_confidence']
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Adaptive learning update failed: {e}")
            return {'adaptive_learning': {'error': str(e)}}

    async def _generate_reward_thesis(self, reward_data: Dict[str, Any], 
                                    result: Dict[str, Any]) -> str:
        """Generate comprehensive reward thesis"""
        try:
            # Core metrics
            reward = result.get('shaped_reward', 0.0)
            components = result.get('reward_components', {})
            pnl = components.get('pnl', 0.0)
            trades_count = components.get('trades_count', 0)
            regime = reward_data.get('regime', 'unknown')
            volatility = reward_data.get('volatility_level', 'medium')
            
            thesis_parts = [
                f"Reward System: Generated {reward:.4f} reward from €{pnl:.2f} PnL across {trades_count} trades",
                f"Market Context: {regime.upper()} regime with {volatility.upper()} volatility"
            ]
            
            # Performance analysis
            if self._reward_quality > 0.7:
                thesis_parts.append(f"Performance: HIGH quality ({self._reward_quality:.2f}) with {self._win_rate:.1%} win rate")
            elif self._reward_quality < 0.3:
                thesis_parts.append(f"Performance: LOW quality ({self._reward_quality:.2f}) requiring attention")
            else:
                thesis_parts.append(f"Performance: MODERATE quality ({self._reward_quality:.2f})")
            
            # Component analysis
            major_components = []
            for key, value in components.items():
                if (key.endswith('_penalty') or key.endswith('_bonus')) and abs(value) > 0.01:
                    major_components.append(f"{key}: {value:.3f}")
            
            if major_components:
                thesis_parts.append(f"Major components: {', '.join(major_components[:3])}")
            
            # Adaptive learning status
            adaptation_confidence = self._adaptive_params.get('adaptation_confidence', 0.5)
            if adaptation_confidence > 0.8:
                thesis_parts.append("Adaptation: HIGH confidence in parameter adjustments")
            elif adaptation_confidence < 0.3:
                thesis_parts.append("Adaptation: LOW confidence - system learning")
            
            # Health status
            if self.circuit_breaker['state'] == 'OPEN':
                thesis_parts.append("ALERT: Circuit breaker OPEN - degraded mode")
            elif self._health_status == 'warning':
                thesis_parts.append("WARNING: System health requires monitoring")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Reward thesis generation failed: {str(e)} - Core reward calculation functional"

    async def _update_reward_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with reward results"""
        try:
            # Shaped reward
            reward_data = {
                'reward': result.get('shaped_reward', 0.0),
                'components': result.get('reward_components', {}),
                'calculation_method': result.get('calculation_method', 'enhanced_async'),
                'timestamp': utcnow()
            }
            
            self.smart_bus.set(
                'shaped_reward',
                reward_data,
                module='RiskAdjustedReward',
                thesis=thesis
            )
            
            # Reward components
            components_data = result.get('reward_components', {})
            self.smart_bus.set(
                'reward_components',
                components_data,
                module='RiskAdjustedReward',
                thesis="Detailed breakdown of reward calculation components"
            )
            
            # Reward analytics
            analytics_data = result.get('reward_analytics', {})
            self.smart_bus.set(
                'reward_analytics',
                analytics_data,
                module='RiskAdjustedReward',
                thesis="Comprehensive reward system analytics and component effectiveness"
            )
            
            # Reward performance
            performance_data = {
                'reward_quality': self._reward_quality,
                'sharpe_ratio': self._sharpe_ratio,
                'consistency_score': self._consistency_score,
                'win_rate': self._win_rate,
                'avg_reward': self._avg_reward,
                'reward_volatility': self._reward_volatility,
                'adaptive_params': self._adaptive_params.copy(),
                'health_status': self._health_status,
                'circuit_breaker_state': self.circuit_breaker['state']
            }
            
            self.smart_bus.set(
                'reward_performance',
                performance_data,
                module='RiskAdjustedReward',
                thesis="Real-time reward system performance metrics and health status"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no reward data is available"""
        self.logger.warning("No reward data available - using fallback calculation")
        
        # Generate minimal reward with penalty for no data
        fallback_reward = -0.1  # Small penalty for missing data
        
        return {
            'shaped_reward': fallback_reward,
            'reward_components': {
                'fallback_penalty': -0.1,
                'reason': 'no_reward_data'
            },
            'fallback_reason': 'no_reward_data'
        }

    async def _handle_reward_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle reward calculation errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self._health_status = 'warning'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "RiskAdjustedReward")
        explanation = self.english_explainer.explain_error(
            "RiskAdjustedReward", str(error), "reward calculation"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "REWARD_CALCULATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                circuit_breaker_state=self.circuit_breaker['state'],
                context="reward_error"
            )
        )
        
        # Record error for analysis
        self._error_recovery_metrics.append({
            'timestamp': time.time(),
            'error_type': type(error).__name__,
            'processing_time': processing_time,
            'circuit_breaker_state': self.circuit_breaker['state']
        })
        
        # Record failure
        self._record_failure(error)
        
        return self._create_error_fallback_response(f"error: {str(error)}")

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        # Provide conservative negative reward during errors
        error_reward = -0.5 if self.circuit_breaker['state'] == 'OPEN' else -0.2
        
        return {
            'shaped_reward': error_reward,
            'reward_components': {
                'error_penalty': error_reward,
                'reason': reason
            },
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    def _update_reward_health(self):
        """Update reward system health metrics"""
        try:
            # Check reward quality
            if self._reward_quality < self.config.min_reward_quality:
                self._health_status = 'warning'
            else:
                self._health_status = 'healthy'
            
            # Check circuit breaker
            if self.circuit_breaker['state'] == 'OPEN':
                self._health_status = 'warning'
            
            # Check for excessive volatility
            if self._reward_volatility > 2.0:
                self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Reward health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_reward_effectiveness(self):
        """Analyze reward calculation effectiveness"""
        try:
            if len(self._reward_history) >= 10:
                recent_rewards = list(self._reward_history)[-10:]
                
                # Analyze reward distribution
                positive_ratio = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
                
                if positive_ratio > 0.8:
                    self.logger.info(
                        format_operator_message(
                            "🎯", "HIGH_REWARD_EFFECTIVENESS",
                            positive_ratio=f"{positive_ratio:.2f}",
                            avg_reward=f"{np.mean(recent_rewards):.4f}",
                            context="reward_analysis"
                        )
                    )
                elif positive_ratio < 0.2:
                    self.logger.warning(
                        format_operator_message(
                            "⚠️", "LOW_REWARD_EFFECTIVENESS",
                            positive_ratio=f"{positive_ratio:.2f}",
                            avg_reward=f"{np.mean(recent_rewards):.4f}",
                            context="reward_analysis"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Reward effectiveness analysis failed: {e}")

    def _adapt_parameters(self):
        """Continuous parameter adaptation based on performance"""
        try:
            # Adapt based on regime performance
            if len(self._regime_performance) >= 2:
                regime_rewards = {}
                for regime, data in self._regime_performance.items():
                    if data['rewards']:
                        regime_rewards[regime] = np.mean(data['rewards'][-10:])
                
                if regime_rewards:
                    reward_variance = np.var(list(regime_rewards.values()))
                    
                    # Increase regime sensitivity if regimes show different performance
                    if reward_variance > 0.1:
                        self._adaptive_params['regime_sensitivity'] = min(
                            1.5, 
                            self._adaptive_params['regime_sensitivity'] * 1.001
                        )
                    else:
                        self._adaptive_params['regime_sensitivity'] = max(
                            0.7, 
                            self._adaptive_params['regime_sensitivity'] * 0.9999
                        )
            
        except Exception as e:
            self.logger.warning(f"Parameter adaptation failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'RiskAdjustedReward', 'reward_calculation', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'RiskAdjustedReward', 'reward_calculation', 0, False
        )

    # ================== EVOLUTIONARY METHODS ==================

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with validation"""
        for key, value in genome.items():
            if hasattr(self.config, key):
                if key == "regime_weights":
                    weights = np.array(value, dtype=np.float32)
                    if weights.sum() > 0:
                        self.regime_weights = weights / weights.sum()
                    else:
                        self.regime_weights = np.array([0.3, 0.4, 0.3], dtype=np.float32)
                    setattr(self.config, key, self.regime_weights.tolist())
                else:
                    setattr(self.config, key, value)
        
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
        
        # Mutate weights with adaptive rates based on performance
        performance_factor = max(0.5, min(2.0, 1.0 + (self._avg_reward - 0.0) * 2.0))
        
        weight_params = [
            ("dd_pen_weight", 0.2, 5.0),
            ("risk_pen_weight", 0.1, 2.0),
            ("tail_pen_weight", 0.1, 2.0),
            ("win_bonus_weight", 0.1, 3.0),
            ("consistency_bonus_weight", 0.1, 2.0),
            ("sharpe_bonus_weight", 0.1, 2.0)
        ]
        
        for weight_name, std, max_val in weight_params:
            if np.random.rand() < mutation_rate * performance_factor:
                old_val = g[weight_name]
                g[weight_name] = float(np.clip(
                    old_val + np.random.normal(0, std), 
                    0.0, 
                    max_val
                ))
                mutations.append(f"{weight_name}: {old_val:.3f} → {g[weight_name]:.3f}")
        
        if mutations:
            self.logger.info(
                format_operator_message(
                    "🧬", "REWARD_SYSTEM_MUTATION",
                    changes=", ".join(mutations),
                    performance_factor=f"{performance_factor:.2f}",
                    context="evolution"
                )
            )
            
        self.set_genome(g)

    def crossover(self, other) -> "RiskAdjustedReward":
        """Enhanced crossover with performance-based selection"""
        if not isinstance(other, RiskAdjustedReward):
            self.logger.warning("Crossover with incompatible type")
            return self
        
        # Performance-based crossover
        self_performance = self._avg_reward
        other_performance = getattr(other, '_avg_reward', 0.0)
        
        # Favor higher performance parent
        if self_performance > other_performance:
            bias = 0.7  # Favor self
        else:
            bias = 0.3  # Favor other
        
        new_g = {}
        other_genome = getattr(other, 'genome', {})
        for key in self.genome:
            if np.random.rand() < bias:
                new_g[key] = self.genome[key]
            else:
                new_g[key] = other_genome.get(key, self.genome[key])
        
        child = RiskAdjustedReward(**{'genome': new_g, 'env': self.env})
        
        return child

    # ================== LEGACY COMPATIBILITY METHODS ==================



    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        # Reset mixin states
        # Note: Mixin reset methods will be implemented as needed
        
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
        self._reward_quality = 0.5
        
        # Reset adaptive parameters
        self._adaptive_params = {
            'dynamic_penalty_scaling': 1.0,
            'regime_sensitivity': 1.0,
            'activity_threshold': 1.0,
            'risk_tolerance': 1.0,
            'learning_momentum': 0.0,
            'adaptation_confidence': 0.5
        }
        
        # Reset circuit breaker
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['state'] = 'CLOSED'
        self._health_status = 'healthy'

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with reward metrics"""
        try:
            if not self._reward_history:
                return np.zeros(10, np.float32)
                
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
            reward_quality = self._reward_quality
            adaptation_confidence = self._adaptive_params.get('adaptation_confidence', 0.5)
            
            return np.array([
                self._last_reward,      # Last reward
                recent_mean,            # Recent average
                recent_std,             # Recent volatility
                win_rate,               # Win rate
                activity,               # Activity level
                reward_trend,           # Trend
                consistency,            # Consistency score
                sharpe,                 # Normalized Sharpe
                reward_quality,         # Reward quality
                adaptation_confidence   # Adaptation confidence
            ], np.float32)
            
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(10, dtype=np.float32)

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
            "dd_pen_weight": self.config.dd_pen_weight,
            "risk_pen_weight": self.config.risk_pen_weight,
            "tail_pen_weight": self.config.tail_pen_weight,
            "mistake_pen_weight": self.config.mistake_pen_weight,
            "no_trade_penalty_weight": self.config.no_trade_penalty_weight,
            "win_bonus_weight": self.config.win_bonus_weight,
            "consistency_bonus_weight": self.config.consistency_bonus_weight,
            "sharpe_bonus_weight": self.config.sharpe_bonus_weight,
            "trade_frequency_bonus": self.config.trade_frequency_bonus,
            "volatility_adjustment": self.config.volatility_adjustment,
            "regime_bonus_weight": self.config.regime_bonus_weight,
            "momentum_bonus_weight": self.config.momentum_bonus_weight,
        }

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'current_mode': self.current_mode.value,
            'reward_history': list(self._reward_history),
            'pnl_history': list(self._pnl_history),
            'trade_count_history': list(self._trade_count_history),
            'last_reward': self._last_reward,
            'last_reason': self._last_reason,
            'call_count': self._call_count,
            'genome': self.genome.copy(),
            'adaptive_params': self._adaptive_params.copy(),
            'performance_metrics': {
                'sharpe_ratio': self._sharpe_ratio,
                'consistency_score': self._consistency_score,
                'win_rate': self._win_rate,
                'avg_reward': self._avg_reward,
                'reward_volatility': self._reward_volatility,
                'reward_quality': self._reward_quality
            },
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self._health_status,
            'regime_performance': {
                regime: {
                    'rewards': data['rewards'][-20:],
                    'pnl': data['pnl'][-20:]
                }
                for regime, data in self._regime_performance.items()
            }
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'current_mode' in state:
            try:
                self.current_mode = RewardMode(state['current_mode'])
            except ValueError:
                self.current_mode = RewardMode.TRAINING
        
        # Restore core state
        self._reward_history = deque(
            state.get("reward_history", []), 
            maxlen=self.config.history_size
        )
        self._pnl_history = deque(
            state.get("pnl_history", []), 
            maxlen=self.config.history_size
        )
        self._trade_count_history = deque(
            state.get("trade_count_history", []), 
            maxlen=20
        )
        self._last_reward = state.get("last_reward", 0.0)
        self._last_reason = state.get("last_reason", "")
        self._call_count = state.get("call_count", 0)
        
        # Restore genome and parameters
        if 'genome' in state:
            self.set_genome(state['genome'])
        
        if 'adaptive_params' in state:
            self._adaptive_params.update(state['adaptive_params'])
        
        # Restore performance metrics
        performance_metrics = state.get("performance_metrics", {})
        self._sharpe_ratio = performance_metrics.get('sharpe_ratio', 0.0)
        self._consistency_score = performance_metrics.get('consistency_score', 0.0)
        self._win_rate = performance_metrics.get('win_rate', 0.0)
        self._avg_reward = performance_metrics.get('avg_reward', 0.0)
        self._reward_volatility = performance_metrics.get('reward_volatility', 0.0)
        self._reward_quality = performance_metrics.get('reward_quality', 0.5)
        
        # Restore system state
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])
        
        if 'health_status' in state:
            self._health_status = state['health_status']

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'reward_quality': self._reward_quality,
            'avg_reward': self._avg_reward,
            'win_rate': self._win_rate,
            'sharpe_ratio': self._sharpe_ratio,
            'adaptation_confidence': self._adaptive_params.get('adaptation_confidence', 0.5)
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

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
        
        # Quality status
        if self._reward_quality > 0.8:
            quality_status = "🎯 High"
        elif self._reward_quality > 0.6:
            quality_status = "✅ Good"
        elif self._reward_quality > 0.4:
            quality_status = "⚡ Fair"
        else:
            quality_status = "❌ Low"
        
        # Health status
        health_emoji = "✅" if self._health_status == 'healthy' else "⚠️"
        cb_status = "🔴 OPEN" if self.circuit_breaker['state'] == 'OPEN' else "🟢 CLOSED"
        
        return f"""
🎯 ENHANCED RISK-ADJUSTED REWARD SYSTEM v4.0
═══════════════════════════════════════════════════
📊 Performance: {performance_status} ({self._avg_reward:.4f} avg)
🎯 Quality: {quality_status} ({self._reward_quality:.3f})
💎 Consistency: {self._consistency_score:.3f}
📈 Sharpe Ratio: {self._sharpe_ratio:.3f}
💰 Win Rate: {self._win_rate:.1%}

🏥 SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Mode: {self.current_mode.value.upper()}
• Adaptation Confidence: {self._adaptive_params.get('adaptation_confidence', 0.5):.2f}

⚖️ CONFIGURATION
• Regime Weights: [{', '.join(f'{w:.2f}' for w in self.regime_weights)}]
• Drawdown Penalty: {self.config.dd_pen_weight:.2f}
• Win Bonus: {self.config.win_bonus_weight:.2f}
• Consistency Bonus: {self.config.consistency_bonus_weight:.2f}

🔧 ADAPTIVE PARAMETERS
• Penalty Scaling: {self._adaptive_params['dynamic_penalty_scaling']:.2f}
• Regime Sensitivity: {self._adaptive_params['regime_sensitivity']:.2f}
• Activity Threshold: {self._adaptive_params['activity_threshold']:.2f}
• Risk Tolerance: {self._adaptive_params['risk_tolerance']:.2f}

📊 ACTIVITY METRICS
• Total Calls: {self._call_count:,}
• Reward History: {len(self._reward_history)} records
• Audit Trail: {len(self.audit_trail)} entries
• Last Reward: {self._last_reward:.4f} ({self._last_reason})

🧬 EVOLUTION
• Genome Parameters: {len(self.genome)} configured
• Performance Factor: {max(0.5, min(2.0, 1.0 + self._avg_reward * 2.0)):.2f}
        """

# End of enhanced RiskAdjustedReward class