

import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from .reward_config import RewardConfig, RewardMode


class RewardState:

    def __init__(self, config: RewardConfig):

        self.cfg = config


        self.current_mode = RewardMode.TRAINING
        self.mode_start_time = datetime.now()


        self.reward_history: Deque[float] = deque(maxlen=config.history_size)
        self.pnl_history: Deque[float] = deque(maxlen=config.history_size)
        self.trade_count_history: Deque[int] = deque(maxlen=20)
        self.reward_components_history: Deque[Dict] = deque(maxlen=50)


        self.sharpe_ratio = 0.0
        self.consistency_score = 0.0
        self.win_rate = 0.0
        self.avg_reward = 0.0
        self.reward_volatility = 0.0
        self.reward_quality = 0.5


        self.last_reward = 0.0
        self.last_reason = ""
        self.last_regime = "unknown"
        self.call_count = 0


        self.baseline_balance: Optional[float] = None
        self.last_balance_observed: Optional[float] = None
        self.last_risk_metrics: Optional[Dict] = None


        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0.0,
            'state': 'CLOSED',
            'threshold': config.circuit_breaker_threshold
        }


        self.health_status = 'healthy'
        self.last_health_check = time.time()


        self.adaptive_params = {
            'dynamic_penalty_scaling': 1.0,
            'regime_sensitivity': 1.0,
            'activity_threshold': 1.0,
            'risk_tolerance': 1.0,
            'learning_momentum': 0.0,
            'adaptation_confidence': 0.5
        }


        self._regime_maxlen = 100
        self._session_maxlen = 200
        self.regime_performance = defaultdict(lambda: {
            'rewards': deque(maxlen=100),
            'trades': deque(maxlen=100),
            'pnl': deque(maxlen=100)
        })
        self.regime_transition_rewards = defaultdict(lambda: deque(maxlen=50))
        self.volatility_performance = defaultdict(lambda: deque(maxlen=100))
        self.session_analytics = defaultdict(lambda: deque(maxlen=200))


        self.trades_processed = 0
        self.winning_trades = 0


        self.audit_trail: Deque[Dict[str, Any]] = deque(maxlen=config.history_size)
        self.audit_log_size = config.history_size


        self.genome = self._initialize_genome()

    def _initialize_genome(self) -> Dict[str, Any]:

        return {
            'initial_balance': self.cfg.initial_balance,
            'history_size': self.cfg.history_size,
            'min_trade_bonus': self.cfg.min_trade_bonus,
            'regime_weights': self.cfg.regime_weights,
            'dd_pen_weight': self.cfg.dd_pen_weight,
            'risk_pen_weight': self.cfg.risk_pen_weight,
            'tail_pen_weight': self.cfg.tail_pen_weight,
            'mistake_pen_weight': self.cfg.mistake_pen_weight,
            'no_trade_penalty_weight': self.cfg.no_trade_penalty_weight,
            'win_bonus_weight': self.cfg.win_bonus_weight,
            'consistency_bonus_weight': self.cfg.consistency_bonus_weight,
            'sharpe_bonus_weight': self.cfg.sharpe_bonus_weight,
            'trade_frequency_bonus': self.cfg.trade_frequency_bonus,
            'volatility_adjustment': self.cfg.volatility_adjustment,
            'regime_bonus_weight': self.cfg.regime_bonus_weight,
            'momentum_bonus_weight': self.cfg.momentum_bonus_weight,
            'confidence_decay': self.cfg.confidence_decay,
            'performance_smoothing': self.cfg.performance_smoothing
        }

    def record_calculation(
        self,
        trades: List[Dict],
        pnl: float,
        reward: float
    ) -> None:

        self.pnl_history.append(float(pnl))
        self.trade_count_history.append(len(trades))
        self.reward_history.append(float(reward))

        self.last_reward = float(reward)
        self.last_reason = "trade" if trades else "no-trade"
        self.call_count += 1


        if trades:
            self.trades_processed += len(trades)
            self.winning_trades += sum(1 for t in trades if t.get('pnl', 0) > 0)


        session = datetime.now().strftime('%Y-%m-%d')
        self.session_analytics[session].append({
            'timestamp': time.time(),
            'pnl': float(pnl),
            'reward': float(reward),
            'trades': len(trades)
        })

    def record_regime_performance(self, regime: str, pnl: float) -> None:

        self.regime_performance[regime]['pnl'].append(float(pnl))
        self.regime_performance[regime]['rewards'].append(float(self.last_reward))

    def record_volatility_performance(self, level: str, pnl: float) -> None:

        self.volatility_performance[level].append(float(pnl))

    def update_performance_metrics(self) -> None:


        if self.trades_processed > 0:
            self.win_rate = float(self.winning_trades / self.trades_processed)


        if self.reward_history:
            rewards = np.array(list(self.reward_history), dtype=np.float32)
            self.avg_reward = float(rewards.mean())
            self.reward_volatility = float(rewards.std())


        if len(self.reward_history) >= 10:
            recent = list(self.reward_history)[-10:]

            positive_ratio = float(sum(1 for r in recent if r > 0) / len(recent))
            stability_np = 1.0 - (np.std(recent) / (abs(np.mean(recent)) + 1e-8))
            stability = float(stability_np)
            self.reward_quality = float((positive_ratio + max(0.0, stability)) / 2.0)

    def update_health_status(self) -> None:

        if self.reward_quality < self.cfg.min_reward_quality:
            self.health_status = 'warning'
        elif self.circuit_breaker['state'] == 'OPEN':
            self.health_status = 'warning'
        elif self.reward_volatility > 2.0:
            self.health_status = 'warning'
        else:
            self.health_status = 'healthy'

        self.last_health_check = time.time()

    def record_success(self) -> None:


        if self.circuit_breaker['state'] in ('OPEN', 'HALF_OPEN'):
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def record_failure(self) -> None:

        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()

        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self.health_status = 'warning'

    def calculate_base_confidence(self) -> float:

        base = 0.8


        base += (self.reward_quality - 0.5) * 0.3


        if abs(self.sharpe_ratio) < 2.0:
            base += 0.1
        elif abs(self.sharpe_ratio) > 5.0:
            base -= 0.2


        if len(self.reward_history) >= 10:
            recent = list(self.reward_history)[-10:]
            stability_np = 1.0 - (np.std(recent) / (abs(np.mean(recent)) + 0.1))
            stability = float(stability_np)
            base += stability * 0.1

        return float(base)

    def get_performance_metrics(self) -> Dict[str, Any]:

        return {
            'reward_quality': float(self.reward_quality),
            'sharpe_ratio': float(self.sharpe_ratio),
            'consistency_score': float(self.consistency_score),
            'win_rate': float(self.win_rate),
            'avg_reward': float(self.avg_reward),
            'reward_volatility': float(self.reward_volatility),
            'adaptive_params': self.adaptive_params.copy(),
            'health_status': self.health_status,
            'circuit_breaker_state': self.circuit_breaker['state'],
            'last_balance_observed': self.last_balance_observed,
            'baseline_balance': self.baseline_balance
        }

    def get_state_summary(self) -> Dict[str, Any]:

        reward_trend = 'neutral'
        if len(self.reward_history) >= 5:
            recent = list(self.reward_history)[-5:]
            slope = float(np.polyfit(range(len(recent)), recent, 1)[0])
            if slope > 0.05:
                reward_trend = 'improving'
            elif slope < -0.05:
                reward_trend = 'declining'

        return {
            'last_reward': float(self.last_reward),
            'reward_trend': reward_trend,
            'sharpe_ratio': float(self.sharpe_ratio),
            'reward_quality': float(self.reward_quality)
        }

    def get_observation_components(self) -> np.ndarray:

        try:
            if not self.reward_history:
                return np.zeros(10, dtype=np.float32)

            rewards = np.array(list(self.reward_history), dtype=np.float32)


            last_reward = float(self.last_reward)
            recent_mean = float(rewards[-10:].mean() if len(rewards) >= 10 else rewards.mean())
            recent_std = float(rewards[-10:].std() if len(rewards) >= 10 else 0.1)

            activity = float(np.mean(list(self.trade_count_history)) if self.trade_count_history else 0.0)
            win_rate = float(self.win_rate)


            trend = 0.0
            if len(rewards) >= 5:
                trend = float(np.polyfit(range(5), rewards[-5:], 1)[0])

            consistency = float(self.consistency_score)
            sharpe_norm = float(np.tanh(self.sharpe_ratio / 3.0))
            quality = float(self.reward_quality)
            confidence = float(self.adaptive_params.get('adaptation_confidence', 0.5))

            return np.array([
                last_reward, recent_mean, recent_std, win_rate, activity,
                trend, consistency, sharpe_norm, quality, confidence
            ], dtype=np.float32)

        except Exception:
            return np.zeros(10, dtype=np.float32)

    def apply_genome(self, genome: Dict[str, Any]) -> None:

        self.genome = genome.copy()
        self.cfg.apply_genome(genome)

    def mutate_genome(self, mutation_rate: float = 0.2) -> None:
        pass


    def get_genome(self) -> Dict[str, Any]:

        return self.genome.copy()

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:

        if not self.audit_trail:
            return []
        trail: List[Dict[str, Any]] = list(self.audit_trail)
        return trail[-n:] if n > 0 else trail

    def get_health_status(self) -> Dict[str, Any]:

        return {
            'status': self.health_status,
            'last_check': self.last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'reward_quality': float(self.reward_quality),
            'avg_reward': float(self.avg_reward),
            'win_rate': float(self.win_rate),
            'sharpe_ratio': float(self.sharpe_ratio),
            'adaptation_confidence': float(self.adaptive_params.get('adaptation_confidence', 0.5)),
            'baseline_balance': self.baseline_balance,
            'last_balance_observed': self.last_balance_observed
        }

    def reset(self) -> None:

        self.reward_history.clear()
        self.pnl_history.clear()
        self.trade_count_history.clear()
        self.reward_components_history.clear()
        self.regime_performance.clear()
        self.volatility_performance.clear()
        self.session_analytics.clear()
        self.audit_trail.clear()

        self.last_reward = 0.0
        self.last_reason = ""
        self.call_count = 0

        self.sharpe_ratio = 0.0
        self.consistency_score = 0.0
        self.win_rate = 0.0
        self.avg_reward = 0.0
        self.reward_volatility = 0.0
        self.reward_quality = 0.5

        self.trades_processed = 0
        self.winning_trades = 0

        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['state'] = 'CLOSED'
        self.health_status = 'healthy'

        self.baseline_balance = None
        self.last_balance_observed = None

        self.adaptive_params = {
            'dynamic_penalty_scaling': 1.0,
            'regime_sensitivity': 1.0,
            'activity_threshold': 1.0,
            'risk_tolerance': 1.0,
            'learning_momentum': 0.0,
            'adaptation_confidence': 0.5
        }

    def get_state(self) -> Dict[str, Any]:

        return {
            'current_mode': self.current_mode.value,
            'reward_history': list(self.reward_history),
            'pnl_history': list(self.pnl_history),
            'trade_count_history': list(self.trade_count_history),
            'last_reward': float(self.last_reward),
            'last_reason': self.last_reason,
            'call_count': int(self.call_count),
            'genome': self.genome.copy(),
            'adaptive_params': self.adaptive_params.copy(),
            'performance_metrics': self.get_performance_metrics(),
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self.health_status,
            'baseline_balance': self.baseline_balance,
            'last_balance_observed': self.last_balance_observed
        }

    def set_state(self, state: Dict[str, Any]) -> None:

        if 'current_mode' in state:
            try:
                self.current_mode = RewardMode(state['current_mode'])
            except ValueError:
                self.current_mode = RewardMode.TRAINING

        for key in ['reward_history', 'pnl_history', 'trade_count_history']:
            if key in state:
                buffer = getattr(self, key)
                buffer.clear()
                buffer.extend(state[key])

        for key in ['last_reward', 'last_reason', 'call_count', 'health_status',
                    'baseline_balance', 'last_balance_observed']:
            if key in state:
                setattr(self, key, state[key])

        if 'genome' in state:
            self.apply_genome(state['genome'])

        if 'adaptive_params' in state:
            self.adaptive_params.update(state['adaptive_params'])

        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])


        perf = state.get('performance_metrics', {})
        for key in ['sharpe_ratio', 'consistency_score', 'win_rate',
                    'avg_reward', 'reward_volatility', 'reward_quality']:
            if key in perf:
                setattr(self, key, perf[key])
