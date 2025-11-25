# modules/reward/components/analytics_engine.py
"""
Analytics Engine for Reward System (Hardened & Typed)
- Strong typing to satisfy Pylance/mypy
- Thread-safe internal mutations (RLock; no lock held across awaits)
- Bounded deques for memory safety
- Defensive reads from shared state; resilient to missing fields
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, DefaultDict, Deque, Tuple, Optional
from collections import defaultdict, deque
import threading
import numpy as np
import time
from datetime import datetime


@dataclass
class _ComponentStats:
    total_magnitude: float = 0.0
    occurrences: int = 0
    avg_magnitude: float = 0.0
    impact_score: float = 0.0  # smoothed importance score


@dataclass
class _RegimeBucket:
    rewards: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    pnls: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    trades: Deque[int] = field(default_factory=lambda: deque(maxlen=100))
    win_rate: float = 0.0
    avg_reward: float = 0.0
    volatility: float = 0.0


class RewardAnalyticsEngine:
    """
    Analytics engine for reward system

    Features:
    - Performance metrics calculation
    - Component effectiveness analysis
    - Regime performance tracking
    - Recommendation generation
    - Trend analysis
    """

    # trend buffers
    TREND_MAXLEN = 50
    # performance analytics max entries per key
    PERF_ANALYTICS_MAXLEN = 100
    # session analytics max entries per day
    SESSION_ANALYTICS_MAXLEN = 500

    def __init__(
        self,
        state: Any,
        logger: Any,
        debug_manager: Any
    ):
        """Initialize analytics engine"""

        self.state = state
        self.logger = logger
        self.debug_manager = debug_manager

        # Internal lock for engine-owned structures
        self._lock: threading.RLock = threading.RLock()

        # Analytics storage (typed) - using deque for bounded memory
        self.performance_analytics: DefaultDict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=self.PERF_ANALYTICS_MAXLEN)
        )
        self.component_effectiveness: DefaultDict[str, _ComponentStats] = defaultdict(_ComponentStats)

        # Regime analytics (typed buckets)
        self.regime_analytics: DefaultDict[str, _RegimeBucket] = defaultdict(_RegimeBucket)

        # Session analytics (per-day raw entries) - bounded to prevent memory growth
        self.session_analytics: DefaultDict[str, Deque[Dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=self.SESSION_ANALYTICS_MAXLEN)
        )

        # Daily aggregates
        self.daily_performance: DefaultDict[str, Dict[str, float]] = defaultdict(
            lambda: {
                'total_reward': 0.0,
                'total_pnl': 0.0,
                'trade_count': 0.0,
                'win_count': 0.0,
                'loss_count': 0.0,
            }
        )

        # Trends (typed, bounded)
        self.reward_trend: Deque[float] = deque(maxlen=self.TREND_MAXLEN)
        self.quality_trend: Deque[float] = deque(maxlen=self.TREND_MAXLEN)
        self.effectiveness_trend: Deque[float] = deque(maxlen=self.TREND_MAXLEN)

        self.analysis_count: int = 0

    # ─────────────────────────────────────────────────────────────
    # Public async API
    # ─────────────────────────────────────────────────────────────
    async def update_analytics(
        self,
        reward_result: Dict[str, Any],
        reward_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update all analytics based on reward calculation"""

        with self._lock:
            self.analysis_count += 1

        try:
            # Update core performance metrics in state (no engine lock required)
            await self._update_performance_metrics()

            # Component effectiveness (purely derived)
            component_analysis = await self._analyze_component_effectiveness(reward_result)

            # Regime analysis (mutates engine buckets)
            regime_analysis = await self._update_regime_analysis(reward_data)

            # Session analytics (mutates engine collections)
            await self._update_session_analytics(reward_result, reward_data)

            # Trends (mutates small bounded deques)
            trends = await self._calculate_trends()

            # Insights (derived)
            insights = await self._generate_insights()

            # Assemble payload (use state’s public API for perf metrics)
            return {
                'reward_analytics': {
                    'performance_metrics': self.state.get_performance_metrics(),
                    'component_analysis': component_analysis,
                    'regime_analysis': regime_analysis,
                    'trends': trends,
                    'insights': insights,
                    'analysis_timestamp': datetime.now().isoformat(),
                }
            }

        except Exception as e:
            self._warn(f"Analytics update failed: {e}", tag="ANALYTICS_ERROR")
            return { 'reward_analytics': self.get_baseline_analytics() }

    # ─────────────────────────────────────────────────────────────
    # Core metric updates
    # ─────────────────────────────────────────────────────────────
    async def _update_performance_metrics(self) -> None:
        """Update core performance metrics on the shared state and track derived series"""

        # 1) Let state compute its canonical metrics
        self.state.update_performance_metrics()

        # 2) Track stability & consistency on the engine side
        rewards = list(getattr(self.state, "reward_history", []))
        if len(rewards) >= 10:
            recent = rewards[-10:]
            mean_recent = float(np.mean(recent))
            std_recent = float(np.std(recent))
            stability = 1.0 - (std_recent / (abs(mean_recent) + 1e-8))
            positive_ratio = float(sum(1 for r in recent if r > 0)) / float(len(recent))

            with self._lock:
                self.performance_analytics['stability'].append(float(stability))
                self.performance_analytics['consistency'].append(float(positive_ratio))

    # ─────────────────────────────────────────────────────────────
    # Component effectiveness
    # ─────────────────────────────────────────────────────────────
    async def _analyze_component_effectiveness(
        self,
        reward_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze effectiveness of each reward component"""

        components = reward_result.get('reward_components', {}) or {}

        # Track component magnitudes (derived this tick)
        component_magnitudes: Dict[str, float] = {}
        total_magnitude = 0.0

        # Signal-bearing keys
        def _is_signal_key(k: str) -> bool:
            return k.endswith('_penalty') or k.endswith('_bonus') or k.endswith('_adjustment')

        # First pass: compute magnitudes
        for key, value in components.items():
            if not _is_signal_key(key):
                continue
            try:
                magnitude = abs(float(value))
            except Exception:
                continue
            component_magnitudes[key] = magnitude
            total_magnitude += magnitude

            # Update rolling stats (under lock)
            with self._lock:
                comp = self.component_effectiveness[key]
                comp.total_magnitude += magnitude
                comp.occurrences += 1
                comp.avg_magnitude = comp.total_magnitude / max(1, comp.occurrences)

        # Second pass: contributions & smoothed impact
        contributions: Dict[str, float] = {}
        if total_magnitude > 0.0:
            for key, magnitude in component_magnitudes.items():
                contrib = float(magnitude / total_magnitude)
                contributions[key] = contrib
                with self._lock:
                    comp = self.component_effectiveness[key]
                    # EWMA-ish smoothing
                    comp.impact_score = contrib * 0.3 + comp.impact_score * 0.7

        # Build top impact list (snapshot under lock)
        with self._lock:
            top_components = sorted(
                self.component_effectiveness.items(),
                key=lambda kv: kv[1].impact_score,
                reverse=True
            )[:5]
            comp_stats_serialized = {
                k: {
                    'total_magnitude': v.total_magnitude,
                    'occurrences': v.occurrences,
                    'avg_magnitude': v.avg_magnitude,
                    'impact_score': v.impact_score,
                }
                for k, v in self.component_effectiveness.items()
            }

        return {
            'component_magnitudes': component_magnitudes,
            'component_contributions': contributions,
            'total_component_magnitude': total_magnitude,
            'top_impact_components': {k: v.impact_score for k, v in top_components},
            'component_statistics': comp_stats_serialized,
        }

    # ─────────────────────────────────────────────────────────────
    # Regime analytics
    # ─────────────────────────────────────────────────────────────
    async def _update_regime_analysis(
        self,
        reward_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update regime-specific performance analysis"""

        regime: str = str(reward_data.get('regime', 'unknown') or 'unknown')
        trades: List[Dict[str, Any]] = list(reward_data.get('trades', []) or [])
        last_reward: float = float(getattr(self.state, "last_reward", 0.0))

        # Update bucket
        with self._lock:
            bucket = self.regime_analytics[regime]
            bucket.rewards.append(last_reward)

            if trades:
                total_pnl = float(sum(float(t.get('pnl', 0.0)) for t in trades))
                bucket.pnls.append(total_pnl)
                bucket.trades.append(int(len(trades)))

            # Update stats if we have enough samples
            if len(bucket.rewards) >= 5:
                bucket.avg_reward = float(np.mean(list(bucket.rewards)))
                bucket.volatility = float(np.std(list(bucket.rewards)))
                if len(bucket.pnls) > 0:
                    positive_pnls = sum(1 for p in bucket.pnls if p > 0)
                    bucket.win_rate = float(positive_pnls) / float(len(bucket.pnls))

            # Snapshot stats for all regimes
            regime_stats: Dict[str, Dict[str, float]] = {}
            for reg, data in self.regime_analytics.items():
                if len(data.rewards) == 0:
                    continue
                regime_stats[reg] = {
                    'avg_reward': float(data.avg_reward),
                    'volatility': float(data.volatility),
                    'win_rate': float(data.win_rate),
                    'sample_count': float(len(data.rewards)),
                    'avg_trades': float(np.mean(list(data.trades))) if len(data.trades) > 0 else 0.0,
                }

        # Best/worst determination (outside lock; we use snapshot dict)
        if regime_stats:
            best_regime = max(regime_stats.items(), key=lambda x: x[1]['avg_reward'])
            worst_regime = min(regime_stats.items(), key=lambda x: x[1]['avg_reward'])
            best_name: Optional[str] = best_regime[0]
            worst_name: Optional[str] = worst_regime[0]
        else:
            best_name = None
            worst_name = None

        return {
            'current_regime': regime,
            'regime_statistics': regime_stats,
            'best_performing_regime': best_name,
            'worst_performing_regime': worst_name,
            'regime_recommendation': self._generate_regime_recommendation(regime_stats),
        }

    # ─────────────────────────────────────────────────────────────
    # Session analytics
    # ─────────────────────────────────────────────────────────────
    async def _update_session_analytics(
        self,
        reward_result: Dict[str, Any],
        reward_data: Dict[str, Any]
    ) -> None:
        """Update session-based analytics"""

        session: str = datetime.now().strftime('%Y-%m-%d')
        components = reward_result.get('reward_components', {}) or {}

        entry = {
            'timestamp': time.time(),
            'reward': float(reward_result.get('shaped_reward', 0.0) or 0.0),
            'pnl': float(components.get('pnl', 0.0) or 0.0),
            'trades': int(len(reward_data.get('trades', []) or [])),
            'regime': str(reward_data.get('regime', 'unknown') or 'unknown'),
            'volatility': str(reward_data.get('volatility_level', 'medium') or 'medium'),
        }

        with self._lock:
            self.session_analytics[session].append(entry)
            daily = self.daily_performance[session]
            daily['total_reward'] += entry['reward']
            daily['total_pnl'] += entry['pnl']
            daily['trade_count'] += entry['trades']
            if entry['pnl'] > 0:
                daily['win_count'] += 1
            elif entry['pnl'] < 0:
                daily['loss_count'] += 1

    # ─────────────────────────────────────────────────────────────
    # Trends & insights
    # ─────────────────────────────────────────────────────────────
    async def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate various performance trends"""

        trends: Dict[str, Any] = {}

        # Reward trend from recent history
        rewards = list(getattr(self.state, "reward_history", []))
        if len(rewards) >= 5:
            recent = rewards[-20:]
            if len(recent) >= 5:
                x = np.arange(len(recent), dtype=np.float64)
                try:
                    slope, intercept = np.polyfit(x, np.asarray(recent, dtype=np.float64), 1)
                except Exception:
                    slope = 0.0
                if slope > 0.01:
                    direction = 'improving'
                elif slope < -0.01:
                    direction = 'declining'
                else:
                    direction = 'stable'
                trends['reward_trend'] = {
                    'direction': direction,
                    'slope': float(slope),
                    'recent_avg': float(np.mean(recent[-5:])),
                    'older_avg': float(np.mean(recent[:5])),
                }

        # Quality trend (bounded deque)
        with self._lock:
            self.quality_trend.append(float(getattr(self.state, "reward_quality", 0.5)))
            qt = list(self.quality_trend)
        if len(qt) >= 10:
            try:
                quality_slope = float(np.polyfit(np.arange(len(qt), dtype=np.float64), np.asarray(qt, dtype=np.float64), 1)[0])
            except Exception:
                quality_slope = 0.0
            trends['quality_trend'] = {
                'direction': 'improving' if quality_slope > 0 else 'declining',
                'current': float(getattr(self.state, "reward_quality", 0.5)),
                'average': float(np.mean(qt)),
            }

        # Effectiveness trend (from component impact scores)
        with self._lock:
            if len(self.component_effectiveness) > 0:
                avg_effectiveness = float(np.mean([c.impact_score for c in self.component_effectiveness.values()]))
                self.effectiveness_trend.append(avg_effectiveness)
                et = list(self.effectiveness_trend)
                if len(et) >= 5:
                    trends['effectiveness_trend'] = {
                        'current': avg_effectiveness,
                        'average': float(np.mean(et)),
                    }

        return trends

    async def _generate_insights(self) -> List[Dict[str, Any]]:
        """Generate actionable insights from analytics"""

        insights: List[Dict[str, Any]] = []

        # Performance insight
        sharpe_ratio = float(getattr(self.state, "sharpe_ratio", 0.0))
        if sharpe_ratio < 0:
            insights.append({
                'type': 'performance',
                'severity': 'high',
                'message': f"Negative Sharpe ratio ({sharpe_ratio:.2f}) indicates poor risk-adjusted returns",
                'recommendation': "Review risk management and position sizing",
            })

        # Component insights (top penalties)
        with self._lock:
            penalties = [(k, v) for k, v in self.component_effectiveness.items() if 'penalty' in k]
        if penalties:
            top_penalty = max(penalties, key=lambda kv: kv[1].impact_score)
            if top_penalty[1].impact_score > 0.2:
                insights.append({
                    'type': 'component',
                    'severity': 'medium',
                    'message': f"{top_penalty[0]} is significantly impacting rewards (impact: {top_penalty[1].impact_score:.2f})",
                    'recommendation': f"Focus on reducing {top_penalty[0].replace('_', ' ')}",
                })

        # Regime insights
        with self._lock:
            buckets = dict(self.regime_analytics)
        if buckets:
            # Consider only regimes with at least one reward
            filtered = {k: v for k, v in buckets.items() if len(v.rewards) > 0}
            if filtered:
                best = max(filtered.items(), key=lambda kv: kv[1].avg_reward)
                if best[1].avg_reward > 0.0:
                    insights.append({
                        'type': 'regime',
                        'severity': 'info',
                        'message': f"Best performance in {best[0]} regime (avg reward: {best[1].avg_reward:.3f})",
                        'recommendation': f"Consider optimizing strategy for {best[0]} conditions",
                    })

        # Consistency insight
        consistency_score = float(getattr(self.state, "consistency_score", 0.0))
        if consistency_score < 0.3:
            insights.append({
                'type': 'consistency',
                'severity': 'high',
                'message': f"Low consistency score ({consistency_score:.2f}) indicates erratic performance",
                'recommendation': "Focus on more consistent trading patterns",
            })

        # Win rate insight
        win_rate = float(getattr(self.state, "win_rate", 0.0))
        if win_rate < 0.4:
            insights.append({
                'type': 'win_rate',
                'severity': 'medium',
                'message': f"Low win rate ({win_rate:.1%}) affecting overall performance",
                'recommendation': "Improve trade selection criteria",
            })

        return insights

    # ─────────────────────────────────────────────────────────────
    # Recommendations
    # ─────────────────────────────────────────────────────────────
    def generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate recommendations based on current analytics"""

        recommendations: List[Dict[str, Any]] = []

        sharpe_ratio = float(getattr(self.state, "sharpe_ratio", 0.0))
        if sharpe_ratio < 0:
            recommendations.append({
                'action': 'reduce_risk_exposure',
                'reason': f'Negative Sharpe ratio: {sharpe_ratio:.2f}',
                'priority': 'high',
            })

        reward_quality = float(getattr(self.state, "reward_quality", 0.5))
        if reward_quality < 0.3:
            recommendations.append({
                'action': 'review_trading_strategy',
                'reason': f'Poor reward quality: {reward_quality:.3f}',
                'priority': 'high',
            })

        reward_volatility = float(getattr(self.state, "reward_volatility", 0.0))
        if reward_volatility > 1.0:
            recommendations.append({
                'action': 'stabilize_reward_variance',
                'reason': f'High reward volatility: {reward_volatility:.3f}',
                'priority': 'medium',
            })

        dynamic_penalty = float(self.state.adaptive_params.get('dynamic_penalty_scaling', 1.0))
        if dynamic_penalty > 1.5:
            recommendations.append({
                'action': 'reduce_penalty_scaling',
                'reason': 'High penalty scaling detected',
                'priority': 'low',
            })

        trade_counts = list(getattr(self.state, "trade_count_history", []))
        if len(trade_counts) >= 10:
            avg_activity = float(np.mean(trade_counts))
            if avg_activity < 0.5:
                recommendations.append({
                    'action': 'increase_trading_activity',
                    'reason': f'Low trading activity: {avg_activity:.1f} trades/period',
                    'priority': 'medium',
                })

        return recommendations

    def _generate_regime_recommendation(self, regime_stats: Dict[str, Dict[str, float]]) -> str:
        """Generate regime-specific recommendation"""

        if not regime_stats:
            return "Insufficient data for regime recommendations"

        best_regime = max(regime_stats.items(), key=lambda x: x[1]['avg_reward'])
        if best_regime[1]['avg_reward'] > 0.1:
            return f"Optimize for {best_regime[0]} regime (best performance)"
        elif all(r['avg_reward'] < 0 for r in regime_stats.values()):
            return "All regimes showing negative performance - review overall strategy"
        else:
            return "Mixed regime performance - consider regime-adaptive strategies"

    # ─────────────────────────────────────────────────────────────
    # Monitoring-side effectiveness check (sync)
    # ─────────────────────────────────────────────────────────────
    def analyze_effectiveness(self) -> None:
        """Analyze overall system effectiveness (used by monitoring thread)"""
        try:
            rewards = list(getattr(self.state, "reward_history", []))
            if len(rewards) >= 10:
                recent = rewards[-10:]
                positive_ratio = float(sum(1 for r in recent if r > 0)) / float(len(recent))
                avg_recent = float(np.mean(recent))
                if positive_ratio > 0.8:
                    self.logger.info(f"High reward effectiveness: {positive_ratio:.1%} positive (avg: {avg_recent:.4f})")
                elif positive_ratio < 0.2:
                    self.logger.warning(f"Low reward effectiveness: {positive_ratio:.1%} positive (avg: {avg_recent:.4f})")
        except Exception as e:
            self.logger.error(f"Effectiveness analysis failed: {e}")

    # ─────────────────────────────────────────────────────────────
    # Baselines / State IO
    # ─────────────────────────────────────────────────────────────
    def get_baseline_analytics(self) -> Dict[str, Any]:
        """Get baseline analytics structure"""
        return {
            'performance_metrics': self.state.get_performance_metrics(),
            'component_analysis': {},
            'regime_analysis': {'current_regime': 'unknown', 'regime_statistics': {}},
            'trends': {},
            'insights': [],
            'analysis_timestamp': datetime.now().isoformat(),
        }

    def reset(self) -> None:
        """Reset analytics state"""
        with self._lock:
            self.performance_analytics.clear()
            self.component_effectiveness.clear()
            self.regime_analytics.clear()
            self.session_analytics.clear()
            self.daily_performance.clear()
            self.reward_trend.clear()
            self.quality_trend.clear()
            self.effectiveness_trend.clear()
            self.analysis_count = 0

    def get_state(self) -> Dict[str, Any]:
        """Get analytics state for persistence (compact summary)"""
        with self._lock:
            # Only persist stable summaries (not large deques)
            return {
                'analysis_count': self.analysis_count,
                'component_effectiveness': {
                    k: {
                        'total_magnitude': v.total_magnitude,
                        'occurrences': v.occurrences,
                        'avg_magnitude': v.avg_magnitude,
                        'impact_score': v.impact_score,
                    }
                    for k, v in self.component_effectiveness.items()
                },
                'regime_analytics': {
                    k: {
                        'avg_reward': v.avg_reward,
                        'win_rate': v.win_rate,
                        'sample_count': float(len(v.rewards)),
                    }
                    for k, v in self.regime_analytics.items()
                },
            }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set analytics state from persistence"""
        with self._lock:
            self.analysis_count = int(state.get('analysis_count', 0))
            if 'component_effectiveness' in state:
                incoming = state['component_effectiveness'] or {}
                for k, v in incoming.items():
                    cs = self.component_effectiveness[k]
                    cs.total_magnitude = float(v.get('total_magnitude', cs.total_magnitude))
                    cs.occurrences = int(v.get('occurrences', cs.occurrences))
                    cs.avg_magnitude = float(v.get('avg_magnitude', cs.avg_magnitude))
                    cs.impact_score = float(v.get('impact_score', cs.impact_score))

    # ─────────────────────────────────────────────────────────────
    # Logging helpers (never raise)
    # ─────────────────────────────────────────────────────────────
    def _warn(self, msg: str, tag: str = "WARN") -> None:
        try:
            if getattr(self.debug_manager, "enabled", False) and hasattr(self.debug_manager, "log_error"):
                self.debug_manager.log_error(tag, RuntimeError(msg))
            else:
                self.logger.error(msg)
        except Exception:
            try:
                self.logger.error(msg)
            except Exception:
                pass
