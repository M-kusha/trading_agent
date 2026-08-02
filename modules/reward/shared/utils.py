# modules/reward/shared/utils.py
"""
Shared Utilities for Reward System
Common helper functions and utilities
"""

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

import numpy as np


class RewardUtils:
    """Utility functions for reward system"""

    @staticmethod
    def utcnow() -> str:
        """Get current UTC timestamp as ISO string (timezone-aware)."""
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def generate_thesis(
        reward_data: Dict[str, Any],
        result: Dict[str, Any],
        state: Any
    ) -> str:
        """Generate comprehensive thesis for reward calculation."""
        try:
            parts: List[str] = []

            # Core metrics
            reward = float(result.get('shaped_reward', 0.0))
            components = result.get('reward_components', {}) or {}
            pnl = float(components.get('pnl', 0.0) or 0.0)
            trades_count = int(components.get('trades_count', 0) or 0)

            parts.append(f"Reward: {reward:.4f} from {pnl:.2f} PnL over {trades_count} trades")

            # Context
            regime = str(reward_data.get('regime', 'unknown'))
            volatility = str(reward_data.get('volatility_level', 'medium'))
            parts.append(f"Context: {regime.upper()} regime, {volatility.upper()} vol")

            # Quality assessment
            rq = float(getattr(state, 'reward_quality', 0.5))
            win_rate = float(getattr(state, 'win_rate', 0.0))
            if rq > 0.7:
                parts.append(f"Quality: HIGH ({rq:.2f}) | WinRate {win_rate:.1%}")
            elif rq < 0.3:
                parts.append(f"Quality: LOW ({rq:.2f})")
            else:
                parts.append(f"Quality: MODERATE ({rq:.2f})")

            # Major components (penalties, bonuses, and adjustments), sorted by |impact|
            majors: List[tuple[str, float]] = []
            for k, v in components.items():
                if not isinstance(v, (int, float)):
                    continue
                if k.endswith(('_penalty', '_bonus', '_adjustment')):
                    if abs(float(v)) > 0.01:
                        majors.append((k, float(v)))

            if majors:
                majors.sort(key=lambda kv: abs(kv[1]), reverse=True)
                parts.append("Major: " + ", ".join(f"{k}={v:.3f}" for k, v in majors[:3]))

            # Adaptation confidence
            adapt_conf = float(getattr(state, 'adaptive_params', {}).get('adaptation_confidence', 0.5))
            if adapt_conf > 0.8:
                parts.append("Adaptation: HIGH confidence")
            elif adapt_conf < 0.3:
                parts.append("Adaptation: LOW confidence")

            # Warnings
            cb_state = getattr(state, 'circuit_breaker', {}).get('state', 'CLOSED')
            if cb_state == 'OPEN':
                parts.append("ALERT: Circuit breaker OPEN")
            elif getattr(state, 'health_status', 'healthy') == 'warning':
                parts.append("WARNING: Health degraded")

            return " | ".join(parts)

        except Exception as e:
            return f"Reward calculation completed (thesis generation failed: {e})"

    @staticmethod
    def generate_system_report(
        state: Any,
        analytics_engine: Any,   # kept for API parity/future use
        adaptation_manager: Any,  # kept for API parity/future use
        config: Any
    ) -> str:
        """Generate comprehensive system report."""

        # Performance status
        avg_reward = float(getattr(state, 'avg_reward', 0.0))
        if avg_reward > 0.5:
            performance_status = "🚀 Excellent"
        elif avg_reward > 0.0:
            performance_status = "✅ Good"
        elif avg_reward > -0.5:
            performance_status = "⚡ Fair"
        else:
            performance_status = "⚠️ Poor"

        # Quality status
        reward_quality = float(getattr(state, 'reward_quality', 0.5))
        if reward_quality > 0.8:
            quality_status = "🎯 High"
        elif reward_quality > 0.6:
            quality_status = "✅ Good"
        elif reward_quality > 0.4:
            quality_status = "⚡ Fair"
        else:
            quality_status = "❌ Low"

        # Circuit breaker & health
        cb_state = getattr(state, 'circuit_breaker', {}).get('state', 'CLOSED')
        cb_status = "🔴 OPEN" if cb_state == 'OPEN' else "🟢 CLOSED"
        health = getattr(state, 'health_status', 'healthy')
        health_emoji = "✅" if health == 'healthy' else "⚠️"

        # Regime weights formatting (defensive)
        weights_raw = getattr(config, 'regime_weights', []) or []
        try:
            regime_weights_str = ', '.join(f"{float(w):.2f}" for w in list(weights_raw))
        except Exception:
            regime_weights_str = "n/a"

        last_balance = getattr(state, 'last_balance_observed', None)
        baseline = getattr(state, 'baseline_balance', None)

        report = f"""
═══════════════════════════════════════════════════════════════
🎯 ENHANCED RISK-ADJUSTED REWARD SYSTEM v4.2
═══════════════════════════════════════════════════════════════

📊 PERFORMANCE
- Status: {performance_status} ({avg_reward:.4f} avg)
- Quality: {quality_status} ({reward_quality:.3f})
- Consistency: {float(getattr(state, 'consistency_score', 0.0)):.3f}
- Sharpe Ratio: {float(getattr(state, 'sharpe_ratio', 0.0)):.3f}
- Win Rate: {float(getattr(state, 'win_rate', 0.0)):.1%}

🏥 SYSTEM HEALTH
- Status: {health_emoji} {health.upper()}
- Circuit Breaker: {cb_status}
- Mode: {getattr(getattr(state, 'current_mode', None), 'value', 'TRAINING').upper()}
- Adaptation Confidence: {float(getattr(state, 'adaptive_params', {}).get('adaptation_confidence', 0.5)):.2f}

💰 ACCOUNT
- Last Balance: {last_balance if last_balance is not None else 'N/A'}
- Baseline: {baseline if baseline is not None else 'N/A'}

⚖️ CONFIGURATION
- Regime Weights: [{regime_weights_str}]
- Drawdown Penalty: {float(getattr(config, 'dd_pen_weight', 0.0)):.2f}
- Win Bonus: {float(getattr(config, 'win_bonus_weight', 0.0)):.2f}
- Consistency Bonus: {float(getattr(config, 'consistency_bonus_weight', 0.0)):.2f}

🔧 ADAPTIVE PARAMETERS
- Penalty Scaling: {float(getattr(state, 'adaptive_params', {}).get('dynamic_penalty_scaling', 1.0)):.2f}
- Regime Sensitivity: {float(getattr(state, 'adaptive_params', {}).get('regime_sensitivity', 1.0)):.2f}
- Activity Threshold: {float(getattr(state, 'adaptive_params', {}).get('activity_threshold', 1.0)):.2f}
- Risk Tolerance: {float(getattr(state, 'adaptive_params', {}).get('risk_tolerance', 1.0)):.2f}

📈 ACTIVITY
- Total Calls: {int(getattr(state, 'call_count', 0)):,}
- Reward History: {len(getattr(state, 'reward_history', []))} records
- Audit Trail: {len(getattr(state, 'audit_trail', []))} entries
- Last Reward: {float(getattr(state, 'last_reward', 0.0)):.4f} ({getattr(state, 'last_reason', '')})

═══════════════════════════════════════════════════════════════
"""
        return report

    @staticmethod
    def safe_float(value: Any, default: float = 0.0) -> float:
        """Safely convert value to float."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def safe_mean(values: Iterable[float]) -> float:
        """Calculate mean with safety checks (handles empty iterables)."""
        try:
            values_list = list(values)
            if not values_list:
                return 0.0
            return float(np.mean(values_list))
        except Exception:
            return 0.0
