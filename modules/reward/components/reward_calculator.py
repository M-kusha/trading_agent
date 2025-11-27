# modules/reward/components/reward_calculator.py
"""
Reward Calculation Component
Handles core reward calculation logic with all penalties and bonuses
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Sequence
import numpy as np
from datetime import datetime


class RewardCalculator:
    """
    Core reward calculation engine

    Features:
    - PnL-based reward calculation
    - Dynamic penalty application
    - Bonus calculations
    - Component tracking
    - Debug integration
    """

    def __init__(
        self,
        config: Any,
        state: Any,
        logger: Any,
        debug_manager: Any,
        env: Optional[Any] = None,   # NEW: make env explicit & optional to satisfy Pylance
    ):
        """Initialize calculator"""

        self.cfg = config
        self.state = state
        self.logger = logger
        self.debug_manager = debug_manager
        self.env: Optional[Any] = env  # ensure attribute exists → fixes Pylance 'env unknown'

        # Calculation tracking
        self.calculation_count: int = 0
        self.component_magnitudes: Dict[str, float] = {}
        
        # Step-by-step PnL tracking (for proper RL reward signals)
        # RL needs the CHANGE in value per step, not cumulative totals
        self._last_step_balance: Optional[float] = None

    # ─────────────────────────────────────────────────────────────
    # Public
    # ─────────────────────────────────────────────────────────────

    async def calculate_enhanced_reward(
        self,
        reward_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate reward with all components

        Returns shaped reward and detailed components
        """

        self.calculation_count += 1

        # Extract core data
        trades: List[Dict[str, Any]] = list(reward_data.get('trades', []) or [])
        balance_now: float = self._to_float_safe(reward_data.get('balance_now', 0.0))
        baseline_balance_raw = reward_data.get('baseline_balance', balance_now)
        baseline_balance: Optional[float] = (
            None if baseline_balance_raw is None else self._to_float_safe(baseline_balance_raw)
        )

        # Calculate denominator (never zero)
        denom = baseline_balance if (baseline_balance and baseline_balance > 0) else max(1e-9, balance_now)

        # FIX: Calculate STEP-BY-STEP PnL change (not cumulative total)
        # RL needs per-step rewards that reflect the action's immediate impact
        # Using cumulative PnL causes the reward to always be large negative after losses
        if self._last_step_balance is None:
            self._last_step_balance = baseline_balance if baseline_balance else balance_now
        
        # Step delta: change since last step (what RL actually learns from)
        step_pnl_delta = balance_now - self._last_step_balance
        self._last_step_balance = balance_now
        
        # Normalize by initial balance to keep in reasonable range
        step_reward = step_pnl_delta / denom * 100.0  # Scale: 1% balance change = 1.0 reward
        
        # Also track cumulative for metrics/debugging
        total_pnl = balance_now - (baseline_balance if baseline_balance else balance_now)

        # Also calculate realized-only PnL from trades for metrics/debugging
        realised_pnl_from_trades = self._calculate_realised_pnl(trades)

        base_component = step_pnl_delta / denom  # Use step delta, not cumulative

        # Initialize components
        components = self._initialize_components(
            reward_data, total_pnl, base_component,
            balance_now, baseline_balance
        )
        components['step_pnl_delta'] = step_pnl_delta  # Track for debugging

        # FIX: Start with STEP delta as base reward (not cumulative PnL!)
        # This gives the RL agent a proper learning signal
        reward = step_reward

        # Apply penalties
        penalties = await self._calculate_penalties(reward_data, components)
        if penalties:
            reward -= sum(penalties.values())
            components.update(penalties)

        # Apply bonuses
        bonuses = await self._calculate_bonuses(reward_data, trades, realised_pnl_from_trades, components)
        if bonuses:
            reward += sum(bonuses.values())
            components.update(bonuses)

        # Apply consensus factor (clamped to [0.25, 1.75] to avoid runaway scaling)
        consensus_factor = float(np.clip(components['consensus_factor'], 0.25, 1.75))
        components['consensus_factor'] = consensus_factor
        reward *= consensus_factor

        # Final clipping
        final_reward = float(np.clip(reward, -10.0, 10.0))
        components['final_reward'] = final_reward
        components['method'] = 'enhanced_async_calculation'

        # Update state
        await self._update_calculation_state(trades, realised_pnl_from_trades, final_reward)

        # Log if significant
        if abs(final_reward) > 0.2 or self.calculation_count % 10 == 1:
            self._log_calculation(final_reward, components, reward_data)

        return {
            'shaped_reward': final_reward,
            'reward_components': components,
            'calculation_method': 'enhanced_async'
        }

    # ─────────────────────────────────────────────────────────────
    # Initialization & Base
    # ─────────────────────────────────────────────────────────────

    def _initialize_components(
        self,
        reward_data: Dict[str, Any],
        realised_pnl: float,
        base_component: float,
        balance_now: float,
        baseline_balance: Optional[float]
    ) -> Dict[str, Any]:
        """Initialize component dictionary with safe defaults"""

        consensus_raw = reward_data.get('consensus', 0.5)
        try:
            consensus_val = float(consensus_raw)
        except Exception:
            consensus_val = 0.5
        consensus_val = float(np.clip(consensus_val, 0.0, 1.0))

        return {
            'pnl': float(realised_pnl),
            'base_component': float(base_component),
            'balance_now': float(balance_now),
            'baseline_balance': float(baseline_balance) if baseline_balance else 0.0,
            'drawdown': self._extract_drawdown(reward_data),
            'consensus': consensus_val,
            'trades_count': int(len(reward_data.get('trades', []) or [])),
            'timestamp': reward_data.get('timestamp', datetime.now().isoformat()),
            'step_idx': int(reward_data.get('step_idx', 0) or 0),
            'market_regime': str(reward_data.get('regime', 'unknown') or 'unknown'),
            'volatility_level': str(reward_data.get('volatility_level', 'medium') or 'medium'),
            'consensus_factor': 0.5 + consensus_val,
            # Initialize all component slots
            'drawdown_penalty': 0.0,
            'risk_penalty': 0.0,
            'tail_penalty': 0.0,
            'mistake_penalty': 0.0,
            'no_trade_penalty': 0.0,
            'win_bonus': 0.0,
            'activity_bonus': 0.0,
            'consistency_bonus': 0.0,
            'sharpe_bonus': 0.0,
            'regime_bonus': 0.0,
            'volatility_adjustment': 0.0,
        }

    def _calculate_realised_pnl(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate total realised PnL from trades.

        Accepts multiple common field names to be robust across executors:
        - 'pnl' (direct)
        - 'realized_pnl' / 'realised_pnl' (executor fill schema)
        - fallback: 0.0 if none present
        """
        if not trades:
            return 0.0
        total_pnl = 0.0
        for trade in trades:
            if not isinstance(trade, dict):
                continue
            val = None
            for key in ("pnl", "realized_pnl", "realised_pnl", "realized", "realised"):
                v = trade.get(key)
                if isinstance(v, (int, float)):
                    val = float(v)
                    break
            if val is None:
                continue
            try:
                total_pnl += float(val)
            except Exception:
                continue
        return float(total_pnl)

    def _extract_drawdown(self, reward_data: Dict[str, Any]) -> float:
        """Extract drawdown from various sources, safely"""
        risk_metrics = reward_data.get('risk_metrics') or {}
        drawdown = (risk_metrics.get('current_drawdown')
                    if isinstance(risk_metrics, dict) else None)
        if drawdown is None and isinstance(risk_metrics, dict):
            drawdown = risk_metrics.get('drawdown', 0.0)

        if not drawdown:
            market_state = reward_data.get('market_state') or {}
            if isinstance(market_state, dict):
                drawdown = market_state.get('drawdown', 0.0)

        try:
            return float(drawdown) if drawdown is not None else 0.0
        except Exception:
            return 0.0

    # ─────────────────────────────────────────────────────────────
    # Penalty Calculations
    # ─────────────────────────────────────────────────────────────

    async def _calculate_penalties(
        self,
        reward_data: Dict[str, Any],
        components: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate all penalties"""

        penalties: Dict[str, float] = {}

        # Drawdown penalty
        drawdown_penalty = self._calculate_drawdown_penalty(
            components['drawdown'],
            components['market_regime']
        )
        if drawdown_penalty > 0:
            penalties['drawdown_penalty'] = drawdown_penalty

        # Risk penalty
        risk_penalty = self._calculate_risk_penalty(
            reward_data.get('actions'),
            components['volatility_level']
        )
        if risk_penalty > 0:
            penalties['risk_penalty'] = risk_penalty

        # Tail risk penalty
        tail_penalty = self._calculate_tail_penalty(
            reward_data.get('trades', []) or []
        )
        if tail_penalty > 0:
            penalties['tail_penalty'] = tail_penalty

        # Mistake penalty
        mistake_penalty = await self._calculate_mistake_penalty(reward_data)
        if mistake_penalty > 0:
            penalties['mistake_penalty'] = mistake_penalty

        # No trade penalty (if no trades)
        trades_any = bool(reward_data.get('trades'))
        if not trades_any:
            no_trade_penalty = self._calculate_no_trade_penalty(
                components['drawdown'],
                components['volatility_level']
            )
            if no_trade_penalty > 0:
                penalties['no_trade_penalty'] = no_trade_penalty

        return penalties

    def _calculate_drawdown_penalty(self, drawdown: float, regime: str) -> float:
        """Calculate drawdown penalty"""
        try:
            d = float(drawdown)
        except Exception:
            d = 0.0

        if d <= 0.05:
            return 0.0

        # Quadratic penalty
        penalty = (d ** 2) * float(self.cfg.dd_pen_weight)

        # Adjust for regime
        regime_multipliers = {
            'volatile': 0.8,   # Less penalty in volatile
            'trending': 1.2,   # More penalty in trending
            'ranging': 1.0,
            'unknown': 1.0
        }
        penalty *= regime_multipliers.get(str(regime), 1.0)

        # Apply adaptive scaling
        penalty *= float(self.state.adaptive_params.get('dynamic_penalty_scaling', 1.0))

        return float(penalty)

    def _calculate_risk_penalty(
        self,
        actions: Optional[Any],
        volatility_level: str
    ) -> float:
        """Calculate risk penalty based on actions"""
        if actions is None:
            return 0.0

        try:
            # Calculate action magnitude
            if isinstance(actions, (list, tuple, np.ndarray)):
                action_magnitude = float(np.linalg.norm(np.array(actions, dtype=np.float32)))
            else:
                action_magnitude = abs(float(actions))

            # Base penalty
            penalty = min(action_magnitude * float(self.cfg.risk_pen_weight), 0.2)

            # Adjust for volatility
            vol_multipliers = {
                'low': 1.2,
                'medium': 1.0,
                'high': 0.8,
                'extreme': 0.6
            }
            penalty *= float(vol_multipliers.get(str(volatility_level), 1.0))

            return float(penalty)

        except Exception:
            return 0.0

    def _calculate_tail_penalty(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate tail risk penalty"""
        if not trades:
            return 0.0

        losses: List[float] = []
        for trade in trades:
            if isinstance(trade, dict):
                pnl_raw = trade.get('pnl', 0.0)
                try:
                    pnl = float(pnl_raw)
                except Exception:
                    continue
                if pnl < 0.0:
                    losses.append(pnl)

        if not losses:
            return 0.0

        # Calculate penalty based on average loss
        avg_loss = abs(float(np.mean(losses)))
        penalty = avg_loss * float(self.cfg.tail_pen_weight) * 0.1

        # Extra penalty for extreme losses
        extreme_losses = [l for l in losses if l < -100.0]
        if extreme_losses:
            penalty *= 1.5

        return float(penalty)

    async def _calculate_mistake_penalty(self, reward_data: Dict[str, Any]) -> float:
        """Calculate mistake-based penalty. Robust to missing sources."""
        try:
            # Try reward_data (preferred)
            mistake_memory = reward_data.get('mistake_memory')
            if not isinstance(mistake_memory, dict):
                # Some pipelines place it inside raw_inputs
                mistake_memory = (reward_data.get('raw_inputs') or {}).get('mistake_memory')

            if isinstance(mistake_memory, dict):
                score_raw = mistake_memory.get('current_score', 0.0)
                try:
                    score = float(score_raw)
                except Exception:
                    score = 0.0
                penalty = score * float(self.cfg.mistake_pen_weight)
                penalty *= float(self.state.adaptive_params.get('dynamic_penalty_scaling', 1.0))
                return float(penalty)

            # Try environment (optional)
            if self.env is not None and hasattr(self.env, 'mistake_memory'):
                try:
                    mm = self.env.mistake_memory.get_observation_components()  # type: ignore[attr-defined]
                    if mm is not None and len(mm) > 0:
                        first = float(mm[0])
                        return float(first * float(self.cfg.mistake_pen_weight))
                except Exception:
                    pass

        except Exception as e:
            if getattr(self.debug_manager, "enabled", False):
                try:
                    self.debug_manager.log_error("MISTAKE_PENALTY_ERROR", e)  # type: ignore[attr-defined]
                except Exception:
                    self.logger.debug(f"Mistake penalty error: {e}")

        return 0.0

    def _calculate_no_trade_penalty(
        self,
        drawdown: float,
        volatility_level: str
    ) -> float:
        """Calculate penalty for not trading"""
        penalty = float(self.cfg.no_trade_penalty_weight)
        penalty *= float(self.state.adaptive_params.get('activity_threshold', 1.0))

        # Reduce penalty if in drawdown
        try:
            if float(drawdown) > 0.1:
                penalty *= 0.3
        except Exception:
            pass

        # Reduce penalty in extreme volatility
        if str(volatility_level) == 'extreme':
            penalty *= 0.5

        return float(penalty)

    # ─────────────────────────────────────────────────────────────
    # Bonus Calculations
    # ─────────────────────────────────────────────────────────────

    async def _calculate_bonuses(
        self,
        reward_data: Dict[str, Any],
        trades: List[Dict[str, Any]],
        realised_pnl: float,
        components: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate all bonuses"""

        bonuses: Dict[str, float] = {}

        if trades:
            # Win bonus
            win_bonus = self._calculate_win_bonus(trades)
            if win_bonus > 0:
                bonuses['win_bonus'] = win_bonus

            # Activity bonus
            activity_bonus = self._calculate_activity_bonus(
                len(trades), realised_pnl
            )
            if activity_bonus > 0:
                bonuses['activity_bonus'] = activity_bonus

        # Consistency bonus
        consistency_bonus = await self._calculate_consistency_bonus()
        if consistency_bonus > 0:
            bonuses['consistency_bonus'] = consistency_bonus

        # Sharpe bonus (can be negative)
        sharpe_bonus = await self._calculate_sharpe_bonus()
        if sharpe_bonus != 0:
            bonuses['sharpe_bonus'] = sharpe_bonus

        # Regime bonus
        regime_bonus = await self._calculate_regime_bonus(
            components['market_regime'], realised_pnl
        )
        if regime_bonus != 0:
            bonuses['regime_bonus'] = regime_bonus

        # Volatility adjustment
        vol_adjustment = await self._calculate_volatility_adjustment(
            components['volatility_level'], realised_pnl
        )
        if vol_adjustment != 0:
            bonuses['volatility_adjustment'] = vol_adjustment

        return bonuses

    def _calculate_win_bonus(self, trades: List[Dict[str, Any]]) -> float:
        """Calculate winning trade bonus.
        
        FIXED: Prevents positive feedback loop by:
        1. Requiring minimum history before streak bonuses
        2. Capping streak bonus multiplier
        3. Using diminishing returns for high win rates
        """
        if not trades:
            return 0.0

        winning_trades = 0
        for t in trades:
            try:
                if float(t.get('pnl', 0.0)) > 0.0:
                    winning_trades += 1
            except Exception:
                continue

        win_ratio = winning_trades / max(1, len(trades))

        # Use sqrt for diminishing returns - prevents overfitting to early wins
        # 50% win rate -> 0.71 factor, 80% win rate -> 0.89 factor
        diminishing_factor = float(np.sqrt(win_ratio))
        bonus = diminishing_factor * float(self.cfg.win_bonus_weight)

        # Streak bonus ONLY if we have sufficient history (prevents early overfitting)
        history_len = len(self.state.pnl_history)
        if history_len >= 10:  # Need 10+ samples before streak bonus
            recent = list(self.state.pnl_history)[-3:]
            recent_wins = []
            for p in recent:
                try:
                    recent_wins.append(float(p) > 0.0)
                except Exception:
                    recent_wins.append(False)
            if all(recent_wins):
                # Cap streak bonus at 1.15x (was 1.3x - too aggressive)
                bonus *= 1.15

        return float(bonus)

    def _calculate_activity_bonus(
        self,
        trade_count: int,
        realised_pnl: float
    ) -> float:
        """Calculate trading activity bonus"""
        bonus = min(int(trade_count) * 0.1, 0.3)

        # Extra bonus for profitable activity
        try:
            if float(realised_pnl) > 0.0:
                bonus *= 1.2
        except Exception:
            pass

        return float(bonus)

    async def _calculate_consistency_bonus(self) -> float:
        """Calculate consistency bonus"""
        if len(self.state.pnl_history) < 3:
            return 0.0

        recent_pnls = list(self.state.pnl_history)[-10:]
        if not recent_pnls:
            return 0.0

        positives = 0
        for p in recent_pnls:
            try:
                if float(p) > 0.0:
                    positives += 1
            except Exception:
                continue

        positive_ratio = positives / max(1, len(recent_pnls))

        # Quadratic for consistency
        consistency_score = positive_ratio ** 2

        # Momentum bonus
        if len(recent_pnls) >= 5:
            early_half = recent_pnls[:len(recent_pnls)//2]
            late_half = recent_pnls[len(recent_pnls)//2:]

            def _ratio(arr: Sequence[float]) -> float:
                pos = 0
                cnt = 0
                for v in arr:
                    try:
                        cnt += 1
                        if float(v) > 0.0:
                            pos += 1
                    except Exception:
                        continue
                return (pos / cnt) if cnt else 0.0

            early_ratio = _ratio(early_half)
            late_ratio = _ratio(late_half)
            momentum = late_ratio - early_ratio
            consistency_score *= (1.0 + momentum * 0.2)

        # Streak bonus
        streak = 0
        for pnl in reversed(recent_pnls):
            try:
                if float(pnl) > 0.0:
                    streak += 1
                else:
                    break
            except Exception:
                break

        if streak >= 3:
            consistency_score *= (1.0 + streak * 0.1)

        self.state.consistency_score = float(consistency_score)

        return float(consistency_score * float(self.cfg.consistency_bonus_weight))

    async def _calculate_sharpe_bonus(self) -> float:
        """Calculate Sharpe ratio bonus"""
        if len(self.state.reward_history) < 5:
            return 0.0

        rewards = np.array(list(self.state.reward_history), dtype=np.float32)
        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards))

        # Minimum std to avoid division issues
        min_std = max(0.1, abs(mean_reward) * 0.1)
        std_reward = max(std_reward, min_std)

        # Calculate Sharpe (annualized-like scaling with sqrt(N))
        sharpe = mean_reward / std_reward * np.sqrt(min(len(rewards), 252))

        # Regime adjustment
        if hasattr(self.state, 'last_regime'):
            regime_multipliers = {
                'trending': 1.2,
                'ranging': 1.0,
                'volatile': 0.8,
                'unknown': 0.9
            }
            sharpe *= regime_multipliers.get(getattr(self.state, 'last_regime', 'unknown'), 1.0)

        # Apply sensitivity
        sensitivity = float(self.state.adaptive_params.get('regime_sensitivity', 1.0))
        normalized_sharpe = np.tanh(sharpe / (6.0 / max(1e-6, sensitivity)))

        bonus = float(np.clip(normalized_sharpe * float(self.cfg.sharpe_bonus_weight), -0.5, 0.5))

        self.state.sharpe_ratio = float(sharpe)

        return bonus

    async def _calculate_regime_bonus(
        self,
        regime: str,
        pnl: float
    ) -> float:
        """Calculate regime-specific bonus"""
        regime_str = str(regime)

        # track in state for analytics
        try:
            self.state.last_regime = regime_str
            self.state.record_regime_performance(regime_str, float(pnl))
        except Exception:
            pass

        bonus = 0.0
        sensitivity = float(self.state.adaptive_params.get('regime_sensitivity', 1.0))
        pnl_f = float(pnl)

        if regime_str == 'trending' and pnl_f > 0.0:
            bonus = min(pnl_f / 100.0, 0.2) * float(self.cfg.regime_bonus_weight) * sensitivity
        elif regime_str == 'ranging' and abs(pnl_f) < 20.0:
            bonus = 0.1 * float(self.cfg.regime_bonus_weight) * sensitivity
        elif regime_str == 'volatile':
            if pnl_f < -50.0:
                bonus = -0.15 * float(self.cfg.regime_bonus_weight) * sensitivity
            elif 0.0 < pnl_f < 30.0:
                bonus = 0.1 * float(self.cfg.regime_bonus_weight) * sensitivity

        return float(bonus)

    async def _calculate_volatility_adjustment(
        self,
        volatility_level: str,
        pnl: float
    ) -> float:
        """Calculate volatility-based adjustment"""

        try:
            self.state.record_volatility_performance(str(volatility_level), float(pnl))
        except Exception:
            pass

        vol_multipliers = {
            'low': 1.1,
            'medium': 1.0,
            'high': 0.9,
            'extreme': 0.8
        }

        base = (vol_multipliers.get(str(volatility_level), 1.0) - 1.0) * abs(float(pnl)) * 0.1

        # Bonus for profitable trading in high volatility
        if str(volatility_level) in ('high', 'extreme') and float(pnl) > 0.0:
            base += float(pnl) * 0.05

        adaptive_factor = float(self.state.adaptive_params.get('risk_tolerance', 1.0))

        return float(base * float(self.cfg.volatility_adjustment) * adaptive_factor)

    # ─────────────────────────────────────────────────────────────
    # State & Logging
    # ─────────────────────────────────────────────────────────────

    async def _update_calculation_state(
        self,
        trades: List[Dict[str, Any]],
        pnl: float,
        reward: float
    ) -> None:
        """Update state after calculation"""
        try:
            self.state.record_calculation(trades, float(pnl), float(reward))
            
            # SPARSE REWARD DETECTION: Warn if PnL is always zero (training will fail)
            if self.calculation_count > 20:
                recent_pnls = list(self.state.pnl_history)[-20:] if hasattr(self.state, 'pnl_history') else []
                if recent_pnls and all(abs(p) < 1e-6 for p in recent_pnls):
                    self.logger.warning(
                        "⚠️ SPARSE REWARD: PnL has been zero for 20+ steps. "
                        "Model cannot learn. Check: 1) Trades closing properly, "
                        "2) Balance updates flowing, 3) Position manager publishing fills"
                    )
        except Exception as e:
            # never fail hard on state updates
            self.logger.debug(f"record_calculation failed: {e}")

    def _log_calculation(
        self,
        reward: float,
        components: Dict[str, Any],
        reward_data: Dict[str, Any]
    ) -> None:
        """Log calculation details"""
        try:
            self.logger.info(
                f"Reward calculated: {float(reward):.4f} | "
                f"PnL: {float(components.get('pnl', 0.0)):.2f} | "
                f"Trades: {int(components.get('trades_count', 0))} | "
                f"Regime: {components.get('market_regime', 'unknown')} | "
                f"Volatility: {components.get('volatility_level', 'medium')}"
            )
        except Exception:
            # logging must not break pipeline
            pass

    def reset(self) -> None:
        """Reset step-by-step tracking for new episode"""
        self._last_step_balance = None
        self.calculation_count = 0

    # ─────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _to_float_safe(v: Any, default: float = 0.0) -> float:
        try:
            f = float(v)
            if not np.isfinite(f):
                return default
            return f
        except Exception:
            return default
