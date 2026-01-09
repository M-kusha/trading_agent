# envs/prop_firm/rewards/trade_reward.py
# pyright: reportAttributeAccessIssue=false
"""
Trade reward computation mixin for PropFirmTradingEnv.

Contains the main _compute_trade_reward method with all reward components.

Upgrades:
- Replace hard-coded EUR thresholds with R-multiple thresholds (safer across sizing/instruments)
- Scale shaping terms by trade magnitude to reduce reward hacking
- Instrument-aware session timing fallback
- Clamp capture ratio for robustness
- Backward compatible: new config knobs are accessed via getattr
"""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from envs.core.env_types import TradeResult, PropFirmConfig


class TradeRewardMixin:
    """Mixin providing trade reward computation methods.

    Expected attributes from PropFirmTradingEnv:
    - config: PropFirmConfig
    - instruments: List[str]
    - consecutive_wins: int
    - consecutive_losses: int
    - daily_trades: int
    - _episode_reward_components: Dict[str, float]
    - _episode_reward_component_counts: Dict[str, int]
    - _last_reward_components: Dict[str, float]
    """

    config: "PropFirmConfig"
    instruments: List[str]
    consecutive_wins: int
    consecutive_losses: int
    daily_trades: int
    _episode_reward_components: Dict[str, float]
    _episode_reward_component_counts: Dict[str, int]
    _last_reward_components: Dict[str, float]

    @staticmethod
    def _clamp(x: float, lo: float, hi: float) -> float:
        return float(np.clip(x, lo, hi))

    def _compute_trade_reward(self, result: "TradeResult", current_dd: float) -> float:
        from envs.core.env_types import CloseReason

        cfg = self.config.reward

        net_pnl = float(result.net_pnl)
        risk_eur = max(float(result.initial_risk_eur), 1e-6)

        mae = abs(float(result.mae))
        mfe = float(result.mfe)

        total_fees = float(getattr(result, "total_fees", 0.0))
        gross_pnl = net_pnl + total_fees  # consistent MFE capture comparisons

        bars_held = max(int(result.bars_held), 0)
        close_reason = result.close_reason
        entry_quality = float(result.entry_quality)

        initial_balance = max(float(self.config.initial_balance), 1.0)
        pnl_pct = net_pnl / initial_balance
        r_multiple = net_pnl / risk_eur

        reward = 0.0
        reward_components: Dict[str, float] = {}

        def add(name: str, value: float) -> None:
            nonlocal reward
            v = float(value)
            reward_components[name] = reward_components.get(name, 0.0) + v
            reward += v

        # ---------------------------------------------------------------------
        # Shaping magnitude scale (reduces "farm small trades for bonuses")
        # ---------------------------------------------------------------------
        shaping_scale_by_magnitude = bool(getattr(cfg, "shaping_scale_by_magnitude", True))
        shape_scale_min = float(getattr(cfg, "shape_scale_min", 0.35))
        if shaping_scale_by_magnitude:
            # 0.35..1.0 depending on |R| (conservative default)
            shape_scale = float(np.clip(abs(r_multiple), shape_scale_min, 1.0))
        else:
            shape_scale = 1.0

        # 1) Base PnL reward (asymmetric optional)
        if net_pnl > 0:
            base_reward = pnl_pct * cfg.reward_scale
            add("base_pnl", base_reward)
        else:
            base_penalty = abs(pnl_pct) * cfg.reward_scale * cfg.loss_multiplier
            add("base_pnl", -base_penalty)

        # 2) R-multiple bonus (winners only)
        if net_pnl > 0 and r_multiple >= cfg.r_multiple_bonus_threshold:
            excess_r = r_multiple - cfg.r_multiple_bonus_threshold
            r_bonus = min(excess_r * cfg.r_multiple_bonus_scale, cfg.r_multiple_bonus_cap)
            add("r_multiple_bonus", r_bonus)

        # 3) MAE efficiency
        if cfg.mae_efficiency_enabled and net_pnl > 0 and mae > 0:
            efficiency_ratio = net_pnl / mae
            if efficiency_ratio >= cfg.mae_efficiency_threshold:
                eff_bonus = min(
                    (efficiency_ratio - cfg.mae_efficiency_threshold) * 0.1,
                    cfg.mae_efficiency_scale,
                )
                add("mae_efficiency", eff_bonus)

        elif cfg.mae_efficiency_enabled and net_pnl < 0 and mfe > 0:
            # Missed profit: had meaningful MFE but ended losing
            if mfe > abs(net_pnl) * 0.5:
                missed_profit_penalty = min((mfe / risk_eur) * 0.05, 0.15)
                add("missed_profit_penalty", -missed_profit_penalty)

        # 4) Time efficiency
        if cfg.time_efficiency_enabled and net_pnl > 0:
            if bars_held <= cfg.optimal_trade_bars:
                time_bonus = cfg.time_efficiency_scale * (
                    1.0 - (bars_held / max(cfg.optimal_trade_bars, 1)) * 0.5
                )
                add("time_efficiency", time_bonus)
            elif bars_held <= cfg.max_trade_bars_for_bonus:
                denom = max(cfg.max_trade_bars_for_bonus - cfg.optimal_trade_bars, 1)
                duration_factor = 1.0 - (bars_held - cfg.optimal_trade_bars) / denom
                time_bonus = cfg.time_efficiency_scale * 0.3 * duration_factor
                add("time_efficiency", time_bonus)

        elif cfg.time_efficiency_enabled and net_pnl < 0 and bars_held > cfg.max_trade_bars_for_bonus:
            time_penalty = 0.05 * min(
                (bars_held - cfg.max_trade_bars_for_bonus) / max(cfg.max_trade_bars_for_bonus, 1), 1.0
            )
            add("time_penalty", -time_penalty)

        # 5) Exit quality modifier
        if cfg.exit_quality_enabled:
            exit_modifier = 0.0

            if close_reason == CloseReason.TRAILING_STOP:
                exit_modifier = cfg.trailing_stop_bonus

            elif close_reason == CloseReason.AGENT_CLOSE:
                exit_modifier = cfg.agent_close_bonus

                # Premature close penalty if captured too little of MFE
                if net_pnl > 0 and mfe > 0:
                    capture_ratio = gross_pnl / max(mfe, 1e-9)
                    capture_ratio = self._clamp(capture_ratio, 0.0, 1.5)
                    if capture_ratio < cfg.premature_close_capture_threshold:
                        left_on_table = 1.0 - capture_ratio
                        premature_close_penalty = min(
                            left_on_table * cfg.premature_close_penalty_scale,
                            cfg.premature_close_penalty_cap,
                        )
                        add("premature_close_penalty", -premature_close_penalty)

            elif close_reason == CloseReason.HARD_STOP:
                exit_modifier = -cfg.hard_stop_penalty

            elif close_reason == CloseReason.RISK_LIQUIDATION:
                exit_modifier = -cfg.risk_liquidation_penalty

            elif close_reason == CloseReason.EMERGENCY_CLOSE:
                exit_modifier = -(cfg.hard_stop_penalty * 1.2)

            if exit_modifier != 0.0:
                add("exit_quality", exit_modifier)

        # 6) Truncation handling
        if close_reason == CloseReason.EPISODE_TRUNCATE:
            if net_pnl > 0:
                trunc_discount = reward_components.get("base_pnl", 0.0) * cfg.truncation_winner_discount
                add("truncation_discount", -trunc_discount)
            else:
                add("truncation_penalty", -cfg.truncation_loser_extra_penalty)

        # 7) Entry quality integration
        if cfg.entry_quality_integration:
            qdev = entry_quality - 0.5
            base_mag = abs(reward_components.get("base_pnl", 0.0))
            if net_pnl > 0 and qdev > 0:
                add("entry_quality_bonus", qdev * cfg.entry_quality_weight * base_mag)
            elif net_pnl < 0 and qdev < 0:
                add("entry_quality_penalty", -(abs(qdev) * cfg.entry_quality_weight * base_mag))

        # 7b) Session timing reward/penalty
        if cfg.session_timing_enabled:
            entry_dt = getattr(result, "entry_dt", None)
            if entry_dt is None:
                # Prefer trade instrument if available (multi-instrument correctness)
                inst = getattr(result, "instrument", None)
                if inst is None and self.instruments:
                    inst = self.instruments[0]
                entry_dt = self._get_bar_dt(inst) if inst is not None else None

            if entry_dt is not None:
                if self._in_no_new_trades_window(entry_dt):
                    add("off_hours_penalty", -cfg.off_hours_trade_penalty)
                elif self._in_prime_window(entry_dt):
                    add("prime_hours_bonus", cfg.prime_hours_trade_bonus)

        # 7c) Market structure rewards (teach WHERE to trade)
        if cfg.market_structure_enabled:
            entry_context = getattr(result, "entry_context", {}) or {}
            trade_direction = getattr(result, "direction", None)

            near_support = float(entry_context.get("near_support", 0.0))
            near_resistance = float(entry_context.get("near_resistance", 0.0))

            # Replace hard-coded EUR threshold with R threshold
            trade_worked_r_threshold = float(getattr(cfg, "trade_worked_r_threshold", -0.25))
            trade_worked = (r_multiple > trade_worked_r_threshold)

            ss = shape_scale

            if trade_direction == "long" and near_support > 0.5:
                if trade_worked:
                    add("sr_support_bonus", near_support * cfg.sr_proximity_bonus * ss)
                else:
                    add("sr_support_failed_penalty", -(near_support * cfg.sr_proximity_penalty * 1.5 * ss))

            elif trade_direction == "short" and near_resistance > 0.5:
                if trade_worked:
                    add("sr_resistance_bonus", near_resistance * cfg.sr_proximity_bonus * ss)
                else:
                    add("sr_resistance_failed_penalty", -(near_resistance * cfg.sr_proximity_penalty * 1.5 * ss))

            elif trade_direction == "long" and near_resistance > 0.5:
                add("sr_bad_entry_penalty", -(near_resistance * cfg.sr_proximity_penalty * ss))

            elif trade_direction == "short" and near_support > 0.5:
                add("sr_bad_entry_penalty", -(near_support * cfg.sr_proximity_penalty * ss))

            structure_trend = float(entry_context.get("structure_trend", 0.0))
            if (trade_direction == "long" and structure_trend > 0.3) or \
               (trade_direction == "short" and structure_trend < -0.3):
                add("structure_alignment_bonus", abs(structure_trend) * cfg.structure_alignment_bonus * ss)

            bos_signal = float(entry_context.get("bos_signal", 0.0))
            if cfg.bos_alignment_bonus > 0:
                if (trade_direction == "long" and bos_signal > 0.3) or \
                   (trade_direction == "short" and bos_signal < -0.3):
                    add("bos_alignment_bonus", abs(bos_signal) * cfg.bos_alignment_bonus * ss)

            ob_bull = float(entry_context.get("order_block_bull", 0.0))
            ob_bear = float(entry_context.get("order_block_bear", 0.0))
            if cfg.order_block_entry_bonus > 0:
                if trade_direction == "long" and ob_bull > 0.5:
                    add("order_block_bonus", ob_bull * cfg.order_block_entry_bonus * ss)
                elif trade_direction == "short" and ob_bear > 0.5:
                    add("order_block_bonus", ob_bear * cfg.order_block_entry_bonus * ss)

        # 7d) Divergence awareness rewards
        if cfg.divergence_awareness_enabled:
            entry_context = getattr(result, "entry_context", {}) or {}
            trade_direction = getattr(result, "direction", None)

            divergence = entry_context.get("divergence_signal")
            overbought = float(entry_context.get("overbought", 0.0))
            oversold = float(entry_context.get("oversold", 0.0))

            ss = shape_scale

            if divergence == "bullish" and trade_direction == "short":
                add("divergence_contra_penalty", -(cfg.divergence_contra_penalty * ss))
            elif divergence == "bearish" and trade_direction == "long":
                add("divergence_contra_penalty", -(cfg.divergence_contra_penalty * ss))

            if divergence == "bullish" and trade_direction == "long":
                add("divergence_aligned_bonus", cfg.divergence_aligned_bonus * ss)
            elif divergence == "bearish" and trade_direction == "short":
                add("divergence_aligned_bonus", cfg.divergence_aligned_bonus * ss)

            if overbought > 0.3 and trade_direction == "long":
                add("overbought_long_penalty", -(overbought * cfg.overbought_long_penalty * ss))
            if oversold > 0.3 and trade_direction == "short":
                add("oversold_short_penalty", -(oversold * cfg.oversold_short_penalty * ss))

        # 7e) Regime awareness rewards
        if cfg.regime_awareness_enabled:
            entry_context = getattr(result, "entry_context", {}) or {}
            risk_regime = entry_context.get("risk_regime", "neutral")
            vol_regime = entry_context.get("volatility_regime", "normal")

            ss = shape_scale

            if risk_regime == "risk_off" and net_pnl < 0:
                add("risk_off_penalty", -(cfg.risk_off_aggressive_penalty * ss))

            # Replace magic -50 EUR with optional R-threshold (fallback keeps old behavior)
            high_vol_loss_r_threshold = float(getattr(cfg, "high_vol_loss_r_threshold", float("nan")))
            if vol_regime == "high":
                if np.isfinite(high_vol_loss_r_threshold):
                    if r_multiple < high_vol_loss_r_threshold:
                        add("high_vol_penalty", -(cfg.high_vol_size_penalty * ss))
                else:
                    if net_pnl < -50:
                        add("high_vol_penalty", -(cfg.high_vol_size_penalty * ss))

        # 8) Streak modifiers
        if cfg.streak_modifier_enabled:
            if net_pnl > 0:
                streak_bonus = min(self.consecutive_wins, 5) * cfg.win_streak_bonus_per_win
                if streak_bonus > 0:
                    add("win_streak_bonus", streak_bonus)
            else:
                streak_pen = min(self.consecutive_losses, 3) * cfg.loss_streak_penalty_per_loss
                if streak_pen > 0:
                    add("loss_streak_penalty", -streak_pen)

        # 9) Anti-churn penalty
        if cfg.anti_churn_enabled and self.daily_trades > cfg.daily_trade_soft_limit:
            excess = self.daily_trades - cfg.daily_trade_soft_limit
            excess_factor = min(1.5 ** min(excess, 8) - 1, 25.0)
            churn_pen = excess_factor * cfg.churn_penalty_per_trade
            add("churn_penalty", -churn_pen)

        # 10) Drawdown shaping
        if cfg.dd_shaping_enabled and current_dd > cfg.dd_threshold:
            denom = float(self.config.max_drawdown_limit - cfg.dd_threshold)
            if denom > 1e-9:
                dd_ratio = (current_dd - cfg.dd_threshold) / denom
                dd_ratio = float(np.clip(dd_ratio, 0.0, cfg.dd_severity_cap))
                dd_severity = dd_ratio ** cfg.dd_severity_exponent
                dd_pen = dd_severity * cfg.dd_penalty_scale * 0.5
                add("dd_shaping", -dd_pen)

        # Final clip
        reward = float(np.clip(reward, cfg.min_reward, cfg.max_reward))
        self._last_reward_components = reward_components

        # Aggregate reward components for dashboard tracking
        if not hasattr(self, "_episode_reward_components") or not isinstance(self._episode_reward_components, dict):
            self._episode_reward_components = {}
        if not hasattr(self, "_episode_reward_component_counts") or not isinstance(self._episode_reward_component_counts, dict):
            self._episode_reward_component_counts = {}

        for key, value in reward_components.items():
            if key not in self._episode_reward_components:
                self._episode_reward_components[key] = 0.0
                self._episode_reward_component_counts[key] = 0
            self._episode_reward_components[key] += float(value)
            self._episode_reward_component_counts[key] += 1

        return reward
