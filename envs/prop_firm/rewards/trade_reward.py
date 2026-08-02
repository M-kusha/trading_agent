
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List

import numpy as np

if TYPE_CHECKING:
    from envs.core.env_types import PropFirmConfig, TradeResult


class TradeRewardMixin:

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
        gross_pnl = net_pnl + total_fees

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


        shaping_scale_by_magnitude = bool(getattr(cfg, "shaping_scale_by_magnitude", True))
        shape_scale_min = float(getattr(cfg, "shape_scale_min", 0.35))
        if shaping_scale_by_magnitude:

            shape_scale = float(np.clip(abs(r_multiple), shape_scale_min, 1.0))
        else:
            shape_scale = 1.0


        pnl_scale_factor = float(getattr(cfg, "pnl_scale_factor", 200.0))
        scaled_pnl_pct = pnl_pct * pnl_scale_factor

        if net_pnl > 0:
            base_reward = scaled_pnl_pct * cfg.reward_scale
            add("base_pnl", base_reward)
        else:
            base_penalty = abs(scaled_pnl_pct) * cfg.reward_scale * cfg.loss_multiplier
            add("base_pnl", -base_penalty)


        base_pnl_magnitude = abs(reward_components.get("base_pnl", 0.0))


        execution_cost_enabled = bool(getattr(cfg, "execution_cost_visibility_enabled", True))
        if execution_cost_enabled and total_fees > 0:
            fee_pct = total_fees / initial_balance
            fee_scale = float(getattr(cfg, "execution_cost_reward_scale", 0.5))

            cost_penalty = fee_pct * pnl_scale_factor * cfg.reward_scale * fee_scale
            add("execution_costs", -cost_penalty)


        if net_pnl > 0 and r_multiple >= cfg.r_multiple_bonus_threshold:
            excess_r = r_multiple - cfg.r_multiple_bonus_threshold
            r_bonus = min(excess_r * cfg.r_multiple_bonus_scale, cfg.r_multiple_bonus_cap)
            add("r_multiple_bonus", r_bonus)


        if cfg.mae_efficiency_enabled and net_pnl > 0 and mae > 0:
            efficiency_ratio = net_pnl / mae
            if efficiency_ratio >= cfg.mae_efficiency_threshold:
                eff_bonus = min(
                    (efficiency_ratio - cfg.mae_efficiency_threshold) * 0.1,
                    cfg.mae_efficiency_scale,
                )
                add("mae_efficiency", eff_bonus)

        elif cfg.mae_efficiency_enabled and net_pnl < 0 and mfe > 0:

            if mfe > abs(net_pnl) * 0.5:
                missed_profit_penalty = min((mfe / risk_eur) * 0.05, 0.15)
                add("missed_profit_penalty", -missed_profit_penalty)


        # Peaks AT optimal_trade_bars. The previous shape was
        # scale * (1 - bars/optimal * 0.5), which is largest at zero bars and
        # falls to half AT the optimum - it paid most for exiting instantly and
        # least for holding to target. With optimal=16 a 1-bar exit collected
        # 0.969 of the bonus and a 16-bar hold collected 0.500, so the reward
        # actively taught the 1-bar scalping the median hold showed.
        #
        # Winners only: cutting a loser fast stays free, which is correct.
        if cfg.time_efficiency_enabled and net_pnl > 0:
            optimal = max(cfg.optimal_trade_bars, 1)
            if bars_held <= optimal:
                # Ramps up to the optimum, so exiting early forfeits the bonus.
                duration_factor = bars_held / optimal
            else:
                denom = max(cfg.max_trade_bars_for_bonus - optimal, 1)
                duration_factor = 1.0 - (bars_held - optimal) / denom
            duration_factor = max(0.0, min(1.0, duration_factor))
            if duration_factor > 0.0:
                add("time_efficiency", cfg.time_efficiency_scale * duration_factor)

        elif cfg.time_efficiency_enabled and net_pnl < 0 and bars_held > cfg.max_trade_bars_for_bonus:
            time_penalty = 0.05 * min(
                (bars_held - cfg.max_trade_bars_for_bonus) / max(cfg.max_trade_bars_for_bonus, 1), 1.0
            )
            add("time_penalty", -time_penalty)


        if cfg.exit_quality_enabled:
            exit_modifier = 0.0

            if close_reason == CloseReason.TRAILING_STOP:
                exit_modifier = cfg.trailing_stop_bonus

            elif close_reason == CloseReason.AGENT_CLOSE:


                good_loss_cut_enabled = bool(getattr(cfg, "good_loss_cut_enabled", True))

                if net_pnl < 0 and good_loss_cut_enabled and mae > 0:


                    loss_cut_efficiency = 1.0 - (abs(net_pnl) / max(mae, abs(net_pnl), 1e-6))
                    loss_cut_efficiency = self._clamp(loss_cut_efficiency, 0.0, 1.0)

                    efficiency_threshold = float(getattr(cfg, "good_loss_cut_efficiency_threshold", 0.3))
                    if loss_cut_efficiency >= efficiency_threshold:

                        base_bonus = float(getattr(cfg, "good_loss_cut_bonus", 0.08))
                        max_bonus = float(getattr(cfg, "good_loss_cut_max_bonus", 0.15))

                        loss_cut_bonus = min(base_bonus * (1.0 + loss_cut_efficiency), max_bonus)
                        add("good_loss_cut_bonus", loss_cut_bonus)

                        exit_modifier = 0.0
                    else:

                        exit_modifier = cfg.agent_close_bonus
                elif net_pnl >= 0:

                    exit_modifier = cfg.agent_close_bonus


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
                else:
                    exit_modifier = cfg.agent_close_bonus

            elif close_reason == CloseReason.HARD_STOP:
                exit_modifier = -cfg.hard_stop_penalty

            elif close_reason == CloseReason.RISK_LIQUIDATION:
                exit_modifier = -cfg.risk_liquidation_penalty

            elif close_reason == CloseReason.EMERGENCY_CLOSE:
                exit_modifier = -(cfg.hard_stop_penalty * 1.2)

            if exit_modifier != 0.0:
                add("exit_quality", exit_modifier)


        if close_reason == CloseReason.EPISODE_TRUNCATE:
            if net_pnl > 0:
                trunc_discount = reward_components.get("base_pnl", 0.0) * cfg.truncation_winner_discount
                add("truncation_discount", -trunc_discount)
            else:
                add("truncation_penalty", -cfg.truncation_loser_extra_penalty)


        if cfg.entry_quality_integration:
            qdev = entry_quality - 0.5
            base_mag = abs(reward_components.get("base_pnl", 0.0))
            if net_pnl > 0 and qdev > 0:
                add("entry_quality_bonus", qdev * cfg.entry_quality_weight * base_mag)
            elif net_pnl < 0 and qdev < 0:
                add("entry_quality_penalty", -(abs(qdev) * cfg.entry_quality_weight * base_mag))


        if bool(getattr(cfg, "setup_quality_enabled", False)):
            setup_q = float(getattr(result, "setup_quality", 0.5))
            setup_thr = float(getattr(cfg, "setup_quality_threshold", 0.7))
            ss = shape_scale
            if setup_q >= setup_thr:
                bonus = (setup_q - setup_thr) * float(getattr(cfg, "setup_quality_bonus_scale", 0.0)) * ss
                if bonus != 0.0:
                    add("setup_quality_bonus", bonus)
            else:
                penalty = (setup_thr - setup_q) * float(getattr(cfg, "hasty_entry_penalty", 0.0)) * ss
                if penalty != 0.0:
                    add("hasty_entry_penalty", -penalty)


        certainty = float(getattr(result, "entry_certainty", 0.5))
        certainty_thr = float(getattr(cfg, "certainty_threshold", 0.7))
        if certainty >= certainty_thr:
            bonus_map = getattr(cfg, "entry_certainty_bonus", {}) or {}
            bonus_val = 0.0
            for k, v in bonus_map.items():
                try:
                    if isinstance(k, str) and "-" in k:
                        lo_s, hi_s = k.split("-", 1)
                        lo = float(lo_s.strip())
                        hi = float(hi_s.strip())
                    elif isinstance(k, (tuple, list)) and len(k) == 2:
                        lo, hi = float(k[0]), float(k[1])
                    else:
                        continue
                    if lo <= certainty < hi:
                        bonus_val = max(bonus_val, float(v))
                except Exception:
                    continue
            if bonus_val:
                add("entry_certainty_bonus", bonus_val * shape_scale)
        else:
            low_pen = float(getattr(cfg, "low_certainty_penalty", 0.0))
            if low_pen > 0.0 and certainty_thr > 0:
                frac = (certainty_thr - certainty) / certainty_thr
                add("low_certainty_penalty", -(low_pen * self._clamp(frac, 0.0, 1.0)))


        if cfg.session_timing_enabled:
            entry_dt = getattr(result, "entry_dt", None)
            if entry_dt is None:

                inst = getattr(result, "instrument", None)
                if inst is None and self.instruments:
                    inst = self.instruments[0]
                entry_dt = self._get_bar_dt(inst) if inst is not None else None

            if entry_dt is not None:
                tod_quality = None
                tod_map = getattr(cfg, "time_of_day_quality", {}) or {}
                if tod_map:
                    t = entry_dt.timetz().replace(tzinfo=None)
                    for k, v in tod_map.items():
                        try:
                            if not isinstance(k, str) or "-" not in k:
                                continue
                            s_raw, e_raw = k.split("-", 1)
                            sh, sm = s_raw.strip().split(":")
                            eh, em = e_raw.strip().split(":")
                            start = (int(sh), int(sm))
                            end = (int(eh), int(em))

                            in_range = False
                            if start == end:
                                in_range = False
                            elif start < end:
                                in_range = (t.hour, t.minute) >= start and (t.hour, t.minute) < end
                            else:
                                in_range = (t.hour, t.minute) >= start or (t.hour, t.minute) < end
                            if in_range:
                                tod_quality = float(v)
                                break
                        except Exception:
                            continue

                if tod_quality is not None:

                    if tod_quality >= 0.5:
                        bonus = ((tod_quality - 0.5) / 0.5) * cfg.prime_hours_trade_bonus
                        if bonus > 0.0:
                            add("time_of_day_bonus", bonus)
                    else:
                        penalty = ((0.5 - tod_quality) / 0.5) * cfg.off_hours_trade_penalty
                        if penalty > 0.0:
                            add("time_of_day_penalty", -penalty)
                else:
                    if self._in_no_new_trades_window(entry_dt):
                        add("off_hours_penalty", -cfg.off_hours_trade_penalty)
                    elif self._in_prime_window(entry_dt):
                        add("prime_hours_bonus", cfg.prime_hours_trade_bonus)


        if cfg.market_structure_enabled:
            entry_context = getattr(result, "entry_context", {}) or {}
            trade_direction = getattr(result, "direction", None)

            near_support = float(entry_context.get("near_support", 0.0))
            near_resistance = float(entry_context.get("near_resistance", 0.0))


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


        if cfg.regime_awareness_enabled:
            entry_context = getattr(result, "entry_context", {}) or {}
            risk_regime = entry_context.get("risk_regime", "neutral")
            vol_regime = entry_context.get("volatility_regime", "normal")

            ss = shape_scale

            if risk_regime == "risk_off" and net_pnl < 0:
                add("risk_off_penalty", -(cfg.risk_off_aggressive_penalty * ss))


            high_vol_loss_r_threshold = float(getattr(cfg, "high_vol_loss_r_threshold", float("nan")))
            if vol_regime == "high":
                if np.isfinite(high_vol_loss_r_threshold):
                    if r_multiple < high_vol_loss_r_threshold:
                        add("high_vol_penalty", -(cfg.high_vol_size_penalty * ss))
                else:
                    if net_pnl < -50:
                        add("high_vol_penalty", -(cfg.high_vol_size_penalty * ss))


        if cfg.streak_modifier_enabled:
            if net_pnl > 0:
                streak_bonus = min(self.consecutive_wins, 5) * cfg.win_streak_bonus_per_win
                if streak_bonus > 0:
                    add("win_streak_bonus", streak_bonus)
            else:


                n_losses = min(self.consecutive_losses, 5)

                streak_pen = cfg.loss_streak_penalty_per_loss * (n_losses ** 1.5)

                streak_pen_cap = float(getattr(cfg, "loss_streak_penalty_cap", 4.0))
                streak_pen = min(streak_pen, streak_pen_cap)
                if streak_pen > 0:
                    add("loss_streak_penalty", -streak_pen)


        if bool(getattr(cfg, "deliberation_time_tracking", False)) and net_pnl > 0:
            opt_range = getattr(cfg, "optimal_deliberation_range", (0, 0))
            try:
                lo = int(opt_range[0])
                hi = int(opt_range[1])
            except Exception:
                lo, hi = 0, 0
            if hi > 0:
                delib = int(getattr(result, "deliberation_bars", 0))
                if lo <= delib <= hi:
                    add("deliberation_bonus", float(getattr(cfg, "deliberation_quality_bonus", 0.0)))


        if bool(getattr(cfg, "compounding_success_enabled", False)):
            quality_r = float(getattr(cfg, "quality_trade_r_multiple", 1.0))
            quality_eq = float(getattr(cfg, "quality_trade_entry_quality", 0.6))
            quality_exit = str(getattr(cfg, "quality_trade_exit_type", "trailing_stop"))
            is_quality_trade = (
                r_multiple >= quality_r
                and float(getattr(result, "entry_quality", 0.0)) >= quality_eq
                and str(result.close_reason.value) == quality_exit
            )
            streak = int(getattr(self, "_quality_trade_streak", 0))
            if is_quality_trade:
                streak += 1
            else:
                streak = 0
            self._quality_trade_streak = streak
            bonus_list = list(getattr(cfg, "consecutive_quality_trades_bonus", []) or [])
            if is_quality_trade and bonus_list:
                idx = min(streak, len(bonus_list) - 1)
                bonus = float(bonus_list[idx])
                if bonus:
                    add("compounding_success_bonus", bonus)


        if cfg.anti_churn_enabled and self.daily_trades > cfg.daily_trade_soft_limit:
            excess = self.daily_trades - cfg.daily_trade_soft_limit
            excess_factor = min(1.5 ** min(excess, 8) - 1, 25.0)
            churn_pen = excess_factor * cfg.churn_penalty_per_trade
            add("churn_penalty", -churn_pen)


        cost_erosion_enabled = bool(getattr(cfg, "cost_erosion_penalty_enabled", True))


        episode_gross = float(getattr(self, "_episode_gross_profit", 0.0))
        episode_costs = float(getattr(self, "_episode_total_costs", 0.0))
        self._episode_total_costs = episode_costs + max(float(total_fees), 0.0)


        if gross_pnl > 0:
            self._episode_gross_profit = episode_gross + float(gross_pnl)

        if cost_erosion_enabled:


            updated_gross = getattr(self, "_episode_gross_profit", 0.0)
            updated_costs = getattr(self, "_episode_total_costs", 0.0)

            if updated_gross > 0:
                cost_ratio = updated_costs / updated_gross
                erosion_threshold = float(getattr(cfg, "cost_erosion_threshold", 0.5))

                if cost_ratio > erosion_threshold:

                    erosion_scale = float(getattr(cfg, "cost_erosion_penalty_scale", 0.15))
                    erosion_cap = float(getattr(cfg, "cost_erosion_penalty_cap", 0.30))
                    erosion_penalty = min((cost_ratio - erosion_threshold) * erosion_scale, erosion_cap)
                    add("cost_erosion_penalty", -erosion_penalty)


        if cfg.dd_shaping_enabled and current_dd > cfg.dd_threshold:
            denom = float(self.config.max_drawdown_limit - cfg.dd_threshold)
            if denom > 1e-9:
                dd_ratio = (current_dd - cfg.dd_threshold) / denom
                dd_ratio = float(np.clip(dd_ratio, 0.0, cfg.dd_severity_cap))
                dd_severity = dd_ratio ** cfg.dd_severity_exponent
                dd_pen = dd_severity * cfg.dd_penalty_scale * 0.5
                add("dd_shaping", -dd_pen)


        max_shaping_ratio = float(getattr(cfg, "max_shaping_to_pnl_ratio", 0.5))
        if max_shaping_ratio > 0 and base_pnl_magnitude > 0:

            non_pnl_keys = [k for k in reward_components.keys()
                          if k not in ("base_pnl", "execution_costs")]

            shaping_mass = sum(abs(reward_components.get(k, 0.0)) for k in non_pnl_keys)

            max_shaping = base_pnl_magnitude * max_shaping_ratio

            if shaping_mass > max_shaping:

                scale_factor = max_shaping / shaping_mass
                shaping_adjustment = 0.0

                for k in non_pnl_keys:
                    old_val = reward_components.get(k, 0.0)
                    new_val = old_val * scale_factor
                    adjustment = new_val - old_val
                    reward_components[k] = new_val
                    shaping_adjustment += adjustment


                reward += shaping_adjustment


                reward_components["shaping_cap_applied"] = shaping_adjustment


        reward = float(np.clip(reward, cfg.min_reward, cfg.max_reward))
        self._last_reward_components = reward_components


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
