# envs/prop_firm/rewards/shaping.py
# pyright: reportAttributeAccessIssue=false
"""
Reward shaping mixin for PropFirmTradingEnv.

Contains per-step shaping and blocked action penalty methods.

Upgrades:
- Reason-aware blocked penalties (hard vs soft blocks get different magnitudes)
- Soft blocks scale with entry quality (good setups get less penalty when blocked by soft rules)
- Optional anti-churn shaping: small cost for "doing too much" without holding a position
- Optional patience shaping: small positive shaping for NOT trading when both entry qualities are poor
- All shaping is bounded and designed to remain subordinate to trade-close rewards
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from envs.core.env_types import PropFirmConfig


class RewardShapingMixin:
    """Mixin providing reward shaping methods.

    Expected attributes from PropFirmTradingEnv:
    - config: PropFirmConfig
    """

    config: "PropFirmConfig"

    # ---------------------------
    # Blocked entry penalty
    # ---------------------------

    def _compute_blocked_action_penalty(self, block_reason: str, entry_quality: float, is_hard_block: bool) -> float:
        """
        Compute penalty for blocked entry action.

        Design:
        - Hard blocks: deterministic strong penalty (agent attempted an illegal/unsafe action).
        - Soft blocks: smaller penalty, scaled down when entry_quality is high (avoid punishing good intent).
        - Reason-aware multipliers: not all blocks are equally "bad behavior".

        Returns:
            Penalty value (positive magnitude). Caller typically subtracts it.
        """
        cfg = self.config.reward

        hard_base = float(getattr(cfg, "hard_block_penalty", 0.10))
        soft_base = float(getattr(cfg, "soft_block_penalty", 0.03))

        # Normalize and clamp quality to [0,1]
        q = float(entry_quality)
        if q != q:  # NaN guard
            q = 0.5
        q = max(0.0, min(1.0, q))

        reason = (block_reason or "").strip().lower()

        # Multipliers: tune these to reflect "severity" of the attempted violation.
        # Strongest: risk / hard-close / drawdown headroom / max-losses.
        # Medium: session/day limits.
        # Mild: spacing / cooldown / quality gate.
        HARD_MULTIPLIERS = {
            "hard_close": 1.40,
            "final_exit_window": 1.20,
            "weekend_block": 1.20,
            "drawdown_headroom": 1.50,
            "max_consecutive_losses": 1.40,
            "max_trades_per_day": 1.10,
            "max_trades_per_session": 1.05,
            "insufficient_bars_for_fill": 1.15,
            # min_entry_spacing is hard-blocked in env.step(), classify consistently
            "min_entry_spacing": 0.90,
            # Governor hard-stop: NOT "bad behavior" - it's a safety mode.
            # Keep at 0 to avoid negative-reward spirals during stop-trading state.
            "loss_layer_stop": 0.0,
        }
        SOFT_MULTIPLIERS = {
            "post_loss_cooldown": 0.95,
            "no_new_trades_window": 1.00,   # this is policy-like but still "soft" in your setup
            "entry_quality_gate": 0.70,     # don't punish too hard: you're already teaching quality
        }

        if is_hard_block:
            # Loss-layer stop is a governor state, not misbehavior - zero penalty
            if reason == "loss_layer_stop":
                stop_pen = float(getattr(cfg, "loss_layer_stop_penalty", 0.0))
                return float(max(0.0, stop_pen))
            mult = HARD_MULTIPLIERS.get(reason, 1.0)
            penalty = hard_base * float(mult)
            return float(max(0.0, penalty))

        # Soft block: scale by (1 - 0.6*q) so higher quality reduces penalty
        mult = SOFT_MULTIPLIERS.get(reason, 1.0)
        penalty = soft_base * float(mult) * (1.0 - 0.60 * q)
        return float(max(0.0, penalty))

    # ---------------------------
    # Per-step shaping
    # ---------------------------

    def _compute_per_step_shaping(
        self,
        has_position: bool,
        bars_in_position: int,
        entry_quality_long: float,
        entry_quality_short: float,
        entry_accepted: bool,
    ) -> float:
        """
        Compute per-step reward shaping.

        Principles:
        - Keep shaping small and bounded.
        - Encourage: not churning, holding cost realism.
        - Encourage patience: don't enter when both entry qualities are low.
        - Encourage exploration: bonus for ACCEPTED entries only (pending_entry created).
        - Avoid: rewarding "do nothing" forever.
        
        CRITICAL: entry_accepted must be True ONLY when a pending_entry was actually
        created this step. This prevents exploration bonus farming by spamming
        blocked entry attempts.

        Returns:
            Shaping reward value (can be negative or small positive).
        """
        cfg = self.config.reward
        if not bool(getattr(cfg, "per_step_shaping_enabled", False)):
            # Even if per_step_shaping is disabled, still apply exploration bonus
            # This is critical for early-stage trade encouragement
            exploration_bonus = float(getattr(cfg, "exploration_bonus", 0.0))
            if exploration_bonus > 0.0 and entry_accepted:
                # Bonus ONLY for accepted entries (pending_entry created)
                return exploration_bonus
            return 0.0

        shaping = 0.0

        # 0) Exploration bonus: encourage trading in early stages
        # Applied ONLY when entry is accepted (pending_entry created)
        exploration_bonus = float(getattr(cfg, "exploration_bonus", 0.0))
        if exploration_bonus > 0.0 and entry_accepted:
            shaping += exploration_bonus

        # 1) Holding cost (your original behavior)
        holding_cost = float(getattr(cfg, "holding_cost_per_bar", 0.0))
        if has_position and int(bars_in_position) > 0 and holding_cost > 0.0:
            shaping -= holding_cost

        # Normalize qualities
        ql = float(entry_quality_long)
        qs = float(entry_quality_short)
        if ql != ql:
            ql = 0.5
        if qs != qs:
            qs = 0.5
        ql = max(0.0, min(1.0, ql))
        qs = max(0.0, min(1.0, qs))
        q_best = max(ql, qs)

        # 2) Optional anti-churn friction:
        # Small cost for accepting entries (encourages selectivity).
        # This should be tiny to avoid training collapse.
        churn_enabled = bool(getattr(cfg, "anti_churn_enabled", False))
        churn_cost = float(getattr(cfg, "churn_action_cost", 0.0))
        if churn_enabled and (not has_position) and entry_accepted and churn_cost > 0.0:
            # Reduce cost when there is a genuinely good setup visible
            # (agent is at least "trying" at the right times).
            # If q_best=1.0 -> cost is reduced by 60%.
            shaping -= churn_cost * (1.0 - 0.60 * q_best)

        # 3) Optional patience shaping:
        # If both entry qualities are poor, slightly reward staying flat AND not entering.
        # This teaches "do nothing is a decision" without paying too much.
        patience_enabled = bool(getattr(cfg, "patience_shaping_enabled", False))
        patience_bonus = float(getattr(cfg, "patience_bonus_per_bar", 0.0))
        patience_threshold = float(getattr(cfg, "patience_quality_threshold", 0.35))
        if patience_enabled and (not has_position) and (not entry_accepted) and patience_bonus > 0.0:
            if q_best < patience_threshold:
                shaping += patience_bonus

        # 3b) Loss streak caution penalty (CRITICAL for consecutive loss control):
        # Penalize ENTRY ATTEMPTS when agent is on a consecutive loss streak.
        # This teaches the agent to STOP trading when tilted, not push through.
        # Only triggers when a new entry is accepted (pending_entry created).
        loss_streak_caution_enabled = bool(getattr(cfg, "loss_streak_caution_enabled", True))
        if loss_streak_caution_enabled and entry_accepted:
            consecutive_losses = int(getattr(self, "consecutive_losses", 0))
            if consecutive_losses >= 2:
                # Escalating penalty for entering while on loss streak
                # Loss 2: -0.03, Loss 3: -0.07, Loss 4+: capped at -0.25
                caution_base = float(getattr(cfg, "loss_streak_caution_base", 0.03))
                caution_penalty = caution_base * (consecutive_losses - 1) ** 1.5
                caution_cap = float(getattr(cfg, "loss_streak_caution_cap", 0.25))
                shaping -= min(caution_penalty, caution_cap)

        # 4) Bound shaping so it cannot dominate
        # Default bounds are conservative; if RewardConfig has explicit bounds, use them.
        min_s = float(getattr(cfg, "per_step_min", -0.05))
        max_s = float(getattr(cfg, "per_step_max", 0.05))
        if min_s > max_s:
            min_s, max_s = -0.05, 0.05
        shaping = max(min_s, min(max_s, shaping))

        return float(shaping)

    # ---------------------------
    # Governor approaching-limit penalties (v5.5)
    # ---------------------------

    def _compute_governor_approaching_penalties(self) -> float:
        """
        Compute gradient signal penalties when approaching hard blocks.

        This provides learning signal BEFORE hard blocks engage, teaching the agent
        to anticipate and avoid limit-hitting situations rather than learning only
        from blocked actions.

        Penalties are progressive - they increase as the agent gets closer to limits:
        - Loss layer: penalty scales with consecutive_losses / loss_layer_stop
        - Session PnL: penalty when session_pnl approaches session_loss_limit
        - Session consecutive losses: penalty as approaching session limit

        Design philosophy (per GPT feedback):
        - Gradient signal BEFORE hard blocks (not after)
        - Smooth scaling (no cliffs)
        - Conservative magnitudes (subordinate to trade rewards)

        Returns:
            Negative penalty value (0.0 if not approaching any limits).
        """
        cfg = self.config.reward

        # Config flags
        governor_penalty_enabled = bool(getattr(cfg, "governor_approaching_penalty_enabled", True))
        if not governor_penalty_enabled:
            return 0.0

        penalty = 0.0

        # Base penalty magnitude (small - should be subordinate to trade rewards)
        base_penalty = float(getattr(cfg, "governor_approaching_base_penalty", 0.02))
        
        # Threshold at which penalty starts (as fraction of limit)
        approach_threshold = float(getattr(cfg, "governor_approach_threshold", 0.5))

        # 1) Loss layer approaching penalty
        # Penalty scales as consecutive_losses / loss_layer_stop approaches 1.0
        loss_layer_stop = int(getattr(self.config, "loss_layer_stop", 4))
        if loss_layer_stop > 0:
            consecutive_losses = int(getattr(self, "consecutive_losses", 0))
            loss_ratio = consecutive_losses / loss_layer_stop
            
            if loss_ratio > approach_threshold:
                # Scale penalty from 0 at threshold to full at limit
                progress = (loss_ratio - approach_threshold) / (1.0 - approach_threshold)
                progress = min(1.0, max(0.0, progress))
                # Quadratic scaling for stronger signal near limit
                penalty -= base_penalty * (progress ** 2) * 2.0

        # 2) Session loss limit approaching penalty
        session_loss_limit = float(getattr(self.config, "session_loss_limit_pct", 0.99))
        if session_loss_limit < 0.99:  # Only apply if limit is actually constraining
            session_start_balance = float(getattr(self, "session_start_balance", 0.0))
            session_pnl = float(getattr(self, "session_pnl", 0.0))
            
            if session_start_balance > 0:
                session_pnl_pct = session_pnl / session_start_balance
                # Negative pnl_pct is bad, approaching -session_loss_limit
                if session_pnl_pct < 0:
                    loss_ratio = abs(session_pnl_pct) / session_loss_limit
                    
                    if loss_ratio > approach_threshold:
                        progress = (loss_ratio - approach_threshold) / (1.0 - approach_threshold)
                        progress = min(1.0, max(0.0, progress))
                        penalty -= base_penalty * (progress ** 2) * 1.5

        # 3) Session consecutive loss limit approaching penalty
        session_consec_limit = int(getattr(self.config, "session_consecutive_loss_limit", 99))
        if session_consec_limit < 99:  # Only apply if limit is actually constraining
            session_consecutive_losses = int(getattr(self, "session_consecutive_losses", 0))
            
            if session_consec_limit > 0:
                consec_ratio = session_consecutive_losses / session_consec_limit
                
                if consec_ratio > approach_threshold:
                    progress = (consec_ratio - approach_threshold) / (1.0 - approach_threshold)
                    progress = min(1.0, max(0.0, progress))
                    penalty -= base_penalty * (progress ** 2) * 1.5

        # Bound total penalty to avoid dominating rewards
        max_penalty = float(getattr(cfg, "governor_max_approaching_penalty", 0.10))
        penalty = max(-max_penalty, penalty)

        return float(penalty)
