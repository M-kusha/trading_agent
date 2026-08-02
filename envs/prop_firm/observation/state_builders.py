
# pyright: reportAttributeAccessIssue=false

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

if TYPE_CHECKING:
    from envs.core.env_types import PropFirmConfig, PropPosition


class ObservationBuildersMixin:

    @staticmethod
    def _dir_sign(direction_str: str) -> float:
        d = str(direction_str).lower().strip()
        if d in ("long", "buy", "bull", "bullish", "up"):
            return 1.0
        elif d in ("short", "sell", "bear", "bearish", "down"):
            return -1.0
        return 0.0

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        if len(close) < period + 1:
            return 50.0
        deltas = np.diff(close[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss < 1e-10:
            return 100.0 if avg_gain > 0 else 50.0
        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    config: "PropFirmConfig"
    total_trades: int
    winning_trades: int
    total_pnl: float
    consecutive_losses: int
    balance: float
    equity: float
    daily_trades: int
    position: Optional["PropPosition"]
    episode_bars: int
    _episode_trade_results: List


    def _prepare_committee_state(self, expert_signals: Dict[str, Any]) -> Dict[str, Any]:
        experts = expert_signals.get("experts", {}) if isinstance(expert_signals, dict) else {}
        dirs: List[float] = []
        wts: List[float] = []
        for _, sig in (experts.items() if isinstance(experts, dict) else []):
            if not isinstance(sig, dict):
                continue
            d = self._dir_sign(str(sig.get("direction", "neutral")))
            s = float(sig.get("score", 0.0) or 0.0)
            c = float(sig.get("confidence", 0.5) or 0.5)
            dirs.append(d * s)
            wts.append(c)

        signed_score = float(sum(d * w for d, w in zip(dirs, wts)) / max(sum(wts), 1e-8)) if wts else 0.0
        action_str = "long" if signed_score > 0.1 else ("short" if signed_score < -0.1 else "flat")

        nonzero_signs = [int(np.sign(d)) for d in dirs if abs(d) > 0.1]

        if not nonzero_signs:
            agreement = 0.5
        else:
            agreement = 1.0 if len(set(nonzero_signs)) == 1 else 0.0

        return {
            "consensus_score": float(np.clip(abs(signed_score), 0.0, 1.0)),
            "score": float(np.clip(signed_score, -1.0, 1.0)),
            "action": action_str,
            "action_value": float(np.clip(signed_score, -1.0, 1.0)),
            "direction": action_str,
            "confidence": float(np.mean(wts)) if wts else 0.5,
            "agreement": float(agreement),
            "fragility": float(1.0 - agreement),
        }

    def _prepare_risk_state(self) -> Dict[str, Any]:
        current_dd, daily_dd = self._calc_dds()


        exposure = 0.0
        if self.position is not None:
            risk_eur = float(self.position.lot_size) * float(self.config.hard_stop_loss_eur)
            exposure = risk_eur / max(float(self.equity), 1.0)


        daily_limit = max(float(self.config.daily_drawdown_limit), 1e-9)
        risk_budget = 1.0 - (float(daily_dd) / daily_limit)

        return {
            "current_drawdown": float(current_dd),
            "daily_drawdown": float(daily_dd),
            "max_drawdown_limit": float(self.config.max_drawdown_limit),
            "daily_drawdown_limit": float(self.config.daily_drawdown_limit),
            "trades_today": int(self.daily_trades),
            "max_trades_per_day": int(self.config.max_trades_per_day),
            "risk_per_trade": float(self.config.risk_per_trade_pct),
            "portfolio_risk": {
                "total_exposure": float(np.clip(exposure, 0.0, 1.0)),
            },
            "risk_budget": float(np.clip(risk_budget, 0.0, 1.0)),
        }


    def _prepare_account_state(self, instrument: str) -> Dict[str, Any]:
        cur_dd, _ = self._calc_dds()


        initial = max(float(self.config.initial_balance), 1.0)
        episode_return = (float(self.equity) - initial) / initial * 100.0


        pnl_trend = 0.0
        if self._episode_trade_results:
            recent = self._episode_trade_results[-5:]
            wins = sum(1 for r in recent if float(getattr(r, "net_pnl", 0.0)) > 0)
            pnl_trend = (2.0 * wins / len(recent)) - 1.0

        state: Dict[str, Any] = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "initial_balance": float(self.config.initial_balance),
            "current_drawdown": float(cur_dd),
            "current_step": int(self.episode_step),
            "max_steps": int(getattr(self.config, "max_steps_per_episode", 2000) or 2000),
            "episode_return": float(episode_return),
            "pnl_trend": float(np.clip(pnl_trend, -1.0, 1.0)),
            "win_rate": float(self.winning_trades / max(self.total_trades, 1)),
            "trades_today": int(self.daily_trades),
            "has_position": self.position is not None,
            "position_direction": 0.0,
            "position_size": 0.0,
            "unrealized_pnl": 0.0,
            "time_in_position": 0.0,
            "on_cooldown": 0.0,
        }

        if self.position is not None:
            bid, ask = self._get_step_bid_ask(instrument, self._get_price_mid(instrument), self._atr_vol_proxy(instrument))
            u = self._mark_unrealized_pnl_from_bid_ask(self.position, bid, ask)
            state["position_direction"] = 1.0 if self.position.direction == "long" else -1.0
            state["position_size"] = float(self.position.lot_size)
            state["unrealized_pnl"] = float(u)
            state["time_in_position"] = float(self.episode_bars - self.position.entry_bar)

        inst_dt = self._get_bar_dt(instrument)
        if inst_dt is not None and self._last_loss_dt is not None:
            mins = (inst_dt - self._last_loss_dt).total_seconds() / 60.0
            tfm = max(1, self._tf_minutes())
            min_bars_after = int(getattr(self.config, "min_bars_after_loss", 0) or 0)
            min_mins_after = float(getattr(self.config, "min_minutes_after_loss", 0) or 0.0)
            effective_min_minutes = max(min_mins_after, min_bars_after * tfm)
            if mins < effective_min_minutes:
                state["on_cooldown"] = 1.0
        elif inst_dt is None and self._last_loss_step is not None:
            tfm = max(1, self._tf_minutes())
            min_bars_after = int(getattr(self.config, "min_bars_after_loss", 0) or 0)
            min_mins_after = float(getattr(self.config, "min_minutes_after_loss", 0) or 0.0)
            base_after_minutes = max(min_mins_after, min_bars_after * tfm)
            post_loss_bars = max(1, min_bars_after, int(np.ceil(base_after_minutes / tfm)))
            if (int(self.current_step) - int(self._last_loss_step)) < post_loss_bars:
                state["on_cooldown"] = 1.0

        return state

    def _get_governor_state(self) -> Dict[str, float]:

        loss_layer_stop = getattr(self.config, "loss_layer_stop", 5)
        session_loss_limit = getattr(self.config, "session_loss_limit_pct", 0.99)
        session_consec_limit = getattr(self.config, "session_consecutive_loss_limit", 99)
        max_session_trades = getattr(self.config, "max_trades_per_session", 99)


        consecutive_losses = getattr(self, "consecutive_losses", 0)
        consecutive_wins = getattr(self, "consecutive_wins", 0)
        loss_layer = min(consecutive_losses, 5)


        session_pnl = getattr(self, "session_pnl", 0.0)
        session_start_balance = getattr(self, "session_start_balance", 0.0)
        session_consecutive_losses = getattr(self, "session_consecutive_losses", 0)
        session_trades = getattr(self, "_session_trades", 0)
        session_start_step = getattr(self, "session_start_step", 0)
        current_step = getattr(self, "current_step", 0)


        if session_start_balance > 0:
            session_pnl_pct = session_pnl / session_start_balance
        else:
            session_pnl_pct = 0.0

        headroom = (session_loss_limit + session_pnl_pct) / max(session_loss_limit, 0.001)


        bars_per_day = getattr(self, "_bars_per_day", lambda: 96)()
        session_duration_bars = max(bars_per_day // 3, 1)
        session_progress = (current_step - session_start_step) / max(session_duration_bars, 1)


        pending_entry = getattr(self, "pending_entry", None)
        pending_exit = getattr(self, "pending_exit", None)
        episode_latency = getattr(self, "_episode_latency_bars", 1)
        max_latency = max(episode_latency, 1)

        pending_bars = 0
        if pending_entry and isinstance(pending_entry, dict):
            fill_step = pending_entry.get("fill_step", current_step)
            pending_bars = max(0, fill_step - current_step)
        elif pending_exit and isinstance(pending_exit, dict):
            fill_step = pending_exit.get("fill_step", current_step)
            pending_bars = max(0, fill_step - current_step)

        return {
            "loss_layer_ratio": float(np.clip(consecutive_losses / max(loss_layer_stop, 1), 0.0, 1.0)),
            "loss_layer_level": float(np.clip(loss_layer / 5.0, 0.0, 1.0)),
            "win_streak_ratio": float(np.clip(consecutive_wins / 5.0, 0.0, 1.0)),
            "session_pnl_headroom": float(np.clip(headroom, 0.0, 2.0)),
            "session_trade_budget": float(np.clip(1.0 - (session_trades / max(max_session_trades, 1)), 0.0, 1.0)),
            "session_consec_loss_ratio": float(np.clip(session_consecutive_losses / max(session_consec_limit, 1), 0.0, 1.0)),
            "session_progress": float(np.clip(session_progress, 0.0, 1.0)),
            "pending_order_progress": float(np.clip(pending_bars / max_latency, 0.0, 1.0) if pending_bars > 0 else 0.0),
        }

    def _prepare_trading_mode_state(self, instrument: str) -> Dict[str, Any]:
        cur_dd, _ = self._calc_dds()
        if cur_dd > 0.05:
            mode = "safe"
        elif cur_dd < 0.02 and self.total_pnl > 0:
            mode = "aggressive"
        else:
            mode = "normal"

        dt = self._get_bar_dt(instrument)
        can_trade = True
        if dt is not None:
            if self._is_weekend(dt) and not self.config.allow_weekend_holding:
                can_trade = False
            if self._in_no_new_trades_window(dt):
                can_trade = False
            if self._in_final_exit_window(dt):
                can_trade = False
            if self._at_or_after_hard_close(dt):
                can_trade = False

        vol_proxy = self._atr_vol_proxy(instrument)
        vol_state = "low" if vol_proxy < 0.3 else ("high" if vol_proxy > 0.7 else "normal")
        zone_type = "good" if vol_state == "normal" else ("bad" if vol_state == "high" else "hot")


        q_long = self._get_step_entry_quality(instrument, "long")
        q_short = self._get_step_entry_quality(instrument, "short")
        cert_long = self._get_step_entry_certainty(instrument, "long")
        cert_short = self._get_step_entry_certainty(instrument, "short")
        setup_long, conf_long = self._get_step_setup_quality(instrument, "long")
        setup_short, conf_short = self._get_step_setup_quality(instrument, "short")

        confluence_count = int(max(conf_long, conf_short))
        bars_since_setup = int(getattr(self, "_bars_since_last_setup", 0) or 0)
        setup_quality_trend = float(getattr(self, "_setup_quality_trend", 0.0) or 0.0)
        confluence_increasing = 1.0 if bool(getattr(self, "_confluence_increasing", False)) else 0.0

        return {
            "trading_mode": mode,
            "regime_stability": float(1.0 - min(vol_proxy, 1.0)),
            "theme_transition": float(min(vol_proxy, 1.0)),
            "theme_strength": float(1.0 - abs(0.5 - min(vol_proxy, 1.0)) * 2.0),
            "regime_accuracy": 0.5,
            "risk_scaling_factor": float(1.0 + 0.5 * min(vol_proxy, 1.0)),
            "liquidity_score": float(1.0 - 0.5 * min(vol_proxy, 1.0)),
            "entry_timing": {
                "entry_allowed": bool(can_trade),
                "entry_quality_long": float(q_long),
                "entry_quality_short": float(q_short),
                "entry_certainty_long": float(cert_long),
                "entry_certainty_short": float(cert_short),
                "setup_quality_long": float(setup_long),
                "setup_quality_short": float(setup_short),
                "confluence_count": int(confluence_count),
                "bars_since_setup": int(bars_since_setup),
                "setup_quality_trend": float(setup_quality_trend),
                "confluence_increasing": float(confluence_increasing),
                "zone_type": zone_type,
                "vol_state": vol_state,
                "hour_normalized": float(dt.hour / 24.0) if dt else 0.5,
                "in_prime_window": 1.0 if (dt and self._in_prime_window(dt)) else 0.0,
            },
            "mode_stats": {"mode_effectiveness": 0.5},
        }


