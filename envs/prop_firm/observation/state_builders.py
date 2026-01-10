# envs/prop_firm/observation/state_builders.py
# pyright: reportAttributeAccessIssue=false
"""
Observation state builder mixin for PropFirmTradingEnv.

Contains methods for preparing state dictionaries for observation building.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from envs.core.env_types import PropFirmConfig, PropPosition


class ObservationBuildersMixin:
    """Mixin providing observation state building methods.
    
    Expected attributes from PropFirmTradingEnv:
    - config: PropFirmConfig
    - total_trades: int
    - winning_trades: int
    - total_pnl: float
    - consecutive_losses: int
    - balance: float
    - equity: float
    - daily_trades: int
    - position: Optional[PropPosition]
    - episode_bars: int
    - _episode_trade_results: List
    """

    @staticmethod
    def _dir_sign(direction_str: str) -> float:
        """Convert direction string to numerical sign: +1 (long/bull), -1 (short/bear), 0 (neutral)."""
        d = str(direction_str).lower().strip()
        if d in ("long", "buy", "bull", "bullish", "up"):
            return 1.0
        elif d in ("short", "sell", "bear", "bearish", "down"):
            return -1.0
        return 0.0

    @staticmethod
    def _compute_rsi(close: np.ndarray, period: int = 14) -> float:
        """Compute RSI from close prices."""
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
    # Type hints for attributes provided by PropFirmTradingEnv
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
        """Prepare committee consensus state from expert signals."""
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
        # When no meaningful signals exist, use 0.5 (uncertain), not 1.0 (full agreement)
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
        """Prepare risk state for observation."""
        current_dd, daily_dd = self._calc_dds()
        return {
            "current_drawdown": float(current_dd),
            "daily_drawdown": float(daily_dd),
            "max_drawdown_limit": float(self.config.max_drawdown_limit),
            "daily_drawdown_limit": float(self.config.daily_drawdown_limit),
            "trades_today": int(self.daily_trades),
            "max_trades_per_day": int(self.config.max_trades_per_day),
            "risk_per_trade": float(self.config.risk_per_trade_pct),
        }

    def _prepare_memory_state(self, instrument: str) -> Dict[str, Any]:
        """Prepare memory/performance state for observation."""
        recent_pnl = float(self.total_pnl)
        recent_trades = int(self.total_trades)
        recent_losses = int(self.total_trades - self.winning_trades)

        if recent_trades == 0:
            memory_gate = 1.0
        else:
            loss_ratio = recent_losses / max(recent_trades, 1)
            memory_gate = float(np.clip(1.0 - loss_ratio * 0.5, 0.3, 1.0))

            cur_dd, _ = self._calc_dds()
            if cur_dd > 0.03:
                memory_gate *= 0.8
            if cur_dd > 0.05:
                memory_gate *= 0.7

        danger_zone_count = 0
        if self._episode_trade_results:
            recent_losses_list = [r for r in self._episode_trade_results[-10:] if r.net_pnl < 0]
            danger_zone_count = len(recent_losses_list)

        return {
            "memory_gate": float(memory_gate),
            "risk_multiplier": float(memory_gate),
            "danger_zones": {"zone_count": int(danger_zone_count), "active": bool(danger_zone_count > 0)},
            "recent_performance": {
                "win_rate": float(self.winning_trades / max(self.total_trades, 1)),
                "total_pnl": float(recent_pnl),
                "consecutive_losses": int(self.consecutive_losses),
            },
        }

    def _prepare_account_state(self, instrument: str) -> Dict[str, Any]:
        """Prepare account state for observation."""
        cur_dd, _ = self._calc_dds()
        state: Dict[str, Any] = {
            "balance": float(self.balance),
            "equity": float(self.equity),
            "initial_balance": float(self.config.initial_balance),
            "current_drawdown": float(cur_dd),
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
            if mins < self.config.min_minutes_after_loss:
                state["on_cooldown"] = 1.0
        elif inst_dt is None and self._last_loss_step is not None:
            tfm = max(1, self._tf_minutes())
            post_loss_bars = int(np.ceil(self.config.min_minutes_after_loss / tfm))
            if (int(self.current_step) - int(self._last_loss_step)) < max(1, post_loss_bars):
                state["on_cooldown"] = 1.0

        return state

    def _prepare_trading_mode_state(self, instrument: str) -> Dict[str, Any]:
        """Prepare trading mode state for observation."""
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

        # Use step cache if available (called from step()), fallback to direct compute
        q_long = self._get_step_entry_quality(instrument, "long")
        q_short = self._get_step_entry_quality(instrument, "short")

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
                "zone_type": zone_type,
                "vol_state": vol_state,
                "hour_normalized": float(dt.hour / 24.0) if dt else 0.5,
                "in_prime_window": 1.0 if (dt and self._in_prime_window(dt)) else 0.0,
            },
            "mode_stats": {"mode_effectiveness": 0.5},
        }

    def _prepare_world_model_state(self, instrument: str, expert_signals: Dict[str, Any], committee_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare world model predictions state for observation."""
        o = self._get_ohlcv(instrument, lookback=120)
        if not o or len(o.get("close", [])) < 60:
            return self._default_world_model_state()

        close = np.asarray(o["close"], dtype=np.float64)
        high = np.asarray(o["high"], dtype=np.float64)
        low = np.asarray(o["low"], dtype=np.float64)

        period = min(14, len(close) - 1)
        if period >= 2:
            hl = high[-period:] - low[-period:]
            hc = np.abs(high[-period:] - close[-period-1:-1])
            lc = np.abs(low[-period:] - close[-period-1:-1])
            atr = float(np.mean(np.maximum(hl, np.maximum(hc, lc))))
        else:
            atr = float(np.std(close[-20:]))

        price = float(close[-1])
        norm_atr = atr / max(price, 1.0)

        price_changes = []
        for lookback in (5, 10, 20, 40):
            if len(close) > lookback:
                ret = np.log(close[-1] / close[-lookback-1])
                normalized = (ret / (norm_atr + 1e-8)) * 0.1
                scaled = float(np.clip(normalized * 10.0, -1.0, 1.0))
            else:
                scaled = 0.0
            price_changes.append(scaled)

        vol_base = float(np.clip(norm_atr * 50.0, 0.0, 1.0))
        volatility_predictions = [vol_base, vol_base * 0.92, vol_base * 0.85, vol_base * 0.78]

        fast = float(np.mean(close[-10:]))
        slow = float(np.mean(close[-30:])) if len(close) >= 30 else fast
        trend = (fast - slow) / (atr + 1e-8)

        returns = np.diff(close[-20:]) / close[-20:-1]
        vol = float(np.std(returns))
        vol_norm = vol / 0.01

        probs = np.array([0.2, 0.2, 0.4, 0.2])

        if vol_norm > 1.5:
            probs[3] += 0.3
            probs[2] -= 0.15
            probs[0] -= 0.075
            probs[1] -= 0.075

        if abs(trend) > 0.5:
            if trend > 0:
                probs[0] += 0.25
                probs[1] -= 0.1
            else:
                probs[1] += 0.25
                probs[0] -= 0.1
            probs[2] -= 0.15

        probs = np.clip(probs, 0.05, 0.8)
        probs = probs / probs.sum()
        predicted_regime = int(np.argmax(probs))

        signs = [np.sign(pc) for pc in price_changes if abs(pc) > 0.05]
        trend_agreement = abs(sum(signs)) / len(signs) if signs else 0.5

        vol_penalty = min(norm_atr * 30, 0.3)

        comm_conf = float(committee_state.get("confidence", 0.5))
        comm_agree = float(committee_state.get("agreement", 0.5))

        rsi = self._compute_rsi(close)
        rsi_penalty = abs(rsi - 50) / 100.0

        confidence = float(np.clip(
            0.25 * trend_agreement +
            0.25 * (1.0 - vol_penalty) +
            0.20 * comm_conf +
            0.15 * comm_agree +
            0.15 * (1.0 - rsi_penalty),
            0.2, 0.9
        ))

        weights = [0.4, 0.3, 0.2, 0.1]
        weighted_mom = sum(w * pc for w, pc in zip(weights, price_changes))
        bullish_prob = float(np.clip(0.5 + weighted_mom * 0.3, 0.1, 0.9))

        if len(close) >= 60:
            short_ret = np.diff(close[-15:]) / close[-15:-1]
            long_ret = np.diff(close[-60:]) / close[-60:-1]
            short_vol = float(np.std(short_ret))
            long_vol = float(np.std(long_ret))
            vol_ratio = short_vol / (long_vol + 1e-8)

            trend_changes = np.diff(np.sign(short_ret))
            trend_consistency = 1.0 - np.sum(np.abs(trend_changes)) / (2 * len(trend_changes))

            stability = float(np.clip(
                0.5 + 0.25 * (1.0 - min(vol_ratio, 2.0) / 2.0) + 0.25 * trend_consistency,
                0.2, 0.9
            ))
        else:
            stability = 0.5

        _ = expert_signals  # reserved for future use

        return {
            "model_confidence": confidence,
            "is_trained": True,
            "stability_score": stability,
            "market_predictions": {
                "latest_predictions": {
                    "price_changes": price_changes,
                    "volatility_predictions": volatility_predictions,
                    "predicted_regime": predicted_regime,
                    "regime_probabilities": probs.tolist(),
                    "confidence": confidence,
                    "model_trained": True,
                },
            },
            "scenario_generation": {
                "bullish_probability": bullish_prob,
                "scenarios": [
                    {"outcome": 1.0, "probability": bullish_prob},
                    {"outcome": -1.0, "probability": 1.0 - bullish_prob},
                ],
            },
        }

    def _default_world_model_state(self) -> Dict[str, Any]:
        """Return default world model state when data unavailable."""
        return {
            "model_confidence": 0.5,
            "is_trained": True,
            "stability_score": 0.5,
            "market_predictions": {
                "latest_predictions": {
                    "price_changes": [0.0, 0.0, 0.0, 0.0],
                    "volatility_predictions": [0.5, 0.5, 0.5, 0.5],
                    "predicted_regime": 2,
                    "regime_probabilities": [0.25, 0.25, 0.25, 0.25],
                    "confidence": 0.5,
                    "model_trained": True,
                },
            },
            "scenario_generation": {
                "bullish_probability": 0.5,
                "scenarios": [
                    {"outcome": 1.0, "probability": 0.5},
                    {"outcome": -1.0, "probability": 0.5},
                ],
            },
        }
