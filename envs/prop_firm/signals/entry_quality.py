# envs/prop_firm/signals/entry_quality.py
# pyright: reportAttributeAccessIssue=false
"""
Entry quality computation mixin for PropFirmTradingEnv.

Contains methods for computing and caching entry quality signals.

Design goals:
- Per-step caching to avoid redundant expensive computations
- Robustness to missing / partial signal dictionaries
- Cost-awareness (spread percentile) and contradiction-awareness (alignment penalties)
- Lightweight telemetry for debugging (optional, no external deps)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np

if TYPE_CHECKING:
    pass


class EntryQualityMixin:
    """Mixin providing entry quality computation methods.

    Expected attributes from PropFirmTradingEnv:
    - config: PropFirmConfig
    - data: Dict[str, Dict[str, pd.DataFrame]]
    - current_step: int

    Expected methods from PropFirmTradingEnv (called here):
    - _prepare_expert_signals(inst) -> dict-like
    - _prepare_committee_state(expert_signals) -> dict-like
    - _prepare_risk_state() -> dict-like
    - _get_ohlcv(inst, lookback=...) -> dict-like (arrays)
    - _get_bar_dt(inst) -> datetime | None
    - _in_prime_window(dt) -> bool
    - _in_no_new_trades_window(dt) -> bool
    - _compute_rsi(close: np.ndarray, period: int) -> float
    - _dir_sign(direction_str: str) -> float  # should return -1,0,+1 style
    """

    # -------------------------------------------------------------------------
    # Small helpers (local, safe, and cheap)
    # -------------------------------------------------------------------------

    @staticmethod
    def _clamp01(x: float) -> float:
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            if x is None:
                return float(default)
            return float(x)
        except Exception:
            return float(default)

    @staticmethod
    def _as_dict(x: Any) -> Dict[str, Any]:
        return x if isinstance(x, dict) else {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Numerically stable-ish sigmoid for moderate magnitudes
        x = float(np.clip(x, -20.0, 20.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def _robust_std(x: np.ndarray) -> float:
        # Cheap robustness: ignore NaNs, require small minimum
        if x.size == 0:
            return 0.0
        x = x[np.isfinite(x)]
        if x.size < 5:
            return 0.0
        return float(np.std(x))

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

    # -------------------------------------------------------------------------
    # Entry quality public-ish accessors
    # -------------------------------------------------------------------------

    def _get_step_entry_quality(self, inst: str, target: str) -> float:
        """
        Get entry quality with per-step caching.
        Avoids redundant computation of the expensive _compute_smart_entry_quality.

        Cache is reset automatically when current_step changes, so values never leak
        across steps.
        """
        cache = getattr(self, "_step_entry_quality_cache", None)
        cache_step = getattr(self, "_step_entry_quality_cache_step", None)

        if cache is None or cache_step != self.current_step:
            cache = {}
            self._step_entry_quality_cache = cache
            self._step_entry_quality_cache_step = self.current_step

        cache_key = f"{inst}_{target}"
        if cache_key in cache:
            return cache[cache_key]

        quality = self._compute_smart_entry_quality(inst, target)
        cache[cache_key] = quality
        return quality

    def _get_step_entry_certainty(self, inst: str, target: str) -> float:
        """
        Get entry certainty with per-step caching.

        Certainty is a lighter-weight confidence signal (0..1) intended for
        patience/selectivity shaping. It is not the same as entry quality.
        """
        cache = getattr(self, "_step_entry_certainty_cache", None)
        cache_step = getattr(self, "_step_entry_certainty_cache_step", None)

        if cache is None or cache_step != self.current_step:
            cache = {}
            self._step_entry_certainty_cache = cache
            self._step_entry_certainty_cache_step = self.current_step

        cache_key = f"{inst}_{target}"
        if cache_key in cache:
            return cache[cache_key]

        certainty = self._compute_entry_certainty(inst, target)
        cache[cache_key] = certainty
        return certainty

    def _get_step_setup_quality(self, inst: str, target: str) -> Tuple[float, int]:
        """
        Get setup quality (0..1) + confluence count with per-step caching.

        Setup quality emphasizes confluence (multiple aligned signals) rather
        than just directional bias.
        """
        cache = getattr(self, "_step_setup_quality_cache", None)
        cache_step = getattr(self, "_step_setup_quality_cache_step", None)

        if cache is None or cache_step != self.current_step:
            cache = {}
            self._step_setup_quality_cache = cache
            self._step_setup_quality_cache_step = self.current_step

        cache_key = f"{inst}_{target}"
        if cache_key in cache:
            return cache[cache_key]

        setup_q, confluence = self._compute_setup_quality(inst, target)
        cache[cache_key] = (setup_q, confluence)
        return setup_q, confluence

    def _compute_entry_certainty(self, inst: str, target: str) -> float:
        """
        Compute a 0..1 entry certainty score.

        Certainty is based on consensus/confidence rather than directional
        alignment strength, so it can be used to reward selectivity without
        duplicating entry quality.
        """
        if target not in ("long", "short"):
            return 0.5

        direction_mult = 1.0 if target == "long" else -1.0

        expert_signals_raw = self._prepare_expert_signals(inst)
        committee = self._as_dict(self._prepare_committee_state(expert_signals_raw))

        consensus = self._safe_float(committee.get("consensus_score"), 0.5)
        conf = self._safe_float(committee.get("confidence"), 0.5)
        agreement = self._safe_float(committee.get("agreement"), 0.5)
        action_value = self._safe_float(committee.get("action_value"), 0.0)
        action_alignment = self._clamp01((action_value * direction_mult + 1.0) / 2.0)

        # Expert confidence (direction-agnostic)
        experts = self._as_dict(self._as_dict(expert_signals_raw).get("experts"))
        expert_conf = 0.5
        if experts:
            confs: List[float] = []
            for sig in experts.values():
                if not isinstance(sig, dict):
                    continue
                confs.append(self._safe_float(sig.get("confidence"), 0.5))
            if confs:
                expert_conf = float(np.clip(np.mean(confs), 0.0, 1.0))

        # Weighted blend (favor consensus/confidence)
        certainty = (
            0.38 * consensus
            + 0.24 * conf
            + 0.18 * agreement
            + 0.10 * action_alignment
            + 0.10 * expert_conf
        )

        return self._clamp01(certainty)

    def _compute_setup_quality(self, inst: str, target: str) -> Tuple[float, int]:
        """
        Compute setup quality (0..1) and confluence count.

        Uses entry-quality components + structure context to quantify
        how many independent signals align.
        """
        if target not in ("long", "short"):
            return 0.5, 0

        entry_quality = self._compute_smart_entry_quality(inst, target)

        # Pull component diagnostics from entry quality (if available)
        confluence_count = 0
        total_components = 0
        dbg = getattr(self, "_last_entry_quality_debug", {}) or {}
        comp = dbg.get(f"{inst}_{target}", {}).get("components", []) if isinstance(dbg, dict) else []
        if isinstance(comp, list):
            for _, v, _w in comp:
                total_components += 1
                if float(v) >= 0.65:
                    confluence_count += 1

        # Add structure confluence from entry context
        try:
            ctx = self._capture_entry_context(inst)
        except Exception:
            ctx = {}
        try:
            # Cache for other mixins (dynamic patience, time-of-day granularity)
            self._last_step_entry_context = ctx
        except Exception:
            pass

        structure_trend = self._safe_float(ctx.get("structure_trend"), 0.0)
        near_support = self._safe_float(ctx.get("near_support"), 0.0)
        near_resistance = self._safe_float(ctx.get("near_resistance"), 0.0)

        total_components += 1
        if target == "long" and structure_trend > 0.25:
            confluence_count += 1
        elif target == "short" and structure_trend < -0.25:
            confluence_count += 1

        total_components += 1
        if target == "long" and near_support > 0.5:
            confluence_count += 1
        elif target == "short" and near_resistance > 0.5:
            confluence_count += 1

        if total_components <= 0:
            total_components = 1
        confluence_ratio = confluence_count / float(total_components)

        setup_quality = self._clamp01(0.55 * entry_quality + 0.45 * confluence_ratio)

        # Optional debug capture
        try:
            dbg_setup = getattr(self, "_last_setup_quality_debug", None)
            if not isinstance(dbg_setup, dict):
                dbg_setup = {}
                self._last_setup_quality_debug = dbg_setup
            dbg_setup[f"{inst}_{target}"] = {
                "entry_quality": float(entry_quality),
                "confluence_count": int(confluence_count),
                "confluence_ratio": float(confluence_ratio),
                "setup_quality": float(setup_quality),
            }
        except Exception:
            pass

        return float(setup_quality), int(confluence_count)

    def _compute_smart_entry_quality(self, inst: str, target: str) -> float:
        """
        Compute a 0..1 entry quality score with multiple components:
        - Experts alignment (trend/momentum/theme)
        - Committee state (action alignment + consensus/confidence/agreement)
        - HTF context (simple SMA context)
        - Risk headroom (overall + daily drawdown)
        - Momentum sanity via RSI bucketization
        - Timing windows (prime / no-new-trades)
        - Cost awareness (spread percentile)

        Includes a lightweight contradiction penalty to reduce scores when key
        components strongly disagree with intended direction.
        """
        if target not in ("long", "short"):
            return 0.5

        direction_mult = 1.0 if target == "long" else -1.0

        # Track components for debugging/telemetry (optional; harmless if unused)
        quality_components: List[Tuple[str, float, float]] = []
        contradictions: List[float] = []

        # -------------------- Experts --------------------
        expert_signals_raw = self._prepare_expert_signals(inst)
        expert_signals = self._as_dict(expert_signals_raw)
        experts = self._as_dict(expert_signals.get("experts"))

        if experts:
            def ex_score(name: str) -> float:
                sig = self._as_dict(experts.get(name))
                s = self._safe_float(sig.get("score"), 0.0)
                c = self._safe_float(sig.get("confidence"), 0.5)
                d = self._dir_sign(str(sig.get("direction", "neutral")))
                return float(d * s * c)

            trend_score = ex_score("trend")
            mom_score = ex_score("momentum")
            theme_score = ex_score("theme")

            # Alignment in [-1, +1] (roughly), then map to 0..1
            alignment = 0.40 * trend_score + 0.35 * mom_score + 0.25 * theme_score
            expert_quality = self._clamp01((alignment * direction_mult + 1.0) / 2.0)

            quality_components.append(("experts", expert_quality, 0.28))

            # Contradiction signal: if alignment is opposite direction strongly
            contradictions.append(self._clamp01((-alignment * direction_mult)))  # 0 good, 1 bad

        # -------------------- Committee --------------------
        committee_raw = self._prepare_committee_state(expert_signals_raw)
        committee = self._as_dict(committee_raw)

        if committee:
            action_value = self._safe_float(committee.get("action_value"), 0.0)
            consensus = self._safe_float(committee.get("consensus_score"), 0.5)
            conf = self._safe_float(committee.get("confidence"), 0.5)
            agreement = self._safe_float(committee.get("agreement"), 0.5)

            action_alignment = self._clamp01((action_value * direction_mult + 1.0) / 2.0)

            committee_quality = self._clamp01(
                0.36 * consensus +
                0.34 * action_alignment +
                0.20 * conf +
                0.10 * agreement
            )
            quality_components.append(("committee", committee_quality, 0.24))

            # Contradiction: committee wants opposite direction
            contradictions.append(self._clamp01((0.5 - action_alignment) * 2.0))  # 0 good, 1 bad

        # -------------------- HTF context --------------------
        htf_quality = 0.5
        ohlcv = self._get_ohlcv(inst, lookback=120)
        close = np.asarray(ohlcv.get("close", np.array([])), dtype=np.float64) if ohlcv else np.array([], dtype=np.float64)

        if close.size >= 50:
            sma_20 = float(np.mean(close[-20:]))
            sma_50 = float(np.mean(close[-50:]))
            current_price = float(close[-1])

            above_sma20 = 1.0 if current_price > sma_20 else -1.0
            above_sma50 = 1.0 if current_price > sma_50 else -1.0
            sma_trend = 1.0 if sma_20 > sma_50 else -1.0

            htf_alignment = (
                0.40 * (above_sma20 * direction_mult) +
                0.35 * (above_sma50 * direction_mult) +
                0.25 * (sma_trend * direction_mult)
            )
            htf_quality = self._clamp01((htf_alignment + 1.0) / 2.0)

            # Contradiction: HTF context strongly against desired direction
            contradictions.append(self._clamp01((-htf_alignment)))  # already in desired direction space
        quality_components.append(("htf_context", float(htf_quality), 0.18))

        # -------------------- Risk headroom --------------------
        risk_state_raw = self._prepare_risk_state()
        risk_state = self._as_dict(risk_state_raw)

        if risk_state:
            dd = self._safe_float(risk_state.get("current_drawdown"), 0.0)
            ddd = self._safe_float(risk_state.get("daily_drawdown"), 0.0)

            dd_limit = max(self._safe_float(getattr(self.config, "max_drawdown_limit", 0.0), 0.0), 1e-6)
            daily_limit = max(self._safe_float(getattr(self.config, "daily_drawdown_limit", 0.0), 0.0), 1e-6)

            # Headroom in 0..1 where 1 = lots of room, 0 = at/over limit
            dd_headroom = max(0.0, (dd_limit - dd) / dd_limit)
            daily_headroom = max(0.0, (daily_limit - ddd) / daily_limit)

            risk_quality = self._clamp01(min(dd_headroom, daily_headroom))
            quality_components.append(("risk_state", risk_quality, 0.14))

        # -------------------- Momentum sanity (RSI buckets) --------------------
        momentum_quality = 0.5
        if close.size >= 14:
            rsi = float(self._compute_rsi(close, 14))
            if target == "long":
                if 30.0 <= rsi <= 50.0:
                    momentum_quality = 0.90
                elif 50.0 < rsi <= 65.0:
                    momentum_quality = 0.60
                elif rsi > 65.0:
                    momentum_quality = 0.30
                else:
                    momentum_quality = 0.70
            else:
                if 50.0 <= rsi <= 70.0:
                    momentum_quality = 0.90
                elif 35.0 <= rsi < 50.0:
                    momentum_quality = 0.60
                elif rsi < 35.0:
                    momentum_quality = 0.30
                else:
                    momentum_quality = 0.70
        quality_components.append(("momentum", float(momentum_quality), 0.10))

        # -------------------- Timing --------------------
        timing_quality = 0.5
        dt = self._get_bar_dt(inst)
        if dt is not None:
            if self._in_prime_window(dt):
                timing_quality = 0.95
            elif not self._in_no_new_trades_window(dt):
                timing_quality = 0.60
            else:
                timing_quality = 0.10
        quality_components.append(("timing", float(timing_quality), 0.12))

        # -------------------- Cost awareness (spread percentile) --------------------
        spread_pct = self._compute_spread_percentile(inst)  # 0..1 (low -> cheap)
        # Map percentile to quality: penalize expensive spreads nonlinearly
        # (cheap: ~1.0, median: ~0.55, very expensive: ~0.1)
        cost_quality = self._clamp01(1.0 - (spread_pct ** 1.35))
        quality_components.append(("costs", float(cost_quality), 0.10))

        # Optional: volatility-aware haircut when both vol and spread are high
        # (prevents "great signal" during chaos with huge execution costs).
        vol_haircut = 0.0
        if close.size >= 30:
            rets = np.diff(close[-60:]) / np.maximum(1e-9, close[-60:-1])
            vol = self._robust_std(rets)  # typical is small numbers (e.g., 0.001..0.02)
            # Convert vol to 0..1 "stress" with a soft threshold; tune constants later
            stress = self._sigmoid((vol - 0.006) / 0.003)  # center ~0.6% per bar
            # Combine stress with spread percentile; only bites when both are high
            vol_haircut = float(0.35 * stress * (spread_pct ** 1.2))

        # -------------------- Aggregate --------------------
        total_weight = sum(w for _, _, w in quality_components)
        if total_weight <= 0.0:
            return 0.5

        weighted_sum = sum(v * w for _, v, w in quality_components)
        base_quality = self._clamp01(weighted_sum / total_weight)

        # -------------------- Contradiction penalty (lightweight, multiplicative) --------------------
        # Take the worst contradiction signals (top-2) and apply a penalty,
        # so “one very wrong” component can meaningfully lower the score.
        contradiction = 0.0
        if contradictions:
            top = sorted([self._clamp01(x) for x in contradictions], reverse=True)[:2]
            contradiction = float(np.mean(top))

        # Penalty in [~0.65 .. 1.0], stronger when contradiction is high
        contradiction_penalty = float(1.0 - 0.35 * (contradiction ** 1.2))

        quality = base_quality * contradiction_penalty

        # Apply volatility haircut (already 0..~0.35)
        quality = self._clamp01(quality * (1.0 - vol_haircut))

        # -------------------- Optional telemetry --------------------
        # Stores per-step debug info without altering behavior elsewhere.
        try:
            dbg = {
                "step": int(getattr(self, "current_step", -1)),
                "target": str(target),
                "components": [(n, float(v), float(w)) for (n, v, w) in quality_components],
                "base_quality": float(base_quality),
                "contradiction": float(contradiction),
                "contradiction_penalty": float(contradiction_penalty),
                "spread_percentile": float(spread_pct),
                "cost_quality": float(cost_quality),
                "vol_haircut": float(vol_haircut),
                "final_quality": float(quality),
            }
            last = getattr(self, "_last_entry_quality_debug", None)
            if not isinstance(last, dict):
                last = {}
                self._last_entry_quality_debug = last
            last[f"{inst}_{target}"] = dbg
        except Exception:
            pass

        return float(quality)

    # -------------------------------------------------------------------------
    # Context capture for reward shaping / analysis
    # -------------------------------------------------------------------------

    def _capture_entry_context(self, instrument: str) -> Dict[str, Any]:
        """
        Capture market structure context at trade entry for reward calculation.

        Returns dict with:
        - near_support, near_resistance: S/R proximity (0-1)
        - structure_trend: -1 (LL/LH) to +1 (HH/HL)
        - bos_signal: -1 (bearish BOS) to +1 (bullish BOS)
        - order_block_bull, order_block_bear: 0-1 proximity
        - divergence_signal: "bullish", "bearish", or None
        - overbought, oversold: 0-1 intensity
        - volatility_regime, risk_regime: string regime labels
        - spread_percentile: 0..1 relative execution cost measure
        """
        expert_signals_raw = self._prepare_expert_signals(instrument)
        expert_signals = self._as_dict(expert_signals_raw)

        context: Dict[str, Any] = {}

        experts = self._as_dict(expert_signals.get("experts"))

        # Trend expert (structure)
        trend = self._as_dict(experts.get("trend"))
        trend_proposal = self._as_dict(trend.get("proposal"))

        context["near_support"] = self._safe_float(trend_proposal.get("near_support"), 0.0)
        context["near_resistance"] = self._safe_float(trend_proposal.get("near_resistance"), 0.0)
        context["structure_trend"] = self._safe_float(trend_proposal.get("structure_trend"), 0.0)
        context["structure_strength"] = self._safe_float(trend_proposal.get("structure_strength"), 0.0)
        context["bos_signal"] = self._safe_float(trend_proposal.get("bos_signal"), 0.0)
        context["liquidity_above"] = self._safe_float(trend_proposal.get("liquidity_above"), 0.0)
        context["liquidity_below"] = self._safe_float(trend_proposal.get("liquidity_below"), 0.0)
        context["order_block_bull"] = self._safe_float(trend_proposal.get("order_block_bull"), 0.0)
        context["order_block_bear"] = self._safe_float(trend_proposal.get("order_block_bear"), 0.0)

        # Momentum expert (divergence + OB/OS)
        momentum = self._as_dict(experts.get("momentum"))
        momentum_proposal = self._as_dict(momentum.get("proposal"))

        context["divergence_signal"] = momentum_proposal.get("divergence_signal")
        context["overbought"] = self._safe_float(momentum_proposal.get("overbought"), 0.0)
        context["oversold"] = self._safe_float(momentum_proposal.get("oversold"), 0.0)
        context["rsi_value"] = self._safe_float(momentum_proposal.get("rsi_value"), 50.0)

        # Theme expert (regime)
        theme = self._as_dict(experts.get("theme"))
        theme_proposal = self._as_dict(theme.get("proposal"))

        context["volatility_regime"] = theme_proposal.get("volatility_regime", "normal")
        context["risk_regime"] = theme_proposal.get("risk_regime", "neutral")
        context["vol_score"] = self._safe_float(theme_proposal.get("vol_score"), 0.5)

        # Spread percentile for regime tracking (Phase 2.2)
        context["spread_percentile"] = self._compute_spread_percentile(instrument)

        return context

    # -------------------------------------------------------------------------
    # Execution-cost proxy
    # -------------------------------------------------------------------------

    def _compute_spread_percentile(self, instrument: str) -> float:
        """Compute current spread as percentile of recent spread history (0..1).

        - 0.0 means extremely tight vs recent history (cheap)
        - 1.0 means extremely wide vs recent history (expensive)

        Fallbacks:
        - If spread column missing, try infer from ask/bid if present.
        - Otherwise return 0.5.
        """
        try:
            tf_map = self.data.get(instrument)
            if not isinstance(tf_map, dict):
                return 0.5

            df = tf_map.get("M15")
            if df is None or len(df) < 25:
                return 0.5

            # Ensure idx is within bounds
            idx = int(getattr(self, "current_step", 0))
            idx = max(0, min(idx, len(df) - 1))
            if idx < 20:
                return 0.5

            lookback = min(200, idx)
            if lookback < 20:
                return 0.5

            # Obtain spread series
            if "spread" in df.columns:
                hist = df["spread"].iloc[idx - lookback:idx].to_numpy(dtype=np.float64, copy=False)
                cur = float(df["spread"].iloc[idx])
            elif ("ask" in df.columns) and ("bid" in df.columns):
                # Infer spread from ask-bid
                hist_ask = df["ask"].iloc[idx - lookback:idx].to_numpy(dtype=np.float64, copy=False)
                hist_bid = df["bid"].iloc[idx - lookback:idx].to_numpy(dtype=np.float64, copy=False)
                hist = (hist_ask - hist_bid)
                cur = float(df["ask"].iloc[idx] - df["bid"].iloc[idx])
            else:
                return 0.5

            # Clean NaNs/Infs
            hist = hist[np.isfinite(hist)]
            if hist.size < 10 or not np.isfinite(cur):
                return 0.5

            # Percentile by rank
            # Use <= for a stable “empirical CDF” percentile.
            pct = float(np.mean(hist <= cur))
            return self._clamp01(pct)
        except Exception:
            return 0.5
