"""
Advanced ThemeExpert - Macro Theme and Risk Regime Voting Module.

This expert analyzes macro themes and market regimes including:
- Volatility regime analysis (ATR, historical vol, implied vol proxy)
- Correlation cluster analysis (cross-asset / cross-timeframe correlations)
- Trend strength aggregation (ADX / DI-style signal)
- Market breadth indicators
- Momentum overlay
- Risk-on / Risk-off scoring

Per-Instrument Voting:
- Analyzes each instrument separately to produce per-instrument votes
- Different asset classes (FX, commodities) may have different regimes
- Each instrument gets its own action/confidence based on its own data

Actions: long, short, flat (plus internal 'hold', 'exit', 'tighten' in position-focus mode)
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.experts.base import VotingExpertBase
from modules.voting.core.per_instrument import (
    PerInstrumentVote,
    InstrumentProposal,
    normalize_instrument,
)
from modules.voting.core.constants import (
    CONFIDENCE_THRESHOLD_F,
    MIN_SIGNAL_STRENGTH_F,
    HIGH_CONFIDENCE_THRESHOLD_F,
)


@module(**module_args("ThemeExpert"))
class ThemeExpert(VotingExpertBase):
    """
    Advanced macro theme and regime analysis expert.

    Combines volatility, trend, correlation, breadth, and momentum into
    a per-instrument macro "theme" signal, then aggregates into a
    global ThemeExpert vote for backward compatibility.

    Per-instrument:
      • EURUSD, XAUUSD, etc. each get their own macro/regime vote.
      • Different instruments can simultaneously be in different regimes.
    """

    # ═══════════════════════════ INIT ═══════════════════════════

    def _expert_specific_init(self) -> None:
        """Initialize the theme expert with configuration and baseline bus keys."""
        self.module_name = self.__class__.__name__

        # Instruments to analyze (from config or default) – normalize for consistency.
        raw_instruments = self.config.get("instruments", ["EURUSD", "XAUUSD"])
        self.instruments = [normalize_instrument(i) for i in raw_instruments]

        # Asset class mapping for different regime interpretations (safe-haven handling, etc.).
        self.asset_classes = {
            "EURUSD": "forex",
            "XAUUSD": "commodity",
            "GBPUSD": "forex",
            "USDJPY": "forex",
        }

        # Volatility configuration
        self.atr_period = int(self.config.get("atr_period", 20))
        self.vol_lookback = int(self.config.get("vol_lookback", 50))
        self.vol_regime_threshold_low = float(
            self.config.get("vol_threshold_low", 0.3)
        )
        self.vol_regime_threshold_high = float(
            self.config.get("vol_threshold_high", 0.7)
        )

        # Correlation configuration (cross-timeframe clustering)
        self.corr_lookback = int(self.config.get("corr_lookback", 30))
        self.corr_cluster_threshold = float(
            self.config.get("corr_cluster_threshold", 0.7)
        )

        # Trend configuration (ADX-like directional score)
        self.adx_period = int(self.config.get("adx_period", 14))
        self.trend_strength_threshold = float(
            self.config.get("trend_strength_threshold", 25.0)
        )
        self.strong_trend_threshold = float(
            self.config.get("strong_trend_threshold", 40.0)
        )

        # Risk scoring thresholds (on/off boundaries)
        self.risk_on_threshold = float(self.config.get("risk_on_threshold", 0.6))
        self.risk_off_threshold = float(self.config.get("risk_off_threshold", 0.4))

        # Breadth configuration (range position + up/down day balance)
        self.breadth_period = int(self.config.get("breadth_period", 20))

        # Sentiment component weights
        self.sentiment_weights = {
            "volatility": 0.25,
            "trend": 0.25,
            "correlation": 0.20,
            "breadth": 0.15,
            "momentum": 0.15,
        }

        # Historical state for regime persistence (GLOBAL theme only)
        self.regime_history: List[str] = []
        self.regime_persistence_count = 0
        self.min_regime_persistence = int(
            self.config.get("min_regime_persistence", 3)
        )

        # Confidence calibration
        self.base_confidence = float(self.config.get("base_confidence", 0.5))
        self.max_confidence = float(self.config.get("max_confidence", 0.95))
        self.min_confidence = float(self.config.get("min_confidence", 0.15))

        # Optional debug throttle per instrument (avoids log spam)
        self._debug_last_log: Dict[str, float] = {}

        self.log_info(
            f"[THEME] ThemeExpert initialized | "
            f"instruments={self.instruments} | ATR period={self.atr_period}"
        )

        # Publish generic baseline keys from VotingExpertBase (if available)
        if hasattr(self, "_publish_baseline_keys"):
            try:
                self._publish_baseline_keys()
            except Exception:
                # Non-fatal; ThemeExpert still publishes its own baseline
                pass

        # Publish ThemeExpert-specific baseline keys
        self._publish_theme_baseline()

    def _publish_theme_baseline(self) -> None:
        """
        Publish baseline theme keys to SmartInfoBus so downstream readers
        never see missing keys, even before the first real tick.
        """
        thesis = "Theme baseline"
        proposal = {
            "action": "flat",
            "signal_strength": 0.1,
            "reason": thesis,
            "proposals": {},
        }
        analysis = {
            "volatility_regime": "unknown",
            "trend_regime": "unknown",
            "risk_regime": "unknown",
            "composite_score": 0.5,
            "action": "flat",
            "confidence": 0.1,
            "per_instrument": {},
        }

        try:
            # Canonical expert keys
            self.smart_bus.set(
                "ThemeExpert_voting_proposal",
                proposal,
                module=self.module_name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "ThemeExpert_confidence",
                0.1,
                module=self.module_name,
                thesis="Theme baseline confidence",
            )

            # Contract aliases used in other modules
            self.smart_bus.set(
                "theme_voting_proposal",
                proposal,
                module=self.module_name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "theme_confidence",
                0.1,
                module=self.module_name,
                thesis="Theme baseline confidence",
            )
            self.smart_bus.set(
                "theme_analysis",
                analysis,
                module=self.module_name,
                thesis="Theme baseline analysis",
            )
        except Exception:
            # Baseline publication failures are not fatal.
            pass

    # ═══════════════════════════ VOTINGEXPERTBASE HOOKS (LEGACY) ═══════════════════════════

    async def _generate_expert_specific_proposal(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Legacy hook for VotingExpertBase.

        ThemeExpert uses a custom per-instrument process() path that does
        richer regime analysis. This method remains as a neutral fallback
        in case some generic caller invokes the base template.
        """
        return {
            "action": "flat",
            "signal_strength": 0.0,
            "reason": "ThemeExpert uses per-instrument process() path",
        }

    async def _calculate_expert_specific_confidence(
        self, proposal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> float:
        """Legacy confidence hook; neutral default for base compat."""
        return 0.3

    # ═══════════════════════════ MAIN PER-INSTRUMENT PROCESS ═══════════════════════════

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Process market data to determine macro theme and regime PER INSTRUMENT.

        Data sourcing:
          • If `market_data` / `features` are provided in `inputs`, they are used.
          • Otherwise, they are pulled from SmartInfoBus under this module's name.

        Outputs:
          • Global ThemeExpert_voting_proposal / ThemeExpert_confidence
          • Per-instrument votes in ThemeExpert_per_instrument_votes
          • Contract aliases: theme_voting_proposal, theme_confidence, theme_analysis
        """
        name = self.__class__.__name__
        start = self._now_ms()

        try:
            # Circuit breaker from VotingExpertBase (risk safety)
            if self._check_circuit_breaker():
                return self._degraded_output("circuit_breaker_open")

            # Pull from inputs first (for tests/offline calls), then InfoBus fallback.
            market_data = inputs.get("market_data")
            features = inputs.get("features")

            if market_data is None:
                try:
                    market_data = self.smart_bus.get("market_data", name, default={})
                except Exception:
                    market_data = {}
            if features is None:
                try:
                    features = self.smart_bus.get("features", name, default={})
                except Exception:
                    features = {}

            if not isinstance(market_data, dict):
                market_data = {}
            if not isinstance(features, dict):
                features = {}

            if not market_data and not features:
                self.log_debug(
                    f"[THEME][BUS] empty fetch: "
                    f"market_data={market_data}, features={features}"
                )
                self.log_warning("[THEME] No market data or features available")
                output = self._neutral_output("No market data available")

            else:
                per_instrument_vote = PerInstrumentVote(member=name)
                per_instrument_analysis: Dict[str, Dict[str, Any]] = {}

                # ── Per-instrument regime analysis ─────────────────────────────
                for inst in self.instruments:
                    inst_norm = normalize_instrument(inst)

                    inst_market = self._extract_instrument_data(market_data, inst_norm)
                    inst_features = self._extract_instrument_data(features, inst_norm)

                    close_prices = self._extract_prices(
                        inst_market, inst_features, "close", inst_norm
                    )
                    high_prices = self._extract_prices(
                        inst_market, inst_features, "high", inst_norm
                    )
                    low_prices = self._extract_prices(
                        inst_market, inst_features, "low", inst_norm
                    )

                    # ═══════════════════════════════════════════════════════════
                    # REAL-TIME RESPONSIVENESS: append forming M15 bar
                    # Makes theme/regime update intra-bar instead of only
                    # on closed candles.
                    # ═══════════════════════════════════════════════════════════
                    try:
                        historical = self.smart_bus.get(
                            "historical_prices", name, default=None
                        )
                    except Exception:
                        historical = None

                    try:
                        if (
                            isinstance(historical, dict)
                            and isinstance(close_prices, np.ndarray)
                            and close_prices.size > 0
                        ):
                            matched_sym = None
                            for sym in historical.keys():
                                if normalize_instrument(sym) == inst_norm:
                                    matched_sym = sym
                                    break

                            if (
                                matched_sym
                                and isinstance(historical.get(matched_sym), dict)
                            ):
                                m15_rec = historical[matched_sym].get("M15", {})
                                if isinstance(m15_rec, dict):
                                    cur_bar = m15_rec.get("current_bar", {})
                                    if isinstance(cur_bar, dict):
                                        forming_close = cur_bar.get("close")
                                        forming_high = cur_bar.get("high")
                                        forming_low = cur_bar.get("low")

                                        if forming_close is not None:
                                            forming_close = float(forming_close)
                                            last_close = float(close_prices[-1])
                                            # Only mutate last bar if there's a meaningful move
                                            if abs(forming_close - last_close) > 0.0001:
                                                close_prices = np.append(
                                                    close_prices[:-1], forming_close
                                                )
                                                if (
                                                    forming_high is not None
                                                    and isinstance(
                                                        high_prices, np.ndarray
                                                    )
                                                    and high_prices.size > 0
                                                ):
                                                    high_prices = np.append(
                                                        high_prices[:-1],
                                                        float(forming_high),
                                                    )
                                                if (
                                                    forming_low is not None
                                                    and isinstance(
                                                        low_prices, np.ndarray
                                                    )
                                                    and low_prices.size > 0
                                                ):
                                                    low_prices = np.append(
                                                        low_prices[:-1],
                                                        float(forming_low),
                                                    )
                    except Exception as e:
                        self.log_warning(f"[THEME] Failed to append forming bar: {e}")

                    # Ensure we have enough history for volatility / breadth logic
                    if len(close_prices) < self.vol_lookback:
                        # Debug snapshot of available data to diagnose insufficiency
                        tf_meta: Dict[str, Any] = {}
                        try:
                            historical = self.smart_bus.get(
                                "historical_prices", name, default=None
                            )
                        except Exception:
                            historical = None

                        if isinstance(historical, dict):
                            matched_sym = None
                            for sym in historical.keys():
                                if normalize_instrument(sym) == inst_norm:
                                    matched_sym = sym
                                    break
                            if matched_sym and isinstance(
                                historical.get(matched_sym), dict
                            ):
                                sym_block = historical[matched_sym]
                                for tf in ("M15", "H1", "H4", "D1"):
                                    rec = sym_block.get(tf)
                                    if isinstance(rec, dict):
                                        bars_avail = rec.get("bars_available")
                                        last_ts = None
                                        cur_bar = rec.get("current_bar")
                                        if isinstance(cur_bar, dict):
                                            last_ts = cur_bar.get("timestamp")
                                        close_seq = rec.get("close")
                                        try:
                                            close_len = (
                                                len(close_seq)
                                                if close_seq is not None
                                                else 0
                                            )
                                        except Exception:
                                            close_len = 0
                                        tf_meta[tf] = {
                                            "close_len": close_len,
                                            "bars_available": bars_avail,
                                            "last_ts": last_ts,
                                        }

                        self.log_debug(
                            f"[THEME][DATA] {inst_norm} insufficient data: "
                            f"close_len={len(close_prices)}, "
                            f"lookback={self.vol_lookback}, tf_meta={tf_meta}"
                        )
                        self.log_debug(
                            f"[THEME] Insufficient data for {inst_norm}, using neutral"
                        )

                        per_instrument_vote.set_proposal(
                            InstrumentProposal(
                                instrument=inst_norm,
                                action="flat",
                                confidence=0.1,
                                magnitude=0.0,
                                rationale=f"Insufficient data for {inst_norm}",
                            )
                        )
                        per_instrument_analysis[inst_norm] = {
                            "volatility_regime": "unknown",
                            "trend_regime": "unknown",
                            "risk_regime": "unknown",
                            "vol_score": 0.5,
                            "trend_score": 0.0,
                            "composite_score": 0.5,
                            "action": "flat",
                            "confidence": 0.1,
                            "correlation_regime": "unknown",
                            "breadth_score": 0.5,
                            "momentum_score": 0.0,
                            "position_focus_mode": False,
                            "supports_position": True,
                            "position_evaluation": "no_position",
                            "original_action": "flat",
                        }
                        continue

                    # Optional periodic debug snapshot
                    try:
                        import time as _time

                        now_ts = _time.time()
                        last_log = self._debug_last_log.get(inst_norm, 0.0)
                        if now_ts - last_log > 30.0:
                            self.log_debug(
                                f"[THEME][DATA] {inst_norm}: "
                                f"close_len={len(close_prices)}, "
                                f"latest={float(close_prices[-1]) if close_prices.size else None}"
                            )
                            self._debug_last_log[inst_norm] = now_ts
                    except Exception:
                        pass

                    # ═══════════════════════════════════════════════════════════
                    # PERFORMANCE CACHE: skip regime recomputation when prices
                    # for this instrument haven't changed.
                    # Uses the shared _get_cached_indicators/_set_cached_indicators
                    # utility from VotingExpertBase.
                    # ═══════════════════════════════════════════════════════════
                    prices_list = (
                        list(close_prices)
                        if hasattr(close_prices, "__iter__")
                        else []
                    )
                    cached = self._get_cached_indicators(inst_norm, prices_list)

                    if cached is not None:
                        # Cache hit - reuse computed regime components
                        vol_regime = cached.get("vol_regime", "unknown")
                        vol_score = float(cached.get("vol_score", 0.5))
                        trend_regime = cached.get("trend_regime", "unknown")
                        trend_score = float(cached.get("trend_score", 0.0))
                        corr_regime = cached.get("corr_regime", "unknown")
                        corr_score = float(cached.get("corr_score", 0.0))
                        breadth_score = float(cached.get("breadth_score", 0.5))
                        momentum_score = float(cached.get("momentum_score", 0.0))
                        composite_score = float(cached.get("composite_score", 0.5))
                    else:
                        # Cache miss - full regime computation
                        vol_regime, vol_score = self._analyze_volatility_regime(
                            close_prices, high_prices, low_prices
                        )
                        trend_regime, trend_score = self._analyze_trend_regime(
                            close_prices, high_prices, low_prices
                        )
                        corr_regime, corr_score = self._analyze_correlation_regime(
                            close_prices, inst_market
                        )
                        breadth_score = self._calculate_market_breadth(
                            close_prices, high_prices, low_prices
                        )
                        momentum_score = self._calculate_momentum_score(
                            close_prices
                        )
                        composite_score = self._calculate_composite_sentiment(
                            vol_score,
                            trend_score,
                            corr_score,
                            breadth_score,
                            momentum_score,
                        )

                        # Store into cache BEFORE any per-call confidence adjustments
                        self._set_cached_indicators(
                            inst_norm,
                            prices_list,
                            {
                                "vol_regime": vol_regime,
                                "vol_score": vol_score,
                                "trend_regime": trend_regime,
                                "trend_score": trend_score,
                                "corr_regime": corr_regime,
                                "corr_score": corr_score,
                                "breadth_score": breadth_score,
                                "momentum_score": momentum_score,
                                "composite_score": composite_score,
                            },
                        )

                    # ── Multi-timeframe confirmation (M15 primary, higher TFs context only) ───
                    from modules.voting.core.constants import (
                        MTF_AGREEMENT_BONUS,
                        MTF_DISAGREEMENT_PENALTY,
                    )

                    mtf_result = self._analyze_multi_timeframe_theme(inst_norm)
                    if mtf_result["valid"]:
                        # Trend confidence adjustment
                        if mtf_result["trend_aligned"]:
                            trend_score = float(
                                np.clip(
                                    trend_score * (1.0 + MTF_AGREEMENT_BONUS), -1, 1
                                )
                            )
                            self.log_debug(
                                f"[THEME MTF] {inst_norm}: Context TFs agree with M15, "
                                f"boosting trend_score to {trend_score:.3f}"
                            )
                        elif mtf_result["trend_opposed"]:
                            trend_score = float(
                                np.clip(
                                    trend_score * (1.0 - MTF_DISAGREEMENT_PENALTY),
                                    -1,
                                    1,
                                )
                            )
                            self.log_debug(
                                f"[THEME MTF] {inst_norm}: Context TFs oppose M15, "
                                f"reducing trend_score to {trend_score:.3f}"
                            )

                        # Lightweight adjustment for volatility consistency
                        if mtf_result["vol_consistent"]:
                            vol_score = float(np.clip(vol_score * 0.95, 0.0, 1.0))

                    asset_class = self.asset_classes.get(inst_norm, "forex")
                    risk_regime = self._determine_risk_regime(
                        vol_regime,
                        trend_regime,
                        composite_score,
                        asset_class=asset_class,
                    )

                    # ── Instrument theme action selection ─────────────────────
                    action, confidence, thesis = self._select_theme_action(
                        vol_regime,
                        trend_regime,
                        risk_regime,
                        vol_score,
                        trend_score,
                        composite_score,
                        instrument=inst_norm,
                    )

                    # Theme magnitude is an overlay, not a primary hammer.
                    min_strength = MIN_SIGNAL_STRENGTH_F()
                    if action == "flat":
                        magnitude = 0.0
                    else:
                        magnitude = float(
                            max(min_strength * 0.5, min(1.0, confidence * 0.8))
                        )

                    # ═══════════════════════════════════════════════════════════
                    # POSITION FOCUS MODE (PER-INSTRUMENT)
                    # If we have a position in THIS instrument, reframe the theme
                    # signal into hold/exit/tighten semantics for position mgmt.
                    # ═══════════════════════════════════════════════════════════
                    position_focus = self._get_position_focus_context()
                    supports_position = True
                    position_eval = "no_position"
                    original_action = action

                    if position_focus and self._has_position_for_instrument(
                        inst_norm
                    ):
                        inst_position = self._get_position_for_instrument(inst_norm)
                        position_side = self._get_position_side_for_instrument(
                            inst_norm
                        )
                        position_pnl = float(
                            inst_position.get(
                                "unrealized_pnl", inst_position.get("pnl", 0.0)
                            )
                        ) if inst_position else 0.0

                        # Determine whether theme supports or threatens existing position.
                        if position_side > 0:  # LONG position
                            if action in ("long", "buy"):
                                supports_position = True
                                position_eval = "supports_long"
                            elif action in ("short", "sell"):
                                supports_position = False
                                position_eval = "threatens_long"
                            else:
                                supports_position = True
                                position_eval = "neutral_for_long"
                        elif position_side < 0:  # SHORT position
                            if action in ("short", "sell"):
                                supports_position = True
                                position_eval = "supports_short"
                            elif action in ("long", "buy"):
                                supports_position = False
                                position_eval = "threatens_short"
                            else:
                                supports_position = True
                                position_eval = "neutral_for_short"

                        # Remap action into position-management language.
                        if supports_position:
                            action = "hold"
                        else:
                            if confidence > 0.7:
                                action = "exit"
                            elif confidence > 0.5:
                                action = "tighten"
                            else:
                                action = "hold"

                        # Adjust confidence based on PnL asymmetry.
                        if position_pnl > 0 and not supports_position:
                            confidence *= 0.8  # reduce eagerness to exit winning trades
                        elif position_pnl < 0 and not supports_position:
                            confidence = min(1.0, confidence * 1.2)

                        thesis = (
                            f"{inst_norm}: POSITION_FOCUS({position_eval}) -> {action} "
                            f"(original_action={original_action}, regime={risk_regime}, pnl={position_pnl:.2f})"
                        )

                        self.log_debug(
                            f"[THEME] {inst_norm} POSITION_FOCUS: "
                            f"original_action={original_action} -> {action}, supports={supports_position}, "
                            f"eval={position_eval}, pnl={position_pnl:.2f}"
                        )

                    # Register per-instrument proposal in the vote container.
                    per_instrument_vote.set_proposal(
                        InstrumentProposal(
                            instrument=inst_norm,
                            action=action,
                            confidence=confidence,
                            magnitude=magnitude,
                            rationale=thesis,
                        )
                    )

                    # Rich per-instrument analysis blob for debugging and dashboards.
                    per_instrument_analysis[inst_norm] = {
                        "volatility_regime": vol_regime,
                        "trend_regime": trend_regime,
                        "risk_regime": risk_regime,
                        "vol_score": vol_score,
                        "trend_score": trend_score,
                        "composite_score": composite_score,
                        "action": action,
                        "confidence": confidence,
                        "correlation_regime": corr_regime,
                        "corr_score": corr_score,
                        "breadth_score": breadth_score,
                        "momentum_score": momentum_score,
                        "position_focus_mode": self._has_position_for_instrument(
                            inst_norm
                        ),
                        "supports_position": supports_position,
                        "position_evaluation": position_eval,
                        "original_action": original_action,
                    }

                    self.log_debug(
                        f"[THEME] {inst_norm}: vol={vol_regime}, "
                        f"trend={trend_regime}, risk={risk_regime}, "
                        f"action={action}, conf={confidence:.2f}, "
                        f"comp={composite_score:.2f}"
                    )

                # ── Global summary / backward compatibility ──────────────────
                if per_instrument_vote.proposals:
                    # Leader = instrument with highest confidence
                    best_proposal = max(
                        per_instrument_vote.proposals.values(),
                        key=lambda p: p.confidence,
                    )
                    global_action = best_proposal.action
                    global_confidence = best_proposal.confidence
                    global_thesis = best_proposal.rationale
                    leader_inst = best_proposal.instrument
                    global_analysis = per_instrument_analysis.get(
                        leader_inst,
                        {
                            "volatility_regime": "unknown",
                            "trend_regime": "unknown",
                            "risk_regime": "unknown",
                            "composite_score": 0.5,
                            "action": global_action,
                            "confidence": global_confidence,
                        },
                    )
                else:
                    global_action = "flat"
                    global_confidence = 0.1
                    global_thesis = "No instrument data available"
                    global_analysis = {
                        "volatility_regime": "unknown",
                        "trend_regime": "unknown",
                        "risk_regime": "unknown",
                        "composite_score": 0.5,
                        "action": "flat",
                        "confidence": 0.1,
                    }

                # Persistence filter on GLOBAL theme action only.
                global_action, global_confidence = self._apply_persistence_filter(
                    global_action, global_confidence
                )

                # Clip with global thresholds and calibration.
                conf_floor = CONFIDENCE_THRESHOLD_F()
                high_conf = HIGH_CONFIDENCE_THRESHOLD_F()
                global_confidence = float(
                    max(conf_floor * 0.5, min(high_conf, global_confidence))
                )

                # Build per-instrument proposals dict for CommitteeCoordinator
                proposals_dict: Dict[str, Dict[str, Any]] = {}
                for inst, prop in per_instrument_vote.proposals.items():
                    proposals_dict[inst] = {
                        "action": prop.action,
                        "confidence": prop.confidence,
                        "magnitude": prop.magnitude,
                        "rationale": prop.rationale,
                    }

                # Main proposal for voting (global theme view)
                proposal = {
                    "action": global_action,
                    "signal_strength": global_confidence,
                    "reason": global_thesis,
                    "proposals": proposals_dict,
                }

                theme_analysis = {
                    "volatility_regime": global_analysis.get(
                        "volatility_regime", "unknown"
                    ),
                    "trend_regime": global_analysis.get("trend_regime", "unknown"),
                    "risk_regime": global_analysis.get("risk_regime", "unknown"),
                    "composite_score": global_analysis.get("composite_score", 0.5),
                    "action": global_action,
                    "confidence": global_confidence,
                    "per_instrument": per_instrument_analysis,
                }

                # Publish to SmartInfoBus (canonical + aliases + per-instrument)
                try:
                    # Canonical expert keys
                    self.smart_bus.set(
                        "ThemeExpert_voting_proposal",
                        proposal,
                        module=name,
                        thesis=global_thesis,
                    )
                    self.smart_bus.set(
                        "ThemeExpert_confidence",
                        global_confidence,
                        module=name,
                        thesis=f"Confidence: {global_confidence:.1%}",
                    )

                    # Contract aliases
                    self.smart_bus.set(
                        "theme_voting_proposal",
                        proposal,
                        module=name,
                        thesis=global_thesis,
                    )
                    self.smart_bus.set(
                        "theme_confidence",
                        global_confidence,
                        module=name,
                        thesis=f"Theme confidence: {global_confidence:.1%}",
                    )
                    self.smart_bus.set(
                        "theme_analysis",
                        theme_analysis,
                        module=name,
                        thesis=f"Theme analysis (leader={global_analysis.get('risk_regime', 'unknown')})",
                    )

                    # Per-instrument votes for dashboards / meta agents
                    per_inst_votes_dict = {
                        inst: prop.to_dict()
                        for inst, prop in per_instrument_vote.proposals.items()
                    }
                    self.smart_bus.set(
                        "ThemeExpert_per_instrument_votes",
                        per_inst_votes_dict,
                        module=name,
                        thesis=(
                            "Per-instrument theme votes: "
                            f"{list(per_inst_votes_dict.keys())}"
                        ),
                    )
                except Exception as e:
                    self.log_warning(f"[THEME] Failed to publish to bus: {e}")

                output = {
                    "ThemeExpert_voting_proposal": proposal,
                    "ThemeExpert_confidence": global_confidence,
                    "ThemeExpert_per_instrument_votes": {
                        inst: prop.to_dict()
                        for inst, prop in per_instrument_vote.proposals.items()
                    },
                    "per_instrument_votes": per_instrument_vote,
                    "theme_voting_proposal": proposal,  # alias
                    "theme_confidence": global_confidence,
                    "theme_analysis": theme_analysis,
                    # NOTE: Renamed from 'agreement_score' to avoid conflict with ConsensusAnalyzer
                    "theme_agreement_score": global_confidence,
                    "theme_expert_analysis": {
                        "volatility_regime": global_analysis.get(
                            "volatility_regime", "unknown"
                        ),
                        "trend_regime": global_analysis.get("trend_regime", "unknown"),
                        "risk_regime": global_analysis.get("risk_regime", "unknown"),
                        "composite_score": global_analysis.get("composite_score", 0.5),
                        "per_instrument": per_instrument_analysis,
                    },
                    "theme_expert_thesis": global_thesis,
                    "theme_volatility_regime": global_analysis.get(
                        "volatility_regime", "unknown"
                    ),
                    "theme_trend_regime": global_analysis.get(
                        "trend_regime", "unknown"
                    ),
                    "theme_risk_regime": global_analysis.get(
                        "risk_regime", "unknown"
                    ),
                    "theme_composite_score": global_analysis.get(
                        "composite_score", 0.5
                    ),
                    "_thesis": global_thesis,
                }

            # Perf metrics / success flag
            elapsed_ms = self._elapsed_ms(start)
            try:
                self.performance_tracker.record_metric(
                    name, "process", elapsed_ms, True
                )
            except Exception:
                pass
            self._record_success()
            return output

        except Exception as e:
            # Error path: degraded mode with contract-compatible shape.
            self._record_error(e)
            if self.error_pinpointer is not None:
                err_ctx = self.error_pinpointer.analyze_error(e, f"{name}_process")
                msg = str(err_ctx)
            else:
                msg = str(e)

            self.log_error(f"[THEME] Process error: {msg}")

            elapsed_ms = self._elapsed_ms(start)
            try:
                self.performance_tracker.record_metric(
                    name, "process", elapsed_ms, False
                )
            except Exception:
                pass

            return self._degraded_output(msg)

    # ═══════════════════════════ HELPERS ═══════════════════════════

    def _now_ms(self) -> float:
        """Monotonic milliseconds helper for perf tracking."""
        import time

        return time.time() * 1000.0

    def _elapsed_ms(self, start_ms: float) -> float:
        """Return elapsed milliseconds from a monotonic start timestamp."""
        import time

        return time.time() * 1000.0 - start_ms

    def _extract_instrument_data(self, data: Dict, instrument: str) -> Dict:
        """
        Extract data for a specific instrument from nested market data.

        Tries multiple aliases (normalized, raw, upper/lower) and falls
        back to the full dict for legacy flat formats.
        """
        if not isinstance(data, dict):
            return {}

        inst_norm = normalize_instrument(instrument)

        # Direct instrument key variants
        for key in [instrument, inst_norm, instrument.upper(), instrument.lower()]:
            if key in data:
                return data[key] if isinstance(data[key], dict) else data

        # Simple canonical mapping for main pairs
        for sep in ["_", "/", "-", ""]:
            for pair in (f"EUR{sep}USD", f"XAU{sep}USD"):
                norm_pair = normalize_instrument(pair)
                if norm_pair == inst_norm and pair in data:
                    return data[pair] if isinstance(data[pair], dict) else data

        # Legacy flat format
        return data

    def _extract_prices(
        self,
        market_data: Dict,
        features: Dict,
        price_type: str,
        instrument: str = "",
    ) -> np.ndarray:
        """
        Extract price array from market data, features, or InfoBus.

        Priority:
          1) Instrument-specific market_data (M15, then H1/H4/D1)
          2) Instrument-specific features
          3) InfoBus historical_prices (M15 primary, H1/H4/D1 context)

        For each candidate source, the "best" series is chosen:
          - Prefer sequences with length >= vol_lookback
          - Otherwise, choose the longest sequence
        """
        inst_norm = normalize_instrument(instrument) if instrument else ""

        def _best_array_from_candidates(candidates: List[np.ndarray]) -> np.ndarray:
            """Pick the best candidate, preferring those that satisfy vol_lookback."""
            if not candidates:
                return np.array([])
            long_enough = [c for c in candidates if len(c) >= self.vol_lookback]
            if long_enough:
                return max(long_enough, key=len)
            return max(candidates, key=len)

        candidates: List[np.ndarray] = []

        # Direct in market_data
        if isinstance(market_data, dict):
            if price_type in market_data:
                data = market_data[price_type]
                if isinstance(data, (list, np.ndarray)):
                    candidates.append(np.array(data, dtype=float))

            # Nested TF structure (M15 primary, H1/H4/D1 context)
            for tf in ("M15", "H1", "H4", "D1"):
                tf_block = market_data.get(tf)
                if isinstance(tf_block, dict) and price_type in tf_block:
                    data = tf_block[price_type]
                    if isinstance(data, (list, np.ndarray)):
                        candidates.append(np.array(data, dtype=float))

        # Features (already instrument-specific at this point)
        if isinstance(features, dict) and price_type in features:
            data = features[price_type]
            if isinstance(data, (list, np.ndarray)):
                candidates.append(np.array(data, dtype=float))

        if candidates:
            return _best_array_from_candidates(candidates)

        # Historical from InfoBus
        try:
            historical = self.smart_bus.get(
                "historical_prices", self.module_name, default=None
            )
        except Exception:
            historical = None

        if isinstance(historical, dict) and historical:
            inst_aliases = {
                "EURUSD": ["EUR_USD", "EURUSD"],
                "XAUUSD": ["XAU_USD", "XAUUSD", "GOLDUSD"],
            }
            aliases = inst_aliases.get(inst_norm, [inst_norm, instrument])

            symbol = None
            for alias in aliases:
                if alias in historical:
                    symbol = alias
                    break

            # Fallback to "first symbol" if nothing matches – better than empty
            if symbol is None:
                symbol = next(iter(historical.keys()))

            sym_block = historical.get(symbol)
            if isinstance(sym_block, dict):
                tf_candidates: List[np.ndarray] = []
                for tf in ("M15", "H1", "H4", "D1"):
                    candidate = sym_block.get(tf)
                    if not isinstance(candidate, dict):
                        continue
                    seq = candidate.get(price_type)
                    if isinstance(seq, (list, np.ndarray)):
                        tf_candidates.append(np.array(seq, dtype=float))
                if tf_candidates:
                    return _best_array_from_candidates(tf_candidates)

        return np.array([])

    def _analyze_volatility_regime(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Analyze volatility regime using ATR and historical volatility.

        Returns:
            regime: 'low_vol', 'normal_vol', 'high_vol', 'extreme_vol'
            score: 0–1 normalized volatility score (higher = more volatile)
        """
        # ATR proxy
        if len(high) >= self.atr_period and len(low) >= self.atr_period:
            tr_list = []
            # Use last atr_period bars; indexing is robust once len >= atr_period
            for i in range(1, min(self.atr_period + 1, len(close))):
                idx = -self.atr_period + i
                tr_val = max(
                    high[idx] - low[idx],
                    abs(high[idx] - close[idx - 1]),
                    abs(low[idx] - close[idx - 1]),
                )
                tr_list.append(tr_val)
            atr = np.mean(tr_list) if tr_list else np.std(close[-20:])
        else:
            atr = np.std(close[-20:]) if len(close) >= 20 else np.std(close)

        # Historical vol (annualized)
        if len(close) >= self.vol_lookback:
            returns = np.diff(np.log(close[-self.vol_lookback :]))
        else:
            returns = np.diff(np.log(close))
        hist_vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

        # Realized volatility percentile vs rolling distribution
        if len(close) >= self.vol_lookback * 2:
            rolling_vols = []
            for i in range(self.vol_lookback, len(close)):
                window_returns = np.diff(np.log(close[i - self.vol_lookback : i]))
                if len(window_returns) > 1:
                    rolling_vols.append(np.std(window_returns) * np.sqrt(252))

            if rolling_vols and max(rolling_vols) > min(rolling_vols):
                current_percentile = (hist_vol - min(rolling_vols)) / (
                    (max(rolling_vols) - min(rolling_vols)) + 1e-8
                )
            else:
                current_percentile = 0.5
        else:
            current_percentile = 0.5

        # ATR as pct of price
        last_price = float(close[-1]) if len(close) > 0 else 0.0
        atr_pct = atr / last_price if last_price > 0 else 0.0

        vol_score = current_percentile * 0.6 + min(atr_pct * 100, 1.0) * 0.4
        vol_score = float(np.clip(vol_score, 0.0, 1.0))

        if vol_score < self.vol_regime_threshold_low:
            regime = "low_vol"
        elif vol_score > self.vol_regime_threshold_high:
            regime = "extreme_vol" if vol_score > 0.9 else "high_vol"
        else:
            regime = "normal_vol"

        return regime, vol_score

    def _analyze_trend_regime(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> Tuple[str, float]:
        """
        Analyze trend regime using ADX-like logic and directional movement.

        Returns:
            regime: 'strong_uptrend', 'weak_uptrend', 'ranging',
                    'weak_downtrend', 'strong_downtrend'
            score: -1 to 1 directional trend score (sign = direction)
        """
        n = len(close)
        if n < self.adx_period + 1:
            return "ranging", 0.0

        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)

        for i in range(1, n):
            up_move = high[i] - high[i - 1]
            down_move = low[i - 1] - low[i]

            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move

            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        smoothed_plus_dm = self._wilder_smooth(plus_dm, self.adx_period)
        smoothed_minus_dm = self._wilder_smooth(minus_dm, self.adx_period)
        smoothed_tr = self._wilder_smooth(tr, self.adx_period)

        plus_di = 100 * smoothed_plus_dm / (smoothed_tr + 1e-8)
        minus_di = 100 * smoothed_minus_dm / (smoothed_tr + 1e-8)

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = self._wilder_smooth(dx, self.adx_period)

        current_adx = float(adx[-1])
        current_plus_di = float(plus_di[-1])
        current_minus_di = float(minus_di[-1])

        di_diff = current_plus_di - current_minus_di
        di_sum = current_plus_di + current_minus_di + 1e-8
        direction = di_diff / di_sum

        trend_score = float(np.clip(direction * (current_adx / 50.0), -1.0, 1.0))

        if current_adx < self.trend_strength_threshold:
            regime = "ranging"
        elif current_adx >= self.strong_trend_threshold:
            regime = "strong_uptrend" if direction > 0 else "strong_downtrend"
        else:
            regime = "weak_uptrend" if direction > 0 else "weak_downtrend"

        return regime, trend_score

    def _analyze_multi_timeframe_theme(self, instrument: str) -> Dict[str, Any]:
        """
        Analyze theme signals across multiple timeframes for confirmation.

        Multi-timeframe confirmation:
          • M15: PRIMARY trading timeframe (signal generation)
          • H1: Hourly confirmation (context)
          • H4: 4-hour filter (context)
          • D1: Daily trend direction (context)

        Returns a dict with:
          - trend_aligned: bool
          - trend_opposed: bool
          - vol_consistent: bool
          - m15_* / h1_* / h4_* / d1_* trend and vol fields
          - valid: bool
        """
        result = {
            "valid": False,
            "trend_aligned": False,
            "trend_opposed": False,
            "vol_consistent": False,
            "m15_trend": 0.0,
            "h1_trend": 0.0,
            "h4_trend": 0.0,
            "d1_trend": 0.0,
            "m15_vol": 0.0,
            "h1_vol": 0.0,
            "h4_vol": 0.0,
            "d1_vol": 0.0,
        }

        try:
            historical = self.smart_bus.get(
                "historical_prices", self.module_name, default=None
            )
            if not isinstance(historical, dict) or not historical:
                return result

            inst_norm = normalize_instrument(instrument)
            inst_aliases = {
                "EURUSD": ["EUR_USD", "EURUSD"],
                "XAUUSD": ["XAU_USD", "XAUUSD", "GOLDUSD"],
            }
            aliases = inst_aliases.get(inst_norm, [inst_norm, instrument])

            symbol = None
            for alias in aliases:
                if alias in historical:
                    symbol = alias
                    break
            if symbol is None:
                return result

            sym_block = historical.get(symbol)
            if not isinstance(sym_block, dict):
                return result

            # Analyze each timeframe
            tf_trends: Dict[str, float] = {}
            tf_vols: Dict[str, float] = {}

            for tf in ("M15", "H1", "H4", "D1"):
                tf_data = sym_block.get(tf)
                if not isinstance(tf_data, dict):
                    continue

                close = tf_data.get("close")
                high = tf_data.get("high")
                low = tf_data.get("low")

                if not all(
                    isinstance(x, (list, np.ndarray)) for x in (close, high, low)
                ):
                    continue

                close_arr = np.array(close, dtype=float)
                high_arr = np.array(high, dtype=float)
                low_arr = np.array(low, dtype=float)

                # Require fewer bars on M15 vs higher TFs
                min_required = 10 if tf == "M15" else 20
                if len(close_arr) < min_required:
                    continue

                # Simple SMA-based trend direction for this TF
                sma_short = np.mean(close_arr[-10:])
                sma_long = np.mean(close_arr[-20:]) if len(close_arr) >= 20 else sma_short

                if sma_long > 0:
                    trend_direction = float((sma_short - sma_long) / sma_long)
                else:
                    trend_direction = 0.0

                tf_trends[tf] = trend_direction

                # Volatility proxy for this TF (ATR% over last 14 bars)
                if len(high_arr) >= 14 and len(low_arr) >= 14:
                    tr_vals = high_arr[-14:] - low_arr[-14:]
                    atr = float(np.mean(tr_vals))
                    avg_price = float(np.mean(close_arr[-14:]))
                    vol_pct = atr / avg_price if avg_price > 0 else 0.0
                    tf_vols[tf] = vol_pct

            if len(tf_trends) < 2:
                return result

            result["valid"] = True

            # Store individual TF values
            result["m15_trend"] = tf_trends.get("M15", 0.0)
            result["h1_trend"] = tf_trends.get("H1", 0.0)
            result["h4_trend"] = tf_trends.get("H4", 0.0)
            result["d1_trend"] = tf_trends.get("D1", 0.0)
            result["m15_vol"] = tf_vols.get("M15", 0.0)
            result["h1_vol"] = tf_vols.get("H1", 0.0)
            result["h4_vol"] = tf_vols.get("H4", 0.0)
            result["d1_vol"] = tf_vols.get("D1", 0.0)

            # Check trend alignment (M15/H1 vs H4/D1)
            m15_trend = tf_trends.get("M15", 0.0)
            h1_trend = tf_trends.get("H1", 0.0)
            h4_trend = tf_trends.get("H4", 0.0)
            d1_trend = tf_trends.get("D1", 0.0)

            signs = [
                np.sign(m15_trend),
                np.sign(h1_trend),
                np.sign(h4_trend),
                np.sign(d1_trend),
            ]
            non_zero_signs = [s for s in signs if s != 0]

            if len(non_zero_signs) >= 2:
                if all(s > 0 for s in non_zero_signs) or all(
                    s < 0 for s in non_zero_signs
                ):
                    result["trend_aligned"] = True
                else:
                    # Check if lower TFs oppose higher TFs
                    ltf_signs = [np.sign(m15_trend), np.sign(h1_trend)]
                    ltf_signs = [s for s in ltf_signs if s != 0]
                    htf_signs = [np.sign(h4_trend), np.sign(d1_trend)]
                    htf_signs = [s for s in htf_signs if s != 0]

                    if ltf_signs and htf_signs:
                        ltf_consensus = (
                            ltf_signs[0] if len(set(ltf_signs)) == 1 else 0
                        )
                        htf_consensus = (
                            htf_signs[0] if len(set(htf_signs)) == 1 else 0
                        )
                        if (
                            ltf_consensus != 0
                            and htf_consensus != 0
                            and ltf_consensus != htf_consensus
                        ):
                            result["trend_opposed"] = True

            # Volatility consistency across TFs (coefficient of variation)
            if len(tf_vols) >= 2:
                vol_values = list(tf_vols.values())
                vol_std = float(np.std(vol_values))
                vol_mean = float(np.mean(vol_values))

                if vol_mean > 0:
                    cv = vol_std / vol_mean
                    result["vol_consistent"] = cv < 0.5

            self.log_debug(
                f"[THEME MTF] {inst_norm}: "
                f"M15={m15_trend:.4f}, H1={h1_trend:.4f}, "
                f"H4={h4_trend:.4f}, D1={d1_trend:.4f}, "
                f"aligned={result['trend_aligned']}, "
                f"opposed={result['trend_opposed']}"
            )

        except Exception as e:
            self.log_debug(f"[THEME MTF] Error analyzing {instrument}: {e}")

        return result

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        Apply Wilder's smoothing method.

        Note: For the initial period, this uses a simple average; subsequent
        values use recursive smoothing with alpha = 1/period.
        """
        result = np.zeros_like(data)
        if period <= 0 or len(data) == 0:
            return result

        # First "true" Wilder value
        if len(data) >= period:
            initial = float(np.sum(data[:period])) / period
            result[period - 1] = initial
        else:
            # If not enough data, fallback to simple running mean
            result[-1] = float(np.mean(data))
            return result

        alpha = 1.0 / period
        for i in range(period, len(data)):
            result[i] = result[i - 1] * (1 - alpha) + data[i] * alpha

        return result

    def _analyze_correlation_regime(
        self,
        close: np.ndarray,
        market_data: Dict,
    ) -> Tuple[str, float]:
        """
        Analyze correlation regime across timeframes for a single instrument.

        Returns:
            regime: 'risk_on_correlation', 'risk_off_correlation', 'decorrelated'
            score: average correlation / autocorrelation measure
        """
        prices_by_tf: Dict[str, np.ndarray] = {}

        if isinstance(market_data, dict):
            for tf in ("M15", "H1", "H4", "D1"):
                tf_block = market_data.get(tf)
                if isinstance(tf_block, dict) and "close" in tf_block:
                    tf_close = np.array(tf_block["close"], dtype=float)
                    if len(tf_close) >= self.corr_lookback:
                        prices_by_tf[tf] = tf_close[-self.corr_lookback :]

        if len(prices_by_tf) >= 2:
            correlations: List[float] = []
            tfs = list(prices_by_tf.keys())

            for i in range(len(tfs)):
                for j in range(i + 1, len(tfs)):
                    p1 = prices_by_tf[tfs[i]]
                    p2 = prices_by_tf[tfs[j]]
                    min_len = min(len(p1), len(p2))
                    r1 = np.diff(np.log(p1[-min_len:]))
                    r2 = np.diff(np.log(p2[-min_len:]))

                    if len(r1) > 5 and len(r2) > 5:
                        corr = float(np.corrcoef(r1, r2)[0, 1])
                        if not np.isnan(corr):
                            correlations.append(corr)

            if correlations:
                avg_corr = float(np.mean(correlations))
                if avg_corr > self.corr_cluster_threshold:
                    return "risk_on_correlation", avg_corr
                if avg_corr < -self.corr_cluster_threshold:
                    return "risk_off_correlation", avg_corr
                return "decorrelated", avg_corr

        # Fallback: simple return autocorrelation if cross-TF isn't available
        if len(close) >= self.corr_lookback:
            returns = np.diff(np.log(close[-self.corr_lookback :]))
            if len(returns) > 1:
                autocorr = float(np.corrcoef(returns[:-1], returns[1:])[0, 1])
                if not np.isnan(autocorr):
                    if autocorr > 0.3:
                        return "risk_on_correlation", autocorr
                    if autocorr < -0.3:
                        return "risk_off_correlation", autocorr

        return "decorrelated", 0.0

    def _calculate_market_breadth(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
    ) -> float:
        """
        Calculate market breadth score for this instrument.

        Components:
          • Position of price within recent range
          • Balance of up vs down closes
          • Recent new highs / lows
        """
        if len(close) < self.breadth_period:
            return 0.5

        period_close = close[-self.breadth_period :]
        period_high = (
            high[-self.breadth_period :]
            if len(high) >= self.breadth_period
            else period_close
        )
        period_low = (
            low[-self.breadth_period :]
            if len(low) >= self.breadth_period
            else period_close
        )

        period_range = float(np.max(period_high) - np.min(period_low))
        if period_range > 0:
            position_score = float(
                (close[-1] - np.min(period_low)) / period_range
            )
        else:
            position_score = 0.5

        returns = np.diff(period_close)
        up_days = int(np.sum(returns > 0))
        down_days = int(np.sum(returns < 0))
        total_days = up_days + down_days
        breadth_ratio = float(up_days / total_days) if total_days > 0 else 0.5

        if len(close) >= 5:
            recent_high = float(np.max(close[-5:]))
            recent_low = float(np.min(close[-5:]))
        else:
            recent_high = float(close[-1])
            recent_low = float(close[-1])

        period_high_max = float(np.max(period_high))
        period_low_min = float(np.min(period_low))

        new_high_score = 1.0 if recent_high >= period_high_max else 0.5
        new_low_score = 0.0 if recent_low <= period_low_min else 0.5

        breadth_score = (
            position_score * 0.35
            + breadth_ratio * 0.35
            + new_high_score * 0.15
            + new_low_score * 0.15
        )

        return float(np.clip(breadth_score, 0.0, 1.0))

    def _calculate_momentum_score(self, close: np.ndarray) -> float:
        """
        Calculate momentum score using multi-horizon rate-of-change.

        Returns:
            score: 0–1 normalized (higher = more bullish momentum)
        """
        if len(close) < 20:
            return 0.5

        roc_5 = (
            (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0.0
        )

        roc_10 = 0.0
        if len(close) > 10 and close[-11] > 0:
            roc_10 = (close[-1] - close[-11]) / close[-11]

        roc_20 = 0.0
        if len(close) > 20 and close[-21] > 0:
            roc_20 = (close[-1] - close[-21]) / close[-21]

        # Normalized to [0, 1] around 0-return baseline
        norm_roc_5 = np.clip(roc_5 * 10 + 0.5, 0.0, 1.0)
        norm_roc_10 = np.clip(roc_10 * 5 + 0.5, 0.0, 1.0)
        norm_roc_20 = np.clip(roc_20 * 2.5 + 0.5, 0.0, 1.0)

        momentum_score = (
            norm_roc_5 * 0.5 + norm_roc_10 * 0.3 + norm_roc_20 * 0.2
        )
        return float(momentum_score)

    def _calculate_composite_sentiment(
        self,
        vol_score: float,
        trend_score: float,
        corr_score: float,
        breadth_score: float,
        momentum_score: float,
    ) -> float:
        """
        Calculate composite sentiment score (0–1).

        Combines all regime scores into a single sentiment value:
          • Lower volatility → higher sentiment (inv_vol)
          • Trend_score mapped from [-1, 1] → [0, 1]
          • Correlation_score mapped from [-1, 1] → [0, 1]
        """
        norm_trend = (trend_score + 1.0) / 2.0
        inv_vol = 1.0 - vol_score
        norm_corr = (corr_score + 1.0) / 2.0

        composite = (
            inv_vol * self.sentiment_weights["volatility"]
            + norm_trend * self.sentiment_weights["trend"]
            + norm_corr * self.sentiment_weights["correlation"]
            + breadth_score * self.sentiment_weights["breadth"]
            + momentum_score * self.sentiment_weights["momentum"]
        )

        return float(np.clip(composite, 0.0, 1.0))

    def _determine_risk_regime(
        self,
        vol_regime: str,
        trend_regime: str,
        composite_score: float,
        asset_class: str = "forex",
    ) -> str:
        """
        Determine overall risk regime label.

        Notes:
          • For commodities (like Gold), "risk-off" often maps to LONG themes.
          • For FX, "risk-off" usually maps to cautious / defensive themes.
        """
        # Extreme volatility overrides everything
        if vol_regime == "extreme_vol":
            return "risk_off"

        if composite_score >= self.risk_on_threshold:
            if trend_regime in ("strong_uptrend", "weak_uptrend"):
                return "risk_on"
            return "neutral"

        if composite_score <= self.risk_off_threshold:
            if trend_regime in ("strong_downtrend", "weak_downtrend"):
                return "risk_off"
            return "cautious"

        return "neutral"

    def _select_theme_action(
        self,
        vol_regime: str,
        trend_regime: str,
        risk_regime: str,
        vol_score: float,
        trend_score: float,
        composite_score: float,
        instrument: str = "",
    ) -> Tuple[str, float, str]:
        """
        Select trading theme action based on regime analysis.

        Instrument-aware:
          • Gold behaves as safe haven in risk-off / extreme volatility.
        Returns:
          (action, confidence, thesis)
        """
        inst_norm = normalize_instrument(instrument) if instrument else ""
        is_safe_haven = inst_norm in ("XAUUSD", "GOLD", "XAU")

        # Extreme volatility: flat for most, long for Gold (safe haven)
        if vol_regime == "extreme_vol":
            if is_safe_haven:
                confidence = 0.6 + (vol_score - 0.9) * 2.0
                confidence = float(np.clip(confidence, 0.5, 0.75))
                return (
                    "long",
                    confidence,
                    f"{inst_norm}: Safe haven bid during extreme volatility "
                    f"(vol_score={vol_score:.2f})",
                )

            confidence = 0.7 + (vol_score - 0.9) * 3.0
            confidence = float(np.clip(confidence, 0.6, 0.9))
            return (
                "flat",
                confidence,
                f"{inst_norm}: Extreme volatility (score={vol_score:.2f}), "
                "defensive positioning",
            )

        # Strong uptrend → long
        if trend_regime == "strong_uptrend":
            confidence = self.base_confidence + abs(trend_score) * 0.4
            confidence = float(np.clip(confidence, 0.5, 0.85))
            return (
                "long",
                confidence,
                f"{inst_norm}: Strong uptrend (trend_score={trend_score:.2f}), "
                "trend following",
            )

        # Strong downtrend → short, except Gold in risk-off → long
        if trend_regime == "strong_downtrend":
            confidence = self.base_confidence + abs(trend_score) * 0.4
            confidence = float(np.clip(confidence, 0.5, 0.85))
            if is_safe_haven and risk_regime in ("risk_off", "cautious"):
                return (
                    "long",
                    float(np.clip(confidence * 0.8, 0.4, 0.7)),
                    f"{inst_norm}: Safe haven bid during risk-off "
                    f"(trend_score={trend_score:.2f})",
                )
            return (
                "short",
                confidence,
                f"{inst_norm}: Strong downtrend (trend_score={trend_score:.2f}), "
                "bearish theme",
            )

        # Risk-on environment
        if risk_regime == "risk_on" and composite_score > 0.55:
            confidence = self.base_confidence + (composite_score - 0.5) * 0.6
            confidence = float(np.clip(confidence, 0.5, 0.8))
            return (
                "long",
                confidence,
                f"{inst_norm}: Risk-on (composite={composite_score:.2f}), bullish theme",
            )

        # Risk-off / cautious environment
        if risk_regime in ("risk_off", "cautious") and composite_score < 0.45:
            confidence = self.base_confidence + (0.5 - composite_score) * 0.6
            confidence = float(np.clip(confidence, 0.5, 0.8))
            if is_safe_haven:
                return (
                    "long",
                    float(np.clip(confidence, 0.5, 0.75)),
                    f"{inst_norm}: Safe haven bid in risk-off "
                    f"(composite={composite_score:.2f})",
                )
            return (
                "short",
                confidence,
                f"{inst_norm}: Risk-off (composite={composite_score:.2f}), bearish theme",
            )

        # Weak trends
        if trend_regime == "weak_uptrend":
            return (
                "long",
                0.45,
                f"{inst_norm}: Weak uptrend, cautious bullish theme",
            )

        if trend_regime == "weak_downtrend":
            if is_safe_haven:
                return (
                    "flat",
                    0.35,
                    f"{inst_norm}: Weak downtrend but safe haven - neutral theme",
                )
            return (
                "short",
                0.45,
                f"{inst_norm}: Weak downtrend, cautious bearish theme",
            )

        # Ranging: direction from composite bias
        if trend_regime == "ranging":
            if composite_score > 0.5:
                return (
                    "long",
                    0.35,
                    f"{inst_norm}: Ranging with bullish bias "
                    f"(composite={composite_score:.2f})",
                )
            if composite_score < 0.5:
                return (
                    "short",
                    0.35,
                    f"{inst_norm}: Ranging with bearish bias "
                    f"(composite={composite_score:.2f})",
                )
            return (
                "flat",
                0.3,
                f"{inst_norm}: Ranging - neutral composite theme",
            )

        # Default mixed-case fallback
        if composite_score > 0.5:
            return (
                "long",
                0.3,
                f"{inst_norm}: Mixed signals favoring long "
                f"(composite={composite_score:.2f})",
            )
        if composite_score < 0.5:
            return (
                "short",
                0.3,
                f"{inst_norm}: Mixed signals favoring short "
                f"(composite={composite_score:.2f})",
            )

        return (
            "flat",
            0.25,
            f"{inst_norm}: Neutral - vol={vol_regime}, trend={trend_regime}, "
            f"risk={risk_regime}",
        )

    def _apply_persistence_filter(
        self,
        action: str,
        confidence: float,
    ) -> Tuple[str, float]:
        """
        Apply regime persistence filter to GLOBAL theme action.

        - Boosts confidence for persistent regimes.
        - Dampens confidence for brand-new flips.
        """
        if self.regime_history and self.regime_history[-1] == action:
            self.regime_persistence_count += 1
            if self.regime_persistence_count >= self.min_regime_persistence:
                boost = min(0.1, self.regime_persistence_count * 0.02)
                confidence = min(confidence + boost, self.max_confidence)
        else:
            self.regime_persistence_count = 1
            confidence = max(confidence * 0.85, self.min_confidence)

        self.regime_history.append(action)
        if len(self.regime_history) > 20:
            self.regime_history = self.regime_history[-20:]

        return action, confidence

    def _neutral_output(self, reason: str) -> Dict[str, Any]:
        """
        Generate neutral output with explanation and publish neutral keys.

        Used when there is no usable market data.
        """
        thesis = f"Theme flat: {reason}"
        name = self.__class__.__name__

        conf_floor = CONFIDENCE_THRESHOLD_F()
        confidence = max(0.1, conf_floor * 0.5)

        proposal = {
            "action": "flat",
            "signal_strength": confidence,
            "reason": thesis,
            "proposals": {},
        }

        theme_analysis = {
            "volatility_regime": "unknown",
            "trend_regime": "unknown",
            "risk_regime": "unknown",
            "composite_score": 0.5,
            "action": "flat",
            "confidence": confidence,
            "per_instrument": {},
        }

        try:
            # Canonical expert keys
            self.smart_bus.set(
                "ThemeExpert_voting_proposal",
                proposal,
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "ThemeExpert_confidence",
                confidence,
                module=name,
                thesis=f"Confidence: {confidence:.1%}",
            )
            # Contract aliases
            self.smart_bus.set(
                "theme_voting_proposal",
                proposal,
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "theme_confidence",
                confidence,
                module=name,
                thesis=f"Theme confidence: {confidence:.1%}",
            )
            self.smart_bus.set(
                "theme_analysis",
                theme_analysis,
                module=name,
                thesis=thesis,
            )
        except Exception:
            pass

        return {
            "ThemeExpert_voting_proposal": proposal,
            "ThemeExpert_confidence": confidence,
            "ThemeExpert_per_instrument_votes": {},
            "per_instrument_votes": PerInstrumentVote(member=name),
            "theme_voting_proposal": proposal,
            "theme_confidence": confidence,
            "theme_analysis": theme_analysis,
            "theme_agreement_score": confidence,
            "theme_expert_analysis": {
                "volatility_regime": "unknown",
                "trend_regime": "unknown",
                "risk_regime": "unknown",
                "composite_score": 0.5,
                "per_instrument": {},
            },
            "theme_expert_thesis": thesis,
            "theme_volatility_regime": "unknown",
            "theme_trend_regime": "unknown",
            "theme_risk_regime": "unknown",
            "theme_composite_score": 0.5,
            "_thesis": thesis,
        }

    def _degraded_output(self, reason: str) -> Dict[str, Any]:
        """
        Degraded-mode output used when circuit breaker is open or an error occurs.

        Wraps _neutral_output but annotates the payload with error metadata
        so monitoring can distinguish "no data" from "error path".
        """
        base = self._neutral_output(f"degraded: {reason}")
        base["theme_degraded"] = True
        base["theme_error_reason"] = str(reason)
        return base

    # ═══════════════════════════════════════════════════════════════════
    # STATE PERSISTENCE - Save/Load module state
    # ═══════════════════════════════════════════════════════════════════

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.

        Saves:
          • Regime history (for persistence filtering)
          • Regime persistence count
          • Confidence calibration settings / sentiment weights
        """
        return {
            "regime_history": list(self.regime_history)[-20:],  # Keep last 20
            "regime_persistence_count": self.regime_persistence_count,
            "base_confidence": self.base_confidence,
            "sentiment_weights": dict(self.sentiment_weights),
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """
        Restore custom state from persistence.
        """
        if not state:
            return

        # Restore regime history & persistence counter
        regime_hist = state.get("regime_history", [])
        self.regime_history = list(regime_hist)
        self.regime_persistence_count = int(
            state.get("regime_persistence_count", 0)
        )

        # Restore confidence settings (if customized)
        if "base_confidence" in state:
            self.base_confidence = float(state["base_confidence"])

        # Restore sentiment weights (if customized)
        weights = state.get("sentiment_weights", {})
        if weights:
            self.sentiment_weights.update(weights)

        self.log_info(
            f"📂 ThemeExpert state restored | "
            f"regime_history={len(self.regime_history)} | "
            f"persistence={self.regime_persistence_count}"
        )
