"""
Advanced ThemeExpert - Macro Theme and Risk Regime Voting Module.

This expert analyzes macro themes and market regimes including:
- Volatility regime analysis (ATR, historical vol, implied vol proxy)
- Correlation cluster analysis (cross-asset correlations)
- Trend strength aggregation (ADX across timeframes)
- Market breadth indicators
- Sentiment aggregation (composite scoring)
- Risk-on/Risk-off scoring

Per-Instrument Voting:
- Analyzes each instrument separately to produce per-instrument votes
- Different asset classes (FX, commodities) may have different regimes
- Each instrument gets its own action/confidence based on its own data

Actions: long, short, flat
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

    Combines multiple regime detection methods to identify optimal
    trading themes and risk positioning.

    Per-instrument:
    - EURUSD, XAUUSD, etc. each get their own macro/regime vote
    """

    # ═══════════════════════════ INIT ═══════════════════════════

    def _expert_specific_init(self) -> None:
        """Initialize the theme expert with configuration."""
        self.module_name = self.__class__.__name__

        # Instruments to analyze (from config or default)
        self.instruments = self.config.get("instruments", ["EURUSD", "XAUUSD"])

        # Asset class mapping for different regime interpretations
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

        # Correlation configuration
        self.corr_lookback = int(self.config.get("corr_lookback", 30))
        self.corr_cluster_threshold = float(
            self.config.get("corr_cluster_threshold", 0.7)
        )

        # Trend configuration
        self.adx_period = int(self.config.get("adx_period", 14))
        self.trend_strength_threshold = float(
            self.config.get("trend_strength_threshold", 25.0)
        )
        self.strong_trend_threshold = float(
            self.config.get("strong_trend_threshold", 40.0)
        )

        # Risk scoring thresholds
        self.risk_on_threshold = float(self.config.get("risk_on_threshold", 0.6))
        self.risk_off_threshold = float(self.config.get("risk_off_threshold", 0.4))

        # Breadth configuration
        self.breadth_period = int(self.config.get("breadth_period", 20))

        # Sentiment weights
        self.sentiment_weights = {
            "volatility": 0.25,
            "trend": 0.25,
            "correlation": 0.20,
            "breadth": 0.15,
            "momentum": 0.15,
        }

        # Historical state for regime persistence (global theme)
        self.regime_history: List[str] = []
        self.regime_persistence_count = 0
        self.min_regime_persistence = int(
            self.config.get("min_regime_persistence", 3)
        )

        # Confidence calibration
        self.base_confidence = float(self.config.get("base_confidence", 0.5))
        self.max_confidence = float(self.config.get("max_confidence", 0.95))
        self.min_confidence = float(self.config.get("min_confidence", 0.15))

        self.log_info(
            f"[THEME] ThemeExpert initialized | "
            f"instruments={self.instruments} | ATR period={self.atr_period}"
        )

        # Publish baseline keys
        self._publish_baseline_keys()
        self._publish_theme_baseline()

    def _publish_theme_baseline(self) -> None:
        """Publish baseline theme keys to avoid stale readers."""
        thesis = "Theme baseline"
        proposal = {
            "action": "flat",
            "signal_strength": 0.1,
            "reason": thesis,
            "proposals": {},
        }
        try:
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
                {
                    "volatility_regime": "unknown",
                    "trend_regime": "unknown",
                    "risk_regime": "unknown",
                    "composite_score": 0.5,
                    "action": "flat",
                    "confidence": 0.1,
                    "per_instrument": {},
                },
                module=self.module_name,
                thesis="Theme baseline analysis",
            )
        except Exception:
            pass

    # ═══════════════════════════ VOTINGEXPERTBASE HOOKS (LEGACY) ═══════════════════════════

    async def _generate_expert_specific_proposal(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Legacy hook for VotingExpertBase.

        ThemeExpert is primarily per-instrument via process(), so this is a
        neutral fallback used only if some generic caller invokes the base
        template method.
        """
        return {
            "action": "flat",
            "signal_strength": 0.0,
            "reason": "ThemeExpert uses per-instrument process() path",
        }

    async def _calculate_expert_specific_confidence(
        self, proposal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> float:
        """Legacy confidence hook; neutral default."""
        return 0.3

    # ═══════════════════════════ MAIN PER-INSTRUMENT PROCESS ═══════════════════════════

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Process market data to determine macro theme and regime PER INSTRUMENT.

        - Each instrument gets its own InstrumentProposal.
        - Different assets can be in different regimes (e.g. XAU risk-on while EURUSD risk-off).
        - Returns a global "ThemeExpert_voting_proposal" for backward compatibility,
          plus per-instrument votes under ThemeExpert_per_instrument_votes.
        """
        name = self.__class__.__name__
        start = self._now_ms()

        try:
            # Circuit breaker
            if self._check_circuit_breaker():
                return self._degraded_output("circuit_breaker_open")

            # Pull data from InfoBus
            market_data = self.smart_bus.get("market_data", name, default={})
            features = self.smart_bus.get("features", name, default={})

            if not market_data and not features:
                self.log_warning("[THEME] No market data or features available")
                output = self._neutral_output("No market data available")
            else:
                per_instrument_vote = PerInstrumentVote(member=name)
                per_instrument_analysis: Dict[str, Dict[str, Any]] = {}

                # ── Per-instrument regime analysis ─────────────────────────────
                for inst in self.instruments:
                    inst_norm = normalize_instrument(inst)

                    inst_market = self._extract_instrument_data(market_data, inst)
                    inst_features = self._extract_instrument_data(features, inst)

                    close_prices = self._extract_prices(
                        inst_market, inst_features, "close", inst_norm
                    )
                    high_prices = self._extract_prices(
                        inst_market, inst_features, "high", inst_norm
                    )
                    low_prices = self._extract_prices(
                        inst_market, inst_features, "low", inst_norm
                    )

                    if len(close_prices) < self.vol_lookback:
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
                        }
                        continue

                    # ── Regime components ─────────────────────────────────────
                    vol_regime, vol_score = self._analyze_volatility_regime(
                        close_prices, high_prices, low_prices
                    )
                    trend_regime, trend_score = self._analyze_trend_regime(
                        close_prices, high_prices, low_prices
                    )
                    corr_regime, corr_score = self._analyze_correlation_regime(
                        close_prices, inst_market
                    )
                    
                    # ── MTF confirmation for trend/vol regimes ───────────────
                    mtf_result = self._analyze_multi_timeframe_theme(inst_norm)
                    if mtf_result["valid"]:
                        # Adjust trend score based on MTF alignment
                        if mtf_result["trend_aligned"]:
                            trend_score = trend_score * 1.15  # +15% for MTF alignment
                            self.log_debug(
                                f"[THEME MTF] {inst_norm}: Trend aligned across TFs, boosting score"
                            )
                        elif mtf_result["trend_opposed"]:
                            trend_score = trend_score * 0.75  # -25% for MTF opposition
                            self.log_debug(
                                f"[THEME MTF] {inst_norm}: Trend opposed across TFs, reducing score"
                            )
                        
                        # Adjust vol score based on MTF volatility consistency
                        if mtf_result["vol_consistent"]:
                            vol_score = vol_score * 0.95  # Slight adjustment for consistency
                        
                        trend_score = float(np.clip(trend_score, -1, 1))
                    breadth_score = self._calculate_market_breadth(
                        close_prices, high_prices, low_prices
                    )
                    momentum_score = self._calculate_momentum_score(close_prices)

                    composite_score = self._calculate_composite_sentiment(
                        vol_score,
                        trend_score,
                        corr_score,
                        breadth_score,
                        momentum_score,
                    )

                    asset_class = self.asset_classes.get(inst_norm, "forex")
                    risk_regime = self._determine_risk_regime(
                        vol_regime,
                        trend_regime,
                        composite_score,
                        asset_class=asset_class,
                    )

                    # ── Instrument theme action ──────────────────────────────
                    action, confidence, thesis = self._select_theme_action(
                        vol_regime,
                        trend_regime,
                        risk_regime,
                        vol_score,
                        trend_score,
                        composite_score,
                        instrument=inst_norm,
                    )

                    # Calibrate magnitude (theme is overlay, not primary hammer)
                    min_strength = MIN_SIGNAL_STRENGTH_F()
                    if action == "flat":
                        magnitude = 0.0
                    else:
                        magnitude = float(
                            max(min_strength * 0.5, min(1.0, confidence * 0.8))
                        )

                    per_instrument_vote.set_proposal(
                        InstrumentProposal(
                            instrument=inst_norm,
                            action=action,
                            confidence=confidence,
                            magnitude=magnitude,
                            rationale=thesis,
                        )
                    )

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
                        "breadth_score": breadth_score,
                        "momentum_score": momentum_score,
                    }

                    self.log_debug(
                        f"[THEME] {inst_norm}: vol={vol_regime}, "
                        f"trend={trend_regime}, risk={risk_regime}, "
                        f"action={action}, conf={confidence:.2f}, "
                        f"comp={composite_score:.2f}"
                    )

                # ── Global summary / backward compatibility ──────────────────
                if per_instrument_analysis:
                    first_inst = list(per_instrument_analysis.keys())[0]
                    global_analysis = per_instrument_analysis[first_inst]
                else:
                    global_analysis = {
                        "volatility_regime": "unknown",
                        "trend_regime": "unknown",
                        "risk_regime": "unknown",
                        "composite_score": 0.5,
                        "action": "flat",
                        "confidence": 0.1,
                    }

                if per_instrument_vote.proposals:
                    best_proposal = max(
                        per_instrument_vote.proposals.values(),
                        key=lambda p: p.confidence,
                    )
                    global_action = best_proposal.action
                    global_confidence = best_proposal.confidence
                    global_thesis = best_proposal.rationale
                else:
                    global_action = "flat"
                    global_confidence = 0.1
                    global_thesis = "No instrument data available"

                # Persistence filter on GLOBAL theme action (not per instrument)
                global_action, global_confidence = self._apply_persistence_filter(
                    global_action, global_confidence
                )

                # Clip with global thresholds
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

                # Main proposal for voting
                proposal = {
                    "action": global_action,
                    "signal_strength": global_confidence,
                    "reason": global_thesis,
                    "proposals": proposals_dict,
                }

                # Publish to SmartInfoBus
                try:
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
                    "agreement_score": global_confidence,
                    "theme_expert_analysis": {
                        "volatility_regime": global_analysis.get(
                            "volatility_regime", "unknown"
                        ),
                        "trend_regime": global_analysis.get(
                            "trend_regime", "unknown"
                        ),
                        "risk_regime": global_analysis.get("risk_regime", "unknown"),
                        "composite_score": global_analysis.get(
                            "composite_score", 0.5
                        ),
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

            # Perf metrics / success
            elapsed_ms = self._elapsed_ms(start)
            try:
                self.performance_tracker.record_metric(name, "process", elapsed_ms, True)
            except Exception:
                pass
            self._record_success()
            return output

        except Exception as e:
            # Error path: degraded mode
            self._record_error(e)
            if self.error_pinpointer is not None:
                err_ctx = self.error_pinpointer.analyze_error(e, f"{name}_process")
                msg = str(err_ctx)
            else:
                msg = str(e)

            self.log_error(f"[THEME] Process error: {msg}")

            elapsed_ms = self._elapsed_ms(start)
            try:
                self.performance_tracker.record_metric(name, "process", elapsed_ms, False)
            except Exception:
                pass

            return self._degraded_output(msg)

    # ═══════════════════════════ HELPERS ═══════════════════════════

    def _now_ms(self) -> float:
        """Monotonic ms helper for perf tracking."""
        import time

        return time.time() * 1000.0

    def _elapsed_ms(self, start_ms: float) -> float:
        import time

        return time.time() * 1000.0 - start_ms

    def _extract_instrument_data(self, data: Dict, instrument: str) -> Dict:
        """Extract data for a specific instrument from nested market data."""
        if not isinstance(data, dict):
            return {}

        inst_norm = normalize_instrument(instrument)

        # Try direct instrument key
        for key in [instrument, inst_norm, instrument.upper(), instrument.lower()]:
            if key in data:
                return data[key] if isinstance(data[key], dict) else data

        # Try with separators and common pairs
        for sep in ["_", "/", "-", ""]:
            for pair in [f"EUR{sep}USD", f"XAU{sep}USD"]:
                norm_pair = normalize_instrument(pair)
                if norm_pair == inst_norm and pair in data:
                    return data[pair] if isinstance(data[pair], dict) else data

        # Return full data if no instrument-specific found (legacy format)
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

        Args:
            market_data: Instrument-specific market data (may be pre-filtered)
            features: Instrument-specific features (may be pre-filtered)
            price_type: 'close', 'high', 'low', 'open'
            instrument: Target instrument (e.g., 'EURUSD', 'XAUUSD') for historical lookup
        """
        inst_norm = normalize_instrument(instrument) if instrument else ""

        def _best_array_from_candidates(candidates: List[np.ndarray]) -> np.ndarray:
            """Pick the best candidate, preferring sequences >= vol_lookback, then longest."""
            if not candidates:
                return np.array([])
            # First, candidates that satisfy lookback
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
                    arr = np.array(data, dtype=float)
                    candidates.append(arr)

            # Nested TF structure (M15 first, but fall back to higher TFs if M15 is short)
            for tf in ["M15", "H1", "H4", "D1"]:
                if tf in market_data and isinstance(market_data[tf], dict):
                    if price_type in market_data[tf]:
                        data = market_data[tf][price_type]
                        if isinstance(data, (list, np.ndarray)):
                            arr = np.array(data, dtype=float)
                            candidates.append(arr)

        # Features
        if isinstance(features, dict):
            if price_type in features:
                data = features[price_type]
                if isinstance(data, (list, np.ndarray)):
                    arr = np.array(data, dtype=float)
                    candidates.append(arr)

        # If we already found usable candidates, choose the best and return
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

            if symbol is None:
                symbol = next(iter(historical.keys()))

            sym_block = historical.get(symbol)
            if isinstance(sym_block, dict):
                tf_candidates: List[np.ndarray] = []
                for tf in ("M15", "H4", "H1", "D1"):
                    candidate = sym_block.get(tf)
                    if not isinstance(candidate, dict):
                        continue
                    seq = candidate.get(price_type)
                    if isinstance(seq, (list, np.ndarray)):
                        arr = np.array(seq, dtype=float)
                        tf_candidates.append(arr)
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
            score: 0-1 normalized volatility score
        """
        # ATR proxy
        if len(high) >= self.atr_period and len(low) >= self.atr_period:
            tr_list = []
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
            returns = np.diff(np.log(close[-self.vol_lookback:]))
        else:
            returns = np.diff(np.log(close))
        hist_vol = np.std(returns) * np.sqrt(252) if len(returns) > 1 else 0.0

        # Realized volatility percentile
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
        atr_pct = atr / close[-1] if close[-1] > 0 else 0.0

        vol_score = current_percentile * 0.6 + min(atr_pct * 100, 1.0) * 0.4
        vol_score = float(np.clip(vol_score, 0, 1))

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
        Analyze trend regime using ADX and directional movement.

        Returns:
            regime: 'strong_uptrend', 'weak_uptrend', 'ranging',
                    'weak_downtrend', 'strong_downtrend'
            score: -1 to 1 directional score
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

        trend_score = float(np.clip(direction * (current_adx / 50.0), -1, 1))

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
        - H1: Primary signal (40% weight)
        - H4: Confirmation signal (35% weight)
        - D1: Strategic direction (25% weight)
        
        Returns:
            Dict with trend_aligned, trend_opposed, vol_consistent, valid flags
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
            tf_trends = {}
            tf_vols = {}
            
            for tf in ["M15", "H1", "H4", "D1"]:
                tf_data = sym_block.get(tf)
                if not isinstance(tf_data, dict):
                    continue
                
                close = tf_data.get("close")
                high = tf_data.get("high")
                low = tf_data.get("low")
                
                if not all(isinstance(x, (list, np.ndarray)) for x in [close, high, low]):
                    continue
                
                close_arr = np.array(close, dtype=float)
                high_arr = np.array(high, dtype=float)
                low_arr = np.array(low, dtype=float)
                
                if len(close_arr) < 20:
                    continue
                
                # Calculate trend direction for this TF
                sma_short = np.mean(close_arr[-10:])
                sma_long = np.mean(close_arr[-20:])
                
                if sma_long > 0:
                    trend_direction = (sma_short - sma_long) / sma_long
                else:
                    trend_direction = 0.0
                
                tf_trends[tf] = trend_direction
                
                # Calculate volatility for this TF (ATR proxy)
                if len(high_arr) >= 14 and len(low_arr) >= 14:
                    tr_vals = high_arr[-14:] - low_arr[-14:]
                    atr = np.mean(tr_vals)
                    avg_price = np.mean(close_arr[-14:])
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
            
            # Check trend alignment (M15/H1 should align with H4/D1)
            m15_trend = tf_trends.get("M15", 0.0)
            h1_trend = tf_trends.get("H1", 0.0)
            h4_trend = tf_trends.get("H4", 0.0)
            d1_trend = tf_trends.get("D1", 0.0)
            
            # Aligned = all same direction (all positive or all negative)
            signs = [np.sign(m15_trend), np.sign(h1_trend), np.sign(h4_trend), np.sign(d1_trend)]
            non_zero_signs = [s for s in signs if s != 0]
            
            if len(non_zero_signs) >= 2:
                if all(s > 0 for s in non_zero_signs) or all(s < 0 for s in non_zero_signs):
                    result["trend_aligned"] = True
                elif len(non_zero_signs) >= 2:
                    # Check if M15/H1 opposes higher TFs
                    ltf_signs = [np.sign(m15_trend), np.sign(h1_trend)]
                    ltf_signs = [s for s in ltf_signs if s != 0]
                    htf_signs = [np.sign(h4_trend), np.sign(d1_trend)]
                    htf_signs = [s for s in htf_signs if s != 0]
                    
                    if ltf_signs and htf_signs:
                        ltf_consensus = ltf_signs[0] if len(set(ltf_signs)) == 1 else 0
                        htf_consensus = htf_signs[0] if len(set(htf_signs)) == 1 else 0
                        if ltf_consensus != 0 and htf_consensus != 0 and ltf_consensus != htf_consensus:
                            result["trend_opposed"] = True
            
            # Check volatility consistency
            if len(tf_vols) >= 2:
                vol_values = list(tf_vols.values())
                vol_std = np.std(vol_values)
                vol_mean = np.mean(vol_values)
                
                # Consistent if coefficient of variation is low
                if vol_mean > 0:
                    cv = vol_std / vol_mean
                    result["vol_consistent"] = cv < 0.5
            
            self.log_debug(
                f"[THEME MTF] {inst_norm}: M15={m15_trend:.4f}, H1={h1_trend:.4f}, H4={h4_trend:.4f}, "
                f"D1={d1_trend:.4f}, aligned={result['trend_aligned']}, "
                f"opposed={result['trend_opposed']}"
            )
            
        except Exception as e:
            self.log_debug(f"[THEME MTF] Error analyzing {instrument}: {e}")
        
        return result

    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing method."""
        result = np.zeros_like(data)
        if period <= 0 or len(data) == 0:
            return result

        result[:period] = np.cumsum(data[:period])
        result[period - 1] = result[period - 1] / period

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
        Analyze correlation regime across assets/timeframes.

        Returns:
            regime: 'risk_on_correlation', 'risk_off_correlation', 'decorrelated'
            score: correlation clustering score
        """
        prices_by_tf: Dict[str, np.ndarray] = {}

        if isinstance(market_data, dict):
            for tf in ["M15", "H1", "H4", "D1"]:
                if tf in market_data and isinstance(market_data[tf], dict):
                    if "close" in market_data[tf]:
                        tf_close = np.array(market_data[tf]["close"], dtype=float)
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
                        corr = np.corrcoef(r1, r2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(float(corr))

            if correlations:
                avg_corr = float(np.mean(correlations))
                if avg_corr > self.corr_cluster_threshold:
                    return "risk_on_correlation", avg_corr
                if avg_corr < -self.corr_cluster_threshold:
                    return "risk_off_correlation", avg_corr
                return "decorrelated", avg_corr

        # Fallback: simple autocorrelation
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
        Calculate market breadth score.

        Uses price position within range and trend consistency.
        """
        if len(close) < self.breadth_period:
            return 0.5

        period_close = close[-self.breadth_period :]
        period_high = (
            high[-self.breadth_period :] if len(high) >= self.breadth_period else period_close
        )
        period_low = (
            low[-self.breadth_period :] if len(low) >= self.breadth_period else period_close
        )

        period_range = np.max(period_high) - np.min(period_low)
        if period_range > 0:
            position_score = (close[-1] - np.min(period_low)) / period_range
        else:
            position_score = 0.5

        returns = np.diff(period_close)
        up_days = np.sum(returns > 0)
        down_days = np.sum(returns < 0)
        total_days = up_days + down_days
        breadth_ratio = up_days / total_days if total_days > 0 else 0.5

        if len(close) >= 5:
            recent_high = np.max(close[-5:])
            recent_low = np.min(close[-5:])
        else:
            recent_high = close[-1]
            recent_low = close[-1]

        period_high_max = np.max(period_high)
        period_low_min = np.min(period_low)

        new_high_score = 1.0 if recent_high >= period_high_max else 0.5
        new_low_score = 0.0 if recent_low <= period_low_min else 0.5

        breadth_score = (
            position_score * 0.35
            + breadth_ratio * 0.35
            + new_high_score * 0.15
            + new_low_score * 0.15
        )

        return float(np.clip(breadth_score, 0, 1))

    def _calculate_momentum_score(self, close: np.ndarray) -> float:
        """Calculate momentum score using rate of change."""
        if len(close) < 20:
            return 0.5

        roc_5 = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0.0

        roc_10 = 0.0
        if len(close) > 10 and close[-11] > 0:
            roc_10 = (close[-1] - close[-11]) / close[-11]

        roc_20 = 0.0
        if len(close) > 20 and close[-21] > 0:
            roc_20 = (close[-1] - close[-21]) / close[-21]

        norm_roc_5 = np.clip(roc_5 * 10 + 0.5, 0, 1)
        norm_roc_10 = np.clip(roc_10 * 5 + 0.5, 0, 1)
        norm_roc_20 = np.clip(roc_20 * 2.5 + 0.5, 0, 1)

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

        Combines all regime scores into single 0-1 sentiment value.
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

        return float(np.clip(composite, 0, 1))

    def _determine_risk_regime(
        self,
        vol_regime: str,
        trend_regime: str,
        composite_score: float,
        asset_class: str = "forex",
    ) -> str:
        """
        Determine overall risk regime.

        For commodities (like Gold), risk-off can mean LONG (safe haven).
        For forex, risk-off typically means cautious/neutral.
        """
        if vol_regime == "extreme_vol":
            return "risk_off"

        if composite_score >= self.risk_on_threshold:
            if trend_regime in ["strong_uptrend", "weak_uptrend"]:
                return "risk_on"
            return "neutral"

        if composite_score <= self.risk_off_threshold:
            if trend_regime in ["strong_downtrend", "weak_downtrend"]:
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
        - Gold behaves as safe haven in risk-off.
        Returns: (action, confidence, thesis)
        """
        inst_norm = normalize_instrument(instrument) if instrument else ""
        is_safe_haven = inst_norm in ["XAUUSD", "GOLD", "XAU"]

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
            if is_safe_haven and risk_regime in ["risk_off", "cautious"]:
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

        # Risk-off / cautious
        if risk_regime in ["risk_off", "cautious"] and composite_score < 0.45:
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

        # Ranging: direction from composite
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

        # Default mixed-case
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
        if len(self.regime_history) > 0 and self.regime_history[-1] == action:
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
        """Generate neutral output with explanation, publish neutral keys."""
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

        try:
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
        except Exception:
            pass

        theme_analysis = {
            "volatility_regime": "unknown",
            "trend_regime": "unknown",
            "risk_regime": "unknown",
            "composite_score": 0.5,
            "action": "flat",
            "confidence": confidence,
            "per_instrument": {},
        }

        return {
            "ThemeExpert_voting_proposal": proposal,
            "ThemeExpert_confidence": confidence,
            "ThemeExpert_per_instrument_votes": {},
            "per_instrument_votes": PerInstrumentVote(member=name),
            "theme_voting_proposal": proposal,
            "theme_confidence": confidence,
            "theme_analysis": theme_analysis,
            "agreement_score": confidence,
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
