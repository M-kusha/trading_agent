"""
Advanced SeasonalityRiskExpert - Time-Based Pattern Voting Module.

This expert analyzes seasonal and temporal patterns including:
- Day-of-week patterns (Monday effect, Friday positioning)
- Hour-of-day patterns (session overlaps, volatility windows)
- Month-of-year patterns (January effect, summer doldrums)
- Session analysis (Asian/European/American sessions)
- Holiday calendar effects (reduced liquidity periods)
- Economic calendar awareness (high-impact event windows)
- Rollover and swap timing

Per-Instrument Voting:
- Different assets have different seasonal patterns
- Gold has different seasonality than EURUSD
- Each instrument gets its own vote based on asset-specific patterns

Actions: long, short, flat
"""

from __future__ import annotations

import calendar
from datetime import datetime, time as dt_time
from typing import Any, Dict, List, Tuple, Optional

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


@module(**module_args("SeasonalityRiskExpert"))
class SeasonalityRiskExpert(VotingExpertBase):
    """
    Advanced seasonality and time-pattern analysis expert.

    Combines multiple temporal factors to identify optimal/suboptimal
    trading windows and seasonal biases.

    Per-instrument:
    - Different assets have different seasonal patterns (e.g. Gold vs EURUSD).
    """

    # ═══════════════════════════ INIT ═══════════════════════════

    def _expert_specific_init(self) -> None:
        """Initialize the seasonality expert with configuration."""
        self.module_name = self.__class__.__name__

        # Instruments to analyze (from config or default)
        self.instruments = self.config.get("instruments", ["EURUSD", "XAUUSD"])

        # Asset class mapping for different seasonal patterns
        self.asset_classes = {
            "EURUSD": "forex",
            "XAUUSD": "commodity",
            "GBPUSD": "forex",
            "USDJPY": "forex",
        }

        # Session times (UTC)
        self.sessions = {
            "asian": {"start": dt_time(0, 0), "end": dt_time(9, 0)},
            "european": {"start": dt_time(7, 0), "end": dt_time(16, 0)},
            "american": {"start": dt_time(13, 0), "end": dt_time(22, 0)},
            "overlap_eu_us": {"start": dt_time(13, 0), "end": dt_time(16, 0)},
            "overlap_asia_eu": {"start": dt_time(7, 0), "end": dt_time(9, 0)},
        }

        # Day-of-week biases (0=Monday, 4=Friday) – stylized FX patterns
        self.dow_biases = {
            0: {
                "name": "Monday",
                "volatility": 0.8,
                "trend_continuation": 0.6,
                "reversal_risk": 0.4,
            },
            1: {
                "name": "Tuesday",
                "volatility": 1.0,
                "trend_continuation": 0.7,
                "reversal_risk": 0.3,
            },
            2: {
                "name": "Wednesday",
                "volatility": 1.1,
                "trend_continuation": 0.8,
                "reversal_risk": 0.3,
            },
            3: {
                "name": "Thursday",
                "volatility": 1.0,
                "trend_continuation": 0.65,
                "reversal_risk": 0.35,
            },
            4: {
                "name": "Friday",
                "volatility": 0.9,
                "trend_continuation": 0.5,
                "reversal_risk": 0.5,
            },
        }

        # Monthly patterns (rough stylized facts)
        self.monthly_patterns = {
            1: {"name": "January", "trend_strength": 1.2, "risk_on": True},
            2: {"name": "February", "trend_strength": 1.0, "risk_on": True},
            3: {"name": "March", "trend_strength": 1.1, "risk_on": True},
            4: {"name": "April", "trend_strength": 1.0, "risk_on": True},
            5: {"name": "May", "trend_strength": 0.8, "risk_on": False},
            6: {"name": "June", "trend_strength": 0.7, "risk_on": False},
            7: {"name": "July", "trend_strength": 0.6, "risk_on": False},
            8: {"name": "August", "trend_strength": 0.5, "risk_on": False},
            9: {"name": "September", "trend_strength": 1.0, "risk_on": False},
            10: {"name": "October", "trend_strength": 1.1, "risk_on": True},
            11: {"name": "November", "trend_strength": 1.0, "risk_on": True},
            12: {"name": "December", "trend_strength": 0.6, "risk_on": False},
        }

        # High-volatility hours (UTC) - typical macro releases
        self.high_impact_hours = [8, 12, 13, 14, 18]

        # Rollover window (usually around 21:00-22:00 UTC)
        self.rollover_start = dt_time(21, 0)
        self.rollover_end = dt_time(22, 0)

        # Weekend gap risk window (Friday after 20:00 UTC)
        self.weekend_risk_start = dt_time(20, 0)

        # Historical pattern tracking (reserved for future calibration)
        self.pattern_history: List[Dict[str, Any]] = []
        self.pattern_accuracy: Dict[str, float] = {}

        # Confidence parameters
        self.base_confidence = float(self.config.get("base_confidence", 0.4))
        self.max_confidence = float(self.config.get("max_confidence", 0.85))
        self.min_confidence = float(self.config.get("min_confidence", 0.15))

        # Session quality weights
        self.session_weights = {
            "overlap_eu_us": 1.3,
            "european": 1.1,
            "american": 1.0,
            "overlap_asia_eu": 0.9,
            "asian": 0.8,
        }

        # Regime persistence for GLOBAL seasonal stance
        self.regime_history: List[str] = []
        self.regime_persistence_count = 0
        self.min_regime_persistence = int(
            self.config.get("min_regime_persistence", 3)
        )

        # Local trading-window configuration (user's timezone)
        # Default: Europe/Berlin, trade 07:00–22:00 local, avoid last 30 minutes before 23:00.
        self.trading_timezone: str = self.config.get(
            "trading_timezone", "Europe/Berlin"
        )
        self.local_trade_start_hour: int = int(
            self.config.get("local_trade_start_hour", 7)
        )
        self.local_trade_end_hour: int = int(
            self.config.get("local_trade_end_hour", 22)
        )
        self.local_hard_close_hour: int = int(
            self.config.get("local_hard_close_hour", 23)
        )
        # In the last N minutes before local_hard_close, we push for exits.
        self.no_trade_last_minutes: int = int(
            self.config.get("no_trade_last_minutes", 30)
        )
        
        # TEST MODE: Allow trades during off-hours (for testing purposes)
        # Set to True to bypass trading window restrictions
        self.allow_off_hours_trading: bool = bool(
            self.config.get("allow_off_hours_trading", False)
        )

        self.log_info(
            f"[SEASONALITY] SeasonalityRiskExpert initialized | "
            f"instruments={self.instruments} | trading_tz={self.trading_timezone} "
            f"| trade_window={self.local_trade_start_hour}:00-{self.local_trade_end_hour}:00 "
            f"| hard_close={self.local_hard_close_hour}:00"
            f"| allow_off_hours={self.allow_off_hours_trading}"
        )

        # Publish baseline keys
        self._publish_baseline_keys()
        self._publish_seasonality_baseline()

    def _publish_seasonality_baseline(self) -> None:
        """Publish baseline seasonality keys to avoid stale consumers."""
        from datetime import timezone as dt_timezone
        thesis = "Seasonality baseline"
        confidence = 0.1
        # Generate trading window - default to blocking new trades in baseline state
        trading_window = self._analyze_trading_window(datetime.now(dt_timezone.utc))
        proposal = {
            "action": "flat",
            "signal_strength": confidence,
            "reason": thesis,
            "proposals": {},
            "trading_window": trading_window,  # Include for arbiter seasonality gate
        }
        try:
            self.smart_bus.set(
                "seasonality_voting_proposal",
                proposal,
                module=self.module_name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "seasonality_confidence",
                confidence,
                module=self.module_name,
                thesis="Seasonality baseline confidence",
            )
            self.smart_bus.set(
                "seasonal_voting_proposal",
                proposal,
                module=self.module_name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "seasonal_confidence",
                confidence,
                module=self.module_name,
                thesis="Seasonal baseline confidence",
            )
            self.smart_bus.set(
                "seasonality_risk_analysis",
                {
                    "session": "unknown",
                    "dow_bias": "unknown",
                    "monthly_pattern": "unknown",
                    "composite_score": 0.5,
                    "rollover_risk": False,
                    "weekend_risk": False,
                    "action": "flat",
                    "confidence": confidence,
                    "per_instrument": {},
                },
                module=self.module_name,
                thesis="Seasonality baseline analysis",
            )
        except Exception:
            pass

    # ═══════════════════════════ VOTINGEXPERTBASE LEGACY HOOKS ═══════════════════════════

    async def _generate_expert_specific_proposal(
        self, market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Legacy hook for VotingExpertBase.

        SeasonalityRiskExpert is primarily per-instrument via process().
        """
        return {
            "action": "flat",
            "signal_strength": 0.0,
            "reason": "SeasonalityRiskExpert uses per-instrument process() path",
        }

    async def _calculate_expert_specific_confidence(
        self, proposal: Dict[str, Any], market_data: Dict[str, Any]
    ) -> float:
        """Legacy confidence hook; neutral default."""
        return 0.3

    # ═══════════════════════════ MAIN PER-INSTRUMENT PROCESS ═══════════════════════════

    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Process temporal data to determine seasonal biases and timing PER INSTRUMENT.

        - Time-based regime (sessions, DOW, month, high-impact windows) is global.
        - Each instrument applies those temporal regimes plus its own historical bias.
        - Local trading window (user's timezone) restricts late-day / overnight exposure.
        """
        name = self.__class__.__name__
        start = self._now_ms()

        try:
            # Circuit breaker
            if self._check_circuit_breaker():
                return self._degraded_output("circuit_breaker_open")

            # Global time context (UTC) – local trading window derived separately
            # IMPORTANT: Use timezone-aware datetime to ensure correct UTC→local conversion
            # datetime.utcnow() returns naive datetime which Python treats as LOCAL time
            # when calling .astimezone(), causing 1-hour DST errors
            from datetime import timezone as dt_timezone
            current_time_utc = datetime.now(dt_timezone.utc)
            trading_window = self._analyze_trading_window(current_time_utc)

            market_data = self.smart_bus.get("market_data", name, default={})
            features = self.smart_bus.get("features", name, default={})

            # Global temporal components (same for all instruments)
            session_analysis = self._analyze_session(current_time_utc)
            dow_analysis = self._analyze_day_of_week(current_time_utc)
            monthly_analysis = self._analyze_monthly_pattern(current_time_utc)
            hour_analysis = self._analyze_hour_patterns(current_time_utc)

            rollover_risk = self._check_rollover_risk(current_time_utc)
            weekend_risk = self._check_weekend_risk(current_time_utc)
            high_impact_window = self._check_high_impact_window(current_time_utc)

            # Per-instrument container
            per_instrument_vote = PerInstrumentVote(member=name)
            per_instrument_analysis: Dict[str, Dict[str, Any]] = {}

            # ── Per-instrument seasonal analysis ─────────────────────────────
            for inst in self.instruments:
                inst_norm = normalize_instrument(inst)
                asset_class = self.asset_classes.get(inst_norm, "forex")

                inst_market = self._extract_instrument_data(market_data, inst)
                inst_features = self._extract_instrument_data(features, inst)

                historical_score = self._calculate_historical_pattern_score(
                    current_time_utc, inst_market, inst_features, inst_norm
                )

                composite_score = self._calculate_composite_score(
                    session_analysis,
                    dow_analysis,
                    monthly_analysis,
                    hour_analysis,
                    historical_score,
                )

                action, confidence, thesis = self._select_seasonal_action(
                    composite_score,
                    session_analysis,
                    dow_analysis,
                    monthly_analysis,
                    rollover_risk,
                    weekend_risk,
                    high_impact_window,
                    instrument=inst_norm,
                    asset_class=asset_class,
                    trading_window=trading_window,
                )

                # Calibrate magnitude: seasonality is an overlay, keep it modest
                min_strength = MIN_SIGNAL_STRENGTH_F()
                if action == "flat":
                    magnitude = 0.0
                else:
                    magnitude = float(
                        max(min_strength * 0.4, min(1.0, confidence * 0.6))
                    )

                # ═══════════════════════════════════════════════════════════════
                # POSITION FOCUS MODE (PER-INSTRUMENT)
                # If we have a position in THIS instrument, reframe signal for position management.
                # End-of-day window will prefer EXIT to avoid overnight fees.
                # ═══════════════════════════════════════════════════════════════
                position_focus = self._get_position_focus_context()
                supports_position = True
                position_eval = "no_position"
                original_action = action

                if position_focus and self._has_position_for_instrument(inst_norm):
                    position_side = self._get_position_side_for_instrument(inst_norm)
                    inst_position = self._get_position_for_instrument(inst_norm)
                    position_pnl = float(
                        inst_position.get(
                            "unrealized_pnl", inst_position.get("pnl", 0.0)
                        )
                    ) if inst_position else 0.0

                    # Determine if signal supports or threatens position
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

                    # Remap action for position management (default behaviour)
                    if supports_position:
                        action = "hold"  # Strong support = confident hold
                    else:
                        if confidence > 0.7:
                            action = "exit"  # Strong opposing signal
                        elif confidence > 0.5:
                            action = "tighten"  # Moderate opposing signal
                        else:
                            action = "hold"  # Weak opposing signal

                    # End-of-day safeguard: prefer exit in final window before hard-close
                    if trading_window.get("final_exit_window", False):
                        action = "exit"
                        position_eval = "eod_exit_window"
                        confidence = max(confidence, 0.65)
                        thesis = (
                            f"{inst_norm}: POSITION_FOCUS(EOD_EXIT_WINDOW) -> exit "
                            f"(local_close_in={trading_window.get('minutes_to_close', 0)}min, "
                            f"pnl={position_pnl:.2f})"
                        )

                    # Adjust confidence based on PnL (risk-aware tweaks)
                    if position_pnl > 0 and not supports_position:
                        confidence *= 0.8  # Reduce exit confidence when in profit
                    elif position_pnl < 0 and not supports_position:
                        confidence = min(1.0, confidence * 1.2)  # Increase exit confidence when losing

                    self.log_debug(
                        f"[SEASONALITY] {inst_norm} POSITION_FOCUS: original_action={original_action} -> {action}, "
                        f"supports={supports_position}, eval={position_eval}, "
                        f"pnl={position_pnl:.2f}, eod_window={trading_window.get('final_exit_window', False)}"
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
                    "session": session_analysis["current_session"],
                    "dow_bias": dow_analysis["bias"],
                    "monthly_pattern": monthly_analysis["pattern"],
                    "composite_score": composite_score,
                    "historical_score": historical_score,
                    "rollover_risk": rollover_risk,
                    "weekend_risk": weekend_risk,
                    "high_impact_window": high_impact_window,
                    "action": action,
                    "confidence": confidence,
                    "position_focus_mode": self._has_position_for_instrument(inst_norm),
                    "supports_position": supports_position,
                    "position_evaluation": position_eval,
                    "original_action": original_action,
                    "trading_window": {
                        "timezone": trading_window.get("timezone"),
                        "local_time": trading_window.get("local_time"),
                        "local_hour": trading_window.get("local_hour"),
                        "local_minute": trading_window.get("local_minute"),
                        "in_primary_window": trading_window.get("in_primary_window"),
                        "no_new_trades": trading_window.get("no_new_trades"),
                        "final_exit_window": trading_window.get("final_exit_window"),
                    },
                }

                self.log_debug(
                    f"[SEASONALITY] {inst_norm}: session={session_analysis['current_session']}, "
                    f"month={monthly_analysis['month_name']}, comp={composite_score:.2f}, "
                    f"action={action}, conf={confidence:.2f}, "
                    f"local_time={trading_window.get('local_time')}, "
                    f"no_new_trades={trading_window.get('no_new_trades')}"
                )

            # ── Global summary / backward compatibility ──────────────────────
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

            # Regime persistence on global seasonal stance
            global_action, global_confidence = self._apply_persistence_filter(
                global_action, global_confidence
            )

            # Clip global confidence with shared thresholds
            conf_floor = CONFIDENCE_THRESHOLD_F()
            high_conf = HIGH_CONFIDENCE_THRESHOLD_F()
            global_confidence = float(
                max(conf_floor * 0.5, min(high_conf, global_confidence))
            )

            # Per-instrument proposals dict for CommitteeCoordinator
            proposals_dict: Dict[str, Dict[str, Any]] = {}
            for inst, prop in per_instrument_vote.proposals.items():
                proposals_dict[inst] = {
                    "action": prop.action,
                    "confidence": prop.confidence,
                    "magnitude": prop.magnitude,
                    "rationale": prop.rationale,
                }

            proposal = {
                "action": global_action,
                "signal_strength": global_confidence,
                "reason": global_thesis,
                "proposals": proposals_dict,
                "trading_window": trading_window,  # Include for arbiter seasonality gate
            }

            # Derive global composite score (first instrument as representative)
            if per_instrument_analysis:
                first_inst_analysis = list(per_instrument_analysis.values())[0]
                global_composite = float(
                    first_inst_analysis.get("composite_score", 0.5)
                )
            else:
                global_composite = 0.5

            # Publish to SmartInfoBus
            try:
                self.smart_bus.set(
                    "SeasonalityRiskExpert_voting_proposal",
                    proposal,
                    module=name,
                    thesis=global_thesis,
                )
                self.smart_bus.set(
                    "SeasonalityRiskExpert_confidence",
                    global_confidence,
                    module=name,
                    thesis=f"Confidence: {global_confidence:.1%}",
                )
                self.smart_bus.set(
                    "seasonality_voting_proposal",
                    proposal,
                    module=name,
                    thesis=global_thesis,
                )
                self.smart_bus.set(
                    "seasonality_confidence",
                    global_confidence,
                    module=name,
                    thesis=f"Seasonality confidence: {global_confidence:.1%}",
                )
                self.smart_bus.set(
                    "seasonal_voting_proposal",
                    proposal,
                    module=name,
                    thesis=global_thesis,
                )
                self.smart_bus.set(
                    "seasonal_confidence",
                    global_confidence,
                    module=name,
                    thesis=f"Seasonal confidence: {global_confidence:.1%}",
                )

                per_inst_votes_dict = {
                    inst: p.to_dict()
                    for inst, p in per_instrument_vote.proposals.items()
                }
                self.smart_bus.set(
                    "SeasonalityRiskExpert_per_instrument_votes",
                    per_inst_votes_dict,
                    module=name,
                    thesis=(
                        "Per-instrument seasonality votes: "
                        f"{list(per_inst_votes_dict.keys())}"
                    ),
                )
            except Exception as e:
                self.log_warning(f"[SEASONALITY] Failed to publish to bus: {e}")

            seasonality_risk_analysis = {
                "session": session_analysis["current_session"],
                "dow_bias": dow_analysis["bias"],
                "monthly_pattern": monthly_analysis["pattern"],
                "composite_score": global_composite,
                "rollover_risk": rollover_risk,
                "weekend_risk": weekend_risk,
                "action": global_action,
                "confidence": global_confidence,
                "per_instrument": per_instrument_analysis,
                "trading_window": trading_window,
            }

            output = {
                "SeasonalityRiskExpert_voting_proposal": proposal,
                "SeasonalityRiskExpert_confidence": global_confidence,
                "SeasonalityRiskExpert_per_instrument_votes": {
                    inst: p.to_dict()
                    for inst, p in per_instrument_vote.proposals.items()
                },
                "per_instrument_votes": per_instrument_vote,
                # Aliases for contracts
                "seasonality_voting_proposal": proposal,
                "seasonality_confidence": global_confidence,
                "seasonal_voting_proposal": proposal,
                "seasonal_confidence": global_confidence,
                "seasonality_risk_analysis": seasonality_risk_analysis,
                "seasonality_analysis": {
                    "session": session_analysis["current_session"],
                    "dow_bias": dow_analysis["bias"],
                    "monthly_pattern": monthly_analysis["pattern"],
                    "composite_score": global_composite,
                    "per_instrument": per_instrument_analysis,
                    "trading_window": trading_window,
                },
                "seasonality_expert_analysis": {
                    "session": session_analysis["current_session"],
                    "dow_bias": dow_analysis["bias"],
                    "monthly_pattern": monthly_analysis["pattern"],
                    "composite_score": global_composite,
                    "rollover_risk": rollover_risk,
                    "weekend_risk": weekend_risk,
                    "per_instrument": per_instrument_analysis,
                    "trading_window": trading_window,
                },
                "seasonality_expert_thesis": global_thesis,
                "seasonal_session": session_analysis["current_session"],
                "seasonal_dow_bias": dow_analysis["bias"],
                "seasonal_monthly_pattern": monthly_analysis["pattern"],
                "seasonal_composite_score": global_composite,
                "seasonal_rollover_risk": rollover_risk,
                "seasonal_weekend_risk": weekend_risk,
                "_thesis": global_thesis,
            }

            # Perf metrics / success
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
            # Degraded-mode path
            self._record_error(e)
            if self.error_pinpointer is not None:
                err_ctx = self.error_pinpointer.analyze_error(e, f"{name}_process")
                msg = str(err_ctx)
            else:
                msg = str(e)

            self.log_error(f"[SEASONALITY] Process error: {msg}")

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
        """Monotonic ms helper for perf tracking."""
        import time as _time

        return _time.time() * 1000.0

    def _elapsed_ms(self, start_ms: float) -> float:
        import time as _time

        return _time.time() * 1000.0 - start_ms

    def _extract_instrument_data(self, data: Dict, instrument: str) -> Dict:
        """Extract data for a specific instrument from nested market data."""
        if not isinstance(data, dict):
            return {}

        inst_norm = normalize_instrument(instrument)

        for key in [instrument, inst_norm, instrument.upper(), instrument.lower()]:
            if key in data:
                return data[key] if isinstance(data[key], dict) else data

        # Fallback: legacy format (single instrument / flat dict)
        return data

    def _analyze_session(self, current_time: datetime) -> Dict[str, Any]:
        """
        Analyze current trading session and quality (UTC-based).

        Returns:
            - current_session
            - active_sessions
            - session_quality
            - liquidity_score
            - session_phase
        """
        current_hour = current_time.time()
        active_sessions: List[str] = []

        for session_name, times in self.sessions.items():
            if self._time_in_range(current_hour, times["start"], times["end"]):
                active_sessions.append(session_name)

        if "overlap_eu_us" in active_sessions:
            primary_session = "overlap_eu_us"
            session_quality = 1.0
        elif "overlap_asia_eu" in active_sessions:
            primary_session = "overlap_asia_eu"
            session_quality = 0.85
        elif "european" in active_sessions:
            primary_session = "european"
            session_quality = 0.9
        elif "american" in active_sessions:
            primary_session = "american"
            session_quality = 0.85
        elif "asian" in active_sessions:
            primary_session = "asian"
            session_quality = 0.7
        else:
            primary_session = "off_hours"
            session_quality = 0.4

        liquidity_score = self.session_weights.get(primary_session, 0.5)
        session_phase = self._get_session_phase(current_time, primary_session)

        return {
            "current_session": primary_session,
            "active_sessions": active_sessions,
            "session_quality": session_quality,
            "liquidity_score": liquidity_score,
            "session_phase": session_phase,
        }

    def _time_in_range(
        self, current: dt_time, start: dt_time, end: dt_time
    ) -> bool:
        """Check if current time is in range (handles overnight sessions)."""
        if start <= end:
            return start <= current <= end
        return current >= start or current <= end

    def _get_session_phase(self, current_time: datetime, session: str) -> str:
        """Determine if we're in early, mid, or late session phase."""
        if session not in self.sessions:
            return "unknown"

        session_times = self.sessions[session]
        start = session_times["start"]
        end = session_times["end"]

        current = current_time.time()

        start_mins = start.hour * 60 + start.minute
        end_mins = end.hour * 60 + end.minute
        current_mins = current.hour * 60 + current.minute

        # Handle overnight sessions
        if end_mins < start_mins:
            end_mins += 24 * 60
            if current_mins < start_mins:
                current_mins += 24 * 60

        duration = end_mins - start_mins
        if duration <= 0:
            return "unknown"

        elapsed = current_mins - start_mins
        progress = elapsed / duration

        if progress < 0.33:
            return "early"
        if progress < 0.67:
            return "mid"
        return "late"

    def _analyze_day_of_week(self, current_time: datetime) -> Dict[str, Any]:
        """Analyze day-of-week patterns and biases."""
        dow = current_time.weekday()
        dow_info = self.dow_biases.get(dow, self.dow_biases[1])

        if dow == 0:
            bias = "cautious"
            continuation_prob = 0.6
        elif dow == 4:
            bias = "closing_bias"
            continuation_prob = 0.5
        elif dow in (1, 2):
            bias = "trending"
            continuation_prob = 0.75
        else:
            bias = "neutral"
            continuation_prob = 0.65

        return {
            "day_name": dow_info["name"],
            "day_number": dow,
            "bias": bias,
            "volatility_factor": dow_info["volatility"],
            "trend_continuation": dow_info["trend_continuation"],
            "reversal_risk": dow_info["reversal_risk"],
            "continuation_probability": continuation_prob,
        }

    def _analyze_monthly_pattern(self, current_time: datetime) -> Dict[str, Any]:
        """Analyze monthly and seasonal patterns."""
        month = current_time.month
        day = current_time.day
        month_info = self.monthly_patterns.get(month, self.monthly_patterns[6])

        days_in_month = calendar.monthrange(current_time.year, month)[1]
        is_month_end = day >= days_in_month - 2
        is_month_start = day <= 3

        quarter = (month - 1) // 3 + 1
        is_quarter_end = month in (3, 6, 9, 12) and is_month_end

        if month in (1, 2, 3, 4, 10, 11):
            seasonal_bias = "bullish"
        elif month in (5, 6, 7, 8, 9):
            seasonal_bias = "cautious"
        else:
            seasonal_bias = "year_end_positioning"

        return {
            "month": month,
            "month_name": month_info["name"],
            "pattern": seasonal_bias,
            "trend_strength": month_info["trend_strength"],
            "risk_on": month_info["risk_on"],
            "is_month_end": is_month_end,
            "is_month_start": is_month_start,
            "is_quarter_end": is_quarter_end,
            "quarter": quarter,
        }

    def _analyze_hour_patterns(self, current_time: datetime) -> Dict[str, Any]:
        """Analyze hourly patterns and volatility expectations."""
        hour = current_time.hour
        minute = current_time.minute

        is_pre_news = hour in self.high_impact_hours and minute >= 50
        is_news_hour = hour in self.high_impact_hours and minute < 30

        # Volatility bands (UTC)
        if 13 <= hour <= 16:
            expected_volatility = 1.3
        elif 7 <= hour <= 9:
            expected_volatility = 1.1
        elif 0 <= hour <= 6:
            expected_volatility = 0.7
        elif 22 <= hour <= 23:
            expected_volatility = 0.6
        else:
            expected_volatility = 0.9

        # Trading quality bands
        if 8 <= hour <= 16:
            trading_quality = 0.9
        elif 17 <= hour <= 20:
            trading_quality = 0.85
        else:
            trading_quality = 0.6

        return {
            "hour": hour,
            "is_pre_news": is_pre_news,
            "is_news_hour": is_news_hour,
            "expected_volatility": expected_volatility,
            "trading_quality": trading_quality,
            "high_impact_window": hour in self.high_impact_hours,
        }

    def _check_rollover_risk(self, current_time: datetime) -> bool:
        """Check if we're in the rollover window (UTC)."""
        current = current_time.time()
        return self._time_in_range(current, self.rollover_start, self.rollover_end)

    def _check_weekend_risk(self, current_time: datetime) -> bool:
        """Check if we're in weekend gap risk window (Friday late UTC)."""
        is_friday = current_time.weekday() == 4
        current = current_time.time()
        return is_friday and current >= self.weekend_risk_start

    def _check_high_impact_window(self, current_time: datetime) -> bool:
        """Check if we're in a high-impact event window."""
        hour = current_time.hour
        minute = current_time.minute

        # On the event hour: 00–30 and 50–59 are dangerous.
        if hour in self.high_impact_hours and (minute <= 30 or minute >= 50):
            return True

        # 10 minutes before the event hour (previous hour, :50–:59)
        if (hour + 1) % 24 in self.high_impact_hours and minute >= 50:
            return True

        return False

    def _analyze_trading_window(self, current_time_utc: datetime) -> Dict[str, Any]:
        """
        Analyze local trading window in the configured timezone.

        Goals:
        - Encourage trading only inside primary local hours (e.g. 07:00–22:00).
        - Avoid opening new trades after cutoff.
        - In last N minutes before local_hard_close, push strongly for exits.
        """
        try:
            from zoneinfo import ZoneInfo  # Python 3.9+
            tz = ZoneInfo(self.trading_timezone)
            local_dt = current_time_utc.astimezone(tz)
            tz_name = self.trading_timezone
        except Exception:
            # Fallback to naive UTC if zoneinfo is unavailable
            local_dt = current_time_utc
            tz_name = "UTC"

        lh = local_dt.hour
        lm = local_dt.minute
        local_minutes = lh * 60 + lm

        start_minutes = self.local_trade_start_hour * 60
        end_minutes = self.local_trade_end_hour * 60
        hard_close_minutes = self.local_hard_close_hour * 60

        in_primary_window = start_minutes <= local_minutes < end_minutes
        after_cutoff = local_minutes >= end_minutes

        minutes_to_close = hard_close_minutes - local_minutes
        if minutes_to_close < 0:
            # Past hard close; treat as "already closed" for this day
            minutes_to_close = 0

        final_exit_window = 0 <= minutes_to_close <= self.no_trade_last_minutes
        
        # TEST MODE: Override no_new_trades if allow_off_hours_trading is enabled
        if self.allow_off_hours_trading:
            no_new_trades = False
        else:
            no_new_trades = after_cutoff or final_exit_window

        return {
            "timezone": tz_name,
            "local_time": local_dt.isoformat(),
            "local_hour": lh,
            "local_minute": lm,
            "in_primary_window": in_primary_window,
            "after_cutoff": after_cutoff,
            "minutes_to_close": minutes_to_close,
            "no_new_trades": no_new_trades,
            "final_exit_window": final_exit_window,
            "allow_off_hours_override": self.allow_off_hours_trading,
        }

    def _calculate_historical_pattern_score(
        self,
        current_time: datetime,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
        instrument: str = "",
    ) -> float:
        """Calculate a simple score based on historical performance at similar times."""
        close_prices = self._extract_prices(
            market_data, features, "close", instrument
        )

        if len(close_prices) < 50:
            return 0.5

        # ═══════════════════════════════════════════════════════════════
        # REAL-TIME RESPONSIVENESS: Append forming bar's close price
        # ═══════════════════════════════════════════════════════════════
        try:
            inst_norm = normalize_instrument(instrument) if instrument else ""
            historical = self.smart_bus.get(
                "historical_prices", self.module_name, default=None
            )
            if isinstance(historical, dict):
                matched_sym = None
                for sym in historical.keys():
                    if normalize_instrument(sym) == inst_norm:
                        matched_sym = sym
                        break
                if matched_sym and isinstance(historical.get(matched_sym), dict):
                    m15_rec = historical[matched_sym].get("M15", {})
                    if isinstance(m15_rec, dict):
                        cur_bar = m15_rec.get("current_bar", {})
                        if isinstance(cur_bar, dict):
                            forming_close = cur_bar.get("close")
                            if forming_close is not None and len(close_prices) > 0:
                                forming_close = float(forming_close)
                                if abs(forming_close - float(close_prices[-1])) > 0.0001:
                                    close_prices = np.append(
                                        close_prices[:-1], forming_close
                                    )
        except Exception:
            # Silently continue with original prices
            pass

        recent_returns = np.diff(np.log(close_prices[-20:]))

        if len(recent_returns) > 0:
            positive_ratio = float(
                np.sum(recent_returns > 0) / len(recent_returns)
            )
        else:
            positive_ratio = 0.5

        historical_score = 0.5 + (positive_ratio - 0.5) * 0.6
        return float(np.clip(historical_score, 0, 1))

    def _extract_prices(
        self,
        market_data: Dict[str, Any],
        features: Dict[str, Any],
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
        if isinstance(market_data, dict):
            if price_type in market_data:
                data = market_data[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)

            for tf in ("M15", "H1", "H4", "D1"):
                if tf in market_data and isinstance(market_data[tf], dict):
                    if price_type in market_data[tf]:
                        data = market_data[tf][price_type]
                        if isinstance(data, (list, np.ndarray)):
                            return np.array(data, dtype=float)

        if isinstance(features, dict):
            if price_type in features:
                data = features[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)

        try:
            historical = self.smart_bus.get(
                "historical_prices", self.module_name, default=None
            )
        except Exception:
            historical = None

        if isinstance(historical, dict) and historical:
            inst_norm = normalize_instrument(instrument) if instrument else ""
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
                tf_rec = None
                # M15 is primary, H1/H4/D1 are context (ordered by granularity)
                for tf in ("M15", "H1", "H4", "D1"):
                    candidate = sym_block.get(tf)
                    if isinstance(candidate, dict):
                        tf_rec = candidate
                        break
                if tf_rec is None and sym_block:
                    tf_rec = sym_block.get(next(iter(sym_block.keys())))
                if isinstance(tf_rec, dict):
                    seq = tf_rec.get(price_type)
                    if isinstance(seq, (list, np.ndarray)):
                        return np.array(seq, dtype=float)

        return np.array([])

    def _calculate_composite_score(
        self,
        session_analysis: Dict[str, Any],
        dow_analysis: Dict[str, Any],
        monthly_analysis: Dict[str, Any],
        hour_analysis: Dict[str, Any],
        historical_score: float,
    ) -> float:
        """
        Calculate composite seasonal/temporal score.

        Higher scores = more favorable trading conditions.
        """
        session_score = float(session_analysis["session_quality"])
        dow_score = float(dow_analysis["trend_continuation"])

        monthly_score = monthly_analysis["trend_strength"] / 1.2
        monthly_score = float(np.clip(monthly_score, 0, 1))

        hour_score = float(hour_analysis["trading_quality"])

        composite = (
            session_score * 0.25
            + dow_score * 0.20
            + monthly_score * 0.20
            + hour_score * 0.15
            + historical_score * 0.20
        )

        return float(np.clip(composite, 0, 1))

    def _select_seasonal_action(
        self,
        composite_score: float,
        session_analysis: Dict[str, Any],
        dow_analysis: Dict[str, Any],
        monthly_analysis: Dict[str, Any],
        rollover_risk: bool,
        weekend_risk: bool,
        high_impact_window: bool,
        instrument: str = "",
        asset_class: str = "forex",
        trading_window: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, float, str]:
        """
        Select trading action based on seasonal analysis.

        Instrument-aware:
        - Gold has different seasonal patterns and safe-haven behaviour.
        - Local trading-window overlay can veto late-day entries.
        """
        inst_norm = normalize_instrument(instrument) if instrument else ""
        is_gold = inst_norm in ("XAUUSD", "GOLD", "XAU")
        tw = trading_window or {}

        # High-impact caution: stay flat; Gold may get safe-haven bid
        if high_impact_window:
            if is_gold:
                return (
                    "long",
                    0.5,
                    f"{inst_norm}: Safe haven bid during high-impact window",
                )
            return (
                "flat",
                0.7,
                f"{inst_norm}: High-impact event window, recommending caution",
            )

        # Weekend gap risk
        if weekend_risk:
            return (
                "flat",
                0.75,
                f"{inst_norm}: Weekend gap risk - Friday late session",
            )

        # Rollover / swap widening
        if rollover_risk:
            return (
                "flat",
                0.6,
                f"{inst_norm}: Rollover window - wider spreads and lower liquidity",
            )

        # Local trading window guard (user-local time).
        # - After local_trade_end_hour or in final_exit_window,
        #   we do NOT want new entries: flat with strong conviction.
        if tw.get("no_new_trades", False):
            lh = tw.get("local_hour")
            lm = tw.get("local_minute")
            tz_name = tw.get("timezone", "local")
            if isinstance(lh, int) and isinstance(lm, int):
                time_str = f"{lh:02d}:{lm:02d} {tz_name}"
            else:
                time_str = f"local time ({tz_name})"
            return (
                "flat",
                0.7,
                f"{inst_norm}: Outside primary trading window at {time_str} - no new entries",
            )

        # Poor session quality: stay flat (Gold can still trade in Asia)
        if session_analysis["session_quality"] < 0.5:
            if not (is_gold and session_analysis["current_session"] == "asian"):
                return (
                    "flat",
                    0.55,
                    f"{inst_norm}: Low session quality "
                    f"({session_analysis['current_session']})",
                )

        # Gold seasonal patterns (stylized: strong in Jan, Aug, Sep, Dec)
        if is_gold:
            month = monthly_analysis.get("month", 0)
            month_name = monthly_analysis.get("month_name", "Unknown")
            if month in (1, 8, 9, 12):
                confidence = self.base_confidence + 0.2
                return (
                    "long",
                    float(np.clip(confidence, 0.5, 0.75)),
                    f"{inst_norm}: Favorable gold seasonality ({month_name})",
                )
            if month in (3, 4, 5):
                confidence = self.base_confidence
                return (
                    "short",
                    float(np.clip(confidence, 0.4, 0.65)),
                    f"{inst_norm}: Weak gold seasonality ({month_name})",
                )

        # Strong seasonal long bias (FX risk-on)
        if monthly_analysis["risk_on"] and composite_score > 0.6:
            confidence = self.base_confidence + (composite_score - 0.5) * 0.6
            return (
                "long",
                float(np.clip(confidence, 0.5, 0.8)),
                f"{inst_norm}: Favorable seasonal "
                f"({monthly_analysis['month_name']}, risk-on)",
            )

        # Cautious seasonal short bias
        if not monthly_analysis["risk_on"] and composite_score < 0.45:
            confidence = self.base_confidence + (0.5 - composite_score) * 0.6
            if is_gold:
                return (
                    "long",
                    float(np.clip(confidence, 0.5, 0.75)),
                    f"{inst_norm}: Safe haven in risk-off season",
                )
            return (
                "short",
                float(np.clip(confidence, 0.5, 0.75)),
                f"{inst_norm}: Unfavorable seasonal "
                f"({monthly_analysis['month_name']}, risk-off)",
            )

        # Optimal session: use DOW bias if strong continuation
        if session_analysis["current_session"] in ("overlap_eu_us", "european"):
            if dow_analysis["trend_continuation"] > 0.55:
                day_name = dow_analysis["day_name"]
                bias = dow_analysis.get("bias", "neutral")
                if bias == "trending":
                    return (
                        "long",
                        0.55,
                        f"{inst_norm}: Optimal session + trending {day_name}",
                    )

        # Quarter-end / month-end → flat
        if monthly_analysis["is_quarter_end"]:
            return (
                "flat",
                0.4,
                f"{inst_norm}: Quarter-end rebalancing",
            )

        if monthly_analysis["is_month_end"]:
            return (
                "flat",
                0.35,
                f"{inst_norm}: Month-end positioning",
            )

        # Default: use composite score for weak directional bias
        if composite_score > 0.52:
            return (
                "long",
                0.45,
                f"{inst_norm}: Slight bullish seasonal "
                f"(composite={composite_score:.2f})",
            )
        if composite_score < 0.48:
            if is_gold:
                return (
                    "flat",
                    0.35,
                    f"{inst_norm}: Slight bearish but gold - neutral overlay",
                )
            return (
                "short",
                0.45,
                f"{inst_norm}: Slight bearish seasonal "
                f"(composite={composite_score:.2f})",
            )

        return (
            "flat",
            0.3,
            f"{inst_norm}: Neutral seasonal - "
            f"session: {session_analysis['current_session']}",
        )

    def _apply_persistence_filter(
        self,
        action: str,
        confidence: float,
    ) -> Tuple[str, float]:
        """
        Apply regime persistence filter to GLOBAL seasonal action.

        - Boosts confidence for persistent regimes.
        - Dampens for brand-new flips.
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
        """Generate neutral output with explanation and publish neutral keys."""
        from datetime import timezone as dt_timezone
        name = self.__class__.__name__
        thesis = f"Seasonal flat: {reason}"

        conf_floor = CONFIDENCE_THRESHOLD_F()
        confidence = max(0.1, conf_floor * 0.5)

        # Generate trading window for the gate check
        trading_window = self._analyze_trading_window(datetime.now(dt_timezone.utc))

        proposal = {
            "action": "flat",
            "signal_strength": confidence,
            "reason": thesis,
            "proposals": {},
            "trading_window": trading_window,  # Include for arbiter seasonality gate
        }

        try:
            self.smart_bus.set(
                "SeasonalityRiskExpert_voting_proposal",
                proposal,
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "SeasonalityRiskExpert_confidence",
                confidence,
                module=name,
                thesis=f"Confidence: {confidence:.1%}",
            )
            self.smart_bus.set(
                "seasonality_voting_proposal",
                proposal,
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "seasonality_confidence",
                confidence,
                module=name,
                thesis=f"Seasonality confidence: {confidence:.1%}",
            )
            self.smart_bus.set(
                "seasonal_voting_proposal",
                proposal,
                module=name,
                thesis=thesis,
            )
            self.smart_bus.set(
                "seasonal_confidence",
                confidence,
                module=name,
                thesis=f"Seasonal confidence: {confidence:.1%}",
            )
        except Exception:
            pass

        per_inst_vote = PerInstrumentVote(member=name)

        seasonality_risk_analysis = {
            "session": "unknown",
            "dow_bias": "unknown",
            "monthly_pattern": "unknown",
            "composite_score": 0.5,
            "rollover_risk": False,
            "weekend_risk": False,
            "action": "flat",
            "confidence": confidence,
            "per_instrument": {},
        }

        return {
            "SeasonalityRiskExpert_voting_proposal": proposal,
            "SeasonalityRiskExpert_confidence": confidence,
            "SeasonalityRiskExpert_per_instrument_votes": {},
            "per_instrument_votes": per_inst_vote,
            "seasonality_voting_proposal": proposal,
            "seasonality_confidence": confidence,
            "seasonal_voting_proposal": proposal,
            "seasonal_confidence": confidence,
            "seasonality_risk_analysis": seasonality_risk_analysis,
            "seasonality_analysis": {
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "composite_score": 0.5,
                "per_instrument": {},
            },
            "seasonality_expert_analysis": {
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "composite_score": 0.5,
                "rollover_risk": False,
                "weekend_risk": False,
                "per_instrument": {},
            },
            "seasonality_expert_thesis": thesis,
            "seasonal_session": "unknown",
            "seasonal_dow_bias": "unknown",
            "seasonal_monthly_pattern": "unknown",
            "seasonal_composite_score": 0.5,
            "seasonal_rollover_risk": False,
            "seasonal_weekend_risk": False,
            "_thesis": thesis,
        }
