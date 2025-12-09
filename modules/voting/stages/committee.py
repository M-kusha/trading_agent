"""
Committee Coordinator
=====================
Coordinates the voting committee by collecting votes from registered experts,
calculating weights, and determining committee decisions through weighted voting.

Refactored from voting_wrappers.py EnhancedVotingCommitteeCoordinator.
~600 lines (down from ~640 in original + ~200 duplicated base code)

v2 Highlights
-------------
- Singleton instance (one committee for the whole process).
- Registry + feed-based vote discovery with strict normalization.
- Directional vs neutral/gate action separation via ignore_actions.
- Performance & regime-aware expert weighting.
- Global decision + per-instrument aggregation (for multi-symbol trading).
- Publishes rich surfaces for:
    • ConsensusAnalyzer
    • CollusionDetector
    • HorizonAligner
    • UncertaintySampler
- Warmup handling and error/warmup/skip outputs are contract-compliant.
"""

from __future__ import annotations

import datetime
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.module_base import module
from modules.voting.core.base import VotingModuleBase
from modules.voting.core.constants import (
    CONSENSUS_THRESHOLD_F,
    MAX_STALENESS_SECONDS,
    VotingBusKeys,
    get_voting_mode,
)
# Per-instrument voting infrastructure
from modules.voting.core.per_instrument import (
    PerInstrumentVote,
    InstrumentProposal,
    DEFAULT_INSTRUMENTS,
    aggregate_all_instruments,
    normalize_instrument,
)


# Singleton instance for committee reuse (one committee across all calls)
_COMMITTEE_INSTANCE: Optional["CommitteeCoordinator"] = None


@module(**module_args("CommitteeCoordinator"))
class CommitteeCoordinator(VotingModuleBase):
    """
    Voting committee coordinator.

    Responsibilities:
    - Discover registered voting members from CONTRACTS + config.
    - Collect votes from experts (feed + per-expert bus keys).
    - Calculate performance + regime-weighted expert weights.
    - Determine committee decision via weighted voting.
    - Aggregate per-instrument decisions (EURUSD / XAUUSD / etc.).
    - Publish rich surfaces for downstream modules:
        • ConsensusAnalyzer (proposal vectors, confidences)
        • CollusionDetector (expert_votes, proposal_vectors)
        • HorizonAligner (committee_member_confidences, expert_weights)
        • UncertaintySampler (committee_proposal_vectors, per-instrument decisions)

    Uses a singleton pattern to maintain committee state across calls.

    Publishes (high level):
    - committee_decision
    - committee_confidence
    - committee_consensus
    - expert_votes
    - committee_votes / votes
    - committee_proposal_vectors
    - committee_decisions_by_instrument
    - member_confidences / committee_member_confidences
    - expert_weights / strategy_weights
    - committee_members / member_performance / committee_analytics
    """

    # ====================================================================== #
    # Singleton wiring
    # ====================================================================== #

    def __new__(cls, *args, **kwargs):
        global _COMMITTEE_INSTANCE
        if _COMMITTEE_INSTANCE is None:
            _COMMITTEE_INSTANCE = super().__new__(cls)
        return _COMMITTEE_INSTANCE

    # ====================================================================== #
    # Initialization
    # ====================================================================== #

    def _module_specific_init(self) -> None:
        """Initialize committee-specific state (only once, singleton-safe)."""
        # Prevent re-initialization of singleton
        if getattr(self, "_singleton_init_done", False):
            return

        # Committee configuration - use dynamic threshold for mode-awareness
        self._config_consensus_threshold = self.config.get("consensus_threshold")
        self.minimum_voters = int(self.config.get("minimum_voters", 2))
        self.performance_weighting = bool(
            self.config.get("performance_weighting", True)
        )
        self.max_vote_age_s = float(
            self.config.get("max_vote_age_s", MAX_STALENESS_SECONDS)
        )

        # Discovery settings
        self.expert_votes_bus_key = str(
            self.config.get("expert_votes_bus_key", "expert_votes")
        )
        self.discovery_mode = str(
            self.config.get("discovery_mode", "registry_only")
        ).lower()
        self.voter_flag_name = str(
            self.config.get("voter_flag_name", "is_voting_member")
        )
        self.voters_from_config = list(self.config.get("voters", []))
        self.ingest_minimum = int(self.config.get("ingest_minimum", self.minimum_voters))

        # Gate/risk actions should not be counted as directional votes.
        # These are risk module signals (proceed/caution/halt) that indicate safety, not direction.
        # NOTE: 'hold' and 'flat' ARE valid neutral signals from directional experts and should NOT
        # be filtered here—they represent "no trade" information. They are still treated as actions
        # in the committee distribution, but not as strong directional signals.
        self.ignore_actions = set(
            self.config.get(
                "ignore_actions",
                [
                    "abstain",
                    None,
                    "unknown",
                    # Risk gate actions (ExecutionQualityMonitor / AnomalyDetector / etc.)
                    "proceed",
                    "caution",
                    "halt",
                    "continue",
                    "confirm",
                    "wait",
                    # Legacy neutral actions that have been superseded by 'flat'
                    "seasonal_neutral",
                    "momentum_neutral",
                    "trend_neutral",
                    "theme_neutral",
                    # Session avoidance (legacy - now mapped to 'flat')
                    "session_avoid",
                    "session_optimal",
                    # High impact caution (legacy - now mapped to 'flat')
                    "high_impact_caution",
                ],
            )
        )
        self.max_votes_per_tick = int(self.config.get("max_votes_per_tick", 128))

        # Warmup configuration - wait for experts to have enough data before trading
        # TUNED: Reduced from 20 to 10 ticks for faster training startup
        self.warmup_ticks = int(self.config.get("warmup_ticks", 10))
        # Need at least 1 long/short vote (PPOAgent usually provides one)
        self.min_directional_votes = int(
            self.config.get("min_directional_votes", 1)
        )
        self.warmup_complete = False
        self._tick_count = 0

        # State
        self.active_experts: List[str] = []
        self.expert_weights: Dict[str, float] = {}
        self.voting_history: deque = deque(maxlen=100)
        self.consensus_history: deque = deque(maxlen=50)

        # Analytics
        self.committee_analytics: Dict[str, Any] = {
            "total_decisions": 0,
            "consensus_decisions": 0,
            "emergency_overrides": 0,
            "average_confidence": 0.5,
            "expert_performance": defaultdict(float),  # optional external updater
        }

        self._decision_counter = 0

        mode = get_voting_mode()
        self.logger.info(
            f"[COMMITTEE] CommitteeCoordinator initialized | MODE={mode} | "
            f"threshold={self.consensus_threshold:.1%} | "
            f"min_voters={self.minimum_voters} | "
            f"warmup_ticks={self.warmup_ticks} | "
            f"min_directional_votes={self.min_directional_votes}"
        )

        # Publish baseline keys
        self._publish_committee_baseline()

        self._singleton_init_done = True

    # ------------------------------------------------------------------ #
    # Consensus threshold (mode-aware)
    # ------------------------------------------------------------------ #

    @property
    def consensus_threshold(self) -> float:
        """Get consensus threshold (mode-aware, lazily delegated to constants)."""
        if self._config_consensus_threshold is not None:
            return float(self._config_consensus_threshold)
        return float(CONSENSUS_THRESHOLD_F())

    # ------------------------------------------------------------------ #
    # Baseline publication
    # ------------------------------------------------------------------ #

    def _publish_committee_baseline(self) -> None:
        """Publish baseline committee keys on the bus."""
        try:
            name = self.__class__.__name__

            self.smart_bus.set(
                "expert_votes",
                [],
                module=name,
                thesis="Baseline expert votes",
            )
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_VOTES,
                [],
                module=name,
                thesis="Baseline committee votes",
            )
            self.smart_bus.set(
                VotingBusKeys.VOTES,
                [],
                module=name,
                thesis="Baseline raw votes",
            )
            self.smart_bus.set(
                "expert_weights",
                {},
                module=name,
                thesis="Baseline expert weights",
            )
            self.smart_bus.set(
                "committee_proposal_vectors",
                [],
                module=name,
                thesis="Baseline proposal vectors",
            )
            self.smart_bus.set(
                "committee_decisions_by_instrument",
                {},
                module=name,
                thesis="Baseline per-instrument decisions",
            )
        except Exception:
            # Baseline is best-effort only
            pass

    # ====================================================================== #
    # Voter discovery
    # ====================================================================== #

    def _discover_voters(self) -> List[str]:
        """
        Discover registered voting members.

        Sources:
        1. Static config: self.voters_from_config
        2. CONTRACTS registry entries with meta[voter_flag_name] truthy
        3. Fallback hard-coded experts if nothing is found
        """
        discovered: List[str] = []

        # 1) Config voters first (explicit > inferred)
        for name in self.voters_from_config:
            if isinstance(name, str) and name and name not in discovered:
                discovered.append(name)

        # 2) Registry voters with voter_flag_name flag
        try:
            from modules.contracts import CONTRACTS

            for name, mc in CONTRACTS.items():
                try:
                    meta = getattr(mc, "meta", {}) or {}
                    if self._to_bool(meta.get(self.voter_flag_name, False)):
                        if name not in discovered:
                            discovered.append(name)
                except Exception:
                    continue
        except Exception:
            # CONTRACTS might not be fully wired in some test contexts
            pass

        # 3) Fallback safety for early development / tests
        if not discovered:
            for fallback in [
                "ThemeExpert",
                "SeasonalityRiskExpert",
                "EnhancedThemeExpert",
                "EnhancedSeasonalityRiskExpert",
            ]:
                if fallback not in discovered:
                    discovered.append(fallback)

        # De-duplicate while preserving order
        seen = set()
        result: List[str] = []
        for name in discovered:
            if name not in seen:
                result.append(name)
                seen.add(name)

        self.active_experts = result
        return result

    @staticmethod
    def _to_bool(v: Any, default: bool = False) -> bool:
        """Convert fuzzy config values to boolean."""
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        s = str(v).strip().lower()
        return s in ("true", "1", "yes", "y", "on")

    # ====================================================================== #
    # Vote collection
    # ====================================================================== #

    def _normalize_vote_entry(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize a raw vote entry to canonical format:

        {
          'expert': str,
          'vote':   dict (at least {'action': ...}),
          'confidence': float [0,1],
          'timestamp': ISO8601 string
        }
        """
        try:
            if not isinstance(raw, dict):
                return None

            expert = str(
                raw.get("expert") or raw.get("name") or "unknown"
            ).strip()
            if not expert:
                return None

            vote = raw.get("vote") or raw.get("proposal") or {}
            if not isinstance(vote, dict):
                vote = {}

            confidence = float(raw.get("confidence", 0.0) or 0.0)
            ts = raw.get("timestamp") or datetime.datetime.now().isoformat()

            return {
                "expert": expert,
                "vote": dict(vote),
                "confidence": max(0.0, min(1.0, confidence)),
                "timestamp": ts,
            }
        except Exception:
            return None

    def _voter_key_pairs(self, name: str) -> List[Tuple[str, str]]:
        """
        Get (proposal_key, confidence_key) pairs for a voter.

        Allows each expert to publish to:
        - canonical keys (VotingBusKeys.expert_proposal/confidence)
        - legacy / specialized keys (theme_voting_proposal, momentum_confidence, etc.)
        """
        pairs = [
            (
                VotingBusKeys.expert_proposal(name),
                VotingBusKeys.expert_confidence(name),
            )
        ]

        # Add alternate keys for known experts
        if name in ["EnhancedThemeExpert", "ThemeExpert"]:
            pairs.append(("theme_voting_proposal", "theme_confidence"))
        if name in ["EnhancedSeasonalityRiskExpert", "SeasonalityRiskExpert"]:
            pairs.append(("seasonality_voting_proposal", "seasonality_confidence"))
        if name == "MomentumExpert":
            pairs.append(("momentum_voting_proposal", "momentum_confidence"))
        if name == "TrendExpert":
            pairs.append(("trend_voting_proposal", "trend_confidence"))
        if name == "PPOAgent":
            pairs.append(("policy_actions", "agent_performance"))
        if name == "MetaAgent":
            pairs.append(("automation_decisions", "meta_performance"))

        return pairs

    async def _collect_expert_votes(self) -> List[Dict[str, Any]]:
        """
        Collect votes from registered experts.

        Collection pipeline:
        1) Discover voter list (config + CONTRACTS).
        2) Optionally read a feed from a shared bus key (expert_votes_bus_key).
        3) Read per-expert bus keys for proposals/confidences.
        4) Filter out purely gate / non-directional actions (ignore_actions)
           if any directional votes are present.
        5) Optionally add a UnifiedMemory meta-vote.
        """
        try:
            voters = self._discover_voters()
            voters_set = set(voters)
            expert_votes: List[Dict[str, Any]] = []
            now_ts = time.time()

            self.logger.debug(
                f"[COLLECT] Discovered {len(voters)} voters: {voters}"
            )
            self.logger.debug(
                f"[COLLECT] Discovery mode={self.discovery_mode}, "
                f"ingest_minimum={self.ingest_minimum}"
            )

            name = self.__class__.__name__

            # -------------------------------------------------------------- #
            # 1) Feed-first: read from expert_votes bus key
            # -------------------------------------------------------------- #
            if self.discovery_mode in ("feed_only", "feed_then_registry"):
                try:
                    feed = (
                        self.smart_bus.get(
                            self.expert_votes_bus_key, name
                        )
                        or []
                    )
                    self.logger.debug(
                        f"[COLLECT] Feed len={len(feed) if isinstance(feed, list) else 'n/a'} "
                        f"from key '{self.expert_votes_bus_key}'"
                    )
                    if isinstance(feed, list):
                        for raw in feed[-self.max_votes_per_tick :]:
                            norm = self._normalize_vote_entry(raw)
                            if norm and norm.get("expert") in voters_set:
                                # Staleness filter
                                ts_val = raw.get("timestamp") or norm.get(
                                    "timestamp"
                                )
                                if ts_val:
                                    try:
                                        age = now_ts - datetime.datetime.fromisoformat(
                                            str(ts_val)
                                        ).timestamp()
                                        if age > self.max_vote_age_s:
                                            continue
                                    except Exception:
                                        pass
                                expert_votes.append(norm)
                except Exception as e:
                    self.logger.warning(f"[COLLECT] Failed to read feed: {e}")

            # De-duplicate by expert (keep latest)
            by_expert: Dict[str, Dict[str, Any]] = {
                v["expert"]: v for v in expert_votes
            }
            expert_votes = list(by_expert.values())

            # -------------------------------------------------------------- #
            # 2) Registry-based per-voter keys
            # -------------------------------------------------------------- #
            if self.discovery_mode in ("registry_only", "feed_then_registry"):
                for voter_name in voters:
                    if voter_name in by_expert:
                        # Already have a fresh vote from feed
                        continue
                    try:
                        for prop_key, conf_key in self._voter_key_pairs(
                            voter_name
                        ):
                            proposal = self.smart_bus.get(
                                prop_key, name, default=None
                            )
                            if proposal is None:
                                continue
                            confidence = self.smart_bus.get(
                                conf_key, name, default=None
                            )
                            if confidence is None:
                                continue

                            raw = {
                                "expert": voter_name,
                                "vote": dict(proposal)
                                if isinstance(proposal, dict)
                                else {
                                    "action": str(proposal),
                                },
                                "confidence": float(confidence)
                                if isinstance(confidence, (int, float))
                                else 0.5,
                                "timestamp": datetime.datetime.now().isoformat(),
                            }
                            norm = self._normalize_vote_entry(raw)
                            if norm:
                                by_expert[voter_name] = norm
                            break
                    except Exception as e:
                        self.logger.warning(
                            f"[COLLECT] Failed to collect vote from {voter_name}: {e}"
                        )

                expert_votes = list(by_expert.values())

            # -------------------------------------------------------------- #
            # 3) Filter gate / neutral actions if directional votes exist
            # -------------------------------------------------------------- #
            before_count = len(expert_votes)
            actions_collected = [
                v.get("vote", {}).get("action", "unknown")
                for v in expert_votes
            ]

            if any(
                v.get("vote", {}).get("action") not in self.ignore_actions
                for v in expert_votes
            ):
                expert_votes = [
                    v
                    for v in expert_votes
                    if v.get("vote", {}).get("action") not in self.ignore_actions
                ]
                filtered_count = before_count - len(expert_votes)
                if filtered_count > 0:
                    self.logger.debug(
                        f"[FILTER] Filtered {filtered_count} neutral/gate votes. "
                        f"Actions: {actions_collected}"
                    )

            # -------------------------------------------------------------- #
            # 4) Add UnifiedMemory vote if available (meta-information)
            # -------------------------------------------------------------- #
            expert_votes = await self._add_memory_vote(expert_votes)

            # Cap total votes for safety
            if len(expert_votes) > self.max_votes_per_tick:
                expert_votes = expert_votes[-self.max_votes_per_tick :]

            self.logger.info(
                f"[VOTES] Collected {len(expert_votes)} votes from {len(voters)} voters"
            )

            return expert_votes

        except Exception as e:
            self.logger.error(f"[COLLECT] Vote collection failed: {e}")
            return []

    async def _add_memory_vote(
        self, expert_votes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add UnifiedMemory meta-vote if available.

        Memory vote carries:
        - signed_bias: direction hint (-1..+1)
        - vote_value: magnitude / strength
        - confidence: how sure memory is
        - neural_risk_hint: how dangerous conditions look (0-1)

        When bias is small or confidence low, we skip adding it.
        """
        try:
            name = self.__class__.__name__
            memory_vote = self.smart_bus.get("memory_vote", name)
            if not isinstance(memory_vote, dict):
                return expert_votes

            signed_bias = float(memory_vote.get("signed_bias", 0.0))
            mem_confidence = float(memory_vote.get("confidence", 0.5))

            if abs(signed_bias) <= 0.05 or mem_confidence <= 0.2:
                return expert_votes

            # Determine action from bias
            if signed_bias > 0.1:
                mem_action = "long"
            elif signed_bias < -0.1:
                mem_action = "short"
            else:
                mem_action = "abstain"

            # Adjust confidence by neural risk hint (higher risk → lower confidence)
            neural_risk = float(memory_vote.get("neural_risk_hint", 0.5))
            adjusted_confidence = mem_confidence * (1.0 - neural_risk * 0.3)

            memory_entry = {
                "expert": "UnifiedMemory",
                "vote": {
                    "action": mem_action,
                    "signal_strength": abs(
                        float(memory_vote.get("vote_value", 0.0))
                    ),
                    "signed_bias": signed_bias,
                },
                "confidence": max(0.0, min(1.0, adjusted_confidence)),
                "timestamp": datetime.datetime.now().isoformat(),
            }

            expert_votes.append(memory_entry)

            self.logger.info(
                f"[MEMORY] MEMORY_VOTE_ADDED | action={mem_action} | "
                f"confidence={adjusted_confidence:.2f}"
            )

            return expert_votes

        except Exception as e:
            self.logger.debug(f"[MEMORY] Memory vote skipped: {e}")
            return expert_votes

    # ====================================================================== #
    # Expert weights
    # ====================================================================== #

    async def _calculate_expert_weights(
        self, expert_votes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Calculate performance-weighted expert weights.

        Base weight per expert:
        - Start from confidence (0..1).
        - Optionally multiply by (0.5 + perf) where perf is 0..1
          from bus['expert_performance'] or committee_analytics.
        - Optionally adjust by market regime (trending/volatile/ranging).
        - Normalize to sum to 1.
        """
        try:
            if not expert_votes:
                return {}

            name = self.__class__.__name__

            # Get bus-level performance (optional)
            bus_perf = (
                self.smart_bus.get("expert_performance", name) or {}
            )
            if not isinstance(bus_perf, dict):
                bus_perf = {}

            # Filter gate/neutral actions if directional votes exist
            actions = [v.get("vote", {}).get("action") for v in expert_votes]
            if any(a not in self.ignore_actions for a in actions):
                expert_votes = [
                    v
                    for v in expert_votes
                    if v.get("vote", {}).get("action") not in self.ignore_actions
                ]

            # Normalized regime for adjustments
            regime_raw = self.smart_bus.get("market_regime", name) or "unknown"
            regime = str(regime_raw).lower()

            weights: Dict[str, float] = {}
            total = 0.0

            for v in expert_votes:
                name_expert = v["expert"]
                conf = float(v.get("confidence", 0.0))
                base = max(0.0, conf)

                if self.performance_weighting:
                    perf = bus_perf.get(name_expert)
                    if perf is None:
                        perf = float(
                            self.committee_analytics["expert_performance"].get(
                                name_expert, 0.5
                            )
                        )
                    else:
                        try:
                            perf = float(perf)
                        except Exception:
                            perf = 0.5

                    base *= 0.5 + max(0.0, min(1.0, perf))

                # Regime adjustment (theme / momentum / trend experts)
                base *= self._get_regime_adjustment(name_expert, regime)

                w = max(1e-6, min(2.0, base))
                weights[name_expert] = w
                total += w

            if total <= 0.0:
                # Fallback: equal weight among participants
                n = len(expert_votes)
                if n == 0:
                    return {}
                return {v["expert"]: 1.0 / n for v in expert_votes}

            # Normalize to probability simplex
            for k in weights:
                weights[k] = weights[k] / total

            self.expert_weights = weights
            return weights

        except Exception as e:
            self.logger.error(f"[WEIGHTS] Weight calculation failed: {e}")
            return {}

    def _get_regime_adjustment(self, expert_name: str, regime: str) -> float:
        """
        Get regime-based weight adjustment for expert.

        Intended to give:
        - Theme/Trend experts more weight in trending regimes.
        - Seasonality more weight in ranging/seasonal regimes.
        - Momentum more weight in volatile/trending regimes.
        """
        regime = str(regime).lower()
        adjustments = {
            "ThemeExpert": {"trending": 1.2, "volatile": 0.9, "ranging": 1.0},
            "EnhancedThemeExpert": {
                "trending": 1.2,
                "volatile": 0.9,
                "ranging": 1.0,
            },
            "SeasonalityRiskExpert": {
                "trending": 1.0,
                "volatile": 1.1,
                "ranging": 1.2,
            },
            "EnhancedSeasonalityRiskExpert": {
                "trending": 1.0,
                "volatile": 1.1,
                "ranging": 1.2,
            },
            # MomentumExpert: strong in volatile/trending, weaker in ranging
            "MomentumExpert": {
                "trending": 1.3,
                "volatile": 1.2,
                "ranging": 0.8,
            },
            # TrendExpert: strong in trending, weaker in volatile/ranging
            "TrendExpert": {"trending": 1.4, "volatile": 0.85, "ranging": 0.7},
        }
        return float(adjustments.get(expert_name, {}).get(regime, 1.0))

    # ====================================================================== #
    # Decision making
    # ====================================================================== #

    @staticmethod
    def _action_sign(action: str) -> float:
        """
        Map action string to directional sign.

        Returns:
            +1.0 for bullish/long actions
            -1.0 for bearish/short actions
            0.0 for neutral/gate actions
        """
        if not action:
            return 0.0

        a = action.strip().lower()

        # Long/bullish actions
        if a.startswith("long") or "long_" in a or a == "buy":
            return 1.0
        # Short/bearish actions
        if a.startswith("short") or "short_" in a or a == "sell":
            return -1.0
        # Risk reduction (bearish tilt)
        if a in ("reduce_risk", "halt", "emergency_stop"):
            return -1.0
        # Risk increase (bullish tilt)
        if a in ("increase_risk", "aggressive"):
            return 1.0
        # Neutral/gate actions
        if a in (
            "hold",
            "abstain",
            "wait",
            "neutral",
            "proceed",
            "continue",
            "caution",
        ):
            return 0.0
        # Thematic bullish actions
        if a in (
            "trend_following",
            "seasonal_long_bias",
            "long_risk_assets",
            "risk_on",
            "bullish",
            "breakout",
        ):
            return 1.0
        # Thematic bearish actions
        if a in (
            "seasonal_short_bias",
            "short_risk_assets",
            "safe_haven_rotation",
            "volatility_hedging",
            "risk_off",
            "bearish",
            "mean_reversion",
        ):
            return -1.0
        # MomentumExpert actions
        if a in ("momentum_long",):
            return 1.0
        if a in ("momentum_short",):
            return -1.0
        if a in ("momentum_neutral",):
            return 0.0
        # TrendExpert actions
        if a in ("trend_bullish",):
            return 1.0
        if a in ("trend_bearish",):
            return -1.0
        if a in ("trend_neutral",):
            return 0.0
        # ThemeExpert neutral actions
        if a in ("theme_neutral",):
            return 0.0
        # SeasonalityRiskExpert neutral/timing actions
        if a in (
            "seasonal_neutral",
            "session_avoid",
            "session_optimal",
            "high_impact_caution",
        ):
            return 0.0

        return 0.0

    async def _determine_committee_decision(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Determine final committee decision via weighted voting.

        - Build action→weight map.
        - If any non-ignore action exists, exclude ignore_actions when computing
          the winning action and consensus strength.
        """
        try:
            if not expert_votes:
                return {"action": "abstain", "reason": "no_expert_votes"}

            weighted_actions: Dict[str, float] = defaultdict(float)
            total_weight = 0.0

            for v in expert_votes:
                name_expert = v["expert"]
                w = float(expert_weights.get(name_expert, 1.0))
                action = v.get("vote", {}).get("action", "abstain")
                weighted_actions[action] += w
                total_weight += w

            if not weighted_actions or total_weight <= 0.0:
                return {"action": "abstain", "reason": "no_valid_actions"}

            # If non-abstain votes exist, ignore pure gate actions in argmax
            has_non_ignore = any(
                a not in self.ignore_actions for a in weighted_actions
            )
            if has_non_ignore:
                filtered = {
                    a: w
                    for a, w in weighted_actions.items()
                    if a not in self.ignore_actions
                }
                if not filtered:
                    # All actions were ignore_actions; treat as abstain
                    return {
                        "action": "abstain",
                        "reason": "only_gate_actions",
                    }
                best_action, best_weight = max(
                    filtered.items(), key=lambda x: x[1]
                )
                denom = sum(filtered.values()) or total_weight
            else:
                best_action, best_weight = max(
                    weighted_actions.items(), key=lambda x: x[1]
                )
                denom = total_weight

            consensus_strength = best_weight / denom if denom > 0 else 0.0

            return {
                "action": best_action,
                "consensus_strength": consensus_strength,
                "total_weight": total_weight,
                "action_weights": dict(weighted_actions),
                "decision_type": "consensus"
                if consensus_strength >= self.consensus_threshold
                else "plurality",
            }

        except Exception as e:
            self.logger.error(f"[DECISION] Decision determination failed: {e}")
            return {"action": "abstain", "reason": f"error: {e}"}

    async def _analyze_voting_consensus(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Analyze voting consensus and identify conflicts.

        Measures:
        - consensus_strength: dominant action weight / total weight
        - conflict_level: entropy of action distribution
        """
        try:
            if not expert_votes:
                return {"consensus_exists": False, "reason": "no_votes"}

            action_weights: Dict[str, float] = defaultdict(float)
            for v in expert_votes:
                action = v.get("vote", {}).get("action", "abstain")
                w = float(expert_weights.get(v["expert"], 1.0))
                action_weights[action] += w

            total_weight = sum(action_weights.values())
            if total_weight <= 0.0:
                return {"consensus_exists": False, "reason": "zero_weight"}

            dominant_action, dom_w = max(
                action_weights.items(), key=lambda x: x[1]
            )
            consensus_strength = dom_w / total_weight

            conflict_level = self._assess_conflict_level(
                action_weights, total_weight
            )

            return {
                "consensus_exists": consensus_strength
                >= self.consensus_threshold,
                "consensus_strength": consensus_strength,
                "dominant_action": dominant_action,
                "action_distribution": dict(action_weights),
                "conflict_level": conflict_level,
                "vote_count": len(expert_votes),
                "total_weight": total_weight,
            }

        except Exception as e:
            return {"consensus_exists": False, "error": str(e)}

    @staticmethod
    def _assess_conflict_level(
        action_weights: Dict[str, float], total_weight: float
    ) -> str:
        """Assess conflict level via normalized entropy over actions."""
        try:
            if len(action_weights) <= 1:
                return "NONE"

            probs = [
                w / total_weight for w in action_weights.values() if total_weight > 0
            ]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(len(action_weights))
            normalized = (entropy / max_entropy) if max_entropy > 0 else 0.0

            if normalized < 0.3:
                return "LOW"
            if normalized < 0.6:
                return "MEDIUM"
            if normalized < 0.8:
                return "HIGH"
            return "SEVERE"

        except Exception:
            return "UNKNOWN"

    async def _calculate_committee_confidence(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float],
        consensus: Dict[str, Any],
    ) -> float:
        """
        Calculate overall committee confidence.

        - Weighted average of expert confidences.
        - Scaled by consensus_strength (more consensus → stronger confidence).
        """
        try:
            if not expert_votes or not expert_weights:
                return 0.1

            weighted_conf = 0.0
            total_w = 0.0

            for v in expert_votes:
                name_expert = v["expert"]
                conf = float(v.get("confidence", 0.0))
                w = float(expert_weights.get(name_expert, 1.0))
                weighted_conf += conf * w
                total_w += w

            avg_conf = weighted_conf / total_w if total_w > 0 else 0.0
            consensus_strength = float(consensus.get("consensus_strength", 0.0))

            # Scale by consensus (0.7..1.3 factor)
            final = avg_conf * (0.7 + 0.6 * consensus_strength)
            return max(0.1, min(1.0, final))

        except Exception:
            return 0.3

    async def _generate_committee_thesis(
        self,
        decision: Dict[str, Any],
        confidence: float,
        consensus: Dict[str, Any],
        expert_votes: List[Dict[str, Any]],
    ) -> str:
        """Generate a human-readable committee thesis string."""
        try:
            action = decision.get("action", "unknown")
            cs = float(consensus.get("consensus_strength", 0.0))
            label = (
                "HIGH"
                if confidence > 0.7
                else "MODERATE"
                if confidence > 0.4
                else "LOW"
            )

            parts = [
                f"COMMITTEE DECISION: {action.upper()} with {label} confidence ({confidence:.1%})"
            ]

            if consensus.get("consensus_exists"):
                parts.append(
                    f"STRONG CONSENSUS: {cs:.1%} agreement among {len(expert_votes)} experts"
                )
            else:
                parts.append(
                    f"DIVIDED OPINION: {consensus.get('conflict_level', 'UNKNOWN')} conflict"
                )

            parts.append(
                f"DECISION TYPE: {decision.get('decision_type', 'unknown').upper()}"
            )

            return " | ".join(parts)

        except Exception as e:
            return f"Committee thesis failed: {e}"

    # ====================================================================== #
    # Per-instrument voting
    # ====================================================================== #

    def _convert_to_per_instrument_votes(
        self, expert_votes: List[Dict[str, Any]]
    ) -> List[PerInstrumentVote]:
        """
        Convert expert votes to PerInstrumentVote format.

        Priority order for per-instrument data:
        1. SmartInfoBus: {Expert}_per_instrument_votes (preferred modern format)
        2. vote_data['proposals'] dict
        3. vote['proposals'] dict
        4. Legacy global vote: apply same action to all DEFAULT_INSTRUMENTS
        """
        per_inst_votes: List[PerInstrumentVote] = []
        name = self.__class__.__name__

        for vote_data in expert_votes:
            try:
                expert_name = str(vote_data.get("expert", "unknown"))
                vote = vote_data.get("vote", {}) or {}
                confidence = float(vote_data.get("confidence", 0.5))

                piv: Optional[PerInstrumentVote] = None

                # PRIORITY 1: Bus-level per-instrument votes
                per_inst_key = f"{expert_name}_per_instrument_votes"
                bus_per_inst = self.smart_bus.get(
                    per_inst_key, name, default=None
                )

                if (
                    bus_per_inst
                    and isinstance(bus_per_inst, dict)
                    and len(bus_per_inst) > 0
                ):
                    piv = PerInstrumentVote(member=expert_name)
                    for inst, inst_vote in bus_per_inst.items():
                        if not isinstance(inst_vote, dict):
                            continue
                        inst_norm = normalize_instrument(inst)
                        inst_action = str(inst_vote.get("action", "flat")).lower()
                        inst_conf = float(
                            inst_vote.get("confidence", confidence)
                        )
                        inst_mag = float(
                            inst_vote.get(
                                "magnitude",
                                inst_vote.get(
                                    "signal_strength", inst_conf
                                ),
                            )
                        )
                        inst_rationale = str(
                            inst_vote.get(
                                "rationale",
                                f"Per-inst vote from {expert_name}",
                            )
                        )

                        piv.set_proposal(
                            InstrumentProposal(
                                instrument=inst_norm,
                                action=inst_action,
                                confidence=inst_conf,
                                magnitude=inst_mag,
                                rationale=inst_rationale,
                            )
                        )

                    if piv.proposals:
                        self.logger.debug(
                            f"[PER-INST] {expert_name}: bus per-instrument votes "
                            f"({len(piv.proposals)} instruments)"
                        )
                        per_inst_votes.append(piv)
                        continue

                # PRIORITY 2: vote_data['proposals'] dict
                if "proposals" in vote_data and isinstance(
                    vote_data["proposals"], dict
                ):
                    piv = PerInstrumentVote.from_dict(
                        {
                            "member": expert_name,
                            "proposals": vote_data["proposals"],
                            "action": vote_data.get("action", "flat"),
                            "confidence": confidence,
                        }
                    )
                    self.logger.debug(
                        f"[PER-INST] {expert_name}: proposals in vote_data "
                        f"({len(piv.proposals)} instruments)"
                    )

                # PRIORITY 3: nested vote['proposals']
                elif "proposals" in vote and isinstance(
                    vote["proposals"], dict
                ):
                    piv = PerInstrumentVote.from_dict(
                        {
                            "member": expert_name,
                            "proposals": vote["proposals"],
                            "action": vote.get("action", "flat"),
                            "confidence": confidence,
                        }
                    )
                    self.logger.debug(
                        f"[PER-INST] {expert_name}: proposals in vote dict "
                        f"({len(piv.proposals)} instruments)"
                    )

                # PRIORITY 4: legacy global vote
                else:
                    action = str(vote.get("action", "flat")).lower()
                    magnitude = float(
                        vote.get(
                            "signal_strength",
                            vote.get("magnitude", confidence),
                        )
                    )

                    piv = PerInstrumentVote(member=expert_name)
                    for inst in DEFAULT_INSTRUMENTS:
                        piv.set_proposal(
                            InstrumentProposal(
                                instrument=inst,
                                action=action,
                                confidence=confidence,
                                magnitude=magnitude,
                                rationale=f"Legacy global vote from {expert_name}",
                            )
                        )
                    self.logger.debug(
                        f"[PER-INST] {expert_name}: LEGACY global vote ({action}) "
                        f"for {len(DEFAULT_INSTRUMENTS)} instruments"
                    )

                if piv is not None:
                    per_inst_votes.append(piv)

            except Exception as e:
                self.logger.warning(
                    f"[PER-INST] Failed to convert vote to per-instrument: {e}"
                )

        return per_inst_votes

    async def _aggregate_per_instrument(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate votes per instrument.

        Returns:
            Dict[instrument -> aggregated decision dict] with:
            - action
            - confidence
            - consensus_score
            - vote_count (long/short/flat counts)
            - weighted_score (signed committee score)
        """
        per_inst_votes = self._convert_to_per_instrument_votes(expert_votes)

        # Debug trace per-instrument votes
        for piv in per_inst_votes:
            for inst, prop in piv.proposals.items():
                self.logger.debug(
                    f"[PER-INST VOTE] {piv.member} → {inst}: "
                    f"action={prop.action}, conf={prop.confidence:.2f}, "
                    f"mag={prop.magnitude:.2f}"
                )

        aggregated = aggregate_all_instruments(
            votes=per_inst_votes,
            instruments=DEFAULT_INSTRUMENTS,
            weights=expert_weights,
        )

        result: Dict[str, Dict[str, Any]] = {}
        for inst, decision in aggregated.items():
            result[inst] = {
                "action": decision.action,
                "confidence": decision.confidence,
                "consensus_score": decision.consensus_score,
                "vote_count": decision.vote_count,
                "long_votes": decision.long_votes,
                "short_votes": decision.short_votes,
                "flat_votes": decision.flat_votes,
                "weighted_score": decision.weighted_score,
                "instrument": inst,
            }
            self.logger.info(
                f"[COMMITTEE] {inst}: action={decision.action}, "
                f"conf={decision.confidence:.2f}, "
                f"consensus={decision.consensus_score:.2f}, "
                f"votes={decision.long_votes}L/"
                f"{decision.short_votes}S/"
                f"{decision.flat_votes}F"
            )

        return result

    # ====================================================================== #
    # Main processing
    # ====================================================================== #

    async def process(self, **inputs: Any) -> Dict[str, Any]:
        """Main committee processing entrypoint."""
        start_time = time.time()
        name = self.__class__.__name__

        try:
            # -------------------------------------------------------------- #
            # 0) Price-change short-circuit: if feed says nothing changed,
            #    skip heavy committee work and output a flat/skip decision.
            # -------------------------------------------------------------- #
            provider_status = self.smart_bus.get("provider_status", name)
            if isinstance(provider_status, dict) and not provider_status.get(
                "price_changed", True
            ):
                skip_decision_id = (
                    self.smart_bus.get("kernel_decision_id", name) or "skip"
                )
                # Also update a minimal, flat-like state on the bus to avoid
                # stale downstream signals.
                self.smart_bus.set(
                    VotingBusKeys.COMMITTEE_DECISION,
                    {"action": "flat", "reason": "no_price_change"},
                    module=name,
                    thesis="No price change - skipped voting",
                )
                self.smart_bus.set(
                    "committee_proposal_vectors",
                    [],
                    module=name,
                    thesis="No price change - no new proposals",
                )
                self.smart_bus.set(
                    "committee_decisions_by_instrument",
                    {},
                    module=name,
                    thesis="No price change - no per-instrument decisions",
                )

                return {
                    "action": "flat",
                    "confidence": 0.0,
                    "magnitude": 0.0,
                    "decision_id": skip_decision_id,
                    "reasoning": "No price change detected - skipped voting",
                    "skipped": True,
                    # Contract-required outputs
                    VotingBusKeys.COMMITTEE_DECISION: {
                        "action": "flat",
                        "reason": "no_price_change",
                    },
                    VotingBusKeys.COMMITTEE_CONSENSUS: {
                        "consensus_exists": False,
                        "skipped": True,
                    },
                    VotingBusKeys.COMMITTEE_CONFIDENCE: 0.0,
                    VotingBusKeys.COMMITTEE_VOTES: [],
                    VotingBusKeys.COMMITTEE_DECISION_ID: skip_decision_id,
                    "committee_summary": {
                        "decision": "flat",
                        "skipped": True,
                    },
                    "committee_proposal_vectors": [],
                    "committee_decisions_by_instrument": {},
                    "raw_proposals": {},
                    "member_confidences": {},
                    "voting_weights": {},
                    "committee_member_confidences": {},
                    "strategy_weights": {},
                    "member_performance": {},
                    VotingBusKeys.VOTES: [],
                    VotingBusKeys.VOTING_SUMMARY: {
                        "action": "flat",
                        "skipped": True,
                    },
                    VotingBusKeys.STRATEGY_ARBITER_WEIGHTS: {},
                    "expert_votes": [],
                    "expert_weights": {},
                    "committee_analytics": {},
                    "committee_members": [],
                    "n_members": 0,
                    "proposal_vectors": [],
                    "time_of_day": 0,
                    "trade_vote": {
                        "action": "flat",
                        "confidence": 0.0,
                    },
                    "trade_vote_v2": {
                        "action": "flat",
                        "size": 0.0,
                        "confidence": 0.0,
                        "consensus_score": 0.0,
                    },
                    "_thesis": "No price change - skipped voting",
                }

            # Track warmup progress
            self._tick_count += 1

            # Get or synthesize decision_id
            decision_id = self.smart_bus.get("kernel_decision_id", name)
            if not decision_id:
                self._decision_counter += 1
                decision_id = (
                    f"{datetime.datetime.now().isoformat()}#{self._decision_counter}"
                )

            # -------------------------------------------------------------- #
            # 1) Collect votes & compute weights
            # -------------------------------------------------------------- #
            expert_votes = await self._collect_expert_votes()
            expert_weights = await self._calculate_expert_weights(expert_votes)

            # Count directional votes (non-gate actions)
            directional_votes = [
                v
                for v in expert_votes
                if v.get("vote", {}).get("action") not in self.ignore_actions
            ]
            n_directional = len(directional_votes)

            # -------------------------------------------------------------- #
            # 2) Warmup handling
            # -------------------------------------------------------------- #
            if not self.warmup_complete:
                if (
                    self._tick_count >= self.warmup_ticks
                    and n_directional >= self.min_directional_votes
                ):
                    self.warmup_complete = True
                    self.logger.info(
                        f"[WARMUP] ✅ Complete after {self._tick_count} ticks with "
                        f"{n_directional} directional votes"
                    )
                    self.smart_bus.set(
                        "warmup_status",
                        {
                            "complete": True,
                            "tick_count": self._tick_count,
                            "warmup_ticks": self.warmup_ticks,
                            "directional_votes": n_directional,
                            "min_directional_votes": self.min_directional_votes,
                            "progress_pct": 100,
                        },
                        module=name,
                        thesis="Warmup complete - trading enabled",
                    )
                else:
                    warmup_reason = (
                        f"Warmup in progress: tick {self._tick_count}/{self.warmup_ticks}, "
                        f"{n_directional}/{self.min_directional_votes} directional votes"
                    )
                    self.logger.info(f"[WARMUP] ⏳ {warmup_reason}")
                    self.smart_bus.set(
                        "warmup_status",
                        {
                            "complete": False,
                            "tick_count": self._tick_count,
                            "warmup_ticks": self.warmup_ticks,
                            "directional_votes": n_directional,
                            "min_directional_votes": self.min_directional_votes,
                            "progress_pct": min(
                                100,
                                (self._tick_count / max(1, self.warmup_ticks))
                                * 100,
                            ),
                        },
                        module=name,
                        thesis=warmup_reason,
                    )
                    return self._warmup_output(warmup_reason, decision_id)

            # -------------------------------------------------------------- #
            # 3) Global decision / consensus / confidence / thesis
            # -------------------------------------------------------------- #
            decision = await self._determine_committee_decision(
                expert_votes, expert_weights
            )
            consensus = await self._analyze_voting_consensus(
                expert_votes, expert_weights
            )
            confidence = await self._calculate_committee_confidence(
                expert_votes, expert_weights, consensus
            )
            thesis = await self._generate_committee_thesis(
                decision, confidence, consensus, expert_votes
            )

            # -------------------------------------------------------------- #
            # 4) Per-instrument decisions
            # -------------------------------------------------------------- #
            per_instrument_decisions = await self._aggregate_per_instrument(
                expert_votes, expert_weights
            )

            # Publish per-instrument decisions to bus (for UncertaintySampler, etc.)
            self.smart_bus.set(
                "committee_decisions_by_instrument",
                per_instrument_decisions,
                module=name,
                thesis=f"Per-instrument decisions for {len(per_instrument_decisions)} instruments",
            )

            # -------------------------------------------------------------- #
            # 5) Build analytics surfaces
            # -------------------------------------------------------------- #
            committee_members = [
                v.get("expert", "unknown") for v in expert_votes
            ]
            n_members = len(committee_members)
            member_confidences_map: Dict[str, float] = {
                v["expert"]: float(v.get("confidence", 0.0)) for v in expert_votes
            }

            # Vectorize proposals for Consensus/Collusion/Uncertainty
            proposal_vectors: List[List[float]] = []
            for v in expert_votes:
                vote = v.get("vote") or {}
                action = str(vote.get("action", "abstain")).lower()
                ss = float(
                    vote.get("signal_strength", v.get("confidence", 0.0)) or 0.0
                )
                conf_v = float(v.get("confidence", 0.0))
                proposal_vectors.append(
                    [self._action_sign(action) * ss, conf_v]
                )

            # -------------------------------------------------------------- #
            # 6) Bus update
            # -------------------------------------------------------------- #
            await self._update_bus(
                decision=decision,
                consensus=consensus,
                confidence=confidence,
                expert_votes=expert_votes,
                expert_weights=expert_weights,
                committee_members=committee_members,
                proposal_vectors=proposal_vectors,
                member_confidences_map=member_confidences_map,
                decision_id=decision_id,
                thesis=thesis,
            )

            # Record decision for analytics
            self._record_decision(
                decision, confidence, consensus, expert_votes
            )

            # -------------------------------------------------------------- #
            # 7) Build return payload
            # -------------------------------------------------------------- #
            elapsed_ms = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric(
                name, "process", elapsed_ms, True
            )

            now = datetime.datetime.now()
            time_of_day = (
                now.hour * 60 + now.minute + now.second / 60.0
            )

            # Simplified committee_votes format
            committee_votes = [
                {
                    "action": v.get("vote", {}).get("action", "abstain"),
                    "confidence": float(v.get("confidence", 0.0)),
                    "expert": v.get("expert", "unknown"),
                }
                for v in expert_votes
            ]

            # raw_proposals per expert
            raw_proposals = {
                v.get("expert", "unknown"): v.get("vote", {}) for v in expert_votes
            }

            committee_summary = {
                "total_members": n_members,
                "participating_members": len(expert_votes),
                "decision": decision.get("action", "abstain"),
                "confidence": confidence,
                "consensus_strength": consensus.get("consensus_strength", 0.0),
                "timestamp": datetime.datetime.now().isoformat(),
            }

            voting_summary = {
                "action": decision.get("action", "abstain"),
                "confidence": confidence,
                "member_count": n_members,
                "consensus": consensus,
            }

            return {
                # Primary outputs (new v5.0)
                VotingBusKeys.COMMITTEE_DECISION: decision,
                VotingBusKeys.COMMITTEE_CONSENSUS: consensus,
                VotingBusKeys.COMMITTEE_CONFIDENCE: confidence,
                VotingBusKeys.COMMITTEE_VOTES: committee_votes,
                "committee_summary": committee_summary,
                VotingBusKeys.COMMITTEE_DECISION_ID: decision_id,
                "raw_proposals": raw_proposals,
                "member_confidences": member_confidences_map,
                "voting_weights": expert_weights,
                # Per-instrument decisions
                "committee_decisions_by_instrument": per_instrument_decisions,
                # Backward compatibility
                VotingBusKeys.VOTES: committee_votes,
                VotingBusKeys.VOTING_SUMMARY: voting_summary,
                VotingBusKeys.STRATEGY_ARBITER_WEIGHTS: expert_weights,
                "expert_votes": expert_votes,
                "expert_weights": expert_weights,
                "committee_member_confidences": member_confidences_map,
                "strategy_weights": expert_weights,
                "member_performance": {
                    m: {
                        "confidence": float(
                            member_confidences_map.get(m, 0.5)
                        ),
                        "weight": float(expert_weights.get(m, 0.5)),
                    }
                    for m in committee_members
                },
                "committee_analytics": dict(self.committee_analytics),
                "committee_members": committee_members,
                "n_members": n_members,
                "proposal_vectors": proposal_vectors,
                "committee_proposal_vectors": proposal_vectors,
                "decision_id": decision_id,
                "time_of_day": time_of_day,
                # Trade vote outputs
                "trade_vote": {
                    "action": decision.get("action", "abstain"),
                    "confidence": float(confidence),
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                "trade_vote_v2": {
                    "action": decision.get("action", "abstain"),
                    "size": float(confidence),
                    "confidence": float(confidence),
                    "consensus_score": float(
                        consensus.get("consensus_strength", 0.0)
                    ),
                    "decision_id": decision_id,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                "_thesis": thesis,
            }

        except Exception as e:
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(
                    e, "committee_process"
                )
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)

    # ====================================================================== #
    # Bus update, recording, warmup/error outputs
    # ====================================================================== #

    async def _update_bus(
        self,
        decision: Dict[str, Any],
        consensus: Dict[str, Any],
        confidence: float,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float],
        committee_members: List[str],
        proposal_vectors: List[List[float]],
        member_confidences_map: Dict[str, float],
        decision_id: str,
        thesis: str,
    ) -> None:
        """Update SmartInfoBus with committee results for downstream modules."""
        try:
            name = self.__class__.__name__

            # Primary decision + consensus surfaces
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_DECISION,
                decision,
                module=name,
                thesis=thesis,
                confidence=confidence,
            )
            # Alias for older consumers
            self.smart_bus.set(
                "committee_decision",
                decision,
                module=name,
                thesis="Committee decision (alias)",
            )

            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_CONSENSUS,
                consensus,
                module=name,
                thesis="Committee consensus",
            )
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_CONFIDENCE,
                confidence,
                module=name,
                thesis=f"Confidence: {confidence:.1%}",
            )

            # Expert votes snapshot
            self.smart_bus.set(
                "expert_votes",
                list(expert_votes),
                module=name,
                thesis="Expert votes snapshot",
            )

            # Simplified committee votes (action/confidence/expert)
            simplified = [
                {
                    "action": v.get("vote", {}).get("action", "abstain"),
                    "confidence": float(v.get("confidence", 0.0)),
                    "expert": v.get("expert", "unknown"),
                }
                for v in expert_votes
            ]
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_VOTES,
                simplified,
                module=name,
                thesis="Committee votes",
            )

            # Analytics surfaces
            self.smart_bus.set(
                "committee_members",
                committee_members,
                module=name,
                thesis="Members list",
            )
            self.smart_bus.set(
                "committee_proposal_vectors",
                proposal_vectors,
                module=name,
                thesis="Proposal vectors",
            )

            # Order confidences aligned with committee_members
            ordered_confidences = [
                float(member_confidences_map.get(m, 0.0))
                for m in committee_members
            ]
            self.smart_bus.set(
                VotingBusKeys.MEMBER_CONFIDENCES,
                ordered_confidences,
                module=name,
                thesis="Member confidences",
            )
            # Alias for Consensus/Horizon/Uncertainty modules
            self.smart_bus.set(
                "committee_member_confidences",
                ordered_confidences,
                module=name,
                thesis="Committee member confidences (alias)",
            )

            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_DECISION_ID,
                decision_id,
                module=name,
                thesis="Decision ID",
            )
            self.smart_bus.set(
                "committee_analytics",
                dict(self.committee_analytics),
                module=name,
                thesis="Committee analytics",
            )

            # Publish expert weights for HorizonAligner & others
            self.smart_bus.set(
                "expert_weights",
                expert_weights,
                module=name,
                thesis="Expert performance/weight map",
            )

            # Strategy weights (meta surface)
            self.smart_bus.set(
                "strategy_weights",
                {
                    "by_member": expert_weights,
                    "members": committee_members,
                    "weights": [
                        float(expert_weights.get(m, 0.0))
                        for m in committee_members
                    ],
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                module=name,
                thesis="Strategy weights per expert",
            )

            # Member performance summary (simple: weight + confidence)
            member_perf = {}
            for expert in committee_members:
                member_perf[expert] = {
                    "contribution_score": float(expert_weights.get(expert, 0.5)),
                    "confidence": float(
                        member_confidences_map.get(expert, 0.5)
                    ),
                }
            self.smart_bus.set(
                "member_performance",
                {
                    "by_member": member_perf,
                    "timestamp": datetime.datetime.now().isoformat(),
                },
                module=name,
                thesis="Member performance summary",
            )

        except Exception as e:
            self.logger.error(f"[BUS] Bus update failed: {e}")

    def _record_decision(
        self,
        decision: Dict[str, Any],
        confidence: float,
        consensus: Dict[str, Any],
        expert_votes: List[Dict[str, Any]],
    ) -> None:
        """Record decision for in-module analytics and history."""
        try:
            record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "decision": decision,
                "confidence": confidence,
                "consensus": consensus,
                "expert_count": len(expert_votes),
            }
            self.voting_history.append(record)

            self.committee_analytics["total_decisions"] += 1
            if consensus.get("consensus_exists"):
                self.committee_analytics["consensus_decisions"] += 1

            # Update rolling average confidence
            n = self.committee_analytics["total_decisions"]
            old = float(
                self.committee_analytics.get("average_confidence", 0.5)
            )
            self.committee_analytics["average_confidence"] = (
                old * (n - 1) + confidence
            ) / n

        except Exception as e:
            self.logger.warning(f"[ANALYTICS] Decision recording failed: {e}")

    def _warmup_output(self, reason: str, decision_id: str) -> Dict[str, Any]:
        """Return contract-compliant warmup output (abstain during warmup)."""
        warmup_thesis = f"WARMUP: {reason}"
        return {
            # Primary outputs (new v5.0)
            VotingBusKeys.COMMITTEE_DECISION: {
                "action": "abstain",
                "reason": reason,
            },
            VotingBusKeys.COMMITTEE_CONSENSUS: {
                "consensus_exists": False,
                "warmup": True,
                "reason": reason,
            },
            VotingBusKeys.COMMITTEE_CONFIDENCE: 0.0,
            VotingBusKeys.COMMITTEE_VOTES: [],
            "committee_summary": {
                "warmup": True,
                "total_members": 0,
                "decision": "abstain",
                "reason": reason,
                "tick_count": self._tick_count,
                "warmup_ticks": self.warmup_ticks,
            },
            VotingBusKeys.COMMITTEE_DECISION_ID: decision_id,
            "raw_proposals": {},
            "member_confidences": {},
            "voting_weights": {},
            "committee_member_confidences": {},
            "strategy_weights": {},
            "member_performance": {},
            # Backward compatibility
            VotingBusKeys.VOTES: [],
            VotingBusKeys.VOTING_SUMMARY: {
                "action": "abstain",
                "warmup": True,
                "reason": reason,
            },
            VotingBusKeys.STRATEGY_ARBITER_WEIGHTS: {},
            "expert_votes": [],
            "expert_weights": {},
            "committee_analytics": dict(self.committee_analytics),
            "committee_members": [],
            "n_members": 0,
            "proposal_vectors": [],
            "decision_id": decision_id,
            "time_of_day": 0,
            # Trade vote outputs
            "trade_vote": {
                "action": "abstain",
                "confidence": 0.0,
                "warmup": True,
            },
            "trade_vote_v2": {
                "action": "abstain",
                "size": 0.0,
                "confidence": 0.0,
                "consensus_score": 0.0,
                "warmup": True,
                "decision_id": decision_id,
                "timestamp": datetime.datetime.now().isoformat(),
            },
            # Contract-required v5.0 extras
            "committee_proposal_vectors": [],
            "committee_decisions_by_instrument": {},
            "_thesis": warmup_thesis,
        }

    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        thesis = f"Committee error: {error}"
        return {
            # Primary outputs (new v5.0)
            VotingBusKeys.COMMITTEE_DECISION: {
                "action": "abstain",
                "reason": f"error: {error}",
            },
            VotingBusKeys.COMMITTEE_CONSENSUS: {
                "consensus_exists": False,
                "error": error,
            },
            VotingBusKeys.COMMITTEE_CONFIDENCE: 0.1,
            VotingBusKeys.COMMITTEE_VOTES: [],
            "committee_summary": {
                "error": error,
                "total_members": 0,
                "decision": "abstain",
            },
            VotingBusKeys.COMMITTEE_DECISION_ID: None,
            "raw_proposals": {},
            "member_confidences": {},
            "voting_weights": {},
            "committee_member_confidences": {},
            "strategy_weights": {},
            "member_performance": {},
            # Backward compatibility
            VotingBusKeys.VOTES: [],
            VotingBusKeys.VOTING_SUMMARY: {
                "action": "abstain",
                "error": error,
            },
            VotingBusKeys.STRATEGY_ARBITER_WEIGHTS: {},
            "expert_votes": [],
            "expert_weights": {},
            "committee_analytics": {"error": error},
            "committee_members": [],
            "n_members": 0,
            "proposal_vectors": [],
            "decision_id": None,
            "time_of_day": 0,
            # Trade vote outputs
            "trade_vote": {"action": "abstain", "confidence": 0.1},
            "trade_vote_v2": {
                "action": "abstain",
                "size": 0.0,
                "confidence": 0.1,
                "consensus_score": 0.0,
            },
            # Contract-required v5.0 extras
            "committee_proposal_vectors": [],
            "committee_decisions_by_instrument": {},
            "_thesis": thesis,
        }

    # ====================================================================== #
    # STATE PERSISTENCE - Save/Load module state
    # ====================================================================== #

    def _get_custom_state(self) -> Dict[str, Any]:
        """
        Get custom state for persistence.

        Saves:
        - voting_history (truncated)
        - committee_analytics
        - warmup state
        - tick counter
        - expert_weights (last snapshot)
        """
        return {
            "voting_history": list(self.voting_history)[-50:],
            "committee_analytics": dict(self.committee_analytics),
            "warmup_complete": bool(self.warmup_complete),
            "tick_count": int(self._tick_count),
            "expert_weights": dict(self.expert_weights),
        }

    def _set_custom_state(self, state: Dict[str, Any]) -> None:
        """Restore custom state from persistence."""
        if not state:
            return

        try:
            hist = state.get("voting_history", [])
            self.voting_history = deque(hist, maxlen=100)

            analytics = state.get("committee_analytics", {})
            self.committee_analytics.update(analytics)

            self.warmup_complete = bool(
                state.get("warmup_complete", self.warmup_complete)
            )
            self._tick_count = int(state.get("tick_count", self._tick_count))
            self.expert_weights = dict(
                state.get("expert_weights", self.expert_weights)
            )

            self.logger.info(
                f"📂 CommitteeCoordinator state restored | "
                f"decisions={self.committee_analytics.get('total_decisions', 0)} | "
                f"warmup_complete={self.warmup_complete}"
            )
        except Exception as e:
            self.logger.warning(f"[STATE] Failed to restore state: {e}")
