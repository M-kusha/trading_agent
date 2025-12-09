"""
Voting system constants and enumerations.
Single source of truth for thresholds, defaults, and configuration values.

MODE-AWARE:
- Thresholds automatically switch between LIVE (conservative) and TRAINING (exploratory)
- Use get_thresholds() / *_F() helpers instead of hard-coded constants
- Per-instrument thresholds go through DynamicThresholdManager in LIVE mode
"""

from enum import Enum
from typing import Dict, Any, Optional


class VotingAction(str, Enum):
    """Possible voting actions from experts and final decisions."""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    ABSTAIN = "abstain"

    @classmethod
    def from_string(cls, value: Any) -> "VotingAction":
        """
        Parse arbitrary input into a VotingAction.

        Accepts:
        - VotingAction enums (idempotent)
        - Strings with legacy aliases:
          - 'buy', 'bull', 'bullish'    -> LONG
          - 'sell', 'bear', 'bearish'   -> SHORT
          - 'flat', 'neutral', 'none',
            'no_trade', 'no-trade'      -> HOLD
          - 'abstain', 'skip', 'ignore' -> ABSTAIN

        Unknown values fall back to ABSTAIN.
        """
        if isinstance(value, cls):
            return value

        if value is None:
            return cls.ABSTAIN

        try:
            s = str(value).lower().strip()
        except Exception:
            return cls.ABSTAIN

        # Explicit mappings first (legacy compatibility)
        if s in {"long", "buy", "bull", "bullish"}:
            return cls.LONG
        if s in {"short", "sell", "bear", "bearish"}:
            return cls.SHORT
        if s in {"hold", "flat", "neutral", "none", "no_trade", "no-trade"}:
            return cls.HOLD
        if s in {"abstain", "skip", "ignore"}:
            return cls.ABSTAIN

        # Fallback to enum value if exact match
        try:
            return cls(s)
        except ValueError:
            return cls.ABSTAIN

    @property
    def is_directional(self) -> bool:
        """Returns True if this is a directional signal (long/short)."""
        return self in (VotingAction.LONG, VotingAction.SHORT)


class PipelineStage(str, Enum):
    """Voting pipeline stages in execution order."""
    IDLE = "idle"
    COMMITTEE = "committee"
    CONSENSUS = "consensus"
    COLLUSION = "collusion"
    HORIZON = "horizon"
    UNCERTAINTY = "uncertainty"
    ARBITER = "arbiter"


class VotingQuality(str, Enum):
    """Quality levels for voting decisions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


# ═══════════════════════════════════════════════════════════════════
# TIMEFRAME CONFIGURATION – SINGLE SOURCE OF TRUTH
# ═══════════════════════════════════════════════════════════════════

PRIMARY_TIMEFRAME: str = "M15"
CONTEXT_TIMEFRAMES: tuple = ("H1", "H4", "D1")
SUPPORTED_TIMEFRAMES: tuple = ("M15", "H1", "H4", "D1")

# MTF adjustments: M15 drives direction; higher TFs modulate confidence only.
MTF_AGREEMENT_BONUS: float = 0.15
MTF_DISAGREEMENT_PENALTY: float = 0.20
MTF_NEUTRAL_ADJUSTMENT: float = 0.0


def get_primary_timeframe() -> str:
    """Get the canonical primary trading timeframe (M15)."""
    return PRIMARY_TIMEFRAME


def is_primary_timeframe(tf: str) -> bool:
    """Check if a timeframe is the primary trading timeframe."""
    return tf.upper() == PRIMARY_TIMEFRAME


def is_context_timeframe(tf: str) -> bool:
    """Check if a timeframe is a context/confirmation timeframe."""
    return tf.upper() in CONTEXT_TIMEFRAMES


# ═══════════════════════════════════════════════════════════════════
# MODE-AWARE THRESHOLD SYSTEM
# ═══════════════════════════════════════════════════════════════════

_VOTING_MODE: str = "TRAINING"  # "LIVE" or "TRAINING"

# LIVE MODE – conservative, real-money trading
_LIVE_THRESHOLDS: Dict[str, float] = {
    # Confidence
    "CONFIDENCE_THRESHOLD": 0.45,
    "HIGH_CONFIDENCE_THRESHOLD": 0.70,
    "MIN_SIGNAL_STRENGTH": 0.40,

    # Consensus
    "CONSENSUS_THRESHOLD": 0.55,
    "STRONG_CONSENSUS_THRESHOLD": 0.70,
    "WEAK_CONSENSUS_THRESHOLD": 0.40,

    # Arbiter
    "ARBITER_CONFIDENCE_FLOOR": 0.50,
    "ARBITER_INTENSITY_FLOOR": 0.45,
}

# TRAINING MODE – exploratory, allow more variety for learning
_TRAINING_THRESHOLDS: Dict[str, float] = {
    "CONFIDENCE_THRESHOLD": 0.30,
    "HIGH_CONFIDENCE_THRESHOLD": 0.70,
    "MIN_SIGNAL_STRENGTH": 0.15,
    "CONSENSUS_THRESHOLD": 0.55,
    "STRONG_CONSENSUS_THRESHOLD": 0.75,
    "WEAK_CONSENSUS_THRESHOLD": 0.35,
    "ARBITER_CONFIDENCE_FLOOR": 0.20,
    "ARBITER_INTENSITY_FLOOR": 0.15,
}


def set_voting_mode(mode: str) -> None:
    """
    Set the global voting mode. Call this at startup based on config.

    Args:
        mode: "LIVE" for conservative real-money trading,
              "TRAINING" for exploratory learning mode.
    """
    global _VOTING_MODE
    old_mode = _VOTING_MODE
    mode = (mode or "").upper().strip()
    if mode not in ("LIVE", "TRAINING"):
        mode = "TRAINING"
    _VOTING_MODE = mode

    import logging
    logging.getLogger("voting.constants").debug(
        f"[VOTING MODE] Changed from {old_mode} → {_VOTING_MODE}"
    )


def get_voting_mode() -> str:
    """Get the current voting mode."""
    return _VOTING_MODE


def is_live_mode() -> bool:
    """True when voting mode is LIVE."""
    return _VOTING_MODE == "LIVE"


def is_training_mode() -> bool:
    """True when voting mode is TRAINING."""
    return _VOTING_MODE == "TRAINING"


def get_thresholds() -> Dict[str, float]:
    """
    Get the current global thresholds based on voting mode.

    For per-instrument thresholds, use:
      - get_adaptive_thresholds_for_instrument()
      - get_instrument_threshold()
    """
    if _VOTING_MODE == "LIVE":
        return _LIVE_THRESHOLDS.copy()
    return _TRAINING_THRESHOLDS.copy()


def get_adaptive_thresholds_for_instrument(instrument: str) -> Dict[str, Any]:
    """
    Get adaptive thresholds for a specific instrument.

    LIVE mode:
        Uses DynamicThresholdManager for intelligent per-instrument thresholds.
    TRAINING mode:
        Returns standard training thresholds.

    Returns:
        Dict with threshold values plus diagnostic metadata.
    """
    if _VOTING_MODE != "LIVE":
        # Training mode – static thresholds only
        return _TRAINING_THRESHOLDS.copy()

    try:
        from modules.voting.core.dynamic_thresholds import get_adaptive_thresholds

        adaptive = get_adaptive_thresholds(instrument)

        # Map to expected keys used across the system
        conf_th = adaptive.get("confidence_threshold", 0.65)
        int_th = adaptive.get("intensity_threshold", 0.50)

        arbiter_conf = max(0.20, conf_th - 0.05)  # never below 20%
        arbiter_int = max(0.10, int_th)

        return {
            "CONFIDENCE_THRESHOLD": conf_th,
            "HIGH_CONFIDENCE_THRESHOLD": adaptive.get(
                "high_confidence_threshold", 0.80
            ),
            "MIN_SIGNAL_STRENGTH": int_th,
            "CONSENSUS_THRESHOLD": adaptive.get("consensus_threshold", 0.60),
            "STRONG_CONSENSUS_THRESHOLD": adaptive.get(
                "strong_consensus_threshold", 0.75
            ),
            "WEAK_CONSENSUS_THRESHOLD": 0.35,
            "ARBITER_CONFIDENCE_FLOOR": arbiter_conf,
            "ARBITER_INTENSITY_FLOOR": arbiter_int,
            "_adaptive": True,
            "_adjustments": adaptive.get("_adjustments", {}),
            "_profile": adaptive.get("_profile", {}),
        }
    except Exception as e:
        import logging

        logging.getLogger("voting.constants").warning(
            f"Adaptive thresholds failed for {instrument}, using static: {e}"
        )
        return _LIVE_THRESHOLDS.copy()


# ═══════════════════════════════════════════════════════════════════
# Global Threshold Manager – convenience wrapper
# ═══════════════════════════════════════════════════════════════════

_threshold_manager = None  # DynamicThresholdManager singleton


def _get_threshold_manager():
    """Lazy-load the DynamicThresholdManager singleton."""
    global _threshold_manager
    if _threshold_manager is None:
        try:
            from modules.voting.core.dynamic_thresholds import DynamicThresholdManager

            _threshold_manager = DynamicThresholdManager.get_instance()
        except Exception as e:
            import logging

            logging.getLogger("voting.constants").warning(
                f"Failed to initialize DynamicThresholdManager: {e}"
            )
    return _threshold_manager


def get_instrument_threshold(
    instrument: str,
    threshold_type: str,
    context: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Get an adaptive threshold for a specific instrument.

    Args:
        instrument: e.g., "EURUSD", "XAUUSD"
        threshold_type: "confidence" or "consensus"
        context: reserved for future use (custom per-call context)

    Returns:
        Adaptive threshold value for this instrument.
    """
    # Training mode – static values only
    if _VOTING_MODE != "LIVE":
        if threshold_type == "confidence":
            return _TRAINING_THRESHOLDS["CONFIDENCE_THRESHOLD"]
        if threshold_type == "consensus":
            return _TRAINING_THRESHOLDS["CONSENSUS_THRESHOLD"]
        return 0.5

    manager = _get_threshold_manager()
    if manager is None:
        # Hard fallback to static LIVE thresholds
        if threshold_type == "confidence":
            return _LIVE_THRESHOLDS["CONFIDENCE_THRESHOLD"]
        if threshold_type == "consensus":
            return _LIVE_THRESHOLDS["CONSENSUS_THRESHOLD"]
        return 0.65

    try:
        thresholds = manager.get_thresholds(instrument)
        if threshold_type == "confidence":
            return thresholds.get("confidence_threshold", 0.68)
        if threshold_type == "consensus":
            return thresholds.get("consensus_threshold", 0.65)
        return 0.65
    except Exception as e:
        import logging

        logging.getLogger("voting.constants").warning(
            f"get_instrument_threshold failed for {instrument}: {e}"
        )
        if threshold_type == "confidence":
            return _LIVE_THRESHOLDS["CONFIDENCE_THRESHOLD"]
        if threshold_type == "consensus":
            return _LIVE_THRESHOLDS["CONSENSUS_THRESHOLD"]
        return 0.65


# ═══════════════════════════════════════════════════════════════════
# Mode-Aware Threshold Accessors (backward compatible)
# ═══════════════════════════════════════════════════════════════════

def CONFIDENCE_THRESHOLD_F() -> float:
    return get_thresholds()["CONFIDENCE_THRESHOLD"]


def HIGH_CONFIDENCE_THRESHOLD_F() -> float:
    return get_thresholds()["HIGH_CONFIDENCE_THRESHOLD"]


def MIN_SIGNAL_STRENGTH_F() -> float:
    return get_thresholds()["MIN_SIGNAL_STRENGTH"]


def CONSENSUS_THRESHOLD_F() -> float:
    return get_thresholds()["CONSENSUS_THRESHOLD"]


def STRONG_CONSENSUS_THRESHOLD_F() -> float:
    return get_thresholds()["STRONG_CONSENSUS_THRESHOLD"]


def WEAK_CONSENSUS_THRESHOLD_F() -> float:
    return get_thresholds()["WEAK_CONSENSUS_THRESHOLD"]


def ARBITER_CONFIDENCE_FLOOR_F() -> float:
    return get_thresholds()["ARBITER_CONFIDENCE_FLOOR"]


def ARBITER_INTENSITY_FLOOR_F() -> float:
    return get_thresholds()["ARBITER_INTENSITY_FLOOR"]


# ═══════════════════════════════════════════════════════════════════
# Static Constants (import-time defaults – mostly for legacy code)
# ═══════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.30
HIGH_CONFIDENCE_THRESHOLD = 0.70
MIN_SIGNAL_STRENGTH = 0.15

CONSENSUS_THRESHOLD = 0.55
STRONG_CONSENSUS_THRESHOLD = 0.75
WEAK_CONSENSUS_THRESHOLD = 0.35

ARBITER_CONFIDENCE_FLOOR = 0.20
ARBITER_INTENSITY_FLOOR = 0.15

# Collusion detection
COLLUSION_THRESHOLD = 0.85
MAX_CORRELATION = 0.90
MIN_EXPERT_DIVERSITY = 0.30

# Timing
MAX_STALENESS_SECONDS = 15.0
DECISION_TIMEOUT_MS = 5000
CACHE_TTL_SECONDS = 5.0

# Committee
MIN_VOTERS_REQUIRED = 2
MAX_VOTERS = 10
QUORUM_RATIO = 0.5

# Uncertainty sampling
FRAGILITY_THRESHOLD = 0.70
MAX_UNCERTAINTY = 0.50
MONTE_CARLO_SAMPLES = 100

# Performance / safety
MAX_PROCESSING_TIME_MS = 100.0
CIRCUIT_BREAKER_THRESHOLD = 5


# Grouped thresholds
CONSENSUS_THRESHOLDS: Dict[str, float] = {
    "threshold": CONSENSUS_THRESHOLD,
    "strong": STRONG_CONSENSUS_THRESHOLD,
    "weak": WEAK_CONSENSUS_THRESHOLD,
}

COLLUSION_THRESHOLDS: Dict[str, float] = {
    "threshold": COLLUSION_THRESHOLD,
    "max_correlation": MAX_CORRELATION,
    "min_diversity": MIN_EXPERT_DIVERSITY,
}

HORIZON_WEIGHTS: Dict[str, float] = {
    "short": 0.4,
    "medium": 0.35,
    "long": 0.25,
}

# Bulk defaults (used to seed VotingModuleBase._voting_defaults)
VOTING_DEFAULTS: Dict[str, Any] = {
    # Confidence thresholds
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "high_confidence_threshold": HIGH_CONFIDENCE_THRESHOLD,
    "min_signal_strength": MIN_SIGNAL_STRENGTH,

    # Consensus thresholds
    "consensus_threshold": CONSENSUS_THRESHOLD,
    "strong_consensus_threshold": STRONG_CONSENSUS_THRESHOLD,
    "weak_consensus_threshold": WEAK_CONSENSUS_THRESHOLD,

    # Collusion detection
    "collusion_threshold": COLLUSION_THRESHOLD,
    "max_correlation": MAX_CORRELATION,
    "min_expert_diversity": MIN_EXPERT_DIVERSITY,

    # Timing
    "max_staleness_seconds": MAX_STALENESS_SECONDS,
    "decision_timeout_ms": DECISION_TIMEOUT_MS,
    "cache_ttl_seconds": CACHE_TTL_SECONDS,

    # Committee
    "min_voters_required": MIN_VOTERS_REQUIRED,
    "max_voters": MAX_VOTERS,
    "quorum_ratio": QUORUM_RATIO,

    # Uncertainty sampling
    "fragility_threshold": FRAGILITY_THRESHOLD,
    "max_uncertainty": MAX_UNCERTAINTY,
    "monte_carlo_samples": MONTE_CARLO_SAMPLES,

    # Time horizon alignment
    "horizon_weights": {
        "short": 0.4,
        "medium": 0.35,
        "long": 0.25,
    },

    # Arbiter thresholds
    "arbiter_confidence_floor": ARBITER_CONFIDENCE_FLOOR,
    "arbiter_intensity_floor": ARBITER_INTENSITY_FLOOR,

    # Performance
    "max_processing_time_ms": MAX_PROCESSING_TIME_MS,
    "circuit_breaker_threshold": CIRCUIT_BREAKER_THRESHOLD,
}


# ═══════════════════════════════════════════════════════════════════
# Bus Key Names (Single Source of Truth)
# ═══════════════════════════════════════════════════════════════════

class VotingBusKeys:
    """
    Standard bus key names for voting system.
    Use these constants instead of hardcoded strings.
    """

    # Decision Coordination
    DECISION_ID = "decision_id"
    KERNEL_DECISION_ID = "kernel_decision_id"
    TICK_TS = "tick_ts"
    KERNEL_TICK_TS = "kernel_tick_ts"

    # Market Data Inputs
    ACTIVE_INSTRUMENTS = "watched_instruments"
    MARKET_REGIME = "market_regime"
    VOLATILITY_DATA = "volatility_data"
    SESSION_TYPE = "session_type"

    # Committee Stage Outputs
    COMMITTEE_VOTES = "committee_votes"
    COMMITTEE_SUMMARY = "committee_summary"
    COMMITTEE_DECISION = "committee_decision"
    COMMITTEE_CONSENSUS = "committee_consensus"
    COMMITTEE_CONFIDENCE = "committee_confidence"
    COMMITTEE_DECISION_ID = "committee_decision_id"
    RAW_PROPOSALS = "raw_proposals"
    MEMBER_CONFIDENCES = "member_confidences"
    VOTING_WEIGHTS = "voting_weights"
    VOTES = "votes"
    VOTING_SUMMARY = "voting_summary"
    STRATEGY_ARBITER_WEIGHTS = "strategy_arbiter_weights"
    VOTE_BUNDLE = "vote_bundle"

    # Consensus Stage Outputs
    CONSENSUS_RESULT = "consensus_result"
    CONSENSUS_SCORE = "consensus_score"
    AGREEMENT_SCORE = "agreement_score"
    CONSENSUS_DIRECTION = "consensus_direction"
    CONSENSUS_CONFIDENCE = "consensus_confidence"
    CONSENSUS_QUALITY = "consensus_quality"

    # Collusion Stage Outputs
    COLLUSION_RESULT = "collusion_result"
    COLLUSION_SCORE = "collusion_score"
    COLLUSION_DETECTED = "collusion_detected"
    SUSPICIOUS_PAIRS = "suspicious_pairs"

    # Horizon Stage Outputs
    HORIZON_RESULT = "horizon_result"
    HORIZON_ALIGNMENT = "horizon_alignment"
    ALIGNED_WEIGHTS = "aligned_weights"

    # Uncertainty Stage Outputs
    UNCERTAINTY_RESULT = "uncertainty_result"
    FRAGILITY = "fragility"
    FRAGILITY_SCORE = "fragility_score"

    # Arbiter / Final Decision Outputs
    FINAL_DECISION = "final_decision"
    GATE_DECISION = "gate_decision"
    TRADE_VOTE = "trade_vote"
    TRADE_VOTE_V2 = "trade_vote_v2"
    ARBITER_DECISION = "arbiter_decision"

    # Pipeline / Kernel Outputs
    PIPELINE_RESULT = "pipeline_result"
    KERNEL_DECISION = "kernel_decision"
    DECISION_BUNDLE = "decision_bundle"
    VOTING_CONSENSUS = "voting_consensus"
    CONSENSUS_SUMMARY = "consensus_summary"
    VOTING_METRICS = "voting_metrics"
    PIPELINE_STATUS = "pipeline_status"

    # Memory Integration
    MEMORY_GATE = "memory_gate"
    DANGER_ZONES = "danger_zones"

    @staticmethod
    def expert_proposal(expert_name: str) -> str:
        """Get the voting proposal key for an expert."""
        return f"{expert_name}_voting_proposal"

    @staticmethod
    def expert_confidence(expert_name: str) -> str:
        """Get the confidence key for an expert."""
        return f"{expert_name}_confidence"


# Known directional voting members (for discovery / diagnostics)
KNOWN_VOTING_MEMBERS = [
    "TrendExpert",
    "MomentumExpert",
    "ThemeExpert",
    "SeasonalityRiskExpert",
    "PPOAgent",
]

EXPERT_KEY_PREFIXES = {
    "TrendExpert": "TrendExpert",
    "MomentumExpert": "MomentumExpert",
    "ThemeExpert": "ThemeExpert",
    "SeasonalityRiskExpert": "SeasonalityRiskExpert",
    "PPOAgent": "PPOAgent",
}
