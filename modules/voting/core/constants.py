# modules/voting/core/constants.py
"""
Voting system constants and enumerations.
Single source of truth for thresholds, defaults, and configuration values.

MODE-AWARE: Thresholds automatically switch between LIVE (conservative)
and TRAINING (exploratory) modes. Call set_voting_mode() at startup.
"""

from enum import Enum
from typing import Dict, Any


class VotingAction(str, Enum):
    """Possible voting actions from experts and final decisions."""
    LONG = "long"
    SHORT = "short"
    HOLD = "hold"
    ABSTAIN = "abstain"
    
    @classmethod
    def from_string(cls, value: str) -> "VotingAction":
        """Parse string to VotingAction with fallback to ABSTAIN."""
        try:
            return cls(value.lower().strip())
        except (ValueError, AttributeError):
            return cls.ABSTAIN
    
    @property
    def is_directional(self) -> bool:
        """Returns True if this is a directional signal (long/short)."""
        return self in (VotingAction.LONG, VotingAction.SHORT)


class PipelineStage(str, Enum):
    """Pipeline stages in execution order."""
    IDLE = "idle"                # Not processing
    COMMITTEE = "committee"      # Collect votes from experts
    CONSENSUS = "consensus"      # Calculate consensus strength
    COLLUSION = "collusion"      # Detect suspicious voting patterns
    HORIZON = "horizon"          # Align time horizons
    UNCERTAINTY = "uncertainty"  # Sample alternative scenarios
    ARBITER = "arbiter"          # Make final decision


class VotingQuality(str, Enum):
    """Quality levels for voting decisions."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


# ═══════════════════════════════════════════════════════════════════
# MODE-AWARE THRESHOLD SYSTEM
# ═══════════════════════════════════════════════════════════════════

# Global mode flag
_VOTING_MODE: str = "TRAINING"  # "LIVE" or "TRAINING"

# ───────────────────────────────────────────────────────────────────
# LIVE MODE THRESHOLDS (Conservative - protect capital)
# ───────────────────────────────────────────────────────────────────
_LIVE_THRESHOLDS = {
    # Confidence (high requirements)
    "CONFIDENCE_THRESHOLD": 0.45,
    "HIGH_CONFIDENCE_THRESHOLD": 0.80,
    "MIN_SIGNAL_STRENGTH": 0.25,
    
    # Consensus (strong agreement required)
    "CONSENSUS_THRESHOLD": 0.70,
    "STRONG_CONSENSUS_THRESHOLD": 0.85,
    "WEAK_CONSENSUS_THRESHOLD": 0.50,
    
    # Arbiter (strict filtering)
    "ARBITER_CONFIDENCE_FLOOR": 0.40,
    "ARBITER_INTENSITY_FLOOR": 0.30,
}

# ───────────────────────────────────────────────────────────────────
# TRAINING MODE THRESHOLDS (Exploratory - allow learning)
# ───────────────────────────────────────────────────────────────────
_TRAINING_THRESHOLDS = {
    # Confidence (lower requirements for exploration)
    "CONFIDENCE_THRESHOLD": 0.30,
    "HIGH_CONFIDENCE_THRESHOLD": 0.70,
    "MIN_SIGNAL_STRENGTH": 0.15,
    
    # Consensus (easier agreement)
    "CONSENSUS_THRESHOLD": 0.55,
    "STRONG_CONSENSUS_THRESHOLD": 0.75,
    "WEAK_CONSENSUS_THRESHOLD": 0.35,
    
    # Arbiter (more permissive)
    "ARBITER_CONFIDENCE_FLOOR": 0.20,
    "ARBITER_INTENSITY_FLOOR": 0.15,
}


def set_voting_mode(mode: str) -> None:
    """
    Set the global voting mode. Call this at startup based on config.
    
    Args:
        mode: "LIVE" for conservative real-money trading,
              "TRAINING" for exploratory learning mode
    """
    global _VOTING_MODE
    mode = mode.upper().strip()
    if mode not in ("LIVE", "TRAINING"):
        mode = "TRAINING"  # Default to exploratory mode
    _VOTING_MODE = mode


def get_voting_mode() -> str:
    """Get the current voting mode."""
    return _VOTING_MODE


def get_thresholds() -> Dict[str, float]:
    """Get the current thresholds based on voting mode."""
    if _VOTING_MODE == "LIVE":
        return _LIVE_THRESHOLDS.copy()
    return _TRAINING_THRESHOLDS.copy()


# ═══════════════════════════════════════════════════════════════════
# Mode-Aware Threshold Accessors
# (These return the appropriate value based on current mode)
# ═══════════════════════════════════════════════════════════════════

# Backward-compatible constants (now functions that check mode)
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
# Static Constants (for backward compatibility - use mode-aware defaults)
# These are evaluated once at import time with TRAINING mode defaults
# For dynamic mode-aware values, use the _F() functions above
# ═══════════════════════════════════════════════════════════════════

# Confidence thresholds (defaults to TRAINING mode values for backward compat)
CONFIDENCE_THRESHOLD = 0.30
HIGH_CONFIDENCE_THRESHOLD = 0.70
MIN_SIGNAL_STRENGTH = 0.15

# Consensus thresholds
CONSENSUS_THRESHOLD = 0.55
STRONG_CONSENSUS_THRESHOLD = 0.75
WEAK_CONSENSUS_THRESHOLD = 0.35

# Arbiter thresholds
ARBITER_CONFIDENCE_FLOOR = 0.20
ARBITER_INTENSITY_FLOOR = 0.15

# Collusion detection (static - not mode-aware)
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

# Performance
MAX_PROCESSING_TIME_MS = 100.0
CIRCUIT_BREAKER_THRESHOLD = 5


# ═══════════════════════════════════════════════════════════════════
# Grouped Threshold Dicts (for structured access)
# ═══════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════
# Default Values Dict (for bulk access)
# ═══════════════════════════════════════════════════════════════════

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
    Single source of truth - use these constants instead of hardcoded strings.
    """
    
    # ─────────────────────────────────────────────────────────────────
    # Decision Coordination
    # ─────────────────────────────────────────────────────────────────
    DECISION_ID = "decision_id"
    KERNEL_DECISION_ID = "kernel_decision_id"
    TICK_TS = "tick_ts"
    KERNEL_TICK_TS = "kernel_tick_ts"
    
    # ─────────────────────────────────────────────────────────────────
    # Market Data Inputs
    # ─────────────────────────────────────────────────────────────────
    ACTIVE_INSTRUMENTS = "watched_instruments"
    MARKET_REGIME = "market_regime"
    VOLATILITY_DATA = "volatility_data"
    SESSION_TYPE = "session_type"
    
    # ─────────────────────────────────────────────────────────────────
    # Committee Stage Outputs
    # ─────────────────────────────────────────────────────────────────
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
    VOTE_BUNDLE = "vote_bundle"  # Used by SlimVotingKernel
    
    # ─────────────────────────────────────────────────────────────────
    # Consensus Stage Outputs
    # ─────────────────────────────────────────────────────────────────
    CONSENSUS_RESULT = "consensus_result"
    CONSENSUS_SCORE = "consensus_score"
    AGREEMENT_SCORE = "agreement_score"
    CONSENSUS_DIRECTION = "consensus_direction"
    CONSENSUS_CONFIDENCE = "consensus_confidence"
    CONSENSUS_QUALITY = "consensus_quality"
    
    # ─────────────────────────────────────────────────────────────────
    # Collusion Stage Outputs
    # ─────────────────────────────────────────────────────────────────
    COLLUSION_RESULT = "collusion_result"
    COLLUSION_SCORE = "collusion_score"
    COLLUSION_DETECTED = "collusion_detected"
    SUSPICIOUS_PAIRS = "suspicious_pairs"
    
    # ─────────────────────────────────────────────────────────────────
    # Horizon Stage Outputs
    # ─────────────────────────────────────────────────────────────────
    HORIZON_RESULT = "horizon_result"
    HORIZON_ALIGNMENT = "horizon_alignment"
    ALIGNED_WEIGHTS = "aligned_weights"
    
    # ─────────────────────────────────────────────────────────────────
    # Uncertainty Stage Outputs
    # ─────────────────────────────────────────────────────────────────
    UNCERTAINTY_RESULT = "uncertainty_result"
    FRAGILITY = "fragility"
    FRAGILITY_SCORE = "fragility_score"
    
    # ─────────────────────────────────────────────────────────────────
    # Arbiter/Final Decision Outputs
    # ─────────────────────────────────────────────────────────────────
    FINAL_DECISION = "final_decision"
    GATE_DECISION = "gate_decision"
    TRADE_VOTE = "trade_vote"
    TRADE_VOTE_V2 = "trade_vote_v2"
    ARBITER_DECISION = "arbiter_decision"
    
    # ─────────────────────────────────────────────────────────────────
    # Pipeline/Kernel Outputs
    # ─────────────────────────────────────────────────────────────────
    PIPELINE_RESULT = "pipeline_result"
    KERNEL_DECISION = "kernel_decision"
    DECISION_BUNDLE = "decision_bundle"
    VOTING_CONSENSUS = "voting_consensus"
    CONSENSUS_SUMMARY = "consensus_summary"
    VOTING_METRICS = "voting_metrics"
    PIPELINE_STATUS = "pipeline_status"
    
    # ─────────────────────────────────────────────────────────────────
    # Memory Integration
    # ─────────────────────────────────────────────────────────────────
    MEMORY_GATE = "memory_gate"
    DANGER_ZONES = "danger_zones"
    
    # ─────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def expert_proposal(expert_name: str) -> str:
        """Get the voting proposal key for an expert."""
        return f"{expert_name}_voting_proposal"
    
    @staticmethod
    def expert_confidence(expert_name: str) -> str:
        """Get the confidence key for an expert."""
        return f"{expert_name}_confidence"


# ═══════════════════════════════════════════════════════════════════
# Known Voting Members (for discovery)
# ═══════════════════════════════════════════════════════════════════

KNOWN_VOTING_MEMBERS = [
    "ThemeExpert",
    "SeasonalityRiskExpert",
    "DynamicRiskController",
    "PPOAgent",
    "PortfolioRiskSystem",
    "EnhancedAnomalyDetector",
    "ExecutionQualityMonitor",
    "MetaAgent",
]

# Expert name to bus key prefix mapping
EXPERT_KEY_PREFIXES = {
    "ThemeExpert": "ThemeExpert",
    "SeasonalityRiskExpert": "SeasonalityRiskExpert",
    "DynamicRiskController": "DynamicRiskController",
    "PPOAgent": "PPOAgent",
    "PortfolioRiskSystem": "PortfolioRiskSystem",
    "EnhancedAnomalyDetector": "EnhancedAnomalyDetector",
    "ExecutionQualityMonitor": "ExecutionQualityMonitor",
    "MetaAgent": "MetaAgent",
}
