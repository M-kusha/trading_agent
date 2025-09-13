"""
Unified Voting System - Core Type Definitions
Provides standardized data structures for the entire voting system
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import numpy as np
import datetime


class VotingAction(Enum):
    """Standardized voting actions across all experts"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    ABSTAIN = "abstain"
    LONG_RISK_ASSETS = "long_risk_assets"
    SAFE_HAVEN_ROTATION = "safe_haven_rotation"
    VOLATILITY_HEDGING = "volatility_hedging"
    TREND_FOLLOWING = "trend_following"
    SEASONAL_LONG_BIAS = "seasonal_long_bias"
    SEASONAL_SHORT_BIAS = "seasonal_short_bias"
    SEASONAL_NEUTRAL = "seasonal_neutral"
    EMERGENCY_EXIT = "emergency_exit"
    REDUCE_POSITIONS = "reduce_positions"
    CONSERVATIVE_SIZING = "conservative_sizing"
    CAUTIOUS_MONITORING = "cautious_monitoring"


class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING = "trending"
    VOLATILE = "volatile"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    REVERSAL = "reversal"
    NOISE = "noise"
    UNKNOWN = "unknown"


class TradingSession(Enum):
    """Trading session classifications"""
    AMERICAN = "american"
    EUROPEAN = "european"
    ASIAN = "asian"
    OVERLAP = "overlap"
    ROLLOVER = "rollover"
    WEEKEND = "weekend"
    UNKNOWN = "unknown"


class VolatilityLevel(Enum):
    """Volatility level classifications"""
    EXTREME = "extreme"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    VERY_LOW = "very_low"


class AlertSeverity(Enum):
    """Alert severity levels for monitoring"""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class CircuitBreakerState(Enum):
    """Circuit breaker states for error handling"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class MarketContext:
    """Comprehensive market context information"""
    regime: MarketRegime = MarketRegime.UNKNOWN
    session: TradingSession = TradingSession.UNKNOWN
    volatility_level: VolatilityLevel = VolatilityLevel.MEDIUM
    volatility_value: float = 0.02
    risk_score: float = 0.0
    emergency_mode: bool = False
    market_open: bool = True
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'regime': self.regime.value,
            'session': self.session.value,
            'volatility_level': self.volatility_level.value,
            'volatility_value': self.volatility_value,
            'risk_score': self.risk_score,
            'emergency_mode': self.emergency_mode,
            'market_open': self.market_open,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MarketContext:
        """Create from dictionary"""
        return cls(
            regime=MarketRegime(data.get('regime', 'unknown')),
            session=TradingSession(data.get('session', 'unknown')),
            volatility_level=VolatilityLevel(data.get('volatility_level', 'medium')),
            volatility_value=float(data.get('volatility_value', 0.02)),
            risk_score=float(data.get('risk_score', 0.0)),
            emergency_mode=bool(data.get('emergency_mode', False)),
            market_open=bool(data.get('market_open', True)),
            timestamp=data.get('timestamp', datetime.datetime.now().isoformat())
        )


@dataclass
class VotingProposal:
    """Standardized voting proposal structure"""
    action: VotingAction
    signal_strength: float
    position_size: float
    duration: str  # 'short', 'medium', 'long'
    confidence: float
    expert_name: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    emergency_adjustment: Optional[Dict[str, Any]] = None
    market_adjustments: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate and normalize values"""
        self.signal_strength = max(-1.0, min(1.0, self.signal_strength))
        self.position_size = max(0.0, min(1.0, self.position_size))
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'action': self.action.value if isinstance(self.action, VotingAction) else self.action,
            'signal_strength': self.signal_strength,
            'position_size': self.position_size,
            'duration': self.duration,
            'confidence': self.confidence,
            'expert_name': self.expert_name,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'emergency_adjustment': self.emergency_adjustment,
            'market_adjustments': self.market_adjustments
        }


@dataclass
class ConsensusResult:
    """Consensus analysis result"""
    consensus_exists: bool
    consensus_score: float
    dominant_action: VotingAction
    action_distribution: Dict[str, float]
    conflict_level: str  # 'NONE', 'LOW', 'MEDIUM', 'HIGH', 'SEVERE'
    vote_count: int
    total_weight: float
    directional_consensus: float = 0.5
    magnitude_consensus: float = 0.5
    confidence_consensus: float = 0.5
    network_consensus: float = 0.5
    temporal_stability: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'consensus_exists': self.consensus_exists,
            'consensus_score': self.consensus_score,
            'dominant_action': self.dominant_action.value if isinstance(self.dominant_action, VotingAction) else self.dominant_action,
            'action_distribution': self.action_distribution,
            'conflict_level': self.conflict_level,
            'vote_count': self.vote_count,
            'total_weight': self.total_weight,
            'directional_consensus': self.directional_consensus,
            'magnitude_consensus': self.magnitude_consensus,
            'confidence_consensus': self.confidence_consensus,
            'network_consensus': self.network_consensus,
            'temporal_stability': self.temporal_stability
        }


@dataclass
class CollusionEvent:
    """Collusion detection event"""
    timestamp: str
    suspicious_pairs: List[Tuple[int, int]]
    collusion_score: float
    coordination_strength: Dict[Tuple[int, int], float]
    alert_severity: AlertSeverity
    recommended_action: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'timestamp': self.timestamp,
            'suspicious_pairs': [list(p) for p in self.suspicious_pairs],
            'collusion_score': self.collusion_score,
            'coordination_strength': {str(k): v for k, v in self.coordination_strength.items()},
            'alert_severity': self.alert_severity.value,
            'recommended_action': self.recommended_action
        }


@dataclass
class SamplingResult:
    """Alternative reality sampling result"""
    samples: np.ndarray
    uncertainty: float
    diversity_score: float
    effective_samples: int
    confidence_bounds: Dict[str, Any]
    sampling_method: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'samples': self.samples.tolist() if isinstance(self.samples, np.ndarray) else self.samples,
            'uncertainty': self.uncertainty,
            'diversity_score': self.diversity_score,
            'effective_samples': self.effective_samples,
            'confidence_bounds': self.confidence_bounds,
            'sampling_method': self.sampling_method
        }


@dataclass
class HorizonAlignment:
    """Time horizon alignment result"""
    aligned_weights: np.ndarray
    horizon_distances: np.ndarray
    horizon_multipliers: np.ndarray
    regime_adjustments: np.ndarray
    session_patterns: np.ndarray
    alignment_quality: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'aligned_weights': self.aligned_weights.tolist() if isinstance(self.aligned_weights, np.ndarray) else self.aligned_weights,
            'horizon_distances': self.horizon_distances.tolist() if isinstance(self.horizon_distances, np.ndarray) else self.horizon_distances,
            'horizon_multipliers': self.horizon_multipliers.tolist() if isinstance(self.horizon_multipliers, np.ndarray) else self.horizon_multipliers,
            'regime_adjustments': self.regime_adjustments.tolist() if isinstance(self.regime_adjustments, np.ndarray) else self.regime_adjustments,
            'session_patterns': self.session_patterns.tolist() if isinstance(self.session_patterns, np.ndarray) else self.session_patterns,
            'alignment_quality': self.alignment_quality
        }


@dataclass
class CommitteeDecision:
    """Final committee decision"""
    action: VotingAction
    confidence: float
    consensus_strength: float
    decision_type: str  # 'consensus', 'plurality', 'emergency', 'abstain'
    expert_weights: Dict[str, float]
    expert_votes: List[VotingProposal]
    thesis: str
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'action': self.action.value if isinstance(self.action, VotingAction) else self.action,
            'confidence': self.confidence,
            'consensus_strength': self.consensus_strength,
            'decision_type': self.decision_type,
            'expert_weights': self.expert_weights,
            'expert_votes': [v.to_dict() for v in self.expert_votes],
            'thesis': self.thesis,
            'timestamp': self.timestamp
        }


@dataclass
class QualityMetrics:
    """System quality metrics"""
    coherence: float = 0.5
    stability: float = 0.5
    diversity: float = 0.5
    reliability: float = 0.5
    predictive_accuracy: float = 0.5
    temporal_consistency: float = 0.5
    overall_effectiveness: float = 0.5
    
    def update_overall(self):
        """Update overall effectiveness based on components"""
        weights = np.array([0.25, 0.20, 0.15, 0.20, 0.10, 0.10])
        values = np.array([
            self.coherence,
            self.stability,
            self.diversity,
            self.reliability,
            self.predictive_accuracy,
            self.temporal_consistency
        ])
        self.overall_effectiveness = float(np.dot(weights, values))
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'coherence': self.coherence,
            'stability': self.stability,
            'diversity': self.diversity,
            'reliability': self.reliability,
            'predictive_accuracy': self.predictive_accuracy,
            'temporal_consistency': self.temporal_consistency,
            'overall_effectiveness': self.overall_effectiveness
        }


@dataclass
class IntelligenceParameters:
    """Shared intelligence parameters for adaptation"""
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.15
    market_sensitivity: float = 0.8
    performance_memory: float = 0.9
    confidence_momentum: float = 0.85
    emergency_response_factor: float = 0.3
    regime_sensitivity: float = 0.8
    session_memory: float = 0.85
    volatility_adaptation: float = 0.7
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            'learning_rate': self.learning_rate,
            'adaptation_threshold': self.adaptation_threshold,
            'market_sensitivity': self.market_sensitivity,
            'performance_memory': self.performance_memory,
            'confidence_momentum': self.confidence_momentum,
            'emergency_response_factor': self.emergency_response_factor,
            'regime_sensitivity': self.regime_sensitivity,
            'session_memory': self.session_memory,
            'volatility_adaptation': self.volatility_adaptation
        }


@dataclass
class DebugSnapshot:
    """Debug snapshot for tracing"""
    stage: str
    timestamp: str
    inputs: Dict[str, Any]
    calculations: Dict[str, Any]
    outputs: Dict[str, Any]
    rationale: str
    performance_ms: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            'stage': self.stage,
            'timestamp': self.timestamp,
            'inputs': self.inputs,
            'calculations': self.calculations,
            'outputs': self.outputs,
            'rationale': self.rationale,
            'performance_ms': self.performance_ms
        }


# Type aliases for clarity
WeightVector = np.ndarray
ProposalVector = np.ndarray
ConfidenceVector = np.ndarray
ExpertName = str
SignalStrength = float
Confidence = float