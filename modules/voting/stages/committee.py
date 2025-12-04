"""
Committee Coordinator
=====================
Coordinates the voting committee by collecting votes from registered experts,
calculating weights, and determining committee decisions through weighted voting.

Refactored from voting_wrappers.py EnhancedVotingCommitteeCoordinator.
~600 lines (down from ~640 in original + ~200 duplicated base code)
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
    VotingAction,
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


# Singleton instance for committee reuse
_COMMITTEE_INSTANCE: Optional["CommitteeCoordinator"] = None


@module(**module_args("CommitteeCoordinator"))
class CommitteeCoordinator(VotingModuleBase):
    """
    Voting committee coordinator.
    
    Responsibilities:
    - Discover registered voting members
    - Collect votes from experts
    - Calculate performance-weighted expert weights
    - Determine committee decision through weighted voting
    - Analyze voting consensus
    - Generate committee thesis
    
    Uses singleton pattern to maintain state across calls.
    
    Publishes:
    - committee_decision
    - committee_confidence  
    - committee_consensus
    - expert_votes
    - committee_votes
    - member_analytics
    """
    
    def __new__(cls, *args, **kwargs):
        global _COMMITTEE_INSTANCE
        if _COMMITTEE_INSTANCE is None:
            _COMMITTEE_INSTANCE = super().__new__(cls)
        return _COMMITTEE_INSTANCE
    
    def _module_specific_init(self) -> None:
        """Initialize committee-specific state."""
        # Prevent re-initialization of singleton
        if getattr(self, '_singleton_init_done', False):
            return
        
        # Committee configuration - use dynamic threshold for mode-awareness
        self._config_consensus_threshold = self.config.get('consensus_threshold')
        self.minimum_voters = int(self.config.get('minimum_voters', 2))
        self.performance_weighting = bool(self.config.get('performance_weighting', True))
        self.max_vote_age_s = float(self.config.get('max_vote_age_s', MAX_STALENESS_SECONDS))
        
        # Discovery settings
        self.expert_votes_bus_key = str(self.config.get('expert_votes_bus_key', 'expert_votes'))
        self.discovery_mode = str(self.config.get('discovery_mode', 'registry_only'))
        self.voter_flag_name = str(self.config.get('voter_flag_name', 'is_voting_member'))
        self.voters_from_config = list(self.config.get('voters', []))
        self.ingest_minimum = int(self.config.get('ingest_minimum', self.minimum_voters))
        # Gate/risk actions should not be counted as directional votes.
        # These are risk module signals (proceed/caution/halt) that indicate safety, not direction.
        # Also filter 'hold'/'flat' since these indicate no directional guidance.
        self.ignore_actions = set(self.config.get('ignore_actions', [
            'abstain', None, 'unknown', 'neutral', 'hold', 'flat',
            # Risk gate actions (from ExecutionQualityMonitor, AnomalyDetector, etc.)
            'proceed', 'caution', 'halt', 'continue', 'confirm', 'wait',
            # Expert neutral actions (legacy - new experts use 'long'/'short'/'flat')
            'seasonal_neutral', 'momentum_neutral', 'trend_neutral', 'theme_neutral',
            # Session avoidance (legacy - now mapped to 'flat')
            'session_avoid', 'session_optimal',
            # High impact caution (legacy - now mapped to 'flat')
            'high_impact_caution',
        ]))
        self.max_votes_per_tick = int(self.config.get('max_votes_per_tick', 128))
        
        # Warmup configuration - wait for experts to have enough data before trading
        self.warmup_ticks = int(self.config.get('warmup_ticks', 20))  # Wait 50 ticks before trading
        self.min_directional_votes = int(self.config.get('min_directional_votes', 1))  # Need at least 1 long/short vote (PPOAgent usually provides one)
        self.warmup_complete = False
        self._tick_count = 0
        
        # State
        self.active_experts: List[str] = []
        self.expert_weights: Dict[str, float] = {}
        self.voting_history: deque = deque(maxlen=100)
        self.consensus_history: deque = deque(maxlen=50)
        
        # Analytics
        self.committee_analytics: Dict[str, Any] = {
            'total_decisions': 0,
            'consensus_decisions': 0,
            'emergency_overrides': 0,
            'average_confidence': 0.5,
            'expert_performance': defaultdict(float)
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
    
    @property
    def consensus_threshold(self) -> float:
        """Get consensus threshold (mode-aware)."""
        if self._config_consensus_threshold is not None:
            return float(self._config_consensus_threshold)
        return CONSENSUS_THRESHOLD_F()
    
    def _publish_committee_baseline(self) -> None:
        """Publish baseline committee keys."""
        try:
            self.smart_bus.set(
                'expert_votes', 
                [], 
                module=self.__class__.__name__, 
                thesis='Baseline expert votes'
            )
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_VOTES, 
                [], 
                module=self.__class__.__name__, 
                thesis='Baseline committee votes'
            )
            self.smart_bus.set(
                VotingBusKeys.VOTES, 
                [], 
                module=self.__class__.__name__, 
                thesis='Baseline raw votes'
            )
            self.smart_bus.set(
                'expert_weights',
                {},
                module=self.__class__.__name__,
                thesis='Baseline expert weights'
            )
        except Exception:
            pass
    
    # ============ Vote Collection ============
    
    def _discover_voters(self) -> List[str]:
        """Discover registered voting members."""
        discovered: List[str] = []
        
        # 1) Config voters first
        for name in self.voters_from_config:
            if isinstance(name, str) and name and name not in discovered:
                discovered.append(name)
        
        # 2) Registry voters with is_voting_member flag
        try:
            from modules.contracts import CONTRACTS
            for name, mc in CONTRACTS.items():
                try:
                    meta = getattr(mc, 'meta', {}) or {}
                    if self._to_bool(meta.get(self.voter_flag_name, False)):
                        if name not in discovered:
                            discovered.append(name)
                except Exception:
                    continue
        except Exception:
            pass
        
        # 3) Fallback safety
        if not discovered:
            for fallback in ['ThemeExpert', 'SeasonalityRiskExpert', 
                           'EnhancedThemeExpert', 'EnhancedSeasonalityRiskExpert']:
                if fallback not in discovered:
                    discovered.append(fallback)
        
        # De-duplicate
        seen = set()
        result = []
        for name in discovered:
            if name not in seen:
                result.append(name)
                seen.add(name)
        
        return result
    
    def _to_bool(self, v: Any, default: bool = False) -> bool:
        """Convert value to boolean."""
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        s = str(v).strip().lower()
        return s in ('true', '1', 'yes', 'y', 'on')
    
    def _normalize_vote_entry(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalize a raw vote entry to canonical format."""
        try:
            if not isinstance(raw, dict):
                return None
            
            expert = str(raw.get('expert') or raw.get('name') or 'unknown').strip()
            if not expert:
                return None
            
            vote = raw.get('vote') or raw.get('proposal') or {}
            if not isinstance(vote, dict):
                vote = {}
            
            confidence = float(raw.get('confidence', 0.0) or 0.0)
            ts = raw.get('timestamp') or datetime.datetime.now().isoformat()
            
            return {
                'expert': expert,
                'vote': dict(vote),
                'confidence': max(0.0, min(1.0, confidence)),
                'timestamp': ts
            }
        except Exception:
            return None
    
    def _voter_key_pairs(self, name: str) -> List[Tuple[str, str]]:
        """Get (proposal_key, confidence_key) pairs for a voter."""
        pairs = [
            (VotingBusKeys.expert_proposal(name), VotingBusKeys.expert_confidence(name))
        ]
        
        # Add alternate keys for known experts
        if name in ['EnhancedThemeExpert', 'ThemeExpert']:
            pairs.append(('theme_voting_proposal', 'theme_confidence'))
        if name in ['EnhancedSeasonalityRiskExpert', 'SeasonalityRiskExpert']:
            pairs.append(('seasonality_voting_proposal', 'seasonality_confidence'))
        if name == 'MomentumExpert':
            pairs.append(('momentum_voting_proposal', 'momentum_confidence'))
        if name == 'TrendExpert':
            pairs.append(('trend_voting_proposal', 'trend_confidence'))
        if name == 'PPOAgent':
            pairs.append(('policy_actions', 'agent_performance'))
        if name == 'MetaAgent':
            pairs.append(('automation_decisions', 'meta_performance'))
        
        return pairs
    
    async def _collect_expert_votes(self) -> List[Dict[str, Any]]:
        """Collect votes from registered experts."""
        try:
            voters = self._discover_voters()
            voters_set = set(voters)
            expert_votes: List[Dict[str, Any]] = []
            now_ts = time.time()
            
            self.logger.debug(f"[COLLECT] Discovered {len(voters)} voters: {voters}")
            self.logger.debug(f"[COLLECT] Discovery mode: {self.discovery_mode}, ingest_minimum: {self.ingest_minimum}")
            
            # 1) Feed-first: read from expert_votes bus key
            if self.discovery_mode in ('feed_only', 'feed_then_registry'):
                try:
                    feed = self.smart_bus.get(self.expert_votes_bus_key, self.__class__.__name__) or []
                    if isinstance(feed, list):
                        for raw in feed[-self.max_votes_per_tick:]:
                            norm = self._normalize_vote_entry(raw)
                            if norm and norm.get('expert') in voters_set:
                                # Check age
                                ts_val = raw.get('timestamp') or norm.get('timestamp')
                                if ts_val:
                                    try:
                                        age = now_ts - datetime.datetime.fromisoformat(str(ts_val)).timestamp()
                                        if age > self.max_vote_age_s:
                                            continue
                                    except Exception:
                                        pass
                                expert_votes.append(norm)
                except Exception as e:
                    self.logger.warning(f"Failed to read feed: {e}")
            
            # De-duplicate by expert
            by_expert: Dict[str, Dict[str, Any]] = {}
            for v in expert_votes:
                by_expert[v['expert']] = v
            expert_votes = list(by_expert.values())
            
            # 2) Fallback: read per-voter bus keys - ALWAYS do this for registry_only mode
            if self.discovery_mode in ('registry_only', 'feed_then_registry'):
                for name in voters:
                    if name in by_expert:
                        continue
                    try:
                        for prop_key, conf_key in self._voter_key_pairs(name):
                            proposal = self.smart_bus.get(prop_key, self.__class__.__name__, default=None)
                            if proposal is None:
                                self.logger.debug(f"[COLLECT] {name}: No proposal at key '{prop_key}'")
                                continue
                            confidence = self.smart_bus.get(conf_key, self.__class__.__name__, default=None)
                            if confidence is None:
                                self.logger.debug(f"[COLLECT] {name}: No confidence at key '{conf_key}'")
                                continue
                            
                            self.logger.debug(f"[COLLECT] {name}: Found vote at '{prop_key}' with conf={confidence}")
                            
                            raw = {
                                'expert': name,
                                'vote': dict(proposal) if isinstance(proposal, dict) else {'action': str(proposal)},
                                'confidence': float(confidence) if isinstance(confidence, (int, float)) else 0.5,
                                'timestamp': datetime.datetime.now().isoformat(),
                            }
                            norm = self._normalize_vote_entry(raw)
                            if norm:
                                by_expert[name] = norm
                                self.logger.debug(f"[COLLECT] {name}: Normalized action='{norm.get('vote', {}).get('action')}'")
                            break
                    except Exception as e:
                        self.logger.warning(f"Failed to collect vote from {name}: {e}")
                
                expert_votes = list(by_expert.values())
            
            # 3) Filter abstains if non-abstain votes present
            # Log what we collected before filtering
            before_count = len(expert_votes)
            actions_collected = [v.get('vote', {}).get('action', 'unknown') for v in expert_votes]
            
            if any(v.get('vote', {}).get('action') not in self.ignore_actions for v in expert_votes):
                expert_votes = [
                    v for v in expert_votes 
                    if v.get('vote', {}).get('action') not in self.ignore_actions
                ]
                filtered_count = before_count - len(expert_votes)
                if filtered_count > 0:
                    self.logger.debug(
                        f"[FILTER] Filtered {filtered_count} neutral votes. "
                        f"Actions: {actions_collected}"
                    )
            
            # 4) Add memory vote if available
            expert_votes = await self._add_memory_vote(expert_votes)
            
            # 5) Cap
            if len(expert_votes) > self.max_votes_per_tick:
                expert_votes = expert_votes[-self.max_votes_per_tick:]
            
            self.logger.info(
                f"[VOTES] Collected {len(expert_votes)} votes from {len(voters)} voters"
            )
            
            return expert_votes
        
        except Exception as e:
            self.logger.error(f"Vote collection failed: {e}")
            return []
    
    async def _add_memory_vote(self, expert_votes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add memory system vote if available."""
        try:
            memory_vote = self.smart_bus.get('memory_vote', self.__class__.__name__)
            if not isinstance(memory_vote, dict):
                return expert_votes
            
            signed_bias = float(memory_vote.get('signed_bias', 0.0))
            mem_confidence = float(memory_vote.get('confidence', 0.5))
            
            if abs(signed_bias) <= 0.05 or mem_confidence <= 0.2:
                return expert_votes
            
            # Determine action from bias
            if signed_bias > 0.1:
                mem_action = 'long'
            elif signed_bias < -0.1:
                mem_action = 'short'
            else:
                mem_action = 'abstain'
            
            # Adjust confidence by neural risk
            neural_risk = float(memory_vote.get('neural_risk_hint', 0.5))
            adjusted_confidence = mem_confidence * (1.0 - neural_risk * 0.3)
            
            memory_entry = {
                'expert': 'UnifiedMemory',
                'vote': {
                    'action': mem_action,
                    'signal_strength': abs(float(memory_vote.get('vote_value', 0.0))),
                    'signed_bias': signed_bias,
                },
                'confidence': adjusted_confidence,
                'timestamp': datetime.datetime.now().isoformat(),
            }
            
            expert_votes.append(memory_entry)
            
            self.logger.info(
                f"🧠 MEMORY_VOTE_ADDED | action={mem_action} | "
                f"confidence={adjusted_confidence:.2f}"
            )
            
            return expert_votes
        
        except Exception as e:
            self.logger.debug(f"Memory vote skipped: {e}")
            return expert_votes
    
    # ============ Weight Calculation ============
    
    async def _calculate_expert_weights(
        self, 
        expert_votes: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculate performance-weighted expert weights."""
        try:
            if not expert_votes:
                return {}
            
            # Get bus-level performance
            bus_perf = self.smart_bus.get('expert_performance', self.__class__.__name__) or {}
            if not isinstance(bus_perf, dict):
                bus_perf = {}
            
            # Filter abstains if non-abstains present
            actions = [v.get('vote', {}).get('action') for v in expert_votes]
            if any(a not in self.ignore_actions for a in actions):
                expert_votes = [
                    v for v in expert_votes 
                    if v.get('vote', {}).get('action') not in self.ignore_actions
                ]
            
            weights: Dict[str, float] = {}
            total = 0.0
            
            for v in expert_votes:
                name = v['expert']
                conf = float(v.get('confidence', 0.0))
                base = max(0.0, conf)
                
                if self.performance_weighting:
                    perf = bus_perf.get(name)
                    if perf is None:
                        perf = float(self.committee_analytics['expert_performance'].get(name, 0.5))
                    else:
                        try:
                            perf = float(perf)
                        except Exception:
                            perf = 0.5
                    base *= (0.5 + max(0.0, min(1.0, perf)))
                
                # Regime adjustment
                regime = self.smart_bus.get('market_regime', self.__class__.__name__) or 'unknown'
                base *= self._get_regime_adjustment(name, regime)
                
                w = max(1e-6, min(2.0, base))
                weights[name] = w
                total += w
            
            if total <= 0.0:
                n = len(expert_votes)
                return {v['expert']: 1.0 / n for v in expert_votes}
            
            # Normalize
            for k in weights:
                weights[k] = weights[k] / total
            
            return weights
        
        except Exception as e:
            self.logger.error(f"Weight calculation failed: {e}")
            return {}
    
    def _get_regime_adjustment(self, expert_name: str, regime: str) -> float:
        """Get regime-based weight adjustment for expert."""
        adjustments = {
            'ThemeExpert': {'trending': 1.2, 'volatile': 0.9, 'ranging': 1.0},
            'EnhancedThemeExpert': {'trending': 1.2, 'volatile': 0.9, 'ranging': 1.0},
            'SeasonalityRiskExpert': {'trending': 1.0, 'volatile': 1.1, 'ranging': 1.2},
            'EnhancedSeasonalityRiskExpert': {'trending': 1.0, 'volatile': 1.1, 'ranging': 1.2},
            # MomentumExpert: Strong in volatile/trending, weaker in ranging
            'MomentumExpert': {'trending': 1.3, 'volatile': 1.2, 'ranging': 0.8},
            # TrendExpert: Strong in trending, weaker in volatile/ranging
            'TrendExpert': {'trending': 1.4, 'volatile': 0.85, 'ranging': 0.7},
        }
        return adjustments.get(expert_name, {}).get(regime, 1.0)
    
    # ============ Decision Making ============
    
    def _action_sign(self, action: str) -> float:
        """Map action string to directional sign.
        
        Returns:
            +1.0 for bullish/long actions
            -1.0 for bearish/short actions
            0.0 for neutral actions
        """
        if not action:
            return 0.0
        
        a = action.strip().lower()
        
        # Long/bullish actions
        if a.startswith('long') or 'long_' in a or a == 'buy':
            return 1.0
        # Short/bearish actions
        if a.startswith('short') or 'short_' in a or a == 'sell':
            return -1.0
        # Risk reduction (bearish)
        if a in ('reduce_risk', 'halt', 'emergency_stop'):
            return -1.0
        # Risk increase (bullish)
        if a in ('increase_risk', 'aggressive'):
            return 1.0
        # Neutral/gate actions (should be in ignore_actions, but just in case)
        if a in ('hold', 'abstain', 'wait', 'neutral', 'proceed', 'continue', 'caution'):
            return 0.0
        # Thematic bullish actions
        if a in ('trend_following', 'seasonal_long_bias', 'long_risk_assets', 'risk_on', 'bullish', 'breakout'):
            return 1.0
        # Thematic bearish actions
        if a in ('seasonal_short_bias', 'short_risk_assets', 'safe_haven_rotation', 
                 'volatility_hedging', 'risk_off', 'bearish', 'mean_reversion'):
            return -1.0
        # MomentumExpert actions
        if a in ('momentum_long',):
            return 1.0
        if a in ('momentum_short',):
            return -1.0
        if a in ('momentum_neutral',):
            return 0.0
        # TrendExpert actions
        if a in ('trend_bullish',):
            return 1.0
        if a in ('trend_bearish',):
            return -1.0
        if a in ('trend_neutral',):
            return 0.0
        # ThemeExpert neutral actions (not directional)
        if a in ('theme_neutral',):
            return 0.0
        # SeasonalityRiskExpert neutral/timing actions (not directional)
        if a in ('seasonal_neutral', 'session_avoid', 'session_optimal', 'high_impact_caution'):
            return 0.0
        
        return 0.0
    
    async def _determine_committee_decision(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Determine final committee decision via weighted voting."""
        try:
            if not expert_votes:
                return {'action': 'abstain', 'reason': 'no_expert_votes'}
            
            weighted_actions: Dict[str, float] = defaultdict(float)
            total_weight = 0.0
            
            for v in expert_votes:
                name = v['expert']
                w = float(expert_weights.get(name, 1.0))
                action = v.get('vote', {}).get('action', 'abstain')
                weighted_actions[action] += w
                total_weight += w
            
            if not weighted_actions or total_weight <= 0.0:
                return {'action': 'abstain', 'reason': 'no_valid_actions'}
            
            # If non-abstain votes exist, ignore abstains in max
            has_non_abstain = any(a not in self.ignore_actions for a in weighted_actions)
            if has_non_abstain:
                filtered = {a: w for a, w in weighted_actions.items() if a not in self.ignore_actions}
                best_action, best_weight = max(filtered.items(), key=lambda x: x[1])
                denom = sum(filtered.values()) or total_weight
            else:
                best_action, best_weight = max(weighted_actions.items(), key=lambda x: x[1])
                denom = total_weight
            
            consensus_strength = best_weight / denom if denom > 0 else 0.0
            
            return {
                'action': best_action,
                'consensus_strength': consensus_strength,
                'total_weight': total_weight,
                'action_weights': dict(weighted_actions),
                'decision_type': 'consensus' if consensus_strength >= self.consensus_threshold else 'plurality'
            }
        
        except Exception as e:
            self.logger.error(f"Decision determination failed: {e}")
            return {'action': 'abstain', 'reason': f'error: {e}'}
    
    async def _analyze_voting_consensus(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analyze voting consensus and identify conflicts."""
        try:
            if not expert_votes:
                return {'consensus_exists': False, 'reason': 'no_votes'}
            
            action_weights: Dict[str, float] = defaultdict(float)
            for v in expert_votes:
                action = v.get('vote', {}).get('action', 'abstain')
                w = float(expert_weights.get(v['expert'], 1.0))
                action_weights[action] += w
            
            total_weight = sum(action_weights.values())
            if total_weight <= 0:
                return {'consensus_exists': False, 'reason': 'zero_weight'}
            
            dominant_action, dom_w = max(action_weights.items(), key=lambda x: x[1])
            consensus_strength = dom_w / total_weight
            
            # Assess conflict level via entropy
            conflict_level = self._assess_conflict_level(action_weights, total_weight)
            
            return {
                'consensus_exists': consensus_strength >= self.consensus_threshold,
                'consensus_strength': consensus_strength,
                'dominant_action': dominant_action,
                'action_distribution': dict(action_weights),
                'conflict_level': conflict_level,
                'vote_count': len(expert_votes),
                'total_weight': total_weight
            }
        
        except Exception as e:
            return {'consensus_exists': False, 'error': str(e)}
    
    def _assess_conflict_level(self, action_weights: Dict[str, float], total_weight: float) -> str:
        """Assess conflict level via entropy."""
        try:
            if len(action_weights) <= 1:
                return 'NONE'
            
            probs = [w / total_weight for w in action_weights.values() if total_weight > 0]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            max_entropy = np.log2(len(action_weights))
            normalized = (entropy / max_entropy) if max_entropy > 0 else 0.0
            
            if normalized < 0.3:
                return 'LOW'
            elif normalized < 0.6:
                return 'MEDIUM'
            elif normalized < 0.8:
                return 'HIGH'
            return 'SEVERE'
        
        except Exception:
            return 'UNKNOWN'
    
    async def _calculate_committee_confidence(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float],
        consensus: Dict[str, Any]
    ) -> float:
        """Calculate overall committee confidence."""
        try:
            if not expert_votes or not expert_weights:
                return 0.1
            
            weighted_conf = 0.0
            total_w = 0.0
            
            for v in expert_votes:
                name = v['expert']
                conf = float(v.get('confidence', 0.0))
                w = float(expert_weights.get(name, 1.0))
                weighted_conf += conf * w
                total_w += w
            
            avg_conf = (weighted_conf / total_w) if total_w > 0 else 0.0
            consensus_strength = float(consensus.get('consensus_strength', 0.0))
            
            # Scale by consensus
            final = avg_conf * (0.7 + 0.6 * consensus_strength)
            return max(0.1, min(1.0, final))
        
        except Exception:
            return 0.3
    
    async def _generate_committee_thesis(
        self,
        decision: Dict[str, Any],
        confidence: float,
        consensus: Dict[str, Any],
        expert_votes: List[Dict[str, Any]]
    ) -> str:
        """Generate committee decision thesis."""
        try:
            action = decision.get('action', 'unknown')
            cs = float(consensus.get('consensus_strength', 0.0))
            label = 'HIGH' if confidence > 0.7 else 'MODERATE' if confidence > 0.4 else 'LOW'
            
            parts = [
                f"COMMITTEE DECISION: {action.upper()} with {label} confidence ({confidence:.1%})"
            ]
            
            if consensus.get('consensus_exists'):
                parts.append(f"STRONG CONSENSUS: {cs:.1%} agreement among {len(expert_votes)} experts")
            else:
                parts.append(f"DIVIDED OPINION: {consensus.get('conflict_level')} conflict")
            
            parts.append(f"DECISION TYPE: {decision.get('decision_type', 'unknown').upper()}")
            
            return ' | '.join(parts)
        
        except Exception as e:
            return f"Committee thesis failed: {e}"
    
    # ============ Per-Instrument Voting ============
    
    def _convert_to_per_instrument_votes(
        self, 
        expert_votes: List[Dict[str, Any]]
    ) -> List[PerInstrumentVote]:
        """
        Convert expert votes to PerInstrumentVote format.
        
        Priority order for per-instrument data:
        1. Check SmartInfoBus for {Expert}_per_instrument_votes (NEW - experts publish here)
        2. Check for 'proposals' dict in vote_data 
        3. Check for 'proposals' nested in vote dict
        4. Fall back to legacy global vote (applied to all instruments)
        """
        per_inst_votes = []
        
        for vote_data in expert_votes:
            try:
                expert_name = str(vote_data.get('expert', 'unknown'))
                vote = vote_data.get('vote', {})
                confidence = float(vote_data.get('confidence', 0.5))
                
                piv: Optional[PerInstrumentVote] = None
                
                # PRIORITY 1: Check SmartInfoBus for per-instrument votes
                # Experts like MomentumExpert, TrendExpert publish to {Expert}_per_instrument_votes
                per_inst_key = f'{expert_name}_per_instrument_votes'
                bus_per_inst = self.smart_bus.get(per_inst_key, self.__class__.__name__, default=None)
                
                if bus_per_inst and isinstance(bus_per_inst, dict) and len(bus_per_inst) > 0:
                    # Found per-instrument votes on bus!
                    piv = PerInstrumentVote(member=expert_name)
                    for inst, inst_vote in bus_per_inst.items():
                        if isinstance(inst_vote, dict):
                            inst_norm = normalize_instrument(inst)
                            inst_action = str(inst_vote.get('action', 'flat')).lower()
                            inst_conf = float(inst_vote.get('confidence', confidence))
                            inst_mag = float(inst_vote.get('magnitude', inst_vote.get('signal_strength', inst_conf)))
                            inst_rationale = str(inst_vote.get('rationale', f'Per-inst vote from {expert_name}'))
                            
                            piv.set_proposal(InstrumentProposal(
                                instrument=inst_norm,
                                action=inst_action,
                                confidence=inst_conf,
                                magnitude=inst_mag,
                                rationale=inst_rationale,
                            ))
                    
                    if piv.proposals:
                        self.logger.debug(
                            f"[CONVERT] {expert_name}: Using per-instrument votes from bus "
                            f"({len(piv.proposals)} instruments)"
                        )
                        per_inst_votes.append(piv)
                        continue
                
                # PRIORITY 2: Check if vote_data has per-instrument proposals
                if 'proposals' in vote_data and isinstance(vote_data['proposals'], dict):
                    # New per-instrument format
                    piv = PerInstrumentVote.from_dict({
                        'member': expert_name,
                        'proposals': vote_data['proposals'],
                        'action': vote_data.get('action', 'flat'),
                        'confidence': confidence,
                    })
                    self.logger.debug(
                        f"[CONVERT] {expert_name}: Using proposals from vote_data "
                        f"({len(piv.proposals)} instruments)"
                    )
                # PRIORITY 3: Check if proposals nested in vote dict
                elif 'proposals' in vote and isinstance(vote['proposals'], dict):
                    piv = PerInstrumentVote.from_dict({
                        'member': expert_name,
                        'proposals': vote['proposals'],
                        'action': vote.get('action', 'flat'),
                        'confidence': confidence,
                    })
                    self.logger.debug(
                        f"[CONVERT] {expert_name}: Using proposals from vote dict "
                        f"({len(piv.proposals)} instruments)"
                    )
                # PRIORITY 4: Legacy global vote - apply to all instruments
                else:
                    action = str(vote.get('action', 'flat')).lower()
                    magnitude = float(vote.get('signal_strength', vote.get('magnitude', confidence)))
                    
                    piv = PerInstrumentVote(member=expert_name)
                    for inst in DEFAULT_INSTRUMENTS:
                        piv.set_proposal(InstrumentProposal(
                            instrument=inst,
                            action=action,
                            confidence=confidence,
                            magnitude=magnitude,
                            rationale=f"Legacy global vote from {expert_name}",
                        ))
                    self.logger.debug(
                        f"[CONVERT] {expert_name}: Using LEGACY global vote ({action}) "
                        f"for all {len(DEFAULT_INSTRUMENTS)} instruments"
                    )
                
                if piv is not None:
                    per_inst_votes.append(piv)
                
            except Exception as e:
                self.logger.warning(f"Failed to convert vote to per-instrument: {e}")
        
        return per_inst_votes
    
    async def _aggregate_per_instrument(
        self,
        expert_votes: List[Dict[str, Any]],
        expert_weights: Dict[str, float],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Aggregate votes per instrument.
        
        Returns:
            Dict mapping instrument -> aggregated decision dict
        """
        # Convert to PerInstrumentVote format
        per_inst_votes = self._convert_to_per_instrument_votes(expert_votes)
        
        # Log per-instrument vote details for debugging
        for piv in per_inst_votes:
            for inst, prop in piv.proposals.items():
                self.logger.debug(
                    f"[PER-INST VOTE] {piv.member} → {inst}: "
                    f"action={prop.action}, conf={prop.confidence:.2f}, mag={prop.magnitude:.2f}"
                )
        
        # Aggregate all instruments
        aggregated = aggregate_all_instruments(
            votes=per_inst_votes,
            instruments=DEFAULT_INSTRUMENTS,
            weights=expert_weights,
        )
        
        # Convert to dict format and log aggregated decisions
        result = {}
        for inst, decision in aggregated.items():
            result[inst] = {
                'action': decision.action,
                'confidence': decision.confidence,
                'consensus_score': decision.consensus_score,
                'vote_count': decision.vote_count,
                'long_votes': decision.long_votes,
                'short_votes': decision.short_votes,
                'flat_votes': decision.flat_votes,
                'weighted_score': decision.weighted_score,
                'instrument': inst,
            }
            # Log aggregated decision at INFO level
            self.logger.info(
                f"[COMMITTEE] {inst}: action={decision.action}, conf={decision.confidence:.2f}, "
                f"consensus={decision.consensus_score:.2f}, "
                f"votes={decision.long_votes}L/{decision.short_votes}S/{decision.flat_votes}F"
            )
        
        return result

    # ============ Main Processing ============
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """Main committee processing."""
        start_time = time.time()
        name = self.__class__.__name__
        
        try:
            # Track warmup progress
            self._tick_count += 1
            
            # Get decision ID
            decision_id = self.smart_bus.get('kernel_decision_id', name)
            if not decision_id:
                self._decision_counter += 1
                decision_id = f"{datetime.datetime.now().isoformat()}#{self._decision_counter}"
            
            # Collect votes
            expert_votes = await self._collect_expert_votes()
            expert_weights = await self._calculate_expert_weights(expert_votes)
            
            # Count directional votes (non-flat/non-neutral)
            directional_votes = [
                v for v in expert_votes 
                if v.get('vote', {}).get('action') not in self.ignore_actions
            ]
            n_directional = len(directional_votes)
            
            # Check warmup status
            if not self.warmup_complete:
                # Still in warmup - check if we should complete
                if self._tick_count >= self.warmup_ticks and n_directional >= self.min_directional_votes:
                    self.warmup_complete = True
                    self.logger.info(
                        f"[WARMUP] ✅ Complete after {self._tick_count} ticks with "
                        f"{n_directional} directional votes"
                    )
                    # Publish warmup complete status
                    self.smart_bus.set(
                        'warmup_status',
                        {
                            'complete': True,
                            'tick_count': self._tick_count,
                            'warmup_ticks': self.warmup_ticks,
                            'directional_votes': n_directional,
                            'min_directional_votes': self.min_directional_votes,
                            'progress_pct': 100
                        },
                        module=name,
                        thesis='Warmup complete - trading enabled'
                    )
                else:
                    # Still warming up - return abstain
                    warmup_reason = (
                        f"Warmup in progress: tick {self._tick_count}/{self.warmup_ticks}, "
                        f"{n_directional}/{self.min_directional_votes} directional votes"
                    )
                    self.logger.info(f"[WARMUP] ⏳ {warmup_reason}")
                    
                    # Publish warmup status to bus
                    self.smart_bus.set(
                        'warmup_status',
                        {
                            'complete': False,
                            'tick_count': self._tick_count,
                            'warmup_ticks': self.warmup_ticks,
                            'directional_votes': n_directional,
                            'min_directional_votes': self.min_directional_votes,
                            'progress_pct': min(100, (self._tick_count / self.warmup_ticks) * 100)
                        },
                        module=name,
                        thesis=warmup_reason
                    )
                    
                    return self._warmup_output(warmup_reason, decision_id)
            
            # Make decisions (global)
            decision = await self._determine_committee_decision(expert_votes, expert_weights)
            consensus = await self._analyze_voting_consensus(expert_votes, expert_weights)
            confidence = await self._calculate_committee_confidence(expert_votes, expert_weights, consensus)
            thesis = await self._generate_committee_thesis(decision, confidence, consensus, expert_votes)
            
            # ========== NEW: Per-instrument decisions ==========
            per_instrument_decisions = await self._aggregate_per_instrument(expert_votes, expert_weights)
            
            # Publish per-instrument decisions to bus
            self.smart_bus.set(
                'committee_decisions_by_instrument',
                per_instrument_decisions,
                module=name,
                thesis=f"Per-instrument decisions for {len(per_instrument_decisions)} instruments"
            )
            
            # Build analytics surfaces
            committee_members = [v.get('expert', 'unknown') for v in expert_votes]
            n_members = len(committee_members)
            member_confidences_map = {v['expert']: float(v.get('confidence', 0.0)) for v in expert_votes}
            
            # Vectorize proposals
            proposal_vectors = []
            for v in expert_votes:
                vote = v.get('vote') or {}
                action = str(vote.get('action', 'abstain')).lower()
                ss = float(vote.get('signal_strength', v.get('confidence', 0.0)) or 0.0)
                conf = float(v.get('confidence', 0.0))
                proposal_vectors.append([self._action_sign(action) * ss, conf])
            
            # Update SmartInfoBus
            await self._update_bus(
                decision, consensus, confidence, expert_votes, expert_weights,
                committee_members, proposal_vectors, member_confidences_map, decision_id, thesis
            )
            
            # Record decision
            self._record_decision(decision, confidence, consensus, expert_votes)
            
            # Build output
            elapsed_ms = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric(name, 'process', elapsed_ms, True)
            
            now = datetime.datetime.now()
            time_of_day = now.hour * 60 + now.minute + now.second / 60.0
            
            # Build committee_votes in the simplified format
            committee_votes = [
                {
                    'action': v.get('vote', {}).get('action', 'abstain'),
                    'confidence': float(v.get('confidence', 0.0)),
                    'expert': v.get('expert', 'unknown')
                }
                for v in expert_votes
            ]
            
            # Build raw_proposals from expert votes
            raw_proposals = {
                v.get('expert', 'unknown'): v.get('vote', {})
                for v in expert_votes
            }
            
            # Build committee_summary
            committee_summary = {
                'total_members': n_members,
                'participating_members': len(expert_votes),
                'decision': decision.get('action', 'abstain'),
                'confidence': confidence,
                'consensus_strength': consensus.get('consensus_strength', 0.0),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Build voting_summary (legacy compatibility)
            voting_summary = {
                'action': decision.get('action', 'abstain'),
                'confidence': confidence,
                'member_count': n_members,
                'consensus': consensus
            }
            
            return {
                # Primary outputs (new v5.0)
                VotingBusKeys.COMMITTEE_DECISION: decision,
                VotingBusKeys.COMMITTEE_CONSENSUS: consensus,
                VotingBusKeys.COMMITTEE_CONFIDENCE: confidence,
                VotingBusKeys.COMMITTEE_VOTES: committee_votes,
                'committee_summary': committee_summary,
                VotingBusKeys.COMMITTEE_DECISION_ID: decision_id,
                'raw_proposals': raw_proposals,
                'member_confidences': member_confidences_map,
                'voting_weights': expert_weights,
                
                # ========== NEW: Per-instrument decisions ==========
                'committee_decisions_by_instrument': per_instrument_decisions,
                
                # Backward compatibility (old EnhancedVotingCommitteeCoordinator keys)
                VotingBusKeys.VOTES: committee_votes,
                VotingBusKeys.VOTING_SUMMARY: voting_summary,
                VotingBusKeys.STRATEGY_ARBITER_WEIGHTS: expert_weights,
                'expert_votes': expert_votes,
                'expert_weights': expert_weights,
                # FIX: Add committee_member_confidences as contract-required alias
                'committee_member_confidences': member_confidences_map,
                'strategy_weights': expert_weights,
                'member_performance': {m: {'confidence': float(member_confidences_map.get(m, 0.5)), 'weight': float(expert_weights.get(m, 0.5))} for m in committee_members},
                'committee_analytics': dict(self.committee_analytics),
                'committee_members': committee_members,
                'n_members': n_members,
                'proposal_vectors': proposal_vectors,
                'decision_id': decision_id,
                'time_of_day': time_of_day,
                
                # Trade vote outputs
                'trade_vote': {
                    'action': decision.get('action', 'abstain'),
                    'confidence': float(confidence),
                    'timestamp': datetime.datetime.now().isoformat()
                },
                'trade_vote_v2': {
                    'action': decision.get('action', 'abstain'),
                    'size': float(confidence),
                    'confidence': float(confidence),
                    'consensus_score': float(consensus.get('consensus_strength', 0.0)),
                    'decision_id': decision_id,
                    'timestamp': datetime.datetime.now().isoformat()
                },
                '_thesis': thesis
            }
        
        except Exception as e:
            if self.error_pinpointer is not None:
                error_context = self.error_pinpointer.analyze_error(e, "committee_process")
                msg = str(error_context)
            else:
                msg = str(e)
            return self._error_output(msg)
    
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
        thesis: str
    ) -> None:
        """Update SmartInfoBus with committee results."""
        try:
            name = self.__class__.__name__
            
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_DECISION,
                decision,
                module=name,
                thesis=thesis,
                confidence=confidence,
            )
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_CONSENSUS,
                consensus,
                module=name,
                thesis='Committee consensus',
            )
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_CONFIDENCE,
                confidence,
                module=name,
                thesis=f'Confidence: {confidence:.1%}',
            )
            self.smart_bus.set('expert_votes', list(expert_votes), module=name, thesis='Expert votes snapshot')
            
            # Simplified committee votes
            simplified = [
                {
                    'action': v.get('vote', {}).get('action', 'abstain'),
                    'confidence': float(v.get('confidence', 0.0)),
                    'expert': v.get('expert', 'unknown')
                }
                for v in expert_votes
            ]
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_VOTES,
                simplified,
                module=name,
                thesis='Committee votes',
            )
            
            # Analytics surfaces
            self.smart_bus.set('committee_members', committee_members, module=name, thesis='Members list')
            self.smart_bus.set('committee_proposal_vectors', proposal_vectors, module=name, thesis='Proposal vectors')
            self.smart_bus.set(
                VotingBusKeys.MEMBER_CONFIDENCES,
                list(member_confidences_map.values()),
                module=name,
                thesis='Member confidences',
            )
            # FIX: Publish committee_member_confidences alias for ConsensusAnalyzer and HorizonAligner
            self.smart_bus.set(
                'committee_member_confidences',
                list(member_confidences_map.values()),
                module=name,
                thesis='Committee member confidences (alias)',
            )
            self.smart_bus.set(
                VotingBusKeys.COMMITTEE_DECISION_ID,
                decision_id,
                module=name,
                thesis='Decision ID',
            )
            self.smart_bus.set('committee_analytics', dict(self.committee_analytics), module=name, thesis='Analytics')
            
            # FIX: Publish strategy_weights and member_performance for StrategyIntrospector
            self.smart_bus.set(
                'strategy_weights',
                {'by_member': expert_weights, 'members': committee_members, 'weights': list(expert_weights.values()), 'timestamp': datetime.datetime.now().isoformat()},
                module=name,
                thesis='Strategy weights per expert',
            )
            # Member performance: simple summary from analytics
            member_perf = {}
            for expert in committee_members:
                member_perf[expert] = {
                    'contribution_score': float(expert_weights.get(expert, 0.5)),
                    'confidence': float(member_confidences_map.get(expert, 0.5)),
                }
            self.smart_bus.set(
                'member_performance',
                {'by_member': member_perf, 'timestamp': datetime.datetime.now().isoformat()},
                module=name,
                thesis='Member performance summary',
            )
            
        except Exception as e:
            self.logger.error(f"Bus update failed: {e}")
    
    def _record_decision(
        self,
        decision: Dict[str, Any],
        confidence: float,
        consensus: Dict[str, Any],
        expert_votes: List[Dict[str, Any]]
    ) -> None:
        """Record decision for analytics."""
        try:
            record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'decision': decision,
                'confidence': confidence,
                'consensus': consensus,
                'expert_count': len(expert_votes)
            }
            self.voting_history.append(record)
            
            self.committee_analytics['total_decisions'] += 1
            if consensus.get('consensus_exists'):
                self.committee_analytics['consensus_decisions'] += 1
            
            # Update average confidence
            n = self.committee_analytics['total_decisions']
            old = float(self.committee_analytics.get('average_confidence', 0.5))
            self.committee_analytics['average_confidence'] = (old * (n - 1) + confidence) / n
        
        except Exception as e:
            self.logger.warning(f"Decision recording failed: {e}")
    
    def _warmup_output(self, reason: str, decision_id: str) -> Dict[str, Any]:
        """Return contract-compliant warmup output (hold during warmup)."""
        warmup_thesis = f"WARMUP: {reason}"
        return {
            # Primary outputs (new v5.0)
            VotingBusKeys.COMMITTEE_DECISION: {'action': 'abstain', 'reason': reason},
            VotingBusKeys.COMMITTEE_CONSENSUS: {
                'consensus_exists': False,
                'warmup': True,
                'reason': reason,
            },
            VotingBusKeys.COMMITTEE_CONFIDENCE: 0.0,
            VotingBusKeys.COMMITTEE_VOTES: [],
            'committee_summary': {
                'warmup': True,
                'total_members': 0,
                'decision': 'abstain',
                'reason': reason,
                'tick_count': self._tick_count,
                'warmup_ticks': self.warmup_ticks
            },
            VotingBusKeys.COMMITTEE_DECISION_ID: decision_id,
            'raw_proposals': {},
            'member_confidences': {},
            'voting_weights': {},
            # FIX: Add contract-required aliases
            'committee_member_confidences': {},
            'strategy_weights': {},
            'member_performance': {},
            
            # Backward compatibility
            VotingBusKeys.VOTES: [],
            VotingBusKeys.VOTING_SUMMARY: {'action': 'abstain', 'warmup': True, 'reason': reason},
            VotingBusKeys.STRATEGY_ARBITER_WEIGHTS: {},
            'expert_votes': [],
            'expert_weights': {},
            'committee_analytics': dict(self.committee_analytics),
            'committee_members': [],
            'n_members': 0,
            'proposal_vectors': [],
            'decision_id': decision_id,
            'time_of_day': 0,
            
            # Trade vote outputs
            'trade_vote': {'action': 'abstain', 'confidence': 0.0, 'warmup': True},
            'trade_vote_v2': {
                'action': 'abstain',
                'size': 0.0,
                'confidence': 0.0,
                'consensus_score': 0.0,
                'warmup': True,
                'decision_id': decision_id,
                'timestamp': datetime.datetime.now().isoformat()
            },
            '_thesis': warmup_thesis
        }
    
    def _error_output(self, error: str) -> Dict[str, Any]:
        """Return contract-compliant error output."""
        return {
            # Primary outputs (new v5.0)
            VotingBusKeys.COMMITTEE_DECISION: {'action': 'abstain', 'reason': f'error: {error}'},
            VotingBusKeys.COMMITTEE_CONSENSUS: {'consensus_exists': False, 'error': error},
            VotingBusKeys.COMMITTEE_CONFIDENCE: 0.1,
            VotingBusKeys.COMMITTEE_VOTES: [],
            'committee_summary': {'error': error, 'total_members': 0, 'decision': 'abstain'},
            VotingBusKeys.COMMITTEE_DECISION_ID: None,
            'raw_proposals': {},
            'member_confidences': {},
            'voting_weights': {},
            # FIX: Add contract-required aliases
            'committee_member_confidences': {},
            'strategy_weights': {},
            'member_performance': {},
            
            # Backward compatibility
            VotingBusKeys.VOTES: [],
            VotingBusKeys.VOTING_SUMMARY: {'action': 'abstain', 'error': error},
            VotingBusKeys.STRATEGY_ARBITER_WEIGHTS: {},
            'expert_votes': [],
            'expert_weights': {},
            'committee_analytics': {'error': error},
            'committee_members': [],
            'n_members': 0,
            'proposal_vectors': [],
            'decision_id': None,
            'time_of_day': 0,
            
            # Trade vote outputs
            'trade_vote': {'action': 'abstain', 'confidence': 0.1},
            'trade_vote_v2': {'action': 'abstain', 'size': 0.0, 'confidence': 0.1, 'consensus_score': 0.0},
            '_thesis': f'Committee error: {error}'
        }

