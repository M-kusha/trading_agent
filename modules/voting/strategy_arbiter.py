from modules.core.mixins import SmartInfoBusStateMixin, SmartInfoBusTradingMixin
# ─────────────────────────────────────────────────────────────
# File: modules/voting/strategy_arbiter.py
# Enhanced Strategy Arbiter with InfoBus integration
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import copy
from typing import Any, Dict, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit
from utils.get_dir import _BASE_GATE, _smart_gate


class StrategyArbiter(Module, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    Enhanced strategy arbiter with InfoBus integration.
    Coordinates multiple trading experts through sophisticated voting mechanisms,
    applies smart gating, and provides comprehensive decision audit trails.
    """

    # REINFORCE parameters
    REINFORCE_LR: float = 0.001
    REINFORCE_LAMBDA: float = 0.95
    PRIOR_BLEND: float = 0.30

    def __init__(
        self,
        members: List[Module],
        init_weights: List[float],
        action_dim: int,
        adapt_rate: float = 0.01,
        consensus: Optional[Module] = None,
        collusion: Optional[Module] = None,
        horizon_aligner: Optional[Module] = None,
        min_confidence: float = 0.3,
        bootstrap_steps: int = 50,
        debug: bool = True,
        audit_log_size: int = 100,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 200),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_trading_state()
        
        # Validate inputs
        if len(init_weights) != len(members):
            raise ValueError(f"Weight count ({len(init_weights)}) must match member count ({len(members)})")
        
        # Core components
        self.members = members
        self.weights = np.asarray(init_weights, dtype=np.float32)
        self.action_dim = int(action_dim)
        self.adapt_rate = float(adapt_rate)
        self.min_confidence = float(min_confidence)
        self.bootstrap_steps = int(bootstrap_steps)
        
        # Sub-modules
        self.consensus = consensus
        self.collusion = collusion
        self.horizon_aligner = horizon_aligner
        
        # Market state
        self.curr_vol: float = 0.01
        self.market_regime = 'unknown'
        self.market_session = 'unknown'
        
        # REINFORCE learning
        self.last_alpha: Optional[np.ndarray] = None
        self._baseline = 0.0
        self._baseline_beta = 0.98
        
        # Decision tracking
        self._trace: List[Dict[str, Any]] = []
        self._log_size = audit_log_size
        self.decision_history = deque(maxlen=100)
        self.proposal_history = deque(maxlen=50)
        
        # Gate statistics
        self._gate_passes = 0
        self._gate_attempts = 0
        self._step_count = 0
        
        # Enhanced member tracking
        self.member_performance = defaultdict(lambda: {
            'proposals_made': 0,
            'successful_proposals': 0,
            'avg_confidence': 0.5,
            'recent_performance': deque(maxlen=20),
            'weight_evolution': deque(maxlen=50)
        })
        
        # Voting quality metrics
        self.voting_quality = {
            'avg_consensus': 0.5,
            'collusion_risk': 0.0,
            'gate_effectiveness': 0.5,
            'member_diversity': 0.5,
            'decision_confidence': 0.5
        }
        
        # Performance statistics
        self.arbiter_stats = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'weight_adaptations': 0,
            'consensus_failures': 0,
            'collusion_detected': 0,
            'gate_pass_rate': 0.0,
            'avg_proposal_quality': 0.5
        }
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "StrategyArbiter",
            "logs/voting/strategy_arbiter.log",
            max_lines=2000,
            operator_mode=debug
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("StrategyArbiter")
        
        # Log member details
        member_names = [getattr(m, '__class__', type(m)).__name__ for m in self.members]
        self.log_operator_info(
            "🏛️ Strategy Arbiter initialized",
            members=len(self.members),
            action_dim=self.action_dim,
            bootstrap_steps=self.bootstrap_steps,
            member_names=member_names[:5]  # First 5 for brevity
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset learning state
        self._baseline = 0.0
        self.last_alpha = None
        
        # Reset tracking
        self._gate_passes = 0
        self._gate_attempts = 0
        self._step_count = 0
        
        # Reset history
        self._trace.clear()
        self.decision_history.clear()
        self.proposal_history.clear()
        
        # Reset member tracking
        self.member_performance.clear()
        
        # Reset quality metrics
        self.voting_quality = {
            'avg_consensus': 0.5,
            'collusion_risk': 0.0,
            'gate_effectiveness': 0.5,
            'member_diversity': 0.5,
            'decision_confidence': 0.5
        }
        
        # Reset statistics
        self.arbiter_stats = {
            'total_decisions': 0,
            'successful_decisions': 0,
            'weight_adaptations': 0,
            'consensus_failures': 0,
            'collusion_detected': 0,
            'gate_pass_rate': 0.0,
            'avg_proposal_quality': 0.5
        }
        
        # Reset sub-modules
        if self.consensus:
            self.consensus.reset()
        if self.collusion:
            self.collusion.reset()
        if self.horizon_aligner:
            self.horizon_aligner.reset()
        
        self.log_operator_info("🔄 Strategy Arbiter reset - all state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - using standalone mode")
            self._process_legacy_step(**kwargs)
            return
        
        # Extract context and market data
        context = extract_standard_context(info_bus)
        market_data = self._extract_market_data_from_info_bus(info_bus)
        
        # Update market state
        self._update_market_state(context, market_data)
        
        # Analyze member performance
        self._analyze_member_performance(market_data)
        
        # Update voting quality metrics
        self._update_voting_quality_metrics()
        
        # Update InfoBus with arbiter state
        self._update_info_bus(info_bus)

    def _extract_market_data_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract market data for arbiter decision making"""
        
        data = {}
        
        try:
            # Extract market context
            market_context = info_bus.get('market_context', {})
            data['volatility'] = market_context.get('volatility', {})
            data['regime'] = market_context.get('regime', 'unknown')
            data['trend_strength'] = market_context.get('trend_strength', 0.5)
            
            # Extract recent performance
            recent_trades = info_bus.get('recent_trades', [])
            data['recent_trades'] = recent_trades
            
            if recent_trades:
                data['recent_pnl'] = [trade.get('pnl', 0) for trade in recent_trades[-10:]]
                data['recent_success_rate'] = sum(1 for trade in recent_trades[-10:] if trade.get('pnl', 0) > 0) / len(recent_trades[-10:])
            else:
                data['recent_pnl'] = []
                data['recent_success_rate'] = 0.5
            
            # Extract current positions for risk assessment
            positions = InfoBusExtractor.get_positions(info_bus)
            data['current_positions'] = positions
            data['total_exposure'] = sum(abs(pos.get('size', 0)) for pos in positions)
            
            # Extract consensus information
            consensus_data = info_bus.get('consensus', {})
            data['consensus_score'] = consensus_data.get('score', 0.5)
            data['consensus_quality'] = consensus_data.get('quality', 0.5)
            
        except Exception as e:
            self.log_operator_warning(f"Market data extraction failed: {e}")
            data = {
                'volatility': {},
                'regime': 'unknown',
                'trend_strength': 0.5,
                'recent_trades': [],
                'recent_pnl': [],
                'recent_success_rate': 0.5,
                'current_positions': [],
                'total_exposure': 0.0,
                'consensus_score': 0.5,
                'consensus_quality': 0.5
            }
        
        return data

    def _update_market_state(self, context: Dict[str, Any], market_data: Dict[str, Any]) -> None:
        """Update market state for decision making"""
        
        try:
            # Update regime and session
            old_regime = self.market_regime
            self.market_regime = context.get('regime', 'unknown')
            self.market_session = context.get('session', 'unknown')
            
            # Update volatility
            volatilities = market_data.get('volatility', {})
            if volatilities:
                self.curr_vol = max(0.001, np.mean(list(volatilities.values())))
            
            # Log regime changes
            if old_regime != self.market_regime and old_regime != 'unknown':
                self.log_operator_info(
                    f"📊 Market regime change: {old_regime} → {self.market_regime}",
                    volatility=f"{self.curr_vol:.3f}",
                    session=self.market_session
                )
            
        except Exception as e:
            self.log_operator_warning(f"Market state update failed: {e}")

    def propose(self, obs: Any) -> np.ndarray:
        """
        Enhanced proposal generation with comprehensive member coordination.
        
        Args:
            obs: Observation data for decision making
            
        Returns:
            Blended action proposal from all committee members
        """
        
        self._step_count += 1
        trace = {
            "step": self._step_count,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        try:
            # Collect proposals from all members
            proposals, confidences = self._collect_member_proposals(obs, trace)
            
            # Apply consensus analysis
            consensus_score = self._analyze_consensus(proposals, confidences, trace)
            
            # Apply collusion detection
            collusion_score = self._detect_collusion(proposals, trace)
            
            # Apply horizon alignment
            aligned_weights = self._apply_horizon_alignment(trace)
            
            # Compute weighted blend
            action = self._compute_weighted_blend(proposals, confidences, aligned_weights, trace)
            
            # Apply smart gate
            final_action = self._apply_smart_gate(action, proposals, confidences, trace)
            
            # Save trace and update statistics
            self._save_trace(trace)
            self._update_decision_statistics(trace, final_action)
            
            return final_action
            
        except Exception as e:
            self.log_operator_error(f"Proposal generation failed: {e}")
            trace["error"] = str(e)
            self._save_trace(trace)
            return np.zeros(self.action_dim, dtype=np.float32)

    def _collect_member_proposals(self, obs: Any, trace: Dict[str, Any]) -> Tuple[List[np.ndarray], List[float]]:
        """Collect proposals and confidences from all committee members"""
        
        proposals = []
        confidences = []
        member_status = []
        
        for i, member in enumerate(self.members):
            try:
                # Get action proposal
                if hasattr(member, 'propose_action'):
                    prop = member.propose_action(obs)
                elif hasattr(member, 'propose'):
                    prop = member.propose(obs)
                else:
                    # Fallback for modules without propose
                    prop = np.zeros(self.action_dim, dtype=np.float32)
                
                # Ensure proper shape and type
                prop = np.asarray(prop, dtype=np.float32).reshape(-1)
                if prop.size < self.action_dim:
                    prop = np.pad(prop, (0, self.action_dim - prop.size))
                elif prop.size > self.action_dim:
                    prop = prop[:self.action_dim]
                
                proposals.append(prop)
                
                # Get confidence
                if hasattr(member, 'confidence'):
                    conf = float(member.confidence(obs))
                else:
                    conf = 0.5  # Default confidence
                
                confidences.append(max(conf, self.min_confidence))
                member_status.append("success")
                
                # Track member performance
                self._track_member_proposal(i, prop, conf)
                
            except Exception as e:
                member_name = getattr(member, '__class__', type(member)).__name__
                self.log_operator_warning(f"Member {i} ({member_name}) failed: {e}")
                
                proposals.append(np.zeros(self.action_dim, dtype=np.float32))
                confidences.append(self.min_confidence)
                member_status.append(f"failed: {e}")
        
        # Update trace
        trace["member_count"] = len(self.members)
        trace["successful_members"] = sum(1 for status in member_status if status == "success")
        trace["raw_proposals"] = [prop.tolist() for prop in proposals]
        trace["confidences"] = confidences
        trace["member_status"] = member_status
        
        return proposals, confidences

    def _track_member_proposal(self, member_idx: int, proposal: np.ndarray, confidence: float) -> None:
        """Track individual member proposal for performance analysis"""
        
        try:
            perf_data = self.member_performance[member_idx]
            perf_data['proposals_made'] += 1
            
            # Update confidence tracking
            old_conf = perf_data['avg_confidence']
            count = perf_data['proposals_made']
            perf_data['avg_confidence'] = (old_conf * (count - 1) + confidence) / count
            
            # Store proposal quality metrics
            proposal_strength = np.linalg.norm(proposal)
            proposal_diversity = np.std(proposal) if len(proposal) > 1 else 0.0
            
            quality_score = min(1.0, (confidence + proposal_strength + proposal_diversity) / 3.0)
            perf_data['recent_performance'].append(quality_score)
            
        except Exception as e:
            self.log_operator_warning(f"Member {member_idx} tracking failed: {e}")

    def _analyze_consensus(self, proposals: List[np.ndarray], confidences: List[float], 
                          trace: Dict[str, Any]) -> float:
        """Analyze consensus among member proposals"""
        
        try:
            if self.consensus:
                consensus_score = self.consensus.compute_consensus(proposals, confidences)
                trace["consensus_score"] = float(consensus_score)
                
                # Track consensus in voting quality
                self.voting_quality['avg_consensus'] = (
                    self.voting_quality['avg_consensus'] * 0.9 + consensus_score * 0.1
                )
                
                return consensus_score
            else:
                # Fallback consensus calculation
                if len(proposals) < 2:
                    return 0.5
                
                # Simple pairwise agreement
                agreements = []
                for i in range(len(proposals)):
                    for j in range(i + 1, len(proposals)):
                        p1, p2 = proposals[i], proposals[j]
                        if np.linalg.norm(p1) > 0 and np.linalg.norm(p2) > 0:
                            similarity = np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2))
                            agreements.append((similarity + 1.0) / 2.0)
                
                consensus = np.mean(agreements) if agreements else 0.5
                trace["consensus_score"] = float(consensus)
                return consensus
                
        except Exception as e:
            self.log_operator_warning(f"Consensus analysis failed: {e}")
            trace["consensus_score"] = 0.5
            return 0.5

    def _detect_collusion(self, proposals: List[np.ndarray], trace: Dict[str, Any]) -> float:
        """Detect potential collusion among members"""
        
        try:
            if self.collusion:
                collusion_score = self.collusion.check_collusion(proposals)
                trace["collusion_score"] = float(collusion_score)
                
                # Apply penalties for collusion
                if collusion_score > 0.5:
                    self.arbiter_stats['collusion_detected'] += 1
                    # Reduce confidence of suspicious pairs
                    for pair in self.collusion.suspicious_pairs:
                        if len(pair) == 2:
                            i, j = pair
                            if i < len(self.weights) and j < len(self.weights):
                                self.weights[i] *= 0.9
                                self.weights[j] *= 0.9
                
                # Track in voting quality
                self.voting_quality['collusion_risk'] = collusion_score
                
                return collusion_score
            else:
                trace["collusion_score"] = 0.0
                return 0.0
                
        except Exception as e:
            self.log_operator_warning(f"Collusion detection failed: {e}")
            trace["collusion_score"] = 0.0
            return 0.0

    def _apply_horizon_alignment(self, trace: Dict[str, Any]) -> np.ndarray:
        """Apply time horizon alignment to weights"""
        
        try:
            if self.horizon_aligner:
                aligned_weights = self.horizon_aligner.apply(self.weights)
                trace["weight_alignment_applied"] = True
                trace["original_weights"] = self.weights.tolist()
                trace["aligned_weights"] = aligned_weights.tolist()
                return aligned_weights
            else:
                trace["weight_alignment_applied"] = False
                return self.weights.copy()
                
        except Exception as e:
            self.log_operator_warning(f"Horizon alignment failed: {e}")
            trace["weight_alignment_applied"] = False
            trace["alignment_error"] = str(e)
            return self.weights.copy()

    def _compute_weighted_blend(self, proposals: List[np.ndarray], confidences: List[float],
                               weights: np.ndarray, trace: Dict[str, Any]) -> np.ndarray:
        """Compute weighted blend of member proposals"""
        
        try:
            # Normalize weights and confidences
            w_norm = weights / (weights.sum() + 1e-12)
            c_norm = np.array(confidences) / (np.sum(confidences) + 1e-12)
            
            # Combined alpha weights
            alpha = w_norm * c_norm
            alpha = alpha / (alpha.sum() + 1e-12)
            
            # Store for REINFORCE learning
            self.last_alpha = alpha.copy()
            
            # Blend proposals
            action = np.zeros(self.action_dim, dtype=np.float32)
            for i, (prop, a) in enumerate(zip(proposals, alpha)):
                action += a * prop
            
            # Update trace
            trace["alpha_weights"] = alpha.tolist()
            trace["blended_action"] = action.tolist()
            trace["blend_method"] = "confidence_weighted"
            
            return action
            
        except Exception as e:
            self.log_operator_error(f"Weighted blend computation failed: {e}")
            trace["blend_error"] = str(e)
            return np.zeros(self.action_dim, dtype=np.float32)

    def _apply_smart_gate(self, action: np.ndarray, proposals: List[np.ndarray], 
                         confidences: List[float], trace: Dict[str, Any]) -> np.ndarray:
        """Apply enhanced smart gate with multiple criteria"""
        
        self._gate_attempts += 1
        
        try:
            # Calculate gate threshold
            if self._step_count < self.bootstrap_steps:
                gate = _BASE_GATE * 0.5  # Lenient during bootstrap
                trace["bootstrap_mode"] = True
            else:
                gate = _smart_gate(self.curr_vol, 0)
                trace["bootstrap_mode"] = False
            
            # Calculate signal strength
            signal_strength = np.abs(action).mean()
            
            # Enhanced gate criteria
            criteria_met = 0
            total_criteria = 5
            
            # Criterion 1: Signal strength
            if signal_strength >= gate:
                criteria_met += 1
                trace["signal_strength_pass"] = True
            else:
                trace["signal_strength_pass"] = False
            
            # Criterion 2: High confidence from multiple experts
            high_conf_count = sum(1 for c in confidences if c >= 0.6)
            high_conf_ratio = high_conf_count / len(confidences) if confidences else 0
            if high_conf_ratio >= 0.3:
                criteria_met += 1
                trace["confidence_pass"] = True
            else:
                trace["confidence_pass"] = False
            
            # Criterion 3: Directional agreement
            directions = []
            for proposal in proposals:
                if len(proposal) > 0 and abs(proposal[0]) > 1e-6:
                    directions.append(np.sign(proposal[0]))
            
            if directions:
                direction_agreement = abs(np.mean(directions))
                if direction_agreement >= 0.5:
                    criteria_met += 1
                    trace["direction_pass"] = True
                else:
                    trace["direction_pass"] = False
            else:
                trace["direction_pass"] = False
            
            # Criterion 4: Low collusion risk
            collusion_risk = trace.get("collusion_score", 0.0)
            if collusion_risk < 0.3:
                criteria_met += 1
                trace["collusion_pass"] = True
            else:
                trace["collusion_pass"] = False
            
            # Criterion 5: Reasonable consensus
            consensus_score = trace.get("consensus_score", 0.5)
            if consensus_score >= 0.4 or consensus_score <= 0.1:  # Very high or very low consensus both interesting
                criteria_met += 1
                trace["consensus_pass"] = True
            else:
                trace["consensus_pass"] = False
            
            # Gate decision
            min_criteria = 2 if self._step_count < self.bootstrap_steps else 3
            passed = criteria_met >= min_criteria
            
            if passed:
                self._gate_passes += 1
                final_action = action
                trace["gate_decision"] = "pass"
                trace["gate_reason"] = f"{criteria_met}/{total_criteria} criteria met"
            else:
                # Apply dampening instead of complete blocking
                final_action = action * 0.2
                trace["gate_decision"] = "dampened"
                trace["gate_reason"] = f"Only {criteria_met}/{total_criteria} criteria met"
            
            # Update gate statistics
            pass_rate = self._gate_passes / self._gate_attempts
            self.voting_quality['gate_effectiveness'] = pass_rate
            
            # Update trace
            trace["gate_threshold"] = float(gate)
            trace["signal_strength"] = float(signal_strength)
            trace["criteria_met"] = criteria_met
            trace["total_criteria"] = total_criteria
            trace["gate_pass_rate"] = pass_rate
            trace["final_action"] = final_action.tolist()
            
            # Periodic logging
            if self._gate_attempts % 25 == 0:
                self.log_operator_info(
                    f"🚪 Gate statistics: {self._gate_passes}/{self._gate_attempts} ({pass_rate:.1%})",
                    criteria_effectiveness=f"{criteria_met}/{total_criteria}",
                    regime=self.market_regime
                )
            
            return final_action
            
        except Exception as e:
            self.log_operator_error(f"Smart gate application failed: {e}")
            trace["gate_error"] = str(e)
            return action * 0.1  # Very conservative fallback

    def _save_trace(self, trace: Dict[str, Any]) -> None:
        """Save decision trace for debugging and analysis"""
        
        try:
            # Add metadata
            trace["regime"] = self.market_regime
            trace["session"] = self.market_session
            trace["volatility"] = self.curr_vol
            trace["weights"] = self.weights.tolist()
            
            # Store trace
            self._trace.append(trace)
            if len(self._trace) > self._log_size:
                self._trace = self._trace[-self._log_size:]
            
            # Store in decision history for analysis
            decision_summary = {
                'timestamp': trace['timestamp'],
                'step': trace['step'],
                'successful_members': trace.get('successful_members', 0),
                'consensus_score': trace.get('consensus_score', 0.5),
                'collusion_score': trace.get('collusion_score', 0.0),
                'gate_decision': trace.get('gate_decision', 'unknown'),
                'signal_strength': trace.get('signal_strength', 0.0)
            }
            self.decision_history.append(decision_summary)
            
        except Exception as e:
            self.log_operator_warning(f"Trace saving failed: {e}")

    def _update_decision_statistics(self, trace: Dict[str, Any], final_action: np.ndarray) -> None:
        """Update comprehensive decision statistics"""
        
        try:
            self.arbiter_stats['total_decisions'] += 1
            
            # Update quality metrics
            signal_strength = trace.get('signal_strength', 0.0)
            consensus_score = trace.get('consensus_score', 0.5)
            
            # Decision confidence (combination of signal strength and consensus)
            decision_confidence = (signal_strength + consensus_score) / 2.0
            self.voting_quality['decision_confidence'] = (
                self.voting_quality['decision_confidence'] * 0.9 + decision_confidence * 0.1
            )
            
            # Member diversity (based on proposal variance)
            proposals = trace.get('raw_proposals', [])
            if len(proposals) > 1:
                proposal_matrix = np.array(proposals)
                diversity = np.mean(np.std(proposal_matrix, axis=0))
                self.voting_quality['member_diversity'] = (
                    self.voting_quality['member_diversity'] * 0.9 + diversity * 0.1
                )
            
            # Update performance metrics
            self._update_performance_metric('total_decisions', self.arbiter_stats['total_decisions'])
            self._update_performance_metric('gate_pass_rate', self.voting_quality['gate_effectiveness'])
            self._update_performance_metric('avg_consensus', self.voting_quality['avg_consensus'])
            self._update_performance_metric('decision_confidence', self.voting_quality['decision_confidence'])
            
        except Exception as e:
            self.log_operator_warning(f"Statistics update failed: {e}")

    def _analyze_member_performance(self, market_data: Dict[str, Any]) -> None:
        """Analyze individual member performance and adapt weights"""
        
        try:
            recent_success_rate = market_data.get('recent_success_rate', 0.5)
            
            # Update member performance based on recent outcomes
            for member_idx, perf_data in self.member_performance.items():
                if len(perf_data['recent_performance']) > 5:
                    avg_quality = np.mean(list(perf_data['recent_performance']))
                    
                    # Record weight evolution
                    if member_idx < len(self.weights):
                        perf_data['weight_evolution'].append(self.weights[member_idx])
                        
                        # Simple adaptation based on performance
                        if avg_quality > 0.7 and recent_success_rate > 0.6:
                            self.weights[member_idx] = min(2.0, self.weights[member_idx] * 1.05)
                        elif avg_quality < 0.3 or recent_success_rate < 0.3:
                            self.weights[member_idx] = max(0.1, self.weights[member_idx] * 0.95)
            
            # Renormalize weights
            self.weights = self.weights / (self.weights.sum() + 1e-12)
            
        except Exception as e:
            self.log_operator_warning(f"Member performance analysis failed: {e}")

    def _update_voting_quality_metrics(self) -> None:
        """Update comprehensive voting quality metrics"""
        
        try:
            # Update arbiter statistics
            self.arbiter_stats['gate_pass_rate'] = self.voting_quality['gate_effectiveness']
            
            # Calculate proposal quality from recent member performance
            if self.member_performance:
                qualities = []
                for perf_data in self.member_performance.values():
                    if perf_data['recent_performance']:
                        qualities.append(np.mean(list(perf_data['recent_performance'])))
                
                if qualities:
                    self.arbiter_stats['avg_proposal_quality'] = np.mean(qualities)
            
        except Exception as e:
            self.log_operator_warning(f"Quality metrics update failed: {e}")

    def update_weights(self, reward: float) -> None:
        """Enhanced REINFORCE weight update with comprehensive tracking"""
        
        if self.last_alpha is None:
            return
        
        try:
            # Update baseline
            self._baseline = self._baseline_beta * self._baseline + (1 - self._baseline_beta) * reward
            
            # Calculate advantage
            advantage = reward - self._baseline
            
            # REINFORCE update
            grad = advantage * (self.last_alpha - self.weights)
            old_weights = self.weights.copy()
            self.weights += self.adapt_rate * grad
            
            # Ensure positive weights
            self.weights = np.maximum(self.weights, 0.01)
            
            # Normalize
            self.weights = self.weights / self.weights.sum()
            
            # Track adaptation
            weight_change = np.linalg.norm(self.weights - old_weights)
            if weight_change > 0.05:
                self.arbiter_stats['weight_adaptations'] += 1
                self.log_operator_info(
                    f"⚖️ Significant weight adaptation",
                    reward=f"{reward:+.3f}",
                    advantage=f"{advantage:+.3f}",
                    change=f"{weight_change:.3f}"
                )
            
            # Update success tracking if positive reward
            if reward > 0:
                self.arbiter_stats['successful_decisions'] += 1
            
        except Exception as e:
            self.log_operator_error(f"Weight update failed: {e}")

    def _process_legacy_step(self, **kwargs) -> None:
        """Process legacy step for backward compatibility"""
        
        reward = kwargs.get('reward', 0.0)
        if reward != 0:
            self.update_weights(reward)

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with comprehensive arbiter state"""
        
        # Add detailed module data
        InfoBusUpdater.add_module_data(info_bus, 'strategy_arbiter', {
            'weights': self.weights.tolist(),
            'last_alpha': self.last_alpha.tolist() if self.last_alpha is not None else None,
            'action_dim': self.action_dim,
            'member_count': len(self.members),
            'step_count': self._step_count,
            'market_state': {
                'regime': self.market_regime,
                'session': self.market_session,
                'volatility': self.curr_vol
            },
            'voting_quality': self.voting_quality.copy(),
            'statistics': self.arbiter_stats.copy(),
            'gate_stats': {
                'attempts': self._gate_attempts,
                'passes': self._gate_passes,
                'pass_rate': self._gate_passes / max(self._gate_attempts, 1),
                'bootstrap_mode': self._step_count < self.bootstrap_steps
            },
            'member_performance': {
                str(k): {
                    'proposals_made': v['proposals_made'],
                    'avg_confidence': v['avg_confidence'],
                    'recent_quality': np.mean(list(v['recent_performance'])) if v['recent_performance'] else 0.5
                } for k, v in self.member_performance.items()
            }
        })
        
        # Update main voting data in InfoBus
        if 'votes' not in info_bus:
            info_bus['votes'] = []
        
        # Add arbiter summary vote
        if self.last_alpha is not None:
            arbiter_vote = {
                'module': 'StrategyArbiter',
                'action': self.last_alpha.tolist(),
                'confidence': self.voting_quality['decision_confidence'],
                'consensus': self.voting_quality['avg_consensus'],
                'timestamp': datetime.datetime.now().isoformat()
            }
            info_bus['votes'].append(arbiter_vote)

    def get_arbiter_report(self) -> str:
        """Generate operator-friendly arbiter report"""
        
        # Decision quality assessment
        decision_conf = self.voting_quality['decision_confidence']
        if decision_conf > 0.8:
            quality_status = "✅ EXCELLENT"
        elif decision_conf > 0.6:
            quality_status = "⚡ GOOD"
        elif decision_conf > 0.4:
            quality_status = "⚠️ FAIR"
        else:
            quality_status = "🚨 POOR"
        
        # Gate effectiveness
        gate_rate = self.voting_quality['gate_effectiveness']
        if gate_rate > 0.7:
            gate_status = "🟢 EFFECTIVE"
        elif gate_rate > 0.4:
            gate_status = "🟡 MODERATE"
        else:
            gate_status = "🔴 RESTRICTIVE"
        
        # Top performing members
        member_lines = []
        for member_idx, perf_data in list(self.member_performance.items())[:5]:
            if member_idx < len(self.weights):
                weight = self.weights[member_idx]
                quality = np.mean(list(perf_data['recent_performance'])) if perf_data['recent_performance'] else 0.5
                confidence = perf_data['avg_confidence']
                
                if quality > 0.7:
                    emoji = "🌟"
                elif quality > 0.5:
                    emoji = "⚡"
                else:
                    emoji = "⚠️"
                
                member_lines.append(f"  {emoji} Member {member_idx}: Weight {weight:.3f}, Quality {quality:.1%}, Conf {confidence:.1%}")
        
        return f"""
🏛️ STRATEGY ARBITER
═══════════════════════════════════════
🎯 Decision Quality: {quality_status} ({decision_conf:.1%})
🚪 Gate Status: {gate_status} ({gate_rate:.1%})
📊 Consensus Level: {self.voting_quality['avg_consensus']:.1%}

⚖️ Committee Overview:
• Total Members: {len(self.members)}
• Action Dimensions: {self.action_dim}
• Current Step: {self._step_count}
• Bootstrap Mode: {'✅ Active' if self._step_count < self.bootstrap_steps else '❌ Complete'}

📊 Market Context:
• Regime: {self.market_regime.title()}
• Session: {self.market_session.title()}
• Volatility: {self.curr_vol:.2%}

🎯 Voting Quality Metrics:
• Decision Confidence: {self.voting_quality['decision_confidence']:.1%}
• Average Consensus: {self.voting_quality['avg_consensus']:.1%}
• Member Diversity: {self.voting_quality['member_diversity']:.1%}
• Collusion Risk: {self.voting_quality['collusion_risk']:.1%}
• Gate Effectiveness: {self.voting_quality['gate_effectiveness']:.1%}

📈 Performance Statistics:
• Total Decisions: {self.arbiter_stats['total_decisions']}
• Successful Decisions: {self.arbiter_stats['successful_decisions']}
• Success Rate: {(self.arbiter_stats['successful_decisions'] / max(self.arbiter_stats['total_decisions'], 1)):.1%}
• Weight Adaptations: {self.arbiter_stats['weight_adaptations']}
• Collusion Events: {self.arbiter_stats['collusion_detected']}

🚪 Gate Statistics:
• Gate Attempts: {self._gate_attempts}
• Gate Passes: {self._gate_passes}
• Pass Rate: {(self._gate_passes / max(self._gate_attempts, 1)):.1%}
• Baseline: {self._baseline:.3f}

👥 Top Performing Members:
{chr(10).join(member_lines) if member_lines else "  📭 No member performance data available"}

🔧 Configuration:
• Adapt Rate: {self.adapt_rate:.4f}
• Min Confidence: {self.min_confidence:.2f}
• Bootstrap Steps: {self.bootstrap_steps}
• REINFORCE Beta: {self._baseline_beta:.3f}

📊 Recent Activity:
• Decision History: {len(self.decision_history)} entries
• Trace Log: {len(self._trace)} entries
• Member Tracking: {len(self.member_performance)} members
        """

    def get_last_traces(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get recent decision traces for debugging"""
        return self._trace[-n:]

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostic information"""
        
        member_names = [getattr(m, '__class__', type(m)).__name__ for m in self.members]
        
        return {
            "members": member_names,
            "weights": self.weights.tolist(),
            "last_alpha": self.last_alpha.tolist() if self.last_alpha is not None else None,
            "voting_quality": self.voting_quality.copy(),
            "statistics": self.arbiter_stats.copy(),
            "gate_stats": {
                "attempts": self._gate_attempts,
                "passes": self._gate_passes,
                "pass_rate": self._gate_passes / max(self._gate_attempts, 1),
                "bootstrap": self._step_count < self.bootstrap_steps,
            },
            "market_state": {
                "regime": self.market_regime,
                "session": self.market_session,
                "volatility": self.curr_vol
            },
            "learning_state": {
                "baseline": float(self._baseline),
                "step_count": self._step_count,
                "adapt_rate": self.adapt_rate
            }
        }

    def get_observation_components(self) -> np.ndarray:
        """Return arbiter features for observation"""
        
        try:
            features = [
                float(self._gate_passes / max(self._gate_attempts, 1)),  # Pass rate
                float(self.curr_vol),
                float(self._baseline),
                float(self.voting_quality['avg_consensus']),
                float(self.voting_quality['decision_confidence'])
            ]
            
            # Add normalized weights
            features.extend(self.weights.tolist())
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            # Return safe defaults
            default_features = [0.5, 0.02, 0.0, 0.5, 0.5]
            default_features.extend([1.0 / len(self.members)] * len(self.members))
            return np.array(default_features, dtype=np.float32)

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, reward: float = 0.0, **kwargs) -> None:
        """Legacy step interface for backward compatibility"""
        if reward != 0:
            self.update_weights(reward)
        
        # Step sub-modules
        if self.consensus:
            self.consensus.step(**kwargs)
        if self.collusion:
            self.collusion.step(**kwargs)
        if self.horizon_aligner:
            self.horizon_aligner.step(**kwargs)

    def update_market_state(self, volatility: float) -> None:
        """Legacy market state update"""
        self.curr_vol = float(max(volatility, 0.001))