# ─────────────────────────────────────────────────────────────
# File: modules/memory/historical_replay_analyzer.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
from typing import List, Optional, Dict, Any
from collections import deque
import datetime
import random

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import TradingMixin, AnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class HistoricalReplayAnalyzer(Module, TradingMixin, AnalysisMixin):
    """
    Enhanced historical replay analyzer with infrastructure integration.
    Identifies and analyzes profitable trading sequences for learning and replay.
    """
    
    def __init__(self, interval: int = 10, bonus: float = 0.1, sequence_len: int = 5, 
                 debug: bool = True, genome: Optional[Dict[str, Any]] = None, **kwargs):
        
        self.profit_threshold = float(genome.get("profit_threshold", 10.0)) if genome else 10.0
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=200,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome, interval, bonus, sequence_len)
        
        # Enhanced state initialization
        self._initialize_module_state()
        
        self.log_operator_info(
            "Historical replay analyzer initialized",
            replay_interval=self.interval,
            base_bonus=f"{self.bonus:.3f}",
            sequence_length=self.sequence_len,
            pattern_tracking="enabled"
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], interval: int, bonus: float, sequence_len: int):
        """Initialize genome-based parameters"""
        if genome:
            self.interval = int(genome.get("interval", interval))
            self.bonus = float(genome.get("bonus", bonus))
            self.sequence_len = int(genome.get("sequence_len", sequence_len))
            self.profit_threshold = float(genome.get("profit_threshold", 10.0))
            self.pattern_sensitivity = float(genome.get("pattern_sensitivity", 1.0))
            self.replay_decay = float(genome.get("replay_decay", 0.9))
        else:
            self.interval = interval
            self.bonus = bonus
            self.sequence_len = sequence_len
            self.profit_threshold = 10.0
            self.pattern_sensitivity = 1.0
            self.replay_decay = 0.9

        # Store genome for evolution
        self.genome = {
            "interval": self.interval,
            "bonus": self.bonus,
            "sequence_len": self.sequence_len,
            "profit_threshold": self.profit_threshold,
            "pattern_sensitivity": self.pattern_sensitivity,
            "replay_decay": self.replay_decay
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_trading_state()
        self._initialize_analysis_state()
        
        # Replay analysis specific state
        self.episode_buffer = deque(maxlen=100)
        self.profitable_sequences = []
        self.sequence_patterns = {}  # pattern -> (count, avg_pnl, last_seen)
        self.current_sequence = []
        self.replay_bonus = 0.0
        self.best_sequence_pnl = 0.0
        
        # Enhanced tracking
        self._episode_count = 0
        self._pattern_evolution = deque(maxlen=50)
        self._sequence_quality_scores = deque(maxlen=100)
        self._replay_effectiveness = {}  # episode -> replay_result
        self._learning_curve = deque(maxlen=200)
        
        # Pattern analysis
        self._pattern_success_rates = {}
        self._pattern_market_conditions = {}
        self._adaptive_thresholds = {
            'min_profit': self.profit_threshold,
            'min_sequence_len': 3,
            'pattern_confidence': 0.6
        }

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_trading_state()
        self._reset_analysis_state()
        
        # Module-specific reset
        self.episode_buffer.clear()
        self.profitable_sequences.clear()
        self.sequence_patterns.clear()
        self.current_sequence.clear()
        self.replay_bonus = 0.0
        self.best_sequence_pnl = 0.0
        self._episode_count = 0
        self._pattern_evolution.clear()
        self._sequence_quality_scores.clear()
        self._replay_effectiveness.clear()
        self._learning_curve.clear()
        self._pattern_success_rates.clear()
        self._pattern_market_conditions.clear()
        
        # Reset adaptive thresholds
        self._adaptive_thresholds = {
            'min_profit': self.profit_threshold,
            'min_sequence_len': 3,
            'pattern_confidence': 0.6
        }

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract trading sequence data
        sequence_data = self._extract_sequence_data(info_bus, kwargs)
        
        # Process current trading sequence
        self._process_trading_sequence(sequence_data)
        
        # Update learning metrics
        self._update_learning_metrics(sequence_data)

    def _extract_sequence_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract sequence tracking data from InfoBus or kwargs"""
        
        # Try InfoBus first
        if info_bus:
            # Extract recent trades for sequence analysis
            recent_trades = info_bus.get('recent_trades', [])
            market_context = info_bus.get('market_context', {})
            step_idx = info_bus.get('step_idx', 0)
            
            # Get current action if available
            current_action = kwargs.get('action', np.zeros(2))
            
            # Extract features from InfoBus
            features = self._extract_features_from_info_bus(info_bus)
            
            return {
                'recent_trades': recent_trades,
                'current_action': current_action,
                'features': features,
                'market_context': market_context,
                'step_idx': step_idx,
                'regime': InfoBusExtractor.get_market_regime(info_bus),
                'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
                'session': InfoBusExtractor.get_session(info_bus),
                'source': 'info_bus'
            }
        
        # Try kwargs (backward compatibility)
        if "action" in kwargs and "features" in kwargs:
            return {
                'current_action': kwargs["action"],
                'features': kwargs["features"],
                'timestamp': kwargs.get("timestamp", self._step_count),
                'source': 'kwargs'
            }
        
        # Return minimal data if insufficient input
        return {'source': 'insufficient_data'}

    def _extract_features_from_info_bus(self, info_bus: InfoBus) -> np.ndarray:
        """Extract trading features from InfoBus"""
        
        # Combine various InfoBus data into feature vector
        features = []
        
        # Market context features
        market_context = info_bus.get('market_context', {})
        if 'volatility' in market_context:
            vol_data = market_context['volatility']
            if isinstance(vol_data, dict):
                features.extend(list(vol_data.values())[:3])  # First 3 instruments
            else:
                features.append(float(vol_data))
        else:
            features.extend([0.01, 0.01, 0.01])  # Default volatility
        
        # Risk features
        risk_data = info_bus.get('risk', {})
        features.extend([
            risk_data.get('current_drawdown', 0.0),
            risk_data.get('margin_used', 0.0) / max(risk_data.get('equity', 1.0), 1.0),
            len(info_bus.get('positions', []))
        ])
        
        # Price momentum features
        prices = info_bus.get('prices', {})
        if prices:
            price_values = list(prices.values())[:2]  # First 2 instruments
            features.extend(price_values)
        else:
            features.extend([1.0, 1.0])
        
        # Session and regime encoding
        session = InfoBusExtractor.get_session(info_bus)
        session_encoding = {'asian': 1, 'european': 2, 'american': 3, 'closed': 0}.get(session, 0)
        features.append(session_encoding)
        
        regime = InfoBusExtractor.get_market_regime(info_bus)
        regime_encoding = {'trending': 1, 'volatile': 2, 'ranging': 3, 'unknown': 0}.get(regime, 0)
        features.append(regime_encoding)
        
        return np.array(features, dtype=np.float32)

    def _process_trading_sequence(self, sequence_data: Dict[str, Any]):
        """Process current trading sequence with enhanced analytics"""
        
        if sequence_data.get('source') == 'insufficient_data':
            return
        
        try:
            # Build step data
            step_data = {
                "action": sequence_data.get('current_action', np.zeros(2)),
                "features": sequence_data.get('features', np.zeros(10)),
                "timestamp": sequence_data.get('step_idx', self._step_count),
                "market_context": sequence_data.get('market_context', {}),
                "regime": sequence_data.get('regime', 'unknown'),
                "volatility_level": sequence_data.get('volatility_level', 'medium')
            }
            
            # Add to current sequence
            self.current_sequence.append(step_data)
            
            # Maintain sequence length
            if len(self.current_sequence) > self.sequence_len:
                removed = self.current_sequence.pop(0)
                
            # Calculate sequence quality
            quality_score = self._calculate_sequence_quality(self.current_sequence)
            self._sequence_quality_scores.append(quality_score)
            
            # Update performance metrics
            self._update_performance_metric('sequence_length', len(self.current_sequence))
            self._update_performance_metric('sequence_quality', quality_score)
            
        except Exception as e:
            self.log_operator_error(f"Sequence processing failed: {e}")
            self._update_health_status("DEGRADED", f"Sequence processing failed: {e}")

    def _calculate_sequence_quality(self, sequence: List[Dict]) -> float:
        """Calculate quality score for current sequence"""
        
        if not sequence:
            return 0.0
        
        try:
            # Action consistency score
            actions = [step.get('action', np.zeros(2)) for step in sequence]
            if len(actions) >= 2:
                action_vars = [np.var([a[i] if len(a) > i else 0 for a in actions]) for i in range(2)]
                consistency = 1.0 / (1.0 + np.mean(action_vars))
            else:
                consistency = 0.5
            
            # Market condition stability
            regimes = [step.get('regime', 'unknown') for step in sequence]
            regime_stability = len(set(regimes)) / max(len(regimes), 1)
            regime_stability = 1.0 - regime_stability  # Higher is better
            
            # Feature diversity (good for learning)
            features = [step.get('features', np.zeros(10)) for step in sequence]
            if len(features) >= 2:
                feature_matrix = np.vstack(features)
                feature_diversity = np.mean(np.std(feature_matrix, axis=0))
                feature_diversity = min(1.0, feature_diversity)
            else:
                feature_diversity = 0.0
            
            # Combined quality score
            quality = (0.4 * consistency + 0.3 * regime_stability + 0.3 * feature_diversity)
            
            return float(np.clip(quality, 0.0, 1.0))
            
        except Exception as e:
            self.log_operator_warning(f"Quality calculation failed: {e}")
            return 0.5

    def _update_learning_metrics(self, sequence_data: Dict[str, Any]):
        """Update learning curve and effectiveness metrics"""
        
        # Track learning progression
        if len(self._sequence_quality_scores) > 0:
            avg_quality = np.mean(list(self._sequence_quality_scores)[-10:])
            self._learning_curve.append({
                'episode': self._episode_count,
                'avg_quality': avg_quality,
                'patterns_found': len(self.sequence_patterns),
                'best_pnl': self.best_sequence_pnl
            })

    def record_episode(self, data: Dict, actions: np.ndarray, pnl: float, info_bus: Optional[InfoBus] = None):
        """Enhanced episode recording with InfoBus integration"""
        
        try:
            self._episode_count += 1
            
            # Extract market context for episode analysis
            market_context = {}
            if info_bus:
                market_context = {
                    'regime': InfoBusExtractor.get_market_regime(info_bus),
                    'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
                    'session': InfoBusExtractor.get_session(info_bus),
                    'drawdown': InfoBusExtractor.get_drawdown_pct(info_bus),
                    'exposure': InfoBusExtractor.get_exposure_pct(info_bus)
                }
            
            # Create comprehensive episode record
            episode_data = {
                "data": data,
                "actions": actions,
                "pnl": pnl,
                "sequence": list(self.current_sequence),
                "episode": self._episode_count,
                "market_context": market_context,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.episode_buffer.append(episode_data)
            
            # Process trading results
            self._process_trading_results(pnl)
            
            # Analyze profitable sequences
            if pnl > self._adaptive_thresholds['min_profit']:
                self._analyze_profitable_sequence(episode_data)
            
            # Update adaptive thresholds
            self._update_adaptive_thresholds(pnl)
            
            # Log episode summary
            self.log_operator_info(
                f"Episode {self._episode_count} recorded",
                pnl=f"€{pnl:+.2f}",
                sequence_length=len(self.current_sequence),
                patterns_tracked=len(self.sequence_patterns),
                quality_score=f"{self._sequence_quality_scores[-1]:.3f}" if self._sequence_quality_scores else "N/A"
            )
            
            # Reset sequence for next episode
            self.current_sequence.clear()
            
        except Exception as e:
            self.log_operator_error(f"Episode recording failed: {e}")
            self._update_health_status("DEGRADED", f"Episode recording failed: {e}")

    def _analyze_profitable_sequence(self, episode_data: Dict):
        """Analyze profitable trading sequences for patterns"""
        
        try:
            sequence = episode_data['sequence']
            pnl = episode_data['pnl']
            
            if len(sequence) >= self._adaptive_thresholds['min_sequence_len']:
                # Extract sequence pattern
                pattern = self._extract_sequence_pattern(sequence)
                
                # Update pattern statistics
                current_time = datetime.datetime.now().isoformat()
                if pattern in self.sequence_patterns:
                    count, avg_pnl, _ = self.sequence_patterns[pattern]
                    new_avg = (avg_pnl * count + pnl) / (count + 1)
                    self.sequence_patterns[pattern] = (count + 1, new_avg, current_time)
                    
                    self.log_operator_info(
                        f"Pattern '{pattern}' updated",
                        count=count + 1,
                        avg_pnl=f"€{new_avg:.2f}",
                        latest_pnl=f"€{pnl:.2f}"
                    )
                else:
                    self.sequence_patterns[pattern] = (1, pnl, current_time)
                    self.log_operator_info(
                        f"New profitable pattern discovered",
                        pattern=pattern,
                        pnl=f"€{pnl:.2f}",
                        episode=self._episode_count
                    )
                
                # Store market conditions for this pattern
                if pattern not in self._pattern_market_conditions:
                    self._pattern_market_conditions[pattern] = []
                self._pattern_market_conditions[pattern].append(episode_data.get('market_context', {}))
                
                # Update pattern success rate
                self._update_pattern_success_rate(pattern, True)
                
                # Store exceptional sequences
                if pnl > self.best_sequence_pnl:
                    old_best = self.best_sequence_pnl
                    self.best_sequence_pnl = pnl
                    
                    profitable_record = {
                        "sequence": list(sequence),
                        "pnl": pnl,
                        "pattern": pattern,
                        "episode": self._episode_count,
                        "market_context": episode_data.get('market_context', {}),
                        "timestamp": current_time
                    }
                    self.profitable_sequences.append(profitable_record)
                    
                    # Keep only top 10 sequences
                    self.profitable_sequences.sort(key=lambda x: x["pnl"], reverse=True)
                    self.profitable_sequences = self.profitable_sequences[:10]
                    
                    self.log_operator_info(
                        f"New best sequence recorded",
                        new_best=f"€{pnl:.2f}",
                        previous_best=f"€{old_best:.2f}",
                        pattern=pattern
                    )
            else:
                self.log_operator_warning(
                    f"Profitable episode but sequence too short",
                    length=len(sequence),
                    required=self._adaptive_thresholds['min_sequence_len']
                )
                
        except Exception as e:
            self.log_operator_error(f"Profitable sequence analysis failed: {e}")

    def _extract_sequence_pattern(self, sequence: List[Dict]) -> str:
        """Extract actionable pattern from trading sequence"""
        
        try:
            if not sequence:
                return "empty"
            
            # Enhanced pattern extraction
            action_pattern = []
            regime_pattern = []
            volatility_pattern = []
            
            for step in sequence[-self.sequence_len:]:  # Recent steps
                # Action pattern
                action = step.get("action", np.array([0, 0]))
                if isinstance(action, np.ndarray) and len(action) >= 2:
                    direction = "B" if action[0] > 0.1 else "S" if action[0] < -0.1 else "H"
                    duration = "L" if len(action) > 1 and action[1] > 0.5 else "S"
                    action_pattern.append(f"{direction}{duration}")
                else:
                    action_pattern.append("H")
                
                # Market context pattern
                regime = step.get("regime", "unknown")
                regime_short = {"trending": "T", "volatile": "V", "ranging": "R"}.get(regime, "U")
                regime_pattern.append(regime_short)
                
                volatility = step.get("volatility_level", "medium")
                vol_short = {"low": "L", "medium": "M", "high": "H", "extreme": "X"}.get(volatility, "M")
                volatility_pattern.append(vol_short)
            
            # Combine patterns
            pattern = f"{'_'.join(action_pattern)}|{''.join(regime_pattern)}|{''.join(volatility_pattern)}"
            
            return pattern
            
        except Exception as e:
            self.log_operator_warning(f"Pattern extraction failed: {e}")
            return "error"

    def _update_pattern_success_rate(self, pattern: str, success: bool):
        """Update pattern success tracking"""
        
        if pattern not in self._pattern_success_rates:
            self._pattern_success_rates[pattern] = {'successes': 0, 'attempts': 0}
        
        self._pattern_success_rates[pattern]['attempts'] += 1
        if success:
            self._pattern_success_rates[pattern]['successes'] += 1

    def _update_adaptive_thresholds(self, pnl: float):
        """Update adaptive learning thresholds based on performance"""
        
        try:
            # Update minimum profit threshold based on recent performance
            if len(self._learning_curve) >= 10:
                recent_pnls = [pnl]  # Current episode
                recent_avg = np.mean(recent_pnls)
                
                # Adapt threshold to be slightly below average profitable episodes
                if recent_avg > 0:
                    new_threshold = max(5.0, recent_avg * 0.7)
                    if abs(new_threshold - self._adaptive_thresholds['min_profit']) > 2.0:
                        old_threshold = self._adaptive_thresholds['min_profit']
                        self._adaptive_thresholds['min_profit'] = new_threshold
                        
                        self.log_operator_info(
                            f"Adaptive threshold updated",
                            metric="min_profit",
                            old_value=f"€{old_threshold:.2f}",
                            new_value=f"€{new_threshold:.2f}"
                        )
            
            # Update pattern confidence threshold
            if len(self.sequence_patterns) > 5:
                pattern_performances = [avg_pnl for _, avg_pnl, _ in self.sequence_patterns.values()]
                if pattern_performances:
                    performance_std = np.std(pattern_performances)
                    new_confidence = min(0.8, 0.5 + performance_std / 50.0)
                    self._adaptive_thresholds['pattern_confidence'] = new_confidence
                    
        except Exception as e:
            self.log_operator_warning(f"Adaptive threshold update failed: {e}")

    def maybe_replay(self, episode: int) -> float:
        """Enhanced replay bonus calculation with market context awareness"""
        
        try:
            # Base interval replay
            base_bonus = self.bonus if episode % self.interval == 0 else 0.0
            
            # Pattern-based bonus enhancement
            pattern_bonus = 0.0
            if self.sequence_patterns:
                # Find most profitable and recent patterns
                recent_patterns = {
                    pattern: (count, avg_pnl) 
                    for pattern, (count, avg_pnl, last_seen) in self.sequence_patterns.items()
                    if count >= 2  # Minimum pattern frequency
                }
                
                if recent_patterns:
                    best_pattern = max(recent_patterns.items(), key=lambda x: x[1][1])
                    pattern_name, (count, avg_pnl) = best_pattern
                    
                    # Calculate pattern bonus
                    pattern_bonus = min(0.3, avg_pnl / 100.0) * self.pattern_sensitivity
                    
                    # Success rate adjustment
                    if pattern_name in self._pattern_success_rates:
                        success_rate = (self._pattern_success_rates[pattern_name]['successes'] / 
                                      max(self._pattern_success_rates[pattern_name]['attempts'], 1))
                        pattern_bonus *= success_rate
                    
                    self.log_operator_info(
                        f"Pattern-enhanced replay bonus",
                        episode=episode,
                        best_pattern=pattern_name,
                        pattern_pnl=f"€{avg_pnl:.2f}",
                        pattern_count=count,
                        base_bonus=f"{base_bonus:.3f}",
                        pattern_bonus=f"{pattern_bonus:.3f}"
                    )
            
            # Learning curve bonus
            learning_bonus = 0.0
            if len(self._learning_curve) >= 5:
                recent_learning = [lc['avg_quality'] for lc in list(self._learning_curve)[-5:]]
                learning_trend = np.polyfit(range(len(recent_learning)), recent_learning, 1)[0]
                if learning_trend > 0:  # Improving
                    learning_bonus = min(0.1, learning_trend * 2.0)
            
            # Combined replay bonus
            self.replay_bonus = base_bonus + pattern_bonus + learning_bonus
            
            # Apply decay to historical bonus
            self.replay_bonus *= (self.replay_decay ** (episode % 100))
            
            # Track replay effectiveness
            self._replay_effectiveness[episode] = {
                'base_bonus': base_bonus,
                'pattern_bonus': pattern_bonus,
                'learning_bonus': learning_bonus,
                'total_bonus': self.replay_bonus
            }
            
            # Update performance metrics
            self._update_performance_metric('replay_bonus', self.replay_bonus)
            self._update_performance_metric('pattern_bonus', pattern_bonus)
            
            return self.replay_bonus
            
        except Exception as e:
            self.log_operator_error(f"Replay bonus calculation failed: {e}")
            return 0.0

    def get_best_sequence_for_replay(self) -> Optional[List[Dict]]:
        """Get the most profitable sequence with market context"""
        
        try:
            if not self.profitable_sequences:
                return None
            
            # Select best sequence considering recency and profitability
            scored_sequences = []
            
            for seq_data in self.profitable_sequences:
                pnl = seq_data['pnl']
                episode = seq_data['episode']
                
                # Recency score (more recent = better)
                recency = 1.0 - ((self._episode_count - episode) / max(self._episode_count, 1))
                
                # Combined score
                total_score = 0.7 * (pnl / 100.0) + 0.3 * recency
                scored_sequences.append((total_score, seq_data))
            
            # Get best sequence
            best_score, best_seq_data = max(scored_sequences, key=lambda x: x[0])
            best_sequence = best_seq_data["sequence"]
            
            self.log_operator_info(
                f"Best sequence selected for replay",
                pnl=f"€{best_seq_data['pnl']:.2f}",
                episode=best_seq_data['episode'],
                length=len(best_sequence),
                score=f"{best_score:.3f}",
                pattern=best_seq_data.get('pattern', 'unknown')
            )
            
            return best_sequence
            
        except Exception as e:
            self.log_operator_error(f"Best sequence selection failed: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with comprehensive metrics"""
        
        try:
            # Base replay metrics
            n_patterns = float(len(self.sequence_patterns))
            avg_pattern_pnl = 0.0
            pattern_diversity = 0.0
            
            if self.sequence_patterns:
                pattern_pnls = [avg_pnl for _, avg_pnl, _ in self.sequence_patterns.values()]
                avg_pattern_pnl = np.mean(pattern_pnls)
                pattern_diversity = np.std(pattern_pnls) if len(pattern_pnls) > 1 else 0.0
            
            # Learning metrics
            learning_trend = 0.0
            if len(self._learning_curve) >= 3:
                qualities = [lc['avg_quality'] for lc in list(self._learning_curve)[-3:]]
                learning_trend = np.polyfit(range(len(qualities)), qualities, 1)[0]
            
            # Pattern success rate
            avg_success_rate = 0.0
            if self._pattern_success_rates:
                success_rates = [
                    stats['successes'] / max(stats['attempts'], 1) 
                    for stats in self._pattern_success_rates.values()
                ]
                avg_success_rate = np.mean(success_rates)
            
            # Sequence quality
            current_quality = float(self._sequence_quality_scores[-1]) if self._sequence_quality_scores else 0.0
            
            # Enhanced observation vector
            observation = np.array([
                self.replay_bonus,
                n_patterns,
                avg_pattern_pnl / 100.0,  # Normalized
                self.best_sequence_pnl / 100.0,  # Normalized
                pattern_diversity / 100.0,  # Normalized
                learning_trend,
                avg_success_rate,
                current_quality,
                float(len(self.current_sequence)) / self.sequence_len,  # Normalized
                float(self._episode_count) / 1000.0  # Normalized episode count
            ], dtype=np.float32)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(10, dtype=np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose replay-informed actions"""
        
        # This module provides learning insights rather than direct trading actions
        action_dim = 2
        if hasattr(obs, 'shape') and len(obs.shape) > 0:
            action_dim = obs.shape[0]
        
        # Return influence based on pattern confidence
        if self.sequence_patterns:
            # Use most successful pattern to influence action
            best_pattern = max(self.sequence_patterns.items(), key=lambda x: x[1][1])
            pattern_confidence = best_pattern[1][1] / 100.0  # Normalize
            
            # Extract direction from pattern name
            pattern_name = best_pattern[0]
            if 'B' in pattern_name:  # Buy patterns
                influence = np.full(action_dim, pattern_confidence * 0.3)
            elif 'S' in pattern_name:  # Sell patterns  
                influence = np.full(action_dim, -pattern_confidence * 0.3)
            else:
                influence = np.zeros(action_dim)
                
            return influence.astype(np.float32)
        
        return np.zeros(action_dim, dtype=np.float32)

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence in replay recommendations"""
        
        base_confidence = 0.5
        
        # Confidence from pattern count and success
        if len(self.sequence_patterns) > 0:
            pattern_confidence = min(0.3, len(self.sequence_patterns) / 10.0)
            base_confidence += pattern_confidence
        
        # Confidence from success rates
        if self._pattern_success_rates:
            avg_success_rate = np.mean([
                stats['successes'] / max(stats['attempts'], 1)
                for stats in self._pattern_success_rates.values()
            ])
            base_confidence += avg_success_rate * 0.2
        
        # Confidence from learning trend
        if len(self._learning_curve) >= 3:
            qualities = [lc['avg_quality'] for lc in list(self._learning_curve)[-3:]]
            if qualities:
                trend = np.polyfit(range(len(qualities)), qualities, 1)[0]
                if trend > 0:
                    base_confidence += min(0.1, trend)
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with validation"""
        self.interval = int(np.clip(genome.get("interval", self.interval), 5, 50))
        self.bonus = float(np.clip(genome.get("bonus", self.bonus), 0.0, 0.5))
        self.sequence_len = int(np.clip(genome.get("sequence_len", self.sequence_len), 3, 15))
        self.profit_threshold = float(np.clip(genome.get("profit_threshold", self.profit_threshold), 1.0, 50.0))
        self.pattern_sensitivity = float(np.clip(genome.get("pattern_sensitivity", self.pattern_sensitivity), 0.1, 2.0))
        self.replay_decay = float(np.clip(genome.get("replay_decay", self.replay_decay), 0.8, 1.0))
        
        self.genome = {
            "interval": self.interval,
            "bonus": self.bonus,
            "sequence_len": self.sequence_len,
            "profit_threshold": self.profit_threshold,
            "pattern_sensitivity": self.pattern_sensitivity,
            "replay_decay": self.replay_decay
        }
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance tracking"""
        g = self.genome.copy()
        mutations = []
        
        if np.random.rand() < mutation_rate:
            old_val = g["interval"]
            g["interval"] = int(np.clip(self.interval + np.random.randint(-3, 4), 5, 50))
            mutations.append(f"interval: {old_val} → {g['interval']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["bonus"]
            g["bonus"] = float(np.clip(self.bonus + np.random.uniform(-0.05, 0.05), 0.0, 0.5))
            mutations.append(f"bonus: {old_val:.3f} → {g['bonus']:.3f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["sequence_len"]
            g["sequence_len"] = int(np.clip(self.sequence_len + np.random.randint(-1, 2), 3, 15))
            mutations.append(f"sequence_len: {old_val} → {g['sequence_len']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["pattern_sensitivity"]
            g["pattern_sensitivity"] = float(np.clip(self.pattern_sensitivity + np.random.uniform(-0.1, 0.1), 0.1, 2.0))
            mutations.append(f"sensitivity: {old_val:.2f} → {g['pattern_sensitivity']:.2f}")
        
        if mutations:
            self.log_operator_info(f"Replay analyzer mutation applied", changes=", ".join(mutations))
            
        self.set_genome(g)
        
    def crossover(self, other: "HistoricalReplayAnalyzer") -> "HistoricalReplayAnalyzer":
        """Enhanced crossover with compatibility checking"""
        if not isinstance(other, HistoricalReplayAnalyzer):
            self.log_operator_warning("Crossover with incompatible type")
            return self
            
        new_g = {k: random.choice([self.genome[k], other.genome[k]]) for k in self.genome}
        return HistoricalReplayAnalyzer(genome=new_g, debug=self.config.debug)

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check sequence bounds
            if len(self.current_sequence) > self.sequence_len + 5:
                return False
                
            # Check pattern data validity
            for pattern, (count, avg_pnl, _) in self.sequence_patterns.items():
                if count <= 0 or not np.isfinite(avg_pnl):
                    return False
            
            # Check profitable sequences
            if self.profitable_sequences:
                for seq_data in self.profitable_sequences:
                    if not isinstance(seq_data.get('pnl'), (int, float)):
                        return False
                        
            # Check episode count consistency
            if self._episode_count < 0:
                return False
                
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        replay_details = {
            'replay_info': {
                'episodes_processed': self._episode_count,
                'patterns_discovered': len(self.sequence_patterns),
                'profitable_sequences': len(self.profitable_sequences),
                'best_sequence_pnl': self.best_sequence_pnl,
                'current_replay_bonus': self.replay_bonus
            },
            'learning_info': {
                'current_sequence_length': len(self.current_sequence),
                'sequence_quality': float(self._sequence_quality_scores[-1]) if self._sequence_quality_scores else 0.0,
                'learning_curve_points': len(self._learning_curve),
                'pattern_success_rates': len(self._pattern_success_rates)
            },
            'performance_info': {
                'adaptive_profit_threshold': self._adaptive_thresholds['min_profit'],
                'pattern_confidence_threshold': self._adaptive_thresholds['pattern_confidence'],
                'replay_effectiveness_tracked': len(self._replay_effectiveness)
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(replay_details)
            return base_details
        
        return replay_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        return {
            "episode_buffer": list(self.episode_buffer)[-20:],  # Keep recent only
            "profitable_sequences": self.profitable_sequences.copy(),
            "sequence_patterns": dict(self.sequence_patterns),
            "current_sequence": list(self.current_sequence),
            "replay_bonus": self.replay_bonus,
            "best_sequence_pnl": self.best_sequence_pnl,
            "genome": self.genome.copy(),
            "episode_count": self._episode_count,
            "pattern_success_rates": dict(self._pattern_success_rates),
            "adaptive_thresholds": self._adaptive_thresholds.copy(),
            "learning_curve": list(self._learning_curve)[-50:],  # Keep recent only
            "sequence_quality_scores": list(self._sequence_quality_scores)[-50:],
            "replay_effectiveness": dict(list(self._replay_effectiveness.items())[-20:])  # Keep recent only
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        self.episode_buffer = deque(module_state.get("episode_buffer", []), maxlen=100)
        self.profitable_sequences = module_state.get("profitable_sequences", [])
        self.sequence_patterns = module_state.get("sequence_patterns", {})
        self.current_sequence = module_state.get("current_sequence", [])
        self.replay_bonus = module_state.get("replay_bonus", 0.0)
        self.best_sequence_pnl = module_state.get("best_sequence_pnl", 0.0)
        self.set_genome(module_state.get("genome", self.genome))
        self._episode_count = module_state.get("episode_count", 0)
        self._pattern_success_rates = module_state.get("pattern_success_rates", {})
        self._adaptive_thresholds = module_state.get("adaptive_thresholds", {
            'min_profit': self.profit_threshold,
            'min_sequence_len': 3,
            'pattern_confidence': 0.6
        })
        self._learning_curve = deque(module_state.get("learning_curve", []), maxlen=200)
        self._sequence_quality_scores = deque(module_state.get("sequence_quality_scores", []), maxlen=100)
        self._replay_effectiveness = module_state.get("replay_effectiveness", {})

    def get_replay_analysis_report(self) -> str:
        """Generate operator-friendly replay analysis report"""
        
        # Best pattern info
        best_pattern_info = "None"
        if self.sequence_patterns:
            best_pattern = max(self.sequence_patterns.items(), key=lambda x: x[1][1])
            pattern_name, (count, avg_pnl, _) = best_pattern
            best_pattern_info = f"{pattern_name} (€{avg_pnl:.2f}, {count}x)"
        
        # Learning trend
        learning_status = "📊 Analyzing"
        if len(self._learning_curve) >= 5:
            recent_qualities = [lc['avg_quality'] for lc in list(self._learning_curve)[-5:]]
            trend = np.polyfit(range(len(recent_qualities)), recent_qualities, 1)[0]
            if trend > 0.01:
                learning_status = "📈 Improving"
            elif trend < -0.01:
                learning_status = "📉 Declining"
            else:
                learning_status = "➡️ Stable"
        
        return f"""
🔄 HISTORICAL REPLAY ANALYZER
═══════════════════════════════════════
📊 Episodes Processed: {self._episode_count}
🎯 Patterns Discovered: {len(self.sequence_patterns)}
💰 Best Sequence: €{self.best_sequence_pnl:.2f}
🔄 Current Replay Bonus: {self.replay_bonus:.3f}

🧠 LEARNING STATUS
• Learning Trend: {learning_status}
• Current Sequence Quality: {float(self._sequence_quality_scores[-1]):.3f if self._sequence_quality_scores else 0:.3f}
• Sequence Length: {len(self.current_sequence)}/{self.sequence_len}

🎯 PATTERN ANALYSIS
• Best Pattern: {best_pattern_info}
• Pattern Success Rate: {np.mean([s['successes']/max(s['attempts'],1) for s in self._pattern_success_rates.values()]):.1%if self._pattern_success_rates else 0:.1%}
• Profitable Sequences: {len(self.profitable_sequences)}/10

🔧 ADAPTIVE PARAMETERS
• Profit Threshold: €{self._adaptive_thresholds['min_profit']:.2f}
• Pattern Confidence: {self._adaptive_thresholds['pattern_confidence']:.3f}
• Replay Interval: {self.interval} episodes
• Pattern Sensitivity: {self.pattern_sensitivity:.2f}

📈 PERFORMANCE METRICS
• Replay Effectiveness: {len(self._replay_effectiveness)} episodes tracked
• Learning Curve Points: {len(self._learning_curve)}
• Quality Score History: {len(self._sequence_quality_scores)} points
        """

    # Maintain backward compatibility
    def step(self, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, **kwargs)

    def get_state(self):
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)