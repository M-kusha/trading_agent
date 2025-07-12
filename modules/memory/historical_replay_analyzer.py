# ─────────────────────────────────────────────────────────────
# File: modules/memory/historical_replay_analyzer.py
# 🚀 PRODUCTION-READY Historical Replay Analysis System
# Advanced sequence analysis with SmartInfoBus integration
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class ReplayConfig:
    """Configuration for Historical Replay Analyzer"""
    interval: int = 10
    bonus: float = 0.1
    sequence_len: int = 5
    profit_threshold: float = 10.0
    pattern_sensitivity: float = 1.0
    replay_decay: float = 0.9
    
    # Performance thresholds
    max_processing_time_ms: float = 200
    circuit_breaker_threshold: int = 3
    min_sequence_quality: float = 0.6
    
    # Analysis parameters
    max_sequences: int = 100
    lookback_episodes: int = 50
    pattern_confidence_threshold: float = 0.7


@module(
    name="HistoricalReplayAnalyzer",
    version="3.0.0",
    category="memory",
    provides=["replay_sequences", "pattern_analysis", "sequence_quality", "learning_progress"],
    requires=["trades", "actions", "market_data", "episode_data"],
    description="Advanced historical sequence analysis with pattern recognition and replay optimization",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class HistoricalReplayAnalyzer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced historical replay analyzer with SmartInfoBus integration.
    Identifies profitable trading sequences and patterns for learning optimization.
    """

    def __init__(self, 
                 config: Optional[ReplayConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        self.config = config or ReplayConfig()
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize replay analysis state
        self._initialize_replay_state()
        
        self.logger.info(
            format_operator_message(
                "🎭", "HISTORICAL_REPLAY_ANALYZER_INITIALIZED",
                details=f"Sequence length: {self.config.sequence_len}, Profit threshold: {self.config.profit_threshold}",
                result="Historical pattern analysis ready",
                context="memory_analysis"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for replay analysis"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="HistoricalReplayAnalyzer", 
            log_path="logs/replay_analysis.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("HistoricalReplayAnalyzer", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for analysis operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        self._start_monitoring()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters"""
        if genome:
            self.genome = {
                "interval": int(genome.get("interval", self.config.interval)),
                "bonus": float(genome.get("bonus", self.config.bonus)),
                "sequence_len": int(genome.get("sequence_len", self.config.sequence_len)),
                "profit_threshold": float(genome.get("profit_threshold", self.config.profit_threshold)),
                "pattern_sensitivity": float(genome.get("pattern_sensitivity", self.config.pattern_sensitivity)),
                "replay_decay": float(genome.get("replay_decay", self.config.replay_decay))
            }
        else:
            self.genome = {
                "interval": self.config.interval,
                "bonus": self.config.bonus,
                "sequence_len": self.config.sequence_len,
                "profit_threshold": self.config.profit_threshold,
                "pattern_sensitivity": self.config.pattern_sensitivity,
                "replay_decay": self.config.replay_decay
            }

    def _initialize_replay_state(self):
        """Initialize replay analysis state"""
        # Core replay data
        self.episode_buffer = deque(maxlen=self.config.lookback_episodes)
        self.profitable_sequences = []
        self.sequence_patterns = {}  # pattern -> (count, avg_pnl, last_seen, confidence)
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
            'min_profit': self.genome["profit_threshold"],
            'min_sequence_len': 3,
            'pattern_confidence': self.config.pattern_confidence_threshold
        }
        
        # Performance analytics
        self._analysis_performance = {
            'sequences_analyzed': 0,
            'patterns_identified': 0,
            'successful_replays': 0,
            'total_replay_bonus': 0.0
        }

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_replay_health()
                    self._analyze_pattern_effectiveness()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module"""
        try:
            # Set initial replay status in SmartInfoBus
            initial_status = {
                "sequences_stored": 0,
                "patterns_identified": 0,
                "best_sequence_pnl": 0.0,
                "replay_bonus": 0.0
            }
            
            self.smart_bus.set(
                'replay_sequences',
                initial_status,
                module='HistoricalReplayAnalyzer',
                thesis="Initial historical replay analysis status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process historical replay analysis"""
        start_time = time.time()
        
        try:
            # Extract sequence data
            sequence_data = await self._extract_sequence_data(**inputs)
            
            if not sequence_data:
                return await self._handle_no_data_fallback()
            
            # Process current trading sequence
            sequence_result = await self._process_trading_sequence(sequence_data)
            
            # Analyze patterns if episode completed
            if sequence_data.get('episode_completed', False):
                pattern_result = await self._analyze_episode_patterns(sequence_data)
                sequence_result.update(pattern_result)
            
            # Generate replay recommendations
            replay_result = await self._generate_replay_recommendations()
            sequence_result.update(replay_result)
            
            # Generate thesis
            thesis = await self._generate_replay_thesis(sequence_data, sequence_result)
            
            # Update SmartInfoBus
            await self._update_replay_smart_bus(sequence_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return sequence_result
            
        except Exception as e:
            return await self._handle_replay_error(e, start_time)

    async def _extract_sequence_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract sequence data from SmartInfoBus"""
        try:
            # Get recent trades
            trades = self.smart_bus.get('trades', 'HistoricalReplayAnalyzer') or []
            
            # Get actions
            actions = self.smart_bus.get('actions', 'HistoricalReplayAnalyzer') or []
            
            # Get market data
            market_data = self.smart_bus.get('market_data', 'HistoricalReplayAnalyzer') or {}
            
            # Get episode data
            episode_data = self.smart_bus.get('episode_data', 'HistoricalReplayAnalyzer') or {}
            
            # Get current action if provided
            current_action = inputs.get('action', np.zeros(2))
            
            return {
                'trades': trades,
                'actions': actions,
                'market_data': market_data,
                'episode_data': episode_data,
                'current_action': current_action,
                'timestamp': datetime.now().isoformat(),
                'episode_completed': inputs.get('episode_completed', False)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract sequence data: {e}")
            return None

    async def _process_trading_sequence(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process current trading sequence"""
        try:
            # Add current step to sequence
            if sequence_data.get('current_action') is not None:
                current_step = {
                    'action': sequence_data['current_action'],
                    'timestamp': time.time(),
                    'market_context': sequence_data.get('market_data', {})
                }
                self.current_sequence.append(current_step)
            
            # Limit sequence length
            if len(self.current_sequence) > self.genome["sequence_len"]:
                self.current_sequence = self.current_sequence[-self.genome["sequence_len"]:]
            
            # Calculate sequence quality
            sequence_quality = self._calculate_sequence_quality(self.current_sequence)
            self._sequence_quality_scores.append(sequence_quality)
            
            # Update analysis performance
            self._analysis_performance['sequences_analyzed'] += 1
            
            return {
                'current_sequence_length': len(self.current_sequence),
                'sequence_quality': sequence_quality,
                'sequences_processed': self._analysis_performance['sequences_analyzed']
            }
            
        except Exception as e:
            self.logger.error(f"Sequence processing failed: {e}")
            return self._create_fallback_response("sequence processing failed")

    def _calculate_sequence_quality(self, sequence: List[Dict[str, Any]]) -> float:
        """Calculate quality score of a trading sequence"""
        if not sequence or len(sequence) < 2:
            return 0.0
        
        try:
            # Check action consistency
            actions = [step.get('action', np.zeros(2)) for step in sequence]
            
            # Calculate action variance (lower is more consistent)
            action_variance = np.var([np.linalg.norm(action) for action in actions])
            consistency_score = max(0, 1.0 - action_variance)
            
            # Check temporal spacing
            timestamps = [step.get('timestamp', 0) for step in sequence]
            time_diffs = np.diff(timestamps)
            temporal_score = 1.0 if len(time_diffs) == 0 else max(0, 1.0 - np.std(time_diffs) / 100)
            
            # Market context consistency
            context_score = 0.8  # Default if no market context
            
            # Combined quality score
            quality = (consistency_score * 0.4 + temporal_score * 0.3 + context_score * 0.3)
            
            return float(max(0.0, min(1.0, quality)))
            
        except Exception as e:
            self.logger.error(f"Quality calculation failed: {e}")
            return 0.5

    async def _analyze_episode_patterns(self, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze patterns in completed episode"""
        try:
            episode_data = sequence_data.get('episode_data', {})
            episode_pnl = episode_data.get('pnl', 0.0)
            
            # Store episode in buffer
            episode_record = {
                'sequence': self.current_sequence.copy(),
                'pnl': episode_pnl,
                'timestamp': time.time(),
                'market_conditions': sequence_data.get('market_data', {})
            }
            
            self.episode_buffer.append(episode_record)
            self._episode_count += 1
            
            # Analyze if profitable
            if episode_pnl > self.genome["profit_threshold"]:
                await self._analyze_profitable_sequence(episode_record)
            
            # Extract and update patterns
            pattern = self._extract_sequence_pattern(self.current_sequence)
            if pattern:
                self._update_pattern_data(pattern, episode_pnl)
            
            # Reset current sequence for next episode
            self.current_sequence = []
            
            return {
                'episode_analyzed': True,
                'episode_pnl': episode_pnl,
                'pattern_extracted': pattern is not None,
                'profitable_sequence': episode_pnl > self.genome["profit_threshold"]
            }
            
        except Exception as e:
            self.logger.error(f"Episode pattern analysis failed: {e}")
            return {'episode_analyzed': False, 'error': str(e)}

    async def _analyze_profitable_sequence(self, episode_record: Dict[str, Any]):
        """Analyze profitable sequence for learning"""
        try:
            sequence = episode_record['sequence']
            pnl = episode_record['pnl']
            
            # Add to profitable sequences
            sequence_entry = {
                'sequence': sequence,
                'pnl': pnl,
                'timestamp': episode_record['timestamp'],
                'market_conditions': episode_record['market_conditions'],
                'quality_score': self._calculate_sequence_quality(sequence)
            }
            
            self.profitable_sequences.append(sequence_entry)
            
            # Keep only best sequences
            if len(self.profitable_sequences) > self.config.max_sequences:
                self.profitable_sequences.sort(key=lambda x: x['pnl'], reverse=True)
                self.profitable_sequences = self.profitable_sequences[:self.config.max_sequences]
            
            # Update best sequence
            if pnl > self.best_sequence_pnl:
                self.best_sequence_pnl = pnl
                
            self.logger.info(
                format_operator_message(
                    "💰", "PROFITABLE_SEQUENCE_IDENTIFIED",
                    pnl=f"{pnl:.2f}",
                    sequence_length=len(sequence),
                    quality_score=f"{sequence_entry['quality_score']:.3f}",
                    context="pattern_learning"
                )
            )
            
        except Exception as e:
            self.logger.error(f"Profitable sequence analysis failed: {e}")

    def _extract_sequence_pattern(self, sequence: List[Dict[str, Any]]) -> Optional[str]:
        """Extract pattern signature from sequence"""
        if not sequence or len(sequence) < 3:
            return None
        
        try:
            # Extract action directions
            directions = []
            for step in sequence:
                action = step.get('action', np.zeros(2))
                if len(action) >= 2:
                    # Discretize actions
                    direction = 'H'  # Hold
                    if action[0] > 0.1:
                        direction = 'L'  # Long
                    elif action[0] < -0.1:
                        direction = 'S'  # Short
                    directions.append(direction)
            
            if len(directions) >= 3:
                return ''.join(directions)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Pattern extraction failed: {e}")
            return None

    def _update_pattern_data(self, pattern: str, pnl: float):
        """Update pattern tracking data"""
        try:
            if pattern not in self.sequence_patterns:
                self.sequence_patterns[pattern] = {
                    'count': 0,
                    'total_pnl': 0.0,
                    'avg_pnl': 0.0,
                    'last_seen': time.time(),
                    'confidence': 0.0
                }
            
            pattern_data = self.sequence_patterns[pattern]
            pattern_data['count'] += 1
            pattern_data['total_pnl'] += pnl
            pattern_data['avg_pnl'] = pattern_data['total_pnl'] / pattern_data['count']
            pattern_data['last_seen'] = time.time()
            
            # Calculate confidence based on sample size and consistency
            confidence = min(1.0, pattern_data['count'] / 10)  # Max confidence at 10 samples
            if pattern_data['count'] > 1:
                # Adjust for consistency
                consistency = 1.0 - abs(pnl - pattern_data['avg_pnl']) / max(abs(pattern_data['avg_pnl']), 1.0)
                confidence *= consistency
            
            pattern_data['confidence'] = confidence
            
            # Track pattern evolution
            self._pattern_evolution.append({
                'pattern': pattern,
                'pnl': pnl,
                'avg_pnl': pattern_data['avg_pnl'],
                'confidence': confidence,
                'timestamp': time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Pattern data update failed: {e}")

    async def _generate_replay_recommendations(self) -> Dict[str, Any]:
        """Generate replay recommendations based on patterns"""
        try:
            # Check if replay should be triggered
            should_replay = (self._episode_count % self.genome["interval"]) == 0
            
            if not should_replay or not self.profitable_sequences:
                return {
                    'replay_recommended': False,
                    'replay_bonus': 0.0,
                    'best_pattern': None
                }
            
            # Find best sequence for replay
            best_sequence = max(self.profitable_sequences, key=lambda x: x['pnl'])
            
            # Calculate replay bonus
            replay_bonus = self.genome["bonus"] * (best_sequence['pnl'] / 100.0)
            replay_bonus *= self.genome["replay_decay"] ** (time.time() - best_sequence['timestamp']) / 3600
            
            self.replay_bonus = replay_bonus
            
            # Find best pattern
            best_pattern = None
            if self.sequence_patterns:
                best_pattern = max(
                    self.sequence_patterns.items(),
                    key=lambda x: x[1]['avg_pnl'] * x[1]['confidence']
                )[0]
            
            return {
                'replay_recommended': True,
                'replay_bonus': replay_bonus,
                'best_sequence_pnl': best_sequence['pnl'],
                'best_pattern': best_pattern,
                'total_patterns': len(self.sequence_patterns)
            }
            
        except Exception as e:
            self.logger.error(f"Replay recommendation generation failed: {e}")
            return {
                'replay_recommended': False,
                'replay_bonus': 0.0,
                'error': str(e)
            }

    async def _generate_replay_thesis(self, sequence_data: Dict[str, Any], 
                                    replay_result: Dict[str, Any]) -> str:
        """Generate comprehensive replay analysis thesis"""
        try:
            # Performance metrics
            sequences_analyzed = self._analysis_performance['sequences_analyzed']
            patterns_identified = len(self.sequence_patterns)
            best_pnl = self.best_sequence_pnl
            
            # Pattern analysis
            total_patterns = len(self.sequence_patterns)
            profitable_patterns = sum(1 for p in self.sequence_patterns.values() if p['avg_pnl'] > 0)
            
            # Replay status
            replay_recommended = replay_result.get('replay_recommended', False)
            replay_bonus = replay_result.get('replay_bonus', 0.0)
            
            thesis_parts = [
                f"Historical Replay Analysis: Processed {sequences_analyzed} trading sequences across {self._episode_count} episodes",
                f"Pattern Recognition: Identified {total_patterns} unique patterns, {profitable_patterns} profitable",
                f"Best sequence performance: {best_pnl:.2f} PnL with quality-weighted selection",
                f"Learning optimization: {len(self.profitable_sequences)} profitable sequences stored for replay"
            ]
            
            if replay_recommended:
                thesis_parts.append(f"Replay triggered with bonus: {replay_bonus:.4f} based on historical performance")
                best_pattern = replay_result.get('best_pattern')
                if best_pattern:
                    pattern_data = self.sequence_patterns[best_pattern]
                    thesis_parts.append(f"Best pattern '{best_pattern}': {pattern_data['avg_pnl']:.2f} avg PnL, {pattern_data['confidence']:.2f} confidence")
            else:
                thesis_parts.append("No replay recommended this cycle - continuing pattern accumulation")
            
            # Sequence quality analysis
            if self._sequence_quality_scores:
                avg_quality = np.mean(list(self._sequence_quality_scores)[-20:])
                thesis_parts.append(f"Recent sequence quality: {avg_quality:.2f} (target: {self.config.min_sequence_quality}+)")
            
            # Learning curve assessment
            if len(self.profitable_sequences) > 5:
                recent_performance = np.mean([s['pnl'] for s in self.profitable_sequences[-5:]])
                thesis_parts.append(f"Learning trend: Recent avg PnL {recent_performance:.2f} showing pattern effectiveness")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Replay thesis generation failed: {str(e)} - Pattern analysis continuing with basic metrics"

    async def _update_replay_smart_bus(self, replay_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with replay analysis results"""
        try:
            # Replay sequences data
            replay_data = {
                'total_sequences': len(self.profitable_sequences),
                'best_sequence_pnl': self.best_sequence_pnl,
                'replay_bonus': self.replay_bonus,
                'sequences_analyzed': self._analysis_performance['sequences_analyzed']
            }
            
            self.smart_bus.set(
                'replay_sequences',
                replay_data,
                module='HistoricalReplayAnalyzer',
                thesis=thesis
            )
            
            # Pattern analysis
            pattern_summary = {
                'total_patterns': len(self.sequence_patterns),
                'profitable_patterns': sum(1 for p in self.sequence_patterns.values() if p['avg_pnl'] > 0),
                'best_pattern': max(self.sequence_patterns.items(), 
                                  key=lambda x: x[1]['avg_pnl']) if self.sequence_patterns else None,
                'pattern_confidence_avg': np.mean([p['confidence'] for p in self.sequence_patterns.values()]) if self.sequence_patterns else 0.0
            }
            
            self.smart_bus.set(
                'pattern_analysis',
                pattern_summary,
                module='HistoricalReplayAnalyzer',
                thesis=f"Pattern analysis: {pattern_summary['total_patterns']} patterns identified"
            )
            
            # Sequence quality metrics
            quality_metrics = {
                'current_quality': replay_result.get('sequence_quality', 0.0),
                'average_quality': np.mean(list(self._sequence_quality_scores)) if self._sequence_quality_scores else 0.0,
                'quality_trend': self._calculate_quality_trend(),
                'episodes_processed': self._episode_count
            }
            
            self.smart_bus.set(
                'sequence_quality',
                quality_metrics,
                module='HistoricalReplayAnalyzer',
                thesis="Sequence quality assessment and learning progress tracking"
            )
            
            # Learning progress
            learning_metrics = {
                'profitable_sequences_count': len(self.profitable_sequences),
                'total_episodes': self._episode_count,
                'success_rate': len(self.profitable_sequences) / max(self._episode_count, 1),
                'learning_acceleration': self._calculate_learning_acceleration()
            }
            
            self.smart_bus.set(
                'learning_progress',
                learning_metrics,
                module='HistoricalReplayAnalyzer',
                thesis="Learning progress and pattern discovery effectiveness"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    def _calculate_quality_trend(self) -> str:
        """Calculate sequence quality trend"""
        if len(self._sequence_quality_scores) < 10:
            return "insufficient_data"
        
        recent_quality = np.mean(list(self._sequence_quality_scores)[-5:])
        older_quality = np.mean(list(self._sequence_quality_scores)[-10:-5])
        
        if recent_quality > older_quality * 1.1:
            return "improving"
        elif recent_quality < older_quality * 0.9:
            return "declining"
        else:
            return "stable"

    def _calculate_learning_acceleration(self) -> float:
        """Calculate learning acceleration metric"""
        if self._episode_count < 20:
            return 0.0
        
        # Compare recent vs early profitable sequence discovery rate
        recent_episodes = max(10, self._episode_count // 4)
        recent_profitable = sum(1 for s in self.profitable_sequences 
                              if time.time() - s['timestamp'] < recent_episodes * 60)
        
        recent_rate = recent_profitable / recent_episodes
        overall_rate = len(self.profitable_sequences) / self._episode_count
        
        return (recent_rate - overall_rate) / max(overall_rate, 0.01)

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no sequence data is available"""
        self.logger.warning("No sequence data available - using cached analysis")
        
        return {
            'sequences_processed': self._analysis_performance['sequences_analyzed'],
            'total_patterns': len(self.sequence_patterns),
            'best_sequence_pnl': self.best_sequence_pnl,
            'fallback_reason': 'no_sequence_data'
        }

    async def _handle_replay_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle replay analysis errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "HistoricalReplayAnalyzer")
        explanation = self.english_explainer.explain_error(
            "HistoricalReplayAnalyzer", str(error), "replay analysis"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "REPLAY_ANALYSIS_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="replay_analysis"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'sequences_processed': self._analysis_performance['sequences_analyzed'],
            'total_patterns': len(self.sequence_patterns),
            'best_sequence_pnl': self.best_sequence_pnl,
            'replay_bonus': 0.0,
            'fallback_reason': reason,
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    def _update_replay_health(self):
        """Update replay analysis health metrics"""
        try:
            # Check pattern discovery rate
            if self._episode_count > 0:
                pattern_rate = len(self.sequence_patterns) / self._episode_count
                if pattern_rate < 0.1:  # Less than 10% pattern discovery
                    self._health_status = 'warning'
                elif pattern_rate > 0.3:  # Good pattern discovery
                    self._health_status = 'healthy'
            
            # Check sequence quality
            if self._sequence_quality_scores:
                avg_quality = np.mean(list(self._sequence_quality_scores)[-10:])
                if avg_quality < self.config.min_sequence_quality:
                    self._health_status = 'warning'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _analyze_pattern_effectiveness(self):
        """Analyze effectiveness of identified patterns"""
        try:
            for pattern, data in self.sequence_patterns.items():
                if data['count'] >= 5:  # Sufficient sample size
                    effectiveness = data['avg_pnl'] * data['confidence']
                    
                    if effectiveness > 10.0:  # Highly effective pattern
                        self.logger.info(
                            format_operator_message(
                                "🎯", "EFFECTIVE_PATTERN_IDENTIFIED",
                                pattern=pattern,
                                avg_pnl=f"{data['avg_pnl']:.2f}",
                                confidence=f"{data['confidence']:.2f}",
                                count=data['count'],
                                context="pattern_effectiveness"
                            )
                        )
            
        except Exception as e:
            self.logger.error(f"Pattern effectiveness analysis failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'HistoricalReplayAnalyzer', 'analysis_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'HistoricalReplayAnalyzer', 'analysis_cycle', 0, False
        )

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'profitable_sequences': self.profitable_sequences.copy(),
            'sequence_patterns': self.sequence_patterns.copy(),
            'genome': self.genome.copy(),
            'episode_count': self._episode_count,
            'best_sequence_pnl': self.best_sequence_pnl,
            'replay_bonus': self.replay_bonus,
            'analysis_performance': self._analysis_performance.copy(),
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self._health_status
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'profitable_sequences' in state:
            self.profitable_sequences = state['profitable_sequences']
        
        if 'sequence_patterns' in state:
            self.sequence_patterns = state['sequence_patterns']
        
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'episode_count' in state:
            self._episode_count = state['episode_count']
        
        if 'best_sequence_pnl' in state:
            self.best_sequence_pnl = state['best_sequence_pnl']
        
        if 'replay_bonus' in state:
            self.replay_bonus = state['replay_bonus']
        
        if 'analysis_performance' in state:
            self._analysis_performance.update(state['analysis_performance'])
        
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])
        
        if 'health_status' in state:
            self._health_status = state['health_status']

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'total_patterns': len(self.sequence_patterns),
            'profitable_sequences': len(self.profitable_sequences),
            'episodes_processed': self._episode_count
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def propose_action(self, obs: Any = None, **kwargs) -> np.ndarray:
        """Legacy compatibility for action proposal"""
        return np.array([0.0, 0.0])
    
    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        if self._sequence_quality_scores:
            return float(np.mean(list(self._sequence_quality_scores)[-5:]))
        return 0.5