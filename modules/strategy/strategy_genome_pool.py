# ─────────────────────────────────────────────────────────────
# File: modules/strategy/strategy_genome_pool.py
# Enhanced with InfoBus integration & intelligent genetic evolution
# ─────────────────────────────────────────────────────────────

import hashlib
import numpy as np
import datetime
import random
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class StrategyGenomePool(Module, AnalysisMixin, StateManagementMixin, TradingMixin):
    """
    Enhanced strategy genome pool with InfoBus integration.
    Evolves trading strategy parameters using genetic algorithms with intelligent selection and adaptation.
    Provides robust population management and performance-driven evolution.
    """

    def __init__(
        self,
        population_size: int = 20,
        tournament_k: int = 3,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.2,
        max_generations_kept: int = 10000,
        debug: bool = False,
        profit_target: float = 150.0,
        genome_size: int = 4,
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 500),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.genome_size = int(genome_size)
        self.pop_size = int(population_size)
        self.tournament_k = int(tournament_k)
        self.cx_rate = float(crossover_rate)
        self.mut_rate = float(mutation_rate)
        self.mut_scale = float(mutation_scale)
        self.max_generations_kept = int(max_generations_kept)
        self.debug = bool(debug)
        self.profit_target = float(profit_target)
        
        # Enhanced genome bounds for XAUUSD/EURUSD trading
        self.genome_bounds = {
            'sl_base': (0.2, 3.0),      # Stop Loss: 20-300 pips
            'tp_base': (0.3, 4.0),      # Take Profit: 30-400 pips
            'vol_scale': (0.1, 2.5),    # Volatility scaling
            'regime_adapt': (0.0, 0.8), # Regime adaptation factor
        }
        
        # Initialize population with enhanced seeding
        self.population = self._initialize_population()
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)
        
        # Evolution tracking
        self.epoch = 0
        self.best_genome = self.population[0].copy()
        self.best_fitness = -np.inf
        self.generations_without_improvement = 0
        self.active_genome = None
        self.active_genome_idx = 0
        
        # Enhanced evolution analytics
        self.evolution_analytics = {
            'fitness_history': deque(maxlen=100),
            'diversity_history': deque(maxlen=100),
            'selection_pressure': deque(maxlen=50),
            'mutation_success_rate': 0.0,
            'crossover_success_rate': 0.0,
            'adaptation_events': 0,
            'convergence_score': 0.0,
            'population_health': 'healthy'
        }
        
        # Genome evaluation tracking
        self.genome_evaluation_history = deque(maxlen=self.pop_size * 5)
        self.genome_performance_cache = {}
        self.genome_usage_stats = defaultdict(int)
        
        # Advanced selection strategies
        self.selection_strategies = {
            'tournament': self._tournament_selection,
            'roulette': self._roulette_selection,
            'rank': self._rank_selection,
            'adaptive': self._adaptive_selection,
            'diversity': self._diversity_selection
        }
        
        # Current selection strategy
        self.current_selection_strategy = 'adaptive'
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "StrategyGenomePool",
            "logs/strategy/genome_pool.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("StrategyGenomePool")
        
        self.log_operator_info(
            "🧬 Strategy Genome Pool initialized",
            population_size=self.pop_size,
            genome_size=self.genome_size,
            profit_target=f"€{self.profit_target}",
            mutation_rate=f"{self.mut_rate:.1%}"
        )

    def log_operator_debug(self, message: str, **kwargs):
        """Log debug message with proper formatting"""
        if self.debug and hasattr(self, 'logger'):
            details = " | ".join(f"{k}: {v}" for k, v in kwargs.items()) if kwargs else ""
            formatted_message = format_operator_message("🔧", message, details=details)
            if hasattr(self.logger, 'debug'):
                self.logger.debug(formatted_message)

    def log_operator_info(self, message: str, **kwargs):
        """Log info message with proper formatting"""
        if hasattr(self, 'logger'):
            details = " | ".join(f"{k}: {v}" for k, v in kwargs.items()) if kwargs else ""
            formatted_message = format_operator_message("📊", message, details=details)
            self.logger.info(formatted_message)

    def log_operator_warning(self, message: str, **kwargs):
        """Log warning message with proper formatting"""
        if hasattr(self, 'logger'):
            details = " | ".join(f"{k}: {v}" for k, v in kwargs.items()) if kwargs else ""
            formatted_message = format_operator_message("⚠️", message, details=details)
            self.logger.warning(formatted_message)

    def log_operator_error(self, message: str, **kwargs):
        """Log error message with proper formatting"""
        if hasattr(self, 'logger'):
            details = " | ".join(f"{k}: {v}" for k, v in kwargs.items()) if kwargs else ""
            formatted_message = format_operator_message("❌", message, details=details)
            self.logger.error(formatted_message)

    def _initialize_population(self) -> np.ndarray:
        """Initialize population with enhanced seeding strategies"""
        
        # Create diverse seed genomes for different trading approaches
        seed_genomes = [
            np.array([0.5, 0.75, 1.0, 0.2]),   # Conservative scalping
            np.array([1.0, 1.5, 1.2, 0.3]),    # Balanced approach
            np.array([1.5, 2.0, 1.5, 0.4]),    # Aggressive trading
            np.array([0.8, 1.2, 0.8, 0.25]),   # Tight risk management
            np.array([1.2, 1.8, 1.3, 0.35]),   # Medium risk-reward
            np.array([0.3, 0.8, 0.6, 0.15]),   # Ultra-conservative
            np.array([2.0, 2.5, 2.0, 0.5]),    # High-risk high-reward
            np.array([0.7, 1.0, 0.9, 0.2]),    # Defensive approach
        ]
        
        # Fill remaining spots with random genomes within bounds
        remaining_spots = max(0, self.pop_size - len(seed_genomes))
        random_genomes = []
        
        for _ in range(remaining_spots):
            genome = np.array([
                np.random.uniform(*self.genome_bounds['sl_base']),
                np.random.uniform(*self.genome_bounds['tp_base']),
                np.random.uniform(*self.genome_bounds['vol_scale']),
                np.random.uniform(*self.genome_bounds['regime_adapt'])
            ], dtype=np.float32)
            random_genomes.append(genome)
        
        # Combine seed and random genomes
        all_genomes = seed_genomes + random_genomes
        
        # Take only the required number
        population = np.array(all_genomes[:self.pop_size], dtype=np.float32)
        
        self.log_operator_info(
            f"🌱 Population initialized",
            seed_genomes=len(seed_genomes),
            random_genomes=len(random_genomes),
            total_size=len(population)
        )
        
        return population

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Reset evolution state
        self.epoch = 0
        self.fitness[:] = 0.0
        self.generations_without_improvement = 0
        self.active_genome = None
        self.active_genome_idx = 0
        
        # Reset best tracking
        self.best_genome = self.population[0].copy()
        self.best_fitness = -np.inf
        
        # Reset analytics
        self.evolution_analytics = {
            'fitness_history': deque(maxlen=100),
            'diversity_history': deque(maxlen=100),
            'selection_pressure': deque(maxlen=50),
            'mutation_success_rate': 0.0,
            'crossover_success_rate': 0.0,
            'adaptation_events': 0,
            'convergence_score': 0.0,
            'population_health': 'healthy'
        }
        
        # Reset evaluation tracking
        self.genome_evaluation_history.clear()
        self.genome_performance_cache.clear()
        self.genome_usage_stats.clear()
        
        self.log_operator_info("🔄 Strategy Genome Pool reset - population reinitialized")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and adaptive evolution"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited genome evolution")
            return
        
        # Extract context and performance data
        context = extract_standard_context(info_bus)
        evolution_context = self._extract_evolution_context_from_info_bus(info_bus, context)
        
        # Update evolution analytics
        self._update_evolution_analytics(evolution_context)
        
        # Adaptive strategy selection
        self._adapt_selection_strategy(evolution_context)
        
        # Update InfoBus with genome pool status
        self._update_info_bus_with_genome_data(info_bus)

    def _extract_evolution_context_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evolution context from InfoBus"""
        
        try:
            # Get performance data
            recent_trades = info_bus.get('recent_trades', [])
            risk_data = info_bus.get('risk', {})
            
            evolution_context = {
                'timestamp': datetime.datetime.now().isoformat(),
                'session_pnl': context.get('session_pnl', 0),
                'recent_trades_count': len(recent_trades),
                'market_regime': context.get('regime', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium'),
                'drawdown': risk_data.get('current_drawdown', 0),
                'balance': risk_data.get('balance', 0),
                'performance_trend': self._calculate_performance_trend(recent_trades),
                'market_stress_level': self._assess_market_stress(context, risk_data)
            }
            
            return evolution_context
            
        except Exception as e:
            self.log_operator_warning(f"Evolution context extraction failed: {e}")
            return {'timestamp': datetime.datetime.now().isoformat()}

    def _calculate_performance_trend(self, recent_trades: List[Dict]) -> str:
        """Calculate recent performance trend"""
        
        try:
            if not recent_trades or len(recent_trades) < 5:
                return 'insufficient_data'
            
            # Look at last 10 trades
            recent_pnls = [t.get('pnl', 0) for t in recent_trades[-10:]]
            
            # Calculate trend
            if len(recent_pnls) >= 3:
                recent_avg = np.mean(recent_pnls[-3:])
                older_avg = np.mean(recent_pnls[:-3]) if len(recent_pnls) > 3 else recent_avg
                
                if recent_avg > older_avg + 10:
                    return 'improving'
                elif recent_avg < older_avg - 10:
                    return 'declining'
                else:
                    return 'stable'
            
            return 'stable'
            
        except Exception:
            return 'unknown'

    def _assess_market_stress(self, context: Dict[str, Any], risk_data: Dict[str, Any]) -> str:
        """Assess current market stress level"""
        
        stress_score = 0
        
        # High volatility increases stress
        if context.get('volatility_level') == 'high':
            stress_score += 2
        elif context.get('volatility_level') == 'extreme':
            stress_score += 3
        
        # High drawdown increases stress
        drawdown = risk_data.get('current_drawdown', 0)
        if drawdown > 0.05:
            stress_score += 2
        if drawdown > 0.1:
            stress_score += 2
        
        # Market regime considerations
        regime = context.get('regime', 'unknown')
        if regime in ['volatile', 'unknown']:
            stress_score += 1
        
        # Classify stress level
        if stress_score <= 1:
            return 'low'
        elif stress_score <= 3:
            return 'medium'
        elif stress_score <= 5:
            return 'high'
        else:
            return 'extreme'

    def _update_evolution_analytics(self, evolution_context: Dict[str, Any]) -> None:
        """Update comprehensive evolution analytics"""
        
        try:
            # Update fitness history
            if len(self.fitness) > 0:
                current_fitness_stats = {
                    'mean': float(np.mean(self.fitness)),
                    'max': float(np.max(self.fitness)),
                    'std': float(np.std(self.fitness)),
                    'timestamp': evolution_context.get('timestamp')
                }
                self.evolution_analytics['fitness_history'].append(current_fitness_stats)
            
            # Update diversity metrics
            diversity_score = self._calculate_population_diversity()
            diversity_record = {
                'score': diversity_score,
                'timestamp': evolution_context.get('timestamp')
            }
            self.evolution_analytics['diversity_history'].append(diversity_record)
            
            # Update convergence score
            self.evolution_analytics['convergence_score'] = self._calculate_convergence_score()
            
            # Assess population health
            self._assess_population_health()
            
        except Exception as e:
            self.log_operator_warning(f"Evolution analytics update failed: {e}")

    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity score"""
        
        try:
            if len(self.population) < 2:
                return 0.0
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(self.population)):
                for j in range(i + 1, len(self.population)):
                    dist = np.linalg.norm(self.population[i] - self.population[j])
                    distances.append(dist)
            
            return float(np.mean(distances)) if distances else 0.0
            
        except Exception:
            return 0.0

    def _calculate_convergence_score(self) -> float:
        """Calculate convergence score (lower = more converged)"""
        
        try:
            if len(self.fitness) < 2:
                return 1.0
            
            fitness_std = np.std(self.fitness)
            fitness_mean = abs(np.mean(self.fitness))
            
            # Normalized standard deviation
            convergence = fitness_std / (fitness_mean + 1e-6)
            return min(1.0, convergence)
            
        except Exception:
            return 0.5

    def _assess_population_health(self) -> None:
        """Assess overall population health"""
        
        try:
            diversity_score = self.evolution_analytics['diversity_history'][-1]['score'] if self.evolution_analytics['diversity_history'] else 0.5
            convergence_score = self.evolution_analytics['convergence_score']
            
            # Health criteria
            if diversity_score > 1.0 and convergence_score > 0.3:
                health = 'healthy'
            elif diversity_score > 0.5 and convergence_score > 0.1:
                health = 'moderate'
            elif diversity_score < 0.2 or convergence_score < 0.05:
                health = 'poor'
            else:
                health = 'fair'
            
            if health != self.evolution_analytics['population_health']:
                self.log_operator_info(
                    f"🏥 Population health changed: {self.evolution_analytics['population_health']} → {health}",
                    diversity=f"{diversity_score:.3f}",
                    convergence=f"{convergence_score:.3f}"
                )
            
            self.evolution_analytics['population_health'] = health
            
        except Exception as e:
            self.log_operator_warning(f"Population health assessment failed: {e}")

    def _adapt_selection_strategy(self, evolution_context: Dict[str, Any]) -> None:
        """Adapt selection strategy based on context"""
        
        try:
            current_health = self.evolution_analytics['population_health']
            performance_trend = evolution_context.get('performance_trend', 'stable')
            market_stress = evolution_context.get('market_stress_level', 'medium')
            
            # Strategy selection logic
            if current_health == 'poor' or performance_trend == 'declining':
                new_strategy = 'diversity'  # Promote diversity
            elif market_stress in ['high', 'extreme']:
                new_strategy = 'tournament'  # Conservative selection
            elif performance_trend == 'improving':
                new_strategy = 'rank'  # Exploit good performers
            else:
                new_strategy = 'adaptive'  # Balanced approach
            
            if new_strategy != self.current_selection_strategy:
                self.log_operator_info(
                    f"🎯 Selection strategy adapted",
                    from_strategy=self.current_selection_strategy,
                    to_strategy=new_strategy,
                    reason=f"health={current_health}, trend={performance_trend}, stress={market_stress}"
                )
                self.current_selection_strategy = new_strategy
                self.evolution_analytics['adaptation_events'] += 1
            
        except Exception as e:
            self.log_operator_warning(f"Selection strategy adaptation failed: {e}")

    def genome_hash(self, g: np.ndarray) -> str:
        """Generate hash for genome tracking with enhanced precision"""
        
        try:
            # Round to reasonable precision to avoid minor floating point differences
            rounded_genome = np.round(g, decimals=4)
            return hashlib.md5(rounded_genome.tobytes()).hexdigest()
        except Exception as e:
            self.log_operator_warning(f"Genome hashing failed: {e}")
            return "error_hash"

    def evaluate_population(self, eval_fn: Callable[[np.ndarray], float]) -> None:
        """Enhanced population evaluation with comprehensive analytics"""
        
        try:
            self.log_operator_info(f"🧪 Evaluating population generation {self.epoch}")
            
            evaluation_results = []
            fitness_improvements = 0
            
            for i, genome in enumerate(self.population):
                try:
                    # Validate genome
                    if np.any(~np.isfinite(genome)):
                        self.log_operator_warning(f"Invalid genome {i}: {genome}")
                        genome = self._repair_genome(genome)
                        self.population[i] = genome
                    
                    # Check cache first
                    genome_hash = self.genome_hash(genome)
                    if genome_hash in self.genome_performance_cache:
                        raw_fitness = self.genome_performance_cache[genome_hash]
                        cache_hit = True
                    else:
                        raw_fitness = float(eval_fn(genome))
                        self.genome_performance_cache[genome_hash] = raw_fitness
                        cache_hit = False
                    
                    # Validate fitness
                    if not np.isfinite(raw_fitness):
                        self.log_operator_warning(f"Invalid fitness for genome {i}: {raw_fitness}")
                        raw_fitness = 0.0
                    
                    # Apply profit target bonus
                    if raw_fitness >= self.profit_target:
                        profit_bonus = 1.0 + (raw_fitness - self.profit_target) / self.profit_target
                        final_fitness = raw_fitness * profit_bonus
                        
                        self.log_operator_info(
                            f"💰 Genome {i} achieved target",
                            raw_pnl=f"€{raw_fitness:.2f}",
                            bonus_multiplier=f"{profit_bonus:.2f}x",
                            final_fitness=f"{final_fitness:.2f}"
                        )
                    else:
                        final_fitness = raw_fitness
                    
                    # Track improvement
                    if final_fitness > self.fitness[i]:
                        fitness_improvements += 1
                    
                    self.fitness[i] = final_fitness
                    
                    evaluation_results.append({
                        'genome_idx': i,
                        'raw_fitness': raw_fitness,
                        'final_fitness': final_fitness,
                        'cache_hit': cache_hit,
                        'genome_hash': genome_hash
                    })
                    
                except Exception as e:
                    self.log_operator_error(f"Genome {i} evaluation failed: {e}")
                    self.fitness[i] = 0.0
            
            # Update best genome tracking
            self._update_best_genome_tracking()
            
            # Log evaluation summary
            self._log_evaluation_summary(evaluation_results, fitness_improvements)
            
            # Update evaluation history
            self._record_evaluation_event(evaluation_results)
            
        except Exception as e:
            self.log_operator_error(f"Population evaluation failed: {e}")

    def _repair_genome(self, genome: np.ndarray) -> np.ndarray:
        """Repair invalid genome by clamping to valid bounds"""
        
        repaired = genome.copy()
        
        # Clamp to bounds
        repaired[0] = np.clip(repaired[0], *self.genome_bounds['sl_base'])      # SL
        repaired[1] = np.clip(repaired[1], *self.genome_bounds['tp_base'])      # TP
        repaired[2] = np.clip(repaired[2], *self.genome_bounds['vol_scale'])    # Vol scale
        repaired[3] = np.clip(repaired[3], *self.genome_bounds['regime_adapt']) # Regime adapt
        
        # Replace NaN/inf with random valid values
        for i in range(len(repaired)):
            if not np.isfinite(repaired[i]):
                bounds = list(self.genome_bounds.values())[i]
                repaired[i] = np.random.uniform(*bounds)
        
        return repaired.astype(np.float32)

    def _update_best_genome_tracking(self) -> None:
        """Update best genome tracking with detailed analytics"""
        
        try:
            current_max_idx = np.argmax(self.fitness)
            current_max_fitness = self.fitness[current_max_idx]
            
            if current_max_fitness > self.best_fitness:
                improvement = current_max_fitness - self.best_fitness
                old_best = self.best_fitness
                
                self.best_fitness = current_max_fitness
                self.best_genome = self.population[current_max_idx].copy()
                self.generations_without_improvement = 0
                
                self.log_operator_info(
                    f"🏆 New best genome found!",
                    improvement=f"€{improvement:.2f}",
                    old_fitness=f"€{old_best:.2f}",
                    new_fitness=f"€{current_max_fitness:.2f}",
                    genome=f"[{', '.join(f'{x:.3f}' for x in self.best_genome)}]"
                )
            else:
                self.generations_without_improvement += 1
            
        except Exception as e:
            self.log_operator_warning(f"Best genome tracking update failed: {e}")

    def _log_evaluation_summary(self, evaluation_results: List[Dict], fitness_improvements: int) -> None:
        """Log comprehensive evaluation summary"""
        
        try:
            cache_hits = sum(1 for r in evaluation_results if r.get('cache_hit', False))
            target_achievers = sum(1 for r in evaluation_results if r.get('raw_fitness', 0) >= self.profit_target)
            
            fitness_stats = {
                'min': float(np.min(self.fitness)),
                'max': float(np.max(self.fitness)),
                'mean': float(np.mean(self.fitness)),
                'std': float(np.std(self.fitness))
            }
            
            self.log_operator_info(
                f"📊 Generation {self.epoch} evaluation complete",
                fitness_range=f"€{fitness_stats['min']:.1f} - €{fitness_stats['max']:.1f}",
                mean_fitness=f"€{fitness_stats['mean']:.1f}",
                improvements=f"{fitness_improvements}/{len(evaluation_results)}",
                cache_hits=cache_hits,
                target_achievers=target_achievers,
                stagnant_gens=self.generations_without_improvement
            )
            
            # Log top performers
            top_indices = np.argsort(self.fitness)[-3:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                genome_str = ", ".join(f"{x:.3f}" for x in self.population[idx])
                self.log_operator_info(
                    f"  #{rank} Genome {idx}: €{self.fitness[idx]:.2f} [{genome_str}]"
                )
                
        except Exception as e:
            self.log_operator_warning(f"Evaluation summary logging failed: {e}")

    def _record_evaluation_event(self, evaluation_results: List[Dict]) -> None:
        """Record evaluation event in history"""
        
        try:
            evaluation_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'generation': self.epoch,
                'population_size': len(evaluation_results),
                'fitness_stats': {
                    'min': float(np.min(self.fitness)),
                    'max': float(np.max(self.fitness)),
                    'mean': float(np.mean(self.fitness)),
                    'std': float(np.std(self.fitness))
                },
                'best_genome': self.best_genome.copy().tolist(),
                'best_fitness': float(self.best_fitness),
                'generations_without_improvement': self.generations_without_improvement
            }
            
            self.genome_evaluation_history.append(evaluation_record)
            
        except Exception as e:
            self.log_operator_warning(f"Evaluation event recording failed: {e}")

    def evolve_strategies(self) -> None:
        """Enhanced evolution with adaptive strategies and comprehensive tracking"""
        
        try:
            self.log_operator_info(f"🧬 Evolving generation {self.epoch} → {self.epoch + 1}")
            
            # Track pre-evolution state
            old_diversity = self._calculate_population_diversity()
            old_hashes = [self.genome_hash(g) for g in self.population]
            
            # Apply adaptive parameters
            adaptive_params = self._calculate_adaptive_parameters()
            
            # Create new population
            new_population = self._create_new_population(adaptive_params)
            
            # Update population
            self.population = new_population
            self.fitness[:] = 0.0
            self.epoch += 1
            
            # Track post-evolution state
            new_diversity = self._calculate_population_diversity()
            new_hashes = [self.genome_hash(g) for g in self.population]
            evolution_stats = self._calculate_evolution_stats(old_hashes, new_hashes, old_diversity, new_diversity)
            
            # Log evolution results
            self._log_evolution_results(evolution_stats, adaptive_params)
            
        except Exception as e:
            self.log_operator_error(f"Evolution failed: {e}")

    def _calculate_adaptive_parameters(self) -> Dict[str, float]:
        """Calculate adaptive evolution parameters based on current state"""
        
        # Base parameters
        adaptive_mut_rate = self.mut_rate
        adaptive_mut_scale = self.mut_scale
        adaptive_cx_rate = self.cx_rate
        
        # Adapt based on stagnation
        if self.generations_without_improvement > 10:
            adaptive_mut_rate = min(0.4, self.mut_rate * 2.0)
            adaptive_mut_scale = min(0.6, self.mut_scale * 1.5)
            
        # Adapt based on population health
        if self.evolution_analytics['population_health'] == 'poor':
            adaptive_mut_rate = min(0.5, adaptive_mut_rate * 1.5)
            adaptive_cx_rate = max(0.2, adaptive_cx_rate * 0.8)
        
        # Adapt based on convergence
        convergence = self.evolution_analytics['convergence_score']
        if convergence < 0.1:  # Too converged
            adaptive_mut_rate = min(0.6, adaptive_mut_rate * 2.0)
        
        return {
            'mutation_rate': adaptive_mut_rate,
            'mutation_scale': adaptive_mut_scale,
            'crossover_rate': adaptive_cx_rate,
            'elitism_count': max(1, self.pop_size // 10)  # Keep top 10%
        }

    def _create_new_population(self, adaptive_params: Dict[str, float]) -> np.ndarray:
        """Create new population using adaptive evolution strategies"""
        
        new_pop = []
        evolution_stats = {'crossovers': 0, 'mutations': 0, 'elite_preserved': 0}
        
        # Elitism: preserve best genomes
        elite_count = adaptive_params['elitism_count']
        elite_indices = np.argsort(self.fitness)[-elite_count:]
        
        for idx in elite_indices:
            new_pop.append(self.population[idx].copy())
            evolution_stats['elite_preserved'] += 1
        
        # Diversity injection if population health is poor
        if self.evolution_analytics['population_health'] == 'poor':
            diversity_inject_count = min(3, self.pop_size // 10)
            for _ in range(diversity_inject_count):
                diverse_genome = self._generate_diverse_genome()
                new_pop.append(diverse_genome)
        
        # Generate rest of population
        while len(new_pop) < self.pop_size:
            try:
                # Selection
                parent1 = self._select_parent()
                parent2 = self._select_parent()
                
                # Crossover
                if np.random.random() < adaptive_params['crossover_rate']:
                    child = self._crossover(parent1, parent2)
                    evolution_stats['crossovers'] += 1
                else:
                    child = parent1.copy()
                
                # Mutation
                if np.random.random() < adaptive_params['mutation_rate']:
                    child = self._mutate(child, adaptive_params['mutation_scale'])
                    evolution_stats['mutations'] += 1
                
                # Validate and repair
                child = self._repair_genome(child)
                new_pop.append(child)
                
            except Exception as e:
                self.log_operator_warning(f"Child generation failed: {e}")
                # Add random valid genome as fallback
                fallback = self._generate_diverse_genome()
                new_pop.append(fallback)
        
        # Store evolution stats for reporting
        self._last_evolution_stats = evolution_stats
        
        return np.array(new_pop[:self.pop_size], dtype=np.float32)

    def _select_parent(self) -> np.ndarray:
        """Select parent using current selection strategy"""
        
        strategy_func = self.selection_strategies.get(self.current_selection_strategy, self._tournament_selection)
        return strategy_func()

    def _tournament_selection(self) -> np.ndarray:
        """Tournament selection"""
        
        candidates = np.random.choice(self.pop_size, self.tournament_k, replace=False)
        winner_idx = candidates[np.argmax(self.fitness[candidates])]
        return self.population[winner_idx].copy()

    def _roulette_selection(self) -> np.ndarray:
        """Roulette wheel selection"""
        
        # Shift fitness to positive
        shifted_fitness = self.fitness - np.min(self.fitness) + 1e-6
        total_fitness = np.sum(shifted_fitness)
        
        if total_fitness > 0:
            probabilities = shifted_fitness / total_fitness
            selected_idx = np.random.choice(self.pop_size, p=probabilities)
        else:
            selected_idx = np.random.randint(self.pop_size)
        
        return self.population[selected_idx].copy()

    def _rank_selection(self) -> np.ndarray:
        """Rank-based selection"""
        
        ranks = np.argsort(np.argsort(self.fitness)) + 1  # Ranks from 1 to pop_size
        probabilities = ranks / np.sum(ranks)
        selected_idx = np.random.choice(self.pop_size, p=probabilities)
        return self.population[selected_idx].copy()

    def _adaptive_selection(self) -> np.ndarray:
        """Adaptive selection combining multiple strategies"""
        
        # Choose strategy based on current state
        if self.generations_without_improvement < 5:
            return self._tournament_selection()
        elif self.evolution_analytics['population_health'] == 'poor':
            return self._diversity_selection()
        else:
            return self._rank_selection()

    def _diversity_selection(self) -> np.ndarray:
        """Selection that promotes diversity"""
        
        # Select randomly from top 50% to promote diversity
        top_half = np.argsort(self.fitness)[self.pop_size//2:]
        selected_idx = np.random.choice(top_half)
        return self.population[selected_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Enhanced crossover with multiple strategies"""
        
        crossover_type = np.random.choice(['uniform', 'arithmetic', 'single_point'])
        
        if crossover_type == 'uniform':
            # Uniform crossover
            mask = np.random.random(self.genome_size) < 0.5
            child = np.where(mask, parent1, parent2)
        elif crossover_type == 'arithmetic':
            # Arithmetic crossover
            alpha = np.random.random()
            child = alpha * parent1 + (1 - alpha) * parent2
        else:
            # Single point crossover
            point = np.random.randint(1, self.genome_size)
            child = np.concatenate([parent1[:point], parent2[point:]])
        
        return child.astype(np.float32)

    def _mutate(self, genome: np.ndarray, mutation_scale: float) -> np.ndarray:
        """Enhanced mutation with adaptive strategies"""
        
        mutated = genome.copy()
        
        # Choose mutation strategy
        mutation_type = np.random.choice(['gaussian', 'uniform', 'polynomial'])
        
        for i in range(self.genome_size):
            if np.random.random() < 0.3:  # 30% chance per gene
                bounds = list(self.genome_bounds.values())[i]
                
                if mutation_type == 'gaussian':
                    # Gaussian mutation
                    noise = np.random.normal(0, mutation_scale)
                    mutated[i] += noise
                elif mutation_type == 'uniform':
                    # Uniform mutation within bounds
                    range_size = bounds[1] - bounds[0]
                    noise = np.random.uniform(-range_size * mutation_scale, range_size * mutation_scale)
                    mutated[i] += noise
                else:
                    # Polynomial mutation
                    range_size = bounds[1] - bounds[0]
                    u = np.random.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1/3) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1/3)
                    mutated[i] += delta * range_size * mutation_scale
                
                # Clamp to bounds
                mutated[i] = np.clip(mutated[i], bounds[0], bounds[1])
        
        return mutated

    def _generate_diverse_genome(self) -> np.ndarray:
        """Generate a diverse genome for population diversity"""
        
        genome = np.array([
            np.random.uniform(*self.genome_bounds['sl_base']),
            np.random.uniform(*self.genome_bounds['tp_base']),
            np.random.uniform(*self.genome_bounds['vol_scale']),
            np.random.uniform(*self.genome_bounds['regime_adapt'])
        ], dtype=np.float32)
        
        return genome

    def _calculate_evolution_stats(self, old_hashes: List[str], new_hashes: List[str], 
                                 old_diversity: float, new_diversity: float) -> Dict[str, Any]:
        """Calculate comprehensive evolution statistics"""
        
        genomes_changed = sum(1 for o, n in zip(old_hashes, new_hashes) if o != n)
        diversity_change = new_diversity - old_diversity
        
        return {
            'genomes_changed': genomes_changed,
            'change_percentage': (genomes_changed / self.pop_size) * 100,
            'diversity_change': diversity_change,
            'old_diversity': old_diversity,
            'new_diversity': new_diversity,
            **self._last_evolution_stats
        }

    def _log_evolution_results(self, evolution_stats: Dict[str, Any], adaptive_params: Dict[str, float]) -> None:
        """Log comprehensive evolution results"""
        
        self.log_operator_info(
            f"🧬 Evolution {self.epoch} complete",
            changed_genomes=f"{evolution_stats['genomes_changed']}/{self.pop_size} ({evolution_stats['change_percentage']:.1f}%)",
            crossovers=evolution_stats['crossovers'],
            mutations=evolution_stats['mutations'],
            elite_preserved=evolution_stats['elite_preserved'],
            diversity_change=f"{evolution_stats['diversity_change']:+.3f}",
            stagnant_gens=self.generations_without_improvement,
            health=self.evolution_analytics['population_health']
        )
        
        self.log_operator_info(
            f"  📊 Adaptive params: mut_rate={adaptive_params['mutation_rate']:.3f}, "
            f"mut_scale={adaptive_params['mutation_scale']:.3f}, "
            f"cx_rate={adaptive_params['crossover_rate']:.3f}"
        )

    def select_genome(self, mode: str = "smart", k: int = 3, custom_selector: Optional[Callable] = None) -> Union[np.ndarray, Dict[str, Any]]:
        """Enhanced genome selection with comprehensive strategies and format handling"""
        
        try:
            # Ensure population is properly initialized
            if len(self.population) == 0 or len(self.fitness) == 0:
                self.log_operator_warning("Empty population detected, reinitializing")
                self.population = self._initialize_population()
                self.fitness = np.zeros(self.pop_size, dtype=np.float32)
            
            assert len(self.population) == len(self.fitness), "Population/fitness size mismatch"
            N = len(self.population)
            
            self.log_operator_debug(f"Selecting genome with mode={mode}, k={k}")
            
            # Genome selection logic
            if mode == "smart":
                idx = self._smart_selection()
            elif mode == "random":
                idx = np.random.randint(N)
            elif mode == "best":
                idx = int(np.argmax(self.fitness))
            elif mode == "tournament":
                candidates = np.random.choice(N, min(k, N), replace=False)
                idx = candidates[np.argmax(self.fitness[candidates])]
            elif mode == "roulette":
                shifted_fitness = self.fitness - np.min(self.fitness) + 1e-8
                probabilities = shifted_fitness / shifted_fitness.sum() if shifted_fitness.sum() > 0 else np.ones(N) / N
                idx = np.random.choice(N, p=probabilities)
            elif mode == "diversity":
                idx = self._diversity_aware_selection()
            elif mode == "custom":
                assert custom_selector is not None, "Provide custom_selector callable!"
                idx = custom_selector(self.population, self.fitness)
            else:
                self.log_operator_warning(f"Unknown selection mode: {mode}, using best")
                idx = int(np.argmax(self.fitness))
            
            # Validate selection
            if not (0 <= idx < N):
                self.log_operator_warning(f"Invalid genome index {idx}, using best")
                idx = int(np.argmax(self.fitness))
            
            # Get the selected genome
            selected_genome = self.population[idx].copy()
            
            # Validate selected genome
            if np.any(~np.isfinite(selected_genome)):
                self.log_operator_warning(f"Selected genome contains invalid values: {selected_genome}")
                selected_genome = self._repair_genome(selected_genome)
            
            # Store as active genome
            self.active_genome = selected_genome
            self.active_genome_idx = idx
            
            # Update usage statistics
            self.genome_usage_stats[self.genome_hash(selected_genome)] += 1
            
            # Log selection
            fit_val = self.fitness[idx]
            genome_str = ", ".join(f"{x:.3f}" for x in selected_genome)
            self.log_operator_info(
                f"Genome selected",
                index=idx,
                fitness=f"€{fit_val:.2f}",
                genome=f"[{genome_str}]",
                mode=mode
            )
            
            # Return the genome (numpy array format)
            return selected_genome
            
        except Exception as e:
            self.log_operator_error(f"Genome selection failed: {e}")
            # Return safe fallback
            fallback = self._generate_diverse_genome()
            self.active_genome = fallback
            self.active_genome_idx = 0
            return fallback

    def _smart_selection(self) -> int:
        """Smart selection based on current evolution state"""
        
        try:
            if self.generations_without_improvement < 5:
                # Exploit best when improving
                return int(np.argmax(self.fitness))
            elif self.evolution_analytics.get('population_health', 'fair') == 'poor':
                # Explore when population health is poor
                top_k = min(5, max(1, self.pop_size // 2))
                top_indices = np.argsort(self.fitness)[-top_k:]
                return np.random.choice(top_indices)
            else:
                # Balanced tournament selection
                tournament_size = min(self.tournament_k, len(self.population))
                candidates = np.random.choice(self.pop_size, tournament_size, replace=False)
                return candidates[np.argmax(self.fitness[candidates])]
        except Exception as e:
            self.log_operator_warning(f"Smart selection failed: {e}")
            return 0

    def _diversity_aware_selection(self) -> int:
        """Selection that considers both fitness and diversity"""
        
        try:
            # Calculate diversity contribution of each genome
            diversity_scores = np.zeros(len(self.population))
            
            for i, genome in enumerate(self.population):
                distances = [
                    np.linalg.norm(genome - other) 
                    for j, other in enumerate(self.population) 
                    if i != j
                ]
                diversity_scores[i] = np.mean(distances) if distances else 0
            
            # Combine fitness and diversity (weighted)
            fitness_range = np.max(self.fitness) - np.min(self.fitness)
            diversity_range = np.max(diversity_scores) - np.min(diversity_scores)
            
            if fitness_range > 0:
                normalized_fitness = (self.fitness - np.min(self.fitness)) / fitness_range
            else:
                normalized_fitness = np.ones(len(self.fitness)) * 0.5
            
            if diversity_range > 0:
                normalized_diversity = (diversity_scores - np.min(diversity_scores)) / diversity_range
            else:
                normalized_diversity = np.ones(len(diversity_scores)) * 0.5
            
            combined_score = 0.7 * normalized_fitness + 0.3 * normalized_diversity
            return int(np.argmax(combined_score))
            
        except Exception as e:
            self.log_operator_warning(f"Diversity selection failed: {e}")
            return 0

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation with comprehensive evolution metrics"""
        
        try:
            mean_fitness = float(np.mean(self.fitness))
            max_fitness = float(np.max(self.fitness))
            
            # Validate fitness values
            if not np.isfinite(mean_fitness):
                self.log_operator_warning("Invalid mean fitness")
                mean_fitness = 0.0
            if not np.isfinite(max_fitness):
                self.log_operator_warning("Invalid max fitness")
                max_fitness = 0.0
            
            # Calculate additional metrics
            diversity_score = self._calculate_population_diversity()
            convergence_score = self.evolution_analytics['convergence_score']
            profit_ratio = max_fitness / self.profit_target if self.profit_target > 0 else 0.0
            health_score = {'healthy': 1.0, 'moderate': 0.7, 'fair': 0.5, 'poor': 0.2}.get(
                self.evolution_analytics['population_health'], 0.5)
            
            # Stagnation indicator
            stagnation_score = min(1.0, self.generations_without_improvement / 20.0)
            
            # Evolution activity score
            evolution_activity = min(1.0, self.evolution_analytics['adaptation_events'] / 10.0)
            
            observation = np.array([
                mean_fitness / 100.0,      # Normalized mean fitness
                max_fitness / 100.0,       # Normalized max fitness  
                diversity_score,           # Population diversity
                profit_ratio,              # Profit achievement ratio
                convergence_score,         # Convergence indicator
                health_score,              # Population health
                stagnation_score,          # Stagnation indicator
                evolution_activity,        # Evolution activity
                float(self.epoch) / 100.0, # Normalized generation count
                float(len(self.genome_performance_cache)) / 1000.0  # Cache utilization
            ], dtype=np.float32)
            
            # Final validation
            if np.any(~np.isfinite(observation)):
                self.log_operator_error(f"Invalid genome observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            self.log_operator_debug(
                f"Genome observation: mean={mean_fitness:.1f}, max={max_fitness:.1f}, "
                f"diversity={diversity_score:.3f}, health={health_score:.1f}"
            )
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Genome observation generation failed: {e}")
            return np.full(10, 0.5, dtype=np.float32)



    def _update_info_bus_with_genome_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with genome pool status and analytics"""
        
        try:
            # Prepare comprehensive genome data
            genome_data = {
                'population_size': self.pop_size,
                'current_generation': self.epoch,
                'best_genome': self.best_genome.tolist(),
                'best_fitness': float(self.best_fitness),
                'generations_without_improvement': self.generations_without_improvement,
                'active_genome': self.active_genome.tolist() if self.active_genome is not None else None,
                'evolution_analytics': self.evolution_analytics.copy(),
                'fitness_statistics': {
                    'mean': float(np.mean(self.fitness)),
                    'max': float(np.max(self.fitness)),
                    'min': float(np.min(self.fitness)),
                    'std': float(np.std(self.fitness))
                },
                'selection_strategy': self.current_selection_strategy,
                'genome_bounds': self.genome_bounds.copy(),
                'cache_size': len(self.genome_performance_cache)
            }
            
            # Add to InfoBus
            InfoBusUpdater.add_module_data(info_bus, 'strategy_genome_pool', genome_data)
            
            # Add evolution alerts
            if self.generations_without_improvement > 15:
                InfoBusUpdater.add_alert(
                    info_bus,
                    f"Population stagnant for {self.generations_without_improvement} generations",
                    'strategy_genome_pool',
                    'warning',
                    {'stagnation_count': self.generations_without_improvement}
                )
            
            if self.evolution_analytics['population_health'] == 'poor':
                InfoBusUpdater.add_alert(
                    info_bus,
                    "Poor population health detected",
                    'strategy_genome_pool',
                    'warning',
                    {'health': self.evolution_analytics['population_health']}
                )
            
            if self.best_fitness >= self.profit_target:
                InfoBusUpdater.add_alert(
                    info_bus,
                    f"Genome achieved profit target: €{self.best_fitness:.2f}",
                    'strategy_genome_pool',
                    'info',
                    {'best_fitness': self.best_fitness, 'target': self.profit_target}
                )
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus genome update failed: {e}")

    def get_genome_report(self) -> str:
        """Generate comprehensive genome pool report"""
        
        # Population statistics
        fitness_stats = {
            'mean': np.mean(self.fitness),
            'max': np.max(self.fitness),
            'min': np.min(self.fitness),
            'std': np.std(self.fitness)
        }
        
        # Evolution trend
        if len(self.evolution_analytics['fitness_history']) >= 3:
            recent_fitness = [h['max'] for h in list(self.evolution_analytics['fitness_history'])[-3:]]
            if recent_fitness[-1] > recent_fitness[0] + 10:
                evolution_trend = "📈 Improving"
            elif recent_fitness[-1] < recent_fitness[0] - 10:
                evolution_trend = "📉 Declining"
            else:
                evolution_trend = "➡️ Stable"
        else:
            evolution_trend = "📊 Insufficient data"
        
        # Top genomes
        top_genomes = ""
        top_indices = np.argsort(self.fitness)[-3:][::-1]
        for rank, idx in enumerate(top_indices, 1):
            genome_str = ", ".join(f"{x:.3f}" for x in self.population[idx])
            top_genomes += f"  #{rank}: €{self.fitness[idx]:.2f} [{genome_str}]\n"
        
        return f"""
🧬 STRATEGY GENOME POOL REPORT
═══════════════════════════════════════
📊 Population Overview:
• Generation: {self.epoch}
• Population Size: {self.pop_size}
• Genome Size: {self.genome_size}
• Selection Strategy: {self.current_selection_strategy.title()}

🏆 Performance Metrics:
• Best Fitness: €{self.best_fitness:.2f}
• Target Progress: {(self.best_fitness/self.profit_target)*100:.1f}%
• Population Mean: €{fitness_stats['mean']:.2f}
• Fitness Range: €{fitness_stats['min']:.2f} - €{fitness_stats['max']:.2f}
• Stagnant Generations: {self.generations_without_improvement}

🧬 Evolution Analytics:
• Population Health: {self.evolution_analytics['population_health'].title()}
• Diversity Score: {self._calculate_population_diversity():.3f}
• Convergence Score: {self.evolution_analytics['convergence_score']:.3f}
• Adaptation Events: {self.evolution_analytics['adaptation_events']}
• Evolution Trend: {evolution_trend}

🎯 Top Performing Genomes:
{top_genomes}

⚙️ Technical Details:
• Cache Size: {len(self.genome_performance_cache)} evaluations
• Genome Usage Stats: {len(self.genome_usage_stats)} unique genomes used
• Evaluation History: {len(self.genome_evaluation_history)} records

🎚️ Current Parameters:
• Mutation Rate: {self.mut_rate:.1%}
• Crossover Rate: {self.cx_rate:.1%}
• Tournament Size: {self.tournament_k}
• Profit Target: €{self.profit_target}
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "population_size": self.pop_size,
                "tournament_k": self.tournament_k,
                "crossover_rate": self.cx_rate,
                "mutation_rate": self.mut_rate,
                "mutation_scale": self.mut_scale,
                "max_generations_kept": self.max_generations_kept,
                "debug": self.debug,
                "profit_target": self.profit_target,
                "genome_size": self.genome_size
            },
            "evolution_state": {
                "population": self.population.tolist(),
                "fitness": self.fitness.tolist(),
                "epoch": self.epoch,
                "best_genome": self.best_genome.tolist(),
                "best_fitness": float(self.best_fitness),
                "generations_without_improvement": self.generations_without_improvement,
                "active_genome": self.active_genome.tolist() if self.active_genome is not None else None,
                "active_genome_idx": self.active_genome_idx,
                "current_selection_strategy": self.current_selection_strategy
            },
            "analytics": {
                "evolution_analytics": self.evolution_analytics.copy(),
                "genome_usage_stats": dict(self.genome_usage_stats),
                "genome_performance_cache": dict(self.genome_performance_cache),
                "evaluation_history": list(self.genome_evaluation_history)[-50:]  # Keep recent only
            },
            "bounds": self.genome_bounds.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.pop_size = int(config.get("population_size", self.pop_size))
        self.tournament_k = int(config.get("tournament_k", self.tournament_k))
        self.cx_rate = float(config.get("crossover_rate", self.cx_rate))
        self.mut_rate = float(config.get("mutation_rate", self.mut_rate))
        self.mut_scale = float(config.get("mutation_scale", self.mut_scale))
        self.max_generations_kept = int(config.get("max_generations_kept", self.max_generations_kept))
        self.debug = bool(config.get("debug", self.debug))
        self.profit_target = float(config.get("profit_target", self.profit_target))
        self.genome_size = int(config.get("genome_size", self.genome_size))
        
        # Load evolution state
        evolution_state = state.get("evolution_state", {})
        self.population = np.array(evolution_state.get("population", self.population), dtype=np.float32)
        self.fitness = np.array(evolution_state.get("fitness", self.fitness), dtype=np.float32)
        self.epoch = int(evolution_state.get("epoch", 0))
        self.best_genome = np.array(evolution_state.get("best_genome", self.best_genome), dtype=np.float32)
        self.best_fitness = float(evolution_state.get("best_fitness", -np.inf))
        self.generations_without_improvement = int(evolution_state.get("generations_without_improvement", 0))
        
        active_genome_data = evolution_state.get("active_genome")
        if active_genome_data:
            self.active_genome = np.array(active_genome_data, dtype=np.float32)
        
        self.active_genome_idx = int(evolution_state.get("active_genome_idx", 0))
        self.current_selection_strategy = evolution_state.get("current_selection_strategy", "adaptive")
        
        # Load analytics
        analytics = state.get("analytics", {})
        self.evolution_analytics.update(analytics.get("evolution_analytics", {}))
        self.genome_usage_stats = defaultdict(int, analytics.get("genome_usage_stats", {}))
        self.genome_performance_cache = analytics.get("genome_performance_cache", {})
        
        evaluation_history = analytics.get("evaluation_history", [])
        self.genome_evaluation_history = deque(evaluation_history, maxlen=self.pop_size * 5)
        
        # Load bounds if provided
        self.genome_bounds.update(state.get("bounds", {}))
        
        self.log_operator_info(
            f"🔄 Genome pool state loaded",
            generation=self.epoch,
            best_fitness=f"€{self.best_fitness:.2f}",
            population_size=len(self.population)
        )