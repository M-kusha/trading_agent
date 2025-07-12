"""
🧬 Enhanced Strategy Genome Pool with SmartInfoBus Integration v3.0
Advanced genetic algorithm evolution system with intelligent strategy parameter optimization
"""

import asyncio
import time
import hashlib
import numpy as np
import datetime
import random
from typing import Dict, Any, List, Optional, Tuple, Callable, Union
from collections import deque, defaultdict

# ═══════════════════════════════════════════════════════════════════
# MODERN SMARTINFOBUS IMPORTS
# ═══════════════════════════════════════════════════════════════════
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker


@module(
    name="StrategyGenomePool",
    version="3.0.0",
    category="strategy",
    provides=[
        "genome_weights", "genome_analysis", "genome_recommendations",
        "evolution_analytics", "best_genome", "population_metrics"
    ],
    requires=[
        "market_data", "recent_trades", "trading_performance", "risk_data",
        "market_regime", "volatility_data", "session_metrics"
    ],
    description="Advanced genetic algorithm evolution system with intelligent strategy parameter optimization",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True,
    timeout_ms=200,
    priority=7,
    explainable=True,
    hot_reload=True
)
class StrategyGenomePool(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusStateMixin):
    """
    🧬 PRODUCTION-GRADE Strategy Genome Pool v3.0
    
    Advanced genetic algorithm evolution system with:
    - Intelligent population management and genetic operations
    - Performance-driven evolution with adaptive parameters
    - Market context-aware selection and mutation strategies
    - SmartInfoBus zero-wiring architecture
    - Comprehensive thesis generation for all evolution decisions
    """

    def _initialize(self):
        """Initialize advanced genetic evolution and population management systems"""
        # Initialize base mixins
        self._initialize_trading_state()
        self._initialize_state_management()
        self._initialize_advanced_systems()
        
        # Enhanced configuration
        self.population_size = self.config.get('population_size', 20)
        self.tournament_k = self.config.get('tournament_k', 3)
        self.crossover_rate = self.config.get('crossover_rate', 0.5)
        self.mutation_rate = self.config.get('mutation_rate', 0.1)
        self.mutation_scale = self.config.get('mutation_scale', 0.2)
        self.max_generations_kept = self.config.get('max_generations_kept', 10000)
        self.profit_target = self.config.get('profit_target', 150.0)
        self.genome_size = self.config.get('genome_size', 4)
        self.debug = self.config.get('debug', False)
        
        # Enhanced genome bounds for XAUUSD/EURUSD trading
        self.genome_bounds = {
            'sl_base': (0.2, 3.0),      # Stop Loss: 20-300 pips
            'tp_base': (0.3, 4.0),      # Take Profit: 30-400 pips
            'vol_scale': (0.1, 2.5),    # Volatility scaling
            'regime_adapt': (0.0, 0.8), # Regime adaptation factor
        }
        
        # Core evolution state
        self.population = self._initialize_population()
        self.fitness = np.zeros(self.population_size, dtype=np.float32)
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
        self.genome_evaluation_history = deque(maxlen=self.population_size * 5)
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
        
        # Circuit breaker for error handling
        self.error_count = 0
        self.circuit_breaker_threshold = 5
        self.is_disabled = False
        
        # Advanced evolution intelligence
        self.evolution_intelligence = {
            'adaptation_sensitivity': 0.8,
            'diversity_pressure': 0.9,
            'fitness_decay': 0.95,
            'evolution_momentum': 0.85
        }
        
        # Generate initialization thesis
        self._generate_initialization_thesis()
        
        version = getattr(self.metadata, 'version', '3.0.0') if self.metadata else '3.0.0'
        self.logger.info(format_operator_message(
            icon="🧬",
            message=f"Strategy Genome Pool v{version} initialized",
            population=self.population_size,
            genome_size=self.genome_size,
            profit_target=f"€{self.profit_target}",
            mutation_rate=f"{self.mutation_rate:.1%}"
        ))

    def _initialize_advanced_systems(self):
        """Initialize all modern system components"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="StrategyGenomePool",
            log_path="logs/strategy/genome_pool.log",
            max_lines=2000,
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("StrategyGenomePool", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()

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
        remaining_spots = max(0, self.population_size - len(seed_genomes))
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
        population = np.array(all_genomes[:self.population_size], dtype=np.float32)
        
        self.logger.info(format_operator_message(
            icon="🌱",
            message="Population initialized",
            seed_genomes=len(seed_genomes),
            random_genomes=len(random_genomes),
            total_size=len(population)
        ))
        
        return population

    def _generate_initialization_thesis(self):
        """Generate comprehensive initialization thesis"""
        thesis = f"""
        Strategy Genome Pool v3.0 Initialization Complete:
        
        Advanced Genetic Evolution System:
        - Population size: {self.population_size} diverse genomes with intelligent seeding
        - Genome structure: {self.genome_size} parameters optimizing SL, TP, volatility, and regime adaptation
        - Evolution strategies: {len(self.selection_strategies)} adaptive selection methods
        - Performance target: €{self.profit_target} with bonus fitness scaling
        
        Current Configuration:
        - Crossover rate: {self.crossover_rate:.1%} for genetic recombination
        - Mutation rate: {self.mutation_rate:.1%} with {self.mutation_scale:.1%} scale
        - Tournament size: {self.tournament_k} genomes for competitive selection
        - Generation memory: {self.max_generations_kept} generations tracked
        
        Evolution Intelligence Features:
        - Performance-driven fitness evaluation with caching
        - Adaptive parameter tuning based on population health
        - Market context-aware selection pressure adjustment
        - Multi-strategy genetic operations (uniform, arithmetic, polynomial)
        
        Advanced Capabilities:
        - Real-time population diversity monitoring and health assessment
        - Intelligent strategy adaptation based on market conditions
        - Comprehensive analytics tracking for evolution optimization
        - Circuit breaker protection against evolution failures
        
        Expected Outcomes:
        - Optimal trading parameter evolution through genetic algorithms
        - Enhanced performance through intelligent population management
        - Adaptive evolution that responds to market regime changes
        - Transparent evolution decisions with comprehensive explanations
        """
        
        self.smart_bus.set('strategy_genome_pool_initialization', {
            'status': 'initialized',
            'thesis': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'configuration': {
                'population_size': self.population_size,
                'genome_bounds': self.genome_bounds,
                'selection_strategies': list(self.selection_strategies.keys())
            }
        }, module='StrategyGenomePool', thesis=thesis)

    async def process(self) -> Dict[str, Any]:
        """
        Modern async processing with comprehensive genome evolution analysis
        
        Returns:
            Dict containing genome weights, evolution analysis, and recommendations
        """
        start_time = time.time()
        
        try:
            # Circuit breaker check
            if self.is_disabled:
                return self._generate_disabled_response()
            
            # Get comprehensive market data from SmartInfoBus
            market_data = await self._get_comprehensive_market_data()
            
            # Core evolution analysis with error handling
            evolution_analysis = await self._analyze_genome_evolution_comprehensive(market_data)
            
            # Update population performance based on recent results
            await self._update_population_performance_comprehensive(market_data, evolution_analysis)
            
            # Evolve population with intelligent algorithms
            evolution_results = await self._evolve_population_intelligent(evolution_analysis, market_data)
            
            # Generate comprehensive thesis
            thesis = await self._generate_comprehensive_evolution_thesis(evolution_analysis, evolution_results)
            
            # Create comprehensive results
            results = {
                'genome_weights': self._get_current_genome_weights(),
                'genome_analysis': evolution_analysis,
                'genome_recommendations': self._generate_intelligent_recommendations(evolution_analysis),
                'evolution_analytics': self.evolution_analytics.copy(),
                'best_genome': {
                    'parameters': self.best_genome.tolist(),
                    'fitness': float(self.best_fitness),
                    'generation': self.epoch
                },
                'population_metrics': self._get_population_metrics(),
                'health_metrics': self._get_health_metrics()
            }
            
            # Update SmartInfoBus with comprehensive thesis
            await self._update_smartinfobus_comprehensive(results, thesis)
            
            # Record performance metrics
            processing_time = (time.time() - start_time) * 1000
            self.performance_tracker.record_metric('StrategyGenomePool', 'process_time', processing_time, True)
            
            # Reset error count on successful processing
            self.error_count = 0
            
            return results
            
        except Exception as e:
            return await self._handle_processing_error(e, start_time)

    async def _get_comprehensive_market_data(self) -> Dict[str, Any]:
        """Get comprehensive market data using modern SmartInfoBus patterns"""
        try:
            return {
                'market_data': self.smart_bus.get('market_data', 'StrategyGenomePool') or {},
                'recent_trades': self.smart_bus.get('recent_trades', 'StrategyGenomePool') or [],
                'trading_performance': self.smart_bus.get('trading_performance', 'StrategyGenomePool') or {},
                'risk_data': self.smart_bus.get('risk_data', 'StrategyGenomePool') or {},
                'market_regime': self.smart_bus.get('market_regime', 'StrategyGenomePool') or 'unknown',
                'volatility_data': self.smart_bus.get('volatility_data', 'StrategyGenomePool') or {},
                'session_metrics': self.smart_bus.get('session_metrics', 'StrategyGenomePool') or {},
                'market_context': self.smart_bus.get('market_context', 'StrategyGenomePool') or {}
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "StrategyGenomePool")
            self.logger.warning(f"Market data retrieval incomplete: {error_context}")
            return self._get_safe_market_defaults()

    async def _analyze_genome_evolution_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive genome evolution analysis with advanced algorithms"""
        try:
            analysis = {
                'evolution_context': {},
                'population_health': {},
                'fitness_analysis': {},
                'diversity_metrics': {},
                'selection_pressure': {},
                'analysis_timestamp': datetime.datetime.now().isoformat()
            }
            
            # Extract evolution context from market data
            evolution_context = await self._extract_evolution_context_comprehensive(market_data)
            analysis['evolution_context'] = evolution_context
            
            # Analyze population health
            population_health = await self._analyze_population_health_comprehensive()
            analysis['population_health'] = population_health
            
            # Fitness landscape analysis
            fitness_analysis = await self._analyze_fitness_landscape(market_data)
            analysis['fitness_analysis'] = fitness_analysis
            
            # Diversity and convergence metrics
            diversity_metrics = await self._calculate_diversity_metrics_comprehensive()
            analysis['diversity_metrics'] = diversity_metrics
            
            # Selection pressure assessment
            selection_pressure = await self._assess_selection_pressure(evolution_context)
            analysis['selection_pressure'] = selection_pressure
            
            # Log significant analysis results
            await self._log_significant_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "StrategyGenomePool")
            self.logger.error(f"Evolution analysis failed: {error_context}")
            return self._get_safe_analysis_defaults()

    async def _extract_evolution_context_comprehensive(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive evolution context from market data"""
        try:
            recent_trades = market_data.get('recent_trades', [])
            risk_data = market_data.get('risk_data', {})
            trading_performance = market_data.get('trading_performance', {})
            
            evolution_context = {
                'timestamp': datetime.datetime.now().isoformat(),
                'session_pnl': trading_performance.get('session_pnl', 0),
                'recent_trades_count': len(recent_trades),
                'market_regime': market_data.get('market_regime', 'unknown'),
                'volatility_level': market_data.get('market_context', {}).get('volatility_level', 'medium'),
                'drawdown': risk_data.get('current_drawdown', 0),
                'balance': risk_data.get('balance', 0),
                'performance_trend': self._calculate_performance_trend(recent_trades),
                'market_stress_level': self._assess_market_stress(market_data, risk_data),
                'evolution_pressure': self._calculate_evolution_pressure(trading_performance)
            }
            
            return evolution_context
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "evolution_context")
            self.logger.warning(f"Evolution context extraction failed: {error_context}")
            return {'timestamp': datetime.datetime.now().isoformat(), 'extraction_error': str(error_context)}

    def _calculate_performance_trend(self, recent_trades: List[Dict]) -> str:
        """Calculate recent performance trend with enhanced analysis"""
        try:
            if not recent_trades or len(recent_trades) < 5:
                return 'insufficient_data'
            
            # Look at last 10 trades
            recent_pnls = [t.get('pnl', 0) for t in recent_trades[-10:]]
            
            # Calculate trend with momentum
            if len(recent_pnls) >= 3:
                recent_avg = np.mean(recent_pnls[-3:])
                older_avg = np.mean(recent_pnls[:-3]) if len(recent_pnls) > 3 else recent_avg
                
                trend_strength = abs(recent_avg - older_avg)
                
                if recent_avg > older_avg + 10:
                    return 'strongly_improving' if trend_strength > 25 else 'improving'
                elif recent_avg < older_avg - 10:
                    return 'strongly_declining' if trend_strength > 25 else 'declining'
                else:
                    return 'stable'
            
            return 'stable'
            
        except Exception:
            return 'unknown'

    def _assess_market_stress(self, market_data: Dict[str, Any], risk_data: Dict[str, Any]) -> str:
        """Assess current market stress level with enhanced factors"""
        stress_score = 0
        
        # Volatility stress
        volatility_level = market_data.get('market_context', {}).get('volatility_level', 'medium')
        stress_mapping = {'low': 0, 'medium': 1, 'high': 2, 'extreme': 4}
        stress_score += stress_mapping.get(volatility_level, 1)
        
        # Drawdown stress
        drawdown = risk_data.get('current_drawdown', 0)
        if drawdown > 0.15:
            stress_score += 4
        elif drawdown > 0.1:
            stress_score += 3
        elif drawdown > 0.05:
            stress_score += 2
        
        # Market regime stress
        regime = market_data.get('market_regime', 'unknown')
        if regime in ['volatile', 'unknown']:
            stress_score += 1
        elif regime == 'crisis':
            stress_score += 3
        
        # Performance stress
        session_pnl = market_data.get('trading_performance', {}).get('session_pnl', 0)
        if session_pnl < -100:
            stress_score += 2
        elif session_pnl < -50:
            stress_score += 1
        
        # Classify stress level
        if stress_score <= 1:
            return 'low'
        elif stress_score <= 3:
            return 'medium'
        elif stress_score <= 6:
            return 'high'
        else:
            return 'extreme'

    def _calculate_evolution_pressure(self, trading_performance: Dict[str, Any]) -> float:
        """Calculate evolution pressure based on performance metrics"""
        try:
            base_pressure = 0.5
            
            # Performance-based pressure
            session_pnl = trading_performance.get('session_pnl', 0)
            if session_pnl < -50:
                base_pressure += 0.3  # Increase pressure for poor performance
            elif session_pnl > 100:
                base_pressure -= 0.2  # Reduce pressure for good performance
            
            # Win rate pressure
            win_rate = trading_performance.get('win_rate', 0.5)
            if win_rate < 0.4:
                base_pressure += 0.2
            elif win_rate > 0.7:
                base_pressure -= 0.1
            
            # Stagnation pressure
            if self.generations_without_improvement > 10:
                base_pressure += 0.4
            
            return np.clip(base_pressure, 0.1, 1.0)
            
        except Exception:
            return 0.5

    async def _analyze_population_health_comprehensive(self) -> Dict[str, Any]:
        """Comprehensive population health analysis"""
        try:
            health_metrics = {}
            
            # Diversity health
            diversity_score = self._calculate_population_diversity()
            health_metrics['diversity'] = {
                'score': diversity_score,
                'status': 'healthy' if diversity_score > 1.0 else 'poor' if diversity_score < 0.3 else 'moderate'
            }
            
            # Fitness health
            if len(self.fitness) > 0:
                fitness_std = float(np.std(self.fitness))
                fitness_mean = float(np.mean(self.fitness))
                health_metrics['fitness_distribution'] = {
                    'std': fitness_std,
                    'mean': fitness_mean,
                    'coefficient_variation': fitness_std / (abs(fitness_mean) + 1e-6),
                    'status': 'healthy' if fitness_std > 10 else 'converging'
                }
            
            # Evolution health
            health_metrics['evolution'] = {
                'stagnant_generations': self.generations_without_improvement,
                'status': 'healthy' if self.generations_without_improvement < 10 else 'stagnant'
            }
            
            # Overall health assessment
            health_statuses = [m.get('status', 'unknown') for m in health_metrics.values() if isinstance(m, dict)]
            if 'poor' in health_statuses:
                overall_health = 'poor'
            elif 'stagnant' in health_statuses or 'converging' in health_statuses:
                overall_health = 'moderate'
            else:
                overall_health = 'healthy'
            
            health_metrics['overall'] = overall_health
            
            return health_metrics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "population_health")
            return {'overall': 'unknown', 'error': str(error_context)}

    async def _analyze_fitness_landscape(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze fitness landscape and optimization potential"""
        try:
            analysis = {}
            
            if len(self.fitness) > 0:
                # Basic fitness statistics
                analysis['statistics'] = {
                    'min': float(np.min(self.fitness)),
                    'max': float(np.max(self.fitness)),
                    'mean': float(np.mean(self.fitness)),
                    'std': float(np.std(self.fitness)),
                    'range': float(np.max(self.fitness) - np.min(self.fitness))
                }
                
                # Fitness improvement potential
                target_achievement = analysis['statistics']['max'] / self.profit_target
                analysis['optimization_potential'] = {
                    'target_achievement': target_achievement,
                    'improvement_needed': max(0, self.profit_target - analysis['statistics']['max']),
                    'optimization_pressure': 1.0 - min(1.0, target_achievement)
                }
                
                # Elite genome analysis
                elite_count = max(1, self.population_size // 10)
                elite_indices = np.argsort(self.fitness)[-elite_count:]
                analysis['elite_analysis'] = {
                    'count': elite_count,
                    'avg_fitness': float(np.mean(self.fitness[elite_indices])),
                    'diversity': self._calculate_elite_diversity(elite_indices)
                }
            
            return analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "fitness_landscape")
            return {'error': str(error_context)}

    def _calculate_population_diversity(self) -> float:
        """Calculate population diversity score with enhanced metrics"""
        try:
            if len(self.population) < 2:
                return 0.0
            
            # Calculate pairwise distances
            distances = []
            for i in range(len(self.population)):
                for j in range(i + 1, len(self.population)):
                    # Euclidean distance
                    euclidean_dist = np.linalg.norm(self.population[i] - self.population[j])
                    # Manhattan distance for comparison
                    manhattan_dist = np.sum(np.abs(self.population[i] - self.population[j]))
                    # Combined distance metric
                    combined_dist = 0.7 * euclidean_dist + 0.3 * manhattan_dist
                    distances.append(combined_dist)
            
            return float(np.mean(distances)) if distances else 0.0
            
        except Exception:
            return 0.0

    def _calculate_elite_diversity(self, elite_indices: np.ndarray) -> float:
        """Calculate diversity within elite population"""
        try:
            if len(elite_indices) < 2:
                return 0.0
            
            elite_genomes = self.population[elite_indices]
            distances = []
            
            for i in range(len(elite_genomes)):
                for j in range(i + 1, len(elite_genomes)):
                    dist = np.linalg.norm(elite_genomes[i] - elite_genomes[j])
                    distances.append(dist)
            
            return float(np.mean(distances)) if distances else 0.0
            
        except Exception:
            return 0.0

    async def _calculate_diversity_metrics_comprehensive(self) -> Dict[str, Any]:
        """Calculate comprehensive diversity metrics"""
        try:
            metrics = {}
            
            # Population diversity
            metrics['population_diversity'] = self._calculate_population_diversity()
            
            # Convergence analysis
            convergence_score = self._calculate_convergence_score()
            metrics['convergence'] = {
                'score': convergence_score,
                'status': 'converged' if convergence_score < 0.1 else 'diverse' if convergence_score > 0.5 else 'balanced'
            }
            
            # Parameter-specific diversity
            if len(self.population) > 0:
                param_diversity = {}
                for i, param_name in enumerate(['sl_base', 'tp_base', 'vol_scale', 'regime_adapt']):
                    param_values = self.population[:, i]
                    param_diversity[param_name] = {
                        'std': float(np.std(param_values)),
                        'range': float(np.max(param_values) - np.min(param_values)),
                        'coefficient_variation': float(np.std(param_values) / (np.mean(param_values) + 1e-6))
                    }
                metrics['parameter_diversity'] = param_diversity
            
            return metrics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "diversity_metrics")
            return {'error': str(error_context)}

    def _calculate_convergence_score(self) -> float:
        """Calculate convergence score (lower = more converged)"""
        try:
            if len(self.fitness) < 2:
                return 1.0
            
            fitness_std = np.std(self.fitness)
            fitness_mean = abs(np.mean(self.fitness))
            
            # Normalized standard deviation
            convergence = float(fitness_std / (fitness_mean + 1e-6))
            return min(1.0, convergence)
            
        except Exception:
            return 0.5

    async def _assess_selection_pressure(self, evolution_context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess current selection pressure and adaptation needs"""
        try:
            pressure_analysis = {}
            
            # Performance-based pressure
            performance_trend = evolution_context.get('performance_trend', 'stable')
            market_stress = evolution_context.get('market_stress_level', 'medium')
            evolution_pressure = evolution_context.get('evolution_pressure', 0.5)
            
            # Calculate recommended selection strategy
            if performance_trend in ['strongly_declining', 'declining'] or market_stress == 'extreme':
                recommended_strategy = 'diversity'
                pressure_level = 'high'
            elif performance_trend in ['strongly_improving', 'improving'] and market_stress == 'low':
                recommended_strategy = 'tournament'
                pressure_level = 'low'
            elif self.generations_without_improvement > 15:
                recommended_strategy = 'adaptive'
                pressure_level = 'high'
            else:
                recommended_strategy = 'rank'
                pressure_level = 'medium'
            
            pressure_analysis = {
                'current_strategy': self.current_selection_strategy,
                'recommended_strategy': recommended_strategy,
                'pressure_level': pressure_level,
                'evolution_pressure': evolution_pressure,
                'adaptation_needed': recommended_strategy != self.current_selection_strategy,
                'factors': {
                    'performance_trend': performance_trend,
                    'market_stress': market_stress,
                    'stagnation': self.generations_without_improvement > 10
                }
            }
            
            return pressure_analysis
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "selection_pressure")
            return {'error': str(error_context)}

    async def _log_significant_analysis(self, analysis: Dict[str, Any]):
        """Log significant analysis results"""
        try:
            population_health = analysis.get('population_health', {})
            overall_health = population_health.get('overall', 'unknown')
            
            if overall_health == 'poor':
                self.logger.warning(format_operator_message(
                    icon="🏥",
                    message="Population health declining",
                    health=overall_health,
                    stagnation=self.generations_without_improvement
                ))
            
            fitness_analysis = analysis.get('fitness_analysis', {})
            optimization_potential = fitness_analysis.get('optimization_potential', {})
            target_achievement = optimization_potential.get('target_achievement', 0)
            
            if target_achievement >= 1.0:
                self.logger.info(format_operator_message(
                    icon="🎯",
                    message="Profit target achieved",
                    achievement=f"{target_achievement:.1%}",
                    best_fitness=f"€{self.best_fitness:.2f}"
                ))
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "analysis_logging")

    async def _update_population_performance_comprehensive(self, market_data: Dict[str, Any], 
                                                         evolution_analysis: Dict[str, Any]):
        """Update population performance with comprehensive tracking"""
        try:
            recent_trades = market_data.get('recent_trades', [])
            
            if not recent_trades:
                return
            
            # Get the most recent trade result
            last_trade = recent_trades[-1]
            pnl = last_trade.get('pnl', 0)
            
            # Update active genome performance if available
            if self.active_genome is not None and self.active_genome_idx is not None:
                await self._record_genome_result_comprehensive(
                    self.active_genome_idx, pnl, evolution_analysis
                )
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "performance_update")
            self.logger.warning(f"Population performance update failed: {error_context}")

    async def _record_genome_result_comprehensive(self, genome_idx: int, pnl: float, 
                                                evolution_analysis: Dict[str, Any]):
        """Record comprehensive genome result with enhanced analytics"""
        try:
            # Validate inputs
            if not (0 <= genome_idx < len(self.population)):
                return
            
            if np.isnan(pnl):
                return
            
            # Apply profit target bonus
            if pnl >= self.profit_target:
                profit_bonus = 1.0 + (pnl - self.profit_target) / self.profit_target
                final_fitness = pnl * profit_bonus
                
                self.logger.info(format_operator_message(
                    icon="💰",
                    message=f"Genome {genome_idx} achieved target",
                    raw_pnl=f"€{pnl:.2f}",
                    bonus_multiplier=f"{profit_bonus:.2f}x",
                    final_fitness=f"€{final_fitness:.2f}"
                ))
            else:
                final_fitness = pnl
            
            # Update fitness
            old_fitness = self.fitness[genome_idx]
            self.fitness[genome_idx] = final_fitness
            
            # Update best genome tracking
            if final_fitness > self.best_fitness:
                improvement = final_fitness - self.best_fitness
                self.best_fitness = final_fitness
                self.best_genome = self.population[genome_idx].copy()
                self.generations_without_improvement = 0
                
                self.logger.info(format_operator_message(
                    icon="🏆",
                    message="New best genome found",
                    improvement=f"€{improvement:.2f}",
                    new_fitness=f"€{final_fitness:.2f}",
                    generation=self.epoch
                ))
            
            # Record in evaluation history
            evaluation_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'genome_idx': genome_idx,
                'pnl': pnl,
                'final_fitness': final_fitness,
                'improvement': final_fitness - old_fitness,
                'evolution_context': evolution_analysis.get('evolution_context', {})
            }
            self.genome_evaluation_history.append(evaluation_record)
            
            # Update usage statistics
            genome_hash = self.genome_hash(self.population[genome_idx])
            self.genome_usage_stats[genome_hash] += 1
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "genome_result_recording")
            self.logger.error(f"Genome result recording failed: {error_context}")

    def genome_hash(self, genome: np.ndarray) -> str:
        """Generate hash for genome tracking with enhanced precision"""
        try:
            # Round to reasonable precision to avoid minor floating point differences
            rounded_genome = np.round(genome, decimals=4)
            return hashlib.md5(rounded_genome.tobytes()).hexdigest()
        except Exception as e:
            self.logger.warning(f"Genome hashing failed: {e}")
            return "error_hash"

    async def _evolve_population_intelligent(self, evolution_analysis: Dict[str, Any], 
                                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evolve population using intelligent algorithms"""
        try:
            evolution_results = {
                'strategy_adaptation': {},
                'population_changes': {},
                'evolution_metrics': {},
                'adaptation_success': False
            }
            
            # Adapt selection strategy based on analysis
            strategy_adaptation = await self._adapt_selection_strategy_intelligent(evolution_analysis)
            evolution_results['strategy_adaptation'] = strategy_adaptation
            
            # Check if evolution is needed
            evolution_pressure = evolution_analysis.get('evolution_context', {}).get('evolution_pressure', 0.5)
            population_health = evolution_analysis.get('population_health', {}).get('overall', 'unknown')
            
            should_evolve = (
                evolution_pressure > 0.6 or
                population_health == 'poor' or
                self.generations_without_improvement > 5
            )
            
            if should_evolve:
                # Perform population evolution
                population_changes = await self._perform_population_evolution(evolution_analysis)
                evolution_results['population_changes'] = population_changes
                evolution_results['adaptation_success'] = True
                
                self.logger.info(format_operator_message(
                    icon="🧬",
                    message="Population evolution performed",
                    generation=self.epoch,
                    pressure=f"{evolution_pressure:.2f}",
                    health=population_health
                ))
            
            # Update evolution analytics
            await self._update_evolution_analytics_comprehensive(evolution_analysis, evolution_results)
            
            return evolution_results
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "population_evolution")
            self.logger.error(f"Population evolution failed: {error_context}")
            return {'error': str(error_context)}

    async def _adapt_selection_strategy_intelligent(self, evolution_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Adapt selection strategy based on intelligent analysis"""
        try:
            selection_pressure = evolution_analysis.get('selection_pressure', {})
            recommended_strategy = selection_pressure.get('recommended_strategy', 'adaptive')
            adaptation_needed = selection_pressure.get('adaptation_needed', False)
            
            adaptation_info = {
                'old_strategy': self.current_selection_strategy,
                'new_strategy': recommended_strategy,
                'adapted': False,
                'reason': 'no_change_needed'
            }
            
            if adaptation_needed:
                self.current_selection_strategy = recommended_strategy
                self.evolution_analytics['adaptation_events'] += 1
                adaptation_info['adapted'] = True
                adaptation_info['reason'] = 'performance_optimization'
                
                factors = selection_pressure.get('factors', {})
                self.logger.info(format_operator_message(
                    icon="🎯",
                    message="Selection strategy adapted",
                    from_strategy=adaptation_info['old_strategy'],
                    to_strategy=recommended_strategy,
                    factors=str(factors)
                ))
            
            return adaptation_info
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "strategy_adaptation")
            return {'error': str(error_context)}

    async def _perform_population_evolution(self, evolution_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Perform actual population evolution with adaptive parameters"""
        try:
            # Calculate adaptive evolution parameters
            adaptive_params = self._calculate_adaptive_parameters_comprehensive(evolution_analysis)
            
            # Track pre-evolution state
            old_diversity = self._calculate_population_diversity()
            old_hashes = [self.genome_hash(g) for g in self.population]
            
            # Create new population
            new_population = await self._create_new_population_intelligent(adaptive_params)
            
            # Update population
            self.population = new_population
            self.fitness[:] = 0.0  # Reset fitness for new population
            self.epoch += 1
            
            # Track post-evolution state
            new_diversity = self._calculate_population_diversity()
            new_hashes = [self.genome_hash(g) for g in self.population]
            
            # Calculate evolution statistics
            evolution_stats = {
                'genomes_changed': sum(1 for o, n in zip(old_hashes, new_hashes) if o != n),
                'change_percentage': (sum(1 for o, n in zip(old_hashes, new_hashes) if o != n) / self.population_size) * 100,
                'diversity_change': new_diversity - old_diversity,
                'old_diversity': old_diversity,
                'new_diversity': new_diversity,
                'generation': self.epoch,
                'adaptive_params': adaptive_params
            }
            
            self.logger.info(format_operator_message(
                icon="🧬",
                message="Population evolution complete",
                generation=self.epoch,
                changed=f"{evolution_stats['genomes_changed']}/{self.population_size}",
                diversity_change=f"{evolution_stats['diversity_change']:+.3f}"
            ))
            
            return evolution_stats
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "population_evolution_perform")
            return {'error': str(error_context)}

    def _calculate_adaptive_parameters_comprehensive(self, evolution_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calculate adaptive evolution parameters based on comprehensive analysis"""
        try:
            # Base parameters
            adaptive_params = {
                'mutation_rate': self.mutation_rate,
                'mutation_scale': self.mutation_scale,
                'crossover_rate': self.crossover_rate,
                'elitism_count': max(1, self.population_size // 10)
            }
            
            # Evolution context adjustments
            evolution_context = evolution_analysis.get('evolution_context', {})
            performance_trend = evolution_context.get('performance_trend', 'stable')
            market_stress = evolution_context.get('market_stress_level', 'medium')
            evolution_pressure = evolution_context.get('evolution_pressure', 0.5)
            
            # Population health adjustments
            population_health = evolution_analysis.get('population_health', {})
            overall_health = population_health.get('overall', 'healthy')
            
            # Adaptive adjustments
            if performance_trend in ['strongly_declining', 'declining']:
                adaptive_params['mutation_rate'] = min(0.5, self.mutation_rate * 2.0)
                adaptive_params['mutation_scale'] = min(0.6, self.mutation_scale * 1.5)
            
            if overall_health == 'poor':
                adaptive_params['mutation_rate'] = min(0.6, adaptive_params['mutation_rate'] * 1.5)
                adaptive_params['crossover_rate'] = max(0.2, adaptive_params['crossover_rate'] * 0.8)
            
            if market_stress == 'extreme':
                adaptive_params['elitism_count'] = max(2, self.population_size // 5)  # Preserve more elite
            
            # Stagnation adjustments
            if self.generations_without_improvement > 10:
                adaptive_params['mutation_rate'] = min(0.7, adaptive_params['mutation_rate'] * 2.0)
                adaptive_params['mutation_scale'] = min(0.8, adaptive_params['mutation_scale'] * 1.8)
            
            # Convergence adjustments
            diversity_metrics = evolution_analysis.get('diversity_metrics', {})
            convergence = diversity_metrics.get('convergence', {})
            if convergence.get('status') == 'converged':
                adaptive_params['mutation_rate'] = min(0.8, adaptive_params['mutation_rate'] * 3.0)
            
            return adaptive_params
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "adaptive_parameters")
            return {
                'mutation_rate': self.mutation_rate,
                'mutation_scale': self.mutation_scale,
                'crossover_rate': self.crossover_rate,
                'elitism_count': max(1, self.population_size // 10)
            }

    async def _create_new_population_intelligent(self, adaptive_params: Dict[str, float]) -> np.ndarray:
        """Create new population using intelligent evolution strategies"""
        try:
            new_pop = []
            evolution_stats = {'crossovers': 0, 'mutations': 0, 'elite_preserved': 0, 'diversity_injected': 0}
            
            # Elitism: preserve best genomes
            elite_count = adaptive_params['elitism_count']
            if len(self.fitness) > 0:
                elite_indices = np.argsort(self.fitness)[-elite_count:]
                for idx in elite_indices:
                    new_pop.append(self.population[idx].copy())
                    evolution_stats['elite_preserved'] += 1
            
            # Diversity injection for poor health
            population_health = getattr(self, '_last_population_health', 'healthy')
            if population_health == 'poor':
                diversity_inject_count = min(3, self.population_size // 10)
                for _ in range(diversity_inject_count):
                    diverse_genome = self._generate_diverse_genome()
                    new_pop.append(diverse_genome)
                    evolution_stats['diversity_injected'] += 1
            
            # Generate rest of population
            while len(new_pop) < self.population_size:
                try:
                    # Selection
                    parent1 = self._select_parent_intelligent()
                    parent2 = self._select_parent_intelligent()
                    
                    # Crossover
                    if np.random.random() < adaptive_params['crossover_rate']:
                        child = self._crossover_intelligent(parent1, parent2)
                        evolution_stats['crossovers'] += 1
                    else:
                        child = parent1.copy()
                    
                    # Mutation
                    if np.random.random() < adaptive_params['mutation_rate']:
                        child = self._mutate_intelligent(child, adaptive_params['mutation_scale'])
                        evolution_stats['mutations'] += 1
                    
                    # Validate and repair
                    child = self._repair_genome(child)
                    new_pop.append(child)
                    
                except Exception as e:
                    self.logger.warning(f"Child generation failed: {e}")
                    # Add random valid genome as fallback
                    fallback = self._generate_diverse_genome()
                    new_pop.append(fallback)
            
            # Store evolution stats for reporting
            self._last_evolution_stats = evolution_stats
            
            return np.array(new_pop[:self.population_size], dtype=np.float32)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "population_creation")
            self.logger.error(f"Population creation failed: {error_context}")
            return self.population.copy()  # Return current population as fallback

    def _select_parent_intelligent(self) -> np.ndarray:
        """Select parent using current intelligent selection strategy"""
        strategy_func = self.selection_strategies.get(self.current_selection_strategy, self._tournament_selection)
        return strategy_func()

    def _tournament_selection(self) -> np.ndarray:
        """Tournament selection with enhanced competition"""
        if len(self.population) == 0 or len(self.fitness) == 0:
            return self._generate_diverse_genome()
        
        candidates = np.random.choice(self.population_size, 
                                    min(self.tournament_k, self.population_size), 
                                    replace=False)
        winner_idx = candidates[np.argmax(self.fitness[candidates])]
        return self.population[winner_idx].copy()

    def _roulette_selection(self) -> np.ndarray:
        """Roulette wheel selection with fitness scaling"""
        if len(self.fitness) == 0:
            return self._generate_diverse_genome()
        
        # Shift fitness to positive
        shifted_fitness = self.fitness - np.min(self.fitness) + 1e-6
        total_fitness = np.sum(shifted_fitness)
        
        if total_fitness > 0:
            probabilities = shifted_fitness / total_fitness
            selected_idx = np.random.choice(self.population_size, p=probabilities)
        else:
            selected_idx = np.random.randint(self.population_size)
        
        return self.population[selected_idx].copy()

    def _rank_selection(self) -> np.ndarray:
        """Rank-based selection for balanced pressure"""
        if len(self.fitness) == 0:
            return self._generate_diverse_genome()
        
        ranks = np.argsort(np.argsort(self.fitness)) + 1  # Ranks from 1 to population_size
        probabilities = ranks / np.sum(ranks)
        selected_idx = np.random.choice(self.population_size, p=probabilities)
        return self.population[selected_idx].copy()

    def _adaptive_selection(self) -> np.ndarray:
        """Adaptive selection combining multiple strategies"""
        # Choose strategy based on current state
        if self.generations_without_improvement < 5:
            return self._tournament_selection()
        elif getattr(self, '_last_population_health', 'healthy') == 'poor':
            return self._diversity_selection()
        else:
            return self._rank_selection()

    def _diversity_selection(self) -> np.ndarray:
        """Selection that promotes genetic diversity"""
        if len(self.fitness) == 0:
            return self._generate_diverse_genome()
        
        # Select randomly from top 50% to promote diversity
        top_half_count = max(1, self.population_size // 2)
        top_half = np.argsort(self.fitness)[-top_half_count:]
        selected_idx = np.random.choice(top_half)
        return self.population[selected_idx].copy()

    def _crossover_intelligent(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        """Enhanced crossover with multiple intelligent strategies"""
        crossover_type = np.random.choice(['uniform', 'arithmetic', 'single_point', 'blend'])
        
        if crossover_type == 'uniform':
            # Uniform crossover
            mask = np.random.random(self.genome_size) < 0.5
            child = np.where(mask, parent1, parent2)
        elif crossover_type == 'arithmetic':
            # Arithmetic crossover with adaptive alpha
            alpha = np.random.uniform(0.3, 0.7)  # Favor balanced combination
            child = alpha * parent1 + (1 - alpha) * parent2
        elif crossover_type == 'blend':
            # Blend crossover for exploration
            alpha = 0.5
            beta = np.random.uniform(0.1, 0.3, self.genome_size)
            child = parent1 + beta * (parent2 - parent1)
        else:
            # Single point crossover
            point = np.random.randint(1, self.genome_size)
            child = np.concatenate([parent1[:point], parent2[point:]])
        
        return child.astype(np.float32)

    def _mutate_intelligent(self, genome: np.ndarray, mutation_scale: float) -> np.ndarray:
        """Enhanced mutation with adaptive intelligent strategies"""
        mutated = genome.copy()
        
        # Choose mutation strategy based on population state
        mutation_strategies = ['gaussian', 'uniform', 'polynomial', 'boundary']
        mutation_type = np.random.choice(mutation_strategies)
        
        for i in range(self.genome_size):
            if np.random.random() < 0.4:  # 40% chance per gene
                bounds = list(self.genome_bounds.values())[i]
                range_size = bounds[1] - bounds[0]
                
                if mutation_type == 'gaussian':
                    # Gaussian mutation with adaptive scale
                    noise = np.random.normal(0, mutation_scale * range_size * 0.1)
                    mutated[i] += noise
                elif mutation_type == 'uniform':
                    # Uniform mutation within scaled bounds
                    noise = np.random.uniform(-range_size * mutation_scale * 0.2, 
                                            range_size * mutation_scale * 0.2)
                    mutated[i] += noise
                elif mutation_type == 'polynomial':
                    # Polynomial mutation for smooth exploration
                    u = np.random.random()
                    if u <= 0.5:
                        delta = (2 * u) ** (1/3) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1/3)
                    mutated[i] += delta * range_size * mutation_scale * 0.15
                else:  # boundary
                    # Boundary mutation for constraint exploration
                    if np.random.random() < 0.5:
                        mutated[i] = bounds[0] + np.random.random() * range_size * 0.1
                    else:
                        mutated[i] = bounds[1] - np.random.random() * range_size * 0.1
                
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

    async def _update_evolution_analytics_comprehensive(self, evolution_analysis: Dict[str, Any], 
                                                      evolution_results: Dict[str, Any]):
        """Update comprehensive evolution analytics"""
        try:
            # Update fitness history
            if len(self.fitness) > 0:
                current_fitness_stats = {
                    'mean': float(np.mean(self.fitness)),
                    'max': float(np.max(self.fitness)),
                    'std': float(np.std(self.fitness)),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                self.evolution_analytics['fitness_history'].append(current_fitness_stats)
            
            # Update diversity history
            diversity_score = self._calculate_population_diversity()
            diversity_record = {
                'score': diversity_score,
                'timestamp': datetime.datetime.now().isoformat()
            }
            self.evolution_analytics['diversity_history'].append(diversity_record)
            
            # Update convergence score
            self.evolution_analytics['convergence_score'] = self._calculate_convergence_score()
            
            # Update population health tracking
            population_health = evolution_analysis.get('population_health', {})
            overall_health = population_health.get('overall', 'healthy')
            
            if overall_health != self.evolution_analytics['population_health']:
                self.logger.info(format_operator_message(
                    icon="🏥",
                    message="Population health changed",
                    from_health=self.evolution_analytics['population_health'],
                    to_health=overall_health,
                    generation=self.epoch
                ))
            
            self.evolution_analytics['population_health'] = overall_health
            self._last_population_health = overall_health
            
            # Update success rates
            evolution_stats = getattr(self, '_last_evolution_stats', {})
            total_operations = evolution_stats.get('crossovers', 0) + evolution_stats.get('mutations', 0)
            if total_operations > 0:
                # Simplified success rate calculation
                self.evolution_analytics['mutation_success_rate'] = evolution_stats.get('mutations', 0) / total_operations
                self.evolution_analytics['crossover_success_rate'] = evolution_stats.get('crossovers', 0) / total_operations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "analytics_update")
            self.logger.warning(f"Evolution analytics update failed: {error_context}")

    async def _generate_comprehensive_evolution_thesis(self, evolution_analysis: Dict[str, Any], 
                                                     evolution_results: Dict[str, Any]) -> str:
        """Generate comprehensive thesis explaining all evolution decisions"""
        try:
            thesis_parts = []
            
            # Executive Summary
            population_health = evolution_analysis.get('population_health', {})
            overall_health = population_health.get('overall', 'unknown')
            thesis_parts.append(
                f"EVOLUTION ANALYSIS: Population health {overall_health} at generation {self.epoch}"
            )
            
            # Performance Context
            evolution_context = evolution_analysis.get('evolution_context', {})
            performance_trend = evolution_context.get('performance_trend', 'stable')
            market_stress = evolution_context.get('market_stress_level', 'medium')
            thesis_parts.append(
                f"PERFORMANCE CONTEXT: {performance_trend} trend with {market_stress} market stress"
            )
            
            # Best Genome Status
            if self.best_fitness > -np.inf:
                target_progress = (self.best_fitness / self.profit_target) * 100 if self.profit_target > 0 else 0
                thesis_parts.append(
                    f"BEST GENOME: €{self.best_fitness:.2f} ({target_progress:.1f}% of target)"
                )
            
            # Evolution Actions
            strategy_adaptation = evolution_results.get('strategy_adaptation', {})
            if strategy_adaptation.get('adapted', False):
                thesis_parts.append(
                    f"STRATEGY ADAPTED: {strategy_adaptation['old_strategy']} → {strategy_adaptation['new_strategy']}"
                )
            
            population_changes = evolution_results.get('population_changes', {})
            if population_changes:
                changed_count = population_changes.get('genomes_changed', 0)
                thesis_parts.append(
                    f"POPULATION EVOLVED: {changed_count}/{self.population_size} genomes changed"
                )
            
            # Diversity and Health Metrics
            diversity_metrics = evolution_analysis.get('diversity_metrics', {})
            population_diversity = diversity_metrics.get('population_diversity', 0)
            convergence = diversity_metrics.get('convergence', {})
            thesis_parts.append(
                f"DIVERSITY STATUS: {population_diversity:.3f} diversity, {convergence.get('status', 'unknown')} convergence"
            )
            
            # Stagnation Tracking
            if self.generations_without_improvement > 0:
                thesis_parts.append(
                    f"STAGNATION: {self.generations_without_improvement} generations without improvement"
                )
            
            # System Health
            if self.error_count > 0:
                thesis_parts.append(
                    f"SYSTEM HEALTH: {self.error_count} errors recorded"
                )
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Evolution thesis generation failed: {error_context}"

    def _generate_intelligent_recommendations(self, evolution_analysis: Dict[str, Any]) -> List[str]:
        """Generate intelligent recommendations based on evolution analysis"""
        try:
            recommendations = []
            
            # Population health recommendations
            population_health = evolution_analysis.get('population_health', {})
            overall_health = population_health.get('overall', 'healthy')
            
            if overall_health == 'poor':
                recommendations.append(
                    "Population Health: Increase mutation rate and inject diversity to restore genetic variation"
                )
            elif overall_health == 'moderate':
                recommendations.append(
                    "Population Health: Monitor diversity trends and consider adaptive parameter adjustments"
                )
            
            # Performance recommendations
            evolution_context = evolution_analysis.get('evolution_context', {})
            performance_trend = evolution_context.get('performance_trend', 'stable')
            
            if performance_trend in ['strongly_declining', 'declining']:
                recommendations.append(
                    "Performance: Aggressive evolution recommended with increased exploration parameters"
                )
            elif performance_trend in ['strongly_improving', 'improving']:
                recommendations.append(
                    "Performance: Maintain current strategy with slight exploitation bias"
                )
            
            # Fitness landscape recommendations
            fitness_analysis = evolution_analysis.get('fitness_analysis', {})
            optimization_potential = fitness_analysis.get('optimization_potential', {})
            target_achievement = optimization_potential.get('target_achievement', 0)
            
            if target_achievement >= 1.0:
                recommendations.append(
                    "Target Achievement: Consider raising profit target or focusing on consistency"
                )
            elif target_achievement < 0.5:
                recommendations.append(
                    "Target Achievement: Increase evolution pressure and exploration to reach profit targets"
                )
            
            # Stagnation recommendations
            if self.generations_without_improvement > 15:
                recommendations.append(
                    "Stagnation: Critical - implement radical diversity injection and parameter reset"
                )
            elif self.generations_without_improvement > 10:
                recommendations.append(
                    "Stagnation: High - increase mutation rates and consider strategy diversification"
                )
            
            # Diversity recommendations
            diversity_metrics = evolution_analysis.get('diversity_metrics', {})
            convergence = diversity_metrics.get('convergence', {})
            if convergence.get('status') == 'converged':
                recommendations.append(
                    "Convergence: Population converged - inject diversity or restart with best genomes"
                )
            
            # Default recommendation
            if not recommendations:
                recommendations.append(
                    f"System Operating Optimally: Continue current evolution strategy with {overall_health} population"
                )
            
            return recommendations[:5]  # Limit to top 5 recommendations
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "recommendation_generation")
            return [f"Recommendation generation failed: {error_context}"]

    def _get_current_genome_weights(self) -> Dict[str, Any]:
        """Get current genome weights and selection information"""
        try:
            if self.active_genome is not None:
                return {
                    'active_genome': self.active_genome.tolist(),
                    'active_genome_idx': self.active_genome_idx,
                    'active_fitness': float(self.fitness[self.active_genome_idx]) if self.active_genome_idx < len(self.fitness) else 0.0,
                    'selection_strategy': self.current_selection_strategy,
                    'generation': self.epoch
                }
            else:
                # Return best genome as default
                if len(self.fitness) > 0:
                    best_idx = int(np.argmax(self.fitness))
                    return {
                        'active_genome': self.population[best_idx].tolist(),
                        'active_genome_idx': best_idx,
                        'active_fitness': float(self.fitness[best_idx]),
                        'selection_strategy': 'best_available',
                        'generation': self.epoch
                    }
                else:
                    return {
                        'active_genome': [1.0, 1.5, 1.0, 0.3],  # Default safe genome
                        'active_genome_idx': 0,
                        'active_fitness': 0.0,
                        'selection_strategy': 'default',
                        'generation': self.epoch
                    }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "genome_weights")
            return {'error': str(error_context)}

    def _get_population_metrics(self) -> Dict[str, Any]:
        """Get comprehensive population metrics"""
        try:
            metrics = {
                'population_size': self.population_size,
                'current_generation': self.epoch,
                'generations_without_improvement': self.generations_without_improvement,
                'best_fitness': float(self.best_fitness),
                'target_achievement': (self.best_fitness / self.profit_target) * 100 if self.profit_target > 0 else 0,
                'cache_size': len(self.genome_performance_cache),
                'evaluation_history_size': len(self.genome_evaluation_history)
            }
            
            if len(self.fitness) > 0:
                metrics['fitness_statistics'] = {
                    'min': float(np.min(self.fitness)),
                    'max': float(np.max(self.fitness)),
                    'mean': float(np.mean(self.fitness)),
                    'std': float(np.std(self.fitness)),
                    'range': float(np.max(self.fitness) - np.min(self.fitness))
                }
            
            metrics['diversity_score'] = self._calculate_population_diversity()
            metrics['convergence_score'] = self.evolution_analytics['convergence_score']
            metrics['population_health'] = self.evolution_analytics['population_health']
            
            return metrics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "population_metrics")
            return {'error': str(error_context)}

    async def _update_smartinfobus_comprehensive(self, results: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with comprehensive evolution results"""
        try:
            # Core genome weights
            self.smart_bus.set('genome_weights', results['genome_weights'],
                             module='StrategyGenomePool', thesis=thesis)
            
            # Genome analysis
            analysis_thesis = f"Genome evolution analysis: generation {self.epoch} with {results['population_metrics'].get('population_health', 'unknown')} health"
            self.smart_bus.set('genome_analysis', results['genome_analysis'],
                             module='StrategyGenomePool', thesis=analysis_thesis)
            
            # Genome recommendations
            rec_thesis = f"Generated {len(results['genome_recommendations'])} intelligent evolution recommendations"
            self.smart_bus.set('genome_recommendations', results['genome_recommendations'],
                             module='StrategyGenomePool', thesis=rec_thesis)
            
            # Evolution analytics
            analytics_thesis = f"Evolution analytics: {results['evolution_analytics']['adaptation_events']} adaptations performed"
            self.smart_bus.set('evolution_analytics', results['evolution_analytics'],
                             module='StrategyGenomePool', thesis=analytics_thesis)
            
            # Best genome
            best_thesis = f"Best genome: €{results['best_genome']['fitness']:.2f} at generation {results['best_genome']['generation']}"
            self.smart_bus.set('best_genome', results['best_genome'],
                             module='StrategyGenomePool', thesis=best_thesis)
            
            # Population metrics
            metrics_thesis = f"Population metrics: {results['population_metrics']['population_size']} genomes, {results['population_metrics']['diversity_score']:.3f} diversity"
            self.smart_bus.set('population_metrics', results['population_metrics'],
                             module='StrategyGenomePool', thesis=metrics_thesis)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "smartinfobus_update")
            self.logger.error(f"SmartInfoBus update failed: {error_context}")

    # ═══════════════════════════════════════════════════════════════════
    # ERROR HANDLING AND RECOVERY
    # ═══════════════════════════════════════════════════════════════════

    async def _handle_processing_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle processing errors with intelligent recovery"""
        self.error_count += 1
        error_context = self.error_pinpointer.analyze_error(error, "StrategyGenomePool")
        
        # Circuit breaker logic
        if self.error_count >= self.circuit_breaker_threshold:
            self.is_disabled = True
            self.logger.error(format_operator_message(
                icon="🚨",
                message="Strategy Genome Pool disabled due to repeated errors",
                error_count=self.error_count,
                threshold=self.circuit_breaker_threshold
            ))
        
        # Record error performance
        processing_time = (time.time() - start_time) * 1000
        self.performance_tracker.record_metric('StrategyGenomePool', 'process_time', processing_time, False)
        
        return {
            'genome_weights': self._get_current_genome_weights(),
            'genome_analysis': {'error': str(error_context)},
            'genome_recommendations': ["Investigate genome evolution system errors"],
            'evolution_analytics': {'error': str(error_context)},
            'best_genome': {'error': str(error_context)},
            'population_metrics': {'error': str(error_context)},
            'health_metrics': {'status': 'error', 'error_context': str(error_context)}
        }

    def _get_safe_market_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when market data retrieval fails"""
        return {
            'market_data': {},
            'recent_trades': [],
            'trading_performance': {},
            'risk_data': {},
            'market_regime': 'unknown',
            'volatility_data': {},
            'session_metrics': {},
            'market_context': {}
        }

    def _get_safe_analysis_defaults(self) -> Dict[str, Any]:
        """Get safe defaults when analysis fails"""
        return {
            'evolution_context': {'timestamp': datetime.datetime.now().isoformat()},
            'population_health': {'overall': 'unknown'},
            'fitness_analysis': {'error': 'analysis_failed'},
            'diversity_metrics': {'population_diversity': 0.5},
            'selection_pressure': {'current_strategy': self.current_selection_strategy},
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'error': 'analysis_failed'
        }

    def _generate_disabled_response(self) -> Dict[str, Any]:
        """Generate response when module is disabled"""
        return {
            'genome_weights': self._get_current_genome_weights(),
            'genome_analysis': {'status': 'disabled'},
            'genome_recommendations': ["Restart strategy genome pool system"],
            'evolution_analytics': {'status': 'disabled'},
            'best_genome': {'status': 'disabled'},
            'population_metrics': {'status': 'disabled'},
            'health_metrics': {'status': 'disabled', 'reason': 'circuit_breaker_triggered'}
        }

    # ═══════════════════════════════════════════════════════════════════
    # UTILITY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def _get_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics for monitoring"""
        return {
            'module_name': 'StrategyGenomePool',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'error_count': self.error_count,
            'circuit_breaker_threshold': self.circuit_breaker_threshold,
            'population_size': self.population_size,
            'current_generation': self.epoch,
            'generations_without_improvement': self.generations_without_improvement,
            'best_fitness': float(self.best_fitness),
            'population_health': self.evolution_analytics['population_health'],
            'diversity_score': self._calculate_population_diversity(),
            'convergence_score': self.evolution_analytics['convergence_score'],
            'adaptation_events': self.evolution_analytics['adaptation_events']
        }

    def select_genome(self, mode: str = "smart", k: int = 3, custom_selector: Optional[Callable] = None) -> np.ndarray:
        """Enhanced genome selection with comprehensive strategies"""
        try:
            # Ensure population is properly initialized
            if len(self.population) == 0 or len(self.fitness) == 0:
                self.logger.warning("Empty population detected, reinitializing")
                self.population = self._initialize_population()
                self.fitness = np.zeros(self.population_size, dtype=np.float32)
            
            N = len(self.population)
            
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
                self.logger.warning(f"Unknown selection mode: {mode}, using best")
                idx = int(np.argmax(self.fitness))
            
            # Validate selection
            if not (0 <= idx < N):
                self.logger.warning(f"Invalid genome index {idx}, using best")
                idx = int(np.argmax(self.fitness))
            
            # Get the selected genome
            selected_genome = self.population[idx].copy()
            
            # Validate selected genome
            if np.any(~np.isfinite(selected_genome)):
                self.logger.warning(f"Selected genome contains invalid values: {selected_genome}")
                selected_genome = self._repair_genome(selected_genome)
            
            # Store as active genome
            self.active_genome = selected_genome
            self.active_genome_idx = idx
            
            # Update usage statistics
            genome_hash = self.genome_hash(selected_genome)
            self.genome_usage_stats[genome_hash] += 1
            
            # Log selection
            fit_val = self.fitness[idx]
            genome_str = ", ".join(f"{x:.3f}" for x in selected_genome)
            self.logger.info(format_operator_message(
                icon="🎯",
                message="Genome selected",
                index=idx,
                fitness=f"€{fit_val:.2f}",
                genome=f"[{genome_str}]",
                mode=mode
            ))
            
            return selected_genome
            
        except Exception as e:
            self.logger.error(f"Genome selection failed: {e}")
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
                top_k = min(5, max(1, self.population_size // 2))
                top_indices = np.argsort(self.fitness)[-top_k:]
                return np.random.choice(top_indices)
            else:
                # Balanced tournament selection
                tournament_size = min(self.tournament_k, len(self.population))
                candidates = np.random.choice(self.population_size, tournament_size, replace=False)
                return candidates[np.argmax(self.fitness[candidates])]
        except Exception as e:
            self.logger.warning(f"Smart selection failed: {e}")
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
            self.logger.warning(f"Diversity selection failed: {e}")
            return 0

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation with comprehensive evolution metrics"""
        try:
            if len(self.fitness) == 0:
                return np.full(10, 0.5, dtype=np.float32)
            
            mean_fitness = float(np.mean(self.fitness))
            max_fitness = float(np.max(self.fitness))
            
            # Validate fitness values
            if not np.isfinite(mean_fitness):
                mean_fitness = 0.0
            if not np.isfinite(max_fitness):
                max_fitness = 0.0
            
            # Calculate additional metrics
            diversity_score = self._calculate_population_diversity()
            convergence_score = self.evolution_analytics['convergence_score']
            profit_ratio = max_fitness / self.profit_target if self.profit_target > 0 else 0.0
            health_score = {'healthy': 1.0, 'moderate': 0.7, 'poor': 0.3, 'unknown': 0.5}.get(
                self.evolution_analytics['population_health'], 0.5)
            
            # Stagnation indicator
            stagnation_score = min(1.0, self.generations_without_improvement / 20.0)
            
            # Evolution activity score
            evolution_activity = min(1.0, self.evolution_analytics['adaptation_events'] / 10.0)
            
            observation = np.array([
                mean_fitness / 100.0,      # Normalized mean fitness
                max_fitness / 100.0,       # Normalized max fitness  
                diversity_score / 3.0,     # Normalized population diversity
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
                self.logger.error(f"Invalid genome observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            self.logger.error(f"Genome observation generation failed: {e}")
            return np.full(10, 0.5, dtype=np.float32)

    def get_genome_report(self) -> str:
        """Generate comprehensive genome pool report"""
        try:
            # Population statistics
            fitness_stats = {
                'mean': np.mean(self.fitness) if len(self.fitness) > 0 else 0,
                'max': np.max(self.fitness) if len(self.fitness) > 0 else 0,
                'min': np.min(self.fitness) if len(self.fitness) > 0 else 0,
                'std': np.std(self.fitness) if len(self.fitness) > 0 else 0
            }
            
            # Evolution trend
            evolution_trend = "📊 Insufficient data"
            if len(self.evolution_analytics['fitness_history']) >= 3:
                recent_fitness = [h['max'] for h in list(self.evolution_analytics['fitness_history'])[-3:]]
                if recent_fitness[-1] > recent_fitness[0] + 10:
                    evolution_trend = "📈 Improving"
                elif recent_fitness[-1] < recent_fitness[0] - 10:
                    evolution_trend = "📉 Declining"
                else:
                    evolution_trend = "➡️ Stable"
            
            # Top genomes
            top_genomes = ""
            if len(self.fitness) > 0:
                top_indices = np.argsort(self.fitness)[-3:][::-1]
                for rank, idx in enumerate(top_indices, 1):
                    genome_str = ", ".join(f"{x:.3f}" for x in self.population[idx])
                    top_genomes += f"  #{rank}: €{self.fitness[idx]:.2f} [{genome_str}]\n"
            
            return f"""
🧬 STRATEGY GENOME POOL COMPREHENSIVE REPORT
═══════════════════════════════════════════════════════════════
📊 Population Overview:
• Generation: {self.epoch}
• Population Size: {self.population_size}
• Genome Size: {self.genome_size}
• Selection Strategy: {self.current_selection_strategy.title()}
• Population Health: {self.evolution_analytics['population_health'].title()}

🏆 Performance Metrics:
• Best Fitness: €{self.best_fitness:.2f}
• Target Progress: {(self.best_fitness/self.profit_target)*100:.1f}%
• Population Mean: €{fitness_stats['mean']:.2f}
• Fitness Range: €{fitness_stats['min']:.2f} - €{fitness_stats['max']:.2f}
• Stagnant Generations: {self.generations_without_improvement}

🧬 Evolution Analytics:
• Diversity Score: {self._calculate_population_diversity():.3f}
• Convergence Score: {self.evolution_analytics['convergence_score']:.3f}
• Adaptation Events: {self.evolution_analytics['adaptation_events']}
• Evolution Trend: {evolution_trend}
• Mutation Success Rate: {self.evolution_analytics['mutation_success_rate']:.1%}
• Crossover Success Rate: {self.evolution_analytics['crossover_success_rate']:.1%}

🎯 Top Performing Genomes:
{top_genomes if top_genomes else '  📭 No fitness data available yet'}

⚙️ Technical Details:
• Cache Size: {len(self.genome_performance_cache)} evaluations
• Genome Usage Stats: {len(self.genome_usage_stats)} unique genomes used
• Evaluation History: {len(self.genome_evaluation_history)} records
• Error Count: {self.error_count}/{self.circuit_breaker_threshold}

🎚️ Current Parameters:
• Mutation Rate: {self.mutation_rate:.1%}
• Crossover Rate: {self.crossover_rate:.1%}
• Tournament Size: {self.tournament_k}
• Profit Target: €{self.profit_target}

🔧 System Status:
• Module Status: {'DISABLED' if self.is_disabled else 'OPERATIONAL'}
• Circuit Breaker: {'OPEN' if self.error_count >= self.circuit_breaker_threshold else 'CLOSED'}
• Intelligence Level: Advanced Adaptive Evolution
            """
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "genome_report")
            return f"Genome report generation failed: {error_context}"

    # ═══════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT FOR HOT-RELOAD
    # ═══════════════════════════════════════════════════════════════════

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for hot-reload and persistence"""
        return {
            'module_info': {
                'name': 'StrategyGenomePool',
                'version': '3.0.0',
                'last_updated': datetime.datetime.now().isoformat()
            },
            'configuration': {
                'population_size': self.population_size,
                'tournament_k': self.tournament_k,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'mutation_scale': self.mutation_scale,
                'max_generations_kept': self.max_generations_kept,
                'profit_target': self.profit_target,
                'genome_size': self.genome_size,
                'debug': self.debug
            },
            'evolution_state': {
                'population': self.population.tolist(),
                'fitness': self.fitness.tolist(),
                'epoch': self.epoch,
                'best_genome': self.best_genome.tolist(),
                'best_fitness': float(self.best_fitness),
                'generations_without_improvement': self.generations_without_improvement,
                'active_genome': self.active_genome.tolist() if self.active_genome is not None else None,
                'active_genome_idx': self.active_genome_idx,
                'current_selection_strategy': self.current_selection_strategy
            },
            'analytics_state': {
                'evolution_analytics': self.evolution_analytics.copy(),
                'genome_usage_stats': dict(self.genome_usage_stats),
                'genome_performance_cache': dict(self.genome_performance_cache),
                'evaluation_history': list(self.genome_evaluation_history)[-50:],  # Keep recent only
                'evolution_intelligence': self.evolution_intelligence.copy()
            },
            'error_state': {
                'error_count': self.error_count,
                'is_disabled': self.is_disabled
            },
            'genome_bounds': self.genome_bounds.copy(),
            'performance_metrics': self._get_health_metrics()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state for hot-reload and persistence"""
        try:
            # Load configuration
            config = state.get("configuration", {})
            self.population_size = int(config.get("population_size", self.population_size))
            self.tournament_k = int(config.get("tournament_k", self.tournament_k))
            self.crossover_rate = float(config.get("crossover_rate", self.crossover_rate))
            self.mutation_rate = float(config.get("mutation_rate", self.mutation_rate))
            self.mutation_scale = float(config.get("mutation_scale", self.mutation_scale))
            self.max_generations_kept = int(config.get("max_generations_kept", self.max_generations_kept))
            self.profit_target = float(config.get("profit_target", self.profit_target))
            self.genome_size = int(config.get("genome_size", self.genome_size))
            self.debug = bool(config.get("debug", self.debug))
            
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
            
            # Load analytics state
            analytics_state = state.get("analytics_state", {})
            self.evolution_analytics.update(analytics_state.get("evolution_analytics", {}))
            self.genome_usage_stats = defaultdict(int, analytics_state.get("genome_usage_stats", {}))
            self.genome_performance_cache = analytics_state.get("genome_performance_cache", {})
            
            evaluation_history = analytics_state.get("evaluation_history", [])
            self.genome_evaluation_history = deque(evaluation_history, maxlen=self.population_size * 5)
            
            self.evolution_intelligence.update(analytics_state.get("evolution_intelligence", {}))
            
            # Load error state
            error_state = state.get("error_state", {})
            self.error_count = error_state.get("error_count", 0)
            self.is_disabled = error_state.get("is_disabled", False)
            
            # Load genome bounds if provided
            self.genome_bounds.update(state.get("genome_bounds", {}))
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Strategy Genome Pool state restored",
                generation=self.epoch,
                best_fitness=f"€{self.best_fitness:.2f}",
                population_size=len(self.population),
                health=self.evolution_analytics.get('population_health', 'unknown')
            ))
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "state_restoration")
            self.logger.error(f"State restoration failed: {error_context}")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status for system monitoring"""
        return {
            'module_name': 'StrategyGenomePool',
            'status': 'disabled' if self.is_disabled else 'healthy',
            'metrics': self._get_health_metrics(),
            'alerts': self._generate_health_alerts(),
            'recommendations': self._generate_health_recommendations()
        }

    def _generate_health_alerts(self) -> List[Dict[str, Any]]:
        """Generate health-related alerts"""
        alerts = []
        
        if self.is_disabled:
            alerts.append({
                'severity': 'critical',
                'message': 'StrategyGenomePool disabled due to errors',
                'action': 'Investigate error logs and restart module'
            })
        
        if self.error_count > 2:
            alerts.append({
                'severity': 'warning',
                'message': f'High error count: {self.error_count}',
                'action': 'Monitor for recurring evolution issues'
            })
        
        if self.generations_without_improvement > 20:
            alerts.append({
                'severity': 'warning',
                'message': f'Population stagnant for {self.generations_without_improvement} generations',
                'action': 'Consider population reset or parameter adjustment'
            })
        
        if self.evolution_analytics['population_health'] == 'poor':
            alerts.append({
                'severity': 'warning',
                'message': 'Poor population health detected',
                'action': 'Increase diversity and mutation parameters'
            })
        
        if self.best_fitness >= self.profit_target:
            alerts.append({
                'severity': 'info',
                'message': f'Profit target achieved: €{self.best_fitness:.2f}',
                'action': 'Consider raising target or focusing on consistency'
            })
        
        return alerts

    def _generate_health_recommendations(self) -> List[str]:
        """Generate health-related recommendations"""
        recommendations = []
        
        if self.is_disabled:
            recommendations.append("Restart StrategyGenomePool module after investigating errors")
        
        if len(self.genome_evaluation_history) < 10:
            recommendations.append("Insufficient evolution history - continue operations to build performance baseline")
        
        if self.generations_without_improvement > 15:
            recommendations.append("High stagnation detected - consider aggressive evolution parameters")
        
        if self.evolution_analytics['population_health'] == 'poor':
            recommendations.append("Poor population health - inject diversity and increase mutation rates")
        
        convergence_score = self.evolution_analytics.get('convergence_score', 0.5)
        if convergence_score < 0.1:
            recommendations.append("Population converged - restart evolution with best genomes as seeds")
        
        if not recommendations:
            recommendations.append("StrategyGenomePool operating within normal parameters")
        
        return recommendations

    # ═══════════════════════════════════════════════════════════════════
    # PUBLIC API METHODS (for external use)
    # ═══════════════════════════════════════════════════════════════════

    def record_genome_result(self, genome_idx: int, pnl: float, confidence: float = 1.0) -> None:
        """Public method to record genome result (async wrapper)"""
        try:
            # Validate inputs
            if not isinstance(genome_idx, int) or not (0 <= genome_idx < len(self.population)):
                self.logger.warning(f"Invalid genome index: {genome_idx}")
                return
            
            if np.isnan(pnl):
                self.logger.warning(f"NaN PnL for genome {genome_idx}, ignoring")
                return
            
            # Update fitness directly for synchronous call
            old_fitness = self.fitness[genome_idx]
            
            # Apply profit target bonus
            if pnl >= self.profit_target:
                profit_bonus = 1.0 + (pnl - self.profit_target) / self.profit_target
                final_fitness = pnl * profit_bonus
                
                self.logger.info(format_operator_message(
                    icon="💰",
                    message=f"Genome {genome_idx} achieved target",
                    raw_pnl=f"€{pnl:.2f}",
                    bonus_multiplier=f"{profit_bonus:.2f}x",
                    final_fitness=f"€{final_fitness:.2f}"
                ))
            else:
                final_fitness = pnl
            
            self.fitness[genome_idx] = final_fitness
            
            # Update best genome tracking
            if final_fitness > self.best_fitness:
                improvement = final_fitness - self.best_fitness
                self.best_fitness = final_fitness
                self.best_genome = self.population[genome_idx].copy()
                self.generations_without_improvement = 0
                
                self.logger.info(format_operator_message(
                    icon="🏆",
                    message="New best genome found",
                    improvement=f"€{improvement:.2f}",
                    new_fitness=f"€{final_fitness:.2f}",
                    genome_idx=genome_idx
                ))
            
            # Record in evaluation history
            evaluation_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'genome_idx': genome_idx,
                'pnl': pnl,
                'final_fitness': final_fitness,
                'improvement': final_fitness - old_fitness,
                'confidence': confidence
            }
            self.genome_evaluation_history.append(evaluation_record)
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "result_recording_wrapper")
            self.logger.error(f"Genome result recording wrapper failed: {error_context}")

    def evolve_population(self) -> bool:
        """Public method to trigger population evolution"""
        try:
            # Check if evolution is appropriate
            if self.is_disabled:
                self.logger.warning("Cannot evolve population - module is disabled")
                return False
            
            if len(self.population) == 0:
                self.logger.warning("Cannot evolve empty population")
                return False
            
            # Trigger evolution using simplified analysis
            simplified_analysis = {
                'evolution_context': {
                    'evolution_pressure': 0.7,  # Default pressure
                    'performance_trend': 'stable',
                    'market_stress_level': 'medium'
                },
                'population_health': {
                    'overall': self.evolution_analytics.get('population_health', 'healthy')
                }
            }
            
            # Run evolution synchronously
            import asyncio
            if asyncio.get_event_loop().is_running():
                # If we're already in an async context, schedule it
                task = asyncio.create_task(self._perform_population_evolution(simplified_analysis))
                # Don't wait for completion in async context
                return True
            else:
                # Run it directly
                evolution_results = asyncio.run(self._perform_population_evolution(simplified_analysis))
                return 'error' not in evolution_results
                
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "evolution_trigger")
            self.logger.error(f"Population evolution trigger failed: {error_context}")
            return False

    def get_best_genome_parameters(self) -> Dict[str, Any]:
        """Get best genome parameters with detailed breakdown"""
        try:
            if self.best_genome is None or len(self.best_genome) == 0:
                return {
                    'parameters': [1.0, 1.5, 1.0, 0.3],  # Default safe parameters
                    'fitness': 0.0,
                    'generation': 0,
                    'parameter_breakdown': {
                        'sl_base': 1.0,
                        'tp_base': 1.5,
                        'vol_scale': 1.0,
                        'regime_adapt': 0.3
                    },
                    'performance_metrics': {
                        'target_achievement': 0.0,
                        'rank_in_population': 1,
                        'usage_count': 0
                    }
                }
            
            # Parameter breakdown
            param_names = ['sl_base', 'tp_base', 'vol_scale', 'regime_adapt']
            parameter_breakdown = {
                name: float(self.best_genome[i]) 
                for i, name in enumerate(param_names) 
                if i < len(self.best_genome)
            }
            
            # Performance metrics
            target_achievement = (self.best_fitness / self.profit_target) * 100 if self.profit_target > 0 else 0
            best_genome_hash = self.genome_hash(self.best_genome)
            usage_count = self.genome_usage_stats.get(best_genome_hash, 0)
            
            return {
                'parameters': self.best_genome.tolist(),
                'fitness': float(self.best_fitness),
                'generation': self.epoch,
                'parameter_breakdown': parameter_breakdown,
                'performance_metrics': {
                    'target_achievement': target_achievement,
                    'rank_in_population': 1,  # Best genome is always rank 1
                    'usage_count': usage_count,
                    'generations_since_improvement': self.generations_without_improvement
                },
                'metadata': {
                    'genome_hash': best_genome_hash,
                    'population_health': self.evolution_analytics['population_health'],
                    'selection_strategy': self.current_selection_strategy
                }
            }
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "best_genome_parameters")
            self.logger.error(f"Best genome parameters retrieval failed: {error_context}")
            return {'error': str(error_context)}

    def get_evolution_summary(self) -> Dict[str, Any]:
        """Get comprehensive evolution summary"""
        try:
            summary = {
                'evolution_status': {
                    'current_generation': self.epoch,
                    'population_size': self.population_size,
                    'generations_without_improvement': self.generations_without_improvement,
                    'is_stagnant': self.generations_without_improvement > 10,
                    'population_health': self.evolution_analytics['population_health']
                },
                'performance_summary': {
                    'best_fitness': float(self.best_fitness),
                    'target_achievement': (self.best_fitness / self.profit_target) * 100 if self.profit_target > 0 else 0,
                    'profit_target': self.profit_target,
                    'evaluations_cached': len(self.genome_performance_cache)
                },
                'diversity_metrics': {
                    'population_diversity': self._calculate_population_diversity(),
                    'convergence_score': self.evolution_analytics['convergence_score'],
                    'unique_genomes_used': len(self.genome_usage_stats)
                },
                'evolution_analytics': {
                    'adaptation_events': self.evolution_analytics['adaptation_events'],
                    'mutation_success_rate': self.evolution_analytics['mutation_success_rate'],
                    'crossover_success_rate': self.evolution_analytics['crossover_success_rate'],
                    'current_selection_strategy': self.current_selection_strategy
                },
                'system_health': {
                    'module_status': 'disabled' if self.is_disabled else 'operational',
                    'error_count': self.error_count,
                    'circuit_breaker_status': 'open' if self.error_count >= self.circuit_breaker_threshold else 'closed'
                }
            }
            
            # Add fitness statistics if population has fitness data
            if len(self.fitness) > 0:
                summary['fitness_statistics'] = {
                    'min': float(np.min(self.fitness)),
                    'max': float(np.max(self.fitness)),
                    'mean': float(np.mean(self.fitness)),
                    'std': float(np.std(self.fitness)),
                    'range': float(np.max(self.fitness) - np.min(self.fitness))
                }
            
            return summary
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "evolution_summary")
            return {'error': str(error_context)}

    def reset_evolution(self) -> bool:
        """Reset evolution system to initial state"""
        try:
            # Reset population
            self.population = self._initialize_population()
            self.fitness = np.zeros(self.population_size, dtype=np.float32)
            
            # Reset evolution state
            self.epoch = 0
            self.best_genome = self.population[0].copy()
            self.best_fitness = -np.inf
            self.generations_without_improvement = 0
            self.active_genome = None
            self.active_genome_idx = 0
            
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
            
            # Reset tracking
            self.genome_evaluation_history.clear()
            self.genome_performance_cache.clear()
            self.genome_usage_stats.clear()
            
            # Reset error state
            self.error_count = 0
            self.is_disabled = False
            
            # Reset selection strategy
            self.current_selection_strategy = 'adaptive'
            
            self.logger.info(format_operator_message(
                icon="🔄",
                message="Strategy Genome Pool reset completed",
                population_size=self.population_size,
                generation=self.epoch
            ))
            
            return True
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "evolution_reset")
            self.logger.error(f"Evolution reset failed: {error_context}")
            return False

    # ═══════════════════════════════════════════════════════════════════
    # ADVANCED ANALYTICS AND REPORTING
    # ═══════════════════════════════════════════════════════════════════

    def get_detailed_analytics(self) -> Dict[str, Any]:
        """Get detailed evolution analytics for analysis"""
        try:
            analytics = {
                'timestamp': datetime.datetime.now().isoformat(),
                'evolution_overview': {
                    'current_generation': self.epoch,
                    'total_evaluations': len(self.genome_evaluation_history),
                    'unique_genomes': len(self.genome_usage_stats),
                    'cache_hit_rate': len(self.genome_performance_cache) / max(len(self.genome_evaluation_history), 1)
                },
                'performance_trends': [],
                'diversity_trends': [],
                'parameter_analysis': {},
                'selection_analysis': {
                    'current_strategy': self.current_selection_strategy,
                    'adaptation_events': self.evolution_analytics['adaptation_events'],
                    'strategy_effectiveness': {}
                }
            }
            
            # Performance trends
            fitness_history = list(self.evolution_analytics['fitness_history'])
            if fitness_history:
                analytics['performance_trends'] = [
                    {
                        'generation': i,
                        'max_fitness': h['max'],
                        'mean_fitness': h['mean'],
                        'std_fitness': h['std'],
                        'timestamp': h['timestamp']
                    }
                    for i, h in enumerate(fitness_history)
                ]
            
            # Diversity trends
            diversity_history = list(self.evolution_analytics['diversity_history'])
            if diversity_history:
                analytics['diversity_trends'] = [
                    {
                        'generation': i,
                        'diversity_score': h['score'],
                        'timestamp': h['timestamp']
                    }
                    for i, h in enumerate(diversity_history)
                ]
            
            # Parameter analysis
            if len(self.population) > 0:
                param_names = ['sl_base', 'tp_base', 'vol_scale', 'regime_adapt']
                for i, param_name in enumerate(param_names):
                    param_values = self.population[:, i]
                    analytics['parameter_analysis'][param_name] = {
                        'min': float(np.min(param_values)),
                        'max': float(np.max(param_values)),
                        'mean': float(np.mean(param_values)),
                        'std': float(np.std(param_values)),
                        'best_value': float(self.best_genome[i]) if i < len(self.best_genome) else 0.0
                    }
            
            return analytics
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "detailed_analytics")
            return {'error': str(error_context)}

    def export_evolution_data(self) -> Dict[str, Any]:
        """Export comprehensive evolution data for external analysis"""
        try:
            export_data = {
                'metadata': {
                    'export_timestamp': datetime.datetime.now().isoformat(),
                    'module_version': '3.0.0',
                    'current_generation': self.epoch,
                    'population_size': self.population_size
                },
                'configuration': {
                    'genome_bounds': self.genome_bounds,
                    'evolution_parameters': {
                        'mutation_rate': self.mutation_rate,
                        'crossover_rate': self.crossover_rate,
                        'tournament_k': self.tournament_k,
                        'profit_target': self.profit_target
                    },
                    'intelligence_settings': self.evolution_intelligence
                },
                'current_state': {
                    'population': self.population.tolist(),
                    'fitness': self.fitness.tolist(),
                    'best_genome': self.best_genome.tolist(),
                    'best_fitness': float(self.best_fitness),
                    'active_selection_strategy': self.current_selection_strategy
                },
                'historical_data': {
                    'evaluation_history': list(self.genome_evaluation_history),
                    'fitness_history': list(self.evolution_analytics['fitness_history']),
                    'diversity_history': list(self.evolution_analytics['diversity_history'])
                },
                'performance_cache': dict(self.genome_performance_cache),
                'usage_statistics': dict(self.genome_usage_stats),
                'system_metrics': {
                    'error_count': self.error_count,
                    'adaptation_events': self.evolution_analytics['adaptation_events'],
                    'generations_without_improvement': self.generations_without_improvement,
                    'population_health': self.evolution_analytics['population_health']
                }
            }
            
            return export_data
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "evolution_export")
            return {'error': str(error_context)}

    def __str__(self) -> str:
        """String representation of the genome pool"""
        return f"StrategyGenomePool(gen={self.epoch}, pop={self.population_size}, best=€{self.best_fitness:.2f})"

    def __repr__(self) -> str:
        """Detailed representation of the genome pool"""
        return (f"StrategyGenomePool(generation={self.epoch}, population_size={self.population_size}, "
                f"best_fitness={self.best_fitness:.2f}, health='{self.evolution_analytics['population_health']}', "
                f"stagnant_generations={self.generations_without_improvement})")