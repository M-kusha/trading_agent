# ─────────────────────────────────────────────────────────────
# File: modules/strategy/thesis_evolution_engine.py
# Enhanced with InfoBus integration & intelligent thesis evolution
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
import random
from typing import Dict, Any, List, Optional, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import AnalysisMixin, StateManagementMixin, TradingMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit


class ThesisEvolutionEngine(Module, AnalysisMixin, StateManagementMixin, TradingMixin):
    """
    Enhanced thesis evolution engine with InfoBus integration.
    Develops and evolves trading theses based on market behavior and performance feedback.
    Provides intelligent thesis management with adaptive learning and performance tracking.
    """

    def __init__(
        self,
        capacity: int = 20,
        debug: bool = False,
        thesis_lifespan: int = 100,  # Maximum trades per thesis
        performance_threshold: float = 0.6,  # Minimum performance to keep thesis
        evolution_rate: float = 0.1,  # Rate of thesis evolution
        diversity_target: float = 0.7,  # Target diversity score
        **kwargs
    ):
        # Initialize with enhanced config
        enhanced_config = ModuleConfig(
            debug=debug,
            max_history=capacity * 2,
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(enhanced_config)
        
        # Initialize mixins
        self._initialize_analysis_state()
        self._initialize_trading_state()
        
        # Core parameters
        self.capacity = int(capacity)
        self.debug = bool(debug)
        self.thesis_lifespan = int(thesis_lifespan)
        self.performance_threshold = float(performance_threshold)
        self.evolution_rate = float(evolution_rate)
        self.diversity_target = float(diversity_target)
        
        # Thesis management state
        self.theses = []  # Current active theses
        self.thesis_performance = defaultdict(lambda: {
            'pnls': [],
            'trade_count': 0,
            'win_count': 0,
            'total_pnl': 0.0,
            'creation_time': datetime.datetime.now().isoformat(),
            'last_update': datetime.datetime.now().isoformat(),
            'category': 'general',
            'confidence_score': 0.5,
            'market_conditions': [],
            'adaptation_history': []
        })
        
        # Enhanced thesis categorization
        self.thesis_categories = {
            'trend_following': {
                'description': 'Following market momentum and trends',
                'keywords': ['trend', 'momentum', 'breakout', 'direction', 'follow'],
                'performance_weight': 1.2,
                'market_conditions': ['trending', 'breakout']
            },
            'mean_reversion': {
                'description': 'Trading against temporary price movements',
                'keywords': ['reversion', 'oversold', 'overbought', 'bounce', 'correction'],
                'performance_weight': 1.0,
                'market_conditions': ['ranging', 'reversal']
            },
            'volatility_based': {
                'description': 'Capitalizing on volatility patterns',
                'keywords': ['volatility', 'expansion', 'contraction', 'spike', 'calm'],
                'performance_weight': 1.3,
                'market_conditions': ['volatile', 'ranging']
            },
            'news_driven': {
                'description': 'Trading based on fundamental events',
                'keywords': ['news', 'event', 'release', 'announcement', 'report'],
                'performance_weight': 1.1,
                'market_conditions': ['volatile', 'trending']
            },
            'pattern_recognition': {
                'description': 'Trading based on chart patterns',
                'keywords': ['pattern', 'formation', 'setup', 'structure', 'support', 'resistance'],
                'performance_weight': 1.0,
                'market_conditions': ['ranging', 'trending']
            },
            'arbitrage': {
                'description': 'Exploiting price discrepancies',
                'keywords': ['spread', 'arbitrage', 'discrepancy', 'correlation', 'divergence'],
                'performance_weight': 0.9,
                'market_conditions': ['ranging', 'stable']
            }
        }
        
        # Evolution tracking
        self.evolution_history = deque(maxlen=100)
        self.thesis_genealogy = defaultdict(list)  # Track thesis evolution lineage
        self.successful_mutations = []
        self.failed_experiments = []
        
        # Performance analytics
        self.evolution_analytics = {
            'total_theses_created': 0,
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'average_thesis_lifespan': 0.0,
            'best_performing_category': 'general',
            'diversity_score': 0.0,
            'innovation_rate': 0.0,
            'adaptation_success_rate': 0.0
        }
        
        # Market adaptation state
        self.market_adaptation = {
            'current_regime': 'unknown',
            'regime_performance': defaultdict(list),
            'adaptation_triggers': [],
            'last_major_adaptation': None,
            'pending_adaptations': []
        }
        
        # Thesis generation templates
        self.thesis_templates = {
            'market_structure': [
                "Market shows {pattern} structure in {timeframe} suggesting {direction} move",
                "Price action indicates {support_resistance} at {level} creating {opportunity}",
                "Volume patterns suggest {accumulation_distribution} phase developing"
            ],
            'momentum': [
                "Strong {direction} momentum building across {instruments}",
                "Momentum divergence signals potential {reversal_continuation}",
                "Acceleration pattern indicates {entry_exit} opportunity"
            ],
            'volatility': [
                "Volatility {expansion_contraction} creates {trading_opportunity}",
                "Low volatility environment suggests {breakout_breakdown} potential",
                "High volatility offers {scalping_swing} opportunities"
            ],
            'correlation': [
                "Cross-asset correlation changes indicate {portfolio_adjustment}",
                "Currency strength shifts suggest {pair_selection} strategy",
                "Risk-on/risk-off rotation creates {sector_opportunity}"
            ]
        }
        
        # Setup enhanced logging with rotation
        self.logger = RotatingLogger(
            "ThesisEvolutionEngine",
            "logs/strategy/thesis_evolution.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit system
        self.audit_tracker = AuditTracker("ThesisEvolutionEngine")
        
        self.log_operator_info(
            "🧬 Thesis Evolution Engine initialized",
            capacity=self.capacity,
            thesis_lifespan=self.thesis_lifespan,
            performance_threshold=f"{self.performance_threshold:.1%}",
            categories=len(self.thesis_categories)
        )
        
        # Initialize with seed theses
        self._initialize_seed_theses()

    def _initialize_seed_theses(self) -> None:
        """Initialize with diverse seed theses"""
        
        seed_theses = [
            "Strong USD momentum continues across major pairs",
            "Gold shows mean reversion opportunity at key support",
            "EUR/USD range-bound trading offers scalping opportunities",
            "Volatility expansion signals breakout potential",
            "Risk-off sentiment creates safe-haven flows",
            "Central bank divergence drives currency strength",
            "Technical patterns suggest trend continuation",
            "Market structure indicates accumulation phase"
        ]
        
        for thesis in seed_theses[:min(len(seed_theses), self.capacity // 2)]:
            self._add_thesis(thesis, source='seed')
        
        self.log_operator_info(f"🌱 Initialized with {len(self.theses)} seed theses")

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Clear thesis state
        self.theses.clear()
        self.thesis_performance.clear()
        self.evolution_history.clear()
        self.thesis_genealogy.clear()
        self.successful_mutations.clear()
        self.failed_experiments.clear()
        
        # Reset analytics
        self.evolution_analytics = {
            'total_theses_created': 0,
            'successful_evolutions': 0,
            'failed_evolutions': 0,
            'average_thesis_lifespan': 0.0,
            'best_performing_category': 'general',
            'diversity_score': 0.0,
            'innovation_rate': 0.0,
            'adaptation_success_rate': 0.0
        }
        
        # Reset market adaptation
        self.market_adaptation = {
            'current_regime': 'unknown',
            'regime_performance': defaultdict(list),
            'adaptation_triggers': [],
            'last_major_adaptation': None,
            'pending_adaptations': []
        }
        
        # Reinitialize with seed theses
        self._initialize_seed_theses()
        
        self.log_operator_info("🔄 Thesis Evolution Engine reset - reinitialized with seeds")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration and adaptive evolution"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - limited thesis evolution")
            return
        
        # Extract context and market data
        context = extract_standard_context(info_bus)
        evolution_context = self._extract_evolution_context_from_info_bus(info_bus, context)
        
        # Update market adaptation state
        self._update_market_adaptation(evolution_context, context)
        
        # Perform thesis evolution if needed
        if self._should_evolve_theses(evolution_context):
            self._evolve_theses(evolution_context, context)
        
        # Clean up underperforming theses
        self._cleanup_underperforming_theses()
        
        # Generate new theses if needed
        self._generate_new_theses_if_needed(evolution_context, context)
        
        # Update InfoBus with thesis data
        self._update_info_bus_with_thesis_data(info_bus)

    def _extract_evolution_context_from_info_bus(self, info_bus: InfoBus, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract evolution context from InfoBus"""
        
        try:
            # Get trading activity
            recent_trades = info_bus.get('recent_trades', [])
            
            # Get strategy information
            module_data = info_bus.get('module_data', {})
            strategy_data = module_data.get('strategy_arbiter', {})
            
            # Get risk and performance data
            risk_data = info_bus.get('risk', {})
            
            evolution_context = {
                'timestamp': datetime.datetime.now().isoformat(),
                'recent_trades': recent_trades,
                'session_pnl': context.get('session_pnl', 0),
                'market_regime': context.get('regime', 'unknown'),
                'volatility_level': context.get('volatility_level', 'medium'),
                'strategy_performance': strategy_data.get('performance_metrics', {}),
                'current_drawdown': risk_data.get('current_drawdown', 0),
                'balance': risk_data.get('balance', 0),
                'market_stress_level': self._assess_market_stress_level(context, risk_data),
                'innovation_pressure': self._calculate_innovation_pressure(),
                'diversity_gap': self._calculate_diversity_gap()
            }
            
            return evolution_context
            
        except Exception as e:
            self.log_operator_warning(f"Evolution context extraction failed: {e}")
            return {'timestamp': datetime.datetime.now().isoformat()}

    def _assess_market_stress_level(self, context: Dict[str, Any], risk_data: Dict[str, Any]) -> str:
        """Assess current market stress level for adaptation"""
        
        stress_factors = 0
        
        # High volatility increases stress
        if context.get('volatility_level') in ['high', 'extreme']:
            stress_factors += 2
        
        # High drawdown increases stress
        if risk_data.get('current_drawdown', 0) > 0.05:
            stress_factors += 2
        
        # Unknown regime increases stress
        if context.get('regime') == 'unknown':
            stress_factors += 1
        
        # Classify stress
        if stress_factors <= 1:
            return 'low'
        elif stress_factors <= 3:
            return 'medium'
        else:
            return 'high'

    def _calculate_innovation_pressure(self) -> float:
        """Calculate pressure to innovate new theses"""
        
        try:
            if not self.theses or not self.thesis_performance:
                return 1.0  # High pressure when no theses
            
            # Calculate average performance
            total_pnl = 0
            total_trades = 0
            
            for thesis in self.theses:
                perf = self.thesis_performance.get(thesis, {})
                total_pnl += perf.get('total_pnl', 0)
                total_trades += perf.get('trade_count', 0)
            
            if total_trades == 0:
                return 1.0
            
            avg_pnl_per_trade = total_pnl / total_trades
            
            # High pressure if performance is poor
            if avg_pnl_per_trade < -10:
                return 1.0
            elif avg_pnl_per_trade < 0:
                return 0.8
            elif avg_pnl_per_trade < 10:
                return 0.5
            else:
                return 0.2
                
        except Exception:
            return 0.5

    def _calculate_diversity_gap(self) -> float:
        """Calculate gap between current and target diversity"""
        
        try:
            current_diversity = self._calculate_thesis_diversity()
            return max(0.0, self.diversity_target - current_diversity)
        except Exception:
            return 0.5

    def _calculate_thesis_diversity(self) -> float:
        """Calculate current thesis diversity score"""
        
        try:
            if len(self.theses) < 2:
                return 0.0
            
            # Category diversity
            categories = [self._categorize_thesis(thesis) for thesis in self.theses]
            unique_categories = len(set(categories))
            category_diversity = unique_categories / len(self.thesis_categories)
            
            # Performance diversity
            performances = []
            for thesis in self.theses:
                perf = self.thesis_performance.get(thesis, {})
                if perf.get('trade_count', 0) > 0:
                    avg_pnl = perf.get('total_pnl', 0) / perf['trade_count']
                    performances.append(avg_pnl)
            
            if performances and len(performances) > 1:
                performance_diversity = np.std(performances) / (abs(np.mean(performances)) + 1e-6)
                performance_diversity = min(1.0, performance_diversity / 10.0)  # Normalize
            else:
                performance_diversity = 0.0
            
            # Combined diversity score
            return 0.6 * category_diversity + 0.4 * performance_diversity
            
        except Exception:
            return 0.0

    def _update_market_adaptation(self, evolution_context: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Update market adaptation state"""
        
        try:
            current_regime = evolution_context.get('market_regime', 'unknown')
            previous_regime = self.market_adaptation.get('current_regime', 'unknown')
            
            # Detect regime changes
            if current_regime != previous_regime and previous_regime != 'unknown':
                self.market_adaptation['adaptation_triggers'].append({
                    'timestamp': evolution_context.get('timestamp'),
                    'type': 'regime_change',
                    'from_regime': previous_regime,
                    'to_regime': current_regime,
                    'thesis_count': len(self.theses)
                })
                
                self.log_operator_info(
                    f"🌊 Market regime change detected",
                    from_regime=previous_regime,
                    to_regime=current_regime,
                    active_theses=len(self.theses)
                )
            
            self.market_adaptation['current_regime'] = current_regime
            
            # Track regime performance
            session_pnl = evolution_context.get('session_pnl', 0)
            self.market_adaptation['regime_performance'][current_regime].append({
                'timestamp': evolution_context.get('timestamp'),
                'pnl': session_pnl,
                'thesis_count': len(self.theses)
            })
            
            # Limit regime performance history
            for regime in self.market_adaptation['regime_performance']:
                if len(self.market_adaptation['regime_performance'][regime]) > 50:
                    self.market_adaptation['regime_performance'][regime] = \
                        self.market_adaptation['regime_performance'][regime][-50:]
            
        except Exception as e:
            self.log_operator_warning(f"Market adaptation update failed: {e}")

    def _should_evolve_theses(self, evolution_context: Dict[str, Any]) -> bool:
        """Determine if thesis evolution should occur"""
        
        try:
            # Evolution triggers
            innovation_pressure = evolution_context.get('innovation_pressure', 0.5)
            diversity_gap = evolution_context.get('diversity_gap', 0.0)
            market_stress = evolution_context.get('market_stress_level', 'medium')
            
            # High innovation pressure triggers evolution
            if innovation_pressure > 0.8:
                return True
            
            # Large diversity gap triggers evolution
            if diversity_gap > 0.3:
                return True
            
            # High market stress with poor performance triggers evolution
            if market_stress == 'high' and evolution_context.get('current_drawdown', 0) > 0.03:
                return True
            
            # Periodic evolution check
            if len(self.evolution_history) == 0 or len(self.evolution_history) % 20 == 0:
                return True
            
            return False
            
        except Exception:
            return False

    def _evolve_theses(self, evolution_context: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Evolve existing theses based on performance and market conditions"""
        
        try:
            evolution_strategies = []
            
            # Determine evolution strategies
            if evolution_context.get('innovation_pressure', 0) > 0.7:
                evolution_strategies.append('mutation')
            
            if evolution_context.get('diversity_gap', 0) > 0.2:
                evolution_strategies.append('crossover')
            
            if evolution_context.get('market_stress_level') == 'high':
                evolution_strategies.append('adaptation')
            
            if not evolution_strategies:
                evolution_strategies = ['refinement']
            
            evolved_count = 0
            for strategy in evolution_strategies:
                if strategy == 'mutation':
                    evolved_count += self._mutate_theses(evolution_context)
                elif strategy == 'crossover':
                    evolved_count += self._crossover_theses(evolution_context)
                elif strategy == 'adaptation':
                    evolved_count += self._adapt_theses_to_market(evolution_context, context)
                elif strategy == 'refinement':
                    evolved_count += self._refine_theses(evolution_context)
            
            if evolved_count > 0:
                evolution_record = {
                    'timestamp': evolution_context.get('timestamp'),
                    'strategies_used': evolution_strategies,
                    'theses_evolved': evolved_count,
                    'total_theses': len(self.theses),
                    'innovation_pressure': evolution_context.get('innovation_pressure', 0),
                    'diversity_gap': evolution_context.get('diversity_gap', 0)
                }
                
                self.evolution_history.append(evolution_record)
                self.evolution_analytics['successful_evolutions'] += 1
                
                self.log_operator_info(
                    f"🧬 Thesis evolution completed",
                    strategies=', '.join(evolution_strategies),
                    evolved_count=evolved_count,
                    total_theses=len(self.theses)
                )
            
        except Exception as e:
            self.log_operator_error(f"Thesis evolution failed: {e}")
            self.evolution_analytics['failed_evolutions'] += 1

    def _mutate_theses(self, evolution_context: Dict[str, Any]) -> int:
        """Mutate existing theses to create variations"""
        
        mutations = 0
        
        try:
            # Select theses for mutation (prefer underperformers)
            mutation_candidates = []
            for thesis in self.theses:
                perf = self.thesis_performance.get(thesis, {})
                if perf.get('trade_count', 0) >= 3:  # Enough data for assessment
                    avg_pnl = perf.get('total_pnl', 0) / perf['trade_count']
                    if avg_pnl < 5:  # Underperforming
                        mutation_candidates.append(thesis)
            
            # Mutate up to 3 theses
            for _ in range(min(3, len(mutation_candidates))):
                if random.random() < self.evolution_rate * 2:  # Higher rate for mutations
                    original_thesis = random.choice(mutation_candidates)
                    mutated_thesis = self._create_mutation(original_thesis, evolution_context)
                    
                    if mutated_thesis and mutated_thesis not in self.theses:
                        self._add_thesis(mutated_thesis, source='mutation', parent=original_thesis)
                        mutations += 1
                        
                        self.log_operator_info(
                            f"🧬 Thesis mutated",
                            original=original_thesis[:50] + "...",
                            mutated=mutated_thesis[:50] + "..."
                        )
        
        except Exception as e:
            self.log_operator_warning(f"Thesis mutation failed: {e}")
        
        return mutations

    def _create_mutation(self, original_thesis: str, evolution_context: Dict[str, Any]) -> Optional[str]:
        """Create a mutation of an existing thesis"""
        
        try:
            mutation_strategies = [
                'sentiment_flip',    # Change bullish to bearish, etc.
                'timeframe_shift',   # Change timeframe perspective
                'instrument_swap',   # Change focus instrument
                'condition_modify',  # Modify conditions/triggers
                'intensity_adjust'   # Adjust intensity/confidence
            ]
            
            strategy = random.choice(mutation_strategies)
            
            if strategy == 'sentiment_flip':
                # Simple sentiment flipping
                if 'bullish' in original_thesis.lower():
                    return original_thesis.replace('bullish', 'bearish').replace('Bullish', 'Bearish')
                elif 'bearish' in original_thesis.lower():
                    return original_thesis.replace('bearish', 'bullish').replace('Bearish', 'Bullish')
                elif 'buy' in original_thesis.lower():
                    return original_thesis.replace('buy', 'sell').replace('Buy', 'Sell')
                elif 'sell' in original_thesis.lower():
                    return original_thesis.replace('sell', 'buy').replace('Sell', 'Buy')
            
            elif strategy == 'timeframe_shift':
                timeframe_map = {
                    'short-term': 'medium-term',
                    'medium-term': 'long-term',
                    'long-term': 'short-term',
                    'intraday': 'daily',
                    'daily': 'weekly'
                }
                for old_tf, new_tf in timeframe_map.items():
                    if old_tf in original_thesis.lower():
                        return original_thesis.replace(old_tf, new_tf)
            
            elif strategy == 'instrument_swap':
                instrument_map = {
                    'EUR/USD': 'GBP/USD',
                    'GBP/USD': 'USD/JPY',
                    'USD/JPY': 'EUR/USD',
                    'Gold': 'Silver',
                    'Silver': 'Gold'
                }
                for old_inst, new_inst in instrument_map.items():
                    if old_inst in original_thesis:
                        return original_thesis.replace(old_inst, new_inst)
            
            elif strategy == 'condition_modify':
                # Add conditional modifiers
                conditions = [
                    'if volatility increases',
                    'during high volume periods',
                    'with RSI confirmation',
                    'following breakout',
                    'on pullback opportunity'
                ]
                return f"{original_thesis} {random.choice(conditions)}"
            
            elif strategy == 'intensity_adjust':
                # Adjust intensity words
                intensity_map = {
                    'strong': 'moderate',
                    'moderate': 'weak',
                    'weak': 'strong',
                    'significant': 'minor',
                    'minor': 'significant'
                }
                for old_int, new_int in intensity_map.items():
                    if old_int in original_thesis.lower():
                        return original_thesis.replace(old_int, new_int)
            
            # Fallback: add context
            market_regime = evolution_context.get('market_regime', 'current')
            return f"{original_thesis} in {market_regime} market conditions"
            
        except Exception as e:
            self.log_operator_warning(f"Thesis mutation creation failed: {e}")
            return None

    def _crossover_theses(self, evolution_context: Dict[str, Any]) -> int:
        """Create new theses by combining successful ones"""
        
        crossovers = 0
        
        try:
            # Find successful theses for crossover
            successful_theses = []
            for thesis in self.theses:
                perf = self.thesis_performance.get(thesis, {})
                if perf.get('trade_count', 0) >= 2:
                    avg_pnl = perf.get('total_pnl', 0) / perf['trade_count']
                    if avg_pnl > 10:  # Good performance
                        successful_theses.append(thesis)
            
            # Perform crossovers
            if len(successful_theses) >= 2:
                for _ in range(min(2, len(successful_theses) // 2)):
                    if random.random() < self.evolution_rate:
                        parent1, parent2 = random.sample(successful_theses, 2)
                        child_thesis = self._create_crossover(parent1, parent2, evolution_context)
                        
                        if child_thesis and child_thesis not in self.theses:
                            self._add_thesis(child_thesis, source='crossover', 
                                           parent=f"{parent1[:20]}... + {parent2[:20]}...")
                            crossovers += 1
                            
                            self.log_operator_info(
                                f"🧬 Thesis crossover created",
                                parent1=parent1[:30] + "...",
                                parent2=parent2[:30] + "...",
                                child=child_thesis[:50] + "..."
                            )
        
        except Exception as e:
            self.log_operator_warning(f"Thesis crossover failed: {e}")
        
        return crossovers

    def _create_crossover(self, parent1: str, parent2: str, evolution_context: Dict[str, Any]) -> Optional[str]:
        """Create crossover thesis from two successful parents"""
        
        try:
            # Extract key concepts from each parent
            parent1_words = parent1.lower().split()
            parent2_words = parent2.lower().split()
            
            # Find common concepts
            common_words = set(parent1_words) & set(parent2_words)
            
            # Combine unique elements
            unique_p1 = [w for w in parent1_words if w not in common_words and len(w) > 3]
            unique_p2 = [w for w in parent2_words if w not in common_words and len(w) > 3]
            
            # Create crossover combinations
            crossover_templates = [
                f"Market shows {random.choice(unique_p1) if unique_p1 else 'strong'} "
                f"{random.choice(unique_p2) if unique_p2 else 'momentum'} creating opportunity",
                
                f"Combined signals from {random.choice(unique_p1) if unique_p1 else 'technical'} "
                f"and {random.choice(unique_p2) if unique_p2 else 'fundamental'} analysis",
                
                f"Cross-confirmation between {random.choice(unique_p1) if unique_p1 else 'price'} "
                f"action and {random.choice(unique_p2) if unique_p2 else 'volume'} patterns"
            ]
            
            base_thesis = random.choice(crossover_templates)
            
            # Add market context
            market_regime = evolution_context.get('market_regime', 'current')
            return f"{base_thesis} in {market_regime} environment"
            
        except Exception as e:
            self.log_operator_warning(f"Crossover creation failed: {e}")
            return None

    def _adapt_theses_to_market(self, evolution_context: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Adapt theses to current market conditions"""
        
        adaptations = 0
        
        try:
            market_regime = evolution_context.get('market_regime', 'unknown')
            volatility_level = evolution_context.get('volatility_level', 'medium')
            
            # Create market-specific adaptations
            adaptation_templates = {
                'trending': [
                    f"Trend continuation expected in {volatility_level} volatility environment",
                    f"Momentum building across timeframes in trending market",
                    f"Breakout opportunities increase during trend phases"
                ],
                'ranging': [
                    f"Range-bound trading offers mean reversion opportunities",
                    f"Support and resistance levels provide structure in {volatility_level} volatility",
                    f"Oscillator strategies work well in ranging markets"
                ],
                'volatile': [
                    f"High volatility creates both risk and opportunity",
                    f"Volatility expansion signals potential large moves",
                    f"Risk management crucial in volatile conditions"
                ]
            }
            
            # Generate market-adapted theses
            if market_regime in adaptation_templates:
                for template in adaptation_templates[market_regime][:2]:  # Max 2 adaptations
                    if random.random() < self.evolution_rate * 1.5:  # Higher rate for adaptations
                        adapted_thesis = f"{template} - Adapted for current conditions"
                        
                        if adapted_thesis not in self.theses:
                            self._add_thesis(adapted_thesis, source='adaptation', 
                                           parent=f"Market adaptation: {market_regime}")
                            adaptations += 1
        
        except Exception as e:
            self.log_operator_warning(f"Market adaptation failed: {e}")
        
        return adaptations

    def _refine_theses(self, evolution_context: Dict[str, Any]) -> int:
        """Refine existing theses for better performance"""
        
        refinements = 0
        
        try:
            # Select moderately performing theses for refinement
            refinement_candidates = []
            for thesis in self.theses:
                perf = self.thesis_performance.get(thesis, {})
                if perf.get('trade_count', 0) >= 2:
                    avg_pnl = perf.get('total_pnl', 0) / perf['trade_count']
                    if -5 <= avg_pnl <= 15:  # Moderate performance
                        refinement_candidates.append(thesis)
            
            # Refine selected theses
            for thesis in refinement_candidates[:2]:  # Max 2 refinements
                if random.random() < self.evolution_rate:
                    refined_thesis = self._create_refinement(thesis, evolution_context)
                    
                    if refined_thesis and refined_thesis not in self.theses:
                        self._add_thesis(refined_thesis, source='refinement', parent=thesis)
                        refinements += 1
        
        except Exception as e:
            self.log_operator_warning(f"Thesis refinement failed: {e}")
        
        return refinements

    def _create_refinement(self, original_thesis: str, evolution_context: Dict[str, Any]) -> Optional[str]:
        """Create a refined version of an existing thesis"""
        
        try:
            refinement_approaches = [
                'add_timing',      # Add specific timing elements
                'add_confirmation', # Add confirmation requirements
                'add_risk_management', # Add risk considerations
                'add_context',     # Add market context
                'add_precision'    # Add more specific conditions
            ]
            
            approach = random.choice(refinement_approaches)
            
            if approach == 'add_timing':
                timing_elements = [
                    'during European session',
                    'at market open',
                    'following news releases',
                    'during overlap hours',
                    'on weekly close'
                ]
                return f"{original_thesis} {random.choice(timing_elements)}"
            
            elif approach == 'add_confirmation':
                confirmations = [
                    'with volume confirmation',
                    'confirmed by multiple timeframes',
                    'with RSI divergence',
                    'following pattern completion',
                    'with momentum alignment'
                ]
                return f"{original_thesis} {random.choice(confirmations)}"
            
            elif approach == 'add_risk_management':
                risk_elements = [
                    'with tight stops',
                    'using position sizing',
                    'with defined risk-reward',
                    'considering correlation',
                    'managing exposure carefully'
                ]
                return f"{original_thesis} {random.choice(risk_elements)}"
            
            elif approach == 'add_context':
                contexts = [
                    'given current central bank policies',
                    'considering geopolitical factors',
                    'in current volatility environment',
                    'with seasonal considerations',
                    'given recent economic data'
                ]
                return f"{original_thesis} {random.choice(contexts)}"
            
            elif approach == 'add_precision':
                precisions = [
                    'targeting specific price levels',
                    'with precise entry criteria',
                    'focusing on high-probability setups',
                    'using exact technical levels',
                    'with quantified expectations'
                ]
                return f"{original_thesis} {random.choice(precisions)}"
            
            return f"{original_thesis} - Refined version"
            
        except Exception as e:
            self.log_operator_warning(f"Thesis refinement creation failed: {e}")
            return None

    def _cleanup_underperforming_theses(self) -> None:
        """Remove underperforming theses to make room for new ones"""
        
        try:
            if len(self.theses) <= self.capacity // 2:
                return  # Don't clean up if we have few theses
            
            # Identify underperformers
            underperformers = []
            for thesis in self.theses:
                perf = self.thesis_performance.get(thesis, {})
                
                # Criteria for removal
                should_remove = False
                
                # Poor performance after sufficient trades
                if perf.get('trade_count', 0) >= 5:
                    avg_pnl = perf.get('total_pnl', 0) / perf['trade_count']
                    win_rate = perf.get('win_count', 0) / perf['trade_count']
                    
                    if avg_pnl < -15 or win_rate < 0.3:
                        should_remove = True
                
                # Thesis too old
                creation_time = perf.get('creation_time')
                if creation_time:
                    try:
                        created = datetime.datetime.fromisoformat(creation_time)
                        age_hours = (datetime.datetime.now() - created).total_seconds() / 3600
                        if age_hours > 48:  # 48 hours old
                            should_remove = True
                    except Exception:
                        pass
                
                # Too many trades for this thesis
                if perf.get('trade_count', 0) > self.thesis_lifespan:
                    should_remove = True
                
                if should_remove:
                    underperformers.append(thesis)
            
            # Remove underperformers (max 3 at a time)
            removed_count = 0
            for thesis in underperformers[:3]:
                if len(self.theses) > 5:  # Always keep minimum 5 theses
                    self._remove_thesis(thesis, reason="underperformance")
                    removed_count += 1
            
            if removed_count > 0:
                self.log_operator_info(
                    f"🧹 Cleaned up underperforming theses",
                    removed=removed_count,
                    remaining=len(self.theses)
                )
                
        except Exception as e:
            self.log_operator_warning(f"Thesis cleanup failed: {e}")

    def _generate_new_theses_if_needed(self, evolution_context: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Generate new theses if below minimum threshold"""
        
        try:
            min_theses = max(5, self.capacity // 4)
            
            if len(self.theses) < min_theses:
                needed = min_theses - len(self.theses)
                
                for _ in range(needed):
                    new_thesis = self._generate_contextual_thesis(evolution_context, context)
                    if new_thesis and new_thesis not in self.theses:
                        self._add_thesis(new_thesis, source='generation')
                
                self.log_operator_info(
                    f"🌱 Generated new theses",
                    count=needed,
                    total=len(self.theses)
                )
                
        except Exception as e:
            self.log_operator_warning(f"New thesis generation failed: {e}")

    def _generate_contextual_thesis(self, evolution_context: Dict[str, Any], context: Dict[str, Any]) -> Optional[str]:
        """Generate new thesis based on current market context"""
        
        try:
            market_regime = evolution_context.get('market_regime', 'unknown')
            volatility_level = evolution_context.get('volatility_level', 'medium')
            
            # Select appropriate template category
            template_categories = list(self.thesis_templates.keys())
            category = random.choice(template_categories)
            
            # Select template from category
            templates = self.thesis_templates[category]
            template = random.choice(templates)
            
            # Fill template with context
            context_map = {
                '{pattern}': ['ascending', 'descending', 'consolidation', 'expansion'],
                '{timeframe}': ['short-term', 'medium-term', 'daily', 'intraday'],
                '{direction}': ['upward', 'downward', 'sideways', 'reversal'],
                '{support_resistance}': ['strong support', 'key resistance', 'dynamic support'],
                '{level}': ['current levels', 'previous highs', 'previous lows'],
                '{opportunity}': ['buying opportunity', 'selling opportunity', 'range trade'],
                '{accumulation_distribution}': ['accumulation', 'distribution', 'rotation'],
                '{instruments}': ['major pairs', 'USD pairs', 'commodity currencies'],
                '{reversal_continuation}': ['reversal', 'continuation', 'consolidation'],
                '{entry_exit}': ['entry', 'exit', 'scaling'],
                '{expansion_contraction}': ['expansion', 'contraction', 'normalization'],
                '{trading_opportunity}': ['scalping', 'swing trading', 'position trading'],
                '{breakout_breakdown}': ['breakout', 'breakdown', 'false break'],
                '{scalping_swing}': ['scalping', 'swing', 'position'],
                '{portfolio_adjustment}': ['rebalancing', 'hedging', 'exposure change'],
                '{pair_selection}': ['EUR/USD focus', 'GBP/USD preference', 'JPY strength'],
                '{sector_opportunity}': ['commodity focus', 'safe haven flow', 'risk-on trade']
            }
            
            # Replace template variables
            filled_template = template
            for placeholder, options in context_map.items():
                if placeholder in filled_template:
                    filled_template = filled_template.replace(placeholder, random.choice(options))
            
            # Add market context
            thesis = f"{filled_template} in {market_regime} {volatility_level}-volatility environment"
            
            return thesis
            
        except Exception as e:
            self.log_operator_warning(f"Contextual thesis generation failed: {e}")
            return None

    def record_thesis(self, thesis: str) -> None:
        """Record a new thesis with validation"""
        
        try:
            if not isinstance(thesis, str) or not thesis.strip():
                self.log_operator_warning("Invalid thesis provided")
                return
            
            thesis = thesis.strip()
            
            if thesis not in self.theses:
                self._add_thesis(thesis, source='manual')
            
            self.log_operator_info(f"📝 Thesis recorded: {thesis[:50]}...")
            
        except Exception as e:
            self.log_operator_error(f"Thesis recording failed: {e}")

    def record_pnl(self, pnl: float, thesis: Optional[str] = None) -> None:
        """Record P&L for current or specified thesis"""
        
        try:
            # Validate P&L
            if np.isnan(pnl):
                self.log_operator_warning("NaN P&L provided, setting to 0")
                pnl = 0.0
            
            # Determine which thesis to update
            target_thesis = thesis if thesis and thesis in self.theses else (self.theses[-1] if self.theses else None)
            
            if not target_thesis:
                self.log_operator_warning("No thesis available to record P&L against")
                return
            
            # Update thesis performance
            perf = self.thesis_performance[target_thesis]
            perf['pnls'].append(pnl)
            perf['trade_count'] += 1
            perf['total_pnl'] += pnl
            perf['last_update'] = datetime.datetime.now().isoformat()
            
            if pnl > 0:
                perf['win_count'] += 1
            
            # Calculate metrics
            win_rate = perf['win_count'] / perf['trade_count']
            avg_pnl = perf['total_pnl'] / perf['trade_count']
            
            # Update confidence score
            if perf['trade_count'] >= 3:
                perf['confidence_score'] = min(1.0, max(0.1, 
                    0.3 * win_rate + 0.4 * (1 if avg_pnl > 0 else 0) + 0.3 * min(1.0, perf['trade_count'] / 10.0)
                ))
            
            self.log_operator_info(
                f"💰 P&L recorded for thesis",
                pnl=f"€{pnl:+.2f}",
                thesis=target_thesis[:30] + "...",
                total_pnl=f"€{perf['total_pnl']:+.2f}",
                win_rate=f"{win_rate:.1%}",
                trade_count=perf['trade_count']
            )
            
            # Update analytics
            if perf['trade_count'] % 5 == 0:
                self._update_analytics()
                
        except Exception as e:
            self.log_operator_error(f"P&L recording failed: {e}")

    def _add_thesis(self, thesis: str, source: str = 'unknown', parent: Optional[str] = None) -> None:
        """Add new thesis with tracking"""
        
        try:
            # Check capacity
            if len(self.theses) >= self.capacity:
                # Remove oldest thesis if at capacity
                oldest_thesis = min(self.theses, key=lambda t: self.thesis_performance.get(t, {}).get('creation_time', ''))
                self._remove_thesis(oldest_thesis, reason="capacity_limit")
            
            # Add thesis
            self.theses.append(thesis)
            
            # Initialize performance tracking
            perf = self.thesis_performance[thesis]
            perf['creation_time'] = datetime.datetime.now().isoformat()
            perf['category'] = self._categorize_thesis(thesis)
            perf['source'] = source
            
            # Track genealogy
            if parent:
                self.thesis_genealogy[thesis].append({
                    'parent': parent,
                    'source': source,
                    'timestamp': perf['creation_time']
                })
            
            # Update analytics
            self.evolution_analytics['total_theses_created'] += 1
            
        except Exception as e:
            self.log_operator_error(f"Thesis addition failed: {e}")

    def _remove_thesis(self, thesis: str, reason: str = 'unknown') -> None:
        """Remove thesis with tracking"""
        
        try:
            if thesis in self.theses:
                self.theses.remove(thesis)
                
                # Archive performance data
                perf = self.thesis_performance.get(thesis, {})
                perf['removal_time'] = datetime.datetime.now().isoformat()
                perf['removal_reason'] = reason
                
                self.log_operator_info(
                    f"🗑️ Thesis removed",
                    reason=reason,
                    thesis=thesis[:30] + "...",
                    final_pnl=f"€{perf.get('total_pnl', 0):+.2f}",
                    trade_count=perf.get('trade_count', 0)
                )
                
        except Exception as e:
            self.log_operator_warning(f"Thesis removal failed: {e}")

    def _categorize_thesis(self, thesis: str) -> str:
        """Categorize thesis based on content"""
        
        try:
            thesis_lower = thesis.lower()
            
            for category, info in self.thesis_categories.items():
                keywords = info.get('keywords', [])
                if any(keyword in thesis_lower for keyword in keywords):
                    return category
            
            return 'general'
            
        except Exception:
            return 'general'

    def _update_analytics(self) -> None:
        """Update comprehensive analytics"""
        
        try:
            if not self.thesis_performance:
                return
            
            # Calculate average lifespan
            lifespans = []
            for thesis, perf in self.thesis_performance.items():
                if perf.get('trade_count', 0) > 0:
                    lifespans.append(perf['trade_count'])
            
            if lifespans:
                self.evolution_analytics['average_thesis_lifespan'] = np.mean(lifespans)
            
            # Find best performing category
            category_performance = defaultdict(list)
            for thesis, perf in self.thesis_performance.items():
                if perf.get('trade_count', 0) > 0:
                    category = perf.get('category', 'general')
                    avg_pnl = perf.get('total_pnl', 0) / perf['trade_count']
                    category_performance[category].append(avg_pnl)
            
            best_category = 'general'
            best_performance = float('-inf')
            for category, performances in category_performance.items():
                if performances:
                    avg_performance = np.mean(performances)
                    if avg_performance > best_performance:
                        best_performance = avg_performance
                        best_category = category
            
            self.evolution_analytics['best_performing_category'] = best_category
            
            # Update diversity score
            self.evolution_analytics['diversity_score'] = self._calculate_thesis_diversity()
            
            # Calculate innovation rate
            recent_evolutions = len([e for e in self.evolution_history 
                                   if (datetime.datetime.now() - 
                                       datetime.datetime.fromisoformat(e['timestamp'])).days <= 1])
            self.evolution_analytics['innovation_rate'] = recent_evolutions / 24.0  # Per hour
            
        except Exception as e:
            self.log_operator_warning(f"Analytics update failed: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Get thesis metrics for observation"""
        
        try:
            if not self.thesis_performance:
                defaults = np.array([1.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)
                self.log_operator_debug("Using default thesis observations")
                return defaults
            
            # Calculate comprehensive metrics
            unique_theses = len(self.theses)
            total_pnl = sum(perf.get('total_pnl', 0) for perf in self.thesis_performance.values())
            total_trades = sum(perf.get('trade_count', 0) for perf in self.thesis_performance.values())
            
            mean_pnl = total_pnl / max(1, total_trades)
            
            # Best performing thesis
            best_thesis_pnl = 0.0
            if self.thesis_performance:
                for perf in self.thesis_performance.values():
                    if perf.get('trade_count', 0) > 0:
                        thesis_avg = perf.get('total_pnl', 0) / perf['trade_count']
                        best_thesis_pnl = max(best_thesis_pnl, thesis_avg)
            
            # Diversity score
            diversity_score = self._calculate_thesis_diversity()
            
            # Evolution activity score
            evolution_activity = min(1.0, len(self.evolution_history) / 50.0)
            
            observation = np.array([
                float(unique_theses) / self.capacity,  # Thesis capacity utilization
                np.clip(mean_pnl / 50.0, -2.0, 2.0),   # Normalized mean P&L
                np.clip(best_thesis_pnl / 50.0, -2.0, 2.0),  # Normalized best thesis P&L
                diversity_score,                        # Thesis diversity
                evolution_activity                      # Evolution activity
            ], dtype=np.float32)
            
            # Final validation
            if np.any(~np.isfinite(observation)):
                self.log_operator_error(f"Invalid thesis observation: {observation}")
                observation = np.nan_to_num(observation, nan=0.5)
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Thesis observation generation failed: {e}")
            return np.array([0.5, 0.0, 0.0, 0.5, 0.0], dtype=np.float32)

    def _update_info_bus_with_thesis_data(self, info_bus: InfoBus) -> None:
        """Update InfoBus with thesis evolution data"""
        
        try:
            # Prepare thesis data
            thesis_data = {
                'active_theses': self.theses.copy(),
                'thesis_count': len(self.theses),
                'capacity': self.capacity,
                'performance_summary': self._get_performance_summary(),
                'evolution_analytics': self.evolution_analytics.copy(),
                'market_adaptation': self.market_adaptation.copy(),
                'recent_evolutions': list(self.evolution_history)[-5:],
                'best_performing_category': self.evolution_analytics.get('best_performing_category', 'general'),
                'diversity_score': self.evolution_analytics.get('diversity_score', 0.0)
            }
            
            # Add to InfoBus
            InfoBusUpdater.add_module_data(info_bus, 'thesis_evolution_engine', thesis_data)
            
            # Add alerts for important events
            if len(self.theses) < 5:
                InfoBusUpdater.add_alert(
                    info_bus,
                    f"Low thesis count: {len(self.theses)}",
                    'thesis_evolution_engine',
                    'warning',
                    {'thesis_count': len(self.theses)}
                )
            
            if self.evolution_analytics.get('diversity_score', 0) < 0.3:
                InfoBusUpdater.add_alert(
                    info_bus,
                    "Low thesis diversity detected",
                    'thesis_evolution_engine',
                    'info',
                    {'diversity_score': self.evolution_analytics.get('diversity_score', 0)}
                )
            
        except Exception as e:
            self.log_operator_warning(f"InfoBus thesis update failed: {e}")

    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of thesis performance"""
        
        summary = {
            'total_theses': len(self.theses),
            'total_trades': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'best_thesis': None,
            'worst_thesis': None,
            'category_breakdown': defaultdict(int)
        }
        
        try:
            all_trades = 0
            all_wins = 0
            all_pnl = 0.0
            best_performance = float('-inf')
            worst_performance = float('inf')
            
            for thesis, perf in self.thesis_performance.items():
                trade_count = perf.get('trade_count', 0)
                total_pnl = perf.get('total_pnl', 0.0)
                win_count = perf.get('win_count', 0)
                category = perf.get('category', 'general')
                
                all_trades += trade_count
                all_wins += win_count
                all_pnl += total_pnl
                summary['category_breakdown'][category] += 1
                
                if trade_count > 0:
                    avg_pnl = total_pnl / trade_count
                    if avg_pnl > best_performance:
                        best_performance = avg_pnl
                        summary['best_thesis'] = thesis[:50] + ("..." if len(thesis) > 50 else "")
                    
                    if avg_pnl < worst_performance:
                        worst_performance = avg_pnl
                        summary['worst_thesis'] = thesis[:50] + ("..." if len(thesis) > 50 else "")
            
            summary['total_trades'] = all_trades
            summary['total_pnl'] = all_pnl
            summary['win_rate'] = all_wins / max(1, all_trades)
            
        except Exception as e:
            self.log_operator_warning(f"Performance summary generation failed: {e}")
        
        return summary

    def get_thesis_report(self) -> str:
        """Generate comprehensive thesis evolution report"""
        
        performance_summary = self._get_performance_summary()
        
        # Recent evolution activity
        recent_evolution = ""
        if self.evolution_history:
            for evolution in list(self.evolution_history)[-3:]:
                timestamp = evolution['timestamp'][:19].replace('T', ' ')
                strategies = ', '.join(evolution.get('strategies_used', []))
                evolved_count = evolution.get('theses_evolved', 0)
                recent_evolution += f"  • {timestamp}: {strategies} ({evolved_count} theses)\n"
        
        # Active theses summary
        active_theses = ""
        for i, thesis in enumerate(self.theses[:5], 1):  # Show top 5
            perf = self.thesis_performance.get(thesis, {})
            trade_count = perf.get('trade_count', 0)
            total_pnl = perf.get('total_pnl', 0)
            category = perf.get('category', 'general')
            
            status = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "🟡"
            thesis_display = thesis[:60] + ("..." if len(thesis) > 60 else "")
            active_theses += f"  {i}. {thesis_display} ({category}) - {trade_count} trades, €{total_pnl:+.1f} {status}\n"
        
        # Category breakdown
        category_breakdown = ""
        for category, count in performance_summary.get('category_breakdown', {}).items():
            category_breakdown += f"  • {category.title()}: {count} theses\n"
        
        return f"""
🧬 THESIS EVOLUTION ENGINE REPORT
═══════════════════════════════════════
📊 Overview:
• Active Theses: {len(self.theses)}/{self.capacity}
• Total Created: {self.evolution_analytics['total_theses_created']}
• Diversity Score: {self.evolution_analytics.get('diversity_score', 0):.2f}
• Best Category: {self.evolution_analytics.get('best_performing_category', 'general').title()}

💰 Performance Summary:
• Total Trades: {performance_summary['total_trades']}
• Win Rate: {performance_summary['win_rate']:.1%}
• Total P&L: €{performance_summary['total_pnl']:+.2f}
• Average Lifespan: {self.evolution_analytics.get('average_thesis_lifespan', 0):.1f} trades

🔬 Evolution Analytics:
• Successful Evolutions: {self.evolution_analytics['successful_evolutions']}
• Failed Evolutions: {self.evolution_analytics['failed_evolutions']}
• Innovation Rate: {self.evolution_analytics.get('innovation_rate', 0):.2f}/hour
• Current Regime: {self.market_adaptation.get('current_regime', 'unknown').title()}

📝 Active Theses (Top 5):
{active_theses if active_theses else '  📭 No active theses'}

🏷️ Category Breakdown:
{category_breakdown if category_breakdown else '  📭 No categorized theses'}

🔄 Recent Evolution Activity:
{recent_evolution if recent_evolution else '  📭 No recent evolution activity'}

🎯 Current Focus:
• Market Adaptation: {self.market_adaptation.get('current_regime', 'unknown').title()} regime
• Adaptation Triggers: {len(self.market_adaptation.get('adaptation_triggers', []))} recent
• Pending Adaptations: {len(self.market_adaptation.get('pending_adaptations', []))}
        """

    # ================== STATE MANAGEMENT ==================

    def get_state(self) -> Dict[str, Any]:
        """Get complete state for serialization"""
        return {
            "config": {
                "capacity": self.capacity,
                "debug": self.debug,
                "thesis_lifespan": self.thesis_lifespan,
                "performance_threshold": self.performance_threshold,
                "evolution_rate": self.evolution_rate,
                "diversity_target": self.diversity_target
            },
            "thesis_state": {
                "theses": self.theses.copy(),
                "thesis_performance": {k: v.copy() for k, v in self.thesis_performance.items()},
                "evolution_history": list(self.evolution_history),
                "thesis_genealogy": {k: list(v) for k, v in self.thesis_genealogy.items()},
                "successful_mutations": self.successful_mutations.copy(),
                "failed_experiments": self.failed_experiments.copy()
            },
            "analytics": {
                "evolution_analytics": self.evolution_analytics.copy(),
                "market_adaptation": self.market_adaptation.copy()
            },
            "templates": self.thesis_templates.copy(),
            "categories": self.thesis_categories.copy()
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Load state from serialization"""
        
        # Load config
        config = state.get("config", {})
        self.capacity = int(config.get("capacity", self.capacity))
        self.debug = bool(config.get("debug", self.debug))
        self.thesis_lifespan = int(config.get("thesis_lifespan", self.thesis_lifespan))
        self.performance_threshold = float(config.get("performance_threshold", self.performance_threshold))
        self.evolution_rate = float(config.get("evolution_rate", self.evolution_rate))
        self.diversity_target = float(config.get("diversity_target", self.diversity_target))
        
        # Load thesis state
        thesis_state = state.get("thesis_state", {})
        self.theses = list(thesis_state.get("theses", []))
        
        # Restore thesis performance
        performance_data = thesis_state.get("thesis_performance", {})
        self.thesis_performance = defaultdict(lambda: {
            'pnls': [], 'trade_count': 0, 'win_count': 0, 'total_pnl': 0.0,
            'creation_time': datetime.datetime.now().isoformat(),
            'last_update': datetime.datetime.now().isoformat(),
            'category': 'general', 'confidence_score': 0.5,
            'market_conditions': [], 'adaptation_history': []
        })
        for k, v in performance_data.items():
            self.thesis_performance[k] = v
        
        # Restore other state
        self.evolution_history = deque(thesis_state.get("evolution_history", []), maxlen=100)
        
        genealogy_data = thesis_state.get("thesis_genealogy", {})
        self.thesis_genealogy = defaultdict(list)
        for k, v in genealogy_data.items():
            self.thesis_genealogy[k] = list(v)
        
        self.successful_mutations = thesis_state.get("successful_mutations", [])
        self.failed_experiments = thesis_state.get("failed_experiments", [])
        
        # Load analytics
        analytics = state.get("analytics", {})
        self.evolution_analytics = analytics.get("evolution_analytics", self.evolution_analytics)
        self.market_adaptation = analytics.get("market_adaptation", self.market_adaptation)
        
        # Load templates and categories if provided
        self.thesis_templates.update(state.get("templates", {}))
        self.thesis_categories.update(state.get("categories", {}))
        
        self.log_operator_info(
            f"🔄 Thesis evolution engine state loaded",
            theses=len(self.theses),
            total_created=self.evolution_analytics.get('total_theses_created', 0),
            evolutions=len(self.evolution_history)
        )