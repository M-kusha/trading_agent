# ─────────────────────────────────────────────────────────────
# File: modules/memory/memory_budget_optimizer.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
import datetime
import random

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, RiskMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class MemoryBudgetOptimizer(Module, AnalysisMixin, RiskMixin):
    def __init__(self,
                 max_trades: int = 500,
                 max_mistakes: int = 100,
                 max_plays: int = 200,
                 min_size: int = 50,
                 debug: bool = True,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # ensure these exist before Module.__init__ calls _initialize_module_state
        self.max_trades = max_trades
        self.max_mistakes = max_mistakes
        self.max_plays = max_plays
        self.min_size = min_size

        config = ModuleConfig(
            debug=debug,
            max_history=300,
            **kwargs
        )
        super().__init__(config)

        # now safe to initialize genome and state
        self._initialize_genome_parameters(genome, max_trades, max_mistakes, max_plays, min_size)
        self._initialize_module_state()

        self.log_operator_info(
            "Memory budget optimizer initialized",
            total_budget=self.max_trades + self.max_mistakes + self.max_plays,
            trades_allocation=self.max_trades,
            mistakes_allocation=self.max_mistakes,
            plays_allocation=self.max_plays,
            min_size=self.min_size
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict], max_trades: int, 
                                    max_mistakes: int, max_plays: int, min_size: int):
        """Initialize genome-based parameters"""
        if genome:
            self.max_trades = int(genome.get("max_trades", max_trades))
            self.max_mistakes = int(genome.get("max_mistakes", max_mistakes))
            self.max_plays = int(genome.get("max_plays", max_plays))
            self.min_size = int(genome.get("min_size", min_size))
            self.optimization_interval = int(genome.get("optimization_interval", 50))
            self.rebalance_sensitivity = float(genome.get("rebalance_sensitivity", 1.0))
            self.efficiency_weight = float(genome.get("efficiency_weight", 0.7))
            self.recency_weight = float(genome.get("recency_weight", 0.3))
        else:
            self.max_trades = max_trades
            self.max_mistakes = max_mistakes
            self.max_plays = max_plays
            self.min_size = min_size
            self.optimization_interval = 50
            self.rebalance_sensitivity = 1.0
            self.efficiency_weight = 0.7
            self.recency_weight = 0.3

        # Store genome for evolution
        self.genome = {
            "max_trades": self.max_trades,
            "max_mistakes": self.max_mistakes,
            "max_plays": self.max_plays,
            "min_size": self.min_size,
            "optimization_interval": self.optimization_interval,
            "rebalance_sensitivity": self.rebalance_sensitivity,
            "efficiency_weight": self.efficiency_weight,
            "recency_weight": self.recency_weight
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        self._initialize_risk_state()
        
        # Memory performance tracking
        self.memory_performance = {
            "trades": {"size": self.max_trades, "hits": 0, "profit": 0.0, "recent_hits": deque(maxlen=100)},
            "mistakes": {"size": self.max_mistakes, "hits": 0, "profit": 0.0, "recent_hits": deque(maxlen=100)},
            "plays": {"size": self.max_plays, "hits": 0, "profit": 0.0, "recent_hits": deque(maxlen=100)}
        }
        
        # Enhanced tracking
        self.optimization_count = 0
        self.total_profit = 0.0
        self._optimization_history = deque(maxlen=50)
        self._efficiency_trends = {mem_type: deque(maxlen=20) for mem_type in ["trades", "mistakes", "plays"]}
        self._allocation_changes = deque(maxlen=100)
        
        # Performance analytics
        self._memory_utilization_history = deque(maxlen=200)
        self._cost_benefit_analysis = {}
        self._optimal_allocation_predictions = {}
        self._resource_waste_tracking = {"trades": 0, "mistakes": 0, "plays": 0}
        
        # Adaptive thresholds
        self._adaptive_thresholds = {
            'min_efficiency': 0.01,  # Minimum profit per hit
            'rebalance_threshold': 0.1,  # Minimum efficiency difference for rebalancing
            'utilization_target': 0.8,  # Target utilization rate
            'performance_window': 20  # Episodes to consider for performance
        }

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_risk_state()
        
        # Reset memory performance tracking
        for mem_type in self.memory_performance:
            self.memory_performance[mem_type].update({
                "hits": 0, 
                "profit": 0.0,
                "recent_hits": deque(maxlen=100)
            })
        
        # Reset other state
        self.optimization_count = 0
        self.total_profit = 0.0
        self._optimization_history.clear()
        for trend_queue in self._efficiency_trends.values():
            trend_queue.clear()
        self._allocation_changes.clear()
        self._memory_utilization_history.clear()
        self._cost_benefit_analysis.clear()
        self._optimal_allocation_predictions.clear()
        self._resource_waste_tracking = {"trades": 0, "mistakes": 0, "plays": 0}

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract memory usage data
        memory_data = self._extract_memory_data(info_bus, kwargs)
        
        # Process memory utilization
        self._process_memory_utilization(memory_data)
        
        # Update efficiency analytics
        self._update_efficiency_analytics()

    def _extract_memory_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract memory usage data from InfoBus or kwargs"""
        
        # Try InfoBus first
        if info_bus:
            # Extract memory-related information
            recent_trades = info_bus.get('recent_trades', [])
            module_data = info_bus.get('module_data', {})
            
            # Look for memory usage indicators
            memory_usage = {}
            for mem_type in ["trades", "mistakes", "plays"]:
                if mem_type in module_data:
                    memory_usage[mem_type] = module_data[mem_type]
            
            return {
                'recent_trades': recent_trades,
                'memory_usage': memory_usage,
                'step_idx': info_bus.get('step_idx', 0),
                'source': 'info_bus'
            }
        
        # Try kwargs (backward compatibility)
        extracted_data = {}
        
        if "memory_used" in kwargs:
            extracted_data['memory_used'] = kwargs["memory_used"]
            
        if "profit" in kwargs and "source" in kwargs:
            extracted_data['profit'] = float(kwargs["profit"])
            extracted_data['source'] = kwargs["source"]
            
        if extracted_data:
            extracted_data['source'] = 'kwargs'
            return extracted_data
        
        # Return minimal data
        return {'source': 'insufficient_data'}

    def _process_memory_utilization(self, memory_data: Dict[str, Any]):
        """Process memory utilization with enhanced analytics"""
        
        if memory_data.get('source') == 'insufficient_data':
            return
        
        try:
            # Process memory hits
            if 'memory_used' in memory_data:
                memory_type = memory_data['memory_used']
                if memory_type in self.memory_performance:
                    self.memory_performance[memory_type]["hits"] += 1
                    self.memory_performance[memory_type]["recent_hits"].append(self._step_count)
                    
                    self.log_operator_info(
                        f"Memory utilization recorded",
                        type=memory_type,
                        total_hits=self.memory_performance[memory_type]["hits"],
                        step=self._step_count
                    )
            
            # Process profit attribution
            if 'profit' in memory_data and 'source' in memory_data:
                source = memory_data['source']
                profit = memory_data['profit']
                
                if source in self.memory_performance:
                    self.memory_performance[source]["profit"] += profit
                    self.total_profit += profit
                    
                    # Update trading metrics
                    self._update_trading_metrics({'pnl': profit})
                    
                    self.log_operator_info(
                        f"Profit attributed to memory",
                        source=source,
                        profit=f"€{profit:+.2f}",
                        total=f"€{self.memory_performance[source]['profit']:+.2f}"
                    )
            
            # Process InfoBus trades for profit attribution
            if 'recent_trades' in memory_data:
                recent_trades = memory_data['recent_trades']
                self._process_trades_from_info_bus_internal(recent_trades)
            
            # Calculate current utilization
            self._calculate_memory_utilization()
            
            # Update performance metrics
            self._update_performance_metrics()
            
        except Exception as e:
            self.log_operator_error(f"Memory utilization processing failed: {e}")
            self._update_health_status("DEGRADED", f"Processing failed: {e}")

    def _process_trades_from_info_bus_internal(self, recent_trades: List[Dict]):
        """Process recent trades to infer memory usage"""
        
        for trade in recent_trades:
            pnl = trade.get('pnl', 0)
            if abs(pnl) > 0.1:  # Significant trade
                # Infer memory type usage based on trade characteristics
                if pnl > 0:
                    # Profitable trade - likely from trades memory
                    self.memory_performance["trades"]["profit"] += pnl * 0.3  # Partial attribution
                elif pnl < 0:
                    # Loss - could be from mistakes memory learning
                    self.memory_performance["mistakes"]["profit"] += abs(pnl) * 0.1  # Learning value

    def _calculate_memory_utilization(self):
        """Calculate current memory utilization rates"""
        
        try:
            utilization_data = {}
            
            for mem_type, stats in self.memory_performance.items():
                size = stats["size"]
                recent_hits = len(stats["recent_hits"])
                
                # Calculate utilization rate (hits per episode)
                utilization_rate = recent_hits / max(self.optimization_interval, 1)
                
                # Calculate efficiency (profit per hit)
                efficiency = stats["profit"] / max(stats["hits"], 1) if stats["hits"] > 0 else 0.0
                
                utilization_data[mem_type] = {
                    'utilization_rate': utilization_rate,
                    'efficiency': efficiency,
                    'size': size,
                    'waste': max(0, size - recent_hits) / size if size > 0 else 0
                }
            
            # Store utilization history
            self._memory_utilization_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'step': self._step_count,
                'utilization': utilization_data.copy()
            })
            
            # Update waste tracking
            for mem_type, data in utilization_data.items():
                self._resource_waste_tracking[mem_type] = data['waste']
            
        except Exception as e:
            self.log_operator_warning(f"Utilization calculation failed: {e}")

    def _update_efficiency_analytics(self):
        """Update efficiency trend analysis"""
        
        try:
            # Calculate current efficiency for each memory type
            for mem_type, stats in self.memory_performance.items():
                if stats["hits"] > 0:
                    current_efficiency = stats["profit"] / stats["hits"]
                    self._efficiency_trends[mem_type].append(current_efficiency)
                    
                    # Update performance metric
                    self._update_performance_metric(f'{mem_type}_efficiency', current_efficiency)
            
            # Log periodic efficiency summary
            if self._step_count % 100 == 0:
                self._log_efficiency_summary()
                
        except Exception as e:
            self.log_operator_warning(f"Efficiency analytics update failed: {e}")

    def _log_efficiency_summary(self):
        """Log comprehensive efficiency summary"""
        
        try:
            self.log_operator_info(f"Memory efficiency summary (Step {self._step_count})")
            
            for mem_type, stats in self.memory_performance.items():
                size = stats["size"]
                hits = stats["hits"]
                profit = stats["profit"]
                efficiency = profit / max(hits, 1) if hits > 0 else 0.0
                utilization = hits / max(size, 1)
                
                self.log_operator_info(
                    f"  {mem_type.title()}: size={size}, hits={hits}, "
                    f"profit=€{profit:.2f}, efficiency=€{efficiency:.3f}/hit, "
                    f"utilization={utilization:.1%}"
                )
                
        except Exception as e:
            self.log_operator_error(f"Efficiency summary logging failed: {e}")

    def _update_performance_metrics(self):
        """Update comprehensive performance metrics"""
        
        # Calculate total efficiency
        total_hits = sum(stats["hits"] for stats in self.memory_performance.values())
        if total_hits > 0:
            overall_efficiency = self.total_profit / total_hits
            self._update_performance_metric('overall_efficiency', overall_efficiency)
        
        # Calculate allocation optimality score
        optimality = self._calculate_allocation_optimality()
        self._update_performance_metric('allocation_optimality', optimality)
        
        # Update budget utilization
        total_budget = self.max_trades + self.max_mistakes + self.max_plays
        self._update_performance_metric('total_budget', total_budget)

    def _calculate_allocation_optimality(self) -> float:
        """Calculate how optimal current allocation is"""
        
        try:
            if not any(stats["hits"] > 0 for stats in self.memory_performance.values()):
                return 0.5  # No data yet
            
            # Calculate ideal allocation based on efficiency
            efficiencies = {}
            for mem_type, stats in self.memory_performance.items():
                if stats["hits"] > 0:
                    efficiencies[mem_type] = stats["profit"] / stats["hits"]
                else:
                    efficiencies[mem_type] = 0.0
            
            total_efficiency = sum(efficiencies.values())
            if total_efficiency <= 0:
                return 0.5
            
            # Calculate how close current allocation is to optimal
            total_budget = sum(stats["size"] for stats in self.memory_performance.values())
            optimality_score = 0.0
            
            for mem_type, efficiency in efficiencies.items():
                ideal_proportion = efficiency / total_efficiency
                current_proportion = self.memory_performance[mem_type]["size"] / total_budget
                
                # Penalize deviation from ideal
                deviation = abs(ideal_proportion - current_proportion)
                optimality_score += (1.0 - deviation)
            
            return optimality_score / len(efficiencies)
            
        except Exception:
            return 0.5

    def optimize_allocation(self, episode: int, info_bus: Optional[InfoBus] = None):
        """Enhanced allocation optimization with InfoBus context"""
        
        try:
            if episode % self.optimization_interval != 0 or episode == 0:
                return
                
            self.optimization_count += 1
            self.log_operator_info(
                f"Starting memory allocation optimization #{self.optimization_count}",
                episode=episode,
                interval=self.optimization_interval
            )
            
            # Calculate efficiency scores with market context
            efficiency_scores = self._calculate_efficiency_scores(info_bus)
            
            # Determine optimal allocation
            new_allocation = self._calculate_optimal_allocation(efficiency_scores)
            
            # Apply allocation changes
            allocation_changes = self._apply_allocation_changes(new_allocation)
            
            # Record optimization
            self._record_optimization(episode, efficiency_scores, new_allocation, allocation_changes)
            
        except Exception as e:
            self.log_operator_error(f"Allocation optimization failed: {e}")

    def _calculate_efficiency_scores(self, info_bus: Optional[InfoBus]) -> Dict[str, float]:
        """Calculate efficiency scores with market context"""
        
        efficiency_scores = {}
        market_context = {}
        
        # Extract market context if available
        if info_bus:
            market_context = {
                'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
                'regime': InfoBusExtractor.get_market_regime(info_bus),
                'drawdown': InfoBusExtractor.get_drawdown_pct(info_bus)
            }
        
        for mem_type, stats in self.memory_performance.items():
            if stats["hits"] > 0:
                # Base efficiency
                base_efficiency = stats["profit"] / stats["hits"]
                
                # Apply market context adjustments
                market_adj = self._apply_market_context_adjustment(mem_type, base_efficiency, market_context)
                
                # Apply recency weighting
                recent_hits = len(stats["recent_hits"])
                recency_bonus = min(0.2, recent_hits / 50.0)  # Bonus for active use
                
                # Calculate final efficiency score
                final_efficiency = (base_efficiency * market_adj) + recency_bonus
                efficiency_scores[mem_type] = final_efficiency
                
                self.log_operator_info(
                    f"Efficiency calculated for {mem_type}",
                    base=f"€{base_efficiency:.3f}",
                    market_adj=f"{market_adj:.2f}x",
                    recency_bonus=f"+{recency_bonus:.3f}",
                    final=f"€{final_efficiency:.3f}"
                )
            else:
                efficiency_scores[mem_type] = 0.0
        
        return efficiency_scores

    def _apply_market_context_adjustment(self, mem_type: str, base_efficiency: float, 
                                       market_context: Dict[str, Any]) -> float:
        """Apply market context adjustments to efficiency"""
        
        adjustment = 1.0
        
        # Volatility adjustments
        vol_level = market_context.get('volatility_level', 'medium')
        if vol_level == 'high' or vol_level == 'extreme':
            if mem_type == 'mistakes':
                adjustment *= 1.2  # Mistakes memory more valuable in volatile markets
            elif mem_type == 'trades':
                adjustment *= 0.9  # Trade memory less reliable
        
        # Regime adjustments
        regime = market_context.get('regime', 'unknown')
        if regime == 'trending':
            if mem_type == 'plays':
                adjustment *= 1.1  # Playbook more valuable in trends
        elif regime == 'volatile':
            if mem_type == 'mistakes':
                adjustment *= 1.15  # Learn from mistakes in volatility
        
        # Drawdown adjustments
        drawdown = market_context.get('drawdown', 0)
        if drawdown > 10:  # High drawdown
            if mem_type == 'mistakes':
                adjustment *= 1.3  # Critical learning period
            else:
                adjustment *= 0.8  # Conservative on other memories
        
        return adjustment

    def _calculate_optimal_allocation(self, efficiency_scores: Dict[str, float]) -> Dict[str, int]:
        """Calculate optimal memory allocation based on efficiency"""
        
        total_budget = self.max_trades + self.max_mistakes + self.max_plays
        total_efficiency = sum(max(0, score) for score in efficiency_scores.values())
        
        optimal_allocation = {}
        
        if total_efficiency > 0:
            # Proportional allocation based on efficiency
            for mem_type, efficiency in efficiency_scores.items():
                proportion = max(0, efficiency) / total_efficiency
                new_size = int(proportion * total_budget)
                
                # Enforce minimum size
                new_size = max(self.min_size, new_size)
                optimal_allocation[mem_type] = new_size
            
            # Adjust to fit exact budget
            current_total = sum(optimal_allocation.values())
            if current_total != total_budget:
                # Scale proportionally
                scale_factor = total_budget / current_total
                for mem_type in optimal_allocation:
                    scaled_size = int(optimal_allocation[mem_type] * scale_factor)
                    optimal_allocation[mem_type] = max(self.min_size, scaled_size)
                
                # Final adjustment to exact budget
                current_total = sum(optimal_allocation.values())
                difference = total_budget - current_total
                
                # Add/subtract from most efficient memory type
                if difference != 0:
                    best_type = max(efficiency_scores.items(), key=lambda x: x[1])[0]
                    optimal_allocation[best_type] += difference
                    optimal_allocation[best_type] = max(self.min_size, optimal_allocation[best_type])
        else:
            # No efficiency data - equal distribution
            equal_size = (total_budget - 3 * self.min_size) // 3 + self.min_size
            for mem_type in ["trades", "mistakes", "plays"]:
                optimal_allocation[mem_type] = equal_size
        
        return optimal_allocation

    def _apply_allocation_changes(self, new_allocation: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """Apply allocation changes and track modifications"""
        
        changes = {}
        
        # Map memory types to attributes
        type_mapping = {
            "trades": "max_trades",
            "mistakes": "max_mistakes", 
            "plays": "max_plays"
        }
        
        for mem_type, new_size in new_allocation.items():
            if mem_type in type_mapping:
                attr_name = type_mapping[mem_type]
                old_size = getattr(self, attr_name)
                
                # Apply change if significant
                if abs(new_size - old_size) >= max(5, old_size * 0.1):  # At least 5 or 10% change
                    setattr(self, attr_name, new_size)
                    self.memory_performance[mem_type]["size"] = new_size
                    
                    changes[mem_type] = {"old": old_size, "new": new_size}
                    
                    self.log_operator_info(
                        f"Memory allocation updated",
                        type=mem_type,
                        old_size=old_size,
                        new_size=new_size,
                        change=f"{new_size - old_size:+d}"
                    )
        
        return changes

    def _record_optimization(self, episode: int, efficiency_scores: Dict[str, float], 
                           new_allocation: Dict[str, int], changes: Dict[str, Dict[str, int]]):
        """Record optimization for analysis"""
        
        optimization_record = {
            'episode': episode,
            'timestamp': datetime.datetime.now().isoformat(),
            'efficiency_scores': efficiency_scores.copy(),
            'new_allocation': new_allocation.copy(),
            'changes': changes.copy(),
            'total_profit': self.total_profit,
            'optimization_count': self.optimization_count
        }
        
        self._optimization_history.append(optimization_record)
        
        # Store allocation changes
        if changes:
            self._allocation_changes.append({
                'episode': episode,
                'changes': changes,
                'reason': 'efficiency_optimization'
            })

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components with comprehensive metrics"""
        
        try:
            # Current allocation
            allocation = [
                float(self.max_trades),
                float(self.max_mistakes),
                float(self.max_plays)
            ]
            
            # Hit rates (normalized)
            hit_rates = []
            for mem_type in ["trades", "mistakes", "plays"]:
                stats = self.memory_performance[mem_type]
                hit_rate = stats["hits"] / max(1, self.optimization_count * self.optimization_interval)
                hit_rates.append(hit_rate)
            
            # Efficiency scores
            efficiency_scores = []
            for mem_type in ["trades", "mistakes", "plays"]:
                stats = self.memory_performance[mem_type]
                efficiency = stats["profit"] / max(stats["hits"], 1)
                efficiency_scores.append(efficiency / 10.0)  # Normalize
            
            # Utilization rates
            utilization_rates = []
            for mem_type in ["trades", "mistakes", "plays"]:
                stats = self.memory_performance[mem_type]
                recent_hits = len(stats["recent_hits"])
                utilization = recent_hits / max(stats["size"], 1)
                utilization_rates.append(utilization)
            
            # Meta metrics
            meta_metrics = [
                self.total_profit / 1000.0,  # Normalized total profit
                float(self.optimization_count) / 100.0,  # Normalized optimization count
                self._calculate_allocation_optimality(),
                np.mean([self._resource_waste_tracking[t] for t in ["trades", "mistakes", "plays"]])
            ]
            
            # Combine all components
            observation = np.array(
                allocation + hit_rates + efficiency_scores + utilization_rates + meta_metrics,
                dtype=np.float32
            )
            
            return observation
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(15, dtype=np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose memory-informed resource allocation adjustments"""
        
        # Return allocation recommendations as action
        action_dim = 3  # For trades, mistakes, plays
        
        if hasattr(obs, 'shape') and len(obs.shape) > 0:
            action_dim = min(obs.shape[0], 6)  # Cap at reasonable size
        
        # Calculate allocation recommendations
        total_budget = self.max_trades + self.max_mistakes + self.max_plays
        
        recommendations = np.array([
            self.max_trades / total_budget,
            self.max_mistakes / total_budget,
            self.max_plays / total_budget
        ], dtype=np.float32)
        
        # Extend to action_dim if needed
        if action_dim > 3:
            padding = np.zeros(action_dim - 3, dtype=np.float32)
            recommendations = np.concatenate([recommendations, padding])
        else:
            recommendations = recommendations[:action_dim]
        
        return recommendations

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence in allocation recommendations"""
        
        base_confidence = 0.5
        
        # Confidence from optimization history
        if self.optimization_count > 5:
            base_confidence += 0.2
        
        # Confidence from data quality
        total_hits = sum(stats["hits"] for stats in self.memory_performance.values())
        if total_hits > 100:
            base_confidence += 0.2
        
        # Confidence from allocation optimality
        optimality = self._calculate_allocation_optimality()
        base_confidence += optimality * 0.3
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with validation"""
        # Validate and clip genome values
        self.max_trades = int(np.clip(genome.get("max_trades", self.max_trades), self.min_size, 1000))
        self.max_mistakes = int(np.clip(genome.get("max_mistakes", self.max_mistakes), self.min_size, 500))
        self.max_plays = int(np.clip(genome.get("max_plays", self.max_plays), self.min_size, 500))
        self.min_size = int(np.clip(genome.get("min_size", self.min_size), 10, 100))
        self.optimization_interval = int(np.clip(genome.get("optimization_interval", self.optimization_interval), 10, 100))
        self.rebalance_sensitivity = float(np.clip(genome.get("rebalance_sensitivity", self.rebalance_sensitivity), 0.1, 2.0))
        self.efficiency_weight = float(np.clip(genome.get("efficiency_weight", self.efficiency_weight), 0.1, 1.0))
        self.recency_weight = float(np.clip(genome.get("recency_weight", self.recency_weight), 0.0, 0.5))
        
        # Update memory performance sizes
        self.memory_performance["trades"]["size"] = self.max_trades
        self.memory_performance["mistakes"]["size"] = self.max_mistakes
        self.memory_performance["plays"]["size"] = self.max_plays
        
        # Update genome
        self.genome = {
            "max_trades": self.max_trades,
            "max_mistakes": self.max_mistakes,
            "max_plays": self.max_plays,
            "min_size": self.min_size,
            "optimization_interval": self.optimization_interval,
            "rebalance_sensitivity": self.rebalance_sensitivity,
            "efficiency_weight": self.efficiency_weight,
            "recency_weight": self.recency_weight
        }
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance-based adaptation"""
        g = self.genome.copy()
        mutations = []
        
        # Mutate allocation sizes
        if np.random.rand() < mutation_rate:
            # Find least efficient memory type for potential reduction
            efficiencies = {}
            for mem_type, stats in self.memory_performance.items():
                if stats["hits"] > 0:
                    efficiencies[mem_type] = stats["profit"] / stats["hits"]
                else:
                    efficiencies[mem_type] = 0.0
            
            if efficiencies:
                worst_type = min(efficiencies.items(), key=lambda x: x[1])[0]
                param = f"max_{worst_type}"
                old_val = g[param]
                
                # Reduce size of worst performer
                reduction = np.random.randint(10, 50)
                new_val = max(self.min_size, old_val - reduction)
                g[param] = new_val
                mutations.append(f"{param}: {old_val} → {new_val}")
                
                # Give that space to best performer if exists
                if len(efficiencies) > 1:
                    best_type = max(efficiencies.items(), key=lambda x: x[1])[0]
                    best_param = f"max_{best_type}"
                    g[best_param] = g[best_param] + (old_val - new_val)
                    mutations.append(f"{best_param}: +{old_val - new_val}")
        
        # Mutate optimization parameters
        if np.random.rand() < mutation_rate:
            old_val = g["optimization_interval"]
            g["optimization_interval"] = int(np.clip(old_val + np.random.randint(-10, 11), 10, 100))
            mutations.append(f"opt_interval: {old_val} → {g['optimization_interval']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["rebalance_sensitivity"]
            g["rebalance_sensitivity"] = float(np.clip(old_val + np.random.uniform(-0.2, 0.2), 0.1, 2.0))
            mutations.append(f"sensitivity: {old_val:.2f} → {g['rebalance_sensitivity']:.2f}")
        
        if mutations:
            self.log_operator_info(f"Memory budget mutation applied", changes=", ".join(mutations))
            
        self.set_genome(g)
        
    def crossover(self, other: "MemoryBudgetOptimizer") -> "MemoryBudgetOptimizer":
        """Enhanced crossover with efficiency-based selection"""
        if not isinstance(other, MemoryBudgetOptimizer):
            self.log_operator_warning("Crossover with incompatible type")
            return self
        
        # Calculate efficiency to determine better parent
        self_efficiency = self.total_profit / max(sum(s["hits"] for s in self.memory_performance.values()), 1)
        other_efficiency = other.total_profit / max(sum(s["hits"] for s in other.memory_performance.values()), 1)
        
        # Bias crossover toward more efficient parent
        if self_efficiency > other_efficiency:
            # Favor self
            new_g = {k: (self.genome[k] if np.random.rand() < 0.7 else other.genome[k]) for k in self.genome}
        else:
            # Favor other
            new_g = {k: (other.genome[k] if np.random.rand() < 0.7 else self.genome[k]) for k in self.genome}
            
        return MemoryBudgetOptimizer(genome=new_g, debug=self.config.debug)

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check allocation bounds
            if any(getattr(self, f"max_{t}") < self.min_size for t in ["trades", "mistakes", "plays"]):
                return False
                
            # Check memory performance consistency
            for mem_type, stats in self.memory_performance.items():
                if stats["hits"] < 0 or not np.isfinite(stats["profit"]):
                    return False
                    
                if stats["size"] != getattr(self, f"max_{mem_type}"):
                    return False
            
            # Check optimization count
            if self.optimization_count < 0:
                return False
                
            # Check total profit consistency
            calculated_profit = sum(stats["profit"] for stats in self.memory_performance.values())
            if abs(calculated_profit - self.total_profit) > 0.1:
                return False
                
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        budget_details = {
            'allocation_info': {
                'total_budget': self.max_trades + self.max_mistakes + self.max_plays,
                'trades_allocation': self.max_trades,
                'mistakes_allocation': self.max_mistakes,
                'plays_allocation': self.max_plays,
                'min_size_constraint': self.min_size
            },
            'performance_info': {
                'optimization_count': self.optimization_count,
                'total_profit': self.total_profit,
                'allocation_optimality': self._calculate_allocation_optimality(),
                'resource_waste': dict(self._resource_waste_tracking)
            },
            'efficiency_info': {
                'memory_performance': {
                    k: {
                        'hits': v['hits'],
                        'profit': v['profit'],
                        'efficiency': v['profit'] / max(v['hits'], 1),
                        'recent_activity': len(v['recent_hits'])
                    }
                    for k, v in self.memory_performance.items()
                }
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(budget_details)
            return base_details
        
        return budget_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        
        # Convert deques to lists for serialization
        serializable_performance = {}
        for mem_type, stats in self.memory_performance.items():
            serializable_performance[mem_type] = {
                'size': stats['size'],
                'hits': stats['hits'],
                'profit': stats['profit'],
                'recent_hits': list(stats['recent_hits'])
            }
        
        return {
            "memory_performance": serializable_performance,
            "optimization_count": self.optimization_count,
            "total_profit": self.total_profit,
            "genome": self.genome.copy(),
            "optimization_history": list(self._optimization_history)[-20:],  # Keep recent only
            "allocation_changes": list(self._allocation_changes)[-30:],
            "adaptive_thresholds": self._adaptive_thresholds.copy(),
            "resource_waste_tracking": dict(self._resource_waste_tracking),
            "efficiency_trends": {k: list(v) for k, v in self._efficiency_trends.items()}
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        
        # Restore memory performance with deque reconstruction
        performance_data = module_state.get("memory_performance", {})
        for mem_type in ["trades", "mistakes", "plays"]:
            if mem_type in performance_data:
                stats = performance_data[mem_type]
                self.memory_performance[mem_type] = {
                    'size': stats.get('size', getattr(self, f"max_{mem_type}")),
                    'hits': stats.get('hits', 0),
                    'profit': stats.get('profit', 0.0),
                    'recent_hits': deque(stats.get('recent_hits', []), maxlen=100)
                }
        
        # Restore other state
        self.optimization_count = module_state.get("optimization_count", 0)
        self.total_profit = module_state.get("total_profit", 0.0)
        self.set_genome(module_state.get("genome", self.genome))
        self._optimization_history = deque(module_state.get("optimization_history", []), maxlen=50)
        self._allocation_changes = deque(module_state.get("allocation_changes", []), maxlen=100)
        self._adaptive_thresholds = module_state.get("adaptive_thresholds", self._adaptive_thresholds)
        self._resource_waste_tracking = module_state.get("resource_waste_tracking", 
            {"trades": 0, "mistakes": 0, "plays": 0})
        
        # Restore efficiency trends
        efficiency_trends = module_state.get("efficiency_trends", {})
        for mem_type in ["trades", "mistakes", "plays"]:
            if mem_type in efficiency_trends:
                self._efficiency_trends[mem_type] = deque(efficiency_trends[mem_type], maxlen=20)

    def get_budget_optimization_report(self) -> str:
        """Generate operator-friendly budget optimization report"""
        
        # Calculate current efficiency
        total_hits = sum(stats["hits"] for stats in self.memory_performance.values())
        overall_efficiency = self.total_profit / max(total_hits, 1)
        
        # Best performing memory
        best_memory = "None"
        if any(stats["hits"] > 0 for stats in self.memory_performance.values()):
            best_memory = max(
                [(k, v["profit"] / max(v["hits"], 1)) for k, v in self.memory_performance.items() if v["hits"] > 0],
                key=lambda x: x[1]
            )[0]
        
        # Resource utilization
        total_budget = self.max_trades + self.max_mistakes + self.max_plays
        avg_waste = np.mean(list(self._resource_waste_tracking.values()))
        
        return f"""
💰 MEMORY BUDGET OPTIMIZER
═══════════════════════════════════════
💼 Total Budget: {total_budget:,} units
🎯 Optimizations: {self.optimization_count}
📊 Overall Efficiency: €{overall_efficiency:.3f}/hit
🏆 Best Performer: {best_memory.title()}

📈 CURRENT ALLOCATION
• Trades Memory: {self.max_trades:,} ({self.max_trades/total_budget:.1%})
• Mistakes Memory: {self.max_mistakes:,} ({self.max_mistakes/total_budget:.1%})
• Plays Memory: {self.max_plays:,} ({self.max_plays/total_budget:.1%})

⚡ MEMORY PERFORMANCE
""" + "\n".join([
    f"• {mem_type.title()}: "
    f"{stats['hits']:,} hits, "
    f"€{stats['profit']:+.2f} profit, "
    f"€{stats['profit']/max(stats['hits'],1):.3f}/hit"
    for mem_type, stats in self.memory_performance.items()
]) + f"""

🔧 EFFICIENCY METRICS
• Allocation Optimality: {self._calculate_allocation_optimality():.1%}
• Resource Waste: {avg_waste:.1%}
• Total Profit: €{self.total_profit:+.2f}
• Optimization Interval: {self.optimization_interval} episodes

⚙️ CONFIGURATION
• Min Size Constraint: {self.min_size} units
• Rebalance Sensitivity: {self.rebalance_sensitivity:.2f}
• Efficiency Weight: {self.efficiency_weight:.2f}
• Recency Weight: {self.recency_weight:.2f}
        """

    # Maintain backward compatibility
    def step(self, env=None, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, **kwargs)

    def get_state(self):
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state):
        """Backward compatibility state method"""  
        super().set_state(state)