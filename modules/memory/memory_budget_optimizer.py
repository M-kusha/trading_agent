# ─────────────────────────────────────────────────────────────
# File: modules/memory/memory_budget_optimizer.py
# 🚀 PRODUCTION-READY Memory Budget Optimization System
# Advanced memory allocation with SmartInfoBus integration
# ─────────────────────────────────────────────────────────────

import asyncio
import time
import threading
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class MemoryBudgetConfig:
    """Configuration for Memory Budget Optimizer"""
    max_trades: int = 500
    max_mistakes: int = 100
    max_plays: int = 200
    min_size: int = 50
    debug: bool = True
    
    # Performance thresholds
    max_processing_time_ms: float = 150
    circuit_breaker_threshold: int = 3
    optimization_interval: int = 50  # Episodes between optimizations
    
    # Allocation parameters
    rebalance_sensitivity: float = 1.0
    efficiency_weight: float = 0.7
    recency_weight: float = 0.3
    utilization_target: float = 0.8


@module(
    name="MemoryBudgetOptimizer",
    version="3.0.0",
    category="memory",
    provides=["memory_allocation", "budget_optimization", "memory_efficiency", "allocation_strategy"],
    requires=["trades", "mistakes", "playbook_entries", "memory_usage"],
    description="Advanced memory budget optimization with dynamic allocation strategies",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class MemoryBudgetOptimizer(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Advanced memory budget optimizer with SmartInfoBus integration.
    Dynamically allocates memory resources based on performance analytics.
    """

    def __init__(self, 
                 config: Optional[MemoryBudgetConfig] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        self.config = config or MemoryBudgetConfig()
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize memory budget state
        self._initialize_memory_state()
        
        self.logger.info(
            format_operator_message(
                "🧠", "MEMORY_BUDGET_OPTIMIZER_INITIALIZED",
                details=f"Max trades: {self.config.max_trades}, Max mistakes: {self.config.max_mistakes}",
                result="Memory allocation optimization ready",
                context="memory_management"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for memory budget optimization"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="MemoryBudgetOptimizer", 
            log_path="logs/memory_budget.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("MemoryBudgetOptimizer", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for memory operations
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
                "max_trades": int(genome.get("max_trades", self.config.max_trades)),
                "max_mistakes": int(genome.get("max_mistakes", self.config.max_mistakes)),
                "max_plays": int(genome.get("max_plays", self.config.max_plays)),
                "min_size": int(genome.get("min_size", self.config.min_size)),
                "optimization_interval": int(genome.get("optimization_interval", self.config.optimization_interval)),
                "rebalance_sensitivity": float(genome.get("rebalance_sensitivity", self.config.rebalance_sensitivity)),
                "efficiency_weight": float(genome.get("efficiency_weight", self.config.efficiency_weight)),
                "recency_weight": float(genome.get("recency_weight", self.config.recency_weight))
            }
        else:
            self.genome = {
                "max_trades": self.config.max_trades,
                "max_mistakes": self.config.max_mistakes,
                "max_plays": self.config.max_plays,
                "min_size": self.config.min_size,
                "optimization_interval": self.config.optimization_interval,
                "rebalance_sensitivity": self.config.rebalance_sensitivity,
                "efficiency_weight": self.config.efficiency_weight,
                "recency_weight": self.config.recency_weight
            }

    def _initialize_memory_state(self):
        """Initialize memory budget state"""
        # Memory performance tracking
        self.memory_performance = {
            "trades": {
                "size": self.genome["max_trades"], 
                "hits": 0, 
                "profit": 0.0, 
                "recent_hits": deque(maxlen=100),
                "efficiency": 0.0
            },
            "mistakes": {
                "size": self.genome["max_mistakes"], 
                "hits": 0, 
                "profit": 0.0, 
                "recent_hits": deque(maxlen=100),
                "efficiency": 0.0
            },
            "plays": {
                "size": self.genome["max_plays"], 
                "hits": 0, 
                "profit": 0.0, 
                "recent_hits": deque(maxlen=100),
                "efficiency": 0.0
            }
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
            'min_efficiency': 0.01,
            'rebalance_threshold': 0.1,
            'utilization_target': self.config.utilization_target,
            'performance_window': 20
        }

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_memory_health()
                    self._check_allocation_efficiency()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module"""
        try:
            # Set initial allocation in SmartInfoBus
            initial_allocation = {
                "trades": self.genome["max_trades"],
                "mistakes": self.genome["max_mistakes"],
                "plays": self.genome["max_plays"]
            }
            
            self.smart_bus.set(
                'memory_allocation',
                initial_allocation,
                module='MemoryBudgetOptimizer',
                thesis="Initial memory allocation based on configuration"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process memory budget optimization"""
        start_time = time.time()
        
        try:
            # Extract memory usage data
            memory_data = await self._extract_memory_data(**inputs)
            
            if not memory_data:
                return await self._handle_no_data_fallback()
            
            # Process memory utilization
            utilization_result = await self._process_memory_utilization(memory_data)
            
            # Check if optimization is needed
            if self._should_optimize():
                optimization_result = await self._optimize_allocation(memory_data)
                utilization_result.update(optimization_result)
            
            # Generate thesis
            thesis = await self._generate_memory_thesis(memory_data, utilization_result)
            
            # Update SmartInfoBus
            await self._update_memory_smart_bus(utilization_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return utilization_result
            
        except Exception as e:
            return await self._handle_memory_error(e, start_time)

    async def _extract_memory_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract memory usage data from SmartInfoBus"""
        try:
            # Get recent trades
            trades = self.smart_bus.get('trades', 'MemoryBudgetOptimizer') or []
            mistakes = self.smart_bus.get('mistakes', 'MemoryBudgetOptimizer') or []
            playbook_entries = self.smart_bus.get('playbook_entries', 'MemoryBudgetOptimizer') or []
            
            # Get memory usage stats
            memory_usage = self.smart_bus.get('memory_usage', 'MemoryBudgetOptimizer') or {}
            
            # Get performance metrics
            performance_data = self.smart_bus.get('performance_metrics', 'MemoryBudgetOptimizer') or {}
            
            return {
                'trades': trades,
                'mistakes': mistakes,
                'playbook_entries': playbook_entries,
                'memory_usage': memory_usage,
                'performance_data': performance_data,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract memory data: {e}")
            return None

    async def _process_memory_utilization(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process memory utilization with enhanced analytics"""
        try:
            # Update memory performance
            self._update_memory_performance(memory_data)
            
            # Calculate utilization metrics
            utilization_metrics = self._calculate_utilization_metrics()
            
            # Update efficiency analytics
            self._update_efficiency_analytics()
            
            # Calculate allocation optimality
            optimality_score = self._calculate_allocation_optimality()
            
            return {
                'memory_performance': self.memory_performance,
                'utilization_metrics': utilization_metrics,
                'efficiency_trends': dict(self._efficiency_trends),
                'optimality_score': optimality_score,
                'total_profit': self.total_profit,
                'optimization_count': self.optimization_count
            }
            
        except Exception as e:
            self.logger.error(f"Memory utilization processing failed: {e}")
            return self._create_fallback_response("utilization processing failed")

    def _update_memory_performance(self, memory_data: Dict[str, Any]):
        """Update memory performance based on new data"""
        # Process trades
        if 'trades' in memory_data and memory_data['trades']:
            for trade in memory_data['trades'][-10:]:  # Last 10 trades
                if isinstance(trade, dict) and 'pnl' in trade:
                    self.memory_performance['trades']['hits'] += 1
                    self.memory_performance['trades']['profit'] += trade['pnl']
                    self.memory_performance['trades']['recent_hits'].append(time.time())
        
        # Process mistakes
        if 'mistakes' in memory_data and memory_data['mistakes']:
            for mistake in memory_data['mistakes'][-10:]:
                if isinstance(mistake, dict) and 'cost' in mistake:
                    self.memory_performance['mistakes']['hits'] += 1
                    self.memory_performance['mistakes']['profit'] -= mistake.get('cost', 0)
                    self.memory_performance['mistakes']['recent_hits'].append(time.time())
        
        # Process playbook entries
        if 'playbook_entries' in memory_data and memory_data['playbook_entries']:
            for entry in memory_data['playbook_entries'][-10:]:
                if isinstance(entry, dict) and 'value' in entry:
                    self.memory_performance['plays']['hits'] += 1
                    self.memory_performance['plays']['profit'] += entry.get('value', 0)
                    self.memory_performance['plays']['recent_hits'].append(time.time())

    def _calculate_utilization_metrics(self) -> Dict[str, Any]:
        """Calculate memory utilization metrics"""
        metrics = {}
        
        for mem_type, perf in self.memory_performance.items():
            size = perf['size']
            hits = perf['hits']
            profit = perf['profit']
            
            # Calculate efficiency
            efficiency = profit / max(hits, 1)
            perf['efficiency'] = efficiency
            
            # Calculate utilization
            utilization = hits / max(size, 1)
            
            # Calculate recent hit rate
            recent_hits = len([h for h in perf['recent_hits'] if time.time() - h < 3600])  # Last hour
            recent_hit_rate = recent_hits / 60  # Hits per minute
            
            metrics[mem_type] = {
                'efficiency': efficiency,
                'utilization': utilization,
                'recent_hit_rate': recent_hit_rate,
                'total_hits': hits,
                'total_profit': profit,
                'size': size
            }
        
        return metrics

    def _should_optimize(self) -> bool:
        """Check if optimization should be performed"""
        return (self.optimization_count % self.genome["optimization_interval"]) == 0

    async def _optimize_allocation(self, memory_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize memory allocation based on performance"""
        try:
            # Calculate efficiency scores
            efficiency_scores = self._calculate_efficiency_scores()
            
            # Calculate optimal allocation
            optimal_allocation = self._calculate_optimal_allocation(efficiency_scores)
            
            # Apply allocation changes
            changes = self._apply_allocation_changes(optimal_allocation)
            
            # Record optimization
            self._record_optimization(efficiency_scores, optimal_allocation, changes)
            
            self.optimization_count += 1
            
            return {
                'optimization_performed': True,
                'efficiency_scores': efficiency_scores,
                'optimal_allocation': optimal_allocation,
                'allocation_changes': changes
            }
            
        except Exception as e:
            self.logger.error(f"Allocation optimization failed: {e}")
            return {'optimization_performed': False, 'error': str(e)}

    def _calculate_efficiency_scores(self) -> Dict[str, float]:
        """Calculate efficiency scores for each memory type"""
        scores = {}
        
        for mem_type, perf in self.memory_performance.items():
            hits = max(perf['hits'], 1)
            profit = perf['profit']
            size = perf['size']
            
            # Base efficiency (profit per hit)
            base_efficiency = profit / hits
            
            # Utilization bonus/penalty
            utilization = hits / max(size, 1)
            utilization_factor = min(1.0, utilization / self._adaptive_thresholds['utilization_target'])
            
            # Recent performance weight
            recent_hits = len([h for h in perf['recent_hits'] if time.time() - h < 1800])  # Last 30 min
            recency_factor = min(1.0, recent_hits / 10)  # Target 10 hits per 30 min
            
            # Combined score
            efficiency_score = (
                base_efficiency * self.genome["efficiency_weight"] +
                utilization_factor * (1 - self.genome["efficiency_weight"]) * 0.5 +
                recency_factor * self.genome["recency_weight"] * 0.5
            )
            
            scores[mem_type] = efficiency_score
        
        return scores

    def _calculate_optimal_allocation(self, efficiency_scores: Dict[str, float]) -> Dict[str, int]:
        """Calculate optimal memory allocation"""
        total_budget = self.genome["max_trades"] + self.genome["max_mistakes"] + self.genome["max_plays"]
        min_allocation = max(self.genome["min_size"], 20)
        
        # Normalize efficiency scores
        total_efficiency = sum(max(0, score) for score in efficiency_scores.values())
        if total_efficiency == 0:
            # Equal allocation if no efficiency data
            allocation_per_type = (total_budget - 3 * min_allocation) // 3
            return {
                mem_type: min_allocation + allocation_per_type
                for mem_type in efficiency_scores.keys()
            }
        
        # Allocate based on efficiency
        optimal_allocation = {}
        remaining_budget = total_budget - len(efficiency_scores) * min_allocation
        
        for mem_type, efficiency in efficiency_scores.items():
            if efficiency > 0:
                proportion = efficiency / total_efficiency
                allocation = min_allocation + int(remaining_budget * proportion)
            else:
                allocation = min_allocation
            
            optimal_allocation[mem_type] = max(min_allocation, allocation)
        
        # Ensure total doesn't exceed budget
        total_allocated = sum(optimal_allocation.values())
        if total_allocated > total_budget:
            # Scale down proportionally
            scale_factor = total_budget / total_allocated
            for mem_type in optimal_allocation:
                optimal_allocation[mem_type] = max(min_allocation, int(optimal_allocation[mem_type] * scale_factor))
        
        return optimal_allocation

    def _apply_allocation_changes(self, new_allocation: Dict[str, int]) -> Dict[str, Dict[str, int]]:
        """Apply memory allocation changes"""
        changes = {}
        
        for mem_type, new_size in new_allocation.items():
            old_size = self.memory_performance[mem_type]['size']
            change = new_size - old_size
            
            if abs(change) >= self.genome["min_size"] * 0.1:  # Only apply significant changes
                self.memory_performance[mem_type]['size'] = new_size
                changes[mem_type] = {'old': old_size, 'new': new_size, 'change': change}
        
        if changes:
            self._allocation_changes.append({
                'timestamp': time.time(),
                'changes': changes
            })
        
        return changes

    def _record_optimization(self, efficiency_scores: Dict[str, float], 
                           optimal_allocation: Dict[str, int], changes: Dict[str, Dict[str, int]]):
        """Record optimization results"""
        optimization_record = {
            'timestamp': time.time(),
            'efficiency_scores': efficiency_scores.copy(),
            'optimal_allocation': optimal_allocation.copy(),
            'changes_applied': len(changes),
            'total_profit': self.total_profit
        }
        
        self._optimization_history.append(optimization_record)

    def _calculate_allocation_optimality(self) -> float:
        """Calculate how optimal current allocation is"""
        if not self._optimization_history:
            return 0.5  # Neutral score
        
        # Look at recent efficiency trends
        recent_optimizations = list(self._optimization_history)[-5:]
        
        if len(recent_optimizations) < 2:
            return 0.5
        
        # Check if efficiency is improving
        efficiency_trend = 0
        for i in range(1, len(recent_optimizations)):
            prev_scores = recent_optimizations[i-1]['efficiency_scores']
            curr_scores = recent_optimizations[i]['efficiency_scores']
            
            for mem_type in prev_scores:
                if mem_type in curr_scores:
                    if curr_scores[mem_type] > prev_scores[mem_type]:
                        efficiency_trend += 1
                    elif curr_scores[mem_type] < prev_scores[mem_type]:
                        efficiency_trend -= 1
        
        # Normalize to 0-1 range
        max_comparisons = (len(recent_optimizations) - 1) * len(self.memory_performance)
        if max_comparisons > 0:
            optimality = 0.5 + (efficiency_trend / max_comparisons) * 0.5
            return max(0.0, min(1.0, optimality))
        
        return 0.5

    def _update_efficiency_analytics(self):
        """Update efficiency analytics and trends"""
        for mem_type, perf in self.memory_performance.items():
            current_efficiency = perf.get('efficiency', 0.0)
            self._efficiency_trends[mem_type].append(current_efficiency)

    async def _generate_memory_thesis(self, memory_data: Dict[str, Any], 
                                    utilization_result: Dict[str, Any]) -> str:
        """Generate comprehensive memory optimization thesis"""
        try:
            # Analyze performance
            total_hits = sum(perf['hits'] for perf in self.memory_performance.values())
            total_profit = self.total_profit
            avg_efficiency = total_profit / max(total_hits, 1)
            
            # Memory utilization analysis
            utilization_metrics = utilization_result.get('utilization_metrics', {})
            best_performer = max(utilization_metrics.keys(), 
                               key=lambda k: utilization_metrics[k]['efficiency']) if utilization_metrics else 'unknown'
            
            # Optimization status
            optimization_performed = utilization_result.get('optimization_performed', False)
            optimality_score = utilization_result.get('optimality_score', 0.5)
            
            # Recent allocation changes
            recent_changes = len([c for c in self._allocation_changes if time.time() - c['timestamp'] < 3600])
            
            thesis_parts = [
                f"Memory Budget Analysis: Processed {total_hits} total memory accesses generating {total_profit:.2f} total profit",
                f"Average efficiency: {avg_efficiency:.4f} profit per access",
                f"Best performing memory type: {best_performer} with highest efficiency ratio",
                f"Current allocation optimality: {optimality_score:.1%} (0.8+ is considered optimal)",
                f"Recent optimization activity: {recent_changes} allocation changes in last hour"
            ]
            
            if optimization_performed:
                thesis_parts.append("Allocation optimization performed this cycle - memory resources reallocated based on performance")
            
            # Performance trends
            if self._efficiency_trends:
                for mem_type, trend in self._efficiency_trends.items():
                    if len(trend) >= 3:
                        recent_trend = "improving" if trend[-1] > trend[-3] else "declining"
                        thesis_parts.append(f"{mem_type.title()} memory efficiency is {recent_trend}")
            
            # Resource utilization assessment
            total_capacity = sum(perf['size'] for perf in self.memory_performance.values())
            thesis_parts.append(f"Total memory capacity: {total_capacity} entries across all memory types")
            
            if avg_efficiency > 0.01:
                thesis_parts.append("Memory allocation is generating positive returns - system is profitable")
            elif avg_efficiency > 0:
                thesis_parts.append("Memory allocation marginally profitable - optimization recommended")
            else:
                thesis_parts.append("Memory allocation showing losses - immediate rebalancing required")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Memory thesis generation failed: {str(e)} - Basic allocation status maintained"

    async def _update_memory_smart_bus(self, utilization_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with memory optimization results"""
        try:
            # Current allocation
            current_allocation = {
                mem_type: perf['size'] 
                for mem_type, perf in self.memory_performance.items()
            }
            
            self.smart_bus.set(
                'memory_allocation',
                current_allocation,
                module='MemoryBudgetOptimizer',
                thesis=thesis
            )
            
            # Performance metrics
            self.smart_bus.set(
                'memory_efficiency',
                utilization_result.get('utilization_metrics', {}),
                module='MemoryBudgetOptimizer',
                thesis=f"Memory efficiency metrics: {len(self.memory_performance)} types analyzed"
            )
            
            # Budget optimization status
            self.smart_bus.set(
                'budget_optimization',
                {
                    'optimality_score': utilization_result.get('optimality_score', 0.5),
                    'total_profit': self.total_profit,
                    'optimization_count': self.optimization_count,
                    'last_optimization': time.time()
                },
                module='MemoryBudgetOptimizer',
                thesis="Budget optimization status and performance tracking"
            )
            
            # Allocation strategy
            strategy_info = {
                'allocation_method': 'efficiency_based',
                'rebalance_frequency': self.genome["optimization_interval"],
                'efficiency_weight': self.genome["efficiency_weight"],
                'recent_changes': len(self._allocation_changes)
            }
            
            self.smart_bus.set(
                'allocation_strategy',
                strategy_info,
                module='MemoryBudgetOptimizer',
                thesis="Current memory allocation strategy and parameters"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no memory data is available"""
        self.logger.warning("No memory data available - using cached performance metrics")
        
        return {
            'memory_performance': self.memory_performance,
            'utilization_metrics': {},
            'optimality_score': 0.5,
            'fallback_reason': 'no_memory_data'
        }

    async def _handle_memory_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle memory optimization errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "MemoryBudgetOptimizer")
        explanation = self.english_explainer.explain_error(
            "MemoryBudgetOptimizer", str(error), "memory optimization"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "MEMORY_OPTIMIZATION_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="memory_management"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'memory_performance': self.memory_performance,
            'utilization_metrics': {},
            'optimality_score': 0.5,
            'total_profit': self.total_profit,
            'optimization_count': self.optimization_count,
            'fallback_reason': reason,
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    def _update_memory_health(self):
        """Update memory allocation health metrics"""
        try:
            # Check for memory leaks or inefficiencies
            total_hits = sum(perf['hits'] for perf in self.memory_performance.values())
            total_profit = self.total_profit
            
            if total_hits > 0:
                avg_efficiency = total_profit / total_hits
                if avg_efficiency < -0.05:  # Significant losses
                    self._health_status = 'critical'
                elif avg_efficiency < 0:  # Minor losses
                    self._health_status = 'warning'
                else:
                    self._health_status = 'healthy'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    def _check_allocation_efficiency(self):
        """Check allocation efficiency and trigger alerts if needed"""
        try:
            for mem_type, perf in self.memory_performance.items():
                utilization = perf['hits'] / max(perf['size'], 1)
                
                if utilization < 0.1:  # Very low utilization
                    self.logger.warning(
                        format_operator_message(
                            "⚠️", "LOW_MEMORY_UTILIZATION",
                            memory_type=mem_type,
                            utilization=f"{utilization:.1%}",
                            suggestion="Consider reducing allocation size",
                            context="efficiency_monitoring"
                        )
                    )
                elif utilization > 0.95:  # Very high utilization
                    self.logger.warning(
                        format_operator_message(
                            "⚠️", "HIGH_MEMORY_UTILIZATION",
                            memory_type=mem_type,
                            utilization=f"{utilization:.1%}",
                            suggestion="Consider increasing allocation size",
                            context="efficiency_monitoring"
                        )
                    )
            
        except Exception as e:
            self.logger.error(f"Efficiency check failed: {e}")

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'MemoryBudgetOptimizer', 'optimization_cycle', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'MemoryBudgetOptimizer', 'optimization_cycle', 0, False
        )

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'memory_performance': {
                mem_type: {
                    'size': perf['size'],
                    'hits': perf['hits'],
                    'profit': perf['profit'],
                    'efficiency': perf.get('efficiency', 0.0)
                }
                for mem_type, perf in self.memory_performance.items()
            },
            'genome': self.genome.copy(),
            'optimization_count': self.optimization_count,
            'total_profit': self.total_profit,
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self._health_status
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'memory_performance' in state:
            for mem_type, saved_perf in state['memory_performance'].items():
                if mem_type in self.memory_performance:
                    self.memory_performance[mem_type].update(saved_perf)
        
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'optimization_count' in state:
            self.optimization_count = state['optimization_count']
        
        if 'total_profit' in state:
            self.total_profit = state['total_profit']
        
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
            'total_optimizations': self.optimization_count,
            'memory_types': len(self.memory_performance),
            'total_profit': self.total_profit
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def propose_action(self, obs: Any = None, **kwargs) -> np.ndarray:
        """Legacy compatibility for action proposal"""
        return np.array([0.0])
    
    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        return self._calculate_allocation_optimality()