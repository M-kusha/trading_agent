# ─────────────────────────────────────────────────────────────
# File: modules/memory/memory_budget_optimizer.py
# ─────────────────────────────────────────────────────────────

import logging
import os
import numpy as np
import random
from modules.core.core import Module

class MemoryBudgetOptimizer(Module):
    def __init__(self, max_trades: int=500, max_mistakes: int=100, 
                 max_plays: int=200, min_size: int=50, debug=False):
        self.max_trades = max_trades
        self.max_mistakes = max_mistakes
        self.max_plays = max_plays
        self.min_size = min_size  # Minimum memory size
        self.debug = debug
        
        # Enhanced Logger Setup - FIXED
        log_dir = os.path.join("logs", "memory")
        self.logger = logging.getLogger(f"MemoryBudgetOptimizer_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        fh = logging.FileHandler(os.path.join(log_dir, "memory_budget.log"), mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"MemoryBudgetOptimizer initialized - trades={max_trades}, mistakes={max_mistakes}, plays={max_plays}, min_size={min_size}")
        
        self.reset()

    def reset(self):
        # FIX: Track memory utilization and performance with all required keys
        self.memory_performance = {
            "trades": {"size": self.max_trades, "hits": 0, "profit": 0.0},
            "mistakes": {"size": self.max_mistakes, "hits": 0, "profit": 0.0},  # Changed "saves" to "profit"
            "plays": {"size": self.max_plays, "hits": 0, "profit": 0.0}
        }
        self.optimization_count = 0
        self.total_profit = 0.0
        self._step_count = 0
        
        self.logger.info("MemoryBudgetOptimizer reset - performance tracking cleared")

    def step(self, env=None, **kwargs):
        """FIXED: Track memory usage and effectiveness with proper error handling"""
        self._step_count += 1
        
        try:
            # Track memory hits (when memory was useful)
            if "memory_used" in kwargs:
                memory_type = kwargs["memory_used"]
                if memory_type in self.memory_performance:
                    self.memory_performance[memory_type]["hits"] += 1
                    self.logger.debug(f"Step {self._step_count}: Memory hit for {memory_type}")
                    
            # Track profit attribution - FIXED
            if "profit" in kwargs and "source" in kwargs:
                source = kwargs["source"]
                profit = float(kwargs["profit"])  # Ensure it's a float
                
                # FIXED: Ensure the source exists in memory_performance
                if source not in self.memory_performance:
                    self.logger.warning(f"Unknown memory source: {source}")
                    return
                    
                # FIXED: Ensure profit key exists
                if "profit" not in self.memory_performance[source]:
                    self.memory_performance[source]["profit"] = 0.0
                    
                self.memory_performance[source]["profit"] += profit
                self.logger.debug(f"Step {self._step_count}: Profit €{profit:.2f} attributed to {source}")
                self.total_profit += profit
                
            # Log summary periodically
            if self._step_count % 100 == 0:
                self._log_performance_summary()
                
        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            self.logger.error(f"kwargs: {kwargs}")
            self.logger.error(f"memory_performance: {self.memory_performance}")

                
        except Exception as e:
            self.logger.error(f"Error in step: {e}")

    def _log_performance_summary(self):
        """Log current performance summary - FIXED"""
        try:
            self.logger.info(f"Step {self._step_count} - Performance Summary:")
            for mem_type, stats in self.memory_performance.items():
                # FIXED: Use .get() to safely access dictionary values
                size = stats.get("size", 0)
                hits = stats.get("hits", 0)
                profit = stats.get("profit", 0.0)  # This was causing the error
                
                efficiency = profit / max(1, hits) if hits > 0 else 0.0
                
                self.logger.info(f"  {mem_type}: size={size}, hits={hits}, profit=€{profit:.2f}, efficiency=€{efficiency:.2f}/hit")
        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")
            # Log the actual stats structure for debugging
            self.logger.error(f"Memory performance structure: {self.memory_performance}")

    def optimize_allocation(self, episode: int):
        """FIX: Periodically rebalance memory based on performance"""
        try:
            if episode % 50 != 0 or episode == 0:  # Every 50 episodes
                self.logger.debug(f"Episode {episode}: Not optimization interval")
                return
                
            self.optimization_count += 1
            self.logger.info(f"Episode {episode}: Starting optimization #{self.optimization_count}")
            
            # Calculate efficiency for each memory type
            efficiencies = {}
            for mem_type, stats in self.memory_performance.items():
                if stats["hits"] > 0:
                    # Profit per hit
                    efficiency = stats["profit"] / stats["hits"]
                else:
                    efficiency = 0.0
                efficiencies[mem_type] = efficiency
                self.logger.info(f"  {mem_type} efficiency: €{efficiency:.3f}/hit")
                
            # Total memory budget
            total_budget = self.max_trades + self.max_mistakes + self.max_plays
            self.logger.info(f"Total memory budget: {total_budget}")
            
            # Reallocate based on efficiency
            if sum(efficiencies.values()) > 0:
                # Proportional allocation based on efficiency
                total_eff = sum(max(0, e) for e in efficiencies.values())
                
                if total_eff > 0:
                    new_sizes = {}
                    for mem_type, eff in efficiencies.items():
                        proportion = max(0, eff) / total_eff
                        new_size = int(proportion * total_budget)
                        # Enforce minimum size
                        new_size = max(self.min_size, new_size)
                        new_sizes[mem_type] = new_size
                    
                    # Adjust to fit budget exactly
                    size_sum = sum(new_sizes.values())
                    if size_sum > total_budget:
                        # Scale down proportionally
                        scale = total_budget / size_sum
                        for mem_type in new_sizes:
                            new_sizes[mem_type] = max(self.min_size, int(new_sizes[mem_type] * scale))
                    
                    # Apply new sizes
                    old_sizes = {
                        "trades": self.max_trades,
                        "mistakes": self.max_mistakes,
                        "plays": self.max_plays
                    }
                    
                    self.max_trades = new_sizes.get("trades", self.max_trades)
                    self.max_mistakes = new_sizes.get("mistakes", self.max_mistakes)
                    self.max_plays = new_sizes.get("plays", self.max_plays)
                    
                    # Log changes
                    changes_made = False
                    for mem_type in ["trades", "mistakes", "plays"]:
                        old = old_sizes[mem_type]
                        new = getattr(self, f"max_{mem_type}")
                        if old != new:
                            self.logger.info(f"  {mem_type}: {old} -> {new} (efficiency: €{efficiencies[mem_type]:.3f})")
                            changes_made = True
                            
                    if not changes_made:
                        self.logger.info("  No allocation changes needed")
                else:
                    self.logger.warning("  Total efficiency is 0, no reallocation")
            else:
                self.logger.warning("  No efficiency data available")
                
        except Exception as e:
            self.logger.error(f"Error in optimize_allocation: {e}")

    def get_observation_components(self) -> np.ndarray:
        """FIX: Return memory utilization metrics"""
        try:
            # Calculate hit rates
            hit_rates = []
            for mem_type in ["trades", "mistakes", "plays"]:
                stats = self.memory_performance[mem_type]
                size = getattr(self, f"max_{mem_type}")
                hit_rate = stats["hits"] / max(1, self.optimization_count * 50)  # Per episode
                hit_rates.append(hit_rate)
                
            result = np.array([
                float(self.max_trades),
                float(self.max_mistakes),
                float(self.max_plays),
                *hit_rates
            ], dtype=np.float32)
            
            self.logger.debug(f"Observation: sizes=[{self.max_trades}, {self.max_mistakes}, {self.max_plays}], hit_rates={hit_rates}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.zeros(6, np.float32)

    def get_state(self):
        return {
            "max_trades": self.max_trades,
            "max_mistakes": self.max_mistakes,
            "max_plays": self.max_plays,
            "memory_performance": self.memory_performance,
            "optimization_count": self.optimization_count,
            "total_profit": self.total_profit,
            "step_count": self._step_count
        }

    def set_state(self, state):
        self.max_trades = state.get("max_trades", self.max_trades)
        self.max_mistakes = state.get("max_mistakes", self.max_mistakes)
        self.max_plays = state.get("max_plays", self.max_plays)
        self.memory_performance = state.get("memory_performance", self.memory_performance)
        self.optimization_count = state.get("optimization_count", 0)
        self.total_profit = state.get("total_profit", 0.0)
        self._step_count = state.get("step_count", 0)
        
        self.logger.info(f"State restored: optimizations={self.optimization_count}, total_profit=€{self.total_profit:.2f}, steps={self._step_count}")

    def mutate(self):
        """Smart mutation based on performance"""
        try:
            # Only mutate the least efficient memory type
            efficiencies = {}
            for mem_type, stats in self.memory_performance.items():
                if stats["hits"] > 0:
                    efficiencies[mem_type] = stats["profit"] / stats["hits"]
                else:
                    efficiencies[mem_type] = 0.0
                    
            if efficiencies:
                # Find worst performer
                worst_type = min(efficiencies.items(), key=lambda x: x[1])[0]
                param = f"max_{worst_type}"
                old_val = getattr(self, param)
                
                # Reduce size of worst performer
                new_val = max(self.min_size, old_val - random.randint(10, 50))
                setattr(self, param, new_val)
                
                # Give that space to best performer
                if len(efficiencies) > 1:
                    best_type = max(efficiencies.items(), key=lambda x: x[1])[0]
                    best_param = f"max_{best_type}"
                    current_best = getattr(self, best_param)
                    setattr(self, best_param, current_best + (old_val - new_val))
                    
                self.logger.info(f"Mutated {param}: {old_val} -> {new_val} (worst efficiency: €{efficiencies[worst_type]:.3f})")
            else:
                self.logger.warning("No efficiency data for mutation")
                
        except Exception as e:
            self.logger.error(f"Error in mutation: {e}")
                
    def crossover(self, other: "MemoryBudgetOptimizer"):
        child = MemoryBudgetOptimizer(
            max_trades=int((self.max_trades + other.max_trades) / 2),
            max_mistakes=int((self.max_mistakes + other.max_mistakes) / 2),
            max_plays=int((self.max_plays + other.max_plays) / 2),
            min_size=self.min_size,
            debug=self.debug
        )
        self.logger.info("Crossover completed")
        return child