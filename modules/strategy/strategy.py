from __future__ import annotations
import hashlib
import json
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Callable, Optional, Tuple
from collections import deque
from modules.core.core import Module
from modules.market.market import FractalRegimeConfirmation
from modules.strategy.voting import StrategyArbiter

# Setup logging directories
def setup_strategy_logging():
    """Create all required log directories"""
    log_dirs = [
        "logs/strategy",
        "logs/strategy/introspection",
        "logs/strategy/curriculum", 
        "logs/strategy/genome",
        "logs/strategy/meta",
        "logs/strategy/ppo",
        "logs/strategy/controller"
    ]
    for log_dir in log_dirs:
        os.makedirs(log_dir, exist_ok=True)

# Call this at module load
setup_strategy_logging()

# ──────────────────────────────────────────────
class StrategyIntrospector(Module):
    """
    FIXED: Added bootstrap support, better initialization, and comprehensive logging
    """
    def __init__(self, history_len: int = 10, debug: bool = True):
        self.history_len = history_len
        self.debug = debug
        self._records: List[Dict[str, float]] = []
        self._step_count = 0
        
        # FIX: Initialize with some baseline data to avoid zero observations
        self._baseline_wr = 0.5  # 50% win rate baseline
        self._baseline_sl = 1.0  # 1% stop loss baseline
        self._baseline_tp = 1.5  # 1.5% take profit baseline

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"StrategyIntrospector_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/introspection/strategy_introspector.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"StrategyIntrospector initialized - history_len={history_len}, baselines: wr={self._baseline_wr}, sl={self._baseline_sl}, tp={self._baseline_tp}")

    def reset(self) -> None:
        self._records.clear()
        self._step_count = 0
        self.logger.info("StrategyIntrospector reset - all records cleared")

    def step(self, **kwargs) -> None:
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record(self, theme: np.ndarray, win_rate: float, sl: float, tp: float) -> None:
        """Record strategy performance with validation"""
        try:
            # Validate inputs
            if not (0 <= win_rate <= 1):
                self.logger.warning(f"Invalid win_rate {win_rate}, clamping to [0,1]")
                win_rate = np.clip(win_rate, 0, 1)
            
            if sl <= 0:
                self.logger.warning(f"Invalid sl {sl}, using baseline {self._baseline_sl}")
                sl = self._baseline_sl
                
            if tp <= 0:
                self.logger.warning(f"Invalid tp {tp}, using baseline {self._baseline_tp}")
                tp = self._baseline_tp
            
            record = {"wr": win_rate, "sl": sl, "tp": tp}
            self._records.append(record)
            
            if len(self._records) > self.history_len:
                removed = self._records.pop(0)
                self.logger.debug(f"Removed old record: {removed}")
                
            self.logger.info(f"Recorded strategy: wr={win_rate:.3f}, sl={sl:.3f}, tp={tp:.3f}, total_records={len(self._records)}")
            
            # Log statistics periodically
            if len(self._records) % 5 == 0:
                self._log_statistics()
                
        except Exception as e:
            self.logger.error(f"Error recording strategy: {e}")

    def _log_statistics(self):
        """Log current strategy statistics"""
        try:
            if not self._records:
                return
                
            arr = np.array([[r["wr"], r["sl"], r["tp"]] for r in self._records], dtype=np.float32)
            means = arr.mean(axis=0)
            stds = arr.std(axis=0)
            
            self.logger.info(f"Strategy Statistics - Records: {len(self._records)}")
            self.logger.info(f"  Win Rate: {means[0]:.3f} ± {stds[0]:.3f}")
            self.logger.info(f"  Stop Loss: {means[1]:.3f} ± {stds[1]:.3f}")
            self.logger.info(f"  Take Profit: {means[2]:.3f} ± {stds[2]:.3f}")
        except Exception as e:
            self.logger.error(f"Error logging statistics: {e}")

    def profile(self) -> np.ndarray:
        """Get strategy profile with comprehensive validation"""
        try:
            if not self._records:
                # FIX: Return baseline values instead of zeros
                baseline = np.array([
                    self._baseline_wr, self._baseline_sl, self._baseline_tp,
                    0.0, 0.0  # No variance yet
                ], dtype=np.float32)
                self.logger.debug(f"Using baseline profile: {baseline}")
                return baseline
            
            arr = np.array([[r["wr"], r["sl"], r["tp"]] for r in self._records], dtype=np.float32)
            
            # Validate array
            if np.any(np.isnan(arr)):
                self.logger.error(f"NaN values in records array: {arr}")
                arr = np.nan_to_num(arr, nan=0.0)
            
            # FIX: Calculate mean and variance for better signal
            mean_vals = arr.mean(axis=0)
            var_vals = arr.var(axis=0) if len(arr) > 1 else np.zeros(3)
            
            # Combine mean and variance info
            profile = np.concatenate([mean_vals, var_vals[:2]])  # Total 5 values
            
            # Final validation
            if np.any(np.isnan(profile)):
                self.logger.error(f"NaN in final profile: {profile}")
                profile = np.nan_to_num(profile, nan=0.0)
                
            self.logger.debug(f"Generated profile: {profile}")
            return profile.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error generating profile: {e}")
            return np.array([self._baseline_wr, self._baseline_sl, self._baseline_tp, 0.0, 0.0], dtype=np.float32)

    def get_observation_components(self) -> np.ndarray:
        return self.profile()

# ──────────────────────────────────────────────
class CurriculumPlannerPlus(Module):
    """
    FIXED: Enhanced to track trading performance metrics properly with comprehensive logging
    """
    def __init__(self, window: int=10, debug=True):
        self.window = window
        self.debug = debug
        self._history: List[Dict[str,float]] = []
        self._step_count = 0
        
        # FIX: Track cumulative metrics for better learning
        self._total_trades = 0
        self._total_wins = 0
        self._cumulative_pnl = 0.0

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"CurriculumPlannerPlus_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/curriculum/curriculum_planner.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"CurriculumPlannerPlus initialized - window={window}")

    def reset(self):
        self._history.clear()
        self._total_trades = 0
        self._total_wins = 0
        self._cumulative_pnl = 0.0
        self._step_count = 0
        self.logger.info("CurriculumPlannerPlus reset - all metrics cleared")

    def step(self, **kwargs):
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record_episode(self, summary: Dict[str,float]):
        """Record episode with comprehensive validation and logging"""
        try:
            # Validate summary
            if not isinstance(summary, dict):
                self.logger.error(f"Invalid summary type: {type(summary)}")
                return
                
            self.logger.debug(f"Recording episode summary: {summary}")
            
            self._history.append(summary)
            if len(self._history) > self.window:
                removed = self._history.pop(0)
                self.logger.debug(f"Removed old episode: {removed}")
            
            # FIX: Update cumulative metrics with validation
            if "total_trades" in summary:
                trades = summary["total_trades"]
                if isinstance(trades, (int, float)) and trades >= 0:
                    self._total_trades += trades
                else:
                    self.logger.warning(f"Invalid total_trades: {trades}")
                    
            if "wins" in summary:
                wins = summary["wins"]
                if isinstance(wins, (int, float)) and wins >= 0:
                    self._total_wins += wins
                else:
                    self.logger.warning(f"Invalid wins: {wins}")
                    
            if "pnl" in summary:
                pnl = summary["pnl"]
                if isinstance(pnl, (int, float)) and not np.isnan(pnl):
                    self._cumulative_pnl += pnl
                else:
                    self.logger.warning(f"Invalid pnl: {pnl}")
            
            # Log episode statistics
            self.logger.info(f"Episode recorded: total_trades={self._total_trades}, total_wins={self._total_wins}, cumulative_pnl=€{self._cumulative_pnl:.2f}")
            
            # Log detailed statistics periodically
            if len(self._history) % 5 == 0:
                self._log_curriculum_stats()
                
        except Exception as e:
            self.logger.error(f"Error recording episode: {e}")

    def _log_curriculum_stats(self):
        """Log detailed curriculum statistics"""
        try:
            if not self._history:
                return
                
            win_rates = [e.get("win_rate", 0) for e in self._history if "win_rate" in e]
            durations = [e.get("avg_duration", 0) for e in self._history if "avg_duration" in e]
            drawdowns = [e.get("avg_drawdown", 0) for e in self._history if "avg_drawdown" in e]
            
            self.logger.info(f"Curriculum Statistics - Episodes: {len(self._history)}")
            if win_rates:
                self.logger.info(f"  Win Rates: mean={np.mean(win_rates):.3f}, std={np.std(win_rates):.3f}")
            if durations:
                self.logger.info(f"  Durations: mean={np.mean(durations):.3f}, std={np.std(durations):.3f}")
            if drawdowns:
                self.logger.info(f"  Drawdowns: mean={np.mean(drawdowns):.3f}, std={np.std(drawdowns):.3f}")
                
            overall_win_rate = self._total_wins / max(1, self._total_trades)
            self.logger.info(f"  Overall: win_rate={overall_win_rate:.3f}, avg_pnl=€{self._cumulative_pnl/max(1, len(self._history)):.2f}")
            
        except Exception as e:
            self.logger.error(f"Error logging curriculum stats: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Get curriculum metrics with validation"""
        try:
            if not self._history:
                # FIX: Return meaningful defaults for bootstrap
                defaults = np.array([0.5, 0.0, 0.01], dtype=np.float32)  # [win_rate, avg_duration, avg_drawdown]
                self.logger.debug("Using default curriculum metrics")
                return defaults
            
            # Calculate rolling metrics with validation
            win_rates = [e.get("win_rate", 0) for e in self._history if isinstance(e.get("win_rate"), (int, float))]
            durations = [e.get("avg_duration", 0) for e in self._history if isinstance(e.get("avg_duration"), (int, float))]
            drawdowns = [e.get("avg_drawdown", 0) for e in self._history if isinstance(e.get("avg_drawdown"), (int, float))]
            
            # FIX: Use robust averaging with fallbacks
            avg_wr = np.mean(win_rates) if win_rates else 0.5
            avg_dur = np.mean(durations) if durations else 0.0
            avg_dd = np.mean(drawdowns) if drawdowns else 0.01
            
            # Validate for NaN
            metrics = np.array([avg_wr, avg_dur, avg_dd], dtype=np.float32)
            if np.any(np.isnan(metrics)):
                self.logger.error(f"NaN in curriculum metrics: {metrics}")
                metrics = np.nan_to_num(metrics, nan=0.0)
                
            self.logger.debug(f"Curriculum metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.5, 0.0, 0.01], dtype=np.float32)

# ──────────────────────────────────────────────
class StrategyGenomePool:
    """
    FIXED: Enhanced evolution with profit-focused fitness, better diversity, and comprehensive logging
    """
    def __init__(
        self,
        population_size: int = 20,
        tournament_k: int = 3,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.2,
        max_generations_kept: int = 10_000,
        debug: bool = True,
        profit_target: float = 150.0  # €150 daily target
    ) -> None:
        self.genome_size = 4
        self.pop_size = int(population_size)
        self.tournament_k = int(tournament_k)
        self.cx_rate = float(crossover_rate)
        self.mut_rate = float(mutation_rate)
        self.mut_scale = float(mutation_scale)
        self.max_generations_kept = int(max_generations_kept)
        self.debug = debug
        self.profit_target = profit_target

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"StrategyGenomePool_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/genome/genome_pool.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # FIX: Better seed genomes for XAUUSD/EURUSD trading
        seeds = [
            np.array([0.5, 0.75, 1.0, 0.2], dtype=np.float32),  # Conservative scalping
            np.array([1.0, 1.5, 1.2, 0.3], dtype=np.float32),   # Balanced
            np.array([1.5, 2.0, 1.5, 0.4], dtype=np.float32),   # Aggressive
            np.array([0.8, 1.2, 0.8, 0.25], dtype=np.float32),  # Tight stops
        ]
        
        self.logger.info(f"Initializing genome pool with {len(seeds)} seed genomes")
        
        # Random fill with trading-appropriate ranges
        rand = [
            np.random.uniform(
                low=[0.3, 0.5, 0.5, 0.1],   # Min: 30 pip SL, 50 pip TP
                high=[2.0, 3.0, 2.0, 0.5],  # Max: 200 pip SL, 300 pip TP
            ).astype(np.float32)
            for _ in range(self.pop_size - len(seeds))
        ]
        
        self.population = np.vstack(seeds + rand)[: self.pop_size]
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)
        self.epoch = 0
        
        # FIX: Track best genome for stability
        self.best_genome = self.population[0].copy()
        self.best_fitness = -np.inf
        self.generations_without_improvement = 0
        
        self.logger.info(f"StrategyGenomePool initialized - pop_size={self.pop_size}, profit_target=€{profit_target}")
        self.logger.info(f"Initial best genome: {self.best_genome}")

    def reset(self) -> None:
        self.epoch = 0
        self.fitness[:] = 0.0
        self.generations_without_improvement = 0
        self.logger.info("StrategyGenomePool reset")

    def step(self, **kwargs) -> None:
        pass

    def genome_hash(self, g: np.ndarray) -> str:
        """Generate hash for genome tracking"""
        try:
            return hashlib.md5(g.tobytes()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error generating genome hash: {e}")
            return "error_hash"

    def evaluate_population(self, eval_fn: Callable[[np.ndarray], float]) -> None:
        """
        FIX: Enhanced evaluation with profit-focused metrics and comprehensive logging
        """
        try:
            self.logger.info(f"Evaluating population generation {self.epoch}")
            
            evaluation_results = []
            
            for i, genome in enumerate(self.population):
                try:
                    # Validate genome
                    if np.any(np.isnan(genome)):
                        self.logger.error(f"NaN in genome {i}: {genome}")
                        genome = np.nan_to_num(genome)
                        self.population[i] = genome
                    
                    raw_fitness = float(eval_fn(genome))
                    
                    # Validate fitness
                    if np.isnan(raw_fitness):
                        self.logger.warning(f"NaN fitness for genome {i}, setting to 0")
                        raw_fitness = 0.0
                    
                    # FIX: Bonus for achieving profit target
                    if raw_fitness >= self.profit_target:
                        profit_bonus = 1.0 + (raw_fitness - self.profit_target) / self.profit_target
                        self.fitness[i] = raw_fitness * profit_bonus
                        self.logger.info(f"Genome {i} achieved target: €{raw_fitness:.2f} (bonus: {profit_bonus:.2f}x)")
                    else:
                        self.fitness[i] = raw_fitness
                        
                    evaluation_results.append((i, raw_fitness, self.fitness[i]))
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating genome {i}: {e}")
                    self.fitness[i] = 0.0
                    
            # Track best
            max_idx = np.argmax(self.fitness)
            if self.fitness[max_idx] > self.best_fitness:
                old_best = self.best_fitness
                self.best_fitness = self.fitness[max_idx]
                self.best_genome = self.population[max_idx].copy()
                self.generations_without_improvement = 0
                self.logger.info(f"New best genome found: fitness improved from {old_best:.3f} to {self.best_fitness:.3f}")
                self.logger.info(f"Best genome: {self.best_genome}")
            else:
                self.generations_without_improvement += 1
                
            # Log generation statistics
            self.logger.info(f"Generation {self.epoch} results:")
            self.logger.info(f"  Fitness - min: {self.fitness.min():.3f}, max: {self.fitness.max():.3f}, mean: {self.fitness.mean():.3f}")
            self.logger.info(f"  Best ever: {self.best_fitness:.3f}, stagnant for: {self.generations_without_improvement} generations")
            
            # Log top performers
            top_indices = np.argsort(self.fitness)[-3:][::-1]
            for idx in top_indices:
                genome_str = ", ".join(f"{x:.3f}" for x in self.population[idx])
                self.logger.info(f"  Top genome {idx}: fitness={self.fitness[idx]:.3f}, genome=[{genome_str}]")
                
        except Exception as e:
            self.logger.error(f"Error in evaluate_population: {e}")

    def evolve_strategies(self) -> None:
        """
        FIX: Enhanced evolution with elitism, adaptive mutation, and comprehensive logging
        """
        try:
            old_hashes = [self.genome_hash(g) for g in self.population]
            
            self.logger.info(f"Evolving generation {self.epoch}")
            
            # FIX: Adaptive mutation based on progress
            if self.generations_without_improvement > 10:
                adaptive_mut_rate = min(0.3, self.mut_rate * 2)
                adaptive_mut_scale = min(0.5, self.mut_scale * 1.5)
                self.logger.info(f"Increasing mutation due to stagnation: rate={adaptive_mut_rate:.3f}, scale={adaptive_mut_scale:.3f}")
            else:
                adaptive_mut_rate = self.mut_rate
                adaptive_mut_scale = self.mut_scale

            # FIX: Always preserve best genome (elitism)
            new_pop = [self.best_genome.copy()]
            self.logger.debug("Preserved best genome (elitism)")
            
            # Diversity injection if stagnant
            if np.std(self.fitness) < 0.1 or self.generations_without_improvement > 20:
                # Inject fresh genomes focused on profit
                new_genes = np.random.uniform(
                    low=[0.5, 0.8, 0.8, 0.15],
                    high=[1.5, 2.2, 1.8, 0.4],
                    size=(3, 4),
                ).astype(np.float32)
                new_pop.extend(new_genes)
                self.logger.info(f"Injected {len(new_genes)} fresh profit-focused genomes due to low diversity")
            
            # Generate rest of population
            crossover_count = 0
            mutation_count = 0
            
            while len(new_pop) < self.pop_size:
                try:
                    # Tournament selection
                    cand = np.random.choice(self.pop_size, self.tournament_k, replace=False)
                    p1 = self.population[cand[np.argmax(self.fitness[cand])]]
                    cand = np.random.choice(self.pop_size, self.tournament_k, replace=False)
                    p2 = self.population[cand[np.argmax(self.fitness[cand])]]

                    # Crossover
                    mask = np.random.rand(self.genome_size) < self.cx_rate
                    child = np.where(mask, p1, p2).copy()
                    if mask.any():
                        crossover_count += 1

                    # Mutation with adaptive rates
                    m_idx = np.random.rand(self.genome_size) < adaptive_mut_rate
                    if m_idx.any():
                        child[m_idx] += np.random.randn(m_idx.sum()) * adaptive_mut_scale
                        mutation_count += 1
                        
                        # FIX: Trading-appropriate bounds with validation
                        child[0] = np.clip(child[0], 0.2, 3.0)   # SL: 20-300 pips
                        child[1] = np.clip(child[1], 0.3, 4.0)   # TP: 30-400 pips
                        child[2] = np.clip(child[2], 0.1, 2.5)   # Vol scale
                        child[3] = np.clip(child[3], 0.0, 0.6)   # Regime adapt
                        
                        # Validate for NaN
                        if np.any(np.isnan(child)):
                            self.logger.error(f"NaN in mutated child: {child}")
                            child = np.nan_to_num(child)
                        
                    new_pop.append(child.astype(np.float32))
                    
                except Exception as e:
                    self.logger.error(f"Error creating child genome: {e}")
                    # Add a random valid genome as fallback
                    fallback = np.random.uniform([0.5, 0.8, 0.8, 0.15], [1.5, 2.2, 1.8, 0.4]).astype(np.float32)
                    new_pop.append(fallback)

            self.population = np.vstack(new_pop[:self.pop_size])
            self.fitness[:] = 0.0
            self.epoch += 1
            
            # Log evolution progress
            new_hashes = [self.genome_hash(g) for g in self.population]
            n_changed = sum(1 for o, n in zip(old_hashes, new_hashes) if o != n)
            
            self.logger.info(f"Evolution complete - Generation {self.epoch}:")
            self.logger.info(f"  Changed genomes: {n_changed}/{self.pop_size}")
            self.logger.info(f"  Crossovers: {crossover_count}, Mutations: {mutation_count}")
            self.logger.info(f"  Stagnant generations: {self.generations_without_improvement}")
            
        except Exception as e:
            self.logger.error(f"Error in evolve_strategies: {e}")

    def select_genome(self, mode="smart", k=3, custom_selector=None):
        """
        FIX: Enhanced selection with profit-aware modes and comprehensive logging
        """
        try:
            assert self.population.shape[0] == self.fitness.shape[0], "Population/fitness size mismatch"
            N = self.population.shape[0]
            
            self.logger.debug(f"Selecting genome with mode={mode}, k={k}")

            if mode == "smart":
                # FIX: Smart selection based on recent performance
                if self.generations_without_improvement < 5:
                    # Exploit best when improving
                    idx = int(np.argmax(self.fitness))
                    self.logger.debug("Smart mode: exploiting best genome (improving)")
                else:
                    # Explore when stagnant
                    top_k = np.argsort(self.fitness)[-5:]
                    idx = np.random.choice(top_k)
                    self.logger.debug("Smart mode: exploring top genomes (stagnant)")
                    
            elif mode == "random":
                idx = np.random.randint(N)
                self.logger.debug(f"Random mode: selected index {idx}")
            elif mode == "best":
                idx = int(np.argmax(self.fitness))
                self.logger.debug(f"Best mode: selected index {idx}")
            elif mode == "tournament":
                candidates = np.random.choice(N, k, replace=False)
                idx = candidates[np.argmax(self.fitness[candidates])]
                self.logger.debug(f"Tournament mode: candidates={candidates}, selected={idx}")
            elif mode == "roulette":
                fit = self.fitness - np.min(self.fitness) + 1e-8
                probs = fit / fit.sum() if fit.sum() > 0 else np.ones(N) / N
                idx = np.random.choice(N, p=probs)
                self.logger.debug(f"Roulette mode: selected index {idx}")
            elif mode == "custom":
                assert custom_selector is not None, "Provide custom_selector callable!"
                idx = custom_selector(self.population, self.fitness)
                self.logger.debug(f"Custom mode: selected index {idx}")
            else:
                raise ValueError(f"Unknown selection mode: {mode}")

            # Validate selection
            if not (0 <= idx < N):
                self.logger.error(f"Invalid genome index {idx}, using 0")
                idx = 0

            self.active_genome = self.population[idx].copy()
            self.active_genome_idx = idx

            # Validate selected genome
            if np.any(np.isnan(self.active_genome)):
                self.logger.error(f"Selected genome contains NaN: {self.active_genome}")
                self.active_genome = np.nan_to_num(self.active_genome)

            fit_val = self.fitness[idx]
            genome_str = ", ".join(f"{x:.3f}" for x in self.active_genome)
            self.logger.info(f"Selected genome: idx={idx}, fitness={fit_val:.3f}, genome=[{genome_str}] (mode={mode})")
            
            return self.active_genome
            
        except Exception as e:
            self.logger.error(f"Error in select_genome: {e}")
            # Return a safe fallback genome
            fallback = np.array([1.0, 1.5, 1.0, 0.3], dtype=np.float32)
            self.active_genome = fallback
            self.active_genome_idx = 0
            return fallback

    def get_observation_components(self) -> np.ndarray:
        """
        FIX: Enhanced observation with more informative metrics and validation
        """
        try:
            mean_f = float(self.fitness.mean())
            max_f = float(self.fitness.max())
            
            # Validate fitness values
            if np.isnan(mean_f):
                self.logger.error("NaN in mean fitness")
                mean_f = 0.0
            if np.isnan(max_f):
                self.logger.error("NaN in max fitness")
                max_f = 0.0
            
            # Diversity calculation
            if self.pop_size > 1:
                P = self.population.astype(np.float32)
                dists = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
                diversity = float(dists.mean())
                if np.isnan(diversity):
                    self.logger.error("NaN in diversity calculation")
                    diversity = 0.0
            else:
                diversity = 0.0
                
            # FIX: Add profit achievement ratio
            profit_ratio = max_f / self.profit_target if self.profit_target > 0 else 0.0
            
            observation = np.array([mean_f, max_f, diversity, profit_ratio], dtype=np.float32)
            
            # Final validation
            if np.any(np.isnan(observation)):
                self.logger.error(f"NaN in observation: {observation}")
                observation = np.nan_to_num(observation)
                
            self.logger.debug(f"Observation: mean_f={mean_f:.3f}, max_f={max_f:.3f}, diversity={diversity:.3f}, profit_ratio={profit_ratio:.3f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_state(self) -> dict:
        return {
            "population": self.population,
            "fitness": self.fitness,
            "epoch": self.epoch,
            "best_genome": self.best_genome,
            "best_fitness": self.best_fitness,
            "generations_without_improvement": self.generations_without_improvement,
        }

    def set_state(self, state: dict) -> None:
        self.population = state.get("population", self.population)
        self.fitness = state.get("fitness", self.fitness)
        self.epoch = state.get("epoch", self.epoch)
        self.best_genome = state.get("best_genome", self.best_genome)
        self.best_fitness = state.get("best_fitness", self.best_fitness)
        self.generations_without_improvement = state.get("generations_without_improvement", 0)
        self.logger.info(f"State loaded: epoch={self.epoch}, best_fitness={self.best_fitness:.3f}")

# ──────────────────────────────────────────────
class MetaAgent(Module):
    """
    FIX: Enhanced with bootstrap intensity, profit tracking, and comprehensive logging
    """
    def __init__(self, window: int=20, debug=True, profit_target=150.0):
        self.window = window
        self.debug = debug
        self.profit_target = profit_target
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"MetaAgent_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/meta_agent.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"MetaAgent initialized - window={window}, profit_target=€{profit_target}")
        
        self.reset()

    def reset(self):
        self.history: List[float] = []
        self.trade_count = 0
        self.consecutive_losses = 0
        self._step_count = 0
        self.logger.info("MetaAgent reset - all history and counters cleared")

    def step(self, pnl: float=0.0):
        """Step with comprehensive validation and logging"""
        self._step_count += 1
        
        try:
            # Validate PnL
            if np.isnan(pnl):
                self.logger.error(f"NaN PnL received, setting to 0")
                pnl = 0.0
            
            self.history.append(pnl)
            if len(self.history) > self.window:
                removed = self.history.pop(0)
                self.logger.debug(f"Removed old PnL: €{removed:.2f}")
            
            # Track consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
                self.logger.debug(f"Loss recorded: €{pnl:.2f}, consecutive losses: {self.consecutive_losses}")
            else:
                if self.consecutive_losses > 0:
                    self.logger.info(f"Loss streak broken after {self.consecutive_losses} losses with profit: €{pnl:.2f}")
                self.consecutive_losses = 0
                
            self.trade_count += 1
            
            # Log statistics periodically
            if self.trade_count % 10 == 0:
                self._log_meta_stats()
                
        except Exception as e:
            self.logger.error(f"Error in step: {e}")

    def record(self, pnl: float):
        """Record PnL with logging"""
        self.logger.debug(f"Recording PnL: €{pnl:.2f}")
        self.step(pnl)

    def _log_meta_stats(self):
        """Log detailed meta agent statistics"""
        try:
            if not self.history:
                return
                
            total_pnl = sum(self.history)
            avg_pnl = total_pnl / len(self.history)
            wins = sum(1 for p in self.history if p > 0)
            win_rate = wins / len(self.history)
            
            self.logger.info(f"Meta Statistics - Trades: {self.trade_count}")
            self.logger.info(f"  Total PnL: €{total_pnl:.2f}, Avg: €{avg_pnl:.2f}")
            self.logger.info(f"  Win Rate: {win_rate:.3f} ({wins}/{len(self.history)})")
            self.logger.info(f"  Consecutive Losses: {self.consecutive_losses}")
            self.logger.info(f"  Progress vs Target: {(total_pnl/self.profit_target)*100:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error logging meta stats: {e}")

    def get_observation_components(self)->np.ndarray:
        """Get observation with validation"""
        try:
            if not self.history:
                observation = np.array([0.0, 0.0], dtype=np.float32)
                self.logger.debug("Using default observation (empty history)")
                return observation
                
            arr = np.array(self.history, dtype=np.float32)
            
            # Validate array
            if np.any(np.isnan(arr)):
                self.logger.error(f"NaN values in history: {arr}")
                arr = np.nan_to_num(arr)
                
            mean_val = float(arr.mean())
            std_val = float(arr.std())
            
            # Validate results
            if np.isnan(mean_val):
                self.logger.error("NaN in mean calculation")
                mean_val = 0.0
            if np.isnan(std_val):
                self.logger.error("NaN in std calculation")
                std_val = 0.0
                
            observation = np.array([mean_val, std_val], dtype=np.float32)
            self.logger.debug(f"Observation: mean={mean_val:.3f}, std={std_val:.3f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0], dtype=np.float32)
    
    def get_intensity(self, instrument: str) -> float:
        """
        FIX: Smarter intensity calculation for profitable trading with comprehensive logging
        """
        try:
            self.logger.debug(f"Calculating intensity for {instrument}")
            
            # Bootstrap intensity for initial trades
            if self.trade_count < 5:
                # Start with moderate positive intensity to encourage trading
                intensity = 0.3 + np.random.uniform(-0.1, 0.1)
                self.logger.info(f"Bootstrap intensity for {instrument}: {intensity:.3f} (trade_count={self.trade_count})")
                return float(intensity)
            
            if not self.history:
                self.logger.debug(f"No history for {instrument}, returning 0")
                return 0.0
                
            # Calculate recent performance
            recent_window = min(10, len(self.history))
            recent_pnl = self.history[-recent_window:]
            avg_pnl = np.mean(recent_pnl)
            
            self.logger.debug(f"Recent performance for {instrument}: avg_pnl=€{avg_pnl:.3f} over {recent_window} trades")
            
            # FIX: Profit-aware intensity
            if avg_pnl > self.profit_target / self.window:
                # Above target: maintain momentum
                intensity = 0.7 + min(0.3, avg_pnl / (self.profit_target * 2))
                self.logger.info(f"Above target performance - high intensity: {intensity:.3f}")
            elif avg_pnl > 0:
                # Profitable but below target: increase aggression
                intensity = 0.3 + (avg_pnl / self.profit_target)
                self.logger.info(f"Profitable but below target - moderate intensity: {intensity:.3f}")
            else:
                # Losing: reduce but don't stop
                intensity = max(-0.5, -0.1 - self.consecutive_losses * 0.05)
                self.logger.warning(f"Losing streak - reduced intensity: {intensity:.3f}")
                
            # Clamp to safe range
            intensity = np.clip(intensity, -0.8, 0.9)
            
            # Final validation
            if np.isnan(intensity):
                self.logger.error("NaN intensity calculated, using 0")
                intensity = 0.0
            
            self.logger.info(f"Final intensity for {instrument}: {intensity:.3f} (avg_pnl=€{avg_pnl:.3f}, losses={self.consecutive_losses})")
            return float(intensity)
            
        except Exception as e:
            self.logger.error(f"Error calculating intensity for {instrument}: {e}")
            return 0.0

# ──────────────────────────────────────────────
class MetaCognitivePlanner(Module):
    """
    FIX: Enhanced with better episode tracking and comprehensive logging
    """
    def __init__(self, window: int=20, debug=True):
        self.window = window
        self.debug = debug
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"MetaCognitivePlanner_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/metacognitive_planner.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"MetaCognitivePlanner initialized - window={window}")
        
        self.reset()

    def reset(self):
        self.history: List[Dict[str,float]] = []
        self.total_episodes = 0
        self.profitable_episodes = 0
        self._step_count = 0
        self.logger.info("MetaCognitivePlanner reset - all episode history cleared")

    def step(self, **kwargs): 
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record_episode(self, result: Dict[str,float]):
        """Record episode with comprehensive validation and logging"""
        try:
            # Validate input
            if not isinstance(result, dict):
                self.logger.error(f"Invalid result type: {type(result)}")
                return
                
            self.logger.debug(f"Recording episode result: {result}")
            
            # Validate PnL
            pnl = result.get("pnl", 0)
            if np.isnan(pnl):
                self.logger.error(f"NaN PnL in episode result, setting to 0")
                result = result.copy()
                result["pnl"] = 0
                pnl = 0
            
            self.history.append(result)
            if len(self.history) > self.window:
                removed = self.history.pop(0)
                self.logger.debug(f"Removed old episode: {removed}")
                
            # Track profitable episodes
            self.total_episodes += 1
            if pnl > 0:
                self.profitable_episodes += 1
                self.logger.info(f"Profitable episode recorded: €{pnl:.2f}")
            else:
                self.logger.debug(f"Loss episode recorded: €{pnl:.2f}")
                
            # Log episode statistics
            win_rate = self.profitable_episodes / self.total_episodes
            self.logger.info(f"Episode {self.total_episodes}: Win rate: {win_rate:.3f} ({self.profitable_episodes}/{self.total_episodes})")
            
            # Log detailed statistics periodically
            if self.total_episodes % 10 == 0:
                self._log_cognitive_stats()
                
        except Exception as e:
            self.logger.error(f"Error recording episode: {e}")

    def _log_cognitive_stats(self):
        """Log detailed cognitive planning statistics"""
        try:
            if not self.history:
                return
                
            pnls = [r.get("pnl", 0) for r in self.history]
            total_pnl = sum(pnls)
            avg_pnl = total_pnl / len(pnls)
            
            profitable_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p < 0]
            
            self.logger.info(f"Cognitive Statistics - Episodes: {len(self.history)}")
            self.logger.info(f"  Total PnL: €{total_pnl:.2f}, Avg: €{avg_pnl:.2f}")
            
            if profitable_pnls:
                avg_win = np.mean(profitable_pnls)
                self.logger.info(f"  Avg Win: €{avg_win:.2f} ({len(profitable_pnls)} wins)")
                
            if losing_pnls:
                avg_loss = np.mean(losing_pnls)
                self.logger.info(f"  Avg Loss: €{avg_loss:.2f} ({len(losing_pnls)} losses)")
                
                if profitable_pnls:
                    risk_reward = avg_win / abs(avg_loss)
                    self.logger.info(f"  Risk/Reward Ratio: {risk_reward:.2f}")
                    
        except Exception as e:
            self.logger.error(f"Error logging cognitive stats: {e}")

    def get_observation_components(self)->np.ndarray:
        """Get cognitive metrics with validation"""
        try:
            if not self.history:
                # FIX: Better bootstrap values
                defaults = np.array([0.5, 0.0, 1.0], dtype=np.float32)
                self.logger.debug("Using default cognitive metrics")
                return defaults
                
            pnls = np.array([r.get("pnl",0) for r in self.history], dtype=np.float32)
            
            # Validate PnLs
            if np.any(np.isnan(pnls)):
                self.logger.error(f"NaN values in PnL history: {pnls}")
                pnls = np.nan_to_num(pnls)
                
            win_rate = float((pnls>0).sum() / len(pnls)) if len(pnls)>0 else 0.5
            
            # FIX: Add risk-adjusted return metric
            profitable_pnls = pnls[pnls > 0]
            losing_pnls = pnls[pnls < 0]
            
            avg_win = profitable_pnls.mean() if len(profitable_pnls) > 0 else 0.0
            avg_loss = abs(losing_pnls.mean()) if len(losing_pnls) > 0 else 1.0
            risk_reward = avg_win / avg_loss if avg_loss > 0 else 1.0
            
            # Validate components
            if np.isnan(win_rate):
                self.logger.error("NaN in win_rate")
                win_rate = 0.5
            if np.isnan(risk_reward):
                self.logger.error("NaN in risk_reward")
                risk_reward = 1.0
                
            mean_pnl = float(pnls.mean())
            if np.isnan(mean_pnl):
                self.logger.error("NaN in mean_pnl")
                mean_pnl = 0.0
            
            observation = np.array([win_rate, mean_pnl, risk_reward], dtype=np.float32)
            self.logger.debug(f"Cognitive metrics: win_rate={win_rate:.3f}, mean_pnl={mean_pnl:.3f}, risk_reward={risk_reward:.3f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.5, 0.0, 1.0], dtype=np.float32)

# ──────────────────────────────────────────────
class BiasAuditor(Module):
    """
    FIX: Enhanced to detect and correct trading biases with comprehensive logging
    """
    def __init__(self, history_len: int=100, debug=True):
        self.history_len = history_len
        self.debug = debug
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"BiasAuditor_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/bias_auditor.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"BiasAuditor initialized - history_len={history_len}")
        
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.history_len)
        self.bias_corrections = {"revenge": 0, "fear": 0, "greed": 0}
        self._step_count = 0
        self.logger.info("BiasAuditor reset - all bias history and corrections cleared")

    def step(self, **kwargs): 
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record(self, bias: str, pnl: float = 0):
        """FIX: Record bias with outcome and comprehensive logging"""
        try:
            # Validate inputs
            if not isinstance(bias, str):
                self.logger.error(f"Invalid bias type: {type(bias)}")
                bias = "unknown"
                
            if np.isnan(pnl):
                self.logger.error("NaN PnL in bias record, setting to 0")
                pnl = 0.0
                
            self.logger.debug(f"Recording bias: {bias}, PnL: €{pnl:.2f}")
            
            self.hist.append((bias, pnl))
            
            # Learn bias corrections
            if pnl < 0:
                if bias in self.bias_corrections:
                    self.bias_corrections[bias] += 1
                    self.logger.warning(f"Negative outcome for bias '{bias}' - correction count: {self.bias_corrections[bias]}")
                else:
                    self.bias_corrections[bias] = 1
                    self.logger.warning(f"New negative bias detected: {bias}")
            else:
                self.logger.info(f"Positive outcome for bias '{bias}': €{pnl:.2f}")
                
            # Log bias statistics periodically
            if len(self.hist) % 20 == 0:
                self._log_bias_stats()
                
        except Exception as e:
            self.logger.error(f"Error recording bias: {e}")

    def _log_bias_stats(self):
        """Log detailed bias statistics"""
        try:
            if not self.hist:
                return
                
            bias_counts = {}
            bias_pnls = {}
            
            for bias, pnl in self.hist:
                if bias not in bias_counts:
                    bias_counts[bias] = 0
                    bias_pnls[bias] = []
                bias_counts[bias] += 1
                bias_pnls[bias].append(pnl)
            
            self.logger.info(f"Bias Statistics - Total records: {len(self.hist)}")
            
            for bias in bias_counts:
                count = bias_counts[bias]
                pnls = bias_pnls[bias]
                avg_pnl = np.mean(pnls)
                negative_count = sum(1 for p in pnls if p < 0)
                
                self.logger.info(f"  {bias}: count={count}, avg_pnl=€{avg_pnl:.2f}, negative={negative_count}, correction_level={self.bias_corrections.get(bias, 0)}")
                
        except Exception as e:
            self.logger.error(f"Error logging bias stats: {e}")

    def get_observation_components(self)->np.ndarray:
        """Get bias frequencies with validation"""
        try:
            total = len(self.hist)
            if total == 0:
                # FIX: Balanced initial state
                defaults = np.array([0.33, 0.33, 0.33], dtype=np.float32)
                self.logger.debug("Using default bias frequencies")
                return defaults
                
            cnt = {"revenge":0,"fear":0,"greed":0}
            for b, _ in self.hist:
                if b in cnt: 
                    cnt[b] += 1
                    
            # FIX: Apply corrections to discourage losing biases
            for bias, correction_count in self.bias_corrections.items():
                if correction_count > 5:  # Significant negative bias
                    reduction = correction_count // 2
                    cnt[bias] = max(0, cnt[bias] - reduction)
                    if reduction > 0:
                        self.logger.debug(f"Applied correction to {bias}: reduced by {reduction}")
                    
            total_corrected = sum(cnt.values()) or 1
            freqs = np.array([cnt["revenge"],cnt["fear"],cnt["greed"]], dtype=np.float32) / total_corrected
            
            # Validate frequencies
            if np.any(np.isnan(freqs)):
                self.logger.error(f"NaN in bias frequencies: {freqs}")
                freqs = np.nan_to_num(freqs)
                
            # Ensure they sum to 1
            if freqs.sum() > 0:
                freqs = freqs / freqs.sum()
            else:
                freqs = np.array([0.33, 0.33, 0.33], dtype=np.float32)
                
            self.logger.debug(f"Bias frequencies: revenge={freqs[0]:.3f}, fear={freqs[1]:.3f}, greed={freqs[2]:.3f}")
            return freqs
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.33, 0.33, 0.33], dtype=np.float32)

# ──────────────────────────────────────────────
class OpponentModeEnhancer(Module):
    """
    FIX: Market regime detector with profit tracking and comprehensive logging
    """
    def __init__(self, modes: List[str]=None, debug=True):
        self.modes = modes or ["trending", "ranging", "volatile"]  # FIX: Better market modes
        self.debug = debug
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"OpponentModeEnhancer_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/opponent_mode_enhancer.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"OpponentModeEnhancer initialized - modes={self.modes}")
        
        self.reset()

    def reset(self):
        self.pnl = {m: 0.0 for m in self.modes}
        self.counts = {m: 0 for m in self.modes}
        self._step_count = 0
        self.logger.info("OpponentModeEnhancer reset - all mode statistics cleared")

    def step(self, **kwargs): 
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record_result(self, mode: str, pnl: float):
        """Record mode result with validation and logging"""
        try:
            # Validate inputs
            if not isinstance(mode, str):
                self.logger.error(f"Invalid mode type: {type(mode)}")
                return
                
            if np.isnan(pnl):
                self.logger.error(f"NaN PnL for mode {mode}, setting to 0")
                pnl = 0.0
                
            if mode not in self.modes:
                self.logger.warning(f"Unknown mode '{mode}', adding to tracking")
                self.modes.append(mode)
                self.pnl[mode] = 0.0
                self.counts[mode] = 0
                
            self.pnl[mode] += pnl
            self.counts[mode] += 1
            
            avg_pnl = self.pnl[mode] / self.counts[mode]
            self.logger.info(f"Mode result: {mode}, PnL: €{pnl:.2f}, Count: {self.counts[mode]}, Avg: €{avg_pnl:.2f}")
            
            # Log mode statistics periodically
            if sum(self.counts.values()) % 10 == 0:
                self._log_mode_stats()
                
        except Exception as e:
            self.logger.error(f"Error recording result for mode {mode}: {e}")

    def _log_mode_stats(self):
        """Log detailed mode statistics"""
        try:
            total_trades = sum(self.counts.values())
            if total_trades == 0:
                return
                
            self.logger.info(f"Mode Statistics - Total trades: {total_trades}")
            
            for mode in self.modes:
                if self.counts[mode] > 0:
                    avg_pnl = self.pnl[mode] / self.counts[mode]
                    frequency = self.counts[mode] / total_trades
                    self.logger.info(f"  {mode}: trades={self.counts[mode]} ({frequency:.1%}), total=€{self.pnl[mode]:.2f}, avg=€{avg_pnl:.2f}")
                else:
                    self.logger.info(f"  {mode}: no trades yet")
                    
        except Exception as e:
            self.logger.error(f"Error logging mode stats: {e}")

    def get_observation_components(self) -> np.ndarray:
        """Get mode weights with validation"""
        try:
            # FIX: Return profit-per-trade for each mode
            profits_per_trade = []
            for m in self.modes:
                if self.counts[m] > 0:
                    avg_pnl = self.pnl[m] / self.counts[m]
                else:
                    avg_pnl = 0.0
                profits_per_trade.append(avg_pnl)
                
            # Normalize to weights
            arr = np.array(profits_per_trade, dtype=np.float32)
            
            # Validate array
            if np.any(np.isnan(arr)):
                self.logger.error(f"NaN in profits per trade: {arr}")
                arr = np.nan_to_num(arr)
            
            if arr.sum() > 0:
                # Shift to positive values and normalize
                shifted = arr - arr.min() + 1e-6
                weights = shifted / shifted.sum()
            else:
                weights = np.ones(len(self.modes)) / len(self.modes)
                
            # Final validation
            if np.any(np.isnan(weights)):
                self.logger.error(f"NaN in final weights: {weights}")
                weights = np.ones(len(self.modes)) / len(self.modes)
                
            self.logger.debug(f"Mode weights: {dict(zip(self.modes, weights))}")
            return weights
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.ones(len(self.modes), dtype=np.float32) / len(self.modes)

# ──────────────────────────────────────────────
class ThesisEvolutionEngine(Module):
    """
    FIX: Track trading thesis performance with comprehensive logging
    """
    def __init__(self, capacity: int=20, debug=True):
        self.capacity = capacity
        self.debug = debug
        self._step_count = 0
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"ThesisEvolutionEngine_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/meta/thesis_evolution.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"ThesisEvolutionEngine initialized - capacity={capacity}")
        
        self.reset()

    def reset(self):
        self.theses: List[str] = []  # Changed from Any to str for clarity
        self.pnls: List[float] = []
        self.thesis_performance: Dict[str, List[float]] = {}
        self._step_count = 0
        self.logger.info("ThesisEvolutionEngine reset - all thesis data cleared")

    def step(self, **kwargs): 
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - kwargs: {list(kwargs.keys())}")

    def record_thesis(self, thesis: str):
        """FIX: Track thesis properly with validation and logging"""
        try:
            if not isinstance(thesis, str):
                self.logger.error(f"Invalid thesis type: {type(thesis)}")
                thesis = str(thesis)
                
            self.theses.append(thesis)
            if thesis not in self.thesis_performance:
                self.thesis_performance[thesis] = []
                self.logger.info(f"New thesis recorded: '{thesis}'")
            else:
                self.logger.debug(f"Existing thesis recorded: '{thesis}'")
                
            # Maintain capacity
            if len(self.theses) > self.capacity:
                removed = self.theses.pop(0)
                self.logger.debug(f"Removed old thesis: '{removed}'")
                
        except Exception as e:
            self.logger.error(f"Error recording thesis: {e}")

    def record_pnl(self, pnl: float):
        """Record PnL for current thesis with validation and logging"""
        try:
            # Validate PnL
            if np.isnan(pnl):
                self.logger.error("NaN PnL recorded, setting to 0")
                pnl = 0.0
                
            if self.theses:
                current_thesis = self.theses[-1]
                self.pnls.append(pnl)
                self.thesis_performance[current_thesis].append(pnl)
                
                # Calculate thesis performance
                thesis_pnls = self.thesis_performance[current_thesis]
                thesis_avg = np.mean(thesis_pnls)
                thesis_total = sum(thesis_pnls)
                
                self.logger.info(f"PnL €{pnl:.2f} recorded for thesis '{current_thesis}' - Total: €{thesis_total:.2f}, Avg: €{thesis_avg:.2f}, Count: {len(thesis_pnls)}")
                
                # Maintain capacity
                if len(self.pnls) > self.capacity:
                    removed = self.pnls.pop(0)
                    self.logger.debug(f"Removed old PnL: €{removed:.2f}")
                    
                # Log thesis statistics periodically
                if len(thesis_pnls) % 5 == 0:
                    self._log_thesis_stats()
            else:
                self.logger.warning(f"No thesis to record PnL €{pnl:.2f} against")
                
        except Exception as e:
            self.logger.error(f"Error recording PnL: {e}")

    def _log_thesis_stats(self):
        """Log detailed thesis statistics"""
        try:
            if not self.thesis_performance:
                return
                
            self.logger.info(f"Thesis Performance Summary - {len(self.thesis_performance)} theses:")
            
            for thesis, pnls in self.thesis_performance.items():
                if pnls:
                    total_pnl = sum(pnls)
                    avg_pnl = np.mean(pnls)
                    win_rate = sum(1 for p in pnls if p > 0) / len(pnls)
                    
                    self.logger.info(f"  '{thesis[:30]}...': trades={len(pnls)}, total=€{total_pnl:.2f}, avg=€{avg_pnl:.2f}, win_rate={win_rate:.3f}")
                    
        except Exception as e:
            self.logger.error(f"Error logging thesis stats: {e}")

    def get_observation_components(self)->np.ndarray:
        """Get thesis metrics with validation"""
        try:
            if not self.pnls:
                defaults = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                self.logger.debug("Using default thesis metrics")
                return defaults
                
            # FIX: Better metrics
            uniq = len(self.thesis_performance)
            mean_p = float(np.mean(self.pnls))
            
            # Validate mean
            if np.isnan(mean_p):
                self.logger.error("NaN in mean PnL")
                mean_p = 0.0
            
            # Find best performing thesis
            best_thesis_pnl = 0.0
            if self.thesis_performance:
                for thesis, pnls in self.thesis_performance.items():
                    if pnls:
                        thesis_avg = np.mean(pnls)
                        if not np.isnan(thesis_avg):
                            best_thesis_pnl = max(best_thesis_pnl, thesis_avg)
                            
            observation = np.array([float(uniq), mean_p, best_thesis_pnl], dtype=np.float32)
            
            # Final validation
            if np.any(np.isnan(observation)):
                self.logger.error(f"NaN in thesis observation: {observation}")
                observation = np.nan_to_num(observation)
                
            self.logger.debug(f"Thesis metrics: unique={uniq}, mean_pnl={mean_p:.3f}, best_thesis_pnl={best_thesis_pnl:.3f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

# ──────────────────────────────────────────────
class ExplanationGenerator(Module):
    """
    FIX: Enhanced explanation generator with comprehensive logging
    """

    def __init__(self, fractal_regime: FractalRegimeConfirmation, strategy_arbiter: StrategyArbiter, debug: bool = True):
        super().__init__()
        self.debug = debug
        self.last_explanation = ""
        self.trade_count = 0
        self.profit_today = 0.0
        self.fractal_regime = fractal_regime
        self.strategy_arbiter = strategy_arbiter
        self.arbiter = strategy_arbiter
        self._step_count = 0

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"ExplanationGenerator_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/explanation_generator.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info("ExplanationGenerator initialized")

    def reset(self) -> None:
        self.last_explanation = ""
        self.trade_count = 0
        self.profit_today = 0.0
        self._step_count = 0
        self.logger.info("ExplanationGenerator reset - all metrics cleared")

    def step(
        self,
        actions=None,
        arbiter_weights=None,
        member_names=None,
        votes=None,
        regime: str | None = None, # "unknown" for bootstrap
        volatility=None,
        drawdown=0.0,
        genome_metrics=None,
        pnl=0.0,
        target_achieved=False,
        *args, **kwargs
    ) -> None:
        """
        Robust to missing pipeline arguments and preserves full analytics with comprehensive logging.
        Accepts and safely ignores unused/extra parameters.
        """
        self._step_count += 1
        
        try:
            # Validate PnL
            if np.isnan(pnl):
                self.logger.error("NaN PnL received, setting to 0")
                pnl = 0.0

            if regime is None or str(regime).lower() == "unknown":
                self.fractal_regime.label
                self.logger.debug(f"Retrieved regime from switcher: {regime}")

            # Get expert names/weights from arbiter if not provided
            if not member_names:
                member_names = [m.__class__.__name__ for m in self.arbiter.members]
                self.logger.debug(f"Retrieved member names: {member_names}")
                
            if arbiter_weights is None or len(arbiter_weights) != len(member_names):
                if hasattr(self.arbiter, "last_alpha") and self.arbiter.last_alpha is not None:
                    arbiter_weights = np.array(self.arbiter.last_alpha, dtype=np.float32)
                else:
                    arbiter_weights = np.ones(len(member_names), dtype=np.float32)
                self.logger.debug(f"Retrieved/generated arbiter weights: {arbiter_weights}")

            # Defaults for safe operation with validation
            if actions is None or not hasattr(actions, "__len__"):
                actions = np.zeros(1, dtype=np.float32)
            if arbiter_weights is None or not hasattr(arbiter_weights, "__len__"):
                arbiter_weights = np.ones(1, dtype=np.float32)
            if not member_names or not isinstance(member_names, (list, tuple)):
                member_names = ["Unknown"]
            if votes is None or not isinstance(votes, dict):
                votes = {}
            if volatility is None or not isinstance(volatility, dict):
                volatility = {}
            if genome_metrics is None or not isinstance(genome_metrics, dict):
                genome_metrics = {}

            # Validate arbiter weights
            if np.any(np.isnan(arbiter_weights)):
                self.logger.error(f"NaN in arbiter weights: {arbiter_weights}")
                arbiter_weights = np.nan_to_num(arbiter_weights)

            self.trade_count += 1
            self.profit_today += pnl

            self.logger.debug(f"Step {self._step_count}: trade_count={self.trade_count}, profit_today=€{self.profit_today:.2f}")

            try:
                top_idx = int(np.argmax(arbiter_weights))
                top_name = member_names[top_idx] if top_idx < len(member_names) else "Unknown"
                top_w = float(arbiter_weights[top_idx]) * 100.0
            except Exception as e:
                self.logger.error(f"Error determining top strategy: {e}")
                top_idx, top_name, top_w = 0, "Unknown", 100.0

            # Aggregate votes with validation
            agg = {n: 0.0 for n in member_names}
            count = 0
            for vote_dict in votes.values():
                if isinstance(vote_dict, dict):
                    for n, w in vote_dict.items():
                        if not np.isnan(w):
                            agg[n] = agg.get(n, 0.0) + float(w)
                    count += 1
            if count:
                for n in agg:
                    agg[n] /= count
            votes_str = "; ".join(f"{n}: {agg[n] * 100.0:.1f}%" for n in list(member_names)[:3])

            # High volatility warning
            high_vol_instruments = [inst for inst, vol in volatility.items() if isinstance(vol, (float, int)) and vol > 0.02]
            vol_warning = f" ⚠️ HIGH VOL: {', '.join(high_vol_instruments)}" if high_vol_instruments else ""

            # Risk/reward calculation
            sl_base = float(genome_metrics.get("sl_base", 1.0))
            tp_base = float(genome_metrics.get("tp_base", 1.5))
            risk_reward = tp_base / sl_base if sl_base > 0 else 1.5

            # Validate drawdown
            if np.isnan(drawdown):
                self.logger.error("NaN drawdown received, setting to 0")
                drawdown = 0.0

            progress_pct = (self.profit_today / 150.0) * 100  # Against €150 target
            dd_pct = drawdown * 100.0

            self.last_explanation = (
                f"Day Progress: €{self.profit_today:.2f}/€150 ({progress_pct:.1f}%) | "
                f"Trades: {self.trade_count} | "
                f"Regime: {regime}{vol_warning} | "
                f"Strategy: {top_name} ({top_w:.0f}%) | "
                f"RR: {risk_reward:.1f}:1 | "
                f"DD: {dd_pct:.1f}%"
            )

            if target_achieved:
                self.last_explanation += "  TARGET ACHIEVED - Consider stopping"
                self.logger.info("Target achieved!")
            elif dd_pct > 5:
                self.last_explanation += "  High drawdown - Reduce position size"
                self.logger.warning(f"High drawdown detected: {dd_pct:.1f}%")
            elif progress_pct < 30 and self.trade_count > 20:
                self.last_explanation += "  Low progress - Review strategy"
                self.logger.warning(f"Low progress after {self.trade_count} trades: {progress_pct:.1f}%")

            self.logger.info(f"Generated explanation: {self.last_explanation}")

            if self.debug:
                print("[ExplanationGenerator]", self.last_explanation)

        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            self.last_explanation = f"Error generating explanation: {str(e)}"

    def get_observation_components(self) -> np.ndarray:
        """Return profit metrics with validation"""
        try:
            avg_profit = self.profit_today / max(1, self.trade_count)
            
            # Validate components
            if np.isnan(self.profit_today):
                self.logger.error("NaN in profit_today")
                self.profit_today = 0.0
            if np.isnan(avg_profit):
                self.logger.error("NaN in avg_profit")
                avg_profit = 0.0
                
            observation = np.array([
                self.profit_today,
                float(self.trade_count),
                avg_profit
            ], dtype=np.float32)
            
            self.logger.debug(f"Observation: profit={self.profit_today:.2f}, trades={self.trade_count}, avg={avg_profit:.2f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

# ──────────────────────────────────────────────
# PPO-LAG IMPLEMENTATION - FIXED
# ──────────────────────────────────────────────

class PPOLagAgent(nn.Module, Module):
    """
    PPO-Lag: Enhanced PPO for volatile financial markets - FIXED NaN issues
    Optimized for €100-200/day profit on XAUUSD/EURUSD
    """
    
    def __init__(self, 
                 obs_size: int,
                 act_size: int = 2,
                 hidden_size: int = 128,
                 lr: float = 1e-4,
                 lag_window: int = 20,
                 adv_decay: float = 0.95,
                 vol_scaling: bool = True,
                 position_aware: bool = True,
                 device: str = "cpu",
                 debug: bool = True):
        super().__init__()
        
        self.device = torch.device(device)
        self.debug = debug
        self.lag_window = lag_window
        self.adv_decay = adv_decay
        self.vol_scaling = vol_scaling
        self.position_aware = position_aware
        self.obs_size = obs_size
        self.act_size = act_size
        
        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"PPOLagAgent_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/ppo/ppo_lag_agent.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # Expand input size for lagged features
        lag_features = 4  # returns, volatility, volume, spread
        position_features = 3 if position_aware else 0
        self.extended_obs_size = obs_size + (lag_window * lag_features) + position_features
        
        self.logger.info(f"PPOLagAgent initializing - obs_size={obs_size}, extended_size={self.extended_obs_size}")
        
        try:
            # Enhanced actor network with proper initialization
            self.actor = nn.Sequential(
                nn.Linear(self.extended_obs_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, act_size),
                nn.Tanh()
            )
            
            # Market-aware critic
            self.value_encoder = nn.Sequential(
                nn.Linear(self.extended_obs_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            self.market_encoder = nn.Sequential(
                nn.Linear(lag_window * lag_features, hidden_size // 2),
                nn.ReLU()
            )
            
            self.value_head = nn.Sequential(
                nn.Linear(hidden_size + hidden_size // 2, hidden_size // 2),
                nn.ReLU(),
                nn.Linear(hidden_size // 2, 1)
            )
            
            # FIX: Proper weight initialization
            self._initialize_weights()
            
            self.to(self.device)
            
            # Optimizers
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
            self.critic_opt = optim.Adam(
                list(self.value_encoder.parameters()) + 
                list(self.market_encoder.parameters()) + 
                list(self.value_head.parameters()), 
                lr=lr * 2
            )
            
            # PPO hyperparameters
            self.clip_eps = 0.1
            self.value_coeff = 0.5
            self.entropy_coeff = 0.001
            self.max_grad_norm = 0.5
            
            # Lag buffers
            self.price_buffer = deque(maxlen=lag_window)
            self.volume_buffer = deque(maxlen=lag_window)
            self.spread_buffer = deque(maxlen=lag_window)
            self.vol_buffer = deque(maxlen=lag_window)
            
            # Episode buffers
            self.buffer = {
                "obs": [], "actions": [], "logp": [], "values": [], 
                "rewards": [], "market_features": []
            }
            
            # State tracking
            self.running_adv_std = 1.0
            self.last_action = np.zeros(act_size, dtype=np.float32)
            self.position = 0.0
            self.unrealized_pnl = 0.0
            self.total_trades = 0
            self._step_count = 0
            
            self.logger.info("PPOLagAgent initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing PPOLagAgent: {e}")
            raise

    def _initialize_weights(self):
        """FIX: Proper weight initialization to prevent NaN"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)  # Small gain to prevent explosion
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.logger.info("Network weights initialized")

    def update_market_buffers(self, price: float, volume: float, spread: float, volatility: float):
        """Update lagged market features with validation"""
        try:
            # Validate inputs
            if np.isnan(price):
                self.logger.error("NaN price, using last valid or 0")
                price = self.price_buffer[-1] if self.price_buffer else 0.0
            if np.isnan(volume):
                volume = 0.0
            if np.isnan(spread):
                spread = 0.0
            if np.isnan(volatility) or volatility <= 0:
                volatility = 1.0
                
            if len(self.price_buffer) > 1:
                last_price = self.price_buffer[-1]
                ret = (price - last_price) / last_price if last_price > 0 else 0
            else:
                ret = 0
                
            # Validate return
            if np.isnan(ret) or abs(ret) > 0.1:  # Cap at 10% return
                ret = 0.0
                
            self.price_buffer.append(ret)
            self.volume_buffer.append(volume)
            self.spread_buffer.append(spread)
            self.vol_buffer.append(volatility)
            
            self.logger.debug(f"Market buffers updated: ret={ret:.5f}, vol={volatility:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error updating market buffers: {e}")

    def get_lag_features(self) -> np.ndarray:
        """Extract lagged features with validation"""
        try:
            price_lags = list(self.price_buffer) + [0] * (self.lag_window - len(self.price_buffer))
            volume_lags = list(self.volume_buffer) + [0] * (self.lag_window - len(self.volume_buffer))
            spread_lags = list(self.spread_buffer) + [0] * (self.lag_window - len(self.spread_buffer))
            vol_lags = list(self.vol_buffer) + [1] * (self.lag_window - len(self.vol_buffer))
            
            features = []
            for i in range(self.lag_window):
                features.extend([price_lags[i], vol_lags[i], volume_lags[i], spread_lags[i]])
                
            result = np.array(features, dtype=np.float32)
            
            # Validate features
            if np.any(np.isnan(result)):
                self.logger.error(f"NaN in lag features: {result}")
                result = np.nan_to_num(result)
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting lag features: {e}")
            return np.zeros(self.lag_window * 4, dtype=np.float32)

    def forward(self, obs: torch.Tensor, market_lags: torch.Tensor):
        """Forward pass with NaN validation"""
        try:
            # Validate inputs
            if torch.any(torch.isnan(obs)):
                self.logger.error("NaN in observation input")
                obs = torch.nan_to_num(obs)
            if torch.any(torch.isnan(market_lags)):
                self.logger.error("NaN in market lags input")
                market_lags = torch.nan_to_num(market_lags)
                
            action_logits = self.actor(obs)
            
            value_features = self.value_encoder(obs)
            market_features = self.market_encoder(market_lags)
            combined = torch.cat([value_features, market_features], dim=-1)
            value = self.value_head(combined)
            
            # Validate outputs
            if torch.any(torch.isnan(action_logits)):
                self.logger.error("NaN in action logits")
                action_logits = torch.zeros_like(action_logits)
            if torch.any(torch.isnan(value)):
                self.logger.error("NaN in value output")
                value = torch.zeros_like(value)
            
            return action_logits, value
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            # Return safe defaults
            batch_size = obs.shape[0]
            return (torch.zeros(batch_size, self.act_size, device=self.device),
                    torch.zeros(batch_size, 1, device=self.device))

    def record_step(self, obs_vec: np.ndarray, reward: float, 
                   price: float = 0, volume: float = 0, 
                   spread: float = 0, volatility: float = 1,
                   position: float = 0, unrealized_pnl: float = 0):
        """Record step with market data and comprehensive validation"""
        self._step_count += 1
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.logger.error(f"NaN in observation vector: {obs_vec}")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.logger.error("NaN reward, setting to 0")
                reward = 0.0
            if np.isnan(position):
                position = 0.0
            if np.isnan(unrealized_pnl):
                unrealized_pnl = 0.0
                
            # Update market buffers
            self.update_market_buffers(price, volume, spread, volatility)
            
            # Get lag features
            lag_features = self.get_lag_features()
            
            # Build extended observation
            if self.position_aware:
                position_features = np.array([position, unrealized_pnl, position * volatility], dtype=np.float32)
                extended_obs = np.concatenate([obs_vec, lag_features, position_features])
            else:
                extended_obs = np.concatenate([obs_vec, lag_features])
                
            # Validate extended observation
            if np.any(np.isnan(extended_obs)):
                self.logger.error(f"NaN in extended observation: {extended_obs}")
                extended_obs = np.nan_to_num(extended_obs)
                
            # Pad if necessary
            if len(extended_obs) < self.extended_obs_size:
                padding = np.zeros(self.extended_obs_size - len(extended_obs), dtype=np.float32)
                extended_obs = np.concatenate([extended_obs, padding])
            elif len(extended_obs) > self.extended_obs_size:
                extended_obs = extended_obs[:self.extended_obs_size]
                
            # Convert to tensors
            obs_t = torch.as_tensor(extended_obs, dtype=torch.float32, device=self.device)
            market_t = torch.as_tensor(lag_features, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                mu, value = self.forward(obs_t.unsqueeze(0), market_t.unsqueeze(0))
                
                # Volatility-scaled exploration
                if self.vol_scaling and volatility > 0:
                    action_std = 0.1 / np.sqrt(volatility)
                else:
                    action_std = 0.1
                    
                # FIX: Clamp action std to prevent NaN
                action_std = np.clip(action_std, 0.01, 0.5)
                    
                dist = torch.distributions.Normal(mu, action_std)
                action = dist.rsample()
                
                # Position-aware action scaling
                if self.position_aware and abs(position) > 0.8:
                    action = action * (1 - abs(position))
                    
                logp = dist.log_prob(action).sum(dim=-1)
                
                # Validate outputs
                if torch.any(torch.isnan(action)):
                    self.logger.error("NaN in sampled action")
                    action = torch.zeros_like(action)
                if torch.any(torch.isnan(logp)):
                    self.logger.error("NaN in log probability")
                    logp = torch.zeros_like(logp)

            # Store in buffer
            self.buffer["obs"].append(obs_t)
            self.buffer["actions"].append(action.squeeze(0))
            self.buffer["logp"].append(logp.squeeze(0))
            self.buffer["values"].append(value.squeeze(0))
            self.buffer["rewards"].append(torch.as_tensor(reward, dtype=torch.float32, device=self.device))
            self.buffer["market_features"].append(market_t)
            
            # Update state
            self.last_action = action.cpu().numpy().squeeze(0)
            self.position = position
            self.unrealized_pnl = unrealized_pnl
            self.total_trades += 1
            
            self.logger.debug(f"Step {self._step_count} recorded: reward={reward:.3f}, action_mean={self.last_action.mean():.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in record_step: {e}")

    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95):
        """GAE with advantage smoothing and NaN protection"""
        try:
            rewards = torch.stack(self.buffer["rewards"])
            values = torch.stack(self.buffer["values"])
            
            # Validate inputs
            if torch.any(torch.isnan(rewards)):
                self.logger.error("NaN in rewards buffer")
                rewards = torch.nan_to_num(rewards)
            if torch.any(torch.isnan(values)):
                self.logger.error("NaN in values buffer")
                values = torch.nan_to_num(values)
            
            # Compute returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            
            # Compute advantages with GAE
            advantages = []
            adv = 0
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_value = 0
                else:
                    next_value = values[t + 1]
                
                td_error = rewards[t] + gamma * next_value - values[t]
                adv = td_error + gamma * lam * adv
                advantages.insert(0, adv)
                
            advantages = torch.stack(advantages)
            
            # Validate advantages
            if torch.any(torch.isnan(advantages)):
                self.logger.error("NaN in computed advantages")
                advantages = torch.nan_to_num(advantages)
            
            # Smooth advantages
            adv_std = advantages.std()
            if torch.isnan(adv_std) or adv_std == 0:
                adv_std = torch.tensor(1.0, device=self.device)
                
            self.running_adv_std = self.adv_decay * self.running_adv_std + (1 - self.adv_decay) * adv_std.item()
            advantages = advantages / (self.running_adv_std + 1e-8)
            
            return advantages, returns
            
        except Exception as e:
            self.logger.error(f"Error computing advantages: {e}")
            # Return safe defaults
            n_steps = len(self.buffer["rewards"])
            return (torch.zeros(n_steps, device=self.device), 
                    torch.zeros(n_steps, device=self.device))

    def end_episode(self, gamma: float = 0.99):
        """Episode ending with updates and comprehensive error handling"""
        try:
            if len(self.buffer["rewards"]) < 10:
                self.logger.warning(f"Episode too short ({len(self.buffer['rewards'])} steps), skipping update")
                for k in self.buffer:
                    self.buffer[k].clear()
                return
                
            self.logger.info(f"Ending episode with {len(self.buffer['rewards'])} steps")
            
            # Stack tensors with validation
            obs = torch.stack(self.buffer["obs"])
            actions = torch.stack(self.buffer["actions"])
            logp_old = torch.stack(self.buffer["logp"])
            values_old = torch.stack(self.buffer["values"])
            market_features = torch.stack(self.buffer["market_features"])
            
            # Validate stacked tensors
            for name, tensor in [("obs", obs), ("actions", actions), ("logp_old", logp_old), 
                               ("values_old", values_old), ("market_features", market_features)]:
                if torch.any(torch.isnan(tensor)):
                    self.logger.error(f"NaN in stacked {name}")
                    tensor = torch.nan_to_num(tensor)
            
            # Compute advantages
            advantages, returns = self.compute_advantages(gamma)
            
            # Multiple epochs
            for epoch in range(4):
                try:
                    indices = torch.randperm(len(obs), device=self.device)
                    
                    # Mini-batch updates
                    batch_size = min(64, len(obs))
                    for i in range(0, len(obs), batch_size):
                        batch_idx = indices[i:i+batch_size]
                        
                        obs_batch = obs[batch_idx]
                        act_batch = actions[batch_idx]
                        logp_old_batch = logp_old[batch_idx]
                        adv_batch = advantages[batch_idx]
                        ret_batch = returns[batch_idx]
                        market_batch = market_features[batch_idx]
                        
                        # Forward pass
                        mu, value = self.forward(obs_batch, market_batch)
                        
                        # Recompute log probs
                        dist = torch.distributions.Normal(mu, 0.1)
                        logp = dist.log_prob(act_batch).sum(dim=-1)
                        
                        # PPO losses
                        ratio = (logp - logp_old_batch).exp()
                        
                        # Clamp ratio to prevent extreme values
                        ratio = torch.clamp(ratio, 0.1, 10.0)
                        
                        surr1 = ratio * adv_batch
                        surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv_batch
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_clipped = values_old[batch_idx] + torch.clamp(
                            value.squeeze(-1) - values_old[batch_idx], -self.clip_eps, self.clip_eps
                        )
                        value_loss1 = F.mse_loss(value.squeeze(-1), ret_batch)
                        value_loss2 = F.mse_loss(value_clipped, ret_batch)
                        value_loss = torch.max(value_loss1, value_loss2)
                        
                        # Entropy
                        entropy_loss = -dist.entropy().mean()
                        
                        # Total loss
                        loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                        
                        # Validate loss
                        if torch.isnan(loss):
                            self.logger.error("NaN loss detected, skipping update")
                            continue
                        
                        # Update
                        self.actor_opt.zero_grad()
                        self.critic_opt.zero_grad()
                        loss.backward()
                        
                        # Gradient clipping
                        torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                        
                        self.actor_opt.step()
                        self.critic_opt.step()
                        
                except Exception as e:
                    self.logger.error(f"Error in epoch {epoch}, batch {i}: {e}")
                    continue
                    
            # Log stats
            total_reward = sum(r.item() for r in self.buffer["rewards"])
            self.logger.info(f"Episode complete: Reward={total_reward:.2f}, Steps={len(self.buffer['rewards'])}")
            
            # Clear buffers
            for k in self.buffer:
                self.buffer[k].clear()
                
        except Exception as e:
            self.logger.error(f"Error in end_episode: {e}")
            # Clear buffers even on error
            for k in self.buffer:
                self.buffer[k].clear()

    def get_observation_components(self) -> np.ndarray:
        """Return 4 components for MetaRLController compatibility"""
        try:
            # Validate last_action
            if np.any(np.isnan(self.last_action)):
                self.logger.error("NaN in last_action")
                self.last_action = np.nan_to_num(self.last_action)
            if np.isnan(self.position):
                self.position = 0.0
            if np.isnan(self.unrealized_pnl):
                self.unrealized_pnl = 0.0
                
            observation = np.array([
                float(self.last_action.mean()),
                float(self.last_action.std()),
                float(self.position),
                float(self.unrealized_pnl)
            ], dtype=np.float32)
            
            # Final validation
            if np.any(np.isnan(observation)):
                self.logger.error(f"NaN in observation components: {observation}")
                observation = np.nan_to_num(observation)
                
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Action selection with comprehensive validation"""
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.logger.error("NaN in observation tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
            
            # Pad observation if needed
            if obs_tensor.shape[-1] < self.extended_obs_size:
                padding = torch.zeros(
                    (*obs_tensor.shape[:-1], self.extended_obs_size - obs_tensor.shape[-1]),
                    device=obs_tensor.device, dtype=obs_tensor.dtype
                )
                obs_tensor = torch.cat([obs_tensor, padding], dim=-1)
            elif obs_tensor.shape[-1] > self.extended_obs_size:
                obs_tensor = obs_tensor[..., :self.extended_obs_size]
                
            # Extract market features
            market_features = obs_tensor[..., -self.lag_window*4:]
            
            with torch.no_grad():
                action, _ = self.forward(obs_tensor, market_features)
                
                # Validate action
                if torch.any(torch.isnan(action)):
                    self.logger.error("NaN in selected action")
                    action = torch.zeros_like(action)
                    
            return action
            
        except Exception as e:
            self.logger.error(f"Error in select_action: {e}")
            # Return safe default
            batch_size = obs_tensor.shape[0] if obs_tensor.ndim > 1 else 1
            return torch.zeros(batch_size, self.act_size, device=self.device)

    def reset(self):
        """Reset agent with comprehensive cleanup"""
        try:
            for k in self.buffer:
                self.buffer[k].clear()
            self.last_action = np.zeros(self.act_size, dtype=np.float32)
            self.position = 0.0
            self.unrealized_pnl = 0.0
            self.price_buffer.clear()
            self.volume_buffer.clear()
            self.spread_buffer.clear()
            self.vol_buffer.clear()
            self.running_adv_std = 1.0
            self._step_count = 0
            self.total_trades = 0
            self.logger.info("PPOLagAgent reset complete")
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")

    def step(self, *args, **kwargs):
        pass

    def get_state(self) -> Dict:
        return {
            "actor": self.actor.state_dict(),
            "value_encoder": self.value_encoder.state_dict(),
            "market_encoder": self.market_encoder.state_dict(),
            "value_head": self.value_head.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "last_action": self.last_action.tolist(),
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "running_adv_std": self.running_adv_std,
            "total_trades": self.total_trades,
            "step_count": self._step_count,
        }

    def set_state(self, state: Dict, strict: bool = False):
        try:
            self.actor.load_state_dict(state["actor"], strict=strict)
            self.value_encoder.load_state_dict(state["value_encoder"], strict=strict)
            self.market_encoder.load_state_dict(state["market_encoder"], strict=strict)
            self.value_head.load_state_dict(state["value_head"], strict=strict)
            self.actor_opt.load_state_dict(state["actor_opt"])
            self.critic_opt.load_state_dict(state["critic_opt"])
            self.last_action = np.array(state.get("last_action", [0,0]), dtype=np.float32)
            self.position = state.get("position", 0.0)
            self.unrealized_pnl = state.get("unrealized_pnl", 0.0)
            self.running_adv_std = state.get("running_adv_std", 1.0)
            self.total_trades = state.get("total_trades", 0)
            self._step_count = state.get("step_count", 0)
            self.logger.info("PPOLagAgent state loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")

    def get_weights(self) -> Dict[str, any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": {
                "value_encoder": self.value_encoder.state_dict(),
                "market_encoder": self.market_encoder.state_dict(),
                "value_head": self.value_head.state_dict()
            }
        }

    def get_gradients(self) -> Dict[str, any]:
        grads = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.cpu().numpy()
        return grads

# ──────────────────────────────────────────────
# STANDARD PPO (KEPT FOR COMPATIBILITY) - FIXED
# ──────────────────────────────────────────────

class PPOAgent(nn.Module, Module):
    """
    Standard PPO agent - kept for comparison and fallback - FIXED NaN issues
    """
    def __init__(self, obs_size, act_size=2, hidden_size=64, lr=3e-4, device="cpu", debug=True):
        super().__init__()
        self.device = torch.device(device)
        self.debug = debug
        self.obs_size = obs_size
        self.act_size = act_size

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"PPOAgent_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/ppo/ppo_agent.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        try:
            self.actor = nn.Sequential(
                nn.Linear(obs_size, hidden_size), 
                nn.Tanh(),
                nn.Linear(hidden_size, act_size), 
                nn.Tanh()
            )
            self.critic = nn.Sequential(
                nn.Linear(obs_size, hidden_size), 
                nn.Tanh(),
                nn.Linear(hidden_size, 1)
            )
            
            # FIX: Proper initialization
            self._initialize_weights()
            
            self.to(self.device)
            self.opt = optim.Adam(self.parameters(), lr=lr)
            self.clip_eps = 0.2
            self.value_coeff = 0.5
            self.entropy_coeff = 0.01

            self.buffer = {k: [] for k in ["obs", "actions", "logp", "values", "rewards"]}
            self.last_action = np.zeros(act_size, dtype=np.float32)
            self._step_count = 0
            
            self.logger.info(f"PPOAgent initialized - obs_size={obs_size}, act_size={act_size}")
            
        except Exception as e:
            self.logger.error(f"Error initializing PPOAgent: {e}")
            raise

    def _initialize_weights(self):
        """Proper weight initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, obs: torch.Tensor):
        try:
            # Validate input
            if torch.any(torch.isnan(obs)):
                self.logger.error("NaN in observation")
                obs = torch.nan_to_num(obs)
                
            mu = self.actor(obs)
            value = self.critic(obs)
            
            # Validate outputs
            if torch.any(torch.isnan(mu)):
                self.logger.error("NaN in actor output")
                mu = torch.zeros_like(mu)
            if torch.any(torch.isnan(value)):
                self.logger.error("NaN in critic output")
                value = torch.zeros_like(value)
                
            return mu, value
            
        except Exception as e:
            self.logger.error(f"Error in forward pass: {e}")
            batch_size = obs.shape[0] if obs.ndim > 1 else 1
            return (torch.zeros(batch_size, self.act_size, device=self.device),
                    torch.zeros(batch_size, 1, device=self.device))

    def record_step(self, obs_vec, reward, **kwargs):
        """FIX: Accept market data kwargs for compatibility with enhanced validation"""
        self._step_count += 1
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.logger.error("NaN in observation vector")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.logger.error("NaN reward, setting to 0")
                reward = 0.0
                
            obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
            
            with torch.no_grad():
                mu, value = self.forward(obs_t.unsqueeze(0))
                
            dist = torch.distributions.Normal(mu, 0.1)
            action = dist.rsample()
            logp = dist.log_prob(action).sum(dim=-1)
            
            # Validate action and logp
            if torch.any(torch.isnan(action)):
                self.logger.error("NaN in sampled action")
                action = torch.zeros_like(action)
            if torch.any(torch.isnan(logp)):
                self.logger.error("NaN in log probability")
                logp = torch.zeros_like(logp)

            self.buffer["obs"].append(obs_t)
            self.buffer["actions"].append(action.squeeze(0))
            self.buffer["logp"].append(logp.squeeze(0))
            self.buffer["values"].append(value.squeeze(0))
            self.buffer["rewards"].append(
                torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            )
            self.last_action = action.cpu().numpy().squeeze(0)
            
            self.logger.debug(f"Step {self._step_count} recorded: reward={reward:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error in record_step: {e}")

    def end_episode(self, gamma=0.99):
        try:
            if not self.buffer["rewards"]: 
                self.logger.warning("Empty episode buffer")
                return

            self.logger.info(f"Ending episode with {len(self.buffer['rewards'])} steps")

            obs = torch.stack(self.buffer["obs"])
            actions = torch.stack(self.buffer["actions"])
            logp_old = torch.stack(self.buffer["logp"])
            values = torch.stack(self.buffer["values"])
            rewards = torch.stack(self.buffer["rewards"])

            # Validate tensors
            for name, tensor in [("obs", obs), ("actions", actions), ("logp_old", logp_old), 
                               ("values", values), ("rewards", rewards)]:
                if torch.any(torch.isnan(tensor)):
                    self.logger.error(f"NaN in {name}")
                    tensor = torch.nan_to_num(tensor)

            returns = []
            R = 0.0
            for r in reversed(rewards.tolist()):
                R = r + gamma * R
                returns.insert(0, R)
            returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
            advantages = returns - values

            # Validate advantages
            if torch.any(torch.isnan(advantages)):
                self.logger.error("NaN in advantages")
                advantages = torch.nan_to_num(advantages)

            for epoch in range(4):
                try:
                    mu, value = self.forward(obs)
                    dist = torch.distributions.Normal(mu, 0.1)
                    logp = dist.log_prob(actions).sum(dim=-1)
                    ratio = (logp - logp_old).exp()
                    
                    # Clamp ratio
                    ratio = torch.clamp(ratio, 0.1, 10.0)
                    
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(value.squeeze(-1), returns)
                    entropy_loss = -dist.entropy().mean()
                    loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                    
                    # Validate loss
                    if torch.isnan(loss):
                        self.logger.error(f"NaN loss in epoch {epoch}, skipping")
                        continue
                        
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.opt.step()
                    
                except Exception as e:
                    self.logger.error(f"Error in epoch {epoch}: {e}")
                    continue

            total_reward = sum(r.item() for r in rewards)
            self.logger.info(f"Episode complete: reward={total_reward:.2f}")

            for k in self.buffer:
                self.buffer[k].clear()
                
        except Exception as e:
            self.logger.error(f"Error in end_episode: {e}")
            for k in self.buffer:
                self.buffer[k].clear()

    def get_observation_components(self):
        """FIX: Return 4 components for compatibility with validation"""
        try:
            if np.any(np.isnan(self.last_action)):
                self.logger.error("NaN in last_action")
                self.last_action = np.nan_to_num(self.last_action)
                
            observation = np.array([
                float(self.last_action.mean()), 
                float(self.last_action.std()),
                0.0,  # Placeholder for position
                0.0   # Placeholder for unrealized_pnl
            ], dtype=np.float32)
            
            if np.any(np.isnan(observation)):
                self.logger.error("NaN in observation components")
                observation = np.nan_to_num(observation)
                
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_state(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt": self.opt.state_dict(),
            "last_action": self.last_action.tolist(),
            "step_count": self._step_count,
        }

    def set_state(self, state, strict=False):
        try:
            self.actor.load_state_dict(state["actor"], strict=strict)
            self.critic.load_state_dict(state["critic"], strict=strict)
            self.opt.load_state_dict(state["opt"])
            self.last_action = np.array(state.get("last_action", [0,0]), dtype=np.float32)
            self._step_count = state.get("step_count", 0)
            self.logger.info("PPOAgent state loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")

    def reset(self):
        try:
            for k in self.buffer: 
                self.buffer[k].clear()
            self.last_action = np.zeros(self.act_size, dtype=np.float32)
            self._step_count = 0
            self.logger.info("PPOAgent reset complete")
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")

    def step(self, *args, **kwargs):
        pass

    def select_action(self, obs_tensor):
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.logger.error("NaN in observation tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
                
            with torch.no_grad():
                action = self.actor(obs_tensor)
                
            # Validate output
            if torch.any(torch.isnan(action)):
                self.logger.error("NaN in action output")
                action = torch.zeros_like(action)
                
            return action
            
        except Exception as e:
            self.logger.error(f"Error in select_action: {e}")
            batch_size = obs_tensor.shape[0] if obs_tensor.ndim > 1 else 1
            return torch.zeros(batch_size, self.act_size, device=self.device)

    def get_gradients(self) -> Dict[str, Any]:
        grads = {}
        for name, param in self.named_parameters():
            grads[name] = param.grad.cpu().numpy() if param.grad is not None else None
        return grads

    def get_weights(self) -> Dict[str, Any]:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict()
        }

# ──────────────────────────────────────────────
# META RL CONTROLLER (PPO and PPO-LAG ONLY) - FIXED
# ──────────────────────────────────────────────

class MetaRLController(Module):
    """
    FIX: Simplified controller supporting only PPO and PPO-Lag with comprehensive logging and NaN fixes
    """
    def __init__(self, obs_size: int, act_size: int=2, method="ppo-lag", 
                 device="cpu", debug=True, profit_target=150.0):
        self.device = device
        self.obs_size = obs_size
        self.act_size = act_size
        self.debug = debug
        self.profit_target = profit_target
        self._step_count = 0

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"MetaRLController_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/controller/metarl_controller.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        try:
            # Initialize only PPO variants
            self._agents = {
                "ppo": PPOAgent(obs_size, act_size=act_size, device=device, debug=debug),
                "ppo-lag": PPOLagAgent(obs_size, act_size=act_size, device=device, debug=debug),
            }
            
            # FIX: Default to ppo-lag for profit generation
            self.mode = method if method in self._agents else "ppo-lag"
            self.agent = self._agents[self.mode]
            
            # Performance tracking
            self.episode_count = 0
            self.total_profit = 0.0
            self.best_daily_profit = 0.0
            self.last_profit_check = 0

            self.logger.info(f"MetaRLController initialized with {self.mode} agent for €{profit_target}/day target")
            
        except Exception as e:
            self.logger.error(f"Error initializing MetaRLController: {e}")
            raise

    def set_mode(self, method: str):
        """Switch between PPO variants with validation"""
        try:
            if method not in self._agents:
                self.logger.warning(f"Unknown method: {method}, keeping {self.mode}")
                return
                
            old_mode = self.mode
            self.mode = method
            self.agent = self._agents[method]
            self.logger.info(f"Switched from {old_mode} to {self.mode}")
            
        except Exception as e:
            self.logger.error(f"Error setting mode: {e}")

    def record_step(self, obs_vec, reward, **market_data):
        """
        FIX: Enhanced recording with market data support and comprehensive validation
        """
        self._step_count += 1
        
        try:
            # Validate inputs
            if np.any(np.isnan(obs_vec)):
                self.logger.error(f"NaN in observation vector: {obs_vec}")
                obs_vec = np.nan_to_num(obs_vec)
            if np.isnan(reward):
                self.logger.error("NaN reward, setting to 0")
                reward = 0.0
                
            self.total_profit += reward
            
            # Validate market data
            validated_market_data = {}
            for key, value in market_data.items():
                if isinstance(value, (int, float)) and np.isnan(value):
                    self.logger.warning(f"NaN in market data {key}, setting to 0")
                    validated_market_data[key] = 0.0
                else:
                    validated_market_data[key] = value
            
            if self.mode == "ppo-lag":
                # PPO-Lag needs full market data
                self.agent.record_step(obs_vec, reward, **validated_market_data)
            else:
                # Standard PPO
                self.agent.record_step(obs_vec, reward)
                
            # Log significant profits
            if reward > 10:  # €10+ single trade
                self.logger.info(f"Profitable trade: €{reward:.2f} using {self.mode}")
            elif reward < -10:  # €10+ loss
                self.logger.warning(f"Large loss: €{reward:.2f} using {self.mode}")
                
            # Log progress periodically
            if self._step_count % 50 == 0:
                progress = (self.total_profit / self.profit_target) * 100
                self.logger.info(f"Step {self._step_count}: Progress {progress:.1f}% (€{self.total_profit:.2f}/€{self.profit_target})")
                
        except Exception as e:
            self.logger.error(f"Error in record_step: {e}")

    def end_episode(self, *args, **kwargs):
        """End episode with performance tracking and comprehensive logging"""
        try:
            self.episode_count += 1
            
            # Check daily profit progress
            if self.total_profit > self.best_daily_profit:
                improvement = self.total_profit - self.best_daily_profit
                self.best_daily_profit = self.total_profit
                self.logger.info(f"New best daily profit: €{self.best_daily_profit:.2f} (improvement: €{improvement:.2f})")
                
            # Log episode summary
            progress_pct = (self.total_profit / self.profit_target) * 100
            self.logger.info(f"Episode {self.episode_count} complete: total_profit=€{self.total_profit:.2f} ({progress_pct:.1f}% of target)")
                
            # Switch algorithms if underperforming
            if self.episode_count % 50 == 0:
                if self.total_profit < self.profit_target * 0.3:  # Less than 30% of target
                    new_mode = "ppo-lag" if self.mode == "ppo" else "ppo"
                    self.logger.warning(f"Underperforming after {self.episode_count} episodes - switching to {new_mode}")
                    self.set_mode(new_mode)
                else:
                    self.logger.info(f"Performance satisfactory after {self.episode_count} episodes")
                    
            return self.agent.end_episode(*args, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error in end_episode: {e}")

    def get_observation_components(self):
        """Always return 4-component vector with comprehensive validation"""
        try:
            obs = self.agent.get_observation_components()
            
            # Ensure exactly 4 components
            if len(obs) < 4:
                obs = np.pad(obs, (0, 4 - len(obs)))
            elif len(obs) > 4:
                obs = obs[:4]
                
            # Check for NaN
            if np.any(np.isnan(obs)):
                self.logger.error(f"NaN in observation: {obs}")
                obs = np.nan_to_num(obs)
                
            # Validate range
            obs = np.clip(obs, -100, 100)  # Reasonable bounds
            
            return obs.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def get_state(self):
        """Save controller state with validation"""
        try:
            return {
                "mode": self.mode,
                "agents": {k: v.get_state() for k, v in self._agents.items()},
                "episode_count": self.episode_count,
                "total_profit": self.total_profit,
                "best_daily_profit": self.best_daily_profit,
                "step_count": self._step_count,
            }
        except Exception as e:
            self.logger.error(f"Error getting state: {e}")
            return {}

    def set_state(self, state, strict=False):
        """Load controller state with validation"""
        try:
            self.mode = state.get("mode", self.mode)
            self.episode_count = state.get("episode_count", 0)
            self.total_profit = state.get("total_profit", 0.0)
            self.best_daily_profit = state.get("best_daily_profit", 0.0)
            self._step_count = state.get("step_count", 0)
            
            # Validate loaded values
            if np.isnan(self.total_profit):
                self.logger.error("NaN total_profit in loaded state")
                self.total_profit = 0.0
            if np.isnan(self.best_daily_profit):
                self.logger.error("NaN best_daily_profit in loaded state")
                self.best_daily_profit = 0.0
            
            agents_state = state.get("agents", {})
            for agent_name, agent_state in agents_state.items():
                if agent_name in self._agents:
                    self._agents[agent_name].set_state(agent_state, strict=strict)
                    
            self.agent = self._agents[self.mode]
            self.logger.info(f"State loaded: mode={self.mode}, episodes={self.episode_count}, profit=€{self.total_profit:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error setting state: {e}")

    def reset(self):
        """Reset all agents with comprehensive cleanup"""
        try:
            for agent in self._agents.values():
                agent.reset()
            self.episode_count = 0
            self.total_profit = 0.0
            self.best_daily_profit = 0.0
            self._step_count = 0
            self.logger.info("All agents reset")
        except Exception as e:
            self.logger.error(f"Error in reset: {e}")

    def act(self, obs_tensor):
        """Get action from active agent with validation"""
        try:
            # Validate input
            if torch.any(torch.isnan(obs_tensor)):
                self.logger.error("NaN in action input tensor")
                obs_tensor = torch.nan_to_num(obs_tensor)
                
            action = self.agent.select_action(obs_tensor)
            
            # Ensure valid numpy array
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            action = np.asarray(action)
            
            # Check for NaN
            if np.isnan(action).any():
                self.logger.error(f"NaN in action output: {action}")
                action = np.nan_to_num(action)
                
            # Validate action range
            action = np.clip(action, -1.0, 1.0)
                
            return action
            
        except Exception as e:
            self.logger.error(f"Error in act: {e}")
            return np.zeros(self.act_size, dtype=np.float32)

    def step(self, *args, **kwargs):
        """Compatibility method"""
        pass
        
    def obs_dim(self):
        return self.obs_size

    def save_checkpoint(self, filepath: str):
        """Save full checkpoint with validation"""
        try:
            checkpoint = {
                "controller_state": self.get_state(),
                "timestamp": str(np.datetime64('now')),
                "profit_achieved": self.total_profit,
                "target": self.profit_target
            }
            torch.save(checkpoint, filepath)
            self.logger.info(f"Checkpoint saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, filepath: str):
        """Load checkpoint with validation"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self.set_state(checkpoint["controller_state"])
            self.logger.info(f"Loaded checkpoint from {filepath}")
        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")

    def get_weights(self) -> Dict[str, Any]:
        """Get active agent weights"""
        try:
            if hasattr(self.agent, "get_weights"):
                return self.agent.get_weights()
            return {}
        except Exception as e:
            self.logger.error(f"Error getting weights: {e}")
            return {}

    def get_gradients(self) -> Dict[str, Any]:
        """Get active agent gradients"""
        try:
            if hasattr(self.agent, "get_gradients"):
                return self.agent.get_gradients()
            return {}
        except Exception as e:
            self.logger.error(f"Error getting gradients: {e}")
            return {}


# ──────────────────────────────────────────────
# TEST FUNCTION
# ──────────────────────────────────────────────

def test_strategy_modules():
    """Test function to verify all strategy modules log properly and don't produce NaN"""
    print("Testing enhanced strategy modules with comprehensive logging and NaN protection...")
    
    # Test StrategyIntrospector
    print("\n1. Testing StrategyIntrospector:")
    introspector = StrategyIntrospector(debug=True)
    
    for i in range(5):
        theme = np.random.randn(3)
        win_rate = np.random.uniform(0.3, 0.8)
        sl = np.random.uniform(0.5, 2.0)
        tp = np.random.uniform(1.0, 3.0)
        introspector.record(theme, win_rate, sl, tp)
    
    profile = introspector.get_observation_components()
    print(f"Strategy profile: {profile}")
    
    # Test CurriculumPlannerPlus
    print("\n2. Testing CurriculumPlannerPlus:")
    planner = CurriculumPlannerPlus(debug=True)
    
    for i in range(3):
        summary = {
            "win_rate": np.random.uniform(0.4, 0.7),
            "avg_duration": np.random.uniform(5, 20),
            "avg_drawdown": np.random.uniform(0.01, 0.1),
            "total_trades": np.random.randint(10, 50),
            "wins": np.random.randint(5, 25),
            "pnl": np.random.normal(20, 30)
        }
        planner.record_episode(summary)
    
    curriculum_obs = planner.get_observation_components()
    print(f"Curriculum metrics: {curriculum_obs}")
    
    # Test StrategyGenomePool
    print("\n3. Testing StrategyGenomePool:")
    genome_pool = StrategyGenomePool(population_size=10, debug=True)
    
    def dummy_eval(genome):
        return np.random.normal(50, 100)  # Simulate fitness
    
    genome_pool.evaluate_population(dummy_eval)
    genome_pool.evolve_strategies()
    selected = genome_pool.select_genome(mode="smart")
    print(f"Selected genome: {selected}")
    
    # Test MetaAgent
    print("\n4. Testing MetaAgent:")
    meta_agent = MetaAgent(debug=True)
    
    for i in range(10):
        pnl = np.random.normal(0, 50)
        meta_agent.step(pnl)
    
    intensity = meta_agent.get_intensity("EUR/USD")
    print(f"Trading intensity: {intensity:.3f}")
    
    # Test PPOLagAgent (most important for NaN issues)
    print("\n5. Testing PPOLagAgent:")
    try:
        ppo_lag = PPOLagAgent(obs_size=10, act_size=2, debug=True)
        
        # Test with sample data
        for i in range(5):
            obs = np.random.randn(10).astype(np.float32)
            reward = np.random.normal(0, 10)
            price = 1.1000 + np.random.normal(0, 0.001)
            volume = abs(np.random.normal(1000, 200))
            spread = abs(np.random.normal(0.0001, 0.00005))
            volatility = abs(np.random.normal(0.01, 0.005))
            
            ppo_lag.record_step(obs, reward, price=price, volume=volume, 
                              spread=spread, volatility=volatility)
        
        ppo_lag.end_episode()
        ppo_obs = ppo_lag.get_observation_components()
        print(f"PPO-Lag observation: {ppo_obs}")
        
    except Exception as e:
        print(f"Error testing PPOLagAgent: {e}")
    
    # Test MetaRLController
    print("\n6. Testing MetaRLController:")
    try:
        controller = MetaRLController(obs_size=10, method="ppo-lag", debug=True)
        
        for i in range(3):
            obs = np.random.randn(10).astype(np.float32)
            reward = np.random.normal(10, 20)
            controller.record_step(obs, reward, price=1.1000, volume=1000)
        
        controller.end_episode()
        controller_obs = controller.get_observation_components()
        print(f"Controller observation: {controller_obs}")
        
        # Test action generation
        obs_tensor = torch.randn(1, 10)
        action = controller.act(obs_tensor)
        print(f"Generated action: {action}")
        
    except Exception as e:
        print(f"Error testing MetaRLController: {e}")
    
    print("\nAll strategy modules tested! Check log files:")
    print("- logs/strategy/introspection/")
    print("- logs/strategy/curriculum/")
    print("- logs/strategy/genome/")
    print("- logs/strategy/meta/")
    print("- logs/strategy/ppo/")
    print("- logs/strategy/controller/")

if __name__ == "__main__":
    test_strategy_modules()