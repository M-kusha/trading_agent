#
# modules/strategy/strategy_genome_pool.py
from __future__ import annotations
import hashlib
import logging
import numpy as np
from typing import Callable
from modules.core.core import Module



class StrategyGenomePool(Module):

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
