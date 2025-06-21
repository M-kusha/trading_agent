from __future__ import annotations
import hashlib
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Callable, Optional, Tuple
from collections import deque
from modules.core.core import Module

# ──────────────────────────────────────────────
class StrategyIntrospector(Module):
    """
    FIXED: Added bootstrap support and better initialization
    """
    def __init__(self, history_len: int = 10, debug: bool = True):
        self.history_len = history_len
        self.debug = debug
        self._records: List[Dict[str, float]] = []
        # FIX: Initialize with some baseline data to avoid zero observations
        self._baseline_wr = 0.5  # 50% win rate baseline
        self._baseline_sl = 1.0  # 1% stop loss baseline
        self._baseline_tp = 1.5  # 1.5% take profit baseline

    def reset(self) -> None:
        self._records.clear()

    def step(self, **kwargs) -> None:
        pass

    def record(self, theme: np.ndarray, win_rate: float, sl: float, tp: float) -> None:
        self._records.append({"wr": win_rate, "sl": sl, "tp": tp})
        if len(self._records) > self.history_len:
            self._records.pop(0)

    def profile(self) -> np.ndarray:
        if not self._records:
            # FIX: Return baseline values instead of zeros
            return np.array([
                self._baseline_wr, self._baseline_sl, self._baseline_tp,
                0.0, 0.0  # No variance yet
            ], dtype=np.float32)
        
        arr = np.array([[r["wr"], r["sl"], r["tp"]] for r in self._records], dtype=np.float32)
        
        # FIX: Calculate mean and variance for better signal
        mean_vals = arr.mean(axis=0)
        var_vals = arr.var(axis=0) if len(arr) > 1 else np.zeros(3)
        
        # Combine mean and variance info
        flat = np.concatenate([mean_vals, var_vals[:2]])  # Total 5 values
        return flat.astype(np.float32)

    def get_observation_components(self) -> np.ndarray:
        return self.profile()

# ──────────────────────────────────────────────
class CurriculumPlannerPlus(Module):
    """
    FIXED: Enhanced to track trading performance metrics properly
    """
    def __init__(self, window: int=10, debug=True):
        self.window = window
        self.debug = debug
        self._history: List[Dict[str,float]] = []
        # FIX: Track cumulative metrics for better learning
        self._total_trades = 0
        self._total_wins = 0
        self._cumulative_pnl = 0.0

    def reset(self):
        self._history.clear()
        self._total_trades = 0
        self._total_wins = 0
        self._cumulative_pnl = 0.0

    def step(self, **kwargs):
        pass

    def record_episode(self, summary: Dict[str,float]):
        self._history.append(summary)
        if len(self._history) > self.window:
            self._history.pop(0)
        
        # FIX: Update cumulative metrics
        if "total_trades" in summary:
            self._total_trades += summary["total_trades"]
        if "wins" in summary:
            self._total_wins += summary["wins"]
        if "pnl" in summary:
            self._cumulative_pnl += summary["pnl"]

    def get_observation_components(self) -> np.ndarray:
        if not self._history:
            # FIX: Return meaningful defaults for bootstrap
            return np.array([0.5, 0.0, 0.01], dtype=np.float32)  # [win_rate, avg_duration, avg_drawdown]
        
        # Calculate rolling metrics
        win_rates = [e.get("win_rate", 0) for e in self._history]
        durations = [e.get("avg_duration", 0) for e in self._history]
        drawdowns = [e.get("avg_drawdown", 0) for e in self._history]
        
        # FIX: Use robust averaging
        avg_wr = np.mean(win_rates) if win_rates else 0.5
        avg_dur = np.mean(durations) if durations else 0.0
        avg_dd = np.mean(drawdowns) if drawdowns else 0.01
        
        return np.array([avg_wr, avg_dur, avg_dd], dtype=np.float32)

# ──────────────────────────────────────────────
class StrategyGenomePool:
    """
    FIXED: Enhanced evolution with profit-focused fitness and better diversity
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
        log_file: str = "logs/sgp.log",
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

        # FIX: Better seed genomes for XAUUSD/EURUSD trading
        seeds = [
            np.array([0.5, 0.75, 1.0, 0.2], dtype=np.float32),  # Conservative scalping
            np.array([1.0, 1.5, 1.2, 0.3], dtype=np.float32),   # Balanced
            np.array([1.5, 2.0, 1.5, 0.4], dtype=np.float32),   # Aggressive
            np.array([0.8, 1.2, 0.8, 0.25], dtype=np.float32),  # Tight stops
        ]
        
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

        # Logger setup
        self.logger = logging.getLogger("SGP")
        if not self.logger.handlers:
            for h in list(self.logger.handlers):
                self.logger.removeHandler(h)
            fh = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def reset(self) -> None:
        self.epoch = 0
        self.fitness[:] = 0.0
        self.generations_without_improvement = 0

    def step(self, **kwargs) -> None:
        pass

    def genome_hash(self, g: np.ndarray) -> str:
        return hashlib.md5(g.tobytes()).hexdigest()

    def evaluate_population(self, eval_fn: Callable[[np.ndarray], float]) -> None:
        """
        FIX: Enhanced evaluation with profit-focused metrics
        """
        for i, genome in enumerate(self.population):
            raw_fitness = float(eval_fn(genome))
            
            # FIX: Bonus for achieving profit target
            if raw_fitness >= self.profit_target:
                profit_bonus = 1.0 + (raw_fitness - self.profit_target) / self.profit_target
                self.fitness[i] = raw_fitness * profit_bonus
            else:
                self.fitness[i] = raw_fitness
                
        # Track best
        max_idx = np.argmax(self.fitness)
        if self.fitness[max_idx] > self.best_fitness:
            self.best_fitness = self.fitness[max_idx]
            self.best_genome = self.population[max_idx].copy()
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
            
        if self.debug:
            self.logger.debug(
                f"[SGP] Gen {self.epoch} fitness: "
                f"min={self.fitness.min():.3f}, max={self.fitness.max():.3f}, "
                f"best_ever={self.best_fitness:.3f}"
            )

    def evolve_strategies(self) -> None:
        """
        FIX: Enhanced evolution with elitism and adaptive mutation
        """
        old_hashes = [self.genome_hash(g) for g in self.population]
        
        # FIX: Adaptive mutation based on progress
        if self.generations_without_improvement > 10:
            adaptive_mut_rate = min(0.3, self.mut_rate * 2)
            adaptive_mut_scale = min(0.5, self.mut_scale * 1.5)
        else:
            adaptive_mut_rate = self.mut_rate
            adaptive_mut_scale = self.mut_scale

        # FIX: Always preserve best genome (elitism)
        new_pop = [self.best_genome.copy()]
        
        # Diversity injection if stagnant
        if np.std(self.fitness) < 0.1 or self.generations_without_improvement > 20:
            # Inject fresh genomes focused on profit
            new_genes = np.random.uniform(
                low=[0.5, 0.8, 0.8, 0.15],
                high=[1.5, 2.2, 1.8, 0.4],
                size=(3, 4),
            ).astype(np.float32)
            new_pop.extend(new_genes)
            if self.debug:
                self.logger.debug(f"[SGP] Injected {len(new_genes)} fresh profit-focused genomes")
        
        # Generate rest of population
        while len(new_pop) < self.pop_size:
            # Tournament selection
            cand = np.random.choice(self.pop_size, self.tournament_k, replace=False)
            p1 = self.population[cand[np.argmax(self.fitness[cand])]]
            cand = np.random.choice(self.pop_size, self.tournament_k, replace=False)
            p2 = self.population[cand[np.argmax(self.fitness[cand])]]

            # Crossover
            mask = np.random.rand(self.genome_size) < self.cx_rate
            child = np.where(mask, p1, p2).copy()

            # Mutation with adaptive rates
            m_idx = np.random.rand(self.genome_size) < adaptive_mut_rate
            if m_idx.any():
                child[m_idx] += np.random.randn(m_idx.sum()) * adaptive_mut_scale
                
                # FIX: Trading-appropriate bounds
                child[0] = np.clip(child[0], 0.2, 3.0)   # SL: 20-300 pips
                child[1] = np.clip(child[1], 0.3, 4.0)   # TP: 30-400 pips
                child[2] = np.clip(child[2], 0.1, 2.5)   # Vol scale
                child[3] = np.clip(child[3], 0.0, 0.6)   # Regime adapt
                
            new_pop.append(child.astype(np.float32))

        self.population = np.vstack(new_pop[:self.pop_size])
        self.fitness[:] = 0.0
        self.epoch += 1
        
        # Log evolution progress
        new_hashes = [self.genome_hash(g) for g in self.population]
        n_changed = sum(1 for o, n in zip(old_hashes, new_hashes) if o != n)
        if n_changed > 0 or self.debug:
            self.logger.info(
                f"[SGP] Gen {self.epoch}: {n_changed}/{self.pop_size} changed, "
                f"stagnant for {self.generations_without_improvement} gens"
            )

    def select_genome(self, mode="smart", k=3, custom_selector=None):
        """
        FIX: Enhanced selection with profit-aware modes
        """
        assert self.population.shape[0] == self.fitness.shape[0], "Population/fitness size mismatch"
        N = self.population.shape[0]

        if mode == "smart":
            # FIX: Smart selection based on recent performance
            if self.generations_without_improvement < 5:
                # Exploit best when improving
                idx = int(np.argmax(self.fitness))
            else:
                # Explore when stagnant
                top_k = np.argsort(self.fitness)[-5:]
                idx = np.random.choice(top_k)
                
        elif mode == "random":
            idx = np.random.randint(N)
        elif mode == "best":
            idx = int(np.argmax(self.fitness))
        elif mode == "tournament":
            candidates = np.random.choice(N, k, replace=False)
            idx = candidates[np.argmax(self.fitness[candidates])]
        elif mode == "roulette":
            fit = self.fitness - np.min(self.fitness) + 1e-8
            probs = fit / fit.sum() if fit.sum() > 0 else np.ones(N) / N
            idx = np.random.choice(N, p=probs)
        elif mode == "custom":
            assert custom_selector is not None, "Provide custom_selector callable!"
            idx = custom_selector(self.population, self.fitness)
        else:
            raise ValueError(f"Unknown selection mode: {mode}")

        self.active_genome = self.population[idx].copy()
        self.active_genome_idx = idx

        if self.debug:
            fit_val = self.fitness[idx]
            genome_str = ", ".join(f"{x:.3f}" for x in self.active_genome)
            self.logger.debug(
                f"[SGP] Selected genome idx={idx}, fitness={fit_val:.3f}, "
                f"genome=[{genome_str}] (mode={mode})"
            )
        return self.active_genome

    def get_observation_components(self) -> np.ndarray:
        """
        FIX: Enhanced observation with more informative metrics
        """
        mean_f = float(self.fitness.mean())
        max_f = float(self.fitness.max())
        
        # Diversity calculation
        if self.pop_size > 1:
            P = self.population.astype(np.float32)
            dists = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
            diversity = float(dists.mean())
        else:
            diversity = 0.0
            
        # FIX: Add profit achievement ratio
        profit_ratio = max_f / self.profit_target if self.profit_target > 0 else 0.0
        
        return np.array([mean_f, max_f, diversity, profit_ratio], dtype=np.float32)

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

# ──────────────────────────────────────────────
class MetaAgent(Module):
    """
    FIX: Enhanced with bootstrap intensity and profit tracking
    """
    def __init__(self, window: int=20, debug=True, profit_target=150.0):
        self.window = window
        self.debug = debug
        self.profit_target = profit_target
        self.reset()

    def reset(self):
        self.history: List[float] = []
        self.trade_count = 0
        self.consecutive_losses = 0

    def step(self, pnl: float=0.0):
        self.history.append(pnl)
        if len(self.history) > self.window:
            self.history.pop(0)
        
        # Track consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        self.trade_count += 1

    def record(self, pnl: float):
        self.step(pnl)

    def get_observation_components(self)->np.ndarray:
        if not self.history:
            return np.array([0.0, 0.0], dtype=np.float32)
        arr = np.array(self.history, dtype=np.float32)
        return np.array([arr.mean(), arr.std()], dtype=np.float32)
    
    def get_intensity(self, instrument: str) -> float:
        """
        FIX: Smarter intensity calculation for profitable trading
        """
        # Bootstrap intensity for initial trades
        if self.trade_count < 5:
            # Start with moderate positive intensity to encourage trading
            intensity = 0.3 + np.random.uniform(-0.1, 0.1)
            if self.debug:
                print(f"[MetaAgent] Bootstrap intensity for {instrument}: {intensity:.3f}")
            return float(intensity)
        
        if not self.history:
            return 0.0
            
        # Calculate recent performance
        recent_pnl = self.history[-min(10, len(self.history)):]
        avg_pnl = np.mean(recent_pnl)
        
        # FIX: Profit-aware intensity
        if avg_pnl > self.profit_target / self.window:
            # Above target: maintain momentum
            intensity = 0.7 + min(0.3, avg_pnl / (self.profit_target * 2))
        elif avg_pnl > 0:
            # Profitable but below target: increase aggression
            intensity = 0.3 + (avg_pnl / self.profit_target)
        else:
            # Losing: reduce but don't stop
            intensity = max(-0.5, -0.1 - self.consecutive_losses * 0.05)
            
        # Clamp to safe range
        intensity = np.clip(intensity, -0.8, 0.9)
        
        if self.debug:
            print(f"[MetaAgent] Intensity for {instrument}: {intensity:.3f} "
                  f"(avg_pnl={avg_pnl:.3f}, losses={self.consecutive_losses})")
        return float(intensity)

# ──────────────────────────────────────────────
class MetaCognitivePlanner(Module):
    """
    FIX: Enhanced with better episode tracking
    """
    def __init__(self, window: int=20, debug=True):
        self.window = window
        self.debug  = debug
        self.reset()

    def reset(self):
        self.history: List[Dict[str,float]] = []
        self.total_episodes = 0
        self.profitable_episodes = 0

    def step(self, **kwargs): 
        pass

    def record_episode(self, result: Dict[str,float]):
        self.history.append(result)
        if len(self.history) > self.window:
            self.history.pop(0)
            
        # Track profitable episodes
        self.total_episodes += 1
        if result.get("pnl", 0) > 0:
            self.profitable_episodes += 1

    def get_observation_components(self)->np.ndarray:
        if not self.history:
            # FIX: Better bootstrap values
            return np.array([0.5, 0.0, 10.0], dtype=np.float32)
            
        pnls = np.array([r.get("pnl",0) for r in self.history], dtype=np.float32)
        win_rate = float((pnls>0).sum() / len(pnls)) if len(pnls)>0 else 0.5
        
        # FIX: Add risk-adjusted return metric
        avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0.0
        avg_loss = abs(pnls[pnls < 0].mean()) if (pnls < 0).any() else 1.0
        risk_reward = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        return np.array([win_rate, float(pnls.mean()), risk_reward], dtype=np.float32)

# ──────────────────────────────────────────────
class BiasAuditor(Module):
    """
    FIX: Enhanced to detect and correct trading biases
    """
    def __init__(self, history_len: int=100, debug=True):
        self.history_len = history_len
        self.debug = debug
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.history_len)
        self.bias_corrections = {"revenge": 0, "fear": 0, "greed": 0}

    def step(self, **kwargs): 
        pass

    def record(self, bias: str, pnl: float = 0):
        """FIX: Record bias with outcome"""
        self.hist.append((bias, pnl))
        
        # Learn bias corrections
        if pnl < 0:
            self.bias_corrections[bias] = self.bias_corrections.get(bias, 0) + 1

    def get_observation_components(self)->np.ndarray:
        total = len(self.hist)
        if total == 0:
            # FIX: Balanced initial state
            return np.array([0.33, 0.33, 0.33], dtype=np.float32)
            
        cnt = {"revenge":0,"fear":0,"greed":0}
        for b, _ in self.hist:
            if b in cnt: 
                cnt[b] += 1
                
        # FIX: Apply corrections to discourage losing biases
        for bias, count in self.bias_corrections.items():
            if count > 5:  # Significant negative bias
                cnt[bias] = max(0, cnt[bias] - count // 2)
                
        total = sum(cnt.values()) or 1
        freqs = np.array([cnt["revenge"],cnt["fear"],cnt["greed"]], dtype=np.float32) / total
        return freqs

# ──────────────────────────────────────────────
class OpponentModeEnhancer(Module):
    """
    FIX: Market regime detector with profit tracking
    """
    def __init__(self, modes: List[str]=None, debug=True):
        self.modes = modes or ["trending", "ranging", "volatile"]  # FIX: Better market modes
        self.debug = debug
        self.reset()

    def reset(self):
        self.pnl = {m: 0.0 for m in self.modes}
        self.counts = {m: 0 for m in self.modes}

    def step(self, **kwargs): 
        pass

    def record_result(self, mode: str, pnl: float):
        if mode in self.pnl:
            self.pnl[mode] += pnl
            self.counts[mode] += 1

    def get_observation_components(self) -> np.ndarray:
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
        if arr.sum() > 0:
            weights = arr / arr.sum()
        else:
            weights = np.ones(len(self.modes)) / len(self.modes)
            
        return weights

# ──────────────────────────────────────────────
class ThesisEvolutionEngine(Module):
    """
    FIX: Track trading thesis performance
    """
    def __init__(self, capacity: int=20, debug=True):
        self.capacity = capacity
        self.debug = debug
        self.reset()

    def reset(self):
        self.theses: List[str] = []  # Changed from Any to str for clarity
        self.pnls: List[float] = []
        self.thesis_performance: Dict[str, List[float]] = {}

    def step(self, **kwargs): 
        pass

    def record_thesis(self, thesis: str):
        """FIX: Track thesis properly"""
        self.theses.append(thesis)
        if thesis not in self.thesis_performance:
            self.thesis_performance[thesis] = []

    def record_pnl(self, pnl: float):
        if self.theses:
            current_thesis = self.theses[-1]
            self.pnls.append(pnl)
            self.thesis_performance[current_thesis].append(pnl)
            
            if len(self.pnls) > self.capacity:
                self.pnls.pop(0)

    def get_observation_components(self)->np.ndarray:
        if not self.pnls:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
            
        # FIX: Better metrics
        uniq = len(self.thesis_performance)
        mean_p = float(np.mean(self.pnls))
        
        # Find best performing thesis
        best_thesis_pnl = 0.0
        for thesis, pnls in self.thesis_performance.items():
            if pnls:
                thesis_avg = np.mean(pnls)
                best_thesis_pnl = max(best_thesis_pnl, thesis_avg)
                
        return np.array([float(uniq), mean_p, best_thesis_pnl], dtype=np.float32)

# ──────────────────────────────────────────────
import numpy as np

class ExplanationGenerator(Module):
    """
    Enhanced explanation generator with robust argument handling and trading insights.
    """

    def __init__(self, debug: bool = True) -> None:
        super().__init__()
        self.debug = debug
        self.last_explanation = ""
        self.trade_count = 0
        self.profit_today = 0.0

    def reset(self) -> None:
        self.last_explanation = ""
        self.trade_count = 0
        self.profit_today = 0.0

    def step(
        self,
        actions=None,
        arbiter_weights=None,
        member_names=None,
        votes=None,
        regime="unknown",
        volatility=None,
        drawdown=0.0,
        genome_metrics=None,
        pnl=0.0,
        target_achieved=False,
        *args, **kwargs
    ) -> None:
        """
        Robust to missing pipeline arguments and preserves full analytics.
        Accepts and safely ignores unused/extra parameters.
        """

        # Defaults for safe operation
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

        self.trade_count += 1
        self.profit_today += pnl

        try:
            top_idx = int(np.argmax(arbiter_weights))
            top_name = member_names[top_idx] if top_idx < len(member_names) else "Unknown"
            top_w = float(arbiter_weights[top_idx]) * 100.0
        except Exception:
            top_idx, top_name, top_w = 0, "Unknown", 100.0

        # Aggregate votes, with strong fallback
        agg = {n: 0.0 for n in member_names}
        count = 0
        for vote_dict in votes.values():
            if isinstance(vote_dict, dict):
                for n, w in vote_dict.items():
                    agg[n] = agg.get(n, 0.0) + float(w)
                count += 1
        if count:
            for n in agg:
                agg[n] /= count
        votes_str = "; ".join(f"{n}: {agg[n] * 100.0:.1f}%" for n in list(member_names)[:3])

        high_vol_instruments = [inst for inst, vol in volatility.items() if isinstance(vol, (float, int)) and vol > 0.02]
        vol_warning = f" ⚠️ HIGH VOL: {', '.join(high_vol_instruments)}" if high_vol_instruments else ""

        sl_base = float(genome_metrics.get("sl_base", 1.0))
        tp_base = float(genome_metrics.get("tp_base", 1.5))
        risk_reward = tp_base / sl_base if sl_base > 0 else 1.5

        progress_pct = (self.profit_today / 150.0) * 100  # Against €150 target
        dd_pct = drawdown * 100.0

        self.last_explanation = (
            f"🎯 Day Progress: €{self.profit_today:.2f}/€150 ({progress_pct:.1f}%) | "
            f"Trades: {self.trade_count} | "
            f"Regime: {regime}{vol_warning} | "
            f"Strategy: {top_name} ({top_w:.0f}%) | "
            f"RR: {risk_reward:.1f}:1 | "
            f"DD: {dd_pct:.1f}%"
        )

        if target_achieved:
            self.last_explanation += " ✅ TARGET ACHIEVED - Consider stopping"
        elif dd_pct > 5:
            self.last_explanation += " ⚠️ High drawdown - Reduce position size"
        elif progress_pct < 30 and self.trade_count > 20:
            self.last_explanation += " 📊 Low progress - Review strategy"

        if self.debug:
            print("[ExplanationGenerator]", self.last_explanation)

    def get_observation_components(self) -> np.ndarray:
        """Return profit metrics. Always robust."""
        avg_profit = self.profit_today / max(1, self.trade_count)
        return np.array([
            self.profit_today,
            float(self.trade_count),
            avg_profit
        ], dtype=np.float32)

# ──────────────────────────────────────────────
# PPO-LAG IMPLEMENTATION
# ──────────────────────────────────────────────

class PPOLagAgent(nn.Module, Module):
    """
    PPO-Lag: Enhanced PPO for volatile financial markets
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
        
        # Expand input size for lagged features
        lag_features = 4  # returns, volatility, volume, spread
        position_features = 3 if position_aware else 0
        self.extended_obs_size = obs_size + (lag_window * lag_features) + position_features
        
        # Enhanced actor network
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
        
        # Logger
        self.logger = logging.getLogger("PPOLag")
        if not self.logger.handlers:
            fh = logging.FileHandler("logs/ppo_lag.log")
            fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(fh)
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def update_market_buffers(self, price: float, volume: float, spread: float, volatility: float):
        """Update lagged market features"""
        if len(self.price_buffer) > 1:
            last_price = self.price_buffer[-1]
            ret = (price - last_price) / last_price if last_price > 0 else 0
        else:
            ret = 0
            
        self.price_buffer.append(ret)
        self.volume_buffer.append(volume)
        self.spread_buffer.append(spread)
        self.vol_buffer.append(volatility)

    def get_lag_features(self) -> np.ndarray:
        """Extract lagged features"""
        price_lags = list(self.price_buffer) + [0] * (self.lag_window - len(self.price_buffer))
        volume_lags = list(self.volume_buffer) + [0] * (self.lag_window - len(self.volume_buffer))
        spread_lags = list(self.spread_buffer) + [0] * (self.lag_window - len(self.spread_buffer))
        vol_lags = list(self.vol_buffer) + [1] * (self.lag_window - len(self.vol_buffer))
        
        features = []
        for i in range(self.lag_window):
            features.extend([price_lags[i], vol_lags[i], volume_lags[i], spread_lags[i]])
            
        return np.array(features, dtype=np.float32)

    def forward(self, obs: torch.Tensor, market_lags: torch.Tensor):
        """Forward pass"""
        action_logits = self.actor(obs)
        
        value_features = self.value_encoder(obs)
        market_features = self.market_encoder(market_lags)
        combined = torch.cat([value_features, market_features], dim=-1)
        value = self.value_head(combined)
        
        return action_logits, value

    def record_step(self, obs_vec: np.ndarray, reward: float, 
                   price: float = 0, volume: float = 0, 
                   spread: float = 0, volatility: float = 1,
                   position: float = 0, unrealized_pnl: float = 0):
        """Record step with market data"""
        
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
                
            dist = torch.distributions.Normal(mu, action_std)
            action = dist.rsample()
            
            # Position-aware action scaling
            if self.position_aware and abs(position) > 0.8:
                action = action * (1 - abs(position))
                
            logp = dist.log_prob(action).sum(dim=-1)

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

    def compute_advantages(self, gamma: float = 0.99, lam: float = 0.95):
        """GAE with advantage smoothing"""
        rewards = torch.stack(self.buffer["rewards"])
        values = torch.stack(self.buffer["values"])
        
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
        
        # Smooth advantages
        adv_std = advantages.std()
        self.running_adv_std = self.adv_decay * self.running_adv_std + (1 - self.adv_decay) * adv_std
        advantages = advantages / (self.running_adv_std + 1e-8)
        
        return advantages, returns

    def end_episode(self, gamma: float = 0.99):
        """Episode ending with updates"""
        if len(self.buffer["rewards"]) < 10:
            for k in self.buffer:
                self.buffer[k].clear()
            return
            
        # Stack tensors
        obs = torch.stack(self.buffer["obs"])
        actions = torch.stack(self.buffer["actions"])
        logp_old = torch.stack(self.buffer["logp"])
        values_old = torch.stack(self.buffer["values"])
        market_features = torch.stack(self.buffer["market_features"])
        
        # Compute advantages
        advantages, returns = self.compute_advantages(gamma)
        
        # Multiple epochs
        for epoch in range(4):
            indices = torch.randperm(len(obs))
            
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
                
                # Update
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                
                self.actor_opt.step()
                self.critic_opt.step()
                
        # Log stats
        total_reward = sum(r.item() for r in self.buffer["rewards"])
        self.logger.info(f"Episode: Reward={total_reward:.2f}, Steps={len(self.buffer['rewards'])}")
        
        # Clear buffers
        for k in self.buffer:
            self.buffer[k].clear()

    def get_observation_components(self) -> np.ndarray:
        """Return 4 components for MetaRLController compatibility"""
        return np.array([
            float(self.last_action.mean()),
            float(self.last_action.std()),
            float(self.position),
            float(self.unrealized_pnl)
        ], dtype=np.float32)

    def select_action(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Action selection"""
        # Pad observation if needed
        if obs_tensor.shape[-1] < self.extended_obs_size:
            padding = torch.zeros(
                (*obs_tensor.shape[:-1], self.extended_obs_size - obs_tensor.shape[-1]),
                device=obs_tensor.device
            )
            obs_tensor = torch.cat([obs_tensor, padding], dim=-1)
            
        # Extract market features
        market_features = obs_tensor[..., -self.lag_window*4:]
        
        with torch.no_grad():
            action, _ = self.forward(obs_tensor, market_features)
            
        return action

    def reset(self):
        """Reset agent"""
        for k in self.buffer:
            self.buffer[k].clear()
        self.last_action = np.zeros_like(self.last_action)
        self.position = 0.0
        self.unrealized_pnl = 0.0
        self.price_buffer.clear()
        self.volume_buffer.clear()
        self.spread_buffer.clear()
        self.vol_buffer.clear()
        self.running_adv_std = 1.0

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
        }

    def set_state(self, state: Dict, strict: bool = False):
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
# STANDARD PPO (KEPT FOR COMPATIBILITY)
# ──────────────────────────────────────────────

class PPOAgent(nn.Module, Module):
    """
    Standard PPO agent - kept for comparison and fallback
    """
    def __init__(self, obs_size, act_size=2, hidden_size=64, lr=3e-4, device="cpu", debug=True):
        super().__init__()
        self.device = torch.device(device)
        self.debug = debug

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, act_size), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.to(self.device)
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.clip_eps = 0.2
        self.value_coeff = 0.5
        self.entropy_coeff = 0.01

        self.buffer = {k: [] for k in ["obs", "actions", "logp", "values", "rewards"]}
        self.last_action = np.zeros(act_size, dtype=np.float32)

    def forward(self, obs: torch.Tensor):
        mu = self.actor(obs)
        value = self.critic(obs)
        return mu, value

    def record_step(self, obs_vec, reward, **kwargs):
        """FIX: Accept market data kwargs for compatibility"""
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu, value = self.forward(obs_t.unsqueeze(0))
        dist = torch.distributions.Normal(mu, 0.1)
        action = dist.rsample()
        logp = dist.log_prob(action).sum(dim=-1)

        self.buffer["obs"].append(obs_t)
        self.buffer["actions"].append(action.squeeze(0))
        self.buffer["logp"].append(logp.squeeze(0))
        self.buffer["values"].append(value.squeeze(0))
        self.buffer["rewards"].append(
            torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        )
        self.last_action = action.cpu().numpy().squeeze(0)

    def end_episode(self, gamma=0.99):
        if not self.buffer["rewards"]: 
            return

        obs = torch.stack(self.buffer["obs"])
        actions = torch.stack(self.buffer["actions"])
        logp_old = torch.stack(self.buffer["logp"])
        values = torch.stack(self.buffer["values"])
        rewards = torch.stack(self.buffer["rewards"])

        returns = []
        R = 0.0
        for r in reversed(rewards.tolist()):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns - values

        for _ in range(4):
            mu, value = self.forward(obs)
            dist = torch.distributions.Normal(mu, 0.1)
            logp = dist.log_prob(actions).sum(dim=-1)
            ratio = (logp - logp_old).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value.squeeze(-1), returns)
            entropy_loss = -dist.entropy().mean()
            loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.opt.step()

        for k in self.buffer:
            self.buffer[k].clear()

    def get_observation_components(self):
        # FIX: Return 4 components for compatibility
        return np.array([
            float(self.last_action.mean()), 
            float(self.last_action.std()),
            0.0,  # Placeholder for position
            0.0   # Placeholder for unrealized_pnl
        ], dtype=np.float32)

    def get_state(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt": self.opt.state_dict(),
            "last_action": self.last_action.tolist(),
        }

    def set_state(self, state, strict=False):
        self.actor.load_state_dict(state["actor"], strict=strict)
        self.critic.load_state_dict(state["critic"], strict=strict)
        self.opt.load_state_dict(state["opt"])
        self.last_action = np.array(state.get("last_action", [0,0]), dtype=np.float32)

    def reset(self):
        for k in self.buffer: 
            self.buffer[k].clear()
        self.last_action = np.zeros_like(self.last_action)

    def step(self, *args, **kwargs):
        pass

    def select_action(self, obs_tensor):
        with torch.no_grad():
            action = self.actor(obs_tensor)
            return action

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
# META RL CONTROLLER (PPO and PPO-LAG ONLY)
# ──────────────────────────────────────────────

class MetaRLController(Module):
    """
    FIX: Simplified controller supporting only PPO and PPO-Lag
    """
    def __init__(self, obs_size: int, act_size: int=2, method="ppo-lag", 
                 device="cpu", debug=True, profit_target=150.0):
        self.device = device
        self.obs_size = obs_size
        self.act_size = act_size
        self.debug = debug
        self.profit_target = profit_target

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

        # Logger
        self.logger = logging.getLogger("MetaRLController")
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        if not self.logger.handlers:
            handler = logging.FileHandler("logs/MetaRLController.log")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

        self.logger.info(f"Initialized with {self.mode} agent for €{profit_target}/day target")

    def set_mode(self, method: str):
        """Switch between PPO variants"""
        if method not in self._agents:
            self.logger.warning(f"Unknown method: {method}, keeping {self.mode}")
            return
            
        old_mode = self.mode
        self.mode = method
        self.agent = self._agents[method]
        self.logger.info(f"Switched from {old_mode} to {self.mode}")

    def record_step(self, obs_vec, reward, **market_data):
        """
        FIX: Enhanced recording with market data support
        """
        self.total_profit += reward
        
        if self.mode == "ppo-lag":
            # PPO-Lag needs full market data
            self.agent.record_step(obs_vec, reward, **market_data)
        else:
            # Standard PPO
            self.agent.record_step(obs_vec, reward)
            
        # Log significant profits
        if reward > 10:  # €10+ single trade
            self.logger.info(f"Profitable trade: €{reward:.2f} using {self.mode}")

    def end_episode(self, *args, **kwargs):
        """End episode with performance tracking"""
        self.episode_count += 1
        
        # Check daily profit progress
        if self.total_profit > self.best_daily_profit:
            self.best_daily_profit = self.total_profit
            self.logger.info(f"New best daily profit: €{self.best_daily_profit:.2f}")
            
        # Switch algorithms if underperforming
        if self.episode_count % 50 == 0 and self.total_profit < self.profit_target * 0.5:
            new_mode = "ppo-lag" if self.mode == "ppo" else "ppo"
            self.logger.warning(f"Underperforming - switching to {new_mode}")
            self.set_mode(new_mode)
            
        return self.agent.end_episode(*args, **kwargs)

    def get_observation_components(self):
        """Always return 4-component vector"""
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
            
        return obs.astype(np.float32)

    def get_state(self):
        """Save controller state"""
        return {
            "mode": self.mode,
            "agents": {k: v.get_state() for k, v in self._agents.items()},
            "episode_count": self.episode_count,
            "total_profit": self.total_profit,
            "best_daily_profit": self.best_daily_profit,
        }

    def set_state(self, state, strict=False):
        """Load controller state"""
        self.mode = state.get("mode", self.mode)
        self.episode_count = state.get("episode_count", 0)
        self.total_profit = state.get("total_profit", 0.0)
        self.best_daily_profit = state.get("best_daily_profit", 0.0)
        
        for agent_name, agent_state in state["agents"].items():
            if agent_name in self._agents:
                self._agents[agent_name].set_state(agent_state, strict=strict)
                
        self.agent = self._agents[self.mode]

    def reset(self):
        """Reset all agents"""
        for agent in self._agents.values():
            agent.reset()
        self.logger.info("All agents reset")

    def act(self, obs_tensor):
        """Get action from active agent"""
        action = self.agent.select_action(obs_tensor)
        
        # Ensure valid numpy array
        action = np.asarray(action)
        
        # Check for NaN
        if np.isnan(action).any():
            self.logger.error(f"NaN in action: {action}")
            action = np.nan_to_num(action)
            
        return action

    def step(self, *args, **kwargs):
        """Compatibility method"""
        pass
    def obs_dim(self):
        return self.obs_size

    def save_checkpoint(self, filepath: str):
        """Save full checkpoint"""
        checkpoint = {
            "controller_state": self.get_state(),
            "timestamp": str(np.datetime64('now')),
            "profit_achieved": self.total_profit,
            "target": self.profit_target
        }
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """Load checkpoint"""
        checkpoint = torch.load(filepath)
        self.set_state(checkpoint["controller_state"])
        self.logger.info(f"Loaded checkpoint from {filepath}")

    def get_weights(self) -> Dict[str, Any]:
        """Get active agent weights"""
        if hasattr(self.agent, "get_weights"):
            return self.agent.get_weights()
        return {}

    def get_gradients(self) -> Dict[str, Any]:
        """Get active agent gradients"""
        if hasattr(self.agent, "get_gradients"):
            return self.agent.get_gradients()
        return {}