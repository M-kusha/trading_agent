from __future__ import annotations
import hashlib
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Callable, Optional
from collections import deque
from modules.core.core import Module

# ──────────────────────────────────────────────
class StrategyIntrospector(Module):
    def __init__(self, history_len: int = 10, debug: bool = False):
        self.history_len = history_len
        self.debug = debug
        self._records: List[Dict[str, float]] = []

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
            return np.zeros(5, dtype=np.float32)
        arr = np.array([[r["wr"], r["sl"], r["tp"]] for r in self._records], dtype=np.float32)
        flat = arr.flatten()[:5]
        if flat.size < 5:
            flat = np.pad(flat, (0, 5 - flat.size))
        return flat

    def get_observation_components(self) -> np.ndarray:
        return self.profile()

# ──────────────────────────────────────────────
class CurriculumPlannerPlus(Module):
    """
    If not actively adapting environment, serves as a performance tracker.
    """
    def __init__(self, window: int=10, debug=False):
        self.window   = window
        self.debug    = debug
        self._history: List[Dict[str,float]] = []

    def reset(self):
        self._history.clear()

    def step(self, **kwargs):
        pass

    def record_episode(self, summary: Dict[str,float]):
        self._history.append(summary)
        if len(self._history) > self.window:
            self._history.pop(0)

    def get_observation_components(self) -> np.ndarray:
        if not self._history:
            return np.zeros(3, dtype=np.float32)
        arr = np.array([
            [e.get("win_rate",0), e.get("avg_duration",0), e.get("avg_drawdown",0)]
            for e in self._history
        ], dtype=np.float32)
        return arr.mean(axis=0)



import hashlib
import logging
import numpy as np
from typing import Callable, List

class StrategyGenomePool:
    """
    Maintains a population of 4-parameter genomes:
    
    [sl_base, tp_base, vol_scale, regime_adapt]

    Evolves them by tournament selection + crossover + mutation, with
    diversity-preserving injection and parameter bounds.
    """

    def __init__(
        self,
        population_size: int = 20,
        tournament_k: int = 3,
        crossover_rate: float = 0.5,
        mutation_rate: float = 0.1,
        mutation_scale: float = 0.2,
        max_generations_kept: int = 10_000,
        debug: bool = False,
        log_file: str = "sgp.log",
    ) -> None:
        self.genome_size = 4
        self.pop_size = int(population_size)
        self.tournament_k = int(tournament_k)
        self.cx_rate = float(crossover_rate)
        self.mut_rate = float(mutation_rate)
        self.mut_scale = float(mutation_scale)
        self.max_generations_kept = int(max_generations_kept)
        self.debug = debug

        # Hand‐crafted seeds
        seeds = [
            np.array([1.0, 1.0, 0.5, 0.1], dtype=np.float32),
            np.array([2.0, 2.0, 1.0, 0.3], dtype=np.float32),
            np.array([0.8, 1.5, 1.8, 0.2], dtype=np.float32),
        ]
        # Random fill across full intended range
        rand = [
            np.random.uniform(
                low=[0.3, 0.3, 0.3, 0.01],
                high=[3.0, 3.0, 2.5, 0.5],
            ).astype(np.float32)
            for _ in range(self.pop_size - len(seeds))
        ]
        self.population = np.vstack(seeds + rand)[: self.pop_size]
        self.fitness = np.zeros(self.pop_size, dtype=np.float32)
        self.epoch = 0

        # Logger for this class → write to file instead of console
        self.logger = logging.getLogger("SGP")
        if not self.logger.handlers:
            # Remove any default handlers (if present)
            for h in list(self.logger.handlers):
                self.logger.removeHandler(h)

            # Create a FileHandler
            fh = logging.FileHandler(log_file)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        # If you still want console output too, uncomment below:
        # ch = logging.StreamHandler()
        # ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        # self.logger.addHandler(ch)

        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

    def reset(self) -> None:
        self.epoch = 0
        self.fitness[:] = 0.0

    def step(self, **kwargs) -> None:
        # No per-step logic — evolution triggers in evolve_strategies()
        pass

    def genome_hash(self, g: np.ndarray) -> str:
        """Compute an MD5 hash of the genome's raw bytes for quick equality checks."""
        return hashlib.md5(g.tobytes()).hexdigest()

    def evaluate_population(self, eval_fn: Callable[[np.ndarray], float]) -> None:
        """
        eval_fn(genome: np.ndarray) -> float  (higher = better)
        """
        for i, genome in enumerate(self.population):
            self.fitness[i] = float(eval_fn(genome))
        if self.debug:
            self.logger.debug(
                f"[SGP] Gen {self.epoch} fitness "
                f"{self.fitness.min():.3f} – {self.fitness.max():.3f}"
            )

    def evolve_strategies(self) -> None:
        """
        Tournament-select parents, crossover, mutate, then inject diversity
        if fitness has stagnated.
        """
        # 1) Hash the old population for change detection
        old_hashes = [self.genome_hash(g) for g in self.population]

        # 2) Diversity preservation: if fitness variance is tiny, inject new random genomes
        if np.std(self.fitness) < 0.1:
            # Inject 2 fresh genomes across full intended band
            new_genes = np.random.uniform(
                low=[0.3, 0.3, 0.3, 0.01],
                high=[3.0, 3.0, 2.5, 0.5],
                size=(2, 4),
            ).astype(np.float32)
            if self.debug:
                for idx, g in enumerate(new_genes):
                    g_str = ", ".join(f"{x:.3f}" for x in g)
                    self.logger.debug(f"[SGP] Injected new random genome #{idx}: [{g_str}]")
        else:
            new_genes = np.zeros((0, 4), dtype=np.float32)

        # 3) Build the next generation
        new_pop: List[np.ndarray] = []
        n_to_generate = self.pop_size - new_genes.shape[0]
        for _ in range(n_to_generate):
            # 3a) Parent 1 via tournament
            cand = np.random.choice(self.pop_size, self.tournament_k, replace=False)
            p1 = self.population[cand[np.argmax(self.fitness[cand])]]
            # 3b) Parent 2 via tournament
            cand = np.random.choice(self.pop_size, self.tournament_k, replace=False)
            p2 = self.population[cand[np.argmax(self.fitness[cand])]]

            # 3c) Crossover
            mask = np.random.rand(self.genome_size) < self.cx_rate
            child = np.where(mask, p1, p2).copy()

            # 3d) Mutation
            m_idx = np.random.rand(self.genome_size) < self.mut_rate
            if m_idx.any():
                child_before = child.copy()
                child[m_idx] += np.random.randn(m_idx.sum()) * self.mut_scale
                # Clip to hard bounds
                child = np.clip(
                    child,
                    a_min=[0.01, 0.01, 0.01, 0.0],
                    a_max=[5.0, 5.0, 5.0, 1.0],
                )
                # Per-parameter tighter bounds
                child[2] = np.clip(child[2], 0.1, 3.0)  # vol_scale
                child[3] = np.clip(child[3], 0.0, 0.8)  # regime_adapt

                if not np.array_equal(child_before, child):
                    before_str = ", ".join(f"{x:.3f}" for x in child_before)
                    after_str = ", ".join(f"{x:.3f}" for x in child)
                    self.logger.info(
                        f"[SGP] Mutated child from [{before_str}] → [{after_str}]"
                    )
            new_pop.append(child.astype(np.float32))

        # 4) If we injected fresh genomes, tack them on
        if new_genes.shape[0] > 0:
            assembled = np.vstack(new_pop + [new_genes])
            self.population = assembled[: self.pop_size]
        else:
            self.population = np.vstack(new_pop)

        # 5) Reset fitness and bump epoch
        self.fitness[:] = 0.0
        self.epoch += 1
        if self.epoch > self.max_generations_kept:
            self.epoch = 0  # simple cap to avoid int overflow

        # 6) Log how many genomes actually changed this generation
        new_hashes = [self.genome_hash(g) for g in self.population]
        n_changed = sum(1 for o, n in zip(old_hashes, new_hashes) if o != n)
        if n_changed > 0:
            self.logger.info(f"[SGP] {n_changed}/{self.pop_size} genomes changed in generation {self.epoch}")
        elif self.debug:
            self.logger.debug(f"[SGP] No genomes changed in generation {self.epoch}")

    def select_genome(self, mode="random", k=3, custom_selector=None):
        """
        Select a genome according to the specified mode:
        - "random": uniform random selection.
        - "best":   highest fitness.
        - "tournament": among k random genomes, pick the best.
        - "roulette": weighted by fitness (proportional).
        - "custom":  use a custom selector function.

        Returns:
            genome (np.ndarray)
        """
        assert self.population.shape[0] == self.fitness.shape[0], "Population/fitness size mismatch"
        N = self.population.shape[0]

        if mode == "random":
            idx = np.random.randint(N)

        elif mode == "best":
            idx = int(np.argmax(self.fitness))

        elif mode == "tournament":
            candidates = np.random.choice(N, k, replace=False)
            idx = candidates[np.argmax(self.fitness[candidates])]

        elif mode == "roulette":
            # Fitness must be non-negative; add a small offset if any are negative
            fit = self.fitness - np.min(self.fitness) + 1e-8
            probs = fit / fit.sum() if fit.sum() > 0 else np.ones(N) / N
            idx = np.random.choice(N, p=probs)

        elif mode == "custom":
            assert custom_selector is not None, "Provide custom_selector callable!"
            idx = custom_selector(self.population, self.fitness)

        else:
            raise ValueError(f"Unknown selection mode: {mode}")

        self.active_genome = self.population[idx].copy()
        self.active_genome_idx = idx  # Track for potential logging/debug

        if self.debug:
            fit_val = self.fitness[idx]
            genome_str = ", ".join(f"{x:.3f}" for x in self.active_genome)
            self.logger.debug(f"[SGP] Selected genome idx={idx}, fitness={fit_val:.3f}, genome=[{genome_str}] (mode={mode})")
        return self.active_genome

    def get_observation_components(self) -> np.ndarray:
        """
        Returns [mean_fitness, max_fitness, diversity],
        where diversity = average pairwise Euclidean distance.
        """
        mean_f = float(self.fitness.mean())
        max_f = float(self.fitness.max())
        if self.pop_size > 1:
            P = self.population.astype(np.float32)  # (N,4)
            dists = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
            diversity = float(dists.mean())
        else:
            diversity = 0.0
        return np.array([mean_f, max_f, diversity], dtype=np.float32)

    def get_state(self) -> dict:
        return {
            "population": self.population,
            "fitness": self.fitness,
            "epoch": self.epoch,
        }

    def set_state(self, state: dict) -> None:
        self.population = state.get("population", self.population)
        self.fitness = state.get("fitness", self.fitness)
        self.epoch = state.get("epoch", self.epoch)


# ──────────────────────────────────────────────
class MetaAgent(Module):
    def __init__(self, window: int=20, debug=False):
        self.window = window
        self.debug  = debug
        self.reset()

    def reset(self):
        self.history: List[float] = []

    def step(self, pnl: float=0.0):
        self.history.append(pnl)
        if len(self.history) > self.window:
            self.history.pop(0)

    def record(self, pnl: float):
        self.step(pnl)

    def get_observation_components(self)->np.ndarray:
        if not self.history:
            return np.zeros(2, dtype=np.float32)
        arr = np.array(self.history, dtype=np.float32)
        return np.array([arr.mean(), arr.std()], dtype=np.float32)
    
    def get_intensity(self, instrument: str) -> float:
        if not self.history:
            intensity = np.random.uniform(-0.3, 0.3)
            if self.debug:
                print(f"[MetaAgent] No history yet — emitting bootstrapped intensity: {intensity:.3f}")
            return float(intensity)
        avg_pnl = np.mean(self.history[-self.window:])
        scale = 0.01 * self.window
        intensity = np.clip(avg_pnl / scale, -1.0, 1.0)
        if self.debug:
            print(f"[MetaAgent] Intensity for {instrument}: {intensity:.3f} (avg_pnl={avg_pnl:.3f})")
        return float(intensity)

# ──────────────────────────────────────────────
class MetaCognitivePlanner(Module):
    def __init__(self, window: int=20, debug=False):
        self.window = window
        self.debug  = debug
        self.reset()

    def reset(self):
        self.history: List[Dict[str,float]] = []

    def step(self, **kwargs): pass

    def record_episode(self, result: Dict[str,float]):
        self.history.append(result)
        if len(self.history) > self.window:
            self.history.pop(0)

    def get_observation_components(self)->np.ndarray:
        if not self.history:
            return np.zeros(3, dtype=np.float32)
        pnls = np.array([r.get("pnl",0) for r in self.history], dtype=np.float32)
        win_rate = float((pnls>0).sum() / len(pnls)) if len(pnls)>0 else 0.0
        return np.array([win_rate, float(pnls.mean()), float(np.abs(pnls).mean())], dtype=np.float32)

# ──────────────────────────────────────────────
class BiasAuditor(Module):
    def __init__(self, history_len: int=100, debug=False):
        self.history_len = history_len
        self.debug       = debug
        self.reset()

    def reset(self):
        self.hist = deque(maxlen=self.history_len)

    def step(self, **kwargs): pass

    def record(self, bias: str):
        self.hist.append(bias)

    def get_observation_components(self)->np.ndarray:
        total = len(self.hist)
        if total == 0:
            return np.zeros(3, dtype=np.float32)
        cnt = {"revenge":0,"fear":0,"greed":0}
        for b in self.hist:
            if b in cnt: cnt[b]+=1
        freqs = np.array([cnt["revenge"],cnt["fear"],cnt["greed"]],dtype=np.float32)/total
        return freqs

# ──────────────────────────────────────────────
class OpponentModeEnhancer(Module):
    def __init__(self, modes: List[str]=None, debug=False):
        self.modes = modes or ["random","shock","reversal"]
        self.debug = debug
        self.reset()

    def reset(self):
        self.pnl = {m:0.0 for m in self.modes}

    def step(self, **kwargs): pass

    def record_result(self, mode: str, pnl: float):
        if mode in self.pnl:
            self.pnl[mode] += pnl

    def get_observation_components(self)-> np.ndarray:
        vals = np.array([self.pnl[m] for m in self.modes],dtype=np.float32)
        inv  = -vals; inv -= inv.min()
        w    = inv/inv.sum() if inv.sum()>0 else np.ones_like(inv)/len(inv)
        return w

# ──────────────────────────────────────────────
class ThesisEvolutionEngine(Module):
    def __init__(self, capacity: int=20, debug=False):
        self.capacity = capacity
        self.debug    = debug
        self.reset()

    def reset(self):
        self.theses: List[Any] = []
        self.pnls:   List[float] = []

    def step(self, **kwargs): pass

    def record_thesis(self, thesis: Any):
        self.theses.append(thesis)

    def record_pnl(self, pnl: float):
        if self.theses:
            self.pnls.append(pnl)
            if len(self.pnls) > self.capacity:
                self.pnls.pop(0)

    def get_observation_components(self)->np.ndarray:
        if not self.pnls:
            return np.zeros(3, dtype=np.float32)
        uniq = len(set(self.theses))
        mean_p = float(np.mean(self.pnls))
        sd_p   = float(np.std(self.pnls))
        return np.array([uniq, mean_p, sd_p], dtype=np.float32)

# ──────────────────────────────────────────────
class ExplanationGenerator(Module):
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.last_explanation = ""

    def reset(self):
        self.last_explanation = ""

    def step(
        self,
        actions: np.ndarray,
        arbiter_weights: np.ndarray,
        member_names: List[str],
        *,
        regime: str,
        volatility: Dict[str, float],
        drawdown: float,
        genome_metrics: Dict[str, float],
    ):
        top_idx  = int(np.argmax(arbiter_weights))
        top_name = member_names[top_idx]
        top_w    = float(arbiter_weights[top_idx]) * 100

        breakdown = "; ".join(
            f"{n}: {w*100:.1f}%" for n, w in zip(member_names, arbiter_weights)
        )

        vol_str = ", ".join(
            f"{inst} vol={vol:.2f}" for inst, vol in volatility.items()
        )

        sl = genome_metrics.get("sl_base", 0.0)
        tp = genome_metrics.get("tp_base", 0.0)
        dd_pct = drawdown * 100

        self.last_explanation = (
            f"Market regime: {regime} ({vol_str}), drawdown={dd_pct:.2f}%. "
            f"Vote breakdown — {breakdown}. "
            f"Strategy genome used: SL_base={sl:.2f}, TP_base={tp:.2f}. "
            f"Final action is a weighted blend; top influencer was “{top_name}” at {top_w:.1f}%."
        )

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


# ──────────────────────────────────────────────
class PPOAgent(nn.Module, Module):
    def __init__(self, obs_size, act_size=2, hidden_size=64, lr=3e-4, device="cpu", debug=False):
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

    def record_step(self, obs_vec, reward):
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
        if not self.buffer["rewards"]: return

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
        return np.array([float(self.last_action.mean()), float(self.last_action.std())], dtype=np.float32)

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
        for k in self.buffer: self.buffer[k].clear()
        self.last_action = np.zeros_like(self.last_action)
    def step(self, *args, **kwargs):
        pass
    def select_action(self, obs_tensor):
        with torch.no_grad():
            action = self.actor(obs_tensor)
            return action


# ──────────────────────────────────────────────
class SACAgent(nn.Module, Module):
    def __init__(self, obs_size, act_size=2, hidden_size=64, lr=3e-4, alpha=0.2, device="cpu", debug=False):
        super().__init__()
        self.device = torch.device(device)
        self.debug = debug

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, act_size), nn.Tanh()
        )
        self.critic1 = nn.Sequential(
            nn.Linear(obs_size+act_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(obs_size+act_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.target_critic1 = nn.Sequential(
            nn.Linear(obs_size+act_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.target_critic2 = nn.Sequential(
            nn.Linear(obs_size+act_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.to(self.device)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)
        self.alpha = alpha
        self.gamma = 0.99
        self.tau = 0.005

        # Buffer
        self.replay_buffer = []
        self.buffer_limit = 10000
        self.batch_size = 32
        self.last_action = np.zeros(act_size, dtype=np.float32)

    def record_step(self, obs_vec, reward, done=False, next_obs=None):
        if next_obs is not None:
            self.replay_buffer.append((
                np.array(obs_vec, dtype=np.float32),
                self.last_action.copy(),
                reward,
                np.array(next_obs, dtype=np.float32),
                done
            ))
            if len(self.replay_buffer) > self.buffer_limit:
                self.replay_buffer.pop(0)
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(obs_t)
        self.last_action = action.cpu().numpy()

    def end_episode(self, gamma=0.99):
        if len(self.replay_buffer) < self.batch_size: return
        # Sample batch
        idx = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in idx]
        obs, act, rew, next_obs, done = zip(*batch)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.float32, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Target action and Q
        with torch.no_grad():
            next_act = self.actor(next_obs)
            next_cat = torch.cat([next_obs, next_act], dim=1)
            q1_next = self.target_critic1(next_cat)
            q2_next = self.target_critic2(next_cat)
            min_q_next = torch.min(q1_next, q2_next)
            target = rew + (1-done) * self.gamma * (min_q_next - self.alpha * 0) # no logp for tanh policy

        # Update critics
        act_cat = torch.cat([obs, act], dim=1)
        q1 = self.critic1(act_cat)
        q2 = self.critic2(act_cat)
        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)
        self.critic1_opt.zero_grad(); critic1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); critic2_loss.backward(); self.critic2_opt.step()

        # Update actor
        new_act = self.actor(obs)
        act_cat = torch.cat([obs, new_act], dim=1)
        actor_loss = -self.critic1(act_cat).mean()
        self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

        # Soft update targets
        for target, main in zip(
            [self.target_critic1, self.target_critic2], [self.critic1, self.critic2]
        ):
            for t, m in zip(target.parameters(), main.parameters()):
                t.data.copy_(t.data * (1-self.tau) + m.data * self.tau)

    def get_observation_components(self):
        return np.array([float(self.last_action.mean()), float(self.last_action.std())], dtype=np.float32)

    def get_state(self):
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "last_action": self.last_action.tolist(),
        }

    def set_state(self, state, strict=False):
        self.actor.load_state_dict(state["actor"], strict=strict)
        self.critic1.load_state_dict(state["critic1"], strict=strict)
        self.critic2.load_state_dict(state["critic2"], strict=strict)
        self.target_critic1.load_state_dict(state["target_critic1"], strict=strict)
        self.target_critic2.load_state_dict(state["target_critic2"], strict=strict)
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic1_opt.load_state_dict(state["critic1_opt"])
        self.critic2_opt.load_state_dict(state["critic2_opt"])
        self.last_action = np.array(state.get("last_action", [0,0]), dtype=np.float32)

    def reset(self):
        self.last_action = np.zeros_like(self.last_action)
    def step(self, *args, **kwargs):
        pass
    def select_action(self, obs_tensor):
        with torch.no_grad():
            action = self.actor(obs_tensor)
            return action

    def select_action(self, obs_tensor):
        with torch.no_grad():
            action = self.actor(obs_tensor)
            return action

# ──────────────────────────────────────────────
class TD3Agent(nn.Module, Module):
    def __init__(self, obs_size, act_size=2, hidden_size=64, lr=3e-4, device="cpu", debug=False):
        super().__init__()
        self.device = torch.device(device)
        self.debug = debug

        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, act_size), nn.Tanh()
        )
        self.critic1 = nn.Sequential(
            nn.Linear(obs_size+act_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.critic2 = nn.Sequential(
            nn.Linear(obs_size+act_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.target_actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, act_size), nn.Tanh()
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic1 = nn.Sequential(
            nn.Linear(obs_size+act_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.target_critic2 = nn.Sequential(
            nn.Linear(obs_size+act_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        self.to(self.device)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=lr)
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.total_it = 0

        self.replay_buffer = []
        self.buffer_limit = 10000
        self.batch_size = 32
        self.last_action = np.zeros(act_size, dtype=np.float32)

    def record_step(self, obs_vec, reward, done=False, next_obs=None):
        if next_obs is not None:
            self.replay_buffer.append((
                np.array(obs_vec, dtype=np.float32),
                self.last_action.copy(),
                reward,
                np.array(next_obs, dtype=np.float32),
                done
            ))
            if len(self.replay_buffer) > self.buffer_limit:
                self.replay_buffer.pop(0)
        obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action = self.actor(obs_t)
        self.last_action = action.cpu().numpy()

    def end_episode(self, gamma=0.99):
        if len(self.replay_buffer) < self.batch_size: return
        self.total_it += 1
        idx = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in idx]
        obs, act, rew, next_obs, done = zip(*batch)
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)
        act = torch.tensor(act, dtype=torch.float32, device=self.device)
        rew = torch.tensor(rew, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        done = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Target action with noise
        with torch.no_grad():
            noise = (torch.randn_like(act) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_act = (self.target_actor(next_obs) + noise).clamp(-1, 1)
            next_cat = torch.cat([next_obs, next_act], dim=1)
            q1_next = self.target_critic1(next_cat)
            q2_next = self.target_critic2(next_cat)
            min_q_next = torch.min(q1_next, q2_next)
            target = rew + (1-done) * self.gamma * min_q_next

        # Critic update
        act_cat = torch.cat([obs, act], dim=1)
        q1 = self.critic1(act_cat)
        q2 = self.critic2(act_cat)
        critic1_loss = F.mse_loss(q1, target)
        critic2_loss = F.mse_loss(q2, target)
        self.critic1_opt.zero_grad(); critic1_loss.backward(); self.critic1_opt.step()
        self.critic2_opt.zero_grad(); critic2_loss.backward(); self.critic2_opt.step()

        # Delayed actor update
        if self.total_it % self.policy_delay == 0:
            actor_act = self.actor(obs)
            act_cat = torch.cat([obs, actor_act], dim=1)
            actor_loss = -self.critic1(act_cat).mean()
            self.actor_opt.zero_grad(); actor_loss.backward(); self.actor_opt.step()

            # Soft update targets
            for target, main in zip(
                [self.target_actor, self.target_critic1, self.target_critic2],
                [self.actor, self.critic1, self.critic2]
            ):
                for t, m in zip(target.parameters(), main.parameters()):
                    t.data.copy_(t.data * (1-self.tau) + m.data * self.tau)

    def get_observation_components(self):
        return np.array([float(self.last_action.mean()), float(self.last_action.std())], dtype=np.float32)

    def get_state(self):
        return {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic1_opt": self.critic1_opt.state_dict(),
            "critic2_opt": self.critic2_opt.state_dict(),
            "last_action": self.last_action.tolist(),
            "total_it": self.total_it,
        }

    def set_state(self, state, strict=False):
        self.actor.load_state_dict(state["actor"], strict=strict)
        self.critic1.load_state_dict(state["critic1"], strict=strict)
        self.critic2.load_state_dict(state["critic2"], strict=strict)
        self.target_actor.load_state_dict(state["target_actor"], strict=strict)
        self.target_critic1.load_state_dict(state["target_critic1"], strict=strict)
        self.target_critic2.load_state_dict(state["target_critic2"], strict=strict)
        self.actor_opt.load_state_dict(state["actor_opt"])
        self.critic1_opt.load_state_dict(state["critic1_opt"])
        self.critic2_opt.load_state_dict(state["critic2_opt"])
        self.last_action = np.array(state.get("last_action", [0,0]), dtype=np.float32)
        self.total_it = state.get("total_it", 0)

    def reset(self):
        self.last_action = np.zeros_like(self.last_action)
    def step(self, *args, **kwargs):
        pass
    def select_action(self, obs_tensor):
        with torch.no_grad():
            action = self.actor(obs_tensor)
            return action

    # ──────────────────────────────────────────────
class MetaRLController(Module):
    """
    Switchable controller for PPO, SAC, TD3 with shared API.
    """
    def __init__(self, obs_size: int, act_size: int=2, method="sac", device="cpu", debug=False):
        self.device = device
        self.obs_size = obs_size
        self.act_size = act_size
        self.debug = debug

        self._agents = {
            "ppo": PPOAgent(obs_size, act_size=act_size, device=device, debug=debug),
            "sac": SACAgent(obs_size, act_size=act_size, device=device, debug=debug),
            "td3": TD3Agent(obs_size, act_size=act_size, device=device, debug=debug),
        }
        self.mode = method
        self.agent = self._agents[self.mode]

    def set_mode(self, method: str):
        assert method in self._agents, f"Unknown method: {method}"
        self.mode = method
        self.agent = self._agents[method]

    def record_step(self, *args, **kwargs):
        return self.agent.record_step(*args, **kwargs)

    def end_episode(self, *args, **kwargs):
        return self.agent.end_episode(*args, **kwargs)

    def get_observation_components(self):
        return self.agent.get_observation_components()

    def get_state(self):
        return {
            "mode": self.mode,
            "agents": {k: v.get_state() for k, v in self._agents.items()}
        }

    def set_state(self, state, strict=False):
        self.mode = state.get("mode", self.mode)
        for k, v in state["agents"].items():
            if k in self._agents:
                self._agents[k].set_state(v, strict=strict)
        self.agent = self._agents[self.mode]

    def reset(self):
        for agent in self._agents.values():
            agent.reset()
    def step(self, *args, **kwargs):   # <--- add this method
        pass



    def act(self, obs_tensor):
        return self.agent.select_action(obs_tensor)
