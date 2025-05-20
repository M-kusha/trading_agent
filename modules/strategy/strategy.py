# modules/strategy.py
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Callable
from collections import deque
from typing import Callable, List
from modules.core.core import Module



# ───────────────────────────────────────────────────────────────────────────────
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


class CurriculumPlannerPlus(Module):
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


# ───────────────────────────────────────────────────────────────────────────────
# modules/strategy.py (only the StrategyGenomePool part)

class StrategyGenomePool(Module):
    """
    Maintains a population of 4‑parameter genomes:

        [sl_base, tp_base, vol_scale, regime_adapt]

    Evolves them by tournament selection + crossover + mutation, with
    diversity‑preserving injection and parameter bounds.

    *NEW*: Optional hard cap (`max_generations_kept`) so the population
           cannot silently explode if you tweak `evolve_strategies`.
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
    ) -> None:
        self.genome_size = 4
        self.pop_size = int(population_size)
        self.tournament_k = int(tournament_k)
        self.cx_rate = float(crossover_rate)
        self.mut_rate = float(mutation_rate)
        self.mut_scale = float(mutation_scale)
        self.max_generations_kept = int(max_generations_kept)
        self.debug = debug

        # Hand‑crafted seeds
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

    # ------------------------------------------------------------------ API #
    def reset(self) -> None:
        self.epoch = 0
        self.fitness[:] = 0.0

    def step(self, **kwargs) -> None:
        # No per‑step logic — evolution triggers in evolve_strategies()
        pass

    def evaluate_population(self, eval_fn: Callable[[np.ndarray], float]) -> None:
        """
        eval_fn(genome: np.ndarray) -> float  (higher = better)
        """
        for i, genome in enumerate(self.population):
            self.fitness[i] = float(eval_fn(genome))
        if self.debug:
            print(
                f"[SGP] Gen {self.epoch} fitness "
                f"{self.fitness.min():.3f} – {self.fitness.max():.3f}"
            )

    def evolve_strategies(self) -> None:
        """
        Tournament‑select parents, crossover, mutate, then inject diversity
        if fitness has stagnated.
        """
        # ── Diversity preservation ─────────────────────────────────────────
        if np.std(self.fitness) < 0.1:
            # Inject 2 fresh genomes across full intended band
            new_genes = np.random.uniform(
                low=[0.3, 0.3, 0.3, 0.01],
                high=[3.0, 3.0, 2.5, 0.5],
                size=(2, 4),
            ).astype(np.float32)
        else:
            new_genes = np.zeros((0, 4), dtype=np.float32)

        new_pop: List[np.ndarray] = []
        n_to_generate = self.pop_size - new_genes.shape[0]

        for _ in range(n_to_generate):
            # Parent 1 via tournament
            cand = np.random.choice(self.pop_size, self.tournament_k, replace=False)
            p1 = self.population[cand[np.argmax(self.fitness[cand])]]
            # Parent 2 via tournament
            cand = np.random.choice(self.pop_size, self.tournament_k, replace=False)
            p2 = self.population[cand[np.argmax(self.fitness[cand])]]

            # Crossover
            mask = np.random.rand(self.genome_size) < self.cx_rate
            child = np.where(mask, p1, p2).copy()

            # Mutation
            m_idx = np.random.rand(self.genome_size) < self.mut_rate
            child[m_idx] += np.random.randn(m_idx.sum()) * self.mut_scale

            # Clip to hard bounds
            child = np.clip(
                child,
                a_min=[0.01, 0.01, 0.01, 0.0],
                a_max=[5.0, 5.0, 5.0, 1.0],
            )
            # Per‑parameter tighter bounds
            child[2] = np.clip(child[2], 0.1, 3.0)  # vol_scale
            child[3] = np.clip(child[3], 0.0, 0.8)  # regime_adapt
            new_pop.append(child.astype(np.float32))

        # Assemble next generation
        if new_genes.shape[0] > 0:
            self.population = np.vstack(new_pop + [new_genes])[: self.pop_size]
        else:
            self.population = np.vstack(new_pop)

        # Reset fitness and bump epoch
        self.fitness[:] = 0.0
        self.epoch += 1
        if self.epoch > self.max_generations_kept:
            self.epoch = 0  # simple cap to avoid int overflow

        if self.debug:
            print(f"[SGP] Evolved to generation {self.epoch}")

    # ---------- observation hook ------------------------------------------ #
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




# ───────────────────────────────────────────────────────────────────────────────
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
        pnls = np.array([r["pnl"] for r in self.history], dtype=np.float32)
        win_rate = float((pnls>0).sum() / len(pnls))
        return np.array([win_rate, float(pnls.mean()), float(np.abs(pnls).mean())], dtype=np.float32)


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


# modules/strategy.py (just the ExplanationGenerator at the bottom)


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
        # 1) top influencer
        top_idx  = int(np.argmax(arbiter_weights))
        top_name = member_names[top_idx]
        top_w    = float(arbiter_weights[top_idx]) * 100

        # 2) vote breakdown
        breakdown = "; ".join(
            f"{n}: {w*100:.1f}%" for n, w in zip(member_names, arbiter_weights)
        )

        # 3) vol string for both instruments
        vol_str = ", ".join(
            f"{inst} vol={vol:.2f}" for inst, vol in volatility.items()
        )

        # 4) genome SL/TP
        sl = genome_metrics.get("sl_base", 0.0)
        tp = genome_metrics.get("tp_base", 0.0)

        # 5) drawdown %
        dd_pct = drawdown * 100

        # 6) compile explanation
        self.last_explanation = (
            f"Market regime: {regime} ({vol_str}), drawdown={dd_pct:.2f}%. "
            f"Vote breakdown — {breakdown}. "
            f"Strategy genome used: SL_base={sl:.2f}, TP_base={tp:.2f}. "
            f"Final action is a weighted blend; top influencer was “{top_name}” at {top_w:.1f}%."
        )

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


# ───────────────────────────────────────────────────────────────────────────────
class MetaRLController(nn.Module, Module):
    """
    Lightweight PPO-style meta-controller that uses genome & liquidity
    to scale its action outputs.
    """

    def __init__(
        self,
        obs_size: int,
        hidden_size: int = 64,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        device: str = "cpu",
        debug: bool = False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.debug  = debug

        # Actor: outputs [delta_sl, delta_tp]
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
            nn.Tanh(),
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        self.to(self.device)
        self.opt            = optim.Adam(self.parameters(), lr=lr)
        self.clip_eps       = clip_eps
        self.value_coeff    = value_coeff
        self.entropy_coeff  = entropy_coeff

        # Rollout buffer
        self.buffer = {
            "obs": [], "actions": [], "logp": [], "values": [], "rewards": []
        }

    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._forward_dist(obs)

    def record_step(self, obs_vec: np.ndarray, reward: float):
            # send obs directly to the right device
            obs_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                out = self._forward_dist(obs_t.unsqueeze(0))

            # store rollouts
            self.buffer["obs"].append(obs_t)
            self.buffer["actions"].append(out["action"].squeeze(0))
            self.buffer["logp"].append(out["logp"].squeeze(0))
            self.buffer["values"].append(out["value"].squeeze(0))
            self.buffer["rewards"].append(
                torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            )

    def end_episode(self, gamma: float = 0.99):
        if not self.buffer["rewards"]:
            return

        obs      = torch.stack(self.buffer["obs"])
        actions  = torch.stack(self.buffer["actions"])
        logp_old = torch.stack(self.buffer["logp"])
        values   = torch.stack(self.buffer["values"])
        rewards  = torch.stack(self.buffer["rewards"])

        # Compute returns & advantages
        returns = []
        R = 0.0
        for r in reversed(rewards.tolist()):
            R = r + gamma * R
            returns.insert(0, R)
        returns    = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns - values

        batch = {
            "obs": obs, "actions": actions,
            "logp_old": logp_old, "advantages": advantages, "returns": returns
        }
        self._ppo_update(batch)

        # Clear buffer
        for k in self.buffer:
            self.buffer[k].clear()

    # Aliases
    store_step     = record_step
    finish_episode = end_episode

    def _forward_dist(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1) Extract genome params
        genome = obs[...,:4]  # [sl_base,tp_base,vol_scale,regime_adapt]
        # 2) Base action
        mu_base = self.actor(obs)
        # 3) Regime scaling
        regime_scale = 0.1 + genome[...,3].unsqueeze(-1) * 0.3
        # 4) Liquidity factor (assumes last obs dim is liquidity score)
        liquidity_factor = obs[..., -1].unsqueeze(-1).clamp(min=0.1, max=1.0)
        scale = regime_scale * liquidity_factor
        mu    = mu_base * scale

        dist = torch.distributions.Normal(mu, 0.1)
        a    = dist.rsample()
        return {
            "action": a,
            "logp":   dist.log_prob(a).sum(dim=-1),
            "value":  self.critic(obs).squeeze(-1)
        }

    def _ppo_update(self, batch: Dict[str, torch.Tensor], epochs: int = 4):
        obs, actions_old = batch["obs"], batch["actions"]
        logp_old, adv, ret = batch["logp_old"], batch["advantages"], batch["returns"]

        for _ in range(epochs):
            out = self._forward_dist(obs)
            logp = out["logp"]
            dist = torch.distributions.Normal(out["action"], 0.1)
            entropy_loss = -dist.entropy().mean()

            value = out["value"]
            ratio = (logp - logp_old).exp()
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * adv

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss  = F.mse_loss(value, ret)
            loss        = policy_loss + self.value_coeff*value_loss + self.entropy_coeff*entropy_loss

            self.opt.zero_grad()
            loss.backward()
            # ─── gradient clipping ───────────────────────────────────────────
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.opt.step()

    def reset(self):
        for k in self.buffer:
            self.buffer[k].clear()

    def step(self, **kwargs):
        pass

    def get_observation_components(self) -> torch.Tensor:
        return torch.zeros(0, dtype=torch.float32)