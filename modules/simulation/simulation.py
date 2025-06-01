# modules/simulation.py
import numpy as np
from typing import List, Any, Dict
from modules.core.core import Module
import random

class ShadowSimulator(Module):
    """
    Simulates shadow trades over a planning horizon.
    Evolutionary: horizon and simulation mode can mutate/crossover.
    """
    MODES = ["greedy", "random", "conservative"]

    def __init__(self, horizon: int = 5, mode: str = "greedy", debug: bool = False):
        self.horizon = int(horizon)
        self.mode = mode if mode in self.MODES else "greedy"
        self.debug = debug
        self.state = {}

    def reset(self):
        self.state = {}

    def step(self, **kwargs):
        # Not used in this stateless module
        pass

    def simulate(self, env, actions: np.ndarray) -> List[dict]:
        """
        Simulate sequence of trades over the next `horizon` steps.
        Does not modify the real env.
        """
        if env is None or actions is None:
            return []
        # Save env state
        state_backup = env.save_state() if hasattr(env, "save_state") else (env.current_step, env.balance)
        trades = []
        for _ in range(self.horizon):
            if env.current_step >= len(next(iter(env.data.values()))["D1"]):
                break
            for i, instr in enumerate(env.instruments):
                # Vary action per mode
                if self.mode == "greedy":
                    act = actions[2 * i: 2 * i + 2]
                elif self.mode == "random":
                    act = np.random.uniform(-1, 1, size=2)
                elif self.mode == "conservative":
                    act = actions[2 * i: 2 * i + 2] * 0.5
                else:
                    act = actions[2 * i: 2 * i + 2]
                tr = env._execute_trade(instr, act[0], act[1])
                if tr:
                    trades.append(tr)
            env.current_step += 1
        # Restore env state
        if hasattr(env, "load_state"):
            env.load_state(state_backup)
        else:
            env.current_step, env.balance = state_backup
        if self.debug:
            print(f"[ShadowSimulator] Simulated {len(trades)} trades with mode={self.mode} over horizon={self.horizon}")
        return trades

    # --- Evolutionary methods ---
    def mutate(self, std: float = 1.0):
        # Mutate horizon (integer > 1) and possibly mode
        self.horizon = max(1, int(self.horizon + np.random.randint(-2, 3)))
        if random.random() < 0.3:
            self.mode = random.choice(self.MODES)
        if self.debug:
            print(f"[ShadowSimulator] Mutated: horizon={self.horizon}, mode={self.mode}")

    def crossover(self, other: "ShadowSimulator") -> "ShadowSimulator":
        # Cross horizon and mode
        horizon = self.horizon if random.random() < 0.5 else other.horizon
        mode = self.mode if random.random() < 0.5 else other.mode
        return ShadowSimulator(horizon=horizon, mode=mode, debug=self.debug or other.debug)

    def get_observation_components(self) -> np.ndarray:
        mode_idx = self.MODES.index(self.mode)
        return np.array([self.horizon, mode_idx], dtype=np.float32)

    def get_state(self):
        return {
            "horizon": self.horizon,
            "mode": self.mode,
        }

    def set_state(self, state):
        self.horizon = int(state.get("horizon", self.horizon))
        self.mode = state.get("mode", self.mode)
class RoleCoach(Module):
    """
    Evaluates trade count per step, recommends discipline/role adaptation.
    Evolutionary: max_trades is mutable.
    """
    def __init__(self, max_trades: int = 2, penalty: float = 1.0, debug=False):
        self.max_trades = int(max_trades)
        self.penalty = float(penalty)
        self.debug = debug

    def reset(self):
        pass

    def step(self, trades: List[dict] = None, actions: np.ndarray = None) -> float:
        n = len(trades) if trades is not None else 0
        score = float(max(0, n - self.max_trades)) * self.penalty
        if self.debug:
            print(f"[RoleCoach] trades={n} penalty={score}")
        return score

    # --- Evolutionary methods ---
    def mutate(self, std=1.0):
        # Mutate max_trades and penalty
        self.max_trades = max(1, int(self.max_trades + np.random.randint(-1, 2)))
        self.penalty = float(np.clip(self.penalty + np.random.normal(0, std), 0.1, 10.0))
        if self.debug:
            print(f"[RoleCoach] Mutated: max_trades={self.max_trades}, penalty={self.penalty}")

    def crossover(self, other: "RoleCoach") -> "RoleCoach":
        max_trades = self.max_trades if np.random.rand() < 0.5 else other.max_trades
        penalty = self.penalty if np.random.rand() < 0.5 else other.penalty
        return RoleCoach(max_trades=max_trades, penalty=penalty, debug=self.debug or other.debug)

    def get_observation_components(self) -> np.ndarray:
        return np.array([self.max_trades, self.penalty], dtype=np.float32)

    def get_state(self):
        return {
            "max_trades": self.max_trades,
            "penalty": self.penalty,
        }

    def set_state(self, state):
        self.max_trades = int(state.get("max_trades", self.max_trades))
        self.penalty = float(state.get("penalty", self.penalty))
