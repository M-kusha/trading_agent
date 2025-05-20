#modules/simulation.py
import numpy as np
from typing import List, Any
from modules.core.core import Module


class ShadowSimulator(Module):
    def __init__(self, horizon: int=5, debug=False):
        self.horizon = horizon
        self.debug   = debug

    def reset(self): pass
    def step(self, **kwargs): pass

    def simulate(self, env, actions: np.ndarray) -> List[dict]:
        # if env or actions is missing, return empty immediately
        if env is None or actions is None:
            return []
        data = env.data
        step0, bal0 = env.current_step, env.balance
        trades = []
        for _ in range(self.horizon):
            if env.current_step >= len(next(iter(data.values()))["D1"]):
                break
            for i, instr in enumerate(env.instruments):
                tr = env._execute_trade(instr, actions[2*i], actions[2*i+1])
                if tr:
                    trades.append(tr)
            env.current_step += 1
        env.current_step, env.balance = step0, bal0
        return trades

    def get_observation_components(self) -> np.ndarray:
        return np.zeros(0, dtype=np.float32)
class RoleCoach(Module):
    def __init__(self, max_trades: int=2, debug=False):
        self.max_trades = max_trades
        self.debug      = debug
    def reset(self): pass
    def step(self, trades: List[dict], actions: np.ndarray) -> float:
        return float(max(0, len(trades)-self.max_trades))
    def get_observation_components(self) -> np.ndarray:
        return np.zeros(1, dtype=np.float32)

