"""
strategy_arbiter.py
"""
from __future__ import annotations

# ── standard lib ──────────────────────────────────────────────────────────
import math
from statistics import mean
from typing import List, Any, Dict, Optional

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np

# ── internal – fall-back stub for type-checking only ──────────────────────
try:
    from modules.core.core import Module        # provided by your framework
except ImportError:
    class Module:                               # noqa: D401, E302 – stub only
        def reset(self) -> None: ...
        def step(self, *a, **kw): ...
        def get_observation_components(self): ...

# ═══════════════════════════════════════════════════════════════════════════
# Helper functions for the new “smart gate”
# ═══════════════════════════════════════════════════════════════════════════
_LAYER_W = dict(lhl=2.0,       # LiquidityHeatmapLayer
                frc=1.5,       # FractalRegimeConfirmation
                mtd=1.0,       # MarketThemeDetector
                switcher=1.0)  # MarketRegimeSwitcher

_SIG_K     = 8.0        # steepness of soft-sigmoid
_SIG_KNEE  = 0.20       # where confidence starts to matter
_BASE_GATE = 0.25       # min |sig_mean| required in low vol
_VOL_REF   = 0.01       # reference σ (adjust to your asset)

def _squash(c: float) -> float:
    """Sharper than logistic for c∈[0,1].  0→0.0, 0.2→~0.1, 0.5→~0.97."""
    return 1.0 / (1.0 + math.exp(-_SIG_K * (c - _SIG_KNEE)))

def _dyn_gate(vol: float,
              base: float = _BASE_GATE,
              vol_ref: float = _VOL_REF) -> float:
    """Increase the required certainty when realised σ is high."""
    return base * (1.0 + vol / max(vol_ref, 1e-9))

# ═══════════════════════════════════════════════════════════════════════════
# Convenience filters (unchanged from your original file, shortened)
# ═══════════════════════════════════════════════════════════════════════════
from modules.utils.info_bus import Vote          # type: ignore  # pylint: disable=import-error
class ConsensusDetector(Module):
    def __init__(self, n_members: int,
                 entropy_th: float = 0.5, dir_alpha: float = 0.3):
        self.n, self.th, self.alpha = n_members, entropy_th, dir_alpha
    def reset(self): ...
    def step(self, **kw): ...
    def get_observation_components(self):
        return np.zeros(self.n, np.float32)
    def apply(self, w: np.ndarray) -> np.ndarray:
        p = w / (w.sum() + 1e-8)
        ent = -np.sum(p * np.log(p + 1e-8))
        if ent < self.th:
            noise = np.random.dirichlet([self.alpha] * self.n)
            return 0.8 * p + 0.2 * noise
        return w

class CollusionAuditor:
    def __init__(self, window: int = 250, corr_threshold: float = .95):
        self.window, self.th = window, corr_threshold
        self._hist: Dict[str, List[int]] = {}
    def add_votes(self, votes: List[Vote]) -> None:
        for v in votes:
            self._hist.setdefault(v["module"], []).append(v["action"])
            h = self._hist[v["module"]]
            if len(h) > self.window:
                del h[0]
    def check(self) -> List[str]:
        out, mods = [], list(self._hist)
        for i, m1 in enumerate(mods):
            for m2 in mods[i+1:]:
                r = self._pearson(self._hist[m1], self._hist[m2])
                if r is not None and r > self.th:
                    out += [m1, m2]
        return sorted(set(out))
    @staticmethod
    def _pearson(xs: List[int], ys: List[int]) -> Optional[float]:
        if len(xs) < 2 or len(xs) != len(ys): return None
        mx, my = mean(xs), mean(ys)
        num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        den = (sum((x-mx)**2 for x in xs)**.5 *
               sum((y-my)**2 for y in ys)**.5)
        return None if den == 0 else num/den

class TimeHorizonAligner(Module):
    def __init__(self, horizons: List[int]): self.horiz = np.asarray(horizons,np.float32); self.clock=0
    def reset(self): self.clock=0
    def step(self, **kw): ...
    def get_observation_components(self): return np.asarray([self.clock],np.float32)
    def apply(self, w: np.ndarray) -> np.ndarray:
        d = 1.0/(1.0+np.abs(self.clock-self.horiz)); self.clock+=1; return w*d

class AlternativeRealitySampler:
    def __init__(self, dim: int, n_samples: int = 5, sigma: float = .05):
        self.dim, self.n, self.sigma = dim, n_samples, sigma
    def sample(self, w: np.ndarray) -> np.ndarray:
        return w[None,:] + np.random.randn(self.n,self.dim)*self.sigma

# ═══════════════════════════════════════════════════════════════════════════
# StrategyArbiter with smart-gate
# ═══════════════════════════════════════════════════════════════════════════
class StrategyArbiter(Module):
    """Committee-of-experts blender with REINFORCE *and* risk gate."""
    PRIOR_BLEND: float = 0.30

    def __init__(self, members: List[Module],
                 init_weights: List[float] | np.ndarray,
                 action_dim: int,
                 adapt_rate: float = 0.01,
                 consensus: Optional[Module] = None,
                 collusion: Optional[Module] = None,
                 horizon_aligner: Optional[Module] = None,
                 debug: bool = True,
                 audit_log_size: int = 100):
        self.members   = members
        self.weights   = np.asarray(init_weights, np.float32)
        assert self.weights.shape == (len(members),)
        self.action_dim      = action_dim
        self.adapt_rate      = adapt_rate
        self.consensus       = consensus
        self.collusion       = collusion
        self.haligner        = horizon_aligner
        self.debug           = debug
        self.curr_vol: float = 0.0            # updated externally each bar
        self.last_alpha: Optional[np.ndarray] = None
        self._b, self._b_beta = 0.0, 0.98
        self._trace: List[Dict[str,Any]] = []
        self._log_size = audit_log_size

    # ─────────── external hook ────────────────────────────────────────────
    def update_market_state(self, volatility: float):
        self.curr_vol = float(volatility)

    # ─────────── helpers ──────────────────────────────────────────────────
    def _save_trace(self, t: Dict[str, Any]):
        self._trace.append(t)
        if len(self._trace) > self._log_size:
            self._trace = self._trace[-self._log_size:]

    def get_last_traces(self, n=5): return self._trace[-n:]

    # ─────────── main propose() ───────────────────────────────────────────
    def propose(self, obs: Any) -> np.ndarray:
        props, confs, lnames = [], [], []
        trace = {"module": [], "conf": [], "maj": None,
                 "sig_mean": None, "gate": None, "passed": None}
        

        # 1. collect votes
        for m in self.members:
            lnames.append(m.__class__.__name__.lower())
            try:
                p = m.propose_action(obs).astype(np.float32)
                if p.shape != (self.action_dim,): raise ValueError("shape")
            except Exception as e:             # noqa: BLE001
                if self.debug: print("[Arbiter] propose_error:", e)
                p = np.zeros(self.action_dim, np.float32)
            try:
                c = float(m.confidence(obs))
                if not np.isfinite(c): raise ValueError("NaN conf")
            except Exception as e:             # noqa: BLE001
                if self.debug: print("[Arbiter] conf_error:", e)
                c = 0.5
            props.append(p); confs.append(c)
        props = np.stack(props)                # (M,A)
        confs = np.asarray(confs, np.float32)  # (M,)
        trace["module"], trace["conf"] = lnames, confs.tolist()

        # 2. consensus + horizon alignment
        if self.consensus is not None: confs = self.consensus.apply(confs)
        if self.haligner is not None:  confs = self.haligner.apply(confs)

        # 3. blend prior-posterior weights
        prior = self.weights / (self.weights.sum()+1e-8)
        like  = confs / (confs.sum()+1e-8)
        w_raw = (1-self.PRIOR_BLEND)*like + self.PRIOR_BLEND*prior
        w     = w_raw / (w_raw.sum()+1e-8)

        # 4. blended preliminary action
        action = np.dot(w, props)              # (A,)

        # 5. ──────── smart-gate ────────────────────────────────────────
        dirs = np.sign(props[:, 0])            # use first dim direction
        maj  = np.sign(sum(d for d,c in zip(dirs, confs) if c >= 0.25))

        sig_num = sig_den = 0.0
        for name, d, c in zip(lnames, dirs, confs):
            s = _squash(c) * _LAYER_W.get(name.split(".")[-1], 1.0)
            sig_num += s * d; sig_den += s
        sig_mean = 0.0 if sig_den == 0 else sig_num / sig_den
        gate = _dyn_gate(self.curr_vol)
        if self.debug:
            print(f"[Gate] maj={maj:+.0f} sig_mean={sig_mean:+.3f} "
                f"gate={gate:.3f}")

        passed = (maj != 0 and math.copysign(1.0, sig_mean) == maj
                  and abs(sig_mean) >= gate)
        if not passed:
            action[:] = 0.0                    # veto

        trace.update(dict(maj=int(maj), sig_mean=float(sig_mean),
                          gate=float(gate), passed=bool(passed)))
        # ────────────────────────────────────────────────────────────────

        if not np.all(np.isfinite(action)):
            if self.debug: print("[Arbiter] NaN in blended action – zeroed")
            action = np.nan_to_num(action)

        self.last_alpha = w.copy(); self._save_trace(trace)
        return action.astype(np.float32, copy=False)

    # ─────────── REINFORCE weight update ─────────────────────────────────
    def update_reward(self, reward: float):
        if self.last_alpha is None: return
        self._b = self._b_beta*self._b + (1-self._b_beta)*reward
        adv = reward - self._b
        grad = adv * (self.last_alpha - self.last_alpha.mean())
        self.weights += self.adapt_rate * grad
        self.weights = np.clip(self.weights, 1e-4, None)
        self.weights /= self.weights.sum()
        self.adapt_rate = max(1e-4, self.adapt_rate*0.999)
        self.last_alpha = None

    # ─────────── boiler-plate for framework ‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐‐
    def reset(self): self.last_alpha=None; self._trace=[]
    def step(self, **kw): raise RuntimeError("use propose()/update_reward()")
    def get_observation_components(self): return self.weights.copy()
    # genome helpers
    @property
    def genome_dim(self): return len(self.weights)
    def get_genome(self): return self.weights.copy()
    def set_genome(self, g: np.ndarray):
        assert g.shape == self.weights.shape; self.weights=np.clip(g,1e-4,None); self.weights/=self.weights.sum()
    # (de)serialisation
    def get_state(self): return {"weights": self.weights}
    def set_state(self, st): self.set_genome(np.asarray(st.get("weights", self.weights)))
