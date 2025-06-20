"""
Committee-of-experts voting and arbitration logic for strategy modules, including consensus, collusion detection, and adaptive risk gating.
FIXED: Less restrictive gating, better consensus logic, and improved debugging.
"""
from __future__ import annotations

# ── standard lib ──────────────────────────────────────────────────────────
import math
from statistics import mean
from typing import List, Any, Dict, Optional
import logging

# ── third-party ───────────────────────────────────────────────────────────
import numpy as np

# ── internal – fall-back stub for type-checking only ──────────────────────
try:
    from modules.core.core import Module        # provided by your framework
except ImportError:
    class Module:                               # noqa: D401, E302 – stub only
        def reset(self) -> None:
            """Stub for reset method (type-checking only)."""
            ...
        def step(self, *a, **kw):
            """Stub for step method (type-checking only)."""
            ...
        def get_observation_components(self):
            """Stub for get_observation_components (type-checking only)."""
            ...

# ═══════════════════════════════════════════════════════════════════════════
# Helper functions for the new "smart gate" - MADE LESS RESTRICTIVE
# ═══════════════════════════════════════════════════════════════════════════
_LAYER_W = dict(
    liquidityheatmaplayer=2.0,      # LiquidityHeatmapLayer
    lhl=2.0,                        # Short form
    fractalregimeconfirmation=1.5,  # FractalRegimeConfirmation  
    frc=1.5,                        # Short form
    marketthemedetector=1.0,        # MarketThemeDetector
    mtd=1.0,                        # Short form
    markerregimeswitcher=1.0,       # MarketRegimeSwitcher
    switcher=1.0,                   # Short form
    # New additions for better voting
    positionmanager=1.5,            # Position manager has good judgment
    themeexpert=1.2,                # Theme expert
    regimebiasexpert=1.3,           # Regime expert
)

# ADJUSTED: Made less restrictive
_SIG_K     = 4.0        # REDUCED from 8.0 - gentler slope
_SIG_KNEE  = 0.15       # REDUCED from 0.20 - lower threshold
_BASE_GATE = 0.15       # REDUCED from 0.25 - easier base gate
_VOL_REF   = 0.02       # INCREASED from 0.01 - less sensitive to volatility

def _squash(c: float) -> float:
    """Gentler squashing function for confidence values."""
    return 1.0 / (1.0 + math.exp(-_SIG_K * (c - _SIG_KNEE)))

def _dyn_gate(vol: float,
              base: float = _BASE_GATE,
              vol_ref: float = _VOL_REF) -> float:
    """Dynamic gate that's less sensitive to volatility"""
    # Cap volatility impact to prevent excessive gating
    vol_factor = min(vol / max(vol_ref, 1e-9), 2.0)
    return base * (1.0 + 0.5 * vol_factor)  # Reduced from full multiplication

# ═══════════════════════════════════════════════════════════════════════════
# Convenience filters
# ═══════════════════════════════════════════════════════════════════════════

class ConsensusDetector(Module):
    def __init__(self, n_members: int,
                 entropy_th: float = 0.5, dir_alpha: float = 0.3):
        self.n, self.th, self.alpha = n_members, entropy_th, dir_alpha
    def reset(self): pass
    def step(self, **kw): pass
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
    def add_votes(self, votes: List[Dict]) -> None:
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
    def __init__(self, horizons: List[int]): 
        self.horiz = np.asarray(horizons,np.float32)
        self.clock=0
    def reset(self): self.clock=0
    def step(self, **kw): pass
    def get_observation_components(self): return np.asarray([self.clock],np.float32)
    def apply(self, w: np.ndarray) -> np.ndarray:
        d = 1.0/(1.0+np.abs(self.clock-self.horiz))
        self.clock+=1
        return w*d

class AlternativeRealitySampler:
    def __init__(self, dim: int, n_samples: int = 5, sigma: float = .05):
        self.dim, self.n, self.sigma = dim, n_samples, sigma
    def sample(self, w: np.ndarray) -> np.ndarray:
        return w[None,:] + np.random.randn(self.n,self.dim)*self.sigma

# ═══════════════════════════════════════════════════════════════════════════
# StrategyArbiter with improved smart-gate
# ═══════════════════════════════════════════════════════════════════════════
class StrategyArbiter(Module):
    """Committee-of-experts blender with REINFORCE and improved risk gate."""
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
        
        # NEW: Track gate statistics for debugging
        self._gate_passes = 0
        self._gate_attempts = 0
        
        # Setup logging
        self.logger = logging.getLogger("StrategyArbiter")
        if self.debug and not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    # ─────────── external hook ────────────────────────────────────────────
    def update_market_state(self, volatility: float):
        self.curr_vol = float(volatility)

    # ─────────── helpers ──────────────────────────────────────────────────
    def _save_trace(self, t: Dict[str, Any]):
        self._trace.append(t)
        if len(self._trace) > self._log_size:
            self._trace = self._trace[-self._log_size:]

    def get_last_traces(self, n=5): return self._trace[-n:]

    # ─────────── main propose() with improved gating ──────────────────────
    def propose(self, obs: Any) -> np.ndarray:
        """
        Propose an action by blending member module proposals using confidence scores,
        consensus, horizon alignment, and an improved risk gate.
        """
        props, confs, lnames = [], [], []
        trace = {"module": [], "conf": [], "maj": None,
                 "sig_mean": None, "gate": None, "passed": None}
        
        # 1. collect votes
        for m in self.members:
            name = m.__class__.__name__.lower()
            lnames.append(name)
            try:
                p = m.propose_action(obs).astype(np.float32)
                if p.shape != (self.action_dim,): 
                    raise ValueError(f"shape mismatch: {p.shape} vs {self.action_dim}")
            except Exception as e:
                if self.debug: 
                    self.logger.warning(f"[Arbiter] propose_error in {name}: {e}")
                p = np.zeros(self.action_dim, np.float32)
            try:
                c = float(m.confidence(obs))
                if not np.isfinite(c): 
                    raise ValueError("NaN conf")
                c = np.clip(c, 0.0, 1.0)
            except Exception as e:
                if self.debug: 
                    self.logger.warning(f"[Arbiter] conf_error in {name}: {e}")
                c = 0.5
            props.append(p)
            confs.append(c)
            
        props = np.stack(props)                # (M,A)
        confs = np.asarray(confs, np.float32)  # (M,)
        trace["module"], trace["conf"] = lnames, confs.tolist()

        # 2. consensus + horizon alignment
        if self.consensus is not None: 
            confs = self.consensus.apply(confs)
        if self.haligner is not None:  
            confs = self.haligner.apply(confs)

        # 3. blend prior-posterior weights
        prior = self.weights / (self.weights.sum()+1e-8)
        like  = confs / (confs.sum()+1e-8)
        w_raw = (1-self.PRIOR_BLEND)*like + self.PRIOR_BLEND*prior
        w     = w_raw / (w_raw.sum()+1e-8)

        # 4. blended preliminary action
        action = np.dot(w, props)              # (A,)

        # 5. ──────── IMPROVED smart-gate ────────────────────────────────────
        # Extract directional signals from all action dimensions
        all_directions = []
        for i in range(0, self.action_dim, 2):
            if i < props.shape[1]:
                dirs = np.sign(props[:, i])
                all_directions.append(dirs)
                
        if all_directions:
            # Use average direction across instruments
            avg_dirs = np.mean(all_directions, axis=0)
        else:
            avg_dirs = np.sign(props[:, 0]) if props.shape[1] > 0 else np.zeros(len(self.members))
            
        # Calculate majority direction with confidence weighting
        weighted_dir = sum(d * c for d, c in zip(avg_dirs, confs) if c >= 0.2)
        maj = np.sign(weighted_dir)

        # Calculate signal strength with layer weighting
        sig_num = sig_den = 0.0
        for name, d, c in zip(lnames, avg_dirs, confs):
            # Get weight for this layer type
            layer_weight = _LAYER_W.get(name, 1.0)
            # Also check for partial matches
            for key, weight in _LAYER_W.items():
                if key in name:
                    layer_weight = max(layer_weight, weight)
                    
            s = _squash(c) * layer_weight
            sig_num += s * d
            sig_den += s
            
        sig_mean = 0.0 if sig_den == 0 else sig_num / sig_den
        
        # Dynamic gate threshold
        gate = _dyn_gate(self.curr_vol)
        
        # IMPROVED: More nuanced gate logic
        # Allow trades if:
        # 1. Strong directional agreement (even with zero majority)
        # 2. OR high confidence from key modules
        # 3. OR sufficient signal strength
        high_conf_modules = sum(1 for c in confs if c >= 0.7)
        strong_signal = abs(sig_mean) >= gate
        good_agreement = abs(weighted_dir) >= 0.5
        high_confidence = high_conf_modules >= len(self.members) * 0.3
        
        passed = strong_signal or (good_agreement and high_confidence)
        
        # Additional override: if most experts agree on direction with decent confidence
        if not passed:
            direction_agreement = sum(1 for d in avg_dirs if d == maj) / len(avg_dirs)
            avg_confidence = np.mean(confs)
            if direction_agreement >= 0.6 and avg_confidence >= 0.4:
                passed = True
                if self.debug:
                    self.logger.info(f"[Gate] Override: {direction_agreement:.1%} agreement, {avg_confidence:.2f} avg conf")
        
        # Track statistics
        self._gate_attempts += 1
        if passed:
            self._gate_passes += 1
        
        if self.debug:
            pass_rate = self._gate_passes / max(self._gate_attempts, 1)
            self.logger.debug(
                f"[Gate] maj={maj:+.0f} sig_mean={sig_mean:+.3f} "
                f"gate={gate:.3f} passed={passed} "
                f"(pass_rate={pass_rate:.1%})"
            )

        if not passed:
            # Don't completely zero - apply strong dampening instead
            action *= 0.1  # Allow small positions through
            
        trace.update(dict(maj=int(maj), sig_mean=float(sig_mean),
                          gate=float(gate), passed=bool(passed)))
        # ────────────────────────────────────────────────────────────────

        # 6. Final validation
        if not np.all(np.isfinite(action)):
            if self.debug: 
                self.logger.warning("[Arbiter] NaN in blended action – zeroed")
            action = np.nan_to_num(action)

        self.last_alpha = w.copy()
        self._save_trace(trace)
        return action.astype(np.float32, copy=False)

    # ─────────── REINFORCE weight update ─────────────────────────────────
    def update_reward(self, reward: float):
        if self.last_alpha is None: 
            return
            
        # Update baseline
        self._b = self._b_beta * self._b + (1 - self._b_beta) * reward
        
        # Calculate advantage
        adv = reward - self._b
        
        # REINFORCE gradient
        grad = adv * (self.last_alpha - self.last_alpha.mean())
        
        # Update weights with gradient clipping
        grad = np.clip(grad, -1.0, 1.0)
        self.weights += self.adapt_rate * grad
        
        # Ensure weights stay positive and normalized
        self.weights = np.clip(self.weights, 0.01, None)
        self.weights /= self.weights.sum()
        
        # Decay learning rate
        self.adapt_rate = max(1e-4, self.adapt_rate * 0.999)
        
        # Clear last alpha
        self.last_alpha = None
        
        if self.debug and abs(adv) > 0.1:
            self.logger.debug(f"[Update] reward={reward:.3f} adv={adv:.3f} new_weights={self.weights}")

    # ─────────── boiler-plate for framework ───────────────────────────────
    def reset(self):
        self.last_alpha = None
        self._trace = []
        self._gate_passes = 0
        self._gate_attempts = 0
        self._b = 0.0
        
    def step(self, **kw):
        """This method should not be called directly; use propose() and update_reward() instead."""
        raise RuntimeError("use propose()/update_reward()")
        
    def get_observation_components(self): 
        return self.weights.copy()
        
    # genome helpers
    @property
    def genome_dim(self): 
        return len(self.weights)
        
    def set_genome(self, g: np.ndarray):
        assert g.shape == self.weights.shape
        self.weights = np.clip(g, 0.01, None)
        self.weights /= self.weights.sum()
        
    def get_state(self): 
        return {
            "weights": self.weights.copy(),
            "baseline": self._b,
            "adapt_rate": self.adapt_rate,
            "gate_stats": {
                "passes": self._gate_passes,
                "attempts": self._gate_attempts
            }
        }
        
    def set_state(self, st):
        """Set the internal state of the StrategyArbiter from a state dictionary."""
        weights = np.asarray(st.get("weights", self.weights))
        self.set_genome(weights)
        self._b = st.get("baseline", 0.0)
        self.adapt_rate = st.get("adapt_rate", 0.01)
        if "gate_stats" in st:
            self._gate_passes = st["gate_stats"].get("passes", 0)
            self._gate_attempts = st["gate_stats"].get("attempts", 0)