# ─────────────────────────────────────────────────────────────
# File: modules/market/liquidity_heatmap_layer.py
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
from typing import Any, List, Tuple
import tensorflow as tf
from ..core.core import Module


class LiquidityHeatmapLayer(Module):
    def __init__(self, action_dim: int, debug: bool = True, genome: dict = None, weights: list = None):
        super().__init__()
        # Evolvable architecture params (genome)
        if genome:
            self.lstm_units = int(genome.get("lstm_units", 32))
            self.seq_len = int(genome.get("seq_len", 10))
            self.dense_units = int(genome.get("dense_units", 2))
            self.train_epochs = int(genome.get("train_epochs", 10))
        else:
            self.lstm_units = 32
            self.seq_len = 10
            self.dense_units = 2
            self.train_epochs = 10

        self.action_dim = action_dim
        self.debug = debug
        self.bids: List[Tuple[float, float]] = []
        self.asks: List[Tuple[float, float]] = []
        self.history: deque[Tuple[float, float]] = deque(maxlen=200)

        self._trained = False
        self._model = self._build_lstm()
        if weights is not None:
            self._set_weights(weights)

        # Genome for reproduction
        self.genome = {
            "lstm_units": self.lstm_units,
            "seq_len": self.seq_len,
            "dense_units": self.dense_units,
            "train_epochs": self.train_epochs,
        }

    def _build_lstm(self):
        with tf.device("/CPU:0"):
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.seq_len, 2)),
                tf.keras.layers.LSTM(self.lstm_units),
                tf.keras.layers.Dense(self.dense_units),
            ])
            model.compile(optimizer="adam", loss="mse")
            return model

    # --------- Weight helpers for neuroevolution ----------
    def _get_weights(self):
        # Returns a list of numpy arrays representing all model weights
        return self._model.get_weights()

    def _set_weights(self, weights):
        self._model.set_weights([np.copy(w) for w in weights])

    def _weights_like(self, other):
        # Make sure weights shapes match (important for crossover)
        ws1 = self._get_weights()
        ws2 = other._get_weights()
        return all(w1.shape == w2.shape for w1, w2 in zip(ws1, ws2))

    def clone_with_weights(self):
        # Return a new instance with the same weights & genome
        return LiquidityHeatmapLayer(
            self.action_dim,
            debug=self.debug,
            genome=self.genome.copy(),
            weights=self._get_weights(),
        )

    def reset(self):
        self.bids.clear()
        self.asks.clear()
        self.history.clear()
        self._trained = False
        # Don't rebuild model on reset, just clear training flag

    # ------------------------------------------------------

    def _make_seqs(self):
        X, y = [], []
        rows = list(self.history)
        for i in range(len(rows) - self.seq_len):
            X.append(rows[i : i + self.seq_len])
            y.append(rows[i + self.seq_len])
        return np.asarray(X, np.float32), np.asarray(y, np.float32)

    def _train_if_ready(self):
        if self._trained or len(self.history) < 2 * self.seq_len:
            return
        X, y = self._make_seqs()
        if X.size:
            self._model.fit(X, y, epochs=self.train_epochs, verbose=0)
            self._trained = True
            if self.debug:
                print(f"[LHL] trained on {len(X)} sequences (units={self.lstm_units}, seq_len={self.seq_len})")

    def step(self, **kwargs):
        # For now, simulate liquidity data since we don't have real order book
        # In production, this would use real order book data
        
        # Simulate spread and depth based on volatility
        if "env" in kwargs:
            env = kwargs["env"]
            vol = env.get_volatility_profile().get(env.instruments[0], 0.01)
            
            # Higher volatility = wider spread, lower depth
            spread = vol * np.random.uniform(0.5, 1.5)
            depth = 1000 / (1 + vol * 10) * np.random.uniform(0.8, 1.2)
            
            self.history.append((spread, depth))
            self._train_if_ready()
            
            if self.debug and len(self.history) % 50 == 0:
                print(f"[LHL] spread={spread:.6f} depth={depth:.1f}")

    def predict_liquidity(self, steps: int = 4) -> Tuple[float, float]:
        if not self._trained or len(self.history) < self.seq_len:
            return 0.0, 0.0
        seq = np.array([list(self.history)[-self.seq_len:]], np.float32)
        pred = self._model.predict(seq, verbose=0)[0]
        return float(pred[0]), float(pred[1])

    def current_score(self) -> float:
        if not self.history:
            return 1.0
        spread, depth = self.history[-1]
        # Lower spread and higher depth = better liquidity
        score = float(np.log1p(depth) * np.exp(-spread * 100.0))
        return np.clip(score, 0.0, 1.0)

    def get_observation_components(self) -> np.ndarray:
        return np.array([self.current_score()], np.float32)

    def propose_action(self, obs: Any) -> np.ndarray:
        """
        Adjust position size based on liquidity conditions
        """
        liq_score = self.current_score()
        action = np.zeros(self.action_dim, np.float32)
        
        if liq_score < 0.3:
            # Poor liquidity - no trading
            return action
            
        # Scale position size by liquidity
        base_size = 0.3
        for i in range(0, self.action_dim, 2):
            action[i] = base_size * liq_score
            action[i+1] = 0.5  # Standard duration
            
        return action

    def confidence(self, obs: Any) -> float:
        score = self.current_score()
        # High confidence when liquidity is good
        conf = float(np.clip(score, 0.1, 1.0))
        if self.debug:
            print(f"[LiquidityHeatmapLayer] Liquidity score={score:.2f}, confidence={conf:.2f}")
        return conf

    # --- Evolutionary methods ---
    def get_genome(self):
        return self.genome.copy()

    def set_genome(self, genome):
        old_units = self.lstm_units
        self.lstm_units = int(genome.get("lstm_units", self.lstm_units))
        self.seq_len = int(genome.get("seq_len", self.seq_len))
        self.dense_units = int(genome.get("dense_units", self.dense_units))
        self.train_epochs = int(genome.get("train_epochs", self.train_epochs))
        self.genome = {
            "lstm_units": self.lstm_units,
            "seq_len": self.seq_len,
            "dense_units": self.dense_units,
            "train_epochs": self.train_epochs,
        }
        # Only rebuild model if architecture changed
        if old_units != self.lstm_units:
            self._model = self._build_lstm()
            self._trained = False

    def mutate(self, mutation_rate=0.2, weight_mutate_std=0.05):
        g = self.genome.copy()
        if np.random.rand() < mutation_rate:
            g["lstm_units"] = int(np.clip(self.lstm_units + np.random.randint(-8, 9), 8, 128))
        if np.random.rand() < mutation_rate:
            g["seq_len"] = int(np.clip(self.seq_len + np.random.randint(-2, 3), 5, 20))
        if np.random.rand() < mutation_rate:
            g["dense_units"] = int(np.clip(self.dense_units + np.random.randint(-1, 2), 1, 8))
        if np.random.rand() < mutation_rate:
            g["train_epochs"] = int(np.clip(self.train_epochs + np.random.randint(-2, 3), 2, 20))
        self.set_genome(g)

        # Mutate weights (add small Gaussian noise)
        if self._trained:
            weights = self._get_weights()
            mutated = []
            for w in weights:
                if np.issubdtype(w.dtype, np.floating):
                    noise = np.random.randn(*w.shape) * weight_mutate_std
                    mutated.append(w + noise)
                else:
                    mutated.append(w)
            self._set_weights(mutated)

    def crossover(self, other, weight_mix_prob=0.5):
        # Mix architecture
        g1, g2 = self.genome, other.genome
        new_g = {k: np.random.choice([g1[k], g2[k]]) for k in g1}

        # Mix weights only if architectures match
        new_ws = None
        if self._weights_like(other):
            ws1, ws2 = self._get_weights(), other._get_weights()
            new_ws = []
            for w1, w2 in zip(ws1, ws2):
                if np.issubdtype(w1.dtype, np.floating) and w1.shape == w2.shape:
                    mask = np.random.rand(*w1.shape) < weight_mix_prob
                    mixed = np.where(mask, w1, w2)
                    new_ws.append(mixed)
                else:
                    new_ws.append(w1)

        return LiquidityHeatmapLayer(
            self.action_dim, debug=self.debug, genome=new_g, weights=new_ws
        )

    def get_state(self):
        return {
            "history": list(self.history),
            "trained": bool(self._trained),
            "genome": self.genome.copy(),
            "weights": [w.copy() for w in self._get_weights()] if self._trained else None,
        }

    def set_state(self, state):
        self.history = deque(state.get("history", []), maxlen=200)
        self._trained = bool(state.get("trained", False))
        self.set_genome(state.get("genome", self.genome))
        if state.get("weights") is not None and self._trained:
            self._set_weights(state["weights"])