#  # ─────────────────────────────────────────────────────────────
#  # File: modules/memory/memory_compressor.py
#  # ─────────────────────────────────────────────────────────────

import logging
import os
from typing import List, Tuple
import numpy as np
from sklearn.decomposition import PCA
from modules.core.core import Module


# ─────────────────────────────────────────────────────────────
class MemoryCompressor(Module):
    def __init__(self, compress_interval: int = 10, n_components: int = 8, 
                 profit_threshold: float = 10.0, debug=False):
        self.compress_interval = compress_interval
        self.n_components = n_components
        self.profit_threshold = profit_threshold
        self.debug = debug
        
        # Ensure log directory exists
        log_dir = os.path.join("logs", "memory")
        os.makedirs(log_dir, exist_ok=True)
        
        # Enhanced Logger Setup - FIXED
        self.logger = logging.getLogger(f"MemoryCompressor_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        fh = logging.FileHandler(os.path.join(log_dir, "memory_compressor.log"), mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"MemoryCompressor initialized - interval={compress_interval}, components={n_components}, profit_threshold=€{profit_threshold}")
        
        self.reset()

    def reset(self):
        # FIX: Separate memories for different outcomes
        self.profit_memory: List[Tuple[np.ndarray, float]] = []  # (features, pnl)
        self.loss_memory: List[Tuple[np.ndarray, float]] = []
        self.intuition_vector = np.zeros(self.n_components, np.float32)
        self.profit_direction = np.zeros(self.n_components, np.float32)  # Direction of profit
        self.loss_direction = np.zeros(self.n_components, np.float32)    # Direction to avoid
        self._compression_count = 0
        self._step_count = 0
        
        self.logger.info("MemoryCompressor reset - all memories and vectors cleared")

    def step(self, *_, **__):
        """Return profit-seeking intuition"""
        self._step_count += 1
        
        try:
            # FIX: Combine intuition with profit direction
            if np.linalg.norm(self.profit_direction) > 0:
                # Blend current intuition with profit direction
                alpha = 0.7  # Weight towards profit
                combined = alpha * self.profit_direction + (1-alpha) * self.intuition_vector
                result = combined / (np.linalg.norm(combined) + 1e-8)
                
                if self._step_count % 100 == 0:
                    self.logger.info(f"Step {self._step_count}: Combined intuition (alpha={alpha}), profit_norm={np.linalg.norm(self.profit_direction):.3f}")
                else:
                    self.logger.debug(f"Step {self._step_count}: Returning combined intuition")
                    
                return result
            else:
                if self._step_count % 100 == 0:
                    self.logger.info(f"Step {self._step_count}: No profit direction, returning base intuition")
                return self.intuition_vector.copy()
        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            return self.intuition_vector.copy()

    def compress(self, episode: int, trades: List[dict]):
        """FIX: Compress with focus on profitable patterns"""
        try:
            self.logger.info(f"Compressing episode {episode} with {len(trades)} trades")
            
            profit_trades = 0
            loss_trades = 0
            
            for tr in trades:
                if "features" in tr and "pnl" in tr:
                    vec = np.asarray(tr["features"], np.float32)
                    if vec.size != self.n_components:
                        vec = np.pad(vec, (0, max(0, self.n_components - vec.size)))[:self.n_components]
                    
                    pnl = tr["pnl"]
                    if pnl > self.profit_threshold:
                        self.profit_memory.append((vec, pnl))
                        profit_trades += 1
                    elif pnl < 0:
                        self.loss_memory.append((vec, abs(pnl)))
                        loss_trades += 1
                        
            self.logger.info(f"Episode {episode}: Added {profit_trades} profitable, {loss_trades} loss trades")
                        
            # Limit memory size
            max_memory = 1000
            if len(self.profit_memory) > max_memory:
                removed = len(self.profit_memory) - max_memory
                self.profit_memory = self.profit_memory[-max_memory:]
                self.logger.debug(f"Trimmed {removed} old profit memories")
                
            if len(self.loss_memory) > max_memory:
                removed = len(self.loss_memory) - max_memory
                self.loss_memory = self.loss_memory[-max_memory:]
                self.logger.debug(f"Trimmed {removed} old loss memories")
                        
            # Compress at intervals
            if episode % self.compress_interval != 0:
                self.logger.debug(f"Episode {episode}: Not compression interval ({self.compress_interval})")
                return
                
            self._compression_count += 1
            self.logger.info(f"Performing compression #{self._compression_count} for episode {episode}")
            
            # FIX: Extract profit and loss directions
            if len(self.profit_memory) >= 5:  # Need minimum profitable trades
                # Weight by profit amount
                profit_vecs = []
                weights = []
                for vec, pnl in self.profit_memory[-50:]:  # Recent 50 trades
                    profit_vecs.append(vec)
                    weights.append(pnl)
                    
                X_profit = np.vstack(profit_vecs)
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                # Weighted average direction
                old_direction = self.profit_direction.copy()
                self.profit_direction = np.average(X_profit, axis=0, weights=weights)
                
                direction_change = np.linalg.norm(self.profit_direction - old_direction)
                self.logger.info(f"Profit direction updated: change={direction_change:.3f}, total_profitable={len(self.profit_memory)}")
                
                # PCA for main profit components
                if X_profit.shape[0] > self.n_components:
                    try:
                        pca = PCA(n_components=min(3, self.n_components))
                        pca.fit(X_profit)
                        # First principal component is main profit direction
                        if hasattr(pca, 'components_'):
                            self.intuition_vector = pca.components_[0]
                            explained_var = pca.explained_variance_ratio_[0]
                            self.logger.info(f"PCA completed: first component explains {explained_var:.3f} of variance")
                    except Exception as e:
                        self.logger.error(f"PCA failed: {e}")
                        
            if len(self.loss_memory) >= 5:
                # Extract loss patterns to avoid
                loss_vecs = [vec for vec, _ in self.loss_memory[-50:]]
                X_loss = np.vstack(loss_vecs)
                old_loss_direction = self.loss_direction.copy()
                self.loss_direction = X_loss.mean(axis=0)
                
                loss_change = np.linalg.norm(self.loss_direction - old_loss_direction)
                self.logger.info(f"Loss direction updated: change={loss_change:.3f}, total_losses={len(self.loss_memory)}")
                
            self.logger.info(f"Compression #{self._compression_count} completed - Profits: {len(self.profit_memory)}, Losses: {len(self.loss_memory)}")
            
        except Exception as e:
            self.logger.error(f"Error in compress: {e}")

    def get_observation_components(self) -> np.ndarray:
        """FIX: Return full intuition plus profit metrics"""
        try:
            profit_strength = np.linalg.norm(self.profit_direction)
            loss_strength = np.linalg.norm(self.loss_direction)
            profit_clarity = profit_strength / (loss_strength + 1e-8)
            
            # Return intuition vector plus metrics
            result = np.concatenate([
                self.intuition_vector,
                [profit_strength, loss_strength, profit_clarity]
            ]).astype(np.float32)
            
            self.logger.debug(f"Observation: profit_strength={profit_strength:.3f}, loss_strength={loss_strength:.3f}, clarity={profit_clarity:.3f}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.zeros(self.n_components + 3, np.float32)

    def get_state(self):
        return {
            "profit_memory": [(v.tolist(), p) for v, p in self.profit_memory[-100:]],  # Keep recent
            "loss_memory": [(v.tolist(), p) for v, p in self.loss_memory[-100:]],
            "intuition_vector": self.intuition_vector.tolist(),
            "profit_direction": self.profit_direction.tolist(),
            "loss_direction": self.loss_direction.tolist(),
            "compression_count": self._compression_count,
            "step_count": self._step_count,
        }

    def set_state(self, state):
        self.profit_memory = [(np.asarray(v, np.float32), p) for v, p in state.get("profit_memory", [])]
        self.loss_memory = [(np.asarray(v, np.float32), p) for v, p in state.get("loss_memory", [])]
        self.intuition_vector = np.asarray(state.get("intuition_vector", np.zeros(self.n_components)), np.float32)
        self.profit_direction = np.asarray(state.get("profit_direction", np.zeros(self.n_components)), np.float32)
        self.loss_direction = np.asarray(state.get("loss_direction", np.zeros(self.n_components)), np.float32)
        self._compression_count = state.get("compression_count", 0)
        self._step_count = state.get("step_count", 0)
        
        self.logger.info(f"State restored: profits={len(self.profit_memory)}, losses={len(self.loss_memory)}, compressions={self._compression_count}")

    def mutate(self, noise_std=0.05):
        """Evolve profit-seeking intuition"""
        try:
            noise = np.random.normal(0, noise_std, self.intuition_vector.shape).astype(np.float32)
            self.intuition_vector += noise
            # Also mutate profit direction slightly
            self.profit_direction += np.random.normal(0, noise_std/2, self.profit_direction.shape).astype(np.float32)
            self.logger.info(f"Mutated with noise_std={noise_std}")
        except Exception as e:
            self.logger.error(f"Error in mutation: {e}")
        
    def crossover(self, other: "MemoryCompressor"):
        child = MemoryCompressor(self.compress_interval, self.n_components, self.profit_threshold, self.debug)
        mask = np.random.rand(*self.intuition_vector.shape) > 0.5
        child.intuition_vector = np.where(mask, self.intuition_vector, other.intuition_vector)
        child.profit_direction = np.where(mask, self.profit_direction, other.profit_direction)
        self.logger.info("Crossover completed")
        return child
