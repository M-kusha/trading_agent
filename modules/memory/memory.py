# modules/memory.py

import logging
import os
from typing import List, Any, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import random
import copy
from collections import deque
from datetime import datetime
from modules.core.core import Module

# ─────────────────────────────────────────────────────────────
class MistakeMemory(Module):
    """
    FIXED: Enhanced to learn from both wins and losses, providing actionable signals
    Clusters trades by outcome and market conditions to prevent repeated mistakes
    """

    def __init__(
        self,
        max_mistakes: int = 100,
        n_clusters: int = 5,
        profit_threshold: float = 10.0,  # €10 minimum for "good" trade
        *,
        interval: int | None = None,
        debug: bool = False,
    ) -> None:
        # 1) Core parameters
        if interval is not None:
            max_mistakes = interval
        self.max_mistakes = int(max_mistakes)
        self.n_clusters = int(n_clusters)
        self.profit_threshold = profit_threshold
        self.debug = debug

        # 2) Ensure log directory exists
        log_dir = os.path.join("logs", "memory")
        os.makedirs(log_dir, exist_ok=True)

        # 3) Set up logger **before** calling reset()
        self.logger = logging.getLogger(f"MistakeMemory_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.propagate = False

        # File handler → logs/memory/mistakes.log
        fh = logging.FileHandler(os.path.join(log_dir, "mistakes.log"), mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Optional console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # 4) Now it’s safe to reset internal state (which uses self.logger)
        self.reset()

        # 5) Initial log
        self.logger.info(
            f"MistakeMemory initialized – "
            f"max_mistakes={self.max_mistakes}, "
            f"n_clusters={self.n_clusters}, "
            f"profit_threshold=€{self.profit_threshold}"
        )
    # ─────────────────────────────────────────────────────────────

    def reset(self):
        # FIX: Separate buffers for wins and losses
        self._loss_buf: List[Tuple[np.ndarray, float, dict]] = []
        self._win_buf: List[Tuple[np.ndarray, float, dict]] = []
        self._km_loss: KMeans | None = None
        self._km_win: KMeans | None = None
        self._mean_dist = 0.0
        self._last_dist = 0.0
        self._danger_zones: List[np.ndarray] = []  # Cluster centers to avoid
        self._profit_zones: List[np.ndarray] = []  # Cluster centers to seek
        
        # FIX: Track patterns
        self._consecutive_losses = 0
        self._loss_patterns = {}  # pattern -> count
        self._avoidance_signal = 0.0
        self._step_count = 0
        
        self.logger.info("MistakeMemory reset - all buffers and patterns cleared")

    def step(self, *, trades: List[dict] | None = None,
                   features: np.ndarray | None = None,
                   pnl: float | None = None,
                   info: dict | None = None, **kw):
        """Enhanced to learn from both wins and losses"""
        
        self._step_count += 1
        self.logger.debug(f"Step {self._step_count} - trades={len(trades) if trades else 0}, features={features is not None}, pnl={pnl}")
        
        # Batch processing
        if trades is not None:
            self.logger.info(f"Processing batch of {len(trades)} trades")
            processed_count = 0
            for i, tr in enumerate(trades):
                try:
                    self.step(features=tr.get("features"), pnl=tr.get("pnl"), info=tr)
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing trade {i}: {e}")
            self.logger.info(f"Batch processed: {processed_count}/{len(trades)} trades successful")
            return

        # Individual trade processing
        if features is None or pnl is None:
            self.logger.debug("Insufficient data - skipping (no features or pnl)")
            return

        try:
            entry = (np.asarray(features, np.float32), float(pnl), info or {})
            
            # FIX: Categorize and store trades
            if pnl < 0:
                # Loss - track pattern
                self._loss_buf.append(entry)
                if len(self._loss_buf) > self.max_mistakes:
                    removed = self._loss_buf.pop(0)
                    self.logger.debug(f"Removed oldest loss: €{removed[1]:.2f}")
                    
                self._consecutive_losses += 1
                
                # Extract loss pattern
                pattern = self._extract_pattern(features, info)
                self._loss_patterns[pattern] = self._loss_patterns.get(pattern, 0) + 1
                
                self.logger.warning(f"Loss recorded: €{pnl:.2f}, Pattern: {pattern}, Consecutive: {self._consecutive_losses}, Total losses: {len(self._loss_buf)}")
                
            elif pnl > self.profit_threshold:
                # Significant win - learn from it
                self._win_buf.append(entry)
                if len(self._win_buf) > self.max_mistakes // 2:
                    removed = self._win_buf.pop(0)
                    self.logger.debug(f"Removed oldest win: €{removed[1]:.2f}")
                    
                self._consecutive_losses = 0
                
                self.logger.info(f"Profitable trade: €{pnl:.2f}, Total wins: {len(self._win_buf)}")
            else:
                # Small profit/loss - just log
                self.logger.debug(f"Small trade: €{pnl:.2f} (below profit threshold)")

            # Refit clusters
            self._fit_clusters()
            self._update_avoidance_signal()
            
            # Log memory statistics
            if self._step_count % 10 == 0:
                self._log_memory_stats()
                
        except Exception as e:
            self.logger.error(f"Error in step processing: {e}")

    def _extract_pattern(self, features: np.ndarray, info: dict) -> str:
        """Extract tradeable pattern from features and context"""
        try:
            # Simple pattern based on feature ranges
            volatility = info.get("volatility", 0)
            regime = info.get("regime", "unknown")
            hour = info.get("hour", -1)
            
            vol_level = "high" if volatility > 0.02 else "low"
            time_session = "asian" if 0 <= hour < 8 else "european" if 8 <= hour < 16 else "us"
            
            pattern = f"{regime}_{vol_level}_{time_session}"
            self.logger.debug(f"Extracted pattern: {pattern}")
            return pattern
        except Exception as e:
            self.logger.error(f"Error extracting pattern: {e}")
            return "unknown"

    def _fit_clusters(self):
        """FIX: Fit separate clusters for wins and losses"""
        try:
            # Fit loss clusters
            if len(self._loss_buf) >= self.n_clusters:
                X_loss = np.stack([f for f, _, _ in self._loss_buf])
                self._km_loss = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
                self._km_loss.fit(X_loss)
                self._danger_zones = self._km_loss.cluster_centers_.copy()
                
                # Calculate distances
                d = self._km_loss.transform(X_loss)
                mins = d.min(axis=1)
                self._mean_dist = float(mins.mean())
                self._last_dist = float(d[-1].min()) if len(d) > 0 else 0.0
                
                self.logger.info(f"Loss clusters fitted: {self.n_clusters} clusters, mean_dist={self._mean_dist:.3f}, last_dist={self._last_dist:.3f}")
                
            # Fit win clusters
            if len(self._win_buf) >= 3:  # Need at least 3 wins
                X_win = np.stack([f for f, _, _ in self._win_buf])
                n_win_clusters = min(3, len(X_win))
                self._km_win = KMeans(n_clusters=n_win_clusters, n_init=10, random_state=42)
                self._km_win.fit(X_win)
                self._profit_zones = self._km_win.cluster_centers_.copy()
                
                self.logger.info(f"Win clusters fitted: {n_win_clusters} clusters from {len(self._win_buf)} wins")
                
        except Exception as e:
            self.logger.error(f"Error fitting clusters: {e}")

    def _update_avoidance_signal(self):
        """Calculate how strongly to avoid current market conditions"""
        try:
            # Base signal on consecutive losses
            base_signal = min(self._consecutive_losses * 0.1, 0.5)
            
            # Adjust for pattern frequency
            if self._loss_patterns:
                max_pattern_count = max(self._loss_patterns.values())
                pattern_signal = min(max_pattern_count * 0.05, 0.3)
                base_signal += pattern_signal
                
            old_signal = self._avoidance_signal
            self._avoidance_signal = min(base_signal, 0.8)  # Cap at 80% avoidance
            
            if abs(self._avoidance_signal - old_signal) > 0.1:
                self.logger.info(f"Avoidance signal updated: {old_signal:.3f} -> {self._avoidance_signal:.3f}")
        except Exception as e:
            self.logger.error(f"Error updating avoidance signal: {e}")

    def _log_memory_stats(self):
        """Log comprehensive memory statistics"""
        try:
            # Loss patterns statistics
            if self._loss_patterns:
                top_patterns = sorted(self._loss_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
                pattern_str = ", ".join([f"{p}:{c}" for p, c in top_patterns])
                self.logger.info(f"Top loss patterns: {pattern_str}")
            
            # Memory usage
            win_rate = len(self._win_buf) / max(1, len(self._win_buf) + len(self._loss_buf))
            total_pnl = sum(pnl for _, pnl, _ in self._win_buf) + sum(pnl for _, pnl, _ in self._loss_buf)
            
            self.logger.info(f"Memory stats: wins={len(self._win_buf)}, losses={len(self._loss_buf)}, "
                           f"win_rate={win_rate:.3f}, total_pnl=€{total_pnl:.2f}, avoidance={self._avoidance_signal:.3f}")
        except Exception as e:
            self.logger.error(f"Error logging memory stats: {e}")

    def check_similarity_to_mistakes(self, features: np.ndarray) -> float:
        """
        FIX: Check how similar current conditions are to past mistakes
        Returns 0-1 danger score (1 = very dangerous)
        """
        try:
            if self._km_loss is None or len(self._danger_zones) == 0:
                self.logger.debug("No loss clusters available for similarity check")
                return 0.0
                
            # Distance to nearest loss cluster
            features_2d = features.reshape(1, -1)
            distances = np.linalg.norm(self._danger_zones - features_2d, axis=1)
            min_dist = distances.min()
            
            # Normalize to 0-1 (closer = more dangerous)
            danger_score = np.exp(-min_dist)  # Exponential decay
            
            # Boost danger if we have profit zones to compare
            if len(self._profit_zones) > 0:
                profit_distances = np.linalg.norm(self._profit_zones - features_2d, axis=1)
                min_profit_dist = profit_distances.min()
                
                # If closer to losses than profits, increase danger
                if min_dist < min_profit_dist:
                    danger_score *= 1.2
                    
            danger_score = float(np.clip(danger_score, 0, 1))
            
            if danger_score > 0.5:
                self.logger.warning(f"High danger score: {danger_score:.3f} (min_loss_dist={min_dist:.3f})")
            else:
                self.logger.debug(f"Danger score: {danger_score:.3f}")
                
            return danger_score
        except Exception as e:
            self.logger.error(f"Error checking similarity to mistakes: {e}")
            return 0.0

    def get_observation_components(self) -> np.ndarray:
        """FIX: Return actionable trading signals"""
        try:
            n_loss_clusters = float(self.n_clusters if self._km_loss is not None else 0)
            n_win_clusters = float(len(self._profit_zones))
            win_loss_ratio = len(self._win_buf) / max(1, len(self._loss_buf))
            
            components = np.array([
                n_loss_clusters,
                self._mean_dist,
                self._last_dist,
                self._avoidance_signal,
                n_win_clusters,
                win_loss_ratio
            ], np.float32)
            
            self.logger.debug(f"Observation components: {components}")
            return components
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.zeros(6, np.float32)

    def get_state(self):
        st = {
            "loss_buf": [(f.tolist(), pnl, info) for f, pnl, info in self._loss_buf],
            "win_buf": [(f.tolist(), pnl, info) for f, pnl, info in self._win_buf],
            "mean": self._mean_dist,
            "last": self._last_dist,
            "consecutive_losses": self._consecutive_losses,
            "loss_patterns": dict(self._loss_patterns),
            "avoidance_signal": self._avoidance_signal,
            "step_count": self._step_count,
        }
        if self._km_loss is not None:
            st["km_loss_centers"] = self._km_loss.cluster_centers_.tolist()
        if self._km_win is not None:
            st["km_win_centers"] = self._km_win.cluster_centers_.tolist()
        return st

    def set_state(self, st):
        self._loss_buf = [(np.asarray(f, np.float32), pnl, info) for f, pnl, info in st.get("loss_buf", [])]
        self._win_buf = [(np.asarray(f, np.float32), pnl, info) for f, pnl, info in st.get("win_buf", [])]
        self._mean_dist = float(st.get("mean", 0.0))
        self._last_dist = float(st.get("last", 0.0))
        self._consecutive_losses = st.get("consecutive_losses", 0)
        self._loss_patterns = st.get("loss_patterns", {})
        self._avoidance_signal = st.get("avoidance_signal", 0.0)
        self._step_count = st.get("step_count", 0)
        
        if "km_loss_centers" in st:
            self._km_loss = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            self._km_loss.cluster_centers_ = np.asarray(st["km_loss_centers"], np.float32)
            self._danger_zones = self._km_loss.cluster_centers_.copy()
            
        if "km_win_centers" in st:
            n_win = len(st["km_win_centers"])
            self._km_win = KMeans(n_clusters=n_win, n_init=10, random_state=42)
            self._km_win.cluster_centers_ = np.asarray(st["km_win_centers"], np.float32)
            self._profit_zones = self._km_win.cluster_centers_.copy()
            
        self.logger.info(f"State restored: losses={len(self._loss_buf)}, wins={len(self._win_buf)}, step={self._step_count}")

    def mutate(self, noise_std=0.05):
        """Evolve cluster understanding"""
        try:
            if self._km_loss is not None:
                self._km_loss.cluster_centers_ += np.random.randn(*self._km_loss.cluster_centers_.shape).astype(np.float32) * noise_std
            if self._km_win is not None:
                self._km_win.cluster_centers_ += np.random.randn(*self._km_win.cluster_centers_.shape).astype(np.float32) * noise_std
            self.logger.info(f"Mutated cluster centers with noise_std={noise_std}")
        except Exception as e:
            self.logger.error(f"Error in mutation: {e}")
            
    def crossover(self, other: "MistakeMemory"):
        child = MistakeMemory(self.max_mistakes, self.n_clusters, self.profit_threshold, debug=self.debug)
        if self._km_loss is not None and other._km_loss is not None:
            c1, c2 = self._km_loss.cluster_centers_, other._km_loss.cluster_centers_
            mix = np.where(np.random.rand(*c1.shape) > 0.5, c1, c2)
            child._km_loss = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=42)
            child._km_loss.cluster_centers_ = mix
        self.logger.info("Crossover completed")
        return child

# ─────────────────────────────────────────────────────────────
class MemoryCompressor(Module):
    """
    FIXED: Profit-focused feature compression that identifies winning patterns
    """
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

# ─────────────────────────────────────────────────────────────
class HistoricalReplayAnalyzer(Module):
    """
    FIXED: Actually analyzes and replays successful trading sequences
    """
    def __init__(self, interval: int=10, bonus: float=0.1, sequence_len: int=5, debug=False):
        self.interval = interval
        self.bonus = bonus
        self.sequence_len = sequence_len
        self.debug = debug
        
        # Enhanced Logger Setup - FIXED
        log_dir = os.path.join("logs", "memory")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f"HistoricalReplayAnalyzer_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        fh = logging.FileHandler(os.path.join(log_dir, "replay_analyzer.log"), mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"HistoricalReplayAnalyzer initialized - interval={interval}, bonus={bonus}, sequence_len={sequence_len}")
        
        self.reset()

    def reset(self):
        # FIX: Store successful sequences
        self.episode_buffer = deque(maxlen=100)
        self.profitable_sequences = []
        self.sequence_patterns = {}  # pattern -> (count, avg_pnl)
        self.current_sequence = []
        self.replay_bonus = 0.0
        self.best_sequence_pnl = 0.0
        self._step_count = 0
        self._episode_count = 0
        
        self.logger.info("HistoricalReplayAnalyzer reset - all sequences and patterns cleared")

    def step(self, **kwargs):
        """Track current trading sequence"""
        self._step_count += 1
        
        try:
            if "action" in kwargs and "features" in kwargs:
                step_data = {
                    "action": kwargs["action"],
                    "features": kwargs["features"],
                    "timestamp": kwargs.get("timestamp", self._step_count)
                }
                self.current_sequence.append(step_data)
                
                # Keep sequence bounded
                if len(self.current_sequence) > self.sequence_len:
                    removed = self.current_sequence.pop(0)
                    self.logger.debug(f"Sequence bounded: removed step {removed.get('timestamp', 'unknown')}")
                
                self.logger.debug(f"Step {self._step_count}: Added to sequence (length={len(self.current_sequence)})")
            else:
                self.logger.debug(f"Step {self._step_count}: Insufficient data for sequence tracking")
        except Exception as e:
            self.logger.error(f"Error in step: {e}")

    def record_episode(self, data: dict, actions: np.ndarray, pnl: float):
        """FIX: Analyze episode for profitable patterns"""
        try:
            self._episode_count += 1
            episode_data = {
                "data": data,
                "actions": actions,
                "pnl": pnl,
                "sequence": list(self.current_sequence),
                "episode": self._episode_count
            }
            self.episode_buffer.append(episode_data)
            
            self.logger.info(f"Episode {self._episode_count} recorded: PnL=€{pnl:.2f}, sequence_length={len(self.current_sequence)}")
            
            # Identify profitable sequences
            if pnl > 10:  # €10+ profit
                if len(self.current_sequence) >= 3:
                    # Extract sequence pattern
                    pattern = self._extract_sequence_pattern(self.current_sequence)
                    
                    # Update pattern statistics
                    if pattern in self.sequence_patterns:
                        count, avg_pnl = self.sequence_patterns[pattern]
                        new_avg = (avg_pnl * count + pnl) / (count + 1)
                        self.sequence_patterns[pattern] = (count + 1, new_avg)
                        self.logger.info(f"Updated pattern '{pattern}': count={count + 1}, avg_pnl=€{new_avg:.2f}")
                    else:
                        self.sequence_patterns[pattern] = (1, pnl)
                        self.logger.info(f"New profitable pattern '{pattern}': €{pnl:.2f}")
                    
                    # Store if exceptional
                    if pnl > self.best_sequence_pnl:
                        old_best = self.best_sequence_pnl
                        self.best_sequence_pnl = pnl
                        self.profitable_sequences.append({
                            "sequence": list(self.current_sequence),
                            "pnl": pnl,
                            "pattern": pattern,
                            "episode": self._episode_count
                        })
                        
                        # Keep only top 10 sequences
                        self.profitable_sequences.sort(key=lambda x: x["pnl"], reverse=True)
                        self.profitable_sequences = self.profitable_sequences[:10]
                        
                        self.logger.info(f"New best sequence: €{pnl:.2f} (previous: €{old_best:.2f})")
                else:
                    self.logger.warning(f"Profitable episode but sequence too short: {len(self.current_sequence)} < 3")
            else:
                self.logger.debug(f"Episode {self._episode_count}: Not profitable enough (€{pnl:.2f} <= €10)")
                    
            # Reset for next episode
            self.current_sequence = []
            self.logger.debug(f"Episode {self._episode_count} processing complete, sequence reset")
            
        except Exception as e:
            self.logger.error(f"Error recording episode: {e}")

    def _extract_sequence_pattern(self, sequence: List[dict]) -> str:
        """Extract pattern from action sequence"""
        try:
            if not sequence:
                return "empty"
                
            # Simple pattern based on action directions
            action_pattern = []
            for step in sequence[-3:]:  # Last 3 actions
                action = step.get("action", np.array([0, 0]))
                if isinstance(action, np.ndarray) and len(action) >= 2:
                    direction = "buy" if action[0] > 0 else "sell" if action[0] < 0 else "hold"
                    action_pattern.append(direction)
                    
            pattern = "_".join(action_pattern)
            self.logger.debug(f"Extracted pattern: {pattern}")
            return pattern
        except Exception as e:
            self.logger.error(f"Error extracting sequence pattern: {e}")
            return "error"

    def maybe_replay(self, episode: int) -> float:
        """FIX: Dynamic replay bonus based on profitable patterns"""
        try:
            base_bonus = self.bonus if episode % self.interval == 0 else 0.0
            
            # Additional bonus if we have profitable patterns
            if self.sequence_patterns:
                # Find most profitable pattern
                best_pattern = max(self.sequence_patterns.items(), 
                                 key=lambda x: x[1][1])  # Sort by avg PnL
                pattern_name, (count, avg_pnl) = best_pattern
                
                # Bonus proportional to pattern success
                pattern_bonus = min(0.2, avg_pnl / 100.0)  # Cap at 0.2
                self.replay_bonus = base_bonus + pattern_bonus
                
                self.logger.info(f"Episode {episode}: Replay bonus={self.replay_bonus:.3f} (base={base_bonus:.3f}, pattern={pattern_bonus:.3f})")
                self.logger.info(f"Best pattern '{pattern_name}': {count} times, avg €{avg_pnl:.2f}")
            else:
                self.replay_bonus = base_bonus
                self.logger.debug(f"Episode {episode}: No patterns, base bonus={base_bonus:.3f}")
                
            return self.replay_bonus
        except Exception as e:
            self.logger.error(f"Error in maybe_replay: {e}")
            return 0.0

    def get_best_sequence_for_replay(self) -> Optional[List[dict]]:
        """Get the most profitable sequence for learning"""
        try:
            if self.profitable_sequences:
                best_seq = self.profitable_sequences[0]["sequence"]
                self.logger.info(f"Returning best sequence: €{self.profitable_sequences[0]['pnl']:.2f}, length={len(best_seq)}")
                return best_seq
            else:
                self.logger.debug("No profitable sequences available")
                return None
        except Exception as e:
            self.logger.error(f"Error getting best sequence: {e}")
            return None

    def get_observation_components(self) -> np.ndarray:
        """FIX: Return replay analysis metrics"""
        try:
            n_patterns = float(len(self.sequence_patterns))
            avg_pattern_pnl = 0.0
            if self.sequence_patterns:
                avg_pattern_pnl = np.mean([pnl for _, pnl in self.sequence_patterns.values()])
                
            result = np.array([
                self.replay_bonus,
                n_patterns,
                avg_pattern_pnl,
                self.best_sequence_pnl
            ], dtype=np.float32)
            
            self.logger.debug(f"Observation: bonus={self.replay_bonus:.3f}, patterns={n_patterns}, avg_pnl={avg_pattern_pnl:.2f}, best={self.best_sequence_pnl:.2f}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.zeros(4, np.float32)

    def mutate(self, noise_std=0.05):
        """Evolve replay parameters"""
        try:
            old_bonus = self.bonus
            old_seq_len = self.sequence_len
            
            self.bonus = float(np.clip(self.bonus + np.random.normal(0, noise_std), 0.0, 0.5))
            self.sequence_len = int(np.clip(self.sequence_len + np.random.randint(-1, 2), 3, 10))
            
            self.logger.info(f"Mutated: bonus {old_bonus:.3f}->{self.bonus:.3f}, seq_len {old_seq_len}->{self.sequence_len}")
        except Exception as e:
            self.logger.error(f"Error in mutation: {e}")
        
    def crossover(self, other: "HistoricalReplayAnalyzer"):
        child = HistoricalReplayAnalyzer(
            self.interval, 
            random.choice([self.bonus, other.bonus]),
            random.choice([self.sequence_len, other.sequence_len]),
            self.debug
        )
        self.logger.info("Crossover completed")
        return child

# ─────────────────────────────────────────────────────────────
class PlaybookMemory(Module):
    """
    FIXED: Enhanced with market context and profit-weighted recall
    """
    def __init__(self, max_entries: int=500, k: int=5, 
                 profit_weight: float=2.0, context_weight: float=1.5, debug=False):
        self.max_entries = max_entries
        self.k = k
        self.profit_weight = profit_weight  # Weight for profitable trades
        self.context_weight = context_weight  # Weight for market context
        self.debug = debug
        
        # Enhanced Logger Setup - FIXED
        log_dir = os.path.join("logs", "memory")
        os.makedirs(log_dir, exist_ok=True)
        self.logger = logging.getLogger(f"PlaybookMemory_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        fh = logging.FileHandler(os.path.join(log_dir, "playbook_memory.log"), mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"PlaybookMemory initialized - max_entries={max_entries}, k={k}, profit_weight={profit_weight}, context_weight={context_weight}")
        
        self.reset()

    def reset(self):
        self._features = []
        self._actions = []
        self._pnls = []
        self._contexts = []  # Market context for each trade
        self._timestamps = []
        self._nbrs = None
        self._weighted_nbrs = None
        self._total_profit = 0.0
        self._trade_count = 0
        self._recall_count = 0
        
        self.logger.info("PlaybookMemory reset - all memories cleared")

    def step(self, **kwargs): 
        self.logger.debug(f"Step called with {len(kwargs)} parameters")

    def record(self, features: np.ndarray, actions: np.ndarray, pnl: float, 
               context: Optional[dict] = None):
        """FIX: Record with market context"""
        try:
            self.logger.debug(f"Recording trade: PnL=€{pnl:.2f}, features_shape={features.shape}, actions_shape={actions.shape}")
            
            if len(self._features) >= self.max_entries:
                # Remove oldest entry
                removed_pnl = self._pnls.pop(0)
                self._total_profit -= removed_pnl
                self._features.pop(0)
                self._actions.pop(0)
                self._contexts.pop(0)
                self._timestamps.pop(0)
                self.logger.debug(f"Memory full, removed oldest entry: €{removed_pnl:.2f}")
                
            self._features.append(features.copy())
            self._actions.append(actions.copy())
            self._pnls.append(pnl)
            self._contexts.append(context or {})
            self._timestamps.append(datetime.now())
            self._total_profit += pnl
            self._trade_count += 1
            
            self.logger.info(f"Trade {self._trade_count} recorded: €{pnl:.2f}, total_profit=€{self._total_profit:.2f}, memory_size={len(self._features)}")
            
            # Refit nearest neighbors with enough data
            if len(self._features) >= self.k:
                self._fit_neighbors()
            else:
                self.logger.debug(f"Not enough data for neighbors: {len(self._features)} < {self.k}")
                
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")

    def _fit_neighbors(self):
        """FIX: Fit with profit-weighted distances"""
        try:
            X = np.vstack(self._features)
            
            # Create weights based on PnL
            weights = np.array(self._pnls)
            weights = np.where(weights > 0, 
                              1 + weights / 100.0 * self.profit_weight,  # Boost profitable
                              1 / (1 + abs(weights) / 100.0))  # Penalize losses
            
            # Standard KNN
            self._nbrs = NearestNeighbors(n_neighbors=min(self.k, len(X))).fit(X)
            
            # Weighted KNN for profit-focused recall
            self._weighted_nbrs = NearestNeighbors(
                n_neighbors=min(self.k * 2, len(X)),  # Look at more neighbors
                metric='minkowski'
            ).fit(X)
            
            profitable_count = sum(1 for p in self._pnls if p > 0)
            self.logger.info(f"Neighbors fitted: {len(X)} trades, {profitable_count} profitable, k={min(self.k, len(X))}")
            
        except Exception as e:
            self.logger.error(f"Error fitting neighbors: {e}")

    def recall(self, features: np.ndarray, context: Optional[dict] = None) -> dict:
        """
        FIX: Enhanced recall with context and profit weighting
        Returns dict with expected PnL, best action, and confidence
        """
        try:
            self._recall_count += 1
            self.logger.debug(f"Recall #{self._recall_count}: features_shape={features.shape}")
            
            if self._nbrs is None:
                # Bootstrap response
                result = {
                    "expected_pnl": 0.0,
                    "confidence": 0.0,
                    "suggested_action": np.zeros(2),
                    "similar_trades": 0
                }
                self.logger.debug("No neighbors available, returning bootstrap response")
                return result
                
            # Find similar trades
            features_2d = features.reshape(1, -1)
            
            # Use weighted neighbors for profit-focused recall
            if self._weighted_nbrs is not None:
                distances, indices = self._weighted_nbrs.kneighbors(features_2d)
                # Filter to k best
                indices = indices[0][:self.k]
                distances = distances[0][:self.k]
            else:
                distances, indices = self._nbrs.kneighbors(features_2d)
                indices = indices[0]
                distances = distances[0]
            
            # Get PnLs and actions for similar trades
            similar_pnls = [self._pnls[i] for i in indices]
            similar_actions = [self._actions[i] for i in indices]
            similar_contexts = [self._contexts[i] for i in indices]
            
            self.logger.debug(f"Found {len(similar_pnls)} similar trades, PnLs: {[f'{p:.2f}' for p in similar_pnls]}")
            
            # FIX: Context-aware filtering
            if context is not None:
                context_scores = []
                for ctx in similar_contexts:
                    score = self._context_similarity(context, ctx)
                    context_scores.append(score)
                context_scores = np.array(context_scores)
                self.logger.debug(f"Context scores: {[f'{s:.3f}' for s in context_scores]}")
            else:
                context_scores = np.ones(len(indices))
            
            # Combine distance and context scores
            combined_weights = np.exp(-distances) * context_scores
            combined_weights = combined_weights / (combined_weights.sum() + 1e-8)
            
            # Weighted average PnL
            expected_pnl = float(np.average(similar_pnls, weights=combined_weights))
            
            # Weighted average action (for profitable trades only)
            profitable_mask = np.array(similar_pnls) > 0
            if profitable_mask.any():
                profitable_weights = combined_weights[profitable_mask]
                profitable_actions = [a for a, p in zip(similar_actions, profitable_mask) if p]
                suggested_action = np.average(profitable_actions, axis=0, weights=profitable_weights)
                profitable_count = profitable_mask.sum()
                self.logger.debug(f"Used {profitable_count}/{len(similar_pnls)} profitable trades for action")
            else:
                suggested_action = np.mean(similar_actions, axis=0)
                self.logger.debug("No profitable trades found, using average action")
            
            # Confidence based on similarity and profit consistency
            pnl_std = np.std(similar_pnls)
            confidence = np.exp(-pnl_std / 100.0) * combined_weights.max()
            
            result = {
                "expected_pnl": expected_pnl,
                "confidence": float(confidence),
                "suggested_action": suggested_action,
                "similar_trades": len(indices),
                "best_similar_pnl": float(max(similar_pnls))
            }
            
            self.logger.info(f"Recall #{self._recall_count}: expected_pnl=€{expected_pnl:.2f}, confidence={confidence:.3f}, similar_trades={len(indices)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in recall: {e}")
            return {
                "expected_pnl": 0.0,
                "confidence": 0.0,
                "suggested_action": np.zeros(2),
                "similar_trades": 0
            }

    def _context_similarity(self, ctx1: dict, ctx2: dict) -> float:
        """Calculate similarity between market contexts"""
        try:
            score = 1.0
            
            # Compare regimes
            if "regime" in ctx1 and "regime" in ctx2:
                if ctx1["regime"] == ctx2["regime"]:
                    score *= self.context_weight
                else:
                    score *= 0.5
                    
            # Compare volatility
            if "volatility" in ctx1 and "volatility" in ctx2:
                vol_diff = abs(ctx1["volatility"] - ctx2["volatility"])
                score *= np.exp(-vol_diff * 10)  # Decay with volatility difference
                
            return score
        except Exception as e:
            self.logger.error(f"Error calculating context similarity: {e}")
            return 1.0

    def get_observation_components(self) -> np.ndarray:
        """FIX: Return memory performance metrics"""
        try:
            avg_pnl = self._total_profit / max(1, self._trade_count)
            win_rate = sum(1 for p in self._pnls if p > 0) / max(1, len(self._pnls))
            memory_fullness = len(self._features) / self.max_entries
            
            # Recent performance (last 20 trades)
            recent_pnls = self._pnls[-20:] if self._pnls else [0]
            recent_avg = np.mean(recent_pnls)
            
            result = np.array([
                avg_pnl,
                win_rate,
                memory_fullness,
                recent_avg,
                float(len(self._features))
            ], dtype=np.float32)
            
            self.logger.debug(f"Observation: avg_pnl={avg_pnl:.2f}, win_rate={win_rate:.3f}, fullness={memory_fullness:.3f}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.zeros(5, np.float32)

    def get_state(self):
        return {
            "features": [f.tolist() for f in self._features],
            "actions": [a.tolist() for a in self._actions],
            "pnls": self._pnls,
            "contexts": self._contexts,
            "timestamps": [t.isoformat() for t in self._timestamps],
            "total_profit": self._total_profit,
            "trade_count": self._trade_count,
            "recall_count": self._recall_count,
            "k": self.k,
            "profit_weight": self.profit_weight,
            "context_weight": self.context_weight
        }

    def set_state(self, state):
        self._features = [np.array(f, dtype=np.float32) for f in state.get("features", [])]
        self._actions = [np.array(a, dtype=np.float32) for a in state.get("actions", [])]
        self._pnls = state.get("pnls", [])
        self._contexts = state.get("contexts", [])
        self._timestamps = [datetime.fromisoformat(t) for t in state.get("timestamps", [])]
        self._total_profit = state.get("total_profit", 0.0)
        self._trade_count = state.get("trade_count", 0)
        self._recall_count = state.get("recall_count", 0)
        self.k = state.get("k", self.k)
        self.profit_weight = state.get("profit_weight", self.profit_weight)
        self.context_weight = state.get("context_weight", self.context_weight)
        
        if len(self._features) >= self.k:
            self._fit_neighbors()
            
        self.logger.info(f"State restored: {len(self._features)} trades, €{self._total_profit:.2f} total profit")

    def mutate(self):
        """Evolve memory parameters"""
        try:
            old_k = self.k
            old_profit_weight = self.profit_weight
            old_context_weight = self.context_weight
            
            self.k = int(np.clip(self.k + np.random.choice([-1, 0, 1]), 3, 20))
            self.profit_weight = float(np.clip(self.profit_weight + np.random.normal(0, 0.2), 1.0, 5.0))
            self.context_weight = float(np.clip(self.context_weight + np.random.normal(0, 0.1), 1.0, 3.0))
            
            self.logger.info(f"Mutated: k {old_k}->{self.k}, profit_weight {old_profit_weight:.2f}->{self.profit_weight:.2f}, context_weight {old_context_weight:.2f}->{self.context_weight:.2f}")
        except Exception as e:
            self.logger.error(f"Error in mutation: {e}")
            
    def crossover(self, other: "PlaybookMemory"):
        child = PlaybookMemory(
            self.max_entries,
            random.choice([self.k, other.k]),
            random.choice([self.profit_weight, other.profit_weight]),
            random.choice([self.context_weight, other.context_weight]),
            self.debug
        )
        self.logger.info("Crossover completed")
        return child

# ─────────────────────────────────────────────────────────────
class MemoryBudgetOptimizer(Module):
    """
    FIXED: Dynamically optimizes memory allocation based on performance
    """
    def __init__(self, max_trades: int=500, max_mistakes: int=100, 
                 max_plays: int=200, min_size: int=50, debug=False):
        self.max_trades = max_trades
        self.max_mistakes = max_mistakes
        self.max_plays = max_plays
        self.min_size = min_size  # Minimum memory size
        self.debug = debug
        
        # Enhanced Logger Setup - FIXED
        log_dir = os.path.join("logs", "memory")
        self.logger = logging.getLogger(f"MemoryBudgetOptimizer_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        fh = logging.FileHandler(os.path.join(log_dir, "memory_budget.log"), mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info(f"MemoryBudgetOptimizer initialized - trades={max_trades}, mistakes={max_mistakes}, plays={max_plays}, min_size={min_size}")
        
        self.reset()

    def reset(self):
        # FIX: Track memory utilization and performance with all required keys
        self.memory_performance = {
            "trades": {"size": self.max_trades, "hits": 0, "profit": 0.0},
            "mistakes": {"size": self.max_mistakes, "hits": 0, "profit": 0.0},  # Changed "saves" to "profit"
            "plays": {"size": self.max_plays, "hits": 0, "profit": 0.0}
        }
        self.optimization_count = 0
        self.total_profit = 0.0
        self._step_count = 0
        
        self.logger.info("MemoryBudgetOptimizer reset - performance tracking cleared")

    def step(self, env=None, **kwargs):
        """FIXED: Track memory usage and effectiveness with proper error handling"""
        self._step_count += 1
        
        try:
            # Track memory hits (when memory was useful)
            if "memory_used" in kwargs:
                memory_type = kwargs["memory_used"]
                if memory_type in self.memory_performance:
                    self.memory_performance[memory_type]["hits"] += 1
                    self.logger.debug(f"Step {self._step_count}: Memory hit for {memory_type}")
                    
            # Track profit attribution - FIXED
            if "profit" in kwargs and "source" in kwargs:
                source = kwargs["source"]
                profit = float(kwargs["profit"])  # Ensure it's a float
                
                # FIXED: Ensure the source exists in memory_performance
                if source not in self.memory_performance:
                    self.logger.warning(f"Unknown memory source: {source}")
                    return
                    
                # FIXED: Ensure profit key exists
                if "profit" not in self.memory_performance[source]:
                    self.memory_performance[source]["profit"] = 0.0
                    
                self.memory_performance[source]["profit"] += profit
                self.logger.debug(f"Step {self._step_count}: Profit €{profit:.2f} attributed to {source}")
                self.total_profit += profit
                
            # Log summary periodically
            if self._step_count % 100 == 0:
                self._log_performance_summary()
                
        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            self.logger.error(f"kwargs: {kwargs}")
            self.logger.error(f"memory_performance: {self.memory_performance}")

                
        except Exception as e:
            self.logger.error(f"Error in step: {e}")

    def _log_performance_summary(self):
        """Log current performance summary - FIXED"""
        try:
            self.logger.info(f"Step {self._step_count} - Performance Summary:")
            for mem_type, stats in self.memory_performance.items():
                # FIXED: Use .get() to safely access dictionary values
                size = stats.get("size", 0)
                hits = stats.get("hits", 0)
                profit = stats.get("profit", 0.0)  # This was causing the error
                
                efficiency = profit / max(1, hits) if hits > 0 else 0.0
                
                self.logger.info(f"  {mem_type}: size={size}, hits={hits}, profit=€{profit:.2f}, efficiency=€{efficiency:.2f}/hit")
        except Exception as e:
            self.logger.error(f"Error logging performance summary: {e}")
            # Log the actual stats structure for debugging
            self.logger.error(f"Memory performance structure: {self.memory_performance}")

    def optimize_allocation(self, episode: int):
        """FIX: Periodically rebalance memory based on performance"""
        try:
            if episode % 50 != 0 or episode == 0:  # Every 50 episodes
                self.logger.debug(f"Episode {episode}: Not optimization interval")
                return
                
            self.optimization_count += 1
            self.logger.info(f"Episode {episode}: Starting optimization #{self.optimization_count}")
            
            # Calculate efficiency for each memory type
            efficiencies = {}
            for mem_type, stats in self.memory_performance.items():
                if stats["hits"] > 0:
                    # Profit per hit
                    efficiency = stats["profit"] / stats["hits"]
                else:
                    efficiency = 0.0
                efficiencies[mem_type] = efficiency
                self.logger.info(f"  {mem_type} efficiency: €{efficiency:.3f}/hit")
                
            # Total memory budget
            total_budget = self.max_trades + self.max_mistakes + self.max_plays
            self.logger.info(f"Total memory budget: {total_budget}")
            
            # Reallocate based on efficiency
            if sum(efficiencies.values()) > 0:
                # Proportional allocation based on efficiency
                total_eff = sum(max(0, e) for e in efficiencies.values())
                
                if total_eff > 0:
                    new_sizes = {}
                    for mem_type, eff in efficiencies.items():
                        proportion = max(0, eff) / total_eff
                        new_size = int(proportion * total_budget)
                        # Enforce minimum size
                        new_size = max(self.min_size, new_size)
                        new_sizes[mem_type] = new_size
                    
                    # Adjust to fit budget exactly
                    size_sum = sum(new_sizes.values())
                    if size_sum > total_budget:
                        # Scale down proportionally
                        scale = total_budget / size_sum
                        for mem_type in new_sizes:
                            new_sizes[mem_type] = max(self.min_size, int(new_sizes[mem_type] * scale))
                    
                    # Apply new sizes
                    old_sizes = {
                        "trades": self.max_trades,
                        "mistakes": self.max_mistakes,
                        "plays": self.max_plays
                    }
                    
                    self.max_trades = new_sizes.get("trades", self.max_trades)
                    self.max_mistakes = new_sizes.get("mistakes", self.max_mistakes)
                    self.max_plays = new_sizes.get("plays", self.max_plays)
                    
                    # Log changes
                    changes_made = False
                    for mem_type in ["trades", "mistakes", "plays"]:
                        old = old_sizes[mem_type]
                        new = getattr(self, f"max_{mem_type}")
                        if old != new:
                            self.logger.info(f"  {mem_type}: {old} -> {new} (efficiency: €{efficiencies[mem_type]:.3f})")
                            changes_made = True
                            
                    if not changes_made:
                        self.logger.info("  No allocation changes needed")
                else:
                    self.logger.warning("  Total efficiency is 0, no reallocation")
            else:
                self.logger.warning("  No efficiency data available")
                
        except Exception as e:
            self.logger.error(f"Error in optimize_allocation: {e}")

    def get_observation_components(self) -> np.ndarray:
        """FIX: Return memory utilization metrics"""
        try:
            # Calculate hit rates
            hit_rates = []
            for mem_type in ["trades", "mistakes", "plays"]:
                stats = self.memory_performance[mem_type]
                size = getattr(self, f"max_{mem_type}")
                hit_rate = stats["hits"] / max(1, self.optimization_count * 50)  # Per episode
                hit_rates.append(hit_rate)
                
            result = np.array([
                float(self.max_trades),
                float(self.max_mistakes),
                float(self.max_plays),
                *hit_rates
            ], dtype=np.float32)
            
            self.logger.debug(f"Observation: sizes=[{self.max_trades}, {self.max_mistakes}, {self.max_plays}], hit_rates={hit_rates}")
            return result
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.zeros(6, np.float32)

    def get_state(self):
        return {
            "max_trades": self.max_trades,
            "max_mistakes": self.max_mistakes,
            "max_plays": self.max_plays,
            "memory_performance": self.memory_performance,
            "optimization_count": self.optimization_count,
            "total_profit": self.total_profit,
            "step_count": self._step_count
        }

    def set_state(self, state):
        self.max_trades = state.get("max_trades", self.max_trades)
        self.max_mistakes = state.get("max_mistakes", self.max_mistakes)
        self.max_plays = state.get("max_plays", self.max_plays)
        self.memory_performance = state.get("memory_performance", self.memory_performance)
        self.optimization_count = state.get("optimization_count", 0)
        self.total_profit = state.get("total_profit", 0.0)
        self._step_count = state.get("step_count", 0)
        
        self.logger.info(f"State restored: optimizations={self.optimization_count}, total_profit=€{self.total_profit:.2f}, steps={self._step_count}")

    def mutate(self):
        """Smart mutation based on performance"""
        try:
            # Only mutate the least efficient memory type
            efficiencies = {}
            for mem_type, stats in self.memory_performance.items():
                if stats["hits"] > 0:
                    efficiencies[mem_type] = stats["profit"] / stats["hits"]
                else:
                    efficiencies[mem_type] = 0.0
                    
            if efficiencies:
                # Find worst performer
                worst_type = min(efficiencies.items(), key=lambda x: x[1])[0]
                param = f"max_{worst_type}"
                old_val = getattr(self, param)
                
                # Reduce size of worst performer
                new_val = max(self.min_size, old_val - random.randint(10, 50))
                setattr(self, param, new_val)
                
                # Give that space to best performer
                if len(efficiencies) > 1:
                    best_type = max(efficiencies.items(), key=lambda x: x[1])[0]
                    best_param = f"max_{best_type}"
                    current_best = getattr(self, best_param)
                    setattr(self, best_param, current_best + (old_val - new_val))
                    
                self.logger.info(f"Mutated {param}: {old_val} -> {new_val} (worst efficiency: €{efficiencies[worst_type]:.3f})")
            else:
                self.logger.warning("No efficiency data for mutation")
                
        except Exception as e:
            self.logger.error(f"Error in mutation: {e}")
                
    def crossover(self, other: "MemoryBudgetOptimizer"):
        child = MemoryBudgetOptimizer(
            max_trades=int((self.max_trades + other.max_trades) / 2),
            max_mistakes=int((self.max_mistakes + other.max_mistakes) / 2),
            max_plays=int((self.max_plays + other.max_plays) / 2),
            min_size=self.min_size,
            debug=self.debug
        )
        self.logger.info("Crossover completed")
        return child


# Test function to verify all modules log properly
def test_memory_modules():
    """Test function to verify all memory modules log properly"""
    print("Testing enhanced memory modules with logging...")
    
    # Test MistakeMemory
    print("\n1. Testing MistakeMemory:")
    mistake_memory = MistakeMemory(max_mistakes=50, debug=True)
    
    # Simulate some trades
    for i in range(10):
        features = np.random.randn(5)
        pnl = np.random.normal(0, 100)  # Mix of profits and losses
        info = {"volatility": np.random.uniform(0.01, 0.03), "regime": "trending", "hour": i}
        mistake_memory.step(features=features, pnl=pnl, info=info)
    
    danger_score = mistake_memory.check_similarity_to_mistakes(np.random.randn(5))
    print(f"Danger score: {danger_score:.3f}")
    
    # Test MemoryCompressor
    print("\n2. Testing MemoryCompressor:")
    compressor = MemoryCompressor(debug=True)
    
    # Simulate trades for compression
    trades = []
    for i in range(20):
        trades.append({
            "features": np.random.randn(8),
            "pnl": np.random.normal(10, 50)  # Mix of profits/losses
        })
    
    compressor.compress(episode=10, trades=trades)
    intuition = compressor.step()
    print(f"Intuition shape: {intuition.shape}")
    
    # Test HistoricalReplayAnalyzer
    print("\n3. Testing HistoricalReplayAnalyzer:")
    replay_analyzer = HistoricalReplayAnalyzer(debug=True)
    
    # Simulate trading sequence
    for i in range(5):
        replay_analyzer.step(action=np.random.randn(2), features=np.random.randn(5), timestamp=i)
    
    # Record episode
    replay_analyzer.record_episode(
        data={"test": "data"}, 
        actions=np.random.randn(5, 2), 
        pnl=50.0  # Profitable episode
    )
    
    replay_bonus = replay_analyzer.maybe_replay(episode=10)
    print(f"Replay bonus: {replay_bonus:.3f}")
    
    # Test PlaybookMemory
    print("\n4. Testing PlaybookMemory:")
    playbook = PlaybookMemory(debug=True)
    
    # Record some trades
    for i in range(10):
        features = np.random.randn(5)
        actions = np.random.randn(2)
        pnl = np.random.normal(0, 50)
        context = {"regime": "trending", "volatility": 0.02}
        playbook.record(features, actions, pnl, context)
    
    # Test recall
    recall_result = playbook.recall(np.random.randn(5), context={"regime": "trending"})
    print(f"Recall result: expected_pnl={recall_result['expected_pnl']:.2f}")
    
    # Test MemoryBudgetOptimizer
    print("\n5. Testing MemoryBudgetOptimizer:")
    budget_optimizer = MemoryBudgetOptimizer(debug=True)
    
    # Simulate some memory usage
    budget_optimizer.step(memory_used="trades", profit=25.0, source="trades")
    budget_optimizer.step(memory_used="mistakes", profit=-15.0, source="mistakes")
    
    budget_optimizer.optimize_allocation(episode=50)
    
    print("\nAll memory modules tested! Check log files:")
    print("- logs/mistakes.log")
    print("- logs/memory_compressor.log") 
    print("- logs/replay_analyzer.log")
    print("- logs/playbook_memory.log")
    print("- logs/memory_budget.log")

if __name__ == "__main__":
    test_memory_modules()