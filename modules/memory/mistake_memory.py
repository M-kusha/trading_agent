# ─────────────────────────────────────────────────────────────
# File: modules/memory/mistake_memory.py
# ─────────────────────────────────────────────────────────────

import logging
import os
from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans
from modules.core.core import Module

# ─────────────────────────────────────────────────────────────
class MistakeMemory(Module):
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
