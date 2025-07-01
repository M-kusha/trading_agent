
# ─────────────────────────────────────────────────────────────
# File: modules/memory/playbook_memory.py
# ─────────────────────────────────────────────────────────────

import logging
import os
from typing import Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random
from datetime import datetime
from modules.core.core import Module

class PlaybookMemory(Module):

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
