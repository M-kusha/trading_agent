# modules/memory/shared/memory_store.py
"""
Unified Memory Store
Central storage backend for all memory components with efficient indexing and retrieval.
"""

from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class MemoryEntry:
    """Single memory entry with metadata."""
    id: str
    timestamp: float
    features: np.ndarray
    action: np.ndarray
    pnl: float
    context: Dict[str, Any]
    metadata: Dict[str, Any]
    importance: float = 0.5
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)

    def __lt__(self, other: "MemoryEntry") -> bool:
        """Order by importance for heap operations (min-heap)."""
        return self.importance < other.importance


class UnifiedMemoryStore:
    """
    Central memory storage with efficient operations.

    Features:
    - Fast insertion and retrieval
    - Similarity search using cosine similarity
    - Automatic memory management under pressure
    - Thread-safe operations
    - (Optional) Persistence to disk (pickle)
    """

    # Numerical stability for norms/divisions
    _EPS: float = 1e-8

    def __init__(self, max_size: int = 10000, batch_size: int = 32) -> None:
        """
        Initialize memory store.

        Args:
            max_size: Maximum number of memories to store.
            batch_size: Default batch size for operations.
        """
        self.max_size = int(max_size)
        self.batch_size = int(batch_size)

        # Primary storage
        self.memories: Dict[str, MemoryEntry] = {}
        self.memory_list: List[MemoryEntry] = []

        # Indices for fast lookup / housekeeping
        self.timestamp_index: List[Tuple[float, str]] = []  # (timestamp, id)
        self.pnl_index: List[Tuple[float, str]] = []        # (pnl, id)
        self.importance_heap: List[MemoryEntry] = []        # min-heap (by importance)

        # Feature matrix for similarity search (row-aligned with feature_ids)
        self.feature_matrix: Optional[np.ndarray] = None
        self.feature_ids: List[str] = []
        self._feature_matrix_dirty: bool = False

        # Statistics
        self.total_stored: int = 0
        self.total_retrieved: int = 0
        self.total_pruned: int = 0

        # Thread safety
        self._lock = threading.RLock()

        # Memory pressure management
        self.pressure_threshold: float = 0.9
        self.cleanup_ratio: float = 0.8

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def add(self, entry: Dict[str, Any]) -> str:
        """
        Add a single memory entry.

        Args:
            entry: Memory entry dictionary.

        Returns:
            The generated entry ID.
        """
        with self._lock:
            entry_id = self._add_unlocked(entry)
            # Pressure check after insert
            if self.size() > self.max_size:
                self._auto_prune()
            return entry_id

    async def add_batch(self, entries: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple memory entries.

        Args:
            entries: List of memory entries.

        Returns:
            List of generated entry IDs.
        """
        ids: List[str] = []
        with self._lock:
            for entry in entries:
                entry_id = self._add_unlocked(entry)
                ids.append(entry_id)
            # Single pressure check for the whole batch
            if self.size() > self.max_size:
                self._auto_prune()
        return ids

    def get(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get memory by ID.

        Args:
            entry_id: Memory entry ID.

        Returns:
            Memory entry or None.
        """
        with self._lock:
            memory = self.memories.get(entry_id)
            if memory is not None:
                memory.access_count += 1
                memory.last_accessed = time.time()
                self.total_retrieved += 1
            return memory

    def query(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        filter_fn: Optional[Callable[[MemoryEntry], bool]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Query memories by cosine similarity.

        Args:
            query_vector: Query vector for similarity search.
            k: Number of results to return.
            filter_fn: Optional predicate over MemoryEntry to include/exclude results.

        Returns:
            List of similar memories (as dictionaries).
        """
        if k <= 0:
            return []

        with self._lock:
            # Rebuild matrix if needed
            if self._feature_matrix_dirty:
                self._rebuild_feature_matrix()

            if self.feature_matrix is None or self.feature_matrix.size == 0:
                return []

            # Prepare query
            query = self._ensure_row_vector(query_vector, self.feature_matrix.shape[1])

            # Similarities: shape (1, n_memories)
            similarities = self._calculate_similarities(query, self.feature_matrix)

            # Top-k indices by similarity (descending)
            k_eff = min(k, similarities.shape[1])
            top_indices = np.argpartition(similarities[0], -k_eff)[-k_eff:]
            # Sort those top candidates
            top_indices = top_indices[np.argsort(similarities[0, top_indices])[::-1]]

            # Collect results
            results: List[Dict[str, Any]] = []
            now = time.time()
            for idx in top_indices:
                if 0 <= idx < len(self.feature_ids):
                    mem_id = self.feature_ids[idx]
                    memory = self.memories.get(mem_id)
                    if memory is None:
                        continue
                    if filter_fn and not filter_fn(memory):
                        continue

                    # Update access info
                    memory.access_count += 1
                    memory.last_accessed = now

                    results.append(self._memory_to_dict(memory))

            self.total_retrieved += len(results)
            return results

    def get_recent(self, n: int) -> List[Dict[str, Any]]:
        """
        Get the n most recent memories.

        Args:
            n: Number of memories to retrieve.

        Returns:
            List of recent memories (as dictionaries).
        """
        if n <= 0:
            return []
        with self._lock:
            # Fast path when few entries
            if not self.timestamp_index:
                return []

            # Get the top-n timestamps without sorting entire list
            # (nlargest is stable and efficient for large lists)
            top = heapq.nlargest(n, self.timestamp_index, key=lambda x: x[0])

            results: List[Dict[str, Any]] = []
            for _, entry_id in top:
                memory = self.memories.get(entry_id)
                if memory is not None:
                    results.append(self._memory_to_dict(memory))
            return results

    def get_by_filter(
        self,
        filter_fn: Callable[[Dict[str, Any]], bool],
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get memories matching a dict-level filter.

        Args:
            filter_fn: Predicate that accepts memory dicts.
            limit: Optional limit on number of results.

        Returns:
            List of matching memories (as dictionaries).
        """
        with self._lock:
            results: List[Dict[str, Any]] = []
            remaining = limit if (limit is not None and limit > 0) else None

            for memory in self.memory_list:
                mem_dict = self._memory_to_dict(memory)
                if filter_fn(mem_dict):
                    results.append(mem_dict)
                    if remaining is not None:
                        remaining -= 1
                        if remaining <= 0:
                            break
            return results

    def update_importance(self, entry_id: str, importance: float) -> bool:
        """
        Update memory importance.

        Args:
            entry_id: Memory entry ID.
            importance: New importance value (0..1).

        Returns:
            True if updated; False if entry not found.
        """
        with self._lock:
            memory = self.memories.get(entry_id)
            if memory is None:
                return False
            memory.importance = float(np.clip(importance, 0.0, 1.0))
            # Re-heapify because key changed
            heapq.heapify(self.importance_heap)
            return True

    def cleanup(self, keep_ratio: float = 0.8) -> int:
        """
        Clean up memory store to reduce size.

        Args:
            keep_ratio: Ratio of max_size to keep (0..1).

        Returns:
            Number of memories removed.
        """
        keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
        with self._lock:
            target_size = int(self.max_size * keep_ratio)
            current_size = self.size()
            if current_size <= target_size:
                return 0

            to_remove = current_size - target_size

            # Score memories (lower score => more likely to remove)
            scores: List[Tuple[float, str]] = [
                (self._calculate_retention_score(m), m.id) for m in self.memory_list
            ]
            scores.sort(key=lambda x: x[0])  # ascending: remove lowest scores first

            removed = 0
            for _, entry_id in scores[:to_remove]:
                if self._remove_memory(entry_id):
                    removed += 1

            self.total_pruned += removed
            return removed

    def size(self) -> int:
        """Get current number of memories."""
        # No lock necessary for a single dict len, but keep it consistent
        return len(self.memories)

    def utilization(self) -> float:
        """Get memory utilization ratio."""
        return self.size() / max(1, self.max_size)

    def get_statistics(self) -> Dict[str, Any]:
        """Get store statistics."""
        with self._lock:
            avg_importance = float(
                np.mean([m.importance for m in self.memory_list])
            ) if self.memory_list else 0.0
            avg_access = float(
                np.mean([m.access_count for m in self.memory_list])
            ) if self.memory_list else 0.0

            return {
                "size": self.size(),
                "max_size": self.max_size,
                "utilization": self.utilization(),
                "total_stored": self.total_stored,
                "total_retrieved": self.total_retrieved,
                "total_pruned": self.total_pruned,
                "avg_importance": avg_importance,
                "avg_access_count": avg_access,
            }

    def save(self, filepath: str) -> bool:
        """
        Save memory store to disk using pickle.

        Args:
            filepath: Path to save file.

        Returns:
            True on success, False otherwise.
        """
        # NOTE: Pickle can be unsafe with untrusted inputs. Only load what you saved yourself.
        try:
            import pickle  # local import to keep optional

            with self._lock:
                state = {
                    "memories": self.memories,
                    "memory_list": self.memory_list,
                    "timestamp_index": self.timestamp_index,
                    "pnl_index": self.pnl_index,
                    "total_stored": self.total_stored,
                    "total_retrieved": self.total_retrieved,
                    "total_pruned": self.total_pruned,
                }
                with open(filepath, "wb") as f:
                    pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            return True
        except Exception:
            return False

    def load(self, filepath: str) -> bool:
        """
        Load memory store from disk (pickle).

        Args:
            filepath: Path to load file.

        Returns:
            True on success, False otherwise.
        """
        try:
            import pickle  # local import to keep optional

            with open(filepath, "rb") as f:
                state = pickle.load(f)

            with self._lock:
                self.memories = state.get("memories", {})
                self.memory_list = state.get("memory_list", [])
                self.timestamp_index = state.get("timestamp_index", [])
                self.pnl_index = state.get("pnl_index", [])
                self.total_stored = int(state.get("total_stored", 0))
                self.total_retrieved = int(state.get("total_retrieved", 0))
                self.total_pruned = int(state.get("total_pruned", 0))

                # Rebuild heap
                self.importance_heap = list(self.memory_list)
                heapq.heapify(self.importance_heap)

                # Mark feature matrix as dirty
                self._feature_matrix_dirty = True

            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _add_unlocked(self, entry: Dict[str, Any]) -> str:
        """Internal: add entry assuming caller holds the lock."""
        entry_id = f"mem_{self.total_stored}_{int(time.time() * 1_000_000)}"

        features = self._as_float_vector(entry.get("features", []))
        action = self._as_float_vector(entry.get("action", [0.0, 0.0]))

        memory = MemoryEntry(
            id=entry_id,
            timestamp=float(entry.get("timestamp", time.time())),
            features=features,
            action=action,
            pnl=float(entry.get("pnl", 0.0)),
            context=dict(entry.get("context", {})),
            metadata=dict(entry.get("metadata", {})),
            importance=float(entry.get("importance", 0.5)),
        )

        # Primary storage
        self.memories[entry_id] = memory
        self.memory_list.append(memory)

        # Indices
        self.timestamp_index.append((memory.timestamp, entry_id))
        self.pnl_index.append((memory.pnl, entry_id))
        heapq.heappush(self.importance_heap, memory)

        # Feature matrix invalidated
        self._feature_matrix_dirty = True

        # Stats
        self.total_stored += 1

        return entry_id

    def _rebuild_feature_matrix(self) -> None:
        """Rebuild feature matrix for similarity search."""
        if not self.memory_list:
            self.feature_matrix = None
            self.feature_ids = []
            self._feature_matrix_dirty = False
            return

        # Collect all feature vectors and ids
        feats: List[np.ndarray] = []
        ids: List[str] = []
        for m in self.memory_list:
            if m.features is not None and m.features.size > 0:
                feats.append(self._as_float_vector(m.features))
                ids.append(m.id)

        if not feats:
            self.feature_matrix = None
            self.feature_ids = []
            self._feature_matrix_dirty = False
            return

        # Ensure same dimensionality by right-padding with zeros
        max_dim = int(max(v.size for v in feats))
        padded = [
            v if v.size == max_dim else np.pad(v, (0, max_dim - v.size), mode="constant")
            for v in feats
        ]
        self.feature_matrix = np.vstack(padded).astype(np.float32, copy=False)
        self.feature_ids = ids
        self._feature_matrix_dirty = False

    def _calculate_similarities(self, query: np.ndarray, features: np.ndarray) -> np.ndarray:
        """Calculate cosine similarities between a single query and all features."""
        # Normalize rows
        q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + self._EPS)
        f_norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + self._EPS)
        # Cosine similarity = q · f^T
        return np.dot(q_norm, f_norm.T)

    def _calculate_retention_score(self, memory: MemoryEntry) -> float:
        """
        Calculate retention score (higher => keep more).
        Combines importance, recency, access, and |pnl|.
        """
        importance_score = float(np.clip(memory.importance, 0.0, 1.0))

        age_hours = max(0.0, (time.time() - memory.timestamp) / 3600.0)
        recency_score = float(np.exp(-age_hours))  # decays with age

        access_score = float(np.log1p(memory.access_count) / 10.0)

        pnl_score = float(abs(memory.pnl) / 100.0)

        score = (
            importance_score * 0.4
            + recency_score * 0.3
            + access_score * 0.2
            + pnl_score * 0.1
        )
        return score

    def _remove_memory(self, entry_id: str) -> bool:
        """Remove memory by ID; caller must hold the lock."""
        memory = self.memories.pop(entry_id, None)
        if memory is None:
            return False

        # Remove from list (preserve order of others)
        self.memory_list = [m for m in self.memory_list if m.id != entry_id]

        # Remove from indices
        self.timestamp_index = [(t, i) for (t, i) in self.timestamp_index if i != entry_id]
        self.pnl_index = [(p, i) for (p, i) in self.pnl_index if i != entry_id]
        self.importance_heap = [m for m in self.importance_heap if m.id != entry_id]
        heapq.heapify(self.importance_heap)

        # Invalidate feature matrix
        self._feature_matrix_dirty = True
        return True

    def _auto_prune(self) -> None:
        """Automatically prune when over capacity."""
        if self.utilization() > self.pressure_threshold:
            self.cleanup(self.cleanup_ratio)

    def _memory_to_dict(self, memory: MemoryEntry) -> Dict[str, Any]:
        """Convert memory entry to a plain dictionary (JSON-safe)."""
        return {
            "id": memory.id,
            "timestamp": memory.timestamp,
            "features": memory.features.tolist() if memory.features is not None else [],
            "action": memory.action.tolist() if memory.action is not None else [],
            "pnl": memory.pnl,
            "context": memory.context,
            "metadata": memory.metadata,
            "importance": memory.importance,
            "access_count": memory.access_count,
            "last_accessed": memory.last_accessed,
        }

    # ----------------------------- helpers ------------------------------ #

    @staticmethod
    def _as_float_vector(x: Any) -> np.ndarray:
        """Coerce input to a contiguous 1-D float32 vector."""
        if isinstance(x, np.ndarray):
            arr = x.astype(np.float32, copy=False).reshape(-1)
        else:
            arr = np.asarray(x, dtype=np.float32).reshape(-1)
        return np.ascontiguousarray(arr)

    @staticmethod
    def _ensure_row_vector(vec: Any, target_dim: int) -> np.ndarray:
        """
        Convert input to shape (1, target_dim) float32, padding or truncating as needed.
        """
        arr = UnifiedMemoryStore._as_float_vector(vec)
        n = arr.size
        if n < target_dim:
            pad = np.zeros(target_dim - n, dtype=np.float32)
            arr = np.concatenate([arr, pad], axis=0)
        elif n > target_dim:
            arr = arr[:target_dim]
        return arr.reshape(1, target_dim)
