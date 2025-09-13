# modules/memory/shared/utils.py
"""
Shared Utilities for Unified Memory System
Common functions and helper classes
"""

from __future__ import annotations

from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union
from collections import OrderedDict, deque, defaultdict
from datetime import datetime
import hashlib
import threading
import time

import numpy as np


class LRUCache:
    """
    Thread-safe LRU cache with optional per-key TTL.
    Always returns the stored VALUE (never the (value, expiry) tuple).
    """

    def __init__(self, maxsize: int = 1000) -> None:
        """
        Initialize LRU cache.

        Args:
            maxsize: Maximum cache size (number of keys).
        """
        self._cache: "OrderedDict[str, Tuple[Any, Optional[float]]]" = OrderedDict()
        self.maxsize = int(maxsize)
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._cache)

    def _is_expired(self, expiry: Optional[float]) -> bool:
        return expiry is not None and expiry <= time.time()

    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache. If TTL expired, evicts and returns None.
        """
        with self._lock:
            item = self._cache.get(key)
            if item is None:
                self.misses += 1
                return None

            value, expiry = item
            if self._is_expired(expiry):
                # expired -> evict and count as miss
                del self._cache[key]
                self.misses += 1
                return None

            # mark as recently used
            self._cache.move_to_end(key)
            self.hits += 1
            return value

    def put(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Insert or update a value.

        Args:
            key: Cache key.
            value: Value to store.
            ttl: Time-to-live in seconds. If None, no expiration.
                 If 0, the entry expires immediately.
        """
        with self._lock:
            expiry = None if ttl is None else time.time() + max(0, int(ttl))
            self._cache[key] = (value, expiry)
            self._cache.move_to_end(key)

            # Evict LRU entries beyond capacity
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)

    def pop(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """Remove and return the value for key if present, else default."""
        with self._lock:
            item = self._cache.pop(key, None)
            if item is None:
                return default
            value, _ = item
            return value

    def clear(self) -> None:
        """Clear all entries and reset stats."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "maxsize": self.maxsize,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
            }

    def cleanup_expired(self) -> int:
        """Evict all expired entries. Returns number removed."""
        with self._lock:
            now = time.time()
            expired_keys: List[str] = []
            for k, (_, expiry) in list(self._cache.items()):
                if expiry is not None and expiry <= now:
                    expired_keys.append(k)
            for k in expired_keys:
                self._cache.pop(k, None)
            return len(expired_keys)


class MemoryUtils:
    """
    Utility functions for memory operations.
    """

    @staticmethod
    def extract_action(trade: Dict[str, Any]) -> np.ndarray:
        """
        Extract action from trade.

        Returns:
            Action vector [position_delta, confidence]
        """
        action = trade.get("action")
        if action is not None:
            if isinstance(action, (list, tuple, np.ndarray)):
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                if arr.size >= 2:
                    return arr[:2]
                if arr.size == 1:
                    return np.array([float(arr[0]), 0.5], dtype=np.float32)
            else:
                # scalar-like
                try:
                    return np.array([float(action), 0.5], dtype=np.float32)
                except Exception:
                    pass

        # Fallback: derive from side/size
        side = str(trade.get("side", "hold")).lower()
        size = float(trade.get("size", 0.0))
        confidence = float(trade.get("confidence", 0.5))

        if side == "buy":
            position_delta = size
        elif side == "sell":
            position_delta = -size
        else:
            position_delta = 0.0

        return np.array([position_delta, confidence], dtype=np.float32)

    @staticmethod
    def calculate_pnl(
        entry_price: float,
        exit_price: float,
        size: float,
        side: str
    ) -> float:
        """Calculate PnL from trade parameters."""
        if side == "buy":
            pnl = (exit_price - entry_price) * size
        elif side == "sell":
            pnl = (entry_price - exit_price) * size
        else:
            pnl = 0.0
        return float(pnl)

    @staticmethod
    def encode_context(context: Dict[str, Any]) -> str:
        """
        Encode key context fields to a compact string for hashing/comparison.
        """
        parts: List[str] = []
        for key in ("regime", "volatility", "session", "trend"):
            if key in context:
                val = context[key]
                if isinstance(val, dict):
                    val = next(iter(val.values())) if val else "unknown"
                parts.append(f"{key}:{val}")
        return "_".join(parts)

    @staticmethod
    def hash_features(features: np.ndarray, precision: int = 2) -> str:
        """
        Create a short hash of a feature vector (rounded for stability).
        """
        rounded = np.round(np.asarray(features, dtype=np.float32), precision)
        return hashlib.md5(rounded.tobytes()).hexdigest()[:16]

    @staticmethod
    def calculate_similarity(
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two vectors.

        metric: 'cosine' | 'euclidean' | 'manhattan'
        """
        a = np.asarray(vec1, dtype=np.float32).reshape(-1)
        b = np.asarray(vec2, dtype=np.float32).reshape(-1)

        # Align lengths if needed
        if a.size != b.size:
            n = min(a.size, b.size)
            a = a[:n]
            b = b[:n]

        if metric == "cosine":
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na == 0.0 or nb == 0.0:
                return 0.0
            sim = float(np.dot(a, b) / (na * nb))
            return float(np.clip(sim, -1.0, 1.0))

        if metric == "euclidean":
            dist = float(np.linalg.norm(a - b))
            return 1.0 / (1.0 + dist)

        if metric == "manhattan":
            dist = float(np.sum(np.abs(a - b)))
            return 1.0 / (1.0 + dist)

        raise ValueError(f"Unknown metric: {metric}")

    @staticmethod
    def smooth_signal(
        signal: Union[float, np.ndarray],
        history: Deque[float],
        alpha: float = 0.1
    ) -> float:
        """
        Exponential smoothing of a signal, using the provided history deque.

        Note: Ensure 'history' has an appropriate maxlen upstream if you want bounded memory.
        """
        val = float(signal) if not isinstance(signal, np.ndarray) else float(np.asarray(signal).reshape(-1)[0])
        history.append(val)
        if len(history) == 1:
            return val

        smoothed = history[0]
        for v in list(history)[1:]:
            smoothed = alpha * v + (1.0 - alpha) * smoothed
        return float(smoothed)

    @staticmethod
    def calculate_entropy(distribution: np.ndarray) -> float:
        """
        Calculate Shannon entropy of a (non-negative) distribution vector.
        Returns 0.0 if sum is zero or distribution is empty.
        """
        arr = np.asarray(distribution, dtype=np.float64).reshape(-1)
        total = float(np.sum(arr))
        if total <= 0.0 or arr.size == 0:
            return 0.0
        p = arr / total
        p = p[p > 0]
        if p.size == 0:
            return 0.0
        return float(-np.sum(p * np.log2(p)))

    @staticmethod
    def detect_outliers(
        values: np.ndarray,
        method: str = "iqr",
        threshold: float = 1.5
    ) -> np.ndarray:
        """
        Detect outliers in a 1D array.

        method: 'iqr' | 'zscore' | 'percentile' (fallback)
        """
        x = np.asarray(values, dtype=np.float32).reshape(-1)
        n = x.size
        if n < 3:
            return np.zeros(n, dtype=bool)

        if method == "iqr":
            q1 = np.percentile(x, 25)
            q3 = np.percentile(x, 75)
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            return (x < lower) | (x > upper)

        if method == "zscore":
            mu = float(np.mean(x))
            sigma = float(np.std(x))
            if sigma == 0.0:
                return np.zeros(n, dtype=bool)
            z = np.abs((x - mu) / sigma)
            return z > threshold

        # percentile fallback
        lower = np.percentile(x, 5)
        upper = np.percentile(x, 95)
        return (x < lower) | (x > upper)

    @staticmethod
    def normalize_array(
        array: np.ndarray,
        method: str = "minmax"
    ) -> np.ndarray:
        """
        Normalize array values.

        method: 'minmax' | 'zscore' | 'robust'
        """
        x = np.asarray(array, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return x

        if method == "minmax":
            mn = float(np.min(x))
            mx = float(np.max(x))
            if mx > mn:
                return ((x - mn) / (mx - mn)).astype(np.float32)
            # constant array -> zeros
            return (x - mn).astype(np.float32)

        if method == "zscore":
            mu = float(np.mean(x))
            sigma = float(np.std(x))
            if sigma > 0.0:
                return ((x - mu) / sigma).astype(np.float32)
            return (x - mu).astype(np.float32)

        if method == "robust":
            med = float(np.median(x))
            mad = float(np.median(np.abs(x - med)))
            if mad > 0.0:
                return ((x - med) / mad).astype(np.float32)
            return (x - med).astype(np.float32)

        return x.astype(np.float32)

    @staticmethod
    def create_time_features(timestamp: float) -> Dict[str, float]:
        """
        Create normalized time-based features from a Unix timestamp.
        """
        dt = datetime.fromtimestamp(float(timestamp))
        return {
            "hour": dt.hour / 24.0,
            "day_of_week": dt.weekday() / 7.0,
            "day_of_month": dt.day / 31.0,
            "month": dt.month / 12.0,
            "quarter": ((dt.month - 1) // 3) / 4.0,
            "is_weekend": 1.0 if dt.weekday() >= 5 else 0.0,
            "is_month_end": 1.0 if dt.day >= 28 else 0.0,
        }

    @staticmethod
    def format_memory_size(bytes_size: int) -> str:
        """
        Human-friendly byte size formatter.
        """
        size = float(bytes_size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024.0 or unit == "TB":
                return f"{size:.2f} {unit}"
            size /= 1024.0
        # Unreachable, but keeps type checkers happy
        return f"{size:.2f} TB"

    @staticmethod
    def estimate_memory_usage(obj: Any) -> int:
        """
        Rough memory usage estimate, non-recursive for nested containers.
        """
        import sys

        size = sys.getsizeof(obj)
        if isinstance(obj, dict):
            size += sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            size += sum(sys.getsizeof(item) for item in obj)
        elif isinstance(obj, np.ndarray):
            size = obj.nbytes
        return int(size)


class RingBuffer:
    """
    Efficient ring buffer for streaming scalar data.
    """

    def __init__(self, capacity: int, dtype: type = float) -> None:
        self.capacity = int(capacity)
        self.buffer = np.zeros(self.capacity, dtype=dtype)
        self.position = 0
        self.size = 0

    def append(self, value: float) -> None:
        self.buffer[self.position] = value
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self) -> np.ndarray:
        if self.size < self.capacity:
            return self.buffer[:self.size]
        # reorder circular buffer
        return np.concatenate([self.buffer[self.position:], self.buffer[:self.position]])

    def mean(self) -> float:
        if self.size == 0:
            return 0.0
        return float(np.mean(self.buffer[:self.size]))

    def std(self) -> float:
        if self.size < 2:
            return 0.0
        return float(np.std(self.buffer[:self.size]))

    def clear(self) -> None:
        self.buffer.fill(0)
        self.position = 0
        self.size = 0


class MemoryStatistics:
    """
    Lightweight statistics tracker for counters, timers, and arbitrary values.
    """

    def __init__(self) -> None:
        self.counters: Dict[str, int] = defaultdict(int)
        self.timers: Dict[str, List[float]] = defaultdict(list)
        self.values: Dict[str, List[float]] = defaultdict(list)
        self.start_times: Dict[str, float] = {}

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] += int(amount)

    def record_time(self, name: str, duration: float) -> None:
        times = self.timers[name]
        times.append(float(duration))
        if len(times) > 1000:
            self.timers[name] = times[-1000:]

    def record_value(self, name: str, value: float) -> None:
        vals = self.values[name]
        vals.append(float(value))
        if len(vals) > 1000:
            self.values[name] = vals[-1000:]

    def start_timer(self, name: str) -> None:
        self.start_times[name] = time.time()

    def stop_timer(self, name: str) -> float:
        start = self.start_times.pop(name, None)
        if start is None:
            return 0.0
        duration = time.time() - start
        self.record_time(name, duration)
        return float(duration)

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"counters": dict(self.counters), "timers": {}, "values": {}}

        # Timers
        for name, times in self.timers.items():
            if not times:
                continue
            arr = np.asarray(times, dtype=np.float64)
            summary["timers"][name] = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "total": float(np.sum(arr)),
            }

        # Values
        for name, vals in self.values.items():
            if not vals:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            summary["values"][name] = {
                "count": int(arr.size),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "last": float(arr[-1]),
            }

        return summary

    def reset(self) -> None:
        self.counters.clear()
        self.timers.clear()
        self.values.clear()
        self.start_times.clear()
