# modules/memory/shared/feature_extractor.py
"""
Unified Feature Extractor
Standardized feature extraction for all memory components.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from collections.abc import Mapping as AbcMapping, Sequence as AbcSequence
from typing import cast

import numpy as np
from modules.utils.session_utils import normalize_session_name


Number = Union[int, float, np.number]
ArrayLike = Union[np.ndarray, Sequence[Number], Mapping[str, Any]]


class UnifiedFeatureExtractor:
    """
    Unified feature extraction for consistent representations.

    Features:
      - Market context encoding
      - Trade feature extraction
      - Observation processing
      - Feature normalization
      - Lightweight LRU caching for efficiency
    """

    # Numerical stability epsilon
    _EPS: float = 1e-12

    def __init__(self) -> None:
        """Initialize feature extractor."""
        # Fixed feature dimensions
        self.market_features_dim: int = 10
        self.trade_features_dim: int = 10
        self.observation_features_dim: int = 20
        self.total_dim: int = (
            self.market_features_dim + self.trade_features_dim + self.observation_features_dim
        )

        # Encoding maps (float32 for downstream models)
        self.regime_map: Dict[str, np.ndarray] = {
            "trending": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "volatile": np.array([0.0, 1.0, 0.0], dtype=np.float32),
            "ranging": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "unknown": np.array([0.33, 0.33, 0.33], dtype=np.float32),
        }

        # Canonical session one-hot (asian, european, american, closed)
        self.session_map: Dict[str, np.ndarray] = {
            "asian": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            "european": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            "american": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
            "closed": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "unknown": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
        }

        self.volatility_map: Dict[str, float] = {
            "low": 0.2,
            "medium": 0.5,
            "high": 0.8,
            "extreme": 1.0,
        }

        # Simple LRU cache (OrderedDict) — key -> np.ndarray
        self._cache: "OrderedDict[Tuple[Any, ...], np.ndarray]" = OrderedDict()
        self._cache_size: int = 1000

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def extract(
        self,
        observations: Optional[ArrayLike] = None,
        market_context: Optional[Mapping[str, Any]] = None,
        trade: Optional[Mapping[str, Any]] = None,
    ) -> np.ndarray:
        """
        Extract a unified feature vector.

        Args:
            observations: Raw observations (ndarray/list/tuple/dict/number).
            market_context: Market context dictionary.
            trade: Trade information dictionary.

        Returns:
            A float32 feature vector of fixed length (self.total_dim).
        """
        key = self._make_cache_key(observations, market_context, trade)
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        chunks: List[np.ndarray] = []

        if market_context is not None:
            chunks.append(self.extract_market_features(dict(market_context)))

        if trade is not None:
            chunks.append(self.extract_trade_features(dict(trade), dict(market_context or {})))

        if observations is not None:
            chunks.append(self.extract_observation_features(observations))

        if not chunks:
            vec = np.zeros(self.total_dim, dtype=np.float32)
            self._cache_put(key, vec)
            return vec

        combined = np.concatenate(chunks).astype(np.float32, copy=False)
        combined = self._ensure_dim(combined, self.total_dim)
        self._cache_put(key, combined)
        return combined

    def extract_market_features(self, market_context: Mapping[str, Any]) -> np.ndarray:
        """
        Extract features from market context.

        Returns:
            Market feature vector of length self.market_features_dim.
        """
        feats: List[float] = []

        # Regime (3)
        regime = str(market_context.get("regime", "unknown")).lower()
        feats.extend(self.regime_map.get(regime, self.regime_map["unknown"]))

        # Volatility (1)
        vol = market_context.get("volatility", 0.5)
        if isinstance(vol, str):
            vol_val = float(self.volatility_map.get(vol.lower(), 0.5))
        elif isinstance(vol, Mapping):
            # First numeric value if dict
            vol_val = 0.5
            for v in vol.values():
                if isinstance(v, (int, float, np.number)):
                    vol_val = float(v)
                    break
        else:
            vol_val = float(vol) if isinstance(vol, (int, float, np.number)) else 0.5
        feats.append(vol_val)

        # Session (4) – prefer canonical keys
        raw_session = market_context.get("current_session") or market_context.get("session_canonical") or market_context.get("session")
        session = normalize_session_name(str(raw_session) if raw_session is not None else "unknown")
        feats.extend(self.session_map.get(session, self.session_map["unknown"]))

        # Risk metrics (2)
        drawdown = float(market_context.get("drawdown_pct", 0.0)) / 100.0
        exposure = float(market_context.get("exposure_pct", 0.0)) / 100.0
        feats.extend([drawdown, exposure])

        arr = np.asarray(feats, dtype=np.float32).reshape(-1)
        return self._ensure_dim(arr, self.market_features_dim)

    def extract_trade_features(
        self,
        trade: Mapping[str, Any],
        market_context: Optional[Mapping[str, Any]] = None,  # kept for signature parity
    ) -> np.ndarray:
        """
        Extract features from a trade.

        Returns:
            Trade feature vector of length self.trade_features_dim.
        """
        feats: List[float] = []

        # Trade characteristics (4)
        feats.append(float(trade.get("size", 0.0)))
        feats.append(float(trade.get("confidence", 0.5)))
        feats.append(float(trade.get("volume", 1.0)))
        feats.append(float(trade.get("duration", 1.0)))

        # Side encoding (1)
        side = str(trade.get("side", "hold")).lower()
        if side == "buy":
            feats.append(1.0)
        elif side == "sell":
            feats.append(-1.0)
        else:
            feats.append(0.0)

        # PnL & risk (2)
        feats.append(float(trade.get("pnl", 0.0)) / 100.0)  # normalize
        feats.append(float(trade.get("risk", 0.0)))

        # Price movement (2)
        entry_price = float(trade.get("entry_price", 1.0))
        exit_price = float(trade.get("exit_price", entry_price))
        denom = entry_price if abs(entry_price) > self._EPS else 1.0
        price_change = (exit_price - entry_price) / denom
        feats.extend([entry_price, price_change])

        # Time of day (1) — normalized hour
        ts = trade.get("timestamp", 0)
        if ts:
            try:
                hour = float(datetime.fromtimestamp(float(ts)).hour)
                feats.append(hour / 24.0)
            except Exception:
                feats.append(0.5)
        else:
            feats.append(0.5)

        arr = np.asarray(feats, dtype=np.float32).reshape(-1)
        return self._ensure_dim(arr, self.trade_features_dim)

    def extract_observation_features(self, observations: ArrayLike) -> np.ndarray:
        """
        Extract features from observations of various formats.

        Returns:
            Observation feature vector of length self.observation_features_dim.
        """
        # numpy array first (fast path)
        if isinstance(observations, np.ndarray):
            arr = observations.astype(np.float32, copy=False).reshape(-1)

        # mapping/dict-like
        elif isinstance(observations, AbcMapping):
            feats: List[float] = []
            # Flatten numeric values deterministically by key order
            for key in sorted(observations.keys(), key=str):
                val = observations[key]
                if isinstance(val, (int, float, np.number)):
                    feats.append(float(val))
                elif isinstance(val, (list, tuple, np.ndarray)):
                    feat_arr = np.asarray(val, dtype=np.float32).reshape(-1)
                    feats.extend(feat_arr[:5].tolist())  # cap per-key contribution
            arr = np.asarray(feats, dtype=np.float32).reshape(-1)

        # any non-string sequence (e.g., tuples, lists, ranges)
        elif isinstance(observations, AbcSequence) and not isinstance(observations, (str, bytes, bytearray)):
            arr = np.asarray(list(observations), dtype=np.float32).reshape(-1)

        # scalar fallback
        else:
            try:
                scalar = float(cast(float, observations))
                arr = np.asarray([scalar], dtype=np.float32)
            except Exception:
                arr = np.zeros(1, dtype=np.float32)

        return self._ensure_dim(arr, self.observation_features_dim)


    def normalize(self, features: np.ndarray) -> np.ndarray:
        """
        Z-score normalize a feature vector with clipping to [-3, 3].

        Args:
            features: Raw feature vector.

        Returns:
            Normalized float32 vector.
        """
        features = np.asarray(features, dtype=np.float32).reshape(-1)
        mean = float(np.mean(features)) if features.size else 0.0
        std = float(np.std(features)) if features.size else 0.0
        if std > self._EPS:
            normalized = (features - mean) / std
            normalized = np.clip(normalized, -3.0, 3.0)
        else:
            normalized = features - mean
        return normalized.astype(np.float32, copy=False)

    def get_feature_names(self) -> List[str]:
        """Return human-readable feature names aligned to `total_dim`."""
        names: List[str] = []

        # Market features (10)
        names.extend(["regime_trending", "regime_volatile", "regime_ranging"])  # 3
        names.append("volatility")  # 1 -> 4
        names.extend(["session_asian", "session_european", "session_american", "session_closed"])  # +4 -> 8
        names.extend(["drawdown_pct", "exposure_pct"])  # +2 -> 10

        # Trade features (10)
        names.extend(["trade_size", "confidence", "volume", "duration", "side"])  # +5 -> 15
        names.extend(["pnl_normalized", "risk", "entry_price", "price_change", "hour_normalized"])  # +5 -> 20

        # Observation features (20)
        names.extend([f"obs_{i}" for i in range(self.observation_features_dim)])  # +20 -> 40

        return names[: self.total_dim]

    def get_feature_importance(self, features: np.ndarray, target: float) -> Dict[str, float]:
        """
        Compute a simple, sign-aware importance signal (heuristic).

        Args:
            features: Feature vector.
            target: Target scalar (e.g., realized PnL).

        Returns:
            Mapping of feature_name -> importance score.
        """
        vec = np.asarray(features, dtype=np.float32).reshape(-1)
        names = self.get_feature_names()
        sign = 0.0 if target == 0 else float(np.sign(target))
        out: Dict[str, float] = {}
        for i, name in enumerate(names):
            val = float(vec[i]) if i < vec.size else 0.0
            out[name] = val * sign
        return out

    # --------------------------------------------------------------------- #
    # Internals
    # --------------------------------------------------------------------- #

    def _ensure_dim(self, arr: np.ndarray, dim: int) -> np.ndarray:
        """Pad or truncate `arr` to exactly `dim` float32 entries."""
        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
        n = arr.size
        if n == dim:
            return arr
        if n < dim:
            pad = np.zeros(dim - n, dtype=np.float32)
            return np.concatenate([arr, pad], dtype=np.float32)
        return arr[:dim]

    def _cache_get(self, key: Tuple[Any, ...]) -> Optional[np.ndarray]:
        """LRU cache get; moves key to the end on hit."""
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: Tuple[Any, ...], value: np.ndarray) -> None:
        """LRU cache put; evicts the least-recently-used item if needed."""
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def _make_cache_key(
        self,
        observations: Optional[ArrayLike],
        market_context: Optional[Mapping[str, Any]],
        trade: Optional[Mapping[str, Any]],
    ) -> Tuple[Any, ...]:
        """
        Build a stable, hashable cache key for inputs without copying large buffers.
        Uses lightweight signatures for arrays (shape, dtype, mean, std, size).
        """
        return (
            self._sig_any(observations),
            self._sig_mapping(market_context),
            self._sig_mapping(trade),
        )

    def _sig_any(self, obj: Any) -> Any:
        """Signature for arbitrary input."""
        if obj is None:
            return None
        if isinstance(obj, np.ndarray):
            arr = obj
            return ("nd", arr.dtype.str, tuple(arr.shape), float(np.mean(arr)), float(np.std(arr)), arr.size)
        if isinstance(obj, (list, tuple)):
            arr = np.asarray(obj, dtype=np.float32).reshape(-1)
            return ("seq", float(np.mean(arr)) if arr.size else 0.0, float(np.std(arr)) if arr.size else 0.0, arr.size)
        if isinstance(obj, Mapping):
            return self._sig_mapping(obj)
        # Scalar
        try:
            return ("scalar", float(obj))
        except Exception:
            return ("other", str(obj))

    def _sig_mapping(self, mp: Optional[Mapping[str, Any]]) -> Any:
        """Deterministic signature for mappings (dicts)."""
        if mp is None:
            return None
        items: List[Tuple[str, Any]] = []
        for k in sorted(mp.keys(), key=str):
            v = mp[k]
            if isinstance(v, (np.ndarray, list, tuple)):
                arr = np.asarray(v, dtype=np.float32).reshape(-1)
                items.append(
                    (str(k), ("arr", float(np.mean(arr)) if arr.size else 0.0, float(np.std(arr)) if arr.size else 0.0, arr.size))
                )
            elif isinstance(v, Mapping):
                items.append((str(k), self._sig_mapping(v)))
            else:
                try:
                    items.append((str(k), float(v)))  # type: ignore[arg-type]
                except Exception:
                    items.append((str(k), str(v)))
        return tuple(items)
