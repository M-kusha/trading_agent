# ─────────────────────────────────────────────────────────────
# File: modules/market/shared/data_extractors.py
# Unified data extraction for all market components — Production Upgrade
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Dict, Any, List, Optional, Union, Callable, Awaitable, Tuple, cast
import asyncio
import time
import hashlib
from collections import defaultdict, deque

import numpy as np
import pandas as pd


Number = Union[int, float, np.number]
ArrayLike = Union[List[Number], np.ndarray, pd.Series]
InstrumentBlock = Dict[str, Any]


class UnifiedDataExtractor:
    """
    Unified data extraction logic for all market components.
    Robust against partial/heterogeneous inputs. Adds:
      • Source registration (custom async sources)
      • Optional parallel extraction
      • Cache with TTL + stale-while-error
      • Strong normalization (dtype/shape) and validation
      • Derived fields (mid, spread if bid/ask present)
      • Per-source stats & latency
    Public methods/keys preserved for backwards compatibility.
    """

    SCHEMA_VERSION = "1.1.0"

    def __init__(
        self,
        logger: Optional[Any] = None,
        cache_enabled: bool = True,
        cache_ttl_seconds: float = 120.0,  # Match execution cycle (was 2.0s)
        enable_parallel: bool = True,
    ):
        self.logger = logger
        self.cache_enabled = cache_enabled
        self.cache_ttl_seconds = float(cache_ttl_seconds)
        self.enable_parallel = bool(enable_parallel)

        # Simple in-memory cache & telemetry
        self._cache: Dict[str, Any] = {}
        self._cache_fingerprint: Optional[str] = None
        self._cache_time: float = 0.0

        self._extraction_stats = defaultdict(int)
        self._latency_ms = defaultdict(lambda: deque(maxlen=100))
        self._errors = deque(maxlen=50)

        # Custom async sources registry: name -> async callable returning Dict[str, Any]
        self._custom_sources: Dict[str, Callable[[], Awaitable[Dict[str, Any]]]] = {}

    # ------------------------------- Logging -------------------------------

    def trace(self, message: str, level: str = "INFO", **kwargs):
        if self.logger:
            self.logger.trace(f"[DataExtractor] {message}", level=level, **kwargs)

    # --------------------------- Public Interface --------------------------

    def register_source(self, name: str, coro: Callable[[], Awaitable[Dict[str, Any]]]) -> None:
        """Register a custom async source provider."""
        self._custom_sources[name] = coro
        self.trace(f"Custom source registered: {name}", level="DEBUG")

    async def extract(
        self,
        sources: Optional[List[str]] = None,
        timeout: float = 2.0,
        **inputs,
    ) -> Dict[str, Any]:
        """
        Extract market data from multiple sources with fallback chain.

        Args:
            sources: e.g. ['smartinfobus', 'inputs', 'cache', 'synthetic', 'my_source']
            timeout: per-source soft timeout (seconds)
            **inputs: direct raw inputs (used by 'inputs' source)

        Returns:
            Unified market data dictionary (normalized + metadata)
        """
        self.trace("Starting unified data extraction", level="TRACE")
        started = time.time()

        if sources is None:
            sources = ['smartinfobus', 'inputs', 'cache', 'synthetic']

        # Build the plan of async tasks (custom + built-ins)
        tasks: List[Tuple[str, Awaitable[Dict[str, Any]]]] = []
        for src in sources:
            if src in self._custom_sources:
                tasks.append((src, self._wrap_with_timeout(src, self._custom_sources[src](), timeout)))
            elif src == 'smartinfobus':
                tasks.append((src, self._wrap_with_timeout(src, self._extract_from_infobus(), timeout)))
            elif src == 'inputs':
                # sync path wrapped to look async
                tasks.append((src, self._wrap_sync("inputs", lambda: self._extract_from_inputs(inputs))))
            elif src == 'cache':
                tasks.append((src, self._wrap_sync("cache", self._extract_from_cache)))
            elif src == 'synthetic':
                tasks.append((src, self._wrap_sync("synthetic", self._generate_synthetic_data)))
            else:
                self.trace(f"Unknown source '{src}' — skipping", level="WARNING")

        # Run tasks either in sequence (preserving priority) or in parallel with short-circuit
        results: Dict[str, Dict[str, Any]] = {}
        data_sources: List[str] = []
        errors: List[Dict[str, Any]] = []

        async def run_in_order():
            for name, aw in tasks:
                begin = time.time()
                try:
                    data = await aw
                    self._latency_ms[name].append((time.time() - begin) * 1000.0)
                    if isinstance(data, dict) and data:
                        results[name] = cast(Dict[str, Any], data)
                        data_sources.append(name)
                        # For priority semantics: continue gathering (we may merge)
                except Exception as e:
                    self._record_error(name, e)
                    errors.append({'source': name, 'error': str(e)})

        async def run_parallel():
            begin_by = {name: time.time() for name, _ in tasks}
            pending = [aw for _, aw in tasks]
            names = [name for name, _ in tasks]
            # Gather all but ignore failures (we capture below)
            done = await asyncio.gather(*pending, return_exceptions=True)
            for name, res in zip(names, done):
                dur = (time.time() - begin_by[name]) * 1000.0
                try:
                    self._latency_ms[name].append(dur)
                    if isinstance(res, Exception):
                        self._record_error(name, res)
                        errors.append({'source': name, 'error': str(res)})
                    elif isinstance(res, dict) and res:
                        results[name] = cast(Dict[str, Any], res)
                        data_sources.append(name)
                except Exception as e:
                    self._record_error(name, e)
                    errors.append({'source': name, 'error': str(e)})

        if self.enable_parallel:
            await run_parallel()
        else:
            await run_in_order()

        # Merge all results with sensible priority: earlier sources override later ones
        # Build priority map from the input ordering
        priority = {name: i for i, (name, _) in enumerate(tasks)}
        merged = self._merge_results(results, priority)

        # Normalize, validate, and (optionally) cache
        merged = self._normalize_market_data(merged)

        # If merged is empty, attempt stale cache (stale-while-error)
        if not merged and self.cache_enabled and self._is_cache_fresh(allow_stale=True):
            self.trace("Using stale cache due to empty result set", level="WARNING")
            merged = dict(self._cache)  # shallow copy
            data_sources.append('cache_stale')

        # Metadata
        merged['_sources'] = data_sources
        merged['_extraction_time'] = pd.Timestamp.now()
        merged['_schema_version'] = self.SCHEMA_VERSION
        merged['_errors'] = errors
        merged['_latency_ms'] = {k: (float(np.mean(v)) if v else None) for k, v in self._latency_ms.items()}

        # Update cache on success
        if self.cache_enabled and self._has_payload(merged):
            self._update_cache(merged)

        # Stats bump
        for s in data_sources:
            self._extraction_stats[s] += 1

        self.trace(
            f"Extraction complete — sources={data_sources}, "
            f"elapsed_ms={(time.time() - started)*1000.0:.1f}, "
            f"errors={len(errors)}",
            level="DEBUG"
        )
        return merged

    # ------------------------------- Sources -------------------------------

    async def _extract_from_infobus(self) -> Dict[str, Any]:
        """Extract data from SmartInfoBus (best-effort, resilient)."""
        name = "smartinfobus"
        self.trace("Extracting from SmartInfoBus", level="TRACE")
        try:
            from modules.utils.info_bus import InfoBusManager  # local import for resilience
            bus = InfoBusManager.get_instance()
            keys_to_try = [
                'market_data',
                'prices',
                'price_data',
                'bid_ask_data',
                'historical_prices',
                'multi_timeframe_data',
                'volatility_data',
                'liquidity_data',
                'volume_data',
                'timestamps',
            ]
            out: Dict[str, Any] = {}
            for k in keys_to_try:
                try:
                    v = bus.get(k, "UnifiedDataExtractor")
                except Exception as e:
                    self._record_error(name, e)
                    continue
                if v is not None:
                    out[k] = v
            return out if out else {}
        except Exception as e:
            self._record_error(name, e)
            return {}

    def _extract_from_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data directly from provided inputs."""
        self.trace("Extracting from inputs", level="TRACE")
        data: Dict[str, Any] = {}

        # Pass-through market_data if present
        if 'market_data' in inputs:
            data.update(inputs['market_data'] if isinstance(inputs['market_data'], dict) else {'market_data': inputs['market_data']})

        # Copy common keys if present
        for key in ['prices', 'volumes', 'spreads', 'depths', 'bid_ask_data', 'timestamps']:
            if key in inputs:
                data[key] = inputs[key]
        return data

    def _extract_from_cache(self) -> Dict[str, Any]:
        """Extract from cache if fresh enough."""
        if not self.cache_enabled or not self._cache:
            return {}
        if self._is_cache_fresh():
            self.trace("Using fresh cache", level="TRACE")
            return dict(self._cache)
        self.trace("Cache present but expired", level="TRACE")
        return {}

    def _generate_synthetic_data(self) -> Dict[str, Any]:
        """Generate realistic synthetic data for testing."""
        self.trace("Generating synthetic data", level="WARNING")
        rng = np.random.default_rng(42)  # reproducible
        instruments = ['EURUSD', 'XAUUSD']
        data: Dict[str, Any] = {}

        for instrument in instruments:
            base_price = 1.1000 if instrument == 'EURUSD' else 1950.0
            n = 300
            # Geometric random walk
            returns = rng.normal(0.0, 0.0008 if instrument == 'EURUSD' else 0.0006, n)
            prices = base_price * np.exp(np.cumsum(returns))
            vol = rng.exponential(1200.0 if instrument == 'EURUSD' else 900.0, n)

            bid = prices * (1.0 - 0.00005)
            ask = prices * (1.0 + 0.00005)

            data[instrument] = {
                'open': prices[:-1],
                'high': prices * 1.001,
                'low': prices * 0.999,
                'close': prices,
                'volume': vol,
                'bid': bid,
                'ask': ask,
            }

        data['prices'] = {inst: float(data[inst]['close'][-1]) for inst in instruments}
        data['spreads'] = {inst: float(data[inst]['ask'][-1] - data[inst]['bid'][-1]) for inst in instruments}
        data['timestamps'] = pd.date_range(end=pd.Timestamp.utcnow(), periods=300, freq="T")
        return data

    # ----------------------------- Normalization ---------------------------

    def _normalize_market_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize market data to consistent format & validate."""
        self.trace("Normalizing market data", level="TRACE")

        normalized = dict(data) if data else {}

        # Normalize timestamps
        if 'timestamps' in normalized:
            try:
                ts = pd.to_datetime(normalized['timestamps'])
                normalized['timestamps'] = ts
            except Exception:
                self.trace("Failed to parse timestamps; dropping", level="WARNING")
                normalized.pop('timestamps', None)

        # Normalize instrument blocks we know about
        instruments = self._discover_instruments(normalized)
        for inst in instruments:
            block = normalized.get(inst)
            if isinstance(block, pd.DataFrame):
                block = block.to_dict(orient='list')
            if isinstance(block, dict):
                # Coerce keys to strings to satisfy InstrumentBlock typing
                safe_block: InstrumentBlock = {str(k): v for k, v in block.items()}
                normalized[inst] = self._normalize_instrument_block(
                    safe_block, inst, normalized.get('timestamps')
                )

        # Backfill aggregates if possible
        if 'prices' not in normalized:
            normalized['prices'] = {
                inst: float(normalized[inst]['close'][-1]) for inst in instruments
                if inst in normalized and 'close' in normalized[inst] and len(normalized[inst]['close']) > 0
            }
        if 'spreads' not in normalized:
            spreads = {}
            for inst in instruments:
                blk = normalized.get(inst) or {}
                if 'ask' in blk and 'bid' in blk and len(blk['ask']) and len(blk['bid']):
                    spreads[inst] = float(blk['ask'][-1] - blk['bid'][-1])
            if spreads:
                normalized['spreads'] = spreads

        # Optional: validate monotonic timestamps/length alignment
        self._validate_lengths(normalized)

        return normalized

    def _normalize_instrument_block(
        self,
        block: InstrumentBlock,
        instrument: str,
        timestamps: Optional[pd.DatetimeIndex],
    ) -> InstrumentBlock:
        """
        Ensure arrays are float64 np.ndarrays of equal length when feasible.
        Compute 'mid' and 'spread' if bid/ask present.
        """
        keys_numeric = ['open', 'high', 'low', 'close', 'volume', 'bid', 'ask']
        out: InstrumentBlock = dict(block)

        # Convert common fields
        for k in keys_numeric:
            if k in out:
                out[k] = self._to_float_array(out[k])

        # If we have bid/ask, derive mid & spread arrays
        if 'bid' in out and 'ask' in out and len(out['bid']) and len(out['ask']):
            L = min(len(out['bid']), len(out['ask']))
            if L > 0:
                out['bid'] = out['bid'][:L]
                out['ask'] = out['ask'][:L]
                out['mid'] = (out['bid'] + out['ask']) / 2.0
                out['spread_series'] = out['ask'] - out['bid']

        # Align lengths of OHLCV to the shortest present (avoid index errors downstream)
        lengths = [len(out[k]) for k in keys_numeric if k in out and isinstance(out[k], np.ndarray)]
        if lengths:
            Lmin = int(min(lengths))
            for k in keys_numeric + ['mid', 'spread_series']:
                if k in out and isinstance(out[k], np.ndarray) and len(out[k]) != Lmin:
                    out[k] = out[k][:Lmin]
            # If timestamps provided, align as well
            if isinstance(timestamps, pd.DatetimeIndex) and len(timestamps) >= Lmin:
                out['timestamps'] = timestamps[-Lmin:]

        return out

    # ------------------------------- Validation ---------------------------

    def _validate_lengths(self, data: Dict[str, Any]) -> None:
        """Best-effort validation; logs warnings but never throws."""
        try:
            ts = data.get('timestamps')
            for inst, blk in data.items():
                if inst.startswith('_') or inst in ('prices', 'spreads', 'timestamps'):
                    continue
                if not isinstance(blk, dict):
                    continue
                # Check equal lengths among present numeric arrays
                arrs = [v for k, v in blk.items() if k in ('open', 'high', 'low', 'close', 'volume', 'bid', 'ask') and isinstance(v, np.ndarray)]
                if not arrs:
                    continue
                lens = {len(a) for a in arrs}
                if len(lens) > 1:
                    self.trace(f"[{inst}] Array length mismatch: {sorted(lens)} — truncated to shortest", level="WARNING")
                # Timestamps alignment
                if isinstance(ts, pd.DatetimeIndex) and len(ts) not in (0, len(arrs[0])):
                    self.trace(f"[{inst}] Timestamp length ({len(ts)}) != series length ({len(arrs[0])})", level="TRACE")
        except Exception as e:
            self._record_error("validation", e)

    # ----------------------------- Merging/Cache ---------------------------

    def _merge_results(self, results: Dict[str, Dict[str, Any]], priority: Dict[str, int]) -> Dict[str, Any]:
        """
        Merge dicts with priority (lower index = higher priority).
        Shallow merge for top-level keys; instrument dicts are deep-merged.
        """
        if not results:
            return {}
        # Order sources by priority (lower value first)
        ordered = sorted(results.items(), key=lambda kv: priority.get(kv[0], 1e9))
        merged: Dict[str, Any] = {}
        for name, payload in ordered:
            if not isinstance(payload, dict):
                continue
            for k, v in payload.items():
                if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                    merged[k] = self._deep_merge_dicts(merged[k], v)
                else:
                    # Overwrite by priority
                    merged[k] = v
        return merged

    @staticmethod
    def _deep_merge_dicts(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(a)
        for k, v in b.items():
            if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                out[k] = UnifiedDataExtractor._deep_merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    def _update_cache(self, data: Dict[str, Any]) -> None:
        """Update cache with TTL and content fingerprint to avoid churn."""
        try:
            fp = self._fingerprint(data)
            if fp != self._cache_fingerprint:
                self._cache = dict(data)
                self._cache_fingerprint = fp
            self._cache_time = time.time()
            self.trace("Cache updated", level="TRACE")
        except Exception as e:
            self._record_error("cache_update", e)

    def _is_cache_fresh(self, allow_stale: bool = False) -> bool:
        if not self._cache:
            return False
        age = time.time() - self._cache_time
        if age <= self.cache_ttl_seconds:
            return True
        # Stale-while-error window (5× TTL) if allowed
        return allow_stale and age <= (self.cache_ttl_seconds * 5.0)

    def _fingerprint(self, data: Dict[str, Any]) -> str:
        """Lightweight content fingerprint (schema aware)."""
        h = hashlib.sha256()
        # Use prices/spreads + last closes to avoid huge payload hashing
        prices = data.get('prices', {})
        spreads = data.get('spreads', {})
        h.update(repr(sorted(prices.items())).encode())
        h.update(repr(sorted(spreads.items())).encode())
        for inst, blk in sorted(((k, v) for k, v in data.items() if isinstance(v, dict) and k not in ('prices', 'spreads'))):
            close = blk.get('close')
            if isinstance(close, np.ndarray) and close.size:
                h.update(f"{inst}:{float(close[-1]):.10f}".encode())
        return h.hexdigest()

    # ------------------------------- Utilities -----------------------------

    @staticmethod
    def _to_float_array(x: Any) -> np.ndarray:
        """Convert to np.float64 1D array (safe)."""
        if x is None:
            return np.array([], dtype=np.float64)
        if isinstance(x, np.ndarray):
            try:
                return x.astype(np.float64, copy=False).ravel()
            except Exception:
                return np.array(x, dtype=np.float64).ravel()
        if isinstance(x, (pd.Series, pd.Index)):
            return x.to_numpy(dtype=np.float64).ravel()
        if isinstance(x, (list, tuple)):
            return np.asarray(x, dtype=np.float64).ravel()
        # scalars
        try:
            return np.asarray(x, dtype=np.float64).ravel()
        except Exception:
            return np.array([], dtype=np.float64)

    def _discover_instruments(self, data: Dict[str, Any]) -> List[str]:
        """Heuristic: instrument keys look like 'AAA/BBB' or are known dict blocks."""
        instruments: List[str] = []
        for k, v in (data or {}).items():
            if isinstance(v, dict) and ("/" in k or {"open","high","low","close"} & set(v.keys())):
                instruments.append(k)
        # Preserve ordering
        return instruments

    def _has_payload(self, data: Dict[str, Any]) -> bool:
        if not data:
            return False
        # Consider we have payload if there is at least one instrument block or prices
        if any("/" in k for k in data.keys()):
            return True
        return bool(data.get('prices'))

    def _record_error(self, source: str, e: Exception) -> None:
        msg = f"{type(e).__name__}: {e}"
        self._errors.append({'source': source, 'error': msg, 'ts': time.time()})
        self.trace(f"{source} error: {msg}", level="WARNING")

    async def _wrap_with_timeout(self, name: str, aw: Awaitable[Dict[str, Any]], timeout: float) -> Dict[str, Any]:
        try:
            return await asyncio.wait_for(aw, timeout=timeout)
        except asyncio.TimeoutError as e:
            raise RuntimeError(f"{name} timed out after {timeout}s") from e

    async def _wrap_sync(self, name: str, fn: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        # Run sync function in default loop executor (non-blocking)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn)

    # ----------------------------- Introspection ---------------------------

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction & latency statistics."""
        return {
            'counts': dict(self._extraction_stats),
            'latency_ms_avg': {k: (float(np.mean(v)) if v else None) for k, v in self._latency_ms.items()},
            'errors_recent': list(self._errors),
            'cache': {
                'enabled': self.cache_enabled,
                'ttl_seconds': self.cache_ttl_seconds,
                'age_seconds': (time.time() - self._cache_time) if self._cache_time else None,
                'fresh': self._is_cache_fresh(),
            }
        }

    def clear_cache(self) -> None:
        """Clear data cache."""
        self._cache.clear()
        self._cache_time = 0.0
        self._cache_fingerprint = None
        self.trace("Cache cleared", level="DEBUG")
