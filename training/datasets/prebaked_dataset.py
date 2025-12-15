# training/datasets/prebaked_dataset.py
"""
Prebaked Dataset Loader (TRAINING-ONLY)
=======================================

Fast, deterministic loader for prebaked observation signals.
Supports CSV and cached NPZ formats.

Key behaviors:
- Schema validation via obs_contract (optional)
- Datetime parsing for simulation time (prefers Europe/Berlin clock semantics)
- Auto-caching to NPZ for faster subsequent loads
- Windows Python 3.13 compatible
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

import numpy as np
import pandas as pd

from training.contracts.obs_contract import (
    OBS_COLUMNS,
    OBS_SCHEMA_VERSION,
    OBS_SIZE,
    schema_hash,
    validate_df,
    extract_obs_matrix,
    validate_obs_matrix,
)

DEFAULT_LOCAL_TZ = "Europe/Berlin"
DT_COL_CANDIDATES = ("datetime", "timestamp", "time", "date")
PRICE_COLS = ("open", "high", "low", "close")


@dataclass
class PrebakedDataset:
    """
    Prebaked observation dataset for fast training.

    Attributes:
        obs: Observation matrix (N, 64) float32
        dt: Datetime array (N,) int64 nanoseconds (naive wall-clock; local semantics)
        prices: Price data dict with open/high/low/close arrays
        meta: Metadata including schema info, instrument, date range
    """

    obs: np.ndarray  # (N, 64) float32
    dt: np.ndarray  # (N,) int64 nanoseconds
    prices: Dict[str, np.ndarray] = field(default_factory=dict)  # OHLC arrays
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize dtypes defensively (keeps downstream deterministic)
        if not isinstance(self.obs, np.ndarray):
            self.obs = np.asarray(self.obs)
        if self.obs.dtype != np.float32:
            self.obs = self.obs.astype(np.float32, copy=False)

        if not isinstance(self.dt, np.ndarray):
            self.dt = np.asarray(self.dt)
        if self.dt.dtype != np.int64:
            self.dt = self.dt.astype(np.int64, copy=False)

        if self.obs.ndim != 2 or self.obs.shape[1] != OBS_SIZE:
            raise ValueError(f"obs must have shape (N, {OBS_SIZE}), got {self.obs.shape}")

        if self.dt.ndim != 1:
            raise ValueError(f"dt must be 1D, got {self.dt.ndim}D")

        if len(self.dt) != len(self.obs):
            raise ValueError(f"dt length {len(self.dt)} != obs rows {len(self.obs)}")

    def __len__(self) -> int:
        return len(self.obs)

    def __getitem__(self, idx: Union[int, slice]) -> np.ndarray:
        """Get observation(s) by index."""
        return self.obs[idx]

    @property
    def n_rows(self) -> int:
        return len(self.obs)

    @property
    def schema_version(self) -> str:
        return str(self.meta.get("schema_version", "unknown"))

    @property
    def instrument(self) -> str:
        return str(self.meta.get("instrument", "UNKNOWN"))

    @property
    def timeframe(self) -> str:
        return str(self.meta.get("timeframe", "M15"))

    def get_datetime(self, idx: int) -> pd.Timestamp:
        """
        Convert int64 ns to pandas Timestamp (naive wall-clock).

        Note:
            Semantically treated as Europe/Berlin local time by the training stack.
        """
        return pd.Timestamp(self.dt[idx], unit="ns")

    def get_price(self, idx: int) -> Dict[str, float]:
        """Get OHLC prices at index."""
        out: Dict[str, float] = {}
        for k, v in self.prices.items():
            out[k] = float(v[idx]) if 0 <= idx < len(v) else 0.0
        return out

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        auto_cache: bool = True,
        validate: bool = True,
    ) -> "PrebakedDataset":
        """
        Load prebaked dataset from CSV file.

        Args:
            path: Path to CSV file (e.g., data/prebaked/EURUSD_M15_signals.csv)
            auto_cache: If True, save NPZ cache for faster loads
            validate: If True, validate schema and data integrity

        Returns:
            PrebakedDataset instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Prebaked file not found: {path}")

        # Check for cached NPZ
        npz_path = path.with_suffix(".npz")
        if npz_path.exists():
            try:
                ds = cls.from_npz(npz_path, validate=validate)
                if ds.meta.get("schema_hash") == schema_hash():
                    return ds
                warnings.warn(f"Cache schema mismatch, reloading from CSV: {path}")
            except Exception as e:
                warnings.warn(f"Failed to load cache, reloading CSV: {e}")

        # Load CSV
        df = pd.read_csv(path)

        # Validate schema presence
        if validate:
            ok, missing = validate_df(df)
            if not ok:
                raise ValueError(f"CSV missing {len(missing)} columns: {missing[:5]}...")

        # Extract observation matrix (validate=False will fill missing cols with zeros per contract)
        obs = extract_obs_matrix(df, validate=validate)

        # Validate matrix
        if validate:
            ok, err = validate_obs_matrix(obs)
            if not ok:
                raise ValueError(f"Invalid observation matrix: {err}")

        # Parse datetime (simulation time)
        dt = cls._parse_datetime_column(df)

        # Extract prices
        prices = cls._extract_prices(df)

        # Build metadata
        instrument, timeframe = cls._parse_filename(path)

        # Safe date range rendering
        start_ts = pd.Timestamp(dt[0], unit="ns") if len(dt) else pd.Timestamp("2020-01-01")
        end_ts = pd.Timestamp(dt[-1], unit="ns") if len(dt) else pd.Timestamp("2020-01-01")

        meta = {
            "source_path": str(path),
            "source_mtime": float(path.stat().st_mtime),
            "schema_version": OBS_SCHEMA_VERSION,
            "schema_hash": schema_hash(),
            "instrument": instrument,
            "timeframe": timeframe,
            "row_count": int(len(obs)),
            "datetime_semantics": f"naive_wallclock::{DEFAULT_LOCAL_TZ}",
            "date_range": {
                "start": str(start_ts),
                "end": str(end_ts),
            },
        }

        dataset = cls(obs=obs, dt=dt, prices=prices, meta=meta)

        # Auto-cache
        if auto_cache:
            try:
                dataset.to_npz(npz_path)
            except Exception as e:
                warnings.warn(f"Failed to cache NPZ: {e}")

        return dataset

    @classmethod
    def from_npz(
        cls,
        path: Union[str, Path],
        validate: bool = True,
    ) -> "PrebakedDataset":
        """
        Load prebaked dataset from NPZ cache.

        Args:
            path: Path to NPZ file
            validate: If True, validate data integrity

        Returns:
            PrebakedDataset instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"NPZ file not found: {path}")

        with np.load(path, allow_pickle=True) as data:
            obs = data["obs"]
            dt = data["dt"]

            # Load prices if present
            prices: Dict[str, np.ndarray] = {}
            for key in PRICE_COLS:
                if key in data:
                    prices[key] = data[key]

            # Load metadata
            meta: Dict[str, Any] = {}
            if "meta" in data:
                try:
                    meta_obj = data["meta"].item()
                    if isinstance(meta_obj, dict):
                        meta = meta_obj
                except Exception:
                    meta = {}

        # Normalize dtypes early (even if validate=False)
        obs = np.asarray(obs, dtype=np.float32)
        dt = np.asarray(dt, dtype=np.int64)

        if validate:
            ok, err = validate_obs_matrix(obs)
            if not ok:
                raise ValueError(f"Invalid NPZ data: {err}")

            # Check schema hash
            cached_hash = meta.get("schema_hash")
            if cached_hash != schema_hash():
                warnings.warn(
                    f"Schema hash mismatch in {path}. "
                    f"Expected {schema_hash()}, got {cached_hash}"
                )

        return cls(obs=obs, dt=dt, prices=prices, meta=meta)

    def to_npz(self, path: Union[str, Path]) -> None:
        """
        Save dataset to NPZ for fast loading.

        Args:
            path: Output path
        """
        path = Path(path)

        # Ensure stable dtypes
        obs = np.asarray(self.obs, dtype=np.float32)
        dt = np.asarray(self.dt, dtype=np.int64)

        data: Dict[str, np.ndarray] = {
            "obs": obs,
            "dt": dt,
            "meta": np.array(self.meta, dtype=object),
        }
        data.update(self.prices)

        np.savez_compressed(str(path), allow_pickle=True, **data)

    @staticmethod
    def _parse_datetime_column(df: pd.DataFrame) -> np.ndarray:
        """
        Parse datetime column to int64 nanoseconds.

        Semantics:
          - If source strings are tz-aware: convert to Europe/Berlin, then drop tz (keep local clock time).
          - If source strings are naive: keep as-is (treated as Europe/Berlin local by convention).
          - Result is stored as naive wall-clock nanoseconds (int64), consistent with env usage.
        """
        dt_col: Optional[str] = None
        for col in DT_COL_CANDIDATES:
            if col in df.columns:
                dt_col = col
                break

        if dt_col is None:
            # Fallback: generate synthetic timestamps (1 bar per 15 min)
            n = len(df)
            start = pd.Timestamp("2020-01-01 00:00:00")
            dt_idx = pd.date_range(start, periods=n, freq="15min")
            return dt_idx.to_numpy(dtype="datetime64[ns]").view("int64")

        raw = pd.to_datetime(df[dt_col], errors="coerce")

        # If tz-aware, convert to Europe/Berlin and drop tz to keep local clock time
        tz = None
        try:
            tz = raw.dt.tz
        except Exception:
            tz = None

        if tz is not None:
            try:
                raw = raw.dt.tz_convert(DEFAULT_LOCAL_TZ).dt.tz_localize(None)
            except Exception:
                # Worst case: drop tz without conversion (still deterministic)
                raw = raw.dt.tz_localize(None)

        # Fill NaT after tz handling (must be naive here)
        raw = raw.fillna(pd.Timestamp("2020-01-01 00:00:00"))

        # Convert to int64 ns
        return raw.to_numpy(dtype="datetime64[ns]").view("int64")

    @staticmethod
    def _extract_prices(df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Extract OHLC price arrays (float64)."""
        prices: Dict[str, np.ndarray] = {}
        for col in PRICE_COLS:
            if col in df.columns:
                prices[col] = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        return prices

    @staticmethod
    def _parse_filename(path: Path) -> tuple[str, str]:
        """Parse instrument and timeframe from filename."""
        # Expected: EURUSD_M15_signals.csv
        stem = path.stem  # EURUSD_M15_signals
        parts = stem.split("_")

        instrument = parts[0] if parts else "UNKNOWN"
        timeframe = parts[1] if len(parts) > 1 else "M15"

        return instrument, timeframe

    def describe(self) -> str:
        """Return human-readable description."""
        h = str(self.meta.get("schema_hash", "unknown"))
        lines = [
            f"PrebakedDataset: {self.instrument} {self.timeframe}",
            f"  Rows: {self.n_rows:,}",
            f"  Schema: {self.schema_version} ({h[:16]}...)",
            f"  Date range: {self.meta.get('date_range', {})}",
            f"  Datetime semantics: {self.meta.get('datetime_semantics', 'unknown')}",
            f"  Has prices: {list(self.prices.keys())}",
        ]
        return "\n".join(lines)


def load_all_prebaked(
    prebaked_dir: Union[str, Path] = "data/prebaked",
    instruments: Optional[List[str]] = None,
    timeframe: str = "M15",
) -> Dict[str, PrebakedDataset]:
    """
    Load all prebaked datasets from a directory.

    Args:
        prebaked_dir: Directory containing prebaked CSV files
        instruments: List of instruments to load (None = all)
        timeframe: Timeframe filter

    Returns:
        Dict mapping instrument name to PrebakedDataset
    """
    prebaked_dir = Path(prebaked_dir)
    datasets: Dict[str, PrebakedDataset] = {}

    if not prebaked_dir.exists():
        warnings.warn(f"Prebaked directory not found: {prebaked_dir}")
        return datasets

    pattern = f"*_{timeframe}_signals.csv"
    for csv_path in prebaked_dir.glob(pattern):
        inst = csv_path.stem.replace(f"_{timeframe}_signals", "")

        if instruments is not None and inst not in instruments:
            continue

        try:
            datasets[inst] = PrebakedDataset.from_csv(csv_path)
        except Exception as e:
            warnings.warn(f"Failed to load {csv_path}: {e}")

    return datasets


if __name__ == "__main__":
    # Self-test
    import sys

    test_dir = Path("data/prebaked")
    if test_dir.exists():
        print("Loading all prebaked datasets...")
        datasets = load_all_prebaked(test_dir)

        for inst, ds in datasets.items():
            print(f"\n{ds.describe()}")
    else:
        print(f"Prebaked directory not found: {test_dir}")
        print("Run scripts/prebake_signals.py first")
        sys.exit(1)
