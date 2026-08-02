#!/usr/bin/env python3


from __future__ import annotations

import json
import logging
import os
import platform
import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PRIMARY_TIMEFRAME: str = "M15"
CONTEXT_TIMEFRAMES: tuple = ("H1", "H4", "D1")
SUPPORTED_TIMEFRAMES: tuple = ("M15", "H1", "H4", "D1")


try:
    from config import get_trade_limits
    _TRADE_LIMITS = get_trade_limits()
except Exception:


    _TRADE_LIMITS = {"max_trades_per_day": 20}


PPO_OBS_VERSION = "7.0"


PPO_OBS_SIZE = 40

DEFAULT_INSTRUMENT = "XAUUSD"

FEATURE_GROUPS: Dict[str, tuple[int, int]] = {
    "m15_price": (0, 10),
    "htf_context": (10, 22),
    "risk": (22, 25),
    "account": (25, 30),
    "trading_mode": (30, 35),
    "governor": (35, 40),
}


class ObservationContractError(RuntimeError):
    pass


def _build_feature_names() -> List[str]:
    names: List[str] = []

    names += [
        "m15_price_vs_mean",
        "m15_range_pct",
        "m15_close_open_pct",
        "m15_sr_proximity",
        "m15_rsi_norm",
        "m15_macd_hist_norm",
        "m15_atr_norm_x10",
        "m15_trend_slope_norm",
        "m15_roc_x10",
        "m15_volatility_norm",
    ]

    for tf in ("H1", "H4", "D1"):
        names += [
            f"htf_{tf}_trend",
            f"htf_{tf}_momentum",
            f"htf_{tf}_rsi_norm",
            f"htf_{tf}_atr_norm",
        ]

    names += [
        "risk_drawdown_norm",
        "risk_budget",
        "risk_portfolio_exposure",
    ]

    names += [
        "account_step_ratio",
        "account_position_direction",
        "account_unrealized_pnl_norm",
        "account_time_in_position_norm",
        "account_on_cooldown",
    ]

    names += [
        "mode_entry_allowed",
        "mode_entry_quality_avg",
        "mode_setup_quality_avg",
        "mode_entry_certainty_avg",
        "mode_confluence_norm",
    ]

    names += [
        "gov_loss_layer_ratio",
        "gov_session_pnl_headroom",
        "gov_session_trade_budget",
        "gov_session_consec_loss_ratio",
        "gov_session_progress",
    ]

    if len(names) != PPO_OBS_SIZE:
        raise ValueError(f"Feature name list mismatch: {len(names)} != {PPO_OBS_SIZE}")
    return names


PPO_OBS_FEATURE_NAMES: List[str] = _build_feature_names()


@dataclass
class PPOObservationConfig:

    obs_size: int = PPO_OBS_SIZE
    version: str = PPO_OBS_VERSION


    instrument: str = DEFAULT_INSTRUMENT


    price_lookback: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    trend_lookback: int = 20
    momentum_lookback: int = 10


    min_bars_m15: int = 60
    min_bars_htf: int = 30


    max_drawdown_clip: float = 0.5




    sr_threshold_pct_m15: float = 0.005
    sr_cluster_tol_pct: float = 0.0015


    forming_bar_rtol: float = 1e-7
    forming_bar_atol: float = 1e-6


    use_forming_bar: bool = False


    # Off by default. This writes one JSON record per observation with no
    # rotation, so a single training run produced a 43 GB
    # logs/ppo_obs_debug_xauusd.jsonl and filled a 98 GB disk, which then took
    # the run down with OSError 28. It is a diagnostic for one-off inspection,
    # not something a multi-million-step run should carry.
    debug: bool = False
    debug_output_path: str = "logs/ppo_obs_debug_xauusd.jsonl"
    # Hard ceiling even when deliberately enabled, so the default is not the
    # only thing standing between a debug session and a full disk.
    debug_max_file_mb: float = 256.0


    debug_full_dump: bool = True
    debug_hard_cap_elems: int = 0


    debug_record_bus_fetches: bool = True


    debug_max_depth: int = 64


class _DebugTrace:

    def __init__(self, enabled: bool, path: str, cfg: PPOObservationConfig) -> None:
        self.enabled = bool(enabled)
        self.path = str(path)
        self.cfg = cfg
        self.session_id = uuid.uuid4().hex
        self._max_bytes = int(max(0.0, float(getattr(cfg, "debug_max_file_mb", 256.0))) * 1024 * 1024)
        self._cap_reported = False

        if not self.enabled:
            self._bytes_written = 0
            return

        # Count what is already on disk: appending across runs is what let the
        # file reach 43 GB.
        try:
            self._bytes_written = os.path.getsize(self.path)
        except OSError:
            self._bytes_written = 0

        d = os.path.dirname(self.path) or "."
        os.makedirs(d, exist_ok=True)

        header = {
            "type": "header",
            "ts": time.time(),
            "session_id": self.session_id,
            "version": cfg.version,
            "instrument": cfg.instrument,
            "obs_size": cfg.obs_size,
            "feature_groups": FEATURE_GROUPS,
            "feature_names": PPO_OBS_FEATURE_NAMES,
            "debug": {
                "full_dump": bool(cfg.debug_full_dump),
                "hard_cap_elems": int(cfg.debug_hard_cap_elems),
                "record_bus_fetches": bool(cfg.debug_record_bus_fetches),
            },
            "runtime": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "pid": os.getpid(),
            },
        }
        self._append(header)

    def _append(self, obj: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        # Stop at the ceiling rather than raise: a diagnostic must never be the
        # reason training dies. Filling the disk previously killed a run with
        # OSError 28 several hours in.
        if self._bytes_written >= self._max_bytes:
            if not self._cap_reported:
                self._cap_reported = True
                logger.warning(
                    "Observation debug trace hit its %.0f MB cap at '%s'; "
                    "further records dropped. Training continues.",
                    self._max_bytes / (1024 * 1024), self.path,
                )
            return

        try:
            line = json.dumps(obj, ensure_ascii=False) + "\n"
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line)
            self._bytes_written += len(line.encode("utf-8"))
        except Exception as e:
            raise ObservationContractError(
                f"[DEBUG] Failed to write debug trace file '{self.path}': {e}"
            ) from e

    def _safe(self, v: Any, depth: int = 0) -> Any:

        if not self.cfg.debug_full_dump and depth > self.cfg.debug_max_depth:
            return "<max_depth>"

        if v is None or isinstance(v, (bool, int, float, str)):
            return v


        if isinstance(v, (np.floating, np.integer)):
            return float(v)


        if isinstance(v, np.ndarray):
            return self._safe_array(v)


        if isinstance(v, (list, tuple)):
            return [self._safe(x, depth + 1) for x in v]


        if isinstance(v, dict):
            out: Dict[str, Any] = {}
            for k, vv in v.items():
                out[str(k)] = self._safe(vv, depth + 1)
            return out


        try:
            return str(v)
        except Exception:
            return "<unrepr>"

    def _safe_array(self, arr: np.ndarray) -> Any:
        arr = np.asarray(arr)
        info: Dict[str, Any] = {
            "type": "ndarray",
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "len": int(arr.size),
        }

        flat = arr.reshape(-1) if arr.ndim > 0 else np.asarray([arr])
        n = int(flat.size)
        if n == 0:
            return info


        try:
            finite = np.isfinite(flat)
            info["finite_ratio"] = float(np.mean(finite))
            if np.any(finite):
                f = flat[finite].astype(np.float64, copy=False)
                info["min"] = float(np.min(f))
                info["max"] = float(np.max(f))
                info["mean"] = float(np.mean(f))
                info["std"] = float(np.std(f))
        except Exception:
            pass

        cap = int(self.cfg.debug_hard_cap_elems) if int(self.cfg.debug_hard_cap_elems) > 0 else 0
        if cap > 0 and n > cap:

            head_n = min(200, n)
            tail_n = min(200, n)
            info["truncated"] = True
            info["cap"] = cap
            info["head"] = [float(x) for x in flat[:head_n].astype(np.float64, copy=False)]
            info["tail"] = [float(x) for x in flat[-tail_n:].astype(np.float64, copy=False)]
            return info


        try:

            info["values"] = arr.astype(np.float64, copy=False).tolist()
        except Exception:
            info["values_flat"] = [float(x) for x in flat.astype(np.float64, copy=False)]
        return info

    def record(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        obj: Dict[str, Any] = {"type": "event", "event": event, "ts": time.time(), "session_id": self.session_id}
        safe_payload = self._safe(payload)
        if isinstance(safe_payload, dict):
            obj.update(safe_payload)
        else:
            obj["payload"] = safe_payload
        self._append(obj)


class PPOObservationBuilder:

    def __init__(self, config: Optional[PPOObservationConfig] = None) -> None:
        self.config = config or PPOObservationConfig()
        self.config.obs_size = PPO_OBS_SIZE
        self._eps: float = 1e-12

        self._instrument = self._norm_symbol(self.config.instrument)
        if self._instrument != self._norm_symbol(DEFAULT_INSTRUMENT):
            raise ObservationContractError(
                f"Only {DEFAULT_INSTRUMENT} is supported in strict mode. "
                f"Got config.instrument='{self.config.instrument}'."
            )

        self._dbg = _DebugTrace(self.config.debug, self.config.debug_output_path, self.config)
        self._build_id: int = 0

    @property
    def obs_size(self) -> int:
        return self.config.obs_size

    @property
    def version(self) -> str:
        return self.config.version

    @property
    def instrument(self) -> str:
        return DEFAULT_INSTRUMENT

    @property
    def feature_groups(self) -> Dict[str, tuple[int, int]]:
        return dict(FEATURE_GROUPS)

    @property
    def feature_names(self) -> List[str]:
        return list(PPO_OBS_FEATURE_NAMES)

    def get_schema(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "instrument": self.instrument,
            "obs_size": int(self.obs_size),
            "feature_groups": {k: {"start": int(v[0]), "end": int(v[1])} for k, v in FEATURE_GROUPS.items()},
            "feature_names": self.feature_names,
        }


    def build(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        expert_signals: Optional[Dict[str, Any]] = None,
        risk_state: Optional[Dict[str, Any]] = None,
        account_state: Optional[Dict[str, Any]] = None,
        trading_mode_state: Optional[Dict[str, Any]] = None,
        governor_state: Optional[Dict[str, Any]] = None,
        smart_bus: Optional[Any] = None,
        module_name: str = "PPOObservationBuilder",
    ) -> np.ndarray:

        self._build_id += 1
        build_id = int(self._build_id)

        if self.config.debug:
            self._dbg.record(
                "build_start",
                {
                    "build_id": build_id,
                    "module_name": module_name,
                    "instrument": self.instrument,
                    "smart_bus_enabled": bool(smart_bus is not None),
                },
            )

        if smart_bus is not None:
            market_data = self._fetch_market_data_xauusd(smart_bus, module_name, build_id=build_id)
            expert_signals = self._fetch_expert_signals_xauusd(smart_bus, module_name, build_id=build_id)
            risk_state = self._fetch_risk_state_strict(smart_bus, module_name, build_id=build_id)
            account_state = self._fetch_account_state_xauusd(smart_bus, module_name, build_id=build_id)
            trading_mode_state = self._fetch_trading_mode_state_xauusd(smart_bus, module_name, build_id=build_id)
            governor_state = self._fetch_governor_state_strict(smart_bus, module_name, build_id=build_id)

        if self.config.debug:
            self._dbg.record(
                "inputs_before_validate",
                {
                    "build_id": build_id,
                    "module_name": module_name,
                    "inputs": {
                        "market_data": market_data,
                        "expert_signals": expert_signals,
                        "risk_state": risk_state,
                        "account_state": account_state,
                        "trading_mode_state": trading_mode_state,
                        "governor_state": governor_state,
                    },
                },
            )


        self._validate_inputs_strict(
            market_data=market_data,
            expert_signals=expert_signals,
            risk_state=risk_state,
            account_state=account_state,
            trading_mode_state=trading_mode_state,
            governor_state=governor_state,
        )


        assert isinstance(market_data, dict)
        assert isinstance(expert_signals, dict)
        assert isinstance(risk_state, dict)
        assert isinstance(account_state, dict)
        assert isinstance(trading_mode_state, dict)
        assert isinstance(governor_state, dict)

        obs = np.zeros(self.config.obs_size, dtype=np.float32)


        m15_feats, m15_dbg = self._build_m15_features(market_data)
        htf_feats, htf_dbg = self._build_htf_context(market_data, expert_signals)
        risk_feats, risk_dbg = self._build_risk_features(risk_state, account_state)
        account_feats, account_dbg = self._build_account_features(account_state)
        mode_feats, mode_dbg = self._build_trading_mode_features(trading_mode_state)
        gov_feats, gov_dbg = self._build_governor_features(governor_state)

        obs[0:10] = m15_feats
        obs[10:22] = htf_feats
        obs[22:25] = risk_feats
        obs[25:30] = account_feats
        obs[30:35] = mode_feats
        obs[35:40] = gov_feats


        self._validate_observation_strict(obs)

        if self.config.debug:
            name_map = {self.feature_names[i]: float(obs[i]) for i in range(self.obs_size)}
            self._dbg.record(
                "build_trace",
                {
                    "build_id": build_id,
                    "instrument": self.instrument,
                    "module_name": module_name,
                    "config": self.get_schema(),
                    "inputs": {
                        "market_data": market_data,
                        "expert_signals": expert_signals,
                        "risk_state": risk_state,
                        "account_state": account_state,
                        "trading_mode_state": trading_mode_state,
                        "governor_state": governor_state,
                    },
                    "internals": {
                        "m15": m15_dbg,
                        "htf": htf_dbg,
                        "risk": risk_dbg,
                        "account": account_dbg,
                        "trading_mode": mode_dbg,
                        "governor": gov_dbg,
                    },
                    "output": {
                        "obs": obs,
                        "obs_name_map": name_map,
                    },
                },
            )

            self._dbg.record(
                "build_end",
                {"build_id": build_id, "module_name": module_name, "instrument": self.instrument},
            )

        return obs

    def build_for_instrument(self, instrument: str, **kwargs: Any) -> np.ndarray:
        if self._norm_symbol(instrument) != self._norm_symbol(DEFAULT_INSTRUMENT):
            raise ObservationContractError(
                f"Only {DEFAULT_INSTRUMENT} is supported. Got instrument='{instrument}'."
            )
        return self.build(**kwargs)


    def _validate_inputs_strict(
        self,
        *,
        market_data: Optional[Dict[str, Any]],
        expert_signals: Optional[Dict[str, Any]],
        risk_state: Optional[Dict[str, Any]],
        account_state: Optional[Dict[str, Any]],
        trading_mode_state: Optional[Dict[str, Any]],
        governor_state: Optional[Dict[str, Any]],
    ) -> None:
        if not isinstance(market_data, dict) or not market_data:
            raise ObservationContractError("market_data must be a non-empty dict (timeframe->ohlcv dict).")

        for tf in SUPPORTED_TIMEFRAMES:
            if tf not in market_data or not isinstance(market_data.get(tf), dict):
                raise ObservationContractError(
                    f"market_data missing timeframe '{tf}' or it is not a dict. "
                    f"Expected market_data['{tf}']={{'open','high','low','close',...}}."
                )

        m15 = market_data[PRIMARY_TIMEFRAME]
        self._validate_ohlc_block(m15, PRIMARY_TIMEFRAME, min_bars=self.config.min_bars_m15)

        for tf in CONTEXT_TIMEFRAMES:
            self._validate_ohlc_block(market_data[tf], tf, min_bars=self.config.min_bars_htf)

        if not isinstance(expert_signals, dict) or "experts" not in expert_signals:
            raise ObservationContractError("expert_signals must be a dict containing key 'experts'.")
        self._validate_expert_signals(expert_signals)


        if not isinstance(risk_state, dict):
            raise ObservationContractError("risk_state must be a dict.")
        for k in ("portfolio_risk", "risk_budget"):
            if k not in risk_state:
                raise ObservationContractError(f"risk_state missing required key '{k}'.")


        if not isinstance(account_state, dict):
            raise ObservationContractError("account_state must be a dict.")
        for k in (
            "balance",
            "initial_balance",
            "current_drawdown",
            "current_step",
            "max_steps",
            "position_direction",
            "unrealized_pnl",
            "time_in_position",
            "on_cooldown",
        ):
            if k not in account_state:
                raise ObservationContractError(f"account_state missing required key '{k}'.")


        if not isinstance(trading_mode_state, dict):
            raise ObservationContractError("trading_mode_state must be a dict.")
        self._validate_trading_mode_state(trading_mode_state)

        if not isinstance(governor_state, dict):
            raise ObservationContractError("governor_state must be a dict.")
        self._validate_governor_state(governor_state)

    def _validate_observation_strict(self, obs: np.ndarray) -> None:
        if not isinstance(obs, np.ndarray) or obs.shape != (self.obs_size,):
            raise ObservationContractError(
                f"observation must be shape ({self.obs_size},). Got {getattr(obs, 'shape', None)}"
            )
        if not np.all(np.isfinite(obs)):
            bad = np.where(~np.isfinite(obs))[0].tolist()
            raise ObservationContractError(f"observation contains NaN/Inf at indices: {bad}")

        if np.max(np.abs(obs.astype(np.float64))) > 50.0:
            mx = float(np.max(np.abs(obs.astype(np.float64))))
            raise ObservationContractError(
                f"observation magnitude exploded (max abs={mx}). Contract likely broken."
            )


    def _build_m15_features(self, market_data: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        feats = np.zeros(10, dtype=np.float32)
        dbg: Dict[str, Any] = {"tf": PRIMARY_TIMEFRAME}

        m15 = self._extract_timeframe_data(market_data, PRIMARY_TIMEFRAME)
        self._require_dict(m15, "market_data[M15]")

        close = self._as_1d_float_array(m15.get("close"), "M15.close")
        high = self._as_1d_float_array(m15.get("high"), "M15.high")
        low = self._as_1d_float_array(m15.get("low"), "M15.low")
        open_ = self._as_1d_float_array(m15.get("open"), "M15.open")

        self._require_min_len(close, self.config.min_bars_m15, "M15.close")
        self._require_same_len([close, high, low, open_], ["close", "high", "low", "open"], "M15")

        c = float(close[-1])
        h = float(high[-1])
        l = float(low[-1])
        o = float(open_[-1])

        lb = min(self.config.price_lookback, int(close.size))
        mean_close = float(np.mean(close[-lb:]))

        feats[0] = float((c / max(mean_close, self._eps)) - 1.0)
        feats[1] = float((h - l) / max(abs(c), self._eps))
        feats[2] = float((c - o) / max(abs(o), self._eps))

        support, resistance = self._find_sr_levels(high[-self.config.price_lookback:], low[-self.config.price_lookback:])
        near_support, near_resistance = self._compute_sr_proximity(
            current_price=c,
            support=support,
            resistance=resistance,
            threshold_pct=self.config.sr_threshold_pct_m15,
        )
        feats[3] = float(np.clip(near_support - near_resistance, -1.0, 1.0))

        rsi = self._compute_rsi(close, self.config.rsi_period)
        feats[4] = float(np.clip((rsi - 50.0) / 50.0, -1.0, 1.0))

        macd_hist = self._compute_macd_histogram(close)
        denom = max(abs(c) * 0.01, self._eps)
        feats[5] = float(np.clip(macd_hist / denom, -1.0, 1.0))

        atr = self._compute_atr(high, low, close, self.config.atr_period)
        feats[6] = float(np.clip(atr / max(c, self._eps), 0.0, 0.1) * 10.0)

        trend_slope_norm = self._compute_trend_slope_norm(close, high, low, lookback=self.config.trend_lookback)
        feats[7] = float(np.clip(trend_slope_norm, -1.0, 1.0))

        base = float(close[-self.config.momentum_lookback])
        roc = (c - base) / max(abs(base), self._eps)
        feats[8] = float(np.clip(roc * 10.0, -1.0, 1.0))

        prev = close[-20:-1]
        curr = close[-19:]
        rets = (curr - prev) / np.maximum(np.abs(prev), self._eps)
        vol = float(np.std(rets))
        ref = float(np.median(np.abs(rets - np.median(rets)))) * 1.4826
        feats[9] = float(np.clip((vol / ref) / 3.0, 0.0, 1.0)) if ref > 1e-12 else 0.33

        dbg.update(
            {
                "last": {"c": c, "h": h, "l": l, "o": o},
                "mean_close": mean_close,
                "sr": {
                    "support": support,
                    "resistance": resistance,
                    "near_support": near_support,
                    "near_resistance": near_resistance,
                },
                "rsi": rsi,
                "macd_hist": macd_hist,
                "atr": atr,
                "trend_slope_norm": trend_slope_norm,
                "roc": roc,
                "vol_std": vol,
                "feats": feats,

                "raw": {
                    "close": close,
                    "high": high,
                    "low": low,
                    "open": open_,
                    "m15_block": m15,
                },
            }
        )
        return feats, dbg

    def _build_htf_context(
        self, market_data: Dict[str, Any], expert_signals: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        feats = np.zeros(12, dtype=np.float32)
        dbg: Dict[str, Any] = {"tfs": list(CONTEXT_TIMEFRAMES)}

        htf_experts = expert_signals.get("htf_experts", {})
        if not isinstance(htf_experts, dict):
            raise ObservationContractError("expert_signals.htf_experts must be a dict in strict mode.")

        per_tf_dbg: Dict[str, Any] = {}

        for tf_idx, tf in enumerate(CONTEXT_TIMEFRAMES):
            tf_data = self._extract_timeframe_data(market_data, tf)
            self._require_dict(tf_data, f"market_data[{tf}]")

            close = self._as_1d_float_array(tf_data.get("close"), f"{tf}.close")
            high = self._as_1d_float_array(tf_data.get("high"), f"{tf}.high")
            low = self._as_1d_float_array(tf_data.get("low"), f"{tf}.low")

            self._require_min_len(close, self.config.min_bars_htf, f"{tf}.close")
            self._require_same_len([close, high, low], ["close", "high", "low"], tf)

            base_idx = tf_idx * 4
            c = float(close[-1])

            htf_sig = htf_experts.get(tf)
            if not isinstance(htf_sig, dict):
                raise ObservationContractError(f"expert_signals.htf_experts missing dict for tf='{tf}'.")

            trend = self._compute_trend_slope_norm(close, high, low, lookback=min(30, int(close.size)))
            trend = self._blend_with_htf_expert_trend(trend, htf_sig)
            feats[base_idx + 0] = float(np.clip(trend, -1.0, 1.0))

            mom_norm = self._compute_momentum_norm(close, high, low, bars=5)
            mom_norm = self._blend_with_htf_expert_momentum(mom_norm, htf_sig)
            feats[base_idx + 1] = float(np.clip(mom_norm, -1.0, 1.0))

            rsi = float(self._to_float_required(htf_sig.get("rsi"), f"htf_experts.{tf}.rsi"))
            rsi_norm = (rsi - 50.0) / 20.0
            feats[base_idx + 2] = float(np.clip(rsi_norm, -1.0, 1.0))

            atr = self._compute_atr(high, low, close, min(self.config.atr_period, int(close.size) - 1))
            atr_norm = atr / max(c, self._eps)
            feats[base_idx + 3] = float(np.clip(atr_norm * 50.0, 0.0, 1.0))


            per_tf_dbg[tf] = {
                "last_close": c,
                "trend_slope_norm": trend,
                "momentum_norm": mom_norm,
                "rsi": rsi,
                "atr": atr,
                "htf_sig": htf_sig,
                "feats": feats[base_idx : base_idx + 4],

                "raw": {
                    "close": close,
                    "high": high,
                    "low": low,
                    "tf_block": tf_data,
                },
            }

        dbg["per_tf"] = per_tf_dbg
        dbg["feats"] = feats
        return feats, dbg



    def _build_risk_features(
        self, risk_state: Dict[str, Any], account_state: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        feats = np.zeros(3, dtype=np.float32)
        dbg: Dict[str, Any] = {}

        drawdown = float(
            np.clip(self._to_float_required(account_state.get("current_drawdown"), "account_state.current_drawdown"), 0.0, 10.0)
        )
        feats[0] = float(np.clip(drawdown / max(self.config.max_drawdown_clip, self._eps), 0.0, 1.0))

        rb_val = float(np.clip(self._to_float_required(risk_state.get("risk_budget"), "risk_state.risk_budget"), 0.0, 1.0))
        feats[1] = rb_val

        portfolio_risk = risk_state.get("portfolio_risk")
        if not isinstance(portfolio_risk, dict):
            raise ObservationContractError("risk_state.portfolio_risk must be dict.")
        exposure = float(
            np.clip(self._to_float_required(portfolio_risk.get("total_exposure"), "portfolio_risk.total_exposure"), 0.0, 1.0)
        )
        feats[2] = exposure

        dbg.update({"drawdown": drawdown, "risk_budget": rb_val, "exposure": exposure, "feats": feats})
        return feats, dbg

    def _build_account_features(self, account_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        feats = np.zeros(5, dtype=np.float32)
        dbg: Dict[str, Any] = {}

        step = int(self._to_int_default(account_state.get("current_step"), 0))
        max_steps = int(self._to_int_default(account_state.get("max_steps"), 1))
        feats[0] = float(np.clip(step / max(max_steps, 1), 0.0, 1.0))

        pos_dir = account_state.get("position_direction")
        if isinstance(pos_dir, str):
            pos_dir = self._extract_direction(pos_dir)
        feats[1] = float(np.clip(self._to_float_required(pos_dir, "account_state.position_direction"), -1.0, 1.0))

        unreal = self._to_float_required(account_state.get("unrealized_pnl"), "account_state.unrealized_pnl")
        initial = self._to_float_required(account_state.get("initial_balance"), "account_state.initial_balance")
        feats[2] = float(np.clip(unreal / max(initial * 0.01, self._eps), -1.0, 1.0))

        tip = float(np.clip(self._to_float_required(account_state.get("time_in_position"), "account_state.time_in_position"), 0.0, 1e9))
        feats[3] = float(np.clip(tip / 100.0, 0.0, 1.0))

        feats[4] = float(np.clip(self._to_float_required(account_state.get("on_cooldown"), "account_state.on_cooldown"), 0.0, 1.0))

        dbg.update({"step": step, "max_steps": max_steps, "feats": feats})
        return feats, dbg


    def _build_trading_mode_features(self, trading_mode_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        feats = np.zeros(5, dtype=np.float32)
        dbg: Dict[str, Any] = {}

        timing = trading_mode_state.get("entry_timing")
        if not isinstance(timing, dict):
            raise ObservationContractError("trading_mode_state.entry_timing must be dict (strict).")

        feats[0] = 1.0 if bool(timing.get("entry_allowed")) else 0.0

        eql = float(np.clip(self._to_float_required(timing.get("entry_quality_long"), "entry_timing.entry_quality_long"), 0.0, 1.0))
        eqs = float(np.clip(self._to_float_required(timing.get("entry_quality_short"), "entry_timing.entry_quality_short"), 0.0, 1.0))
        feats[1] = float(np.clip((eql + eqs) * 0.5, 0.0, 1.0))

        setup_long = float(np.clip(self._to_float_required(timing.get("setup_quality_long"), "entry_timing.setup_quality_long"), 0.0, 1.0))
        setup_short = float(np.clip(self._to_float_required(timing.get("setup_quality_short"), "entry_timing.setup_quality_short"), 0.0, 1.0))
        feats[2] = float(np.clip((setup_long + setup_short) * 0.5, 0.0, 1.0))

        cert_long = float(np.clip(self._to_float_required(timing.get("entry_certainty_long"), "entry_timing.entry_certainty_long"), 0.0, 1.0))
        cert_short = float(np.clip(self._to_float_required(timing.get("entry_certainty_short"), "entry_timing.entry_certainty_short"), 0.0, 1.0))
        feats[3] = float(np.clip((cert_long + cert_short) * 0.5, 0.0, 1.0))

        confluence_count = float(self._to_float_required(timing.get("confluence_count"), "entry_timing.confluence_count"))
        feats[4] = float(np.clip(confluence_count / 8.0, 0.0, 1.0))

        dbg.update({"entry_allowed": bool(timing.get("entry_allowed")), "feats": feats})
        return feats, dbg

    def _build_governor_features(self, governor_state: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        feats = np.zeros(5, dtype=np.float32)
        dbg: Dict[str, Any] = {}

        feats[0] = float(np.clip(self._to_float_required(governor_state.get("loss_layer_ratio"), "governor.loss_layer_ratio"), 0.0, 1.0))

        raw_headroom = self._to_float_required(governor_state.get("session_pnl_headroom"), "governor.session_pnl_headroom")
        feats[1] = float(np.clip(raw_headroom / 2.0, 0.0, 1.0))

        feats[2] = float(np.clip(self._to_float_required(governor_state.get("session_trade_budget"), "governor.session_trade_budget"), 0.0, 1.0))
        feats[3] = float(np.clip(self._to_float_required(governor_state.get("session_consec_loss_ratio"), "governor.session_consec_loss_ratio"), 0.0, 1.0))
        feats[4] = float(np.clip(self._to_float_required(governor_state.get("session_progress"), "governor.session_progress"), 0.0, 1.0))

        dbg.update({"raw": governor_state, "feats": feats})
        return feats, dbg


    def _bus_get_required(self, bus: Any, key: str, module: str, *, build_id: int) -> Any:
        try:
            v = bus.get(key, module)
        except Exception as e:
            if self.config.debug:
                self._dbg.record("bus_get_error", {"build_id": build_id, "module": module, "key": key, "error": str(e)})
            raise ObservationContractError(f"[SmartBus] Failed get('{key}', module='{module}'): {e}") from e

        if self.config.debug and self.config.debug_record_bus_fetches:
            self._dbg.record("bus_get", {"build_id": build_id, "module": module, "key": key, "value": v})

        if v is None:
            raise ObservationContractError(f"[SmartBus] Missing required key '{key}' (module='{module}').")
        return v

    def _norm_symbol(self, s: Any) -> str:
        if not isinstance(s, str):
            return ""
        return "".join(ch for ch in s.strip().upper() if ch.isalnum())

    def _lookup_symbol_block_strict(self, mapping: Any, instrument: str, key_name: str) -> Dict[str, Any]:
        if not isinstance(mapping, dict):
            raise ObservationContractError(f"{key_name} must be a dict. Got: {type(mapping).__name__}")

        target = self._norm_symbol(instrument)
        if not target:
            raise ObservationContractError(f"Invalid instrument '{instrument}'.")

        direct = mapping.get(instrument)
        if isinstance(direct, dict):
            return direct

        for k, v in mapping.items():
            if isinstance(k, str) and isinstance(v, dict) and self._norm_symbol(k) == target:
                return v

        raise ObservationContractError(
            f"{key_name} does not contain instrument '{instrument}' (normalized '{target}'). "
            f"Available keys: {list(mapping.keys())[:50]}"
        )

    def _fetch_market_data_xauusd(self, bus: Any, module: str, *, build_id: int) -> Dict[str, Any]:
        for key in ("multi_timeframe_data", "market_data", "historical_prices"):
            blob = self._bus_get_required(bus, key, module, build_id=build_id)
            if isinstance(blob, dict) and self._is_timeframe_dict(blob):
                return blob
            if isinstance(blob, dict):
                block = self._lookup_symbol_block_strict(blob, DEFAULT_INSTRUMENT, key_name=key)
                if self._is_timeframe_dict(block):
                    return block
                raise ObservationContractError(f"{key}[{DEFAULT_INSTRUMENT}] is not a timeframe dict.")
        raise ObservationContractError("No valid market data source found on SmartBus (strict).")

    def _fetch_expert_signals_xauusd(self, bus: Any, module: str, *, build_id: int) -> Dict[str, Any]:
        def _dir_label(raw: Any) -> str:
            if isinstance(raw, (int, float, np.floating, np.integer)):
                x = float(raw)
                return "bullish" if x > 0 else ("bearish" if x < 0 else "neutral")
            s = str(raw).lower().strip()
            if s in ("bullish", "long", "buy", "up", "uptrend"):
                return "bullish"
            if s in ("bearish", "short", "sell", "down", "downtrend"):
                return "bearish"
            return "neutral"

        def _clip01(x: Any, name: str) -> float:
            return float(np.clip(self._to_float_required(x, name), 0.0, 1.0))

        def _expert_block(vote_key: str, conf_key: str, expert_name: str) -> Dict[str, Any]:
            proposal_raw = self._bus_get_required(bus, vote_key, module, build_id=build_id)
            conf = _clip01(self._bus_get_required(bus, conf_key, module, build_id=build_id), f"{conf_key}")

            def _camel_to_snake(s: str) -> str:
                out: List[str] = []
                prev_is_lower_or_digit = False
                for ch in str(s):
                    if ch.isupper() and out and prev_is_lower_or_digit:
                        out.append("_")
                    out.append(ch.lower())
                    prev_is_lower_or_digit = ch.islower() or ch.isdigit()
                return "".join(out)

            analysis_key = _camel_to_snake(expert_name.replace("Expert", "")) + "_analysis"
            analysis = self._bus_get_required(bus, analysis_key, module, build_id=build_id)
            if not isinstance(analysis, dict):
                raise ObservationContractError(f"{analysis_key} must be dict (strict).")
            per_inst = analysis.get("per_instrument")
            if not isinstance(per_inst, dict):
                raise ObservationContractError(f"{analysis_key}.per_instrument must be dict (strict).")
            inst_blob = self._lookup_symbol_block_strict(per_inst, DEFAULT_INSTRUMENT, key_name=f"{analysis_key}.per_instrument")
            if not isinstance(inst_blob, dict):
                raise ObservationContractError(f"{analysis_key}.per_instrument[{DEFAULT_INSTRUMENT}] must be dict.")

            direction_src: Any = inst_blob.get("action") or inst_blob.get("current_trend")
            if direction_src is None and isinstance(proposal_raw, dict):
                direction_src = proposal_raw.get("action") or proposal_raw.get("direction")
            if direction_src is None:
                direction_src = proposal_raw
            direction = _dir_label(direction_src)

            score = conf
            if "composite_score" in inst_blob:
                score = _clip01(inst_blob.get("composite_score"), f"{analysis_key}.composite_score")
            elif "trend_strength" in inst_blob:
                score = float(np.clip(abs(self._to_float_required(inst_blob.get("trend_strength"), f"{analysis_key}.trend_strength")), 0.0, 1.0))
            elif "composite_momentum" in inst_blob:
                score = float(np.clip(abs(self._to_float_required(inst_blob.get("composite_momentum"), f"{analysis_key}.composite_momentum")), 0.0, 1.0))

            proposal_dict = dict(inst_blob)

            if expert_name == "TrendExpert":
                for k in ("near_support", "near_resistance", "structure_trend", "bos_signal", "order_block_bull", "order_block_bear"):
                    if k not in proposal_dict:
                        raise ObservationContractError(f"trend_analysis missing required '{k}' for strict proposal.")
            if expert_name == "MomentumExpert":
                if "divergence_signal" not in proposal_dict and "divergence" in proposal_dict:
                    proposal_dict["divergence_signal"] = proposal_dict["divergence"]
                if "rsi" not in proposal_dict and "rsi_value" in proposal_dict:
                    proposal_dict["rsi"] = proposal_dict["rsi_value"]
                rsi_val = float(self._to_float_required(proposal_dict.get("rsi", 50.0), "momentum_analysis.rsi"))
                proposal_dict["overbought"] = max(0.0, (rsi_val - 70.0) / 30.0) if rsi_val > 70.0 else 0.0
                proposal_dict["oversold"] = max(0.0, (30.0 - rsi_val) / 30.0) if rsi_val < 30.0 else 0.0
                if "divergence_signal" not in proposal_dict:
                    proposal_dict["divergence_signal"] = "neutral"
            if expert_name == "ThemeExpert":
                for k in ("risk_regime", "volatility_regime"):
                    if k not in proposal_dict:
                        raise ObservationContractError(f"theme_analysis missing required '{k}' for strict proposal.")

            return {
                "proposal": proposal_dict,
                "direction": direction,
                "score": float(np.clip(score, 0.0, 1.0)),
                "confidence": float(np.clip(conf, 0.0, 1.0)),
                "raw_vote": proposal_raw,
            }

        experts = {
            "trend": _expert_block("TrendExpert_voting_proposal", "TrendExpert_confidence", "TrendExpert"),
            "momentum": _expert_block("MomentumExpert_voting_proposal", "MomentumExpert_confidence", "MomentumExpert"),
            "theme": _expert_block("ThemeExpert_voting_proposal", "ThemeExpert_confidence", "ThemeExpert"),
            "seasonality": _expert_block("SeasonalityRiskExpert_voting_proposal", "SeasonalityRiskExpert_confidence", "SeasonalityRiskExpert"),
        }

        market_regime = self._bus_get_required(bus, "market_regime", module, build_id=build_id)
        regime_strength = self._bus_get_required(bus, "regime_strength", module, build_id=build_id)
        market = {
            "regime": str(market_regime) if market_regime is not None else "unknown",
            "regime_strength": float(np.clip(self._to_float_required(regime_strength, "regime_strength"), 0.0, 1.0)),
        }

        htf_experts = self._extract_htf_experts_from_bus_strict(bus, module, build_id=build_id)

        out = {"experts": experts, "htf_experts": htf_experts, "market": market}
        self._validate_expert_signals(out)
        return out

    def _extract_htf_experts_from_bus_strict(self, bus: Any, module: str, *, build_id: int) -> Dict[str, Dict[str, Any]]:
        trend_analysis = self._bus_get_required(bus, "trend_analysis", module, build_id=build_id)
        if not isinstance(trend_analysis, dict):
            raise ObservationContractError("trend_analysis must be dict (strict).")
        per_inst = trend_analysis.get("per_instrument")
        if not isinstance(per_inst, dict):
            raise ObservationContractError("trend_analysis.per_instrument must be dict (strict).")
        inst = self._lookup_symbol_block_strict(per_inst, DEFAULT_INSTRUMENT, "trend_analysis.per_instrument")

        mtf_analysis = inst.get("mtf_analysis")
        if not isinstance(mtf_analysis, dict):


            mtf = inst.get("mtf")
            details = mtf.get("details") if isinstance(mtf, dict) else None
            if isinstance(details, dict):
                mtf_analysis = {"trends": {}}
                for tf in ["H1", "H4", "D1"]:
                    tfd = details.get(tf)
                    if isinstance(tfd, dict):
                        mtf_analysis["trends"][tf] = {
                            "direction": str(tfd.get("dir", "neutral")).lower(),
                            "strength": float(tfd.get("strength", 0.0) or 0.0),
                            "ma_spread": float(tfd.get("spread", 0.0) or 0.0),
                            "slope": float(tfd.get("slope", 0.0) or 0.0),
                        }
                    else:
                        mtf_analysis["trends"][tf] = {"direction": "neutral", "strength": 0.0, "ma_spread": 0.0, "slope": 0.0}
                if self.config.debug:
                    self._dbg.record(
                        "contract_fixup",
                        {"build_id": build_id, "source": "trend_analysis.per_instrument[XAUUSD].mtf", "added": "mtf_analysis"},
                    )
            else:
                raise ObservationContractError("trend_analysis.per_instrument[XAUUSD].mtf_analysis must be dict (strict).")
        mtf_trends = mtf_analysis.get("trends")
        if not isinstance(mtf_trends, dict):
            raise ObservationContractError("mtf_analysis.trends must be dict (strict).")

        mom_analysis = self._bus_get_required(bus, "momentum_analysis", module, build_id=build_id)
        if not isinstance(mom_analysis, dict):
            raise ObservationContractError("momentum_analysis must be dict (strict).")
        mom_per_inst = mom_analysis.get("per_instrument")
        if not isinstance(mom_per_inst, dict):
            raise ObservationContractError("momentum_analysis.per_instrument must be dict (strict).")
        mom_inst = self._lookup_symbol_block_strict(mom_per_inst, DEFAULT_INSTRUMENT, "momentum_analysis.per_instrument")
        rsi_val = float(self._to_float_required(mom_inst.get("rsi"), "momentum_analysis.per_instrument[XAUUSD].rsi"))

        adx_val = float(self._to_float_default(inst.get("adx"), 0.0))

        out: Dict[str, Dict[str, Any]] = {}
        for tf in ["H1", "H4", "D1"]:
            tf_trend = mtf_trends.get(tf)
            if not isinstance(tf_trend, dict):
                raise ObservationContractError(f"mtf_analysis.trends missing dict for '{tf}' (strict).")

            direction = str(tf_trend.get("direction", "neutral")).lower().strip()
            strength = float(np.clip(abs(self._to_float_required(tf_trend.get("strength"), f"mtf_trends.{tf}.strength")), 0.0, 1.0))

            ma_spread = float(self._to_float_default(tf_trend.get("ma_spread"), 0.0))
            ma_alignment = 1 if ma_spread > 0.001 else (-1 if ma_spread < -0.001 else 0)

            rsi_signal = "neutral"
            if rsi_val > 70:
                rsi_signal = "overbought"
            elif rsi_val < 30:
                rsi_signal = "oversold"

            slope = float(self._to_float_default(tf_trend.get("slope"), 0.0))
            structure_bias = float(np.clip(slope * 50.0, -1.0, 1.0))

            out[tf] = {
                "trend_direction": direction,
                "trend_strength": strength,
                "momentum_direction": direction,
                "momentum_strength": float(np.clip(strength * 0.8, 0.0, 1.0)),
                "rsi": rsi_val,
                "rsi_signal": rsi_signal,
                "ma_alignment": ma_alignment,
                "adx": adx_val,
                "structure_bias": structure_bias,
                "raw_tf_trend": tf_trend,
            }
        return out


    def _fetch_risk_state_strict(self, bus: Any, module: str, *, build_id: int) -> Dict[str, Any]:
        portfolio_risk = self._bus_get_required(bus, "portfolio_risk", module, build_id=build_id)
        if not isinstance(portfolio_risk, dict):
            raise ObservationContractError("portfolio_risk must be dict (strict).")
        if "total_exposure" not in portfolio_risk:
            raise ObservationContractError("portfolio_risk.total_exposure missing (strict).")

        risk_data = self._bus_get_required(bus, "risk_data", module, build_id=build_id)
        if not isinstance(risk_data, dict):
            raise ObservationContractError("risk_data must be dict (strict).")

        if "risk_budget_available" in risk_data:
            rb = float(np.clip(self._to_float_required(risk_data.get("risk_budget_available"), "risk_data.risk_budget_available"), 0.0, 1.0))
        elif "risk_budget_used" in risk_data:
            used = float(np.clip(self._to_float_required(risk_data.get("risk_budget_used"), "risk_data.risk_budget_used"), 0.0, 1.0))
            rb = max(0.0, 1.0 - used)
        else:
            raise ObservationContractError("risk_data must contain 'risk_budget_available' or 'risk_budget_used' (strict).")

        return {"risk_data": risk_data, "portfolio_risk": portfolio_risk, "risk_budget": rb}


    def _fetch_account_state_xauusd(self, bus: Any, module: str, *, build_id: int) -> Dict[str, Any]:
        market_state = self._bus_get_required(bus, "market_state", module, build_id=build_id)
        if not isinstance(market_state, dict):
            raise ObservationContractError("market_state must be dict (strict).")

        positions = self._bus_get_required(bus, "positions", module, build_id=build_id)
        if not isinstance(positions, dict):
            raise ObservationContractError("positions must be dict (strict).")

        inst_pos = positions.get(DEFAULT_INSTRUMENT)
        if not isinstance(inst_pos, dict):
            for k, v in positions.items():
                if isinstance(k, str) and isinstance(v, dict) and self._norm_symbol(k) == self._norm_symbol(DEFAULT_INSTRUMENT):
                    inst_pos = v
                    break
        if not isinstance(inst_pos, dict):
            inst_pos = {}

        direction_raw = inst_pos.get("direction")
        if direction_raw is None:
            side = inst_pos.get("side")
            if isinstance(side, (int, float, np.integer, np.floating)):
                if float(side) > 0:
                    direction = 1.0
                elif float(side) < 0:
                    direction = -1.0
                else:
                    direction = 0.0
            else:
                direction = float(self._extract_direction(inst_pos))
        elif isinstance(direction_raw, str):
            direction = float(self._extract_direction(direction_raw))
        else:
            direction = float(np.clip(self._to_float_default(direction_raw, 0.0), -1.0, 1.0))

        pos_size = 0.0
        size_raw = inst_pos.get("size")
        if isinstance(size_raw, (int, float, np.integer, np.floating)) and 0.0 <= float(size_raw) <= 1.0:
            pos_size = float(size_raw)
        else:
            notional = inst_pos.get("notional_eur", inst_pos.get("notional", 0.0))
            if not isinstance(notional, (int, float, np.integer, np.floating)):
                units = inst_pos.get("units")
                entry_price = inst_pos.get("entry_price")
                if isinstance(units, (int, float, np.integer, np.floating)) and isinstance(entry_price, (int, float, np.integer, np.floating)):
                    notional = abs(float(units) * float(entry_price))
                else:
                    notional = 0.0
            initial_balance_for_size = float(
                self._to_float_default(market_state.get("initial_balance"), market_state.get("balance") or 0.0)
            )
            pos_size = float(np.clip(abs(float(notional)) / max(initial_balance_for_size, self._eps), 0.0, 1.0))

        unrealized_pnl = float(self._to_float_default(inst_pos.get("unrealized_pnl"), 0.0))

        try:
            cd = self._bus_get_required(bus, "instrument_cooldown_state", module, build_id=build_id)
        except ObservationContractError:
            cd = {}
        if not isinstance(cd, dict):
            cd = {}

        inst_cd = cd.get(DEFAULT_INSTRUMENT)
        if not isinstance(inst_cd, dict):
            for k, v in cd.items():
                if isinstance(k, str) and isinstance(v, dict) and self._norm_symbol(k) == self._norm_symbol(DEFAULT_INSTRUMENT):
                    inst_cd = v
                    break
        if not isinstance(inst_cd, dict):
            inst_cd = {}

        on_cd = bool(inst_cd.get("on_cooldown"))
        remaining = float(self._to_float_default(inst_cd.get("cooldown_remaining"), 0.0))
        on_cooldown = 1.0 if on_cd and remaining > 0.0 else 0.0

        time_in_position_raw = None
        try:
            time_in_position_raw = self._bus_get_required(bus, "time_in_position", module, build_id=build_id)
        except ObservationContractError:
            time_in_position_raw = None

        if time_in_position_raw is not None:
            time_in_position = float(self._to_float_default(time_in_position_raw, 0.0))
        else:
            age_hours = inst_pos.get("age_hours")
            if isinstance(age_hours, (int, float, np.integer, np.floating)):
                time_in_position = max(0.0, float(age_hours) * 60.0)
            else:
                time_in_position = 0.0

        state: Dict[str, Any] = {
            "balance": float(self._to_float_required(market_state.get("balance"), "market_state.balance")),
            "initial_balance": float(self._to_float_required(market_state.get("initial_balance"), "market_state.initial_balance")),
            "current_drawdown": float(self._to_float_required(market_state.get("drawdown"), "market_state.drawdown")),
            "current_step": int(self._to_int_default(market_state.get("step"), 0)),
            "max_steps": int(self._to_int_default(market_state.get("max_steps"), 1)),
            "win_rate": float(self._to_float_required(market_state.get("win_rate"), "market_state.win_rate")),
            "pnl_trend": float(self._to_float_required(market_state.get("pnl_trend"), "market_state.pnl_trend")),
            "trades_today": int(self._to_int_default(market_state.get("trades_today"), 0)),
            "episode_return": float(self._to_float_default(market_state.get("episode_return"), 0.0)),
            "position_direction": float(direction),
            "position_size": float(pos_size),
            "unrealized_pnl": float(unrealized_pnl),
            "time_in_position": float(self._to_float_default(time_in_position, 0.0)),
            "on_cooldown": float(on_cooldown),
        }
        return state


    def _fetch_trading_mode_state_xauusd(self, bus: Any, module: str, *, build_id: int) -> Dict[str, Any]:
        trading_mode = self._bus_get_required(bus, "trading_mode", module, build_id=build_id)
        mode_stats = self._bus_get_required(bus, "mode_stats", module, build_id=build_id)
        entry_timing_all = self._bus_get_required(bus, "entry_timing", module, build_id=build_id)

        if not isinstance(mode_stats, dict):
            raise ObservationContractError("mode_stats must be dict (strict).")
        if not isinstance(entry_timing_all, dict):
            raise ObservationContractError("entry_timing must be dict (strict, per instrument).")

        timing = entry_timing_all.get(DEFAULT_INSTRUMENT)
        if not isinstance(timing, dict):
            for k, v in entry_timing_all.items():
                if isinstance(k, str) and isinstance(v, dict) and self._norm_symbol(k) == self._norm_symbol(DEFAULT_INSTRUMENT):
                    timing = v
                    break
        if not isinstance(timing, dict):
            raise ObservationContractError("entry_timing[XAUUSD] missing (strict).")

        out: Dict[str, Any] = {
            "trading_mode": str(trading_mode),
            "mode_stats": mode_stats,
            "entry_timing": timing,
            "regime_stability": self._bus_get_required(bus, "regime_stability", module, build_id=build_id),
            "theme_transition": self._bus_get_required(bus, "theme_transition", module, build_id=build_id),
            "regime_accuracy": self._bus_get_required(bus, "regime_accuracy", module, build_id=build_id),
            "risk_scaling_factor": self._bus_get_required(bus, "risk_scaling_factor", module, build_id=build_id),
            "liquidity_score": self._bus_get_required(bus, "liquidity_score", module, build_id=build_id),
            "theme_strength": self._bus_get_required(bus, "theme_strength", module, build_id=build_id),
        }
        self._validate_trading_mode_state(out)
        return out

    def _fetch_governor_state_strict(self, bus: Any, module: str, *, build_id: int) -> Dict[str, Any]:
        st = self._bus_get_required(bus, "governor_state", module, build_id=build_id)
        if not isinstance(st, dict):
            raise ObservationContractError("governor_state must be dict (strict).")
        self._validate_governor_state(st)
        return st


    def _validate_ohlc_block(self, block: Dict[str, Any], tf: str, min_bars: int) -> None:
        if not isinstance(block, dict):
            raise ObservationContractError(f"{tf} block must be dict.")
        for k in ("open", "high", "low", "close"):
            if k not in block:
                raise ObservationContractError(f"{tf} block missing required key '{k}' (strict).")
        close = self._as_1d_float_array(block.get("close"), f"{tf}.close")
        self._require_min_len(close, min_bars, f"{tf}.close")

    def _validate_expert_signals(self, expert_signals: Dict[str, Any]) -> None:
        # Only htf_experts survives v7.0. The per-expert "experts" dict fed the
        # voting, committee and expert_raw blocks, all of which were removed
        # after measuring no skill against an always-long control - it is now
        # read nowhere, so requiring it would force the environment to compute
        # trend/momentum/theme signals that reach nothing.
        htf = expert_signals.get("htf_experts")
        if not isinstance(htf, dict):
            raise ObservationContractError("expert_signals.htf_experts must be a dict (strict).")
        for tf in CONTEXT_TIMEFRAMES:
            if not isinstance(htf.get(tf), dict):
                raise ObservationContractError(f"htf_experts missing dict for '{tf}' (strict).")
            for key in ("rsi", "trend_direction", "ma_alignment"):
                if key not in htf[tf]:
                    raise ObservationContractError(f"htf_experts.{tf} missing key '{key}' (strict).")


    def _validate_trading_mode_state(self, st: Dict[str, Any]) -> None:
        for k in ("trading_mode", "mode_stats", "entry_timing", "regime_stability", "theme_transition", "regime_accuracy", "risk_scaling_factor", "liquidity_score", "theme_strength"):
            if k not in st:
                raise ObservationContractError(f"trading_mode_state missing '{k}' (strict).")
        if not isinstance(st.get("mode_stats"), dict):
            raise ObservationContractError("trading_mode_state.mode_stats must be dict (strict).")
        if not isinstance(st.get("entry_timing"), dict):
            raise ObservationContractError("trading_mode_state.entry_timing must be dict (strict).")
        if "mode_effectiveness" not in st["mode_stats"]:
            raise ObservationContractError("mode_stats.mode_effectiveness missing (strict).")
        timing = st["entry_timing"]
        for k in ("entry_allowed", "entry_quality_long", "entry_quality_short", "zone_type", "vol_state", "in_prime_window"):
            if k not in timing:
                raise ObservationContractError(f"entry_timing missing '{k}' (strict).")

    def _validate_governor_state(self, st: Dict[str, Any]) -> None:
        req = (
            "loss_layer_ratio",
            "loss_layer_level",
            "win_streak_ratio",
            "session_pnl_headroom",
            "session_trade_budget",
            "session_consec_loss_ratio",
            "session_progress",
            "pending_order_progress",
        )
        for k in req:
            if k not in st:
                raise ObservationContractError(f"governor_state missing '{k}' (strict).")


    def _is_timeframe_dict(self, d: Any) -> bool:
        return isinstance(d, dict) and all(tf in d and isinstance(d.get(tf), dict) for tf in SUPPORTED_TIMEFRAMES)

    def _extract_timeframe_data(self, market_data: Dict[str, Any], timeframe: str) -> Dict[str, Any]:
        if timeframe not in market_data or not isinstance(market_data[timeframe], dict):
            raise ObservationContractError(f"market_data missing timeframe '{timeframe}' (strict).")
        return self._apply_forming_bar(market_data[timeframe], timeframe)

    def _apply_forming_bar(self, tf_data: Dict[str, Any], tf: str) -> Dict[str, Any]:
        if not self.config.use_forming_bar:
            return tf_data

        cur_bar = tf_data.get("current_bar")
        if not isinstance(cur_bar, dict):
            raise ObservationContractError(f"{tf}.current_bar missing dict while use_forming_bar=True (strict).")

        required = ("close", "high", "low", "volume")
        for k in required:
            if k not in cur_bar:
                raise ObservationContractError(f"{tf}.current_bar missing '{k}' (strict).")

        result = dict(tf_data)

        close = list(self._as_1d_float_array(result.get("close"), f"{tf}.close"))
        high = list(self._as_1d_float_array(result.get("high"), f"{tf}.high"))
        low = list(self._as_1d_float_array(result.get("low"), f"{tf}.low"))
        vol = result.get("volume")
        vol_list = list(self._as_1d_float_array(vol, f"{tf}.volume")) if vol is not None else None

        last_close = float(close[-1])
        f_close = float(self._to_float_required(cur_bar.get("close"), f"{tf}.current_bar.close"))

        tol = max(self.config.forming_bar_atol, abs(last_close) * self.config.forming_bar_rtol)
        if abs(f_close - last_close) <= tol:
            return tf_data

        close[-1] = f_close
        high[-1] = float(self._to_float_required(cur_bar.get("high"), f"{tf}.current_bar.high"))
        low[-1] = float(self._to_float_required(cur_bar.get("low"), f"{tf}.current_bar.low"))
        if vol_list is not None:
            vol_list[-1] = float(self._to_float_required(cur_bar.get("volume"), f"{tf}.current_bar.volume"))

        result["close"] = close
        result["high"] = high
        result["low"] = low
        if vol_list is not None:
            result["volume"] = vol_list

        return result


    def _find_sr_levels(self, highs: np.ndarray, lows: np.ndarray, window: int = 2) -> Tuple[List[float], List[float]]:
        highs = np.asarray(highs, dtype=np.float64)
        lows = np.asarray(lows, dtype=np.float64)
        if highs.size < (2 * window + 1) or lows.size < (2 * window + 1):
            raise ObservationContractError("Insufficient bars for S/R swing detection (strict).")

        res: List[float] = []
        sup: List[float] = []

        for i in range(window, int(highs.size) - window):
            h = highs[i]
            if np.all(h > highs[i - window : i]) and np.all(h > highs[i + 1 : i + window + 1]):
                res.append(float(h))
            l = lows[i]
            if np.all(l < lows[i - window : i]) and np.all(l < lows[i + 1 : i + window + 1]):
                sup.append(float(l))

        sup = self._cluster_levels(sorted(set(sup)), tol_pct=self.config.sr_cluster_tol_pct)
        res = self._cluster_levels(sorted(set(res), reverse=True), tol_pct=self.config.sr_cluster_tol_pct)

        return sup[:3], res[:3]

    def _cluster_levels(self, levels: List[float], tol_pct: float) -> List[float]:
        if not levels:
            return []
        clustered: List[float] = [levels[0]]
        for x in levels[1:]:
            last = clustered[-1]
            tol = max(abs(last), self._eps) * tol_pct
            if abs(x - last) <= tol:
                clustered[-1] = float((last + x) * 0.5)
            else:
                clustered.append(x)
        return clustered

    def _compute_sr_proximity(self, current_price: float, support: List[float], resistance: List[float], threshold_pct: float) -> Tuple[float, float]:
        if current_price <= 0:
            raise ObservationContractError("current_price must be > 0 for SR proximity (strict).")

        thr = current_price * float(threshold_pct)
        if thr <= 0:
            raise ObservationContractError("Invalid SR threshold (strict).")

        near_support = 0.0
        near_resistance = 0.0

        for s in support:
            dist = abs(current_price - float(s))
            if dist < thr:
                near_support = max(near_support, 1.0 - dist / thr)

        for r in resistance:
            dist = abs(current_price - float(r))
            if dist < thr:
                near_resistance = max(near_resistance, 1.0 - dist / thr)

        return float(np.clip(near_support, 0.0, 1.0)), float(np.clip(near_resistance, 0.0, 1.0))


    def _ema_series(self, data: np.ndarray, period: int) -> np.ndarray:
        data = np.asarray(data, dtype=np.float64)
        if data.size == 0:
            raise ObservationContractError("EMA input array empty (strict).")
        if period < 1:
            raise ObservationContractError("EMA period must be >=1 (strict).")

        alpha = 2.0 / (period + 1.0)
        ema = np.empty_like(data, dtype=np.float64)
        ema[0] = float(data[0])
        for i in range(1, int(data.size)):
            ema[i] = alpha * float(data[i]) + (1.0 - alpha) * float(ema[i - 1])
        return ema

    def _compute_macd_histogram(self, close: np.ndarray) -> float:
        close = np.asarray(close, dtype=np.float64)
        if close.size < 35:
            raise ObservationContractError("M15.close must have >=35 bars for MACD histogram (strict).")
        ema12 = self._ema_series(close, 12)
        ema26 = self._ema_series(close, 26)
        macd_line = ema12 - ema26
        signal = self._ema_series(macd_line, 9)
        hist = macd_line - signal
        return float(hist[-1])

    def _compute_rsi(self, close: np.ndarray, period: int = 14) -> float:
        close = np.asarray(close, dtype=np.float64)
        if close.size < period + 1:
            raise ObservationContractError(f"close must have >= {period + 1} bars for RSI (strict).")

        window = close[-(period + 1):]
        deltas = np.diff(window)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))

        if avg_loss < self._eps:
            return 100.0 if avg_gain > 0 else 50.0

        rs = avg_gain / avg_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _compute_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        close = np.asarray(close, dtype=np.float64)

        if close.size < 2:
            raise ObservationContractError("close must have >=2 bars for ATR (strict).")

        period = min(int(period), int(close.size) - 1)
        if period < 1:
            raise ObservationContractError("ATR period invalid after clipping (strict).")

        trs = []
        for i in range(-period, 0):
            h = float(high[i])
            l = float(low[i])
            pc = float(close[i - 1])
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return float(np.mean(trs))

    def _compute_trend_slope_norm(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, lookback: int) -> float:
        close = np.asarray(close, dtype=np.float64)
        high = np.asarray(high, dtype=np.float64)
        low = np.asarray(low, dtype=np.float64)
        lookback = int(min(lookback, close.size))
        if lookback < 5:
            raise ObservationContractError("lookback too small for trend slope (strict).")

        window = close[-lookback:]
        ema = self._ema_series(window, period=min(20, lookback))
        slope = self._linear_regression_slope(ema)

        atr = self._compute_atr(high[-lookback:], low[-lookback:], close[-lookback:], period=min(self.config.atr_period, lookback - 1))
        denom = max(atr, self._eps)
        return float(np.clip(slope / denom, -2.0, 2.0))

    def _compute_momentum_norm(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, bars: int = 5) -> float:
        close = np.asarray(close, dtype=np.float64)
        if close.size < bars + 1:
            raise ObservationContractError("Insufficient bars for momentum_norm (strict).")
        ret = (float(close[-1]) - float(close[-(bars + 1)])) / max(abs(float(close[-(bars + 1)])), self._eps)
        atr = self._compute_atr(high, low, close, period=min(self.config.atr_period, int(close.size) - 1))
        atr_pct = atr / max(float(close[-1]), self._eps)
        denom = max(atr_pct * float(bars), self._eps)
        return float(np.clip(ret / denom, -2.0, 2.0))

    def _linear_regression_slope(self, y: np.ndarray) -> float:
        y = np.asarray(y, dtype=np.float64)
        n = int(y.size)
        if n < 2:
            raise ObservationContractError("Need >=2 points for slope (strict).")
        x = np.arange(n, dtype=np.float64)
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        num = float(np.sum((x - x_mean) * (y - y_mean)))
        den = float(np.sum((x - x_mean) ** 2))
        return num / max(den, self._eps)

    def _blend_with_htf_expert_trend(self, raw_trend: float, htf_sig: Dict[str, Any]) -> float:
        direction = str(htf_sig.get("trend_direction")).lower().strip()
        strength = float(np.clip(self._to_float_required(htf_sig.get("trend_strength"), "htf_sig.trend_strength"), 0.0, 1.0))
        expert = strength if direction == "bullish" else (-strength if direction == "bearish" else 0.0)

        ma_alignment = int(self._to_int_default(htf_sig.get("ma_alignment"), 0))
        blended = 0.65 * float(raw_trend) + 0.35 * float(expert)
        if ma_alignment != 0 and ma_alignment * np.sign(blended) > 0:
            blended = float(np.sign(blended) * min(abs(blended) * 1.15, 1.5))
        return float(np.clip(blended, -2.0, 2.0))

    def _blend_with_htf_expert_momentum(self, raw_mom: float, htf_sig: Dict[str, Any]) -> float:
        direction = str(htf_sig.get("momentum_direction")).lower().strip()
        strength = float(np.clip(self._to_float_required(htf_sig.get("momentum_strength"), "htf_sig.momentum_strength"), 0.0, 1.0))
        expert = strength if direction == "bullish" else (-strength if direction == "bearish" else 0.0)
        blended = 0.65 * float(raw_mom) + 0.35 * float(expert)
        return float(np.clip(blended, -2.0, 2.0))


    def _require_dict(self, v: Any, name: str) -> None:
        if not isinstance(v, dict):
            raise ObservationContractError(f"{name} must be dict (strict). Got {type(v).__name__}")

    def _require_min_len(self, arr: np.ndarray, n: int, name: str) -> None:
        if int(arr.size) < int(n):
            raise ObservationContractError(f"{name} must have >= {n} bars (strict). Got {int(arr.size)}")

    def _require_same_len(self, arrays: List[np.ndarray], names: List[str], scope: str) -> None:
        lens = [int(a.size) for a in arrays]
        if len(set(lens)) != 1:
            raise ObservationContractError(f"{scope} arrays must have same length (strict). Got {dict(zip(names, lens))}")

    def _as_1d_float_array(self, v: Any, name: str) -> np.ndarray:
        if v is None:
            raise ObservationContractError(f"{name} missing (strict).")
        arr = np.asarray(v, dtype=np.float64)
        if arr.ndim != 1:
            raise ObservationContractError(f"{name} must be 1D array-like (strict). Got shape {arr.shape}")
        if arr.size == 0:
            raise ObservationContractError(f"{name} empty (strict).")
        if not np.all(np.isfinite(arr)):
            raise ObservationContractError(f"{name} contains NaN/Inf (strict).")
        return arr

    def _to_float_required(self, v: Any, name: str) -> float:
        try:
            x = float(v)
        except Exception as e:
            raise ObservationContractError(f"{name} must be float-convertible (strict). Got {v!r}") from e
        if not np.isfinite(x):
            raise ObservationContractError(f"{name} must be finite (strict). Got {x}")
        return x

    def _to_float_default(self, v: Any, default: float) -> float:
        try:
            x = float(v)
            return x if np.isfinite(x) else default
        except Exception:
            return default

    def _to_int_default(self, v: Any, default: int) -> int:
        try:
            return int(v)
        except Exception:
            return default


    def _extract_direction(self, proposal: Any) -> float:
        if isinstance(proposal, dict):
            direction = proposal.get("direction") or proposal.get("action") or proposal.get("global_direction")
        elif isinstance(proposal, (int, float, np.integer, np.floating)):
            return float(np.clip(float(proposal), -1.0, 1.0))
        else:
            direction = proposal
        s = str(direction or "flat").lower().strip()
        if s in ("long", "buy", "bullish", "up"):
            return 1.0
        if s in ("short", "sell", "bearish", "down"):
            return -1.0
        return 0.0


_default_builder: Optional[PPOObservationBuilder] = None


def get_ppo_observation_builder(config: Optional[PPOObservationConfig] = None) -> PPOObservationBuilder:
    global _default_builder
    if config is not None:
        _default_builder = PPOObservationBuilder(config=config)
        return _default_builder
    if _default_builder is None:
        _default_builder = PPOObservationBuilder()
    return _default_builder


def build_ppo_observation(
    market_data: Optional[Dict[str, Any]] = None,
    expert_signals: Optional[Dict[str, Any]] = None,
    committee_state: Optional[Dict[str, Any]] = None,
    risk_state: Optional[Dict[str, Any]] = None,
    memory_state: Optional[Dict[str, Any]] = None,
    account_state: Optional[Dict[str, Any]] = None,
    world_model_state: Optional[Dict[str, Any]] = None,
    trading_mode_state: Optional[Dict[str, Any]] = None,
    governor_state: Optional[Dict[str, Any]] = None,
    smart_bus: Optional[Any] = None,
    module_name: str = "PPOObservationBuilder",
) -> np.ndarray:
    builder = get_ppo_observation_builder()
    return builder.build(
        market_data=market_data,
        expert_signals=expert_signals,
        risk_state=risk_state,
        account_state=account_state,
        trading_mode_state=trading_mode_state,
        governor_state=governor_state,
        smart_bus=smart_bus,
        module_name=module_name,
    )


def build_ppo_observation_for_instrument(instrument: str, **kwargs: Any) -> np.ndarray:
    builder = get_ppo_observation_builder()
    return builder.build_for_instrument(instrument=instrument, **kwargs)
