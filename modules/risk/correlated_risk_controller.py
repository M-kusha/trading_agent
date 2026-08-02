
from __future__ import annotations

import ast
import datetime
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.contracts import module_args
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.module_base import BaseModule, module
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.info_bus import InfoBusManager
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities


@dataclass
class CorrelatedRiskConfig:

    health_check_interval: int = 30
    circuit_breaker_threshold: int = 5
    max_processing_time_ms: float = 60.0
    status_key: str = "correlated_risk_status"
    health_key: str = "correlated_risk_health"


    max_correlation: float = 0.80
    warning_correlation: float = 0.60
    min_diversification: float = 0.30
    cluster_link_threshold: float = 0.55


    lookback_window: int = 120
    min_samples_for_corr: int = 24


    enabled: bool = True


@module(
    **module_args(
        "CorrelatedRiskController",
        description="Enhanced correlation risk monitoring with intelligent clustering and diversification analysis",
        error_handling=True,
        hot_reload=True,
        timeout_ms=3000,
    )
)
class CorrelatedRiskController(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):


    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):

        cfg_dict = asdict(CorrelatedRiskConfig())
        if isinstance(config, dict):
            for k, v in config.items():
                if k in cfg_dict:
                    cfg_dict[k] = v
        self._cfg = CorrelatedRiskConfig(**cfg_dict)
        self.config = config or {}


        self._initialize_advanced_systems()


        self.circuit_breaker: Dict[str, Any] = {
            "failures": 0,
            "last_failure": 0.0,
            "state": "CLOSED",
            "threshold": int(self._cfg.circuit_breaker_threshold),
            "cooldown_sec": 20.0,
        }
        self._processing_times: deque[float] = deque(maxlen=100)
        self._health_status: str = "healthy"
        self._monitoring_active: bool = False
        self._lock = threading.RLock()


        self.max_correlation: float = float(self._cfg.max_correlation)
        self.warning_correlation: float = float(self._cfg.warning_correlation)
        self.min_diversification: float = float(self._cfg.min_diversification)
        self.lookback_window: int = int(self._cfg.lookback_window)
        self.min_samples_for_corr: int = int(self._cfg.min_samples_for_corr)
        self.cluster_link_threshold: float = float(self._cfg.cluster_link_threshold)
        self.enabled: bool = bool(self._cfg.enabled)


        self.correlation_matrix: Dict[Tuple[str, str], float] = {}
        self.correlation_history: deque[Dict[Tuple[str, str], float]] = deque(maxlen=100)
        self.price_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.lookback_window))
        self.return_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.lookback_window))


        self.correlation_risk_score: float = 0.0
        self.diversification_score: float = 1.0
        self.cluster_risk_score: float = 0.0
        self.severity_level: str = "normal"


        self.correlation_clusters: Dict[int, List[str]] = {}
        self.regime_correlations = defaultdict(lambda: defaultdict(list))


        self.step_count: int = 0
        self.correlation_violations: int = 0
        self.diversification_violations: int = 0

        super().__init__()
        self._start_monitoring()

        self.logger.info(
            format_operator_message(
                message="Enhanced Correlated Risk Controller initialized",
                icon="🔗",
                max_correlation=f"{self.max_correlation:.2f}",
                warning_threshold=f"{self.warning_correlation:.2f}",
                min_diversification=f"{self.min_diversification:.2f}",
                lookback=self.lookback_window,
                enabled=self.enabled,
            )
        )

    def _initialize_advanced_systems(self) -> None:
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="CorrelatedRiskController",
            log_path="logs/risk/correlated_risk_controller.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True,
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("CorrelatedRiskController", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()


    def _initialize(self) -> None:
        try:

            status = {
                "enabled": bool(self.enabled),
                "severity_level": str(self.severity_level),
                "risk_score": float(self.correlation_risk_score),
                "ts": datetime.datetime.now().isoformat(),
            }
            self.smart_bus.set(
                self._cfg.status_key,
                status,
                module="CorrelatedRiskController",
                thesis="Initial correlated risk status",
            )


            self.correlation_matrix.clear()
            self.correlation_history.clear()
            self.price_history.clear()
            self.return_history.clear()
            self.correlation_risk_score = 0.0
            self.diversification_score = 1.0
            self.cluster_risk_score = 0.0
            self.severity_level = "normal"
            self.step_count = 0
            self.correlation_violations = 0
            self.diversification_violations = 0


            baseline_corr = {
                "correlation_risk_score": float(self.correlation_risk_score),
                "severity_level": str(self.severity_level),
                "risk_metrics": {
                    "violation_risk": 0.0,
                    "diversification_risk": 0.0,
                    "cluster_risk_score": float(self.cluster_risk_score),
                },
                "summary": {
                    "pairs_analyzed": 0,
                    "instruments_analyzed": 0,
                    "processing_time_ms": 0.0,
                },
                "timestamp": datetime.datetime.now().isoformat(),
                "thesis": "Baseline correlation risk initialized",
            }
            try:
                self.smart_bus.set(
                    "correlation_risk",
                    baseline_corr,
                    module="CorrelatedRiskController",
                    thesis="Baseline correlation risk initialized",
                )
                self.smart_bus.set(
                    "diversification_score",
                    float(self.diversification_score),
                    module="CorrelatedRiskController",
                    thesis="Baseline diversification score initialized",
                )
                self.smart_bus.set(
                    "correlation_clusters",
                    {},
                    module="CorrelatedRiskController",
                    thesis="Baseline correlation clusters initialized",
                )

                self.smart_bus.set(
                    "correlation_matrix",
                    {},
                    module="CorrelatedRiskController",
                    thesis="Baseline correlation matrix initialized",
                )
            except Exception as e:
                self.logger.warning(f"Baseline bus publish soft-fail: {e}")

            self.logger.info("Correlated Risk Controller initialization completed successfully")
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "controller_initialization")
            self.logger.error(f"Controller initialization failed: {error_context}")


    def _start_monitoring(self) -> None:
        if self._monitoring_active:
            return

        def loop() -> None:
            self._monitoring_active = True
            self.logger.info("[MONITOR] CorrelatedRiskController health monitor started.")
            while self._monitoring_active:
                try:
                    self._update_health()


                    health = self.get_health_status()
                    self.smart_bus.set(
                        self._cfg.health_key,
                        health,
                        module="CorrelatedRiskController",
                        thesis="Correlation risk health heartbeat",
                    )


                    if self.circuit_breaker["state"] == "OPEN":
                        if (
                            time.time() - self.circuit_breaker["last_failure"]
                            >= self.circuit_breaker["cooldown_sec"]
                        ):
                            self.circuit_breaker["state"] = "CLOSED"
                            self.circuit_breaker["failures"] = 0
                            self.logger.info("[MONITOR] Circuit breaker auto-reset to CLOSED.")
                except Exception as e:
                    self.logger.error(f"Correlation monitoring error: {e}")
                time.sleep(max(1, int(self._cfg.health_check_interval)))

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def stop_monitoring(self) -> None:
        self._monitoring_active = False

    def _update_health(self) -> None:
        try:
            self._health_status = "healthy"
            if len(self._processing_times) >= 10:
                avg_ms = float(np.mean(list(self._processing_times)[-10:]) * 1000.0)
                if avg_ms > float(self._cfg.max_processing_time_ms):
                    self._health_status = "warning"
            if self.circuit_breaker["state"] == "OPEN":
                self._health_status = "warning"
        except Exception as e:
            self.logger.error(f"Correlation health update failed: {e}")
            self._health_status = "warning"

    def get_health_status(self) -> Dict[str, Any]:
        avg_ms = float(np.mean(self._processing_times) * 1000.0) if self._processing_times else 0.0
        return {
            "status": self._health_status,
            "avg_processing_time_ms": avg_ms,
            "circuit_breaker_state": self.circuit_breaker["state"],
            "ts": datetime.datetime.now().isoformat(),
        }


    async def calculate_confidence(self, action: Dict[str, Any], **kwargs: Any) -> float:
        try:
            confidence = 0.9

            confidence *= float(np.clip(self.diversification_score, 0.0, 1.0))
            confidence *= float(np.clip(1.0 - self.correlation_risk_score, 0.0, 1.0))


            instruments_count = len(self.price_history)
            if instruments_count >= 5:
                data_factor = min(1.0, instruments_count / 10.0)
            else:
                data_factor = max(0.3, instruments_count / 5.0)
            confidence *= float(data_factor)


            severity_penalties = {
                "normal": 1.0,
                "elevated": 0.9,
                "warning": 0.75,
                "critical": 0.55,
                "error": 0.4,
                "disabled": 0.1,
            }
            confidence *= float(severity_penalties.get(self.severity_level, 0.6))

            return float(np.clip(confidence, 0.0, 1.0))
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    async def propose_action(self, **kwargs: Any) -> Dict[str, Any]:
        try:
            proposal: Dict[str, Any] = {
                "action_type": "correlation_risk_management",
                "timestamp": time.time(),
                "correlation_risk_score": float(self.correlation_risk_score),
                "diversification_score": float(self.diversification_score),
                "severity_level": str(self.severity_level),
                "recommendations": [],
                "warnings": [],
                "adjustments": {},
            }

            if self.correlation_risk_score > 0.7:
                proposal["recommendations"].append(
                    {
                        "type": "reduce_correlation",
                        "reason": "High correlation risk detected",
                        "suggested_action": "Reduce exposure in highly correlated pairs",
                        "priority": "high",
                    }
                )

            if self.diversification_score < self.min_diversification:
                proposal["recommendations"].append(
                    {
                        "type": "improve_diversification",
                        "reason": f"Diversification {self.diversification_score:.2f} below {self.min_diversification:.2f}",
                        "suggested_action": "Add positions in uncorrelated instruments",
                        "priority": "medium",
                    }
                )


            if self.correlation_matrix:
                high_pairs = sorted(
                    self.correlation_matrix.items(),
                    key=lambda kv: abs(kv[1]),
                    reverse=True,
                )
                high_pairs = [
                    (pair, c)
                    for (pair, c) in high_pairs
                    if abs(c) >= self.warning_correlation
                ][:3]
                if high_pairs:
                    proposal["adjustments"]["pairs"] = [
                        {
                            "instruments": [str(pair[0]), str(pair[1])],
                            "correlation": float(c),
                            "action": "review_exposure",
                        }
                        for pair, c in high_pairs
                    ]

            return proposal
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "action_proposal")
            self.logger.error(f"Action proposal failed: {error_context}")
            return {
                "action_type": "correlation_risk_management",
                "timestamp": time.time(),
                "error": str(e),
                "recommendations": [],
                "warnings": [],
                "adjustments": {},
            }


    async def process(self, **kwargs: Any) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if not self.enabled:
                payload = self._handle_disabled_fallback()
                self._write_bus_from_payload(payload, payload["_thesis"])
                return payload

            if self.circuit_breaker["state"] == "OPEN":
                thesis = "Circuit breaker OPEN; correlation risk safe fallback emitted."
                payload = self._fallback_payload(thesis)
                self._write_bus_from_payload(payload, thesis)
                return payload

            self.step_count += 1


            positions = self.smart_bus.get("positions", "CorrelatedRiskController") or []
            prices = self.smart_bus.get("prices", "CorrelatedRiskController") or {}
            market_context = self.smart_bus.get("market_context", "CorrelatedRiskController") or {}


            self._update_price_histories(prices)


            correlation_results = await self._analyze_correlations_comprehensive(positions, market_context)


            risk_metrics = self._calculate_correlation_risk_metrics(correlation_results)


            thesis = await self._generate_correlation_thesis(correlation_results, market_context)


            payload = self._format_provides_output(correlation_results, risk_metrics, thesis)


            self._write_bus_from_payload(payload, thesis)


            processing_time_sec = float(time.time() - start_time)
            self._record_success(processing_time_sec)
            try:
                self.performance_tracker.record_metric(
                    "CorrelatedRiskController",
                    "correlation_analysis",
                    processing_time_sec * 1000.0,
                    True,
                )
            except Exception:
                pass

            return payload

        except Exception as e:
            processing_time_sec = float(time.time() - start_time)
            payload = self._handle_error(e, processing_time_sec)
            try:
                self._write_bus_from_payload(payload, payload.get("_thesis", "Correlation error"))
            except Exception:
                pass
            return payload


    def _write_bus_from_payload(self, payload: Dict[str, Any], thesis: str) -> None:
        try:
            self.smart_bus.set(
                "correlation_risk",
                payload["correlation_risk"],
                module="CorrelatedRiskController",
                thesis=thesis,
            )
            self.smart_bus.set(
                "diversification_score",
                payload["diversification_score"],
                module="CorrelatedRiskController",
                thesis="Diversification score update",
            )
            self.smart_bus.set(
                "correlation_clusters",
                payload["correlation_clusters"],
                module="CorrelatedRiskController",
                thesis="Correlation clusters update",
            )

            self.smart_bus.set(
                "correlation_matrix",
                payload["correlation_matrix"],
                module="CorrelatedRiskController",
                thesis="Correlation matrix update",
            )
        except Exception as e:
            err = self.error_pinpointer.analyze_error(e, "bus_write")
            self.logger.error(f"SmartInfoBus update failed: {err}")


    def _serialize_corr_matrix(self, corr: Dict[Tuple[str, str], float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for (a, b), v in corr.items():
            try:
                key = f"({a!r},{b!r})"
            except Exception:
                key = f"({a},{b})"
            out[key] = float(v)
        return out

    def _format_provides_output(
        self,
        correlation_results: Dict[str, Any],
        risk_metrics: Dict[str, Any],
        thesis: str,
    ) -> Dict[str, Any]:

        corr_payload = {
            "correlation_risk_score": float(self.correlation_risk_score),
            "severity_level": str(self.severity_level),
            "risk_metrics": {
                "violation_risk": float(risk_metrics.get("violation_risk", 0.0)),
                "diversification_risk": float(risk_metrics.get("diversification_risk", 0.0)),
                "cluster_risk_score": float(self.cluster_risk_score),
            },
            "summary": {
                "pairs_analyzed": len(correlation_results.get("correlation_matrix", {})),
                "instruments_analyzed": len(correlation_results.get("instruments_analyzed", [])),
                "processing_time_ms": float(
                    correlation_results.get("processing_time_ms", 0.0)
                ),
            },
            "timestamp": datetime.datetime.now().isoformat(),
            "thesis": thesis,
        }

        clusters = correlation_results.get("cluster_analysis", {}).get("clusters", {})


        tuple_corr: Dict[Tuple[str, str], float] = {}
        raw_matrix = correlation_results.get("correlation_matrix", {})
        if isinstance(raw_matrix, dict):
            for k, v in raw_matrix.items():
                if isinstance(k, tuple) and len(k) == 2:
                    a, b = k
                    tuple_corr[(str(a), str(b))] = float(v)
                elif isinstance(k, str) and k.startswith("(") and k.endswith(")"):

                    try:
                        parsed = ast.literal_eval(k)
                        if isinstance(parsed, tuple) and len(parsed) == 2:
                            a, b = parsed
                            tuple_corr[(str(a), str(b))] = float(v)
                    except Exception:
                        continue

        corr_matrix_serialized = self._serialize_corr_matrix(tuple_corr)

        return {
            "correlation_risk": corr_payload,
            "diversification_score": float(self.diversification_score),
            "correlation_clusters": {
                int(k): list(v) for k, v in clusters.items()
            } if clusters else {},
            "correlation_matrix": corr_matrix_serialized,
            "_thesis": thesis,
            "success": True,
        }


    def _update_price_histories(self, prices: Dict[str, float]) -> None:
        try:
            for instrument, price in prices.items():
                try:
                    p = float(price)
                    if p <= 0.0 or not np.isfinite(p):
                        continue
                except Exception:
                    continue

                ph = self.price_history[instrument]
                prev_price = ph[-1] if ph else None
                ph.append(p)


                if prev_price is not None and prev_price > 0.0:
                    ret = float(np.log(p / prev_price))
                    rh = self.return_history[instrument]
                    rh.append(ret)
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "price_history_update")
            self.logger.warning(f"Price history update failed: {error_context}")

    async def _analyze_correlations_comprehensive(
        self,
        positions: List[Dict[str, Any]],
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        start = datetime.datetime.now()
        try:

            raw_positions: Any = positions
            if isinstance(raw_positions, dict):
                pos_iter = list(raw_positions.values())
            elif isinstance(raw_positions, (list, tuple, set)):
                pos_iter = list(raw_positions)
            else:
                pos_iter = [raw_positions]

            norm_positions: List[Dict[str, Any]] = []
            instruments_set: set[str] = set()

            for pos in pos_iter:
                if not pos:
                    continue

                if isinstance(pos, str):
                    sym = pos
                    norm_positions.append({"symbol": sym})
                elif isinstance(pos, dict):
                    sym = str(pos.get("symbol") or pos.get("instrument") or "")
                    norm_positions.append(pos)
                else:
                    sym = str(getattr(pos, "symbol", "") or getattr(pos, "instrument", ""))
                    norm_positions.append({"symbol": sym})

                if sym:
                    instruments_set.add(sym)


            instruments = [
                i for i in instruments_set if i and i in self.return_history
            ]

            if len(instruments) < 2:
                return self._insufficient_data("Need at least 2 instruments with history")


            corr_matrix = self._calculate_correlation_matrix(instruments, market_context)


            cluster_analysis = self._perform_greedy_clustering(corr_matrix, instruments)


            diversification_metrics = self._calculate_diversification_metrics(corr_matrix, norm_positions)


            violation_analysis = self._analyze_correlation_violations(corr_matrix)


            regime_analysis = self._analyze_regime_correlations(corr_matrix, market_context)

            proc_ms = (datetime.datetime.now() - start).total_seconds() * 1000.0
            result = {
                "correlation_matrix": {tuple(k): float(v) for k, v in corr_matrix.items()},
                "cluster_analysis": cluster_analysis,
                "diversification_metrics": diversification_metrics,
                "violation_analysis": violation_analysis,
                "regime_analysis": regime_analysis,
                "instruments_analyzed": instruments,
                "processing_time_ms": float(proc_ms),
                "market_context": dict(market_context),
            }


            self.correlation_matrix = corr_matrix.copy()
            self.correlation_history.append(self.correlation_matrix.copy())

            return result
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "correlation_analysis")
            self.logger.error(f"Correlation analysis failed: {error_context}")
            return self._analysis_error(str(error_context))

    def _calculate_correlation_matrix(
        self,
        instruments: List[str],
        market_context: Dict[str, Any],
    ) -> Dict[Tuple[str, str], float]:
        corr: Dict[Tuple[str, str], float] = {}
        regime = str(market_context.get("regime", "unknown"))
        vol_regime = str(market_context.get("volatility_level", "medium"))

        for i, a in enumerate(instruments):
            ra = [float(x) for x in self.return_history.get(a, []) if np.isfinite(x)]
            for b in instruments[i + 1 :]:
                rb = [float(x) for x in self.return_history.get(b, []) if np.isfinite(x)]
                if len(ra) >= self.min_samples_for_corr and len(rb) >= self.min_samples_for_corr:
                    m = min(len(ra), len(rb))
                    if m < 2:
                        continue
                    r1 = np.asarray(ra[-m:], dtype=np.float64)
                    r2 = np.asarray(rb[-m:], dtype=np.float64)
                    try:
                        c = float(np.corrcoef(r1, r2)[0, 1])
                        if not np.isfinite(c):
                            c = 0.0
                    except Exception:
                        c = 0.0
                    adj = self._apply_regime_adjustments(c, regime, vol_regime)
                    value = float(np.clip(adj, -0.95, 0.95))
                    corr[(a, b)] = value
                    self.regime_correlations[regime][(a, b)].append(value)
                else:

                    corr[(a, b)] = self._heuristic_corr(a, b)
        return corr

    def _apply_regime_adjustments(self, correlation: float, regime: str, vol_regime: str) -> float:
        try:
            c = correlation

            if vol_regime == "high":
                c *= 1.15
            elif vol_regime == "extreme":
                c *= 1.30
            elif vol_regime == "low":
                c *= 0.85


            if regime == "crisis":
                c *= 1.20
            elif regime == "trending":
                c *= 0.95

            return float(c)
        except Exception:
            return correlation

    def _heuristic_corr(self, a: str, b: str) -> float:
        try:
            au = a.upper()
            bu = b.upper()
            if "XAU" in au and "XAU" in bu:
                return 0.8
            if "XAU" in au or "XAU" in bu:
                return -0.2
            if "JPY" in au and "JPY" in bu:
                return 0.6

            if "USD" in au and "USD" in bu:
                return 0.4

            if any(k in au for k in ("EUR", "GBP")) and any(k in bu for k in ("EUR", "GBP")):
                return 0.5
            return 0.1
        except Exception:
            return 0.0

    def _perform_greedy_clustering(
        self,
        corr: Dict[Tuple[str, str], float],
        instruments: List[str],
    ) -> Dict[str, Any]:
        parent: Dict[str, str] = {i: i for i in instruments}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx


        t = float(self.cluster_link_threshold)
        for (a, b), c in corr.items():
            if abs(c) >= t and a in parent and b in parent:
                union(a, b)

        clusters_map: Dict[str, List[str]] = defaultdict(list)
        for inst in instruments:
            clusters_map[find(inst)].append(inst)


        cluster_risks: Dict[int, float] = {}
        id_map: Dict[str, int] = {root: idx + 1 for idx, root in enumerate(clusters_map.keys())}
        for root, members in clusters_map.items():
            values: List[float] = []
            for i, x in enumerate(members):
                for y in members[i + 1 :]:
                    values.append(abs(corr.get((x, y), corr.get((y, x), 0.0))))
            avg_corr = float(np.mean(values)) if values else 0.0
            size_factor = len(members) / 10.0
            cluster_risks[id_map[root]] = float(avg_corr * (1.0 + size_factor))

        mapped_clusters: Dict[int, List[str]] = {id_map[r]: m for r, m in clusters_map.items()}
        self.correlation_clusters = dict(mapped_clusters)

        return {
            "clusters": dict(mapped_clusters),
            "cluster_count": len(mapped_clusters),
            "cluster_risks": cluster_risks,
            "max_cluster_risk": float(max(cluster_risks.values())) if cluster_risks else 0.0,
            "link_threshold": t,
        }

    def _calculate_diversification_metrics(
        self,
        corr: Dict[Tuple[str, str], float],
        positions: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        try:
            if not corr:
                self.diversification_score = 1.0
                return {
                    "diversification_score": 1.0,
                    "effective_positions": len(positions),
                    "avg_correlation": 0.0,
                    "position_count": len(positions),
                }


            avg_abs_corr = float(np.mean([abs(v) for v in corr.values()])) if corr else 0.0
            corr_diversification = float(np.clip(1.0 - avg_abs_corr, 0.0, 1.0))


            exposures: List[float] = []
            total_exposure = 0.0
            for p in positions:
                try:
                    size = abs(float(p.get("size", p.get("volume", 0.0))))
                    price = float(p.get("current_price", p.get("price", 1.0)))
                    e = size * price
                    exposures.append(e)
                    total_exposure += e
                except Exception:
                    continue

            if total_exposure > 0.0 and exposures:
                weights = [e / total_exposure for e in exposures]
                herfindahl = float(sum(w * w for w in weights))
                pos_diversification = float(np.clip(1.0 - herfindahl, 0.0, 1.0))
                effective_positions = float(1.0 / herfindahl) if herfindahl > 0.0 else float(len(positions))
            else:
                pos_diversification = 1.0
                effective_positions = float(len(positions))


            self.diversification_score = float(
                np.clip((corr_diversification + pos_diversification) / 2.0, 0.0, 1.0)
            )

            return {
                "diversification_score": float(self.diversification_score),
                "avg_correlation": avg_abs_corr,
                "effective_positions": float(effective_positions),
                "position_count": len(positions),
                "concentration_index": float(np.clip(1.0 - pos_diversification, 0.0, 1.0)),
                "correlation_pairs": len(corr),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "diversification_metrics")
            self.logger.error(f"Diversification metrics calculation failed: {error_context}")
            self.diversification_score = 0.5
            return {"diversification_score": 0.5, "effective_positions": 1}

    def _analyze_correlation_violations(
        self,
        corr: Dict[Tuple[str, str], float],
    ) -> Dict[str, Any]:
        try:
            critical: List[Dict[str, Any]] = []
            warning: List[Dict[str, Any]] = []
            for (a, b), c in corr.items():
                ac = abs(float(c))
                if ac >= self.max_correlation:
                    critical.append({"instruments": (a, b), "correlation": float(c)})
                elif ac >= self.warning_correlation:
                    warning.append({"instruments": (a, b), "correlation": float(c)})

            self.correlation_violations += len(critical)
            if self.diversification_score < self.min_diversification:
                self.diversification_violations += 1

            max_corr = float(
                max([abs(v) for v in corr.values()])
            ) if corr else 0.0

            return {
                "violations": {
                    "critical": critical,
                    "warning": warning,
                    "info": [],
                },
                "total_violations": int(len(critical) + len(warning)),
                "critical_pairs": len(critical),
                "warning_pairs": len(warning),
                "max_correlation": max_corr,
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "violation_analysis")
            self.logger.error(f"Violation analysis failed: {error_context}")
            return {
                "violations": {"critical": [], "warning": [], "info": []},
                "total_violations": 0,
            }

    def _analyze_regime_correlations(
        self,
        corr: Dict[Tuple[str, str], float],
        market_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            regime = str(market_context.get("regime", "unknown"))
            vol = str(market_context.get("volatility_level", "medium"))

            stats: Dict[str, Dict[str, float]] = {}
            for regime_name, regime_data in self.regime_correlations.items():
                vals: List[float] = []
                for pair_vals in regime_data.values():
                    vals.extend(pair_vals)
                if vals:
                    abs_vals = [abs(float(x)) for x in vals if np.isfinite(x)]
                    if abs_vals:
                        stats[regime_name] = {
                            "avg_correlation": float(np.mean(abs_vals)),
                            "max_correlation": float(np.max(abs_vals)),
                            "correlation_volatility": float(np.std(abs_vals)),
                            "sample_count": len(abs_vals),
                        }

            impact = self._assess_regime_shift_impact(stats, regime) if stats else "insufficient_data"
            return {
                "current_regime": regime,
                "current_volatility": vol,
                "regime_stats": stats,
                "regime_shift_impact": impact,
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "regime_analysis")
            self.logger.warning(f"Regime analysis failed: {error_context}")
            return {"current_regime": "unknown", "regime_stats": {}}

    def _assess_regime_shift_impact(
        self,
        regime_stats: Dict[str, Dict[str, float]],
        current: str,
    ) -> str:
        try:
            if current not in regime_stats or len(regime_stats) < 2:
                return "insufficient_data"
            cur = regime_stats[current]["avg_correlation"]
            others = [v["avg_correlation"] for k, v in regime_stats.items() if k != current]
            if not others:
                return "no_comparison_data"
            if cur > max(others) * 1.2:
                return "high_correlation_regime"
            if cur < min(others) * 0.8:
                return "low_correlation_regime"
            return "normal_correlation_regime"
        except Exception:
            return "assessment_error"

    def _calculate_correlation_risk_metrics(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            viol = results.get("violation_analysis", {})
            div = results.get("diversification_metrics", {})
            clusters = results.get("cluster_analysis", {})

            critical = len(viol.get("violations", {}).get("critical", []))
            warnings = len(viol.get("violations", {}).get("warning", []))
            instruments = max(1, len(results.get("instruments_analyzed", [])))

            violation_risk = float(
                (critical * 1.0 + warnings * 0.6) / instruments
            )
            diversification_risk = float(
                max(
                    0.0,
                    (
                        self.min_diversification
                        - float(div.get("diversification_score", 1.0))
                    )
                    / max(self.min_diversification, 1e-6),
                )
            )
            self.cluster_risk_score = float(
                clusters.get("max_cluster_risk", 0.0)
            )

            self.correlation_risk_score = float(
                np.clip(
                    violation_risk * 0.4
                    + diversification_risk * 0.4
                    + self.cluster_risk_score * 0.2,
                    0.0,
                    1.0,
                )
            )

            if self.correlation_risk_score > 0.7 or critical > 0:
                self.severity_level = "critical"
            elif self.correlation_risk_score > 0.4 or warnings > 0:
                self.severity_level = "warning"
            elif self.correlation_risk_score > 0.1:
                self.severity_level = "elevated"
            else:
                self.severity_level = "normal"

            return {
                "correlation_risk_score": float(self.correlation_risk_score),
                "diversification_score": float(self.diversification_score),
                "cluster_risk_score": float(self.cluster_risk_score),
                "severity_level": str(self.severity_level),
                "violation_risk": float(violation_risk),
                "diversification_risk": float(diversification_risk),
            }
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "risk_metrics")
            self.logger.error(f"Risk metrics calculation failed: {error_context}")
            self.severity_level = "error"
            return {"correlation_risk_score": 0.5, "severity_level": "error"}

    async def _generate_correlation_thesis(
        self,
        results: Dict[str, Any],
        market_context: Dict[str, Any],
    ) -> str:
        try:
            instruments = results.get("instruments_analyzed", [])
            corr_pairs = results.get("correlation_matrix", {})
            violations = results.get("violation_analysis", {})
            diversification = results.get("diversification_metrics", {})
            clusters = results.get("cluster_analysis", {})
            regime = results.get("regime_analysis", {})

            parts: List[str] = []
            parts.append(f"Analyzed {len(instruments)} instruments and {len(corr_pairs)} correlation pairs")

            div_score = float(diversification.get("diversification_score", 1.0))
            if div_score >= 0.7:
                parts.append(f"EXCELLENT diversification ({div_score:.1%})")
            elif div_score >= 0.5:
                parts.append(f"ADEQUATE diversification ({div_score:.1%})")
            else:
                parts.append(f"POOR diversification ({div_score:.1%}) - concentration risk elevated")

            crit = len(violations.get("violations", {}).get("critical", []))
            warn = len(violations.get("violations", {}).get("warning", []))
            if crit > 0:
                parts.append(f"CRITICAL: {crit} correlations exceed {self.max_correlation:.0%}")
            elif warn > 0:
                parts.append(f"WARNING: {warn} correlations approaching limits")
            else:
                parts.append("All correlations within acceptable ranges")

            cluster_count = int(clusters.get("cluster_count", 0))
            max_cluster_risk = float(clusters.get("max_cluster_risk", 0.0))
            if max_cluster_risk > 0.7:
                parts.append(f"HIGH cluster concentration risk across {cluster_count} clusters")
            elif cluster_count > 1:
                parts.append(f"{cluster_count} clusters detected with manageable concentration")

            reg = regime.get("current_regime", "unknown")
            impact = regime.get("regime_shift_impact", "unknown")
            if reg != "unknown":
                if impact == "high_correlation_regime":
                    parts.append(f"Current {reg} regime shows elevated correlations")
                elif impact == "low_correlation_regime":
                    parts.append(f"Current {reg} regime favors diversification")

            parts.append(
                f"Overall correlation risk: {self.severity_level.upper()} (score {self.correlation_risk_score:.2f})"
            )
            return " | ".join(parts)
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "thesis_generation")
            return f"Thesis generation failed: {error_context}"


    def _fallback_payload(self, thesis: str) -> Dict[str, Any]:
        return {
            "correlation_risk": {
                "correlation_risk_score": float(self.correlation_risk_score),
                "severity_level": str(self.severity_level),
                "risk_metrics": {
                    "violation_risk": 0.0,
                    "diversification_risk": 0.0,
                    "cluster_risk_score": float(self.cluster_risk_score),
                },
                "summary": {
                    "pairs_analyzed": 0,
                    "instruments_analyzed": 0,
                    "processing_time_ms": 0.0,
                },
                "timestamp": datetime.datetime.now().isoformat(),
                "thesis": thesis,
            },
            "diversification_score": float(self.diversification_score),
            "correlation_clusters": {},
            "correlation_matrix": {},
            "_thesis": thesis,
            "success": True,
        }

    def _handle_disabled_fallback(self) -> Dict[str, Any]:
        thesis = "Correlated Risk Controller is disabled"
        self.severity_level = "disabled"
        self.correlation_risk_score = 0.0
        self.diversification_score = 1.0
        return self._fallback_payload(thesis)

    def _handle_error(self, error: Exception, processing_time_sec: float) -> Dict[str, Any]:

        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        if self.circuit_breaker["failures"] >= int(self._cfg.circuit_breaker_threshold):
            self.circuit_breaker["state"] = "OPEN"
            self._health_status = "warning"

        explanation = self.english_explainer.explain_error(
            "CorrelatedRiskController", str(error), "correlation analysis"
        )
        self.logger.error(
            format_operator_message(
                message="Correlation module error",
                icon="[CRASH]",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time_sec * 1000.0,
                circuit_breaker_state=self.circuit_breaker["state"],
            )
        )
        self._record_failure(error)
        self.severity_level = "error"
        self.correlation_risk_score = max(0.3, float(self.correlation_risk_score))
        return self._fallback_payload(thesis=f"Correlation error fallback: {error!s}")


    def _record_success(self, processing_time_sec: float) -> None:
        try:
            self._processing_times.append(float(processing_time_sec))
            if self.circuit_breaker["state"] == "CLOSED":
                self.circuit_breaker["failures"] = max(0, self.circuit_breaker["failures"] - 1)
        except Exception:
            pass

    def _record_failure(self, error: Exception) -> None:
        try:

            _ = str(error)
        except Exception:
            pass


    def _insufficient_data(self, reason: str) -> Dict[str, Any]:
        return {
            "correlation_matrix": {},
            "cluster_analysis": {
                "clusters": {},
                "cluster_count": 0,
                "reason": reason,
            },
            "diversification_metrics": {
                "diversification_score": 1.0,
                "effective_positions": 0,
            },
            "violation_analysis": {
                "violations": {"critical": [], "warning": [], "info": []},
                "total_violations": 0,
            },
            "regime_analysis": {"current_regime": "unknown"},
            "instruments_analyzed": [],
            "processing_time_ms": 0.0,
            "status": "insufficient_data",
        }

    def _analysis_error(self, error_context: str) -> Dict[str, Any]:
        return {
            "correlation_matrix": {},
            "cluster_analysis": {"error": error_context},
            "diversification_metrics": {"diversification_score": 0.5},
            "violation_analysis": {
                "violations": {"critical": [], "warning": [], "info": []},
                "total_violations": 0,
            },
            "regime_analysis": {"current_regime": "error"},
            "instruments_analyzed": [],
            "processing_time_ms": 0.0,
            "status": "analysis_error",
            "error": error_context,
        }


    def get_state(self) -> Dict[str, Any]:
        return {
            "correlation_matrix": self._serialize_corr_matrix(self.correlation_matrix),
            "correlation_risk_score": float(self.correlation_risk_score),
            "diversification_score": float(self.diversification_score),
            "severity_level": str(self.severity_level),
            "correlation_clusters": {
                int(k): list(v) for k, v in self.correlation_clusters.items()
            },
            "step_count": int(self.step_count),
            "correlation_violations": int(self.correlation_violations),
            "diversification_violations": int(self.diversification_violations),
            "config": dict(self.config),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        self.correlation_risk_score = float(state.get("correlation_risk_score", 0.0))
        self.diversification_score = float(state.get("diversification_score", 1.0))
        self.severity_level = str(state.get("severity_level", "normal"))
        self.step_count = int(state.get("step_count", 0))
        self.correlation_violations = int(state.get("correlation_violations", 0))
        self.diversification_violations = int(state.get("diversification_violations", 0))
        self.config.update(dict(state.get("config", {})))


        try:
            cm = state.get("correlation_matrix", {})
            if isinstance(cm, dict):
                rebuilt: Dict[Tuple[str, str], float] = {}
                for k, v in cm.items():
                    if isinstance(k, str) and k.startswith("(") and k.endswith(")"):
                        try:
                            parsed = ast.literal_eval(k)
                        except Exception:
                            continue
                        if isinstance(parsed, tuple) and len(parsed) == 2:
                            a, b = parsed
                            rebuilt[(str(a), str(b))] = float(v)
                if rebuilt:
                    self.correlation_matrix = rebuilt
        except Exception:
            pass

        try:
            cc = state.get("correlation_clusters", {})
            if isinstance(cc, dict):
                self.correlation_clusters = {
                    int(k): list(v) for k, v in cc.items()
                }
        except Exception:
            pass

    def get_health_metrics(self) -> Dict[str, Any]:
        return {
            "correlation_risk_score": float(self.correlation_risk_score),
            "diversification_score": float(self.diversification_score),
            "severity_level": str(self.severity_level),
            "instruments_tracked": len(self.price_history),
            "correlation_violations": int(self.correlation_violations),
            "diversification_violations": int(self.diversification_violations),
            "cluster_count": len(self.correlation_clusters),
            "enabled": bool(self.enabled),
        }
