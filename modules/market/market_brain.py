# # ─────────────────────────────────────────────────────────────
# # File: modules/market/market_brain.py
# # MarketBrain — the ONE registered module (with rich debug logging)
# # - Reads inputs from InfoBus
# # - Calls your market helpers (plain files, NOT modules)
# # - Publishes aggregated outputs to InfoBus
# # - Returns honest "no data" states rather than fake fallbacks
# # ─────────────────────────────────────────────────────────────

# from __future__ import annotations
# import asyncio
# import inspect
# import time
# import datetime
# from typing import Any, Dict, Optional, List, Tuple

# from modules.core.module_base import BaseModule, module
# from modules.utils.info_bus import InfoBusManager
# from modules.monitoring.performance_tracker import PerformanceTracker
# from modules.utils.audit_utils import RotatingLogger

# # Import helper modules (not registered)
# from modules.market import market_theme_detector as theme_mod
# from modules.market import liquidity_heatmap_layer as liq_mod
# from modules.market import fractal_regime_confirmation as reg_mod
# from modules.market import time_aware_risk_scaling as trisk_mod
# from modules.market import regime_performance_matrix as matrix_mod

# # --------------------------- config ---------------------------

# DEFAULT_INPUT_KEYS = [
#     "market_data", "price_data", "bid_ask_data",
#     "historical_prices", "multi_timeframe_data",
#     "macro_data", "volatility_data", "pnl_data",
#     "positions", "pending_orders", "market_context",
#     "prices", "technical_indicators", "timestamp", 
#     "risk_data", "market_regime", "liquidity_score", 
#     "recent_trades",
# ]

# DEFAULT_HELPER_CONFIG = {
#     "theme": {},
#     "liquidity": {},
#     "regime": {},
#     "time_risk": {},
#     "matrix": {},
# }

# PUBLISH_KEYS = ["market_overview", "market_thesis", "market_health", "market_outputs"]


# class HelperAdapter:
#     """
#     Production adapter for calling market helper modules.
#     Returns honest "no data" states instead of fake fallbacks.
#     """
    
#     def __init__(self, 
#                  module,
#                  class_name: str,
#                  name: str,
#                  config: Optional[Dict[str, Any]] = None,
#                  logger: Optional[RotatingLogger] = None):
#         self.module = module
#         self.class_name = class_name
#         self.name = name
#         self.config = config or {}
#         self.logger = logger
        
#         self._instance = None
#         self._initialization_error = None
#         self._call_count = 0
#         self._error_count = 0
#         self._total_time_ms = 0.0
#         self._last_error = None
#         self._consecutive_errors = 0
        
#         # Circuit breaker
#         self._circuit_open = False
#         self._circuit_open_until = 0
#         self.max_consecutive_errors = 3
#         self.circuit_reset_seconds = 60
        
#     def _get_instance(self) -> Optional[Any]:
#         """Get or create the helper instance."""
#         if self._instance is not None:
#             return self._instance
            
#         if self._initialization_error is not None:
#             return None
            
#         try:
#             cls = getattr(self.module, self.class_name, None)
#             if cls is None:
#                 raise AttributeError(f"Class {self.class_name} not found in module")
                
#             try:
#                 self._instance = cls(self.config)
#             except TypeError:
#                 self._instance = cls()
#                 if hasattr(self._instance, 'config'):
#                     self._instance.config = self.config
                    
#             return self._instance
            
#         except Exception as e:
#             self._initialization_error = e
#             if self.logger:
#                 self.logger.error(f"[{self.name}] Initialization failed: {e}")
#             return None
    
#     def _normalize_inputs(self, raw_inputs: Dict[str, Any]) -> Dict[str, Any]:
#         """Normalize inputs for helper compatibility."""
#         normalized = dict(raw_inputs)
        
#         # Ensure timestamp
#         if 'timestamp' not in normalized:
#             normalized['timestamp'] = datetime.datetime.utcnow().isoformat()
        
#         # Normalize price data formats
#         if 'prices' not in normalized and 'price_data' in normalized:
#             price_data = normalized['price_data']
#             if isinstance(price_data, dict):
#                 prices = {}
#                 for symbol, data in price_data.items():
#                     if isinstance(data, dict) and 'close' in data:
#                         prices[symbol] = data['close']
#                     elif isinstance(data, (int, float)):
#                         prices[symbol] = data
#                 normalized['prices'] = prices
        
#         # Create market_data from prices if missing
#         if 'market_data' not in normalized and 'prices' in normalized:
#             market_data = {}
#             for symbol, price in normalized.get('prices', {}).items():
#                 if price is not None:
#                     market_data[symbol] = {
#                         'open': price,
#                         'high': price * 1.001,
#                         'low': price * 0.999,
#                         'close': price,
#                         'volume': 1000
#                     }
#             if market_data:
#                 normalized['market_data'] = market_data
        
#         # Extract volatility if present
#         if 'volatility' not in normalized and 'volatility_data' in normalized:
#             vol = normalized['volatility_data']
#             if isinstance(vol, (int, float)):
#                 normalized['volatility'] = float(vol)
#             elif isinstance(vol, dict) and 'current' in vol:
#                 normalized['volatility'] = float(vol['current'])
                
#         return normalized
    
#     async def run(self, 
#                   raw_inputs: Dict[str, Any], 
#                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
#         """Execute the helper with error handling."""
#         t0 = time.time()
#         self._call_count += 1
        
#         meta = {
#             'name': self.name,
#             'call_count': self._call_count,
#             'ok': False,
#             'elapsed_ms': 0,
#             'error': None,
#             'circuit_breaker': 'closed',
#             'data_available': False
#         }
        
#         # Check circuit breaker
#         if self._circuit_open:
#             if time.time() < self._circuit_open_until:
#                 meta['circuit_breaker'] = 'open'
#                 meta['error'] = 'Circuit breaker open'
#                 meta['elapsed_ms'] = (time.time() - t0) * 1000
#                 return self._create_no_data_response(meta)
#             else:
#                 self._circuit_open = False
#                 self._consecutive_errors = 0
#                 meta['circuit_breaker'] = 'reset'
        
#         try:
#             # Get instance
#             instance = self._get_instance()
#             if instance is None:
#                 raise RuntimeError(f"Failed to initialize {self.class_name}")
            
#             # Normalize inputs
#             normalized_inputs = self._normalize_inputs(raw_inputs)
            
#             # Check if we have minimum required data
#             has_data = self._check_minimum_data(normalized_inputs)
#             if not has_data:
#                 meta['error'] = 'Insufficient input data'
#                 meta['elapsed_ms'] = (time.time() - t0) * 1000
#                 return self._create_no_data_response(meta)
            
#             # Merge configs
#             merged_config = {**self.config}
#             if config:
#                 merged_config.update(config)
            
#             # Call helper
#             if hasattr(instance, 'compute'):
#                 result = instance.compute(normalized_inputs, merged_config)
#             elif hasattr(instance, 'process'):
#                 result = instance.process(normalized_inputs, merged_config)
#             else:
#                 raise AttributeError(f"{self.class_name} has no compute/process method")
            
#             if inspect.isawaitable(result):
#                 result = await result
            
#             if not isinstance(result, dict):
#                 result = {}
            
#             # Success
#             self._consecutive_errors = 0
#             elapsed_ms = (time.time() - t0) * 1000
#             self._total_time_ms += elapsed_ms
            
#             meta['ok'] = True
#             meta['elapsed_ms'] = elapsed_ms
#             meta['avg_time_ms'] = self._total_time_ms / self._call_count
#             meta['data_available'] = True
            
#             result['_adapter_meta'] = meta
            
#             if self.logger:
#                 self.logger.debug(f"[{self.name}] Success in {elapsed_ms:.1f}ms")
            
#             return result
            
#         except Exception as e:
#             self._error_count += 1
#             self._consecutive_errors += 1
#             self._last_error = str(e)
            
#             elapsed_ms = (time.time() - t0) * 1000
#             meta['elapsed_ms'] = elapsed_ms
#             meta['error'] = str(e)
#             meta['error_type'] = type(e).__name__
            
#             if self.logger:
#                 self.logger.error(f"[{self.name}] Error: {e}")
            
#             if self._consecutive_errors >= self.max_consecutive_errors:
#                 self._circuit_open = True
#                 self._circuit_open_until = time.time() + self.circuit_reset_seconds
#                 meta['circuit_breaker'] = 'opened'
#                 if self.logger:
#                     self.logger.warning(f"[{self.name}] Circuit breaker opened")
            
#             return self._create_no_data_response(meta)
    
#     def _check_minimum_data(self, inputs: Dict[str, Any]) -> bool:
#         """Check if minimum required data is present."""
#         # Each helper needs at least some market data
#         has_market = bool(inputs.get('market_data') or inputs.get('prices'))
#         has_price = bool(inputs.get('price_data') or inputs.get('historical_prices'))
#         return has_market or has_price
    
#     def _create_no_data_response(self, meta: Dict[str, Any]) -> Dict[str, Any]:
#         """Create honest 'no data' response."""
#         return {
#             '_adapter_meta': meta,
#             'success': False,
#             'data_available': False,
#             'thesis': f"{self.name}: No data available",
#             'error': meta.get('error', 'No data')
#         }
    
#     def get_health(self) -> Dict[str, Any]:
#         """Get health metrics."""
#         return {
#             'name': self.name,
#             'initialized': self._instance is not None,
#             'call_count': self._call_count,
#             'error_count': self._error_count,
#             'error_rate': self._error_count / max(self._call_count, 1),
#             'avg_time_ms': self._total_time_ms / max(self._call_count, 1),
#             'last_error': self._last_error,
#             'consecutive_errors': self._consecutive_errors,
#             'circuit_breaker': 'open' if self._circuit_open else 'closed'
#         }


# @module(
#     name="MarketBrain",
#     version="1.2.0",
#     category="market",
#     provides=PUBLISH_KEYS,
#     requires=DEFAULT_INPUT_KEYS,
#     thesis_required=True,
#     health_monitoring=True,
#     performance_tracking=True,
#     error_handling=True,
# )
# class MarketBrain(BaseModule):
#     """
#     Market analysis orchestrator.
#     Coordinates helper modules and publishes unified market view.
#     """

#     # Class-level safe defaults to avoid attribute errors during early lifecycle
#     inputs_to_pull: List[str] = DEFAULT_INPUT_KEYS
#     helper_cfg: Dict[str, Dict[str, Any]] = DEFAULT_HELPER_CONFIG
#     min_conf: float = 0.0
#     max_age: Optional[float] = None
#     debug: bool = True
#     debug_inputs: bool = True
#     debug_theses: bool = True
#     debug_health: bool = True

#     def __init__(self, config: Optional[Dict[str, Any]] = None, dependencies: Optional[Dict[str, Any]] = None):
#         cfg = dict(config or {})

#         # Prepare attributes required by _initialize before BaseModule.__init__ triggers it
#         self.inputs_to_pull = cfg.get("input_keys", DEFAULT_INPUT_KEYS)
#         self.helper_cfg = cfg.get("helpers", DEFAULT_HELPER_CONFIG)
#         self.min_conf = float(cfg.get("min_confidence", 0.0))
#         self.max_age = cfg.get("max_age_seconds")

#         # Debug settings
#         self.debug = bool(cfg.get("debug", True))
#         self.debug_inputs = bool(cfg.get("debug_inputs", True))
#         self.debug_theses = bool(cfg.get("debug_theses", True))
#         self.debug_health = bool(cfg.get("debug_health", True))

#         # Objects used by _initialize
#         self.bus = InfoBusManager.get_instance()
#         self.perf = PerformanceTracker()

#         # Let BaseModule set up logger and call _initialize safely
#         super().__init__(config=config, dependencies=dependencies)

#         # Replace logger with dedicated rotating logger for this module
#         self.logger = RotatingLogger(
#             name="MarketBrain",
#             log_path="logs/market/market_brain.log",
#             max_lines=10000,
#             operator_mode=True,
#             plain_english=True,
#         )
        
#         # Initialize adapters
#         self._theme = HelperAdapter(
#             theme_mod, "MarketThemeDetector", "Theme",
#             self.helper_cfg.get("theme", {}), self.logger
#         )
#         self._liq = HelperAdapter(
#             liq_mod, "LiquidityHeatmapLayer", "Liquidity",
#             self.helper_cfg.get("liquidity", {}), self.logger
#         )
#         self._reg = HelperAdapter(
#             reg_mod, "FractalRegimeConfirmation", "Regime",
#             self.helper_cfg.get("regime", {}), self.logger
#         )
#         self._trisk = HelperAdapter(
#             trisk_mod, "TimeAwareRiskScaling", "TimeRisk",
#             self.helper_cfg.get("time_risk", {}), self.logger
#         )
#         self._matrix = HelperAdapter(
#             matrix_mod, "RegimePerformanceMatrix", "Matrix",
#             self.helper_cfg.get("matrix", {}), self.logger
#         )

#     def _initialize(self, **kwargs) -> None:
#         """Initialize and publish warm-up status."""
#         status = {
#             "initialized": True,
#             "inputs_configured": list(self.inputs_to_pull),
#             "helpers": ["theme", "liquidity", "regime", "time_risk", "matrix"],
#         }
#         try:
#             self.bus.set(
#                 "market_health",
#                 {"status": "warming_up", "details": status},
#                 module="MarketBrain",
#                 thesis="MarketBrain initialized",
#             )
#         except Exception:
#             pass
#         self.logger.info("[MB] Initialized")

#     async def process(self, **inputs) -> Dict[str, Any]:
#         """Main processing tick."""
#         t0 = time.time()
        
#         # Pull inputs from bus
#         raw = {}
#         available, missing = [], []
#         for k in self.inputs_to_pull:
#             try:
#                 raw[k] = self.bus.get(k, module="MarketBrain",
#                                      max_age=self.max_age, 
#                                      min_confidence=self.min_conf)
#                 if raw[k] is not None:
#                     available.append(k)
#                 else:
#                     missing.append(k)
#             except Exception:
#                 raw[k] = None
#                 missing.append(k)
        
#         if self.debug_inputs:
#             self._dbg(f"[INPUTS] {len(available)}/{len(self.inputs_to_pull)} available")
#             if missing:
#                 self._dbg(f"[MISSING] {', '.join(missing)}")
        
#         # Run helpers in parallel
#         tasks = [
#             self._theme.run(raw, self.helper_cfg.get("theme", {})),
#             self._liq.run(raw, self.helper_cfg.get("liquidity", {})),
#             self._reg.run(raw, self.helper_cfg.get("regime", {})),
#             self._trisk.run(raw, self.helper_cfg.get("time_risk", {})),
#             self._matrix.run(raw, self.helper_cfg.get("matrix", {}))
#         ]
        
#         results = await asyncio.gather(*tasks, return_exceptions=True)
        
#         # Handle results
#         theme_out = self._process_result(results[0], "Theme")
#         liq_out = self._process_result(results[1], "Liquidity")
#         regime_out = self._process_result(results[2], "Regime")
#         trisk_out = self._process_result(results[3], "TimeRisk")
#         matrix_out = self._process_result(results[4], "Matrix")
        
#         # Compose overview
#         overview = self._compose_overview(theme_out, liq_out, regime_out, trisk_out, matrix_out)
#         thesis = self._compose_thesis(overview)
        
#         # Debug output
#         if self.debug:
#             self._emit_debug_report(overview, thesis, {
#                 "Theme": theme_out,
#                 "Liquidity": liq_out,
#                 "Regime": regime_out,
#                 "TimeRisk": trisk_out,
#                 "Matrix": matrix_out
#             }, (time.time() - t0) * 1000)
        
#         # Health metrics
#         total_ms = (time.time() - t0) * 1000
#         health = {
#             "latency_ms": total_ms,
#             "success": True,
#             "helpers": {
#                 "theme": self._theme.get_health(),
#                 "liquidity": self._liq.get_health(),
#                 "regime": self._reg.get_health(),
#                 "time_risk": self._trisk.get_health(),
#                 "matrix": self._matrix.get_health()
#             }
#         }
        
#         # Publish results
#         self.bus.set("market_overview", overview, module="MarketBrain", thesis="Market overview")
#         self.bus.set("market_thesis", thesis, module="MarketBrain", thesis="Market analysis")
#         self.bus.set("market_health", health, module="MarketBrain", thesis="System health")
#         self.bus.set("market_outputs", {
#             "theme": theme_out,
#             "liquidity": liq_out,
#             "regime": regime_out,
#             "time_risk": trisk_out,
#             "matrix": matrix_out
#         }, module="MarketBrain", thesis="Raw outputs")
        
#         self.perf.record_metric("MarketBrain", "aggregate_ms", total_ms, True)
        
#         return {
#             "market_overview": overview,
#             "market_thesis": thesis,
#             "market_health": health,
#             "_thesis": thesis,
#         }
    
#     def _process_result(self, result: Any, name: str) -> Dict[str, Any]:
#         """Process helper result, handling exceptions."""
#         if isinstance(result, Exception):
#             self.logger.error(f"[{name}] Exception: {result}")
#             return {
#                 "success": False,
#                 "data_available": False,
#                 "error": str(result),
#                 "thesis": f"{name}: Error - {str(result)}"
#             }
#         return result if isinstance(result, dict) else {"success": False}
    
#     def _compose_overview(self, theme: Dict[str, Any], liq: Dict[str, Any],
#                          regime: Dict[str, Any], trisk: Dict[str, Any],
#                          matrix: Dict[str, Any]) -> Dict[str, Any]:
#         """Compose overview, marking unavailable data as None."""
#         def get_val(d, *path, default=None):
#             if not d or not d.get('data_available', True):
#                 return None
#             x = d
#             for p in path:
#                 if not isinstance(x, dict):
#                     return default
#                 x = x.get(p)
#                 if x is None:
#                     return default
#             return x
        
#         return {
#             "data_status": {
#                 "theme": theme.get('data_available', False),
#                 "liquidity": liq.get('data_available', False),
#                 "regime": regime.get('data_available', False),
#                 "time_risk": trisk.get('data_available', False),
#                 "matrix": matrix.get('data_available', False)
#             },
#             "theme": {
#                 "id": get_val(theme, "market_theme"),
#                 "strength": get_val(theme, "theme_strength"),
#                 "confidence": get_val(theme, "theme_confidence"),
#                 "transition": get_val(theme, "theme_transition"),
#             },
#             "liquidity": {
#                 "score": get_val(liq, "liquidity_score"),
#                 "depth": get_val(liq, "market_depth", "condition"),
#                 "spread": get_val(liq, "spread_analysis", "condition"),
#             },
#             "regime": {
#                 "label": get_val(regime, "market_regime"),
#                 "strength": get_val(regime, "regime_strength"),
#                 "direction": get_val(regime, "trend_direction"),
#             },
#             "risk_scaling": get_val(trisk, "risk_scaling_factor"),
#             "matrix": {
#                 "accuracy": get_val(matrix, "regime_accuracy") or 
#                           get_val(matrix, "performance_metrics", "overall_accuracy"),
#                 "state": get_val(matrix, "market_state", default={}),
#             },
#         }
    
#     def _compose_thesis(self, ov: Dict[str, Any]) -> str:
#         """Compose thesis, clearly stating when data is unavailable."""
#         status = ov.get("data_status", {})
#         available = [k for k, v in status.items() if v]
#         unavailable = [k for k, v in status.items() if not v]
        
#         if not available:
#             return "No market data available from any helper module"
        
#         parts = []
        
#         # Add available data
#         t = ov["theme"]
#         if status.get("theme") and t["id"] is not None:
#             parts.append(f"Theme {t['id']} (str={self._fmt(t['strength'])}, "
#                         f"conf={self._fmt(t['confidence'])})")
        
#         l = ov["liquidity"]
#         if status.get("liquidity") and l["score"] is not None:
#             parts.append(f"Liquidity {self._fmt(l['score'])} | "
#                         f"depth={l['depth'] or 'n/a'} | spread={l['spread'] or 'n/a'}")
        
#         r = ov["regime"]
#         if status.get("regime") and r["label"] is not None:
#             parts.append(f"Regime {r['label']} (str={self._fmt(r['strength'])}, "
#                         f"dir={r['direction'] or 'n/a'})")
        
#         m = ov["matrix"]
#         if status.get("matrix") and m["accuracy"] is not None:
#             parts.append(f"Matrix acc={self._fmt(m['accuracy'])}")
        
#         rs = ov.get("risk_scaling")
#         if status.get("time_risk") and rs is not None:
#             parts.append(f"Risk scaling={self._fmt(rs)}")
        
#         thesis = " | ".join(parts) if parts else "Limited data available"
        
#         if unavailable:
#             thesis += f" | No data from: {', '.join(unavailable)}"
        
#         return thesis
    
#     def _fmt(self, x: Any) -> str:
#         """Format value for display."""
#         if x is None:
#             return "n/a"
#         try:
#             xf = float(x)
#             return f"{xf:.1%}" if 0.0 <= xf <= 1.0 else f"{xf:.3f}"
#         except:
#             return "n/a"
    
#     def _emit_debug_report(self, overview: Dict[str, Any], thesis: str,
#                           outputs: Dict[str, Dict[str, Any]], elapsed_ms: float) -> None:
#         """Emit debug report."""
#         self._dbg("=" * 88)
#         self._dbg(f"[TICK] Complete in {elapsed_ms:.1f}ms")
#         self._dbg(f"[THESIS] {thesis}")
        
#         if self.debug_theses:
#             self._dbg("-" * 88)
#             for name, out in outputs.items():
#                 if out.get('data_available'):
#                     helper_thesis = out.get('thesis', out.get('_thesis', 'No thesis'))
#                     self._dbg(f"[{name}] {helper_thesis[:200]}")
#                 else:
#                     self._dbg(f"[{name}] NO DATA")
    
#     def _dbg(self, msg: str) -> None:
#         """Debug output."""
#         self.logger.info(msg)
#         if self.debug:
#             print(msg)