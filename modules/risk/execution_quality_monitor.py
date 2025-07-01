# ──────────────────────────────────────────────────────────────
# File: modules/risk/execution_quality_monitor.py
# ──────────────────────────────────────────────────────────────

import numpy as np
import logging
import json
import os
import datetime
from collections import deque
from typing import Dict, Any, List, Optional
from modules.core.core import Module
from utils.get_dir import utcnow


class ExecutionQualityMonitor(Module):
    AUDIT_PATH = "logs/risk/execution_quality_monitor_audit.jsonl"
    LOG_PATH   = "logs/risk/execution_quality_monitor.log"

    def __init__(
        self,
        slip_limit: float = 0.002,
        latency_limit: int = 1000,
        min_fill_rate: float = 0.95,
        enabled: bool = True,
        audit_log_size: int = 100,
        stats_window: int = 50,
        training_mode: bool = True,  # NEW: Training vs live mode
        debug: bool = True
    ):
        super().__init__()
        self.slip_limit = slip_limit
        self.latency_limit = latency_limit
        self.min_fill_rate = min_fill_rate
        self.enabled = enabled
        self.training_mode = training_mode
        self.debug = debug
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(self.AUDIT_PATH), exist_ok=True)
        
        # Enhanced statistics tracking
        self.stats_window = stats_window
        self.slippage_history = deque(maxlen=stats_window)
        self.latency_history = deque(maxlen=stats_window)
        self.fill_history = deque(maxlen=stats_window)
        self.spread_history = deque(maxlen=stats_window)  # NEW: Track spreads
        
        # Current metrics
        self.quality_score = 1.0
        self.step_count = 0
        self.execution_count = 0
        self.last_quality_log = 0
        
        # Enhanced issue tracking
        self.issues: Dict[str, List[str]] = {
            "slippage": [],
            "latency": [],
            "fill_rate": [],
            "spread": []
        }
        
        # Performance metrics
        self.metrics = {
            "avg_slippage": 0.0,
            "avg_latency": 0.0,
            "avg_fill_rate": 1.0,
            "total_executions": 0,
            "issue_rate": 0.0
        }
        
        # Audit
        self._audit: List[Dict[str, Any]] = []
        self._max_audit = audit_log_size

        # FIXED Logger setup
        self.logger = logging.getLogger("ExecutionQualityMonitor")
        if not self.logger.handlers:
            self.logger.handlers.clear()
            self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
            self.logger.propagate = False
            
            fh = logging.FileHandler(self.LOG_PATH, mode='a')
            fh.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
            
            if debug:
                ch = logging.StreamHandler()
                ch.setLevel(logging.INFO)
                ch.setFormatter(formatter)
                self.logger.addHandler(ch)
        
        self.logger.info(f"ExecutionQualityMonitor initialized - slip_limit={slip_limit}, latency_limit={latency_limit}, training={training_mode}")

    def reset(self):
        """Reset monitor state - FULL IMPLEMENTATION"""
        self.slippage_history.clear()
        self.latency_history.clear()
        self.fill_history.clear()
        self.spread_history.clear()
        self.quality_score = 1.0
        self.step_count = 0
        self.execution_count = 0
        self.last_quality_log = 0
        
        for key in self.issues:
            self.issues[key].clear()
            
        self.metrics = {
            "avg_slippage": 0.0,
            "avg_latency": 0.0,
            "avg_fill_rate": 1.0,
            "total_executions": 0,
            "issue_rate": 0.0
        }
        
        self._audit.clear()
        self.logger.info("ExecutionQualityMonitor reset - all state cleared")

    def step(
        self,
        trade_executions: Optional[List[Dict[str, Any]]] = None,
        order_attempts: Optional[List[Dict[str, Any]]] = None,
        trades: Optional[List[Dict[str, Any]]] = None,
        orders: Optional[List[Dict[str, Any]]] = None,
        spread_data: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """
        ENHANCED: Monitor execution quality across multiple dimensions.
        Now handles training environment properly.
        """
        self.step_count += 1
        
        # Handle different input formats
        executions = trade_executions or trades or []
        attempts = order_attempts or orders or []
        
        # Intelligent logging
        should_log = (
            self.debug and 
            (self.step_count - self.last_quality_log > 50 or 
             len(executions) > 0 or 
             self.step_count % 100 == 0)
        )
        
        if should_log:
            self.logger.debug(f"Step {self.step_count} - enabled={self.enabled}, executions={len(executions)}, attempts={len(attempts)}, training={self.training_mode}")
            self.last_quality_log = self.step_count
        
        if not self.enabled:
            self.quality_score = 1.0
            return
            
        try:
            # Clear previous issues
            for key in self.issues:
                self.issues[key].clear()
                
            execution_count = 0
            
            # Process executions with enhanced analysis
            if executions:
                for execution in executions:
                    try:
                        self._analyze_execution_enhanced(execution)
                        execution_count += 1
                    except Exception as e:
                        self.logger.error(f"Error analyzing execution: {e}")
                        
            # Process order attempts for fill rate
            if attempts:
                try:
                    self._analyze_fill_rate_enhanced(attempts)
                except Exception as e:
                    self.logger.error(f"Error analyzing fill rate: {e}")
            
            # Process spread data
            if spread_data:
                self._analyze_spreads(spread_data)
            
            # ENHANCED: Generate realistic synthetic data for training bootstrapping
            if self.training_mode and not executions and not attempts and len(self.slippage_history) < 10:
                self._generate_realistic_training_data()
            
            # Update execution count
            self.execution_count += execution_count
            self.metrics["total_executions"] = self.execution_count
            
            # Calculate overall quality score
            self._calculate_quality_score_enhanced()
            
            # Update metrics
            self._update_metrics()
            
            # Log summary with intelligent frequency
            total_issues = sum(len(issues) for issues in self.issues.values())
            if total_issues > 0 or self.step_count % 200 == 0:
                self.logger.info(f"Step {self.step_count} summary: quality_score={self.quality_score:.3f}, issues={total_issues}, executions={execution_count}")
            
            # Record audit if issues found or periodically
            if total_issues > 0 or self.step_count % 50 == 0:
                self._record_audit()
                
        except Exception as e:
            self.logger.error(f"Error in execution quality monitoring: {e}")

    def _generate_realistic_training_data(self):
        """ENHANCED: Generate realistic execution data for training bootstrapping"""
        try:
            # Generate realistic training execution metrics based on market conditions
            
            # Realistic slippage (based on typical forex/metals spreads)
            realistic_slippage = abs(np.random.gamma(2, 0.0003))  # Gamma distribution for realistic skew
            self.slippage_history.append(realistic_slippage)
            
            # Realistic latency (typical broker execution times)
            realistic_latency = max(50, np.random.gamma(3, 100))  # 50ms minimum, average ~300ms
            self.latency_history.append(realistic_latency)
            
            # Realistic fill rate (high but not perfect)
            realistic_fill_rate = np.random.beta(20, 2)  # High fill rate with occasional issues
            self.fill_history.append(realistic_fill_rate)
            
            # Realistic spread
            realistic_spread = abs(np.random.gamma(2, 0.0001))
            self.spread_history.append(realistic_spread)
            
            if self.debug and self.step_count % 100 == 0:
                self.logger.debug(f"Generated realistic training data: slip={realistic_slippage:.5f}, latency={realistic_latency:.0f}ms, fill={realistic_fill_rate:.3f}")
                
        except Exception as e:
            self.logger.error(f"Error generating training data: {e}")

    def _analyze_execution_enhanced(self, execution: Dict[str, Any]):
        """ENHANCED: Analyze individual execution quality with better data extraction"""
        try:
            instrument = execution.get("instrument", execution.get("symbol", "Unknown"))
            
            # Enhanced slippage analysis
            slippage = self._extract_slippage(execution)
            if slippage is not None:
                self.slippage_history.append(abs(slippage))
                
                if abs(slippage) > self.slip_limit:
                    msg = f"{instrument} slippage {slippage:.5f} > limit {self.slip_limit:.5f}"
                    self.issues["slippage"].append(msg)
                    self.logger.warning(msg)
                    
            # Enhanced latency analysis
            latency = self._extract_latency(execution)
            if latency is not None:
                self.latency_history.append(latency)
                
                if latency > self.latency_limit:
                    msg = f"{instrument} latency {latency:.0f}ms > limit {self.latency_limit}ms"
                    self.issues["latency"].append(msg)
                    self.logger.warning(msg)
                    
            # Spread analysis
            spread = self._extract_spread(execution)
            if spread is not None:
                self.spread_history.append(spread)
                
        except Exception as e:
            self.logger.error(f"Error analyzing execution for {execution.get('instrument', 'unknown')}: {e}")

    def _extract_slippage(self, execution: Dict[str, Any]) -> Optional[float]:
        """Extract slippage from execution data"""
        # Try different slippage field names
        for field in ["slippage", "slip", "price_diff", "execution_slippage"]:
            if field in execution and execution[field] is not None:
                return float(execution[field])
                
        # Calculate from expected vs actual price
        expected_price = execution.get("expected_price", execution.get("order_price"))
        actual_price = execution.get("actual_price", execution.get("fill_price", execution.get("price")))
        
        if expected_price is not None and actual_price is not None:
            return float(actual_price) - float(expected_price)
            
        return None

    def _extract_latency(self, execution: Dict[str, Any]) -> Optional[float]:
        """Extract latency from execution data"""
        # Try different latency field names
        for field in ["latency_ms", "latency", "execution_time", "fill_time_ms"]:
            if field in execution and execution[field] is not None:
                return float(execution[field])
                
        # Calculate from timestamps
        order_time = execution.get("order_time", execution.get("submit_time"))
        fill_time = execution.get("fill_time", execution.get("execution_time"))
        
        if order_time is not None and fill_time is not None:
            try:
                if isinstance(order_time, str):
                    order_time = datetime.datetime.fromisoformat(order_time.replace('Z', '+00:00'))
                if isinstance(fill_time, str):
                    fill_time = datetime.datetime.fromisoformat(fill_time.replace('Z', '+00:00'))
                    
                latency_seconds = (fill_time - order_time).total_seconds()
                return latency_seconds * 1000  # Convert to milliseconds
            except:
                pass
                
        return None

    def _extract_spread(self, execution: Dict[str, Any]) -> Optional[float]:
        """Extract spread from execution data"""
        # Try different spread field names
        for field in ["spread", "bid_ask_spread", "market_spread"]:
            if field in execution and execution[field] is not None:
                return float(execution[field])
                
        # Calculate from bid/ask
        bid = execution.get("bid_price", execution.get("bid"))
        ask = execution.get("ask_price", execution.get("ask"))
        
        if bid is not None and ask is not None:
            return float(ask) - float(bid)
            
        return None

    def _analyze_fill_rate_enhanced(self, attempts: List[Dict[str, Any]]):
        """ENHANCED: Analyze order fill rates with better status detection"""
        if not attempts:
            return
            
        try:
            successful = 0
            total = len(attempts)
            
            for order in attempts:
                # Enhanced status detection
                filled = self._is_order_filled(order)
                if filled:
                    successful += 1
                    
            fill_rate = successful / total if total > 0 else 1.0
            self.fill_history.append(fill_rate)
            
            if fill_rate < self.min_fill_rate:
                msg = f"Fill rate {fill_rate:.2%} below minimum {self.min_fill_rate:.2%} ({successful}/{total})"
                self.issues["fill_rate"].append(msg)
                self.logger.warning(msg)
                
        except Exception as e:
            self.logger.error(f"Error analyzing fill rate: {e}")

    def _is_order_filled(self, order: Dict[str, Any]) -> bool:
        """Enhanced order fill status detection"""
        # Check various status indicators
        status_indicators = [
            order.get("filled", False),
            order.get("status") in ["filled", "completed", "executed"],
            order.get("state") in ["filled", "completed", "executed"],
            order.get("executed", False),
            order.get("fill_status") == "filled"
        ]
        
        # Check quantity filled
        order_qty = order.get("quantity", order.get("size", order.get("volume", 0)))
        filled_qty = order.get("filled_quantity", order.get("filled_size", order.get("executed_quantity", 0)))
        
        if order_qty > 0 and filled_qty > 0:
            fill_ratio = filled_qty / order_qty
            status_indicators.append(fill_ratio >= 0.95)  # Consider 95%+ filled as success
        
        return any(status_indicators)

    def _analyze_spreads(self, spread_data: Dict[str, float]):
        """Analyze spread data for execution quality impact"""
        try:
            for instrument, spread in spread_data.items():
                if spread is not None and spread > 0:
                    self.spread_history.append(spread)
                    
                    # Wide spreads can affect execution quality
                    if spread > 0.01:  # 10 pips for forex, adjust as needed
                        msg = f"{instrument} wide spread {spread:.5f}"
                        self.issues["spread"].append(msg)
                        
        except Exception as e:
            self.logger.error(f"Error analyzing spreads: {e}")

    def _calculate_quality_score_enhanced(self):
        """ENHANCED: Calculate overall execution quality score with multiple factors"""
        try:
            scores = []
            weights = []
            
            # Slippage score
            if self.slippage_history:
                avg_slip = np.mean(self.slippage_history)
                slip_score = max(0, 1.0 - (avg_slip / (self.slip_limit * 2)))
                scores.append(slip_score)
                weights.append(0.3)
                
            # Latency score
            if self.latency_history:
                avg_latency = np.mean(self.latency_history)
                latency_score = max(0, 1.0 - (avg_latency / (self.latency_limit * 2)))
                scores.append(latency_score)
                weights.append(0.3)
                
            # Fill rate score
            if self.fill_history:
                avg_fill = np.mean(self.fill_history)
                scores.append(avg_fill)
                weights.append(0.3)
                
            # Spread score (lower spread = better quality)
            if self.spread_history:
                avg_spread = np.mean(self.spread_history)
                spread_score = max(0, 1.0 - (avg_spread / 0.01))  # Normalize to 10 pips
                scores.append(spread_score)
                weights.append(0.1)
                
            # Calculate weighted average
            if scores and weights:
                self.quality_score = float(np.average(scores, weights=weights))
            else:
                self.quality_score = 1.0
                
            # Apply issue penalty
            total_issues = sum(len(issues) for issues in self.issues.values())
            if total_issues > 0:
                issue_penalty = min(0.2, total_issues * 0.05)
                self.quality_score = max(0.1, self.quality_score - issue_penalty)
                
        except Exception as e:
            self.logger.error(f"Error calculating quality score: {e}")
            self.quality_score = 0.5  # Conservative fallback

    def _update_metrics(self):
        """Update performance metrics"""
        try:
            if self.slippage_history:
                self.metrics["avg_slippage"] = float(np.mean(self.slippage_history))
            if self.latency_history:
                self.metrics["avg_latency"] = float(np.mean(self.latency_history))
            if self.fill_history:
                self.metrics["avg_fill_rate"] = float(np.mean(self.fill_history))
                
            total_issues = sum(len(issues) for issues in self.issues.values())
            self.metrics["issue_rate"] = total_issues / max(self.execution_count, 1)
            
        except Exception as e:
            self.logger.error(f"Error updating metrics: {e}")

    def _record_audit(self):
        """Record audit entry for quality monitoring - ENHANCED"""
        try:
            entry = {
                "timestamp": utcnow(),
                "step": self.step_count,
                "quality_score": float(self.quality_score),
                "issues": {k: len(v) for k, v in self.issues.items()},
                "metrics": self.metrics.copy(),
                "statistics": self.get_execution_stats(),
                "thresholds": {
                    "slip_limit": self.slip_limit,
                    "latency_limit": self.latency_limit,
                    "min_fill_rate": self.min_fill_rate
                },
                "training_mode": self.training_mode
            }
            
            self._audit.append(entry)
            if len(self._audit) > self._max_audit:
                self._audit.pop(0)
                
            # Write to file with reduced frequency
            if any(self.issues.values()) or len(self._audit) % 20 == 0:
                try:
                    with open(self.AUDIT_PATH, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                except Exception as e:
                    if self.debug:
                        self.logger.error(f"Failed to write audit: {e}")
                        
        except Exception as e:
            self.logger.error(f"Error recording audit: {e}")

    def get_execution_stats(self) -> Dict[str, Any]:
        """Get detailed execution statistics - ENHANCED"""
        stats = {}
        
        try:
            if self.slippage_history:
                slips = list(self.slippage_history)
                stats["slippage"] = {
                    "mean": float(np.mean(slips)),
                    "std": float(np.std(slips)),
                    "max": float(np.max(slips)),
                    "min": float(np.min(slips)),
                    "p95": float(np.percentile(slips, 95)),
                    "count": len(slips)
                }
                
            if self.latency_history:
                latencies = list(self.latency_history)
                stats["latency"] = {
                    "mean": float(np.mean(latencies)),
                    "std": float(np.std(latencies)),
                    "max": float(np.max(latencies)),
                    "min": float(np.min(latencies)),
                    "p95": float(np.percentile(latencies, 95)),
                    "count": len(latencies)
                }
                
            if self.fill_history:
                fills = list(self.fill_history)
                stats["fill_rate"] = {
                    "mean": float(np.mean(fills)),
                    "min": float(np.min(fills)),
                    "current": float(fills[-1]) if fills else 1.0,
                    "below_threshold_count": len([f for f in fills if f < self.min_fill_rate]),
                    "count": len(fills)
                }
                
            if self.spread_history:
                spreads = list(self.spread_history)
                stats["spread"] = {
                    "mean": float(np.mean(spreads)),
                    "std": float(np.std(spreads)),
                    "max": float(np.max(spreads)),
                    "min": float(np.min(spreads)),
                    "count": len(spreads)
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating execution stats: {e}")
            
        return stats

    def get_observation_components(self) -> np.ndarray:
        """Return execution quality metrics as observation - ENHANCED"""
        try:
            has_issues = float(any(self.issues.values()))
            
            recent_slip = 0.0
            recent_latency = 0.0
            recent_fill = 1.0
            recent_spread = 0.0
            
            if self.slippage_history:
                recent_slip = np.mean(list(self.slippage_history)[-10:])
            if self.latency_history:
                recent_latency = np.mean(list(self.latency_history)[-10:])
            if self.fill_history:
                recent_fill = np.mean(list(self.fill_history)[-10:])
            if self.spread_history:
                recent_spread = np.mean(list(self.spread_history)[-10:])
            
            return np.array([
                float(self.quality_score),
                has_issues,
                float(np.clip(recent_slip / max(self.slip_limit, 1e-8), 0.0, 10.0)),
                float(np.clip(recent_latency / max(self.latency_limit, 1), 0.0, 10.0)),
                float(np.clip(recent_fill, 0.0, 1.0)),
                float(np.clip(recent_spread / 0.01, 0.0, 10.0))  # Normalized to 10 pips
            ], dtype=np.float32)
            
        except Exception:
            return np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Get complete state - FULL IMPLEMENTATION"""
        return {
            "limits": {
                "slip_limit": self.slip_limit,
                "latency_limit": self.latency_limit,
                "min_fill_rate": self.min_fill_rate
            },
            "enabled": self.enabled,
            "training_mode": self.training_mode,
            "step_count": self.step_count,
            "execution_count": self.execution_count,
            "quality_score": float(self.quality_score),
            "metrics": self.metrics.copy(),
            "statistics": self.get_execution_stats(),
            "history_sizes": {
                "slippage": len(self.slippage_history),
                "latency": len(self.latency_history),
                "fill_rate": len(self.fill_history),
                "spread": len(self.spread_history)
            },
            "current_issues": {k: len(v) for k, v in self.issues.items()},
            "audit_summary": {
                "total_entries": len(self._audit),
                "recent_issues": len([a for a in self._audit if any(a.get("issues", {}).values())])
            }
        }
