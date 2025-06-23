import numpy as np
import datetime
from typing import Any, Tuple, List, Dict, Optional, Union
from collections import defaultdict
from modules.core.core import Module

class TradeThesisTracker(Module):
    """
    Enhanced trade thesis tracker that integrates with the trading environment.
    Tracks performance by strategy, pattern, and context.
    """
    def __init__(self, debug: bool = True):
        super().__init__()
        self.debug = debug
        self.reset()

    def reset(self) -> None:
        """Reset all tracking data"""
        self.records: List[Dict[str, Any]] = []
        self.thesis_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_pnl": 0.0, "wins": 0, "losses": 0}
        )
        self.pattern_performance: Dict[str, Dict[str, float]] = defaultdict(
            lambda: {"count": 0, "total_pnl": 0.0}
        )
        self.context_stats: Dict[str, Any] = {}
        self.active_trades: Dict[str, Dict[str, Any]] = {}

    def step(self, trades: Optional[List[Dict[str, Any]]] = None, **kwargs) -> None:
        """Process trades and extract thesis information"""
        if not trades:
            return
            
        current_time = datetime.datetime.now().isoformat()
        
        for trade in trades:
            # Extract trade information with defaults
            instrument = trade.get("instrument", "UNKNOWN")
            pnl = float(trade.get("pnl", 0.0))
            size = float(trade.get("size", 0.0))
            
            # Determine thesis from various sources
            thesis = self._extract_thesis(trade, kwargs)
            
            # Extract pattern and context
            pattern = self._extract_pattern(trade, kwargs)
            context = self._extract_context(trade, kwargs)
            
            # Create enhanced record
            record = {
                "timestamp": current_time,
                "instrument": instrument,
                "thesis": thesis,
                "pattern": pattern,
                "context": context,
                "pnl": pnl,
                "size": size,
                "side": trade.get("side", "UNKNOWN"),
                "entry_price": trade.get("entry_price", 0.0),
                "exit_price": trade.get("exit_price", 0.0),
                "duration": trade.get("duration", 0),
                "features": trade.get("features", []),
                "metadata": {
                    "exit_reason": trade.get("exit_reason", "unknown"),
                    "confidence": kwargs.get("consensus", 0.5),
                    "mode": kwargs.get("mode", "normal"),
                    "regime": kwargs.get("regime", "unknown")
                }
            }
            
            # Store record
            self.records.append(record)
            
            # Update performance tracking
            self._update_performance(record)
            
            if self.debug:
                print(f"[TradeThesisTracker] {instrument} | {thesis} | "
                      f"PnL: {pnl:.2f} | Pattern: {pattern}")

    def _extract_thesis(self, trade: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract thesis from trade and context"""
        # Check for explicit thesis
        if "thesis" in trade:
            return str(trade["thesis"])
            
        # Infer from voting data
        if "votes" in context:
            # Find dominant voter
            votes = context["votes"]
            if votes:
                # Aggregate votes across timeframes
                vote_totals = defaultdict(float)
                for vote_dict in votes.values():
                    for member, weight in vote_dict.items():
                        vote_totals[member] += weight
                        
                if vote_totals:
                    dominant = max(vote_totals.items(), key=lambda x: x[1])[0]
                    return f"strategy_{dominant}"
                    
        # Infer from features
        if "features" in trade and len(trade["features"]) > 0:
            features = trade["features"]
            if isinstance(features, np.ndarray) and len(features) > 2:
                # Simple classification based on feature values
                if features[0] > features[1]:  # Price up
                    return "momentum_long"
                else:
                    return "momentum_short"
                    
        # Default thesis based on side
        side = trade.get("side", "").upper()
        if side == "BUY":
            return "generic_long"
        elif side == "SELL":
            return "generic_short"
        else:
            return "unknown"

    def _extract_pattern(self, trade: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Extract market pattern from context"""
        # Check regime
        regime = context.get("regime", "").lower()
        
        # Check volatility
        volatility = context.get("volatility", {})
        if isinstance(volatility, dict):
            avg_vol = np.mean(list(volatility.values())) if volatility else 0.02
        else:
            avg_vol = float(volatility) if volatility else 0.02
            
        # Classify pattern
        if regime == "trending":
            if avg_vol < 0.015:
                return "trend_low_vol"
            else:
                return "trend_high_vol"
        elif regime == "volatile":
            return "range_bound"
        elif avg_vol > 0.03:
            return "high_volatility"
        else:
            return "normal"

    def _extract_context(self, trade: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract trading context"""
        return {
            "drawdown": context.get("drawdown", 0.0),
            "balance": context.get("balance", 0.0),
            "mode": context.get("mode", "normal"),
            "consensus": context.get("consensus", 0.5),
            "step": context.get("current_step", 0),
            "episode": context.get("episode", 0)
        }

    def _update_performance(self, record: Dict[str, Any]):
        """Update performance statistics"""
        thesis = record["thesis"]
        pattern = record["pattern"]
        pnl = record["pnl"]
        
        # Update thesis performance
        thesis_stats = self.thesis_performance[thesis]
        thesis_stats["count"] += 1
        thesis_stats["total_pnl"] += pnl
        if pnl > 0:
            thesis_stats["wins"] += 1
        elif pnl < 0:
            thesis_stats["losses"] += 1
            
        # Update pattern performance
        pattern_stats = self.pattern_performance[pattern]
        pattern_stats["count"] += 1
        pattern_stats["total_pnl"] += pnl

    def get_observation_components(self, last_n: int = 50) -> np.ndarray:
        """Return comprehensive performance metrics"""
        if not self.records:
            return np.zeros(12, dtype=np.float32)

        # Use recent records
        recent = self.records[-last_n:] if last_n else self.records
        
        # Calculate metrics
        total_trades = len(recent)
        total_pnl = sum(r["pnl"] for r in recent)
        win_rate = sum(1 for r in recent if r["pnl"] > 0) / max(1, total_trades)
        
        # Thesis diversity
        unique_theses = len(set(r["thesis"] for r in recent))
        
        # Best and worst thesis
        best_thesis_pnl = 0.0
        worst_thesis_pnl = 0.0
        if self.thesis_performance:
            perfs = [(t, s["total_pnl"]/max(1, s["count"])) 
                     for t, s in self.thesis_performance.items()]
            if perfs:
                best_thesis_pnl = max(p[1] for p in perfs)
                worst_thesis_pnl = min(p[1] for p in perfs)
        
        # Pattern analysis
        pattern_counts = defaultdict(int)
        for r in recent:
            pattern_counts[r["pattern"]] += 1
        
        most_common_pattern_pct = (
            max(pattern_counts.values()) / total_trades 
            if pattern_counts else 0.0
        )
        
        # Average metrics
        avg_pnl = total_pnl / max(1, total_trades)
        avg_duration = np.mean([r["duration"] for r in recent if "duration" in r] or [0])
        avg_size = np.mean([abs(r["size"]) for r in recent if "size" in r] or [0])
        
        # Sharpe approximation (daily)
        if total_trades > 1:
            pnls = [r["pnl"] for r in recent]
            sharpe = np.mean(pnls) / (np.std(pnls) + 1e-6) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return np.array([
            float(total_trades),
            total_pnl,
            win_rate,
            float(unique_theses),
            best_thesis_pnl,
            worst_thesis_pnl,
            most_common_pattern_pct,
            avg_pnl,
            avg_duration,
            avg_size,
            sharpe,
            float(len(self.thesis_performance))
        ], dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        stats = {
            "total_records": len(self.records),
            "thesis_performance": dict(self.thesis_performance),
            "pattern_performance": dict(self.pattern_performance),
            "top_strategies": self.get_top_strategies(5),
            "worst_strategies": self.get_worst_strategies(5),
            "pattern_analysis": self.analyze_patterns(),
            "time_analysis": self.analyze_by_time()
        }
        return stats

    def get_top_strategies(self, n: int = 5) -> List[Tuple[str, Dict[str, float]]]:
        """Get top performing strategies"""
        if not self.thesis_performance:
            return []
            
        # Calculate average PnL per trade
        thesis_avg = []
        for thesis, stats in self.thesis_performance.items():
            if stats["count"] > 0:
                avg_pnl = stats["total_pnl"] / stats["count"]
                win_rate = stats["wins"] / stats["count"] if stats["count"] > 0 else 0
                thesis_avg.append((
                    thesis,
                    {
                        "avg_pnl": avg_pnl,
                        "total_pnl": stats["total_pnl"],
                        "count": stats["count"],
                        "win_rate": win_rate
                    }
                ))
        
        # Sort by average PnL
        thesis_avg.sort(key=lambda x: x[1]["avg_pnl"], reverse=True)
        return thesis_avg[:n]

    def get_worst_strategies(self, n: int = 5) -> List[Tuple[str, Dict[str, float]]]:
        """Get worst performing strategies"""
        strategies = self.get_top_strategies(len(self.thesis_performance))
        return strategies[-n:] if len(strategies) >= n else strategies

    def analyze_patterns(self) -> Dict[str, Any]:
        """Analyze performance by market pattern"""
        pattern_analysis = {}
        
        for pattern, stats in self.pattern_performance.items():
            if stats["count"] > 0:
                pattern_analysis[pattern] = {
                    "avg_pnl": stats["total_pnl"] / stats["count"],
                    "total_pnl": stats["total_pnl"],
                    "count": stats["count"],
                    "percentage": stats["count"] / len(self.records) if self.records else 0
                }
                
        return pattern_analysis

    def analyze_by_time(self) -> Dict[str, Any]:
        """Analyze performance over time"""
        if not self.records:
            return {}
            
        # Group by hour of day (if timestamps available)
        hourly_pnl = defaultdict(lambda: {"count": 0, "total_pnl": 0.0})
        
        for record in self.records:
            try:
                timestamp = datetime.datetime.fromisoformat(record["timestamp"])
                hour = timestamp.hour
                hourly_pnl[hour]["count"] += 1
                hourly_pnl[hour]["total_pnl"] += record["pnl"]
            except:
                pass
                
        # Calculate averages
        hourly_avg = {}
        for hour, stats in hourly_pnl.items():
            if stats["count"] > 0:
                hourly_avg[hour] = stats["total_pnl"] / stats["count"]
                
        return {
            "hourly_average_pnl": hourly_avg,
            "best_hour": max(hourly_avg.items(), key=lambda x: x[1])[0] if hourly_avg else None,
            "worst_hour": min(hourly_avg.items(), key=lambda x: x[1])[0] if hourly_avg else None
        }

    def get_state(self) -> Dict[str, Any]:
        """Save complete state"""
        return {
            "records": self.records.copy(),
            "thesis_performance": dict(self.thesis_performance),
            "pattern_performance": dict(self.pattern_performance),
            "context_stats": self.context_stats.copy(),
            "active_trades": self.active_trades.copy()
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore complete state"""
        self.records = state.get("records", []).copy()
        
        # Restore defaultdicts
        self.thesis_performance = defaultdict(
            lambda: {"count": 0, "total_pnl": 0.0, "wins": 0, "losses": 0}
        )
        self.thesis_performance.update(state.get("thesis_performance", {}))
        
        self.pattern_performance = defaultdict(
            lambda: {"count": 0, "total_pnl": 0.0}
        )
        self.pattern_performance.update(state.get("pattern_performance", {}))
        
        self.context_stats = state.get("context_stats", {}).copy()
        self.active_trades = state.get("active_trades", {}).copy()
        
        if self.debug:
            print(f"[TradeThesisTracker] State restored ({len(self.records)} records)")

    def export_as_jsonl(self, path: str):
        """Export records as JSONL"""
        import json
        with open(path, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r, default=str) + "\n")

    def to_dataframe(self):
        """Convert to pandas DataFrame"""
        import pandas as pd
        if not self.records:
            return pd.DataFrame()
            
        # Flatten nested structures
        flat_records = []
        for r in self.records:
            flat = r.copy()
            # Flatten context
            if "context" in flat and isinstance(flat["context"], dict):
                for k, v in flat["context"].items():
                    flat[f"context_{k}"] = v
                del flat["context"]
            # Flatten metadata
            if "metadata" in flat and isinstance(flat["metadata"], dict):
                for k, v in flat["metadata"].items():
                    flat[f"meta_{k}"] = v
                del flat["metadata"]
            flat_records.append(flat)
            
        return pd.DataFrame(flat_records)

    def get_recommendation(self) -> Dict[str, Any]:
        """Get trading recommendations based on historical performance"""
        if not self.thesis_performance:
            return {"recommendation": "Insufficient data"}
            
        top_strategies = self.get_top_strategies(3)
        worst_strategies = self.get_worst_strategies(3)
        pattern_analysis = self.analyze_patterns()
        
        recommendations = {
            "preferred_strategies": [s[0] for s in top_strategies],
            "avoid_strategies": [s[0] for s in worst_strategies],
            "best_pattern": max(pattern_analysis.items(), 
                               key=lambda x: x[1]["avg_pnl"])[0] if pattern_analysis else None,
            "total_pnl": sum(r["pnl"] for r in self.records),
            "win_rate": sum(1 for r in self.records if r["pnl"] > 0) / len(self.records) if self.records else 0
        }
        
        return recommendations