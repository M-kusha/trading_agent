#  ─────────────────────────────────────────────────────────────
#  # modules/risk/portofilio_risk_system.py
#  ─────────────────────────────────────────────────────────────

import numpy as np
import logging
import random
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from modules.core.core import Module
from utils.get_dir import utcnow


class PortfolioRiskSystem(Module):
    def __init__(
        self,
        var_window: int = 20,  # Reduced for faster adaptation
        dd_limit: float = 0.20,
        instruments: Optional[List[str]] = None,
        risk_mult: float = 2.0,  # More conservative default
        min_position_pct: float = 0.01,  # Minimum 1% position
        max_position_pct: float = 0.25,  # Maximum 25% position
        correlation_window: int = 50,
        debug: bool = True,
        audit_log_path: str = "logs/risk/portfolio_risk_audit.jsonl",
        bootstrap_trades: int = 10,  # Number of trades before full risk kicks in
    ):
        super().__init__()
        
        # Core configuration
        self.var_window = int(var_window)
        self.dd_limit = float(dd_limit)
        self.instruments = instruments or ["EUR/USD", "XAU/USD"]  # Default instruments
        self.risk_mult = float(risk_mult)
        self.min_position_pct = float(min_position_pct)
        self.max_position_pct = float(max_position_pct)
        self.correlation_window = int(correlation_window)
        self.debug = debug
        self.audit_log_path = audit_log_path
        self.bootstrap_trades = bootstrap_trades

        # State tracking
        self.returns_history: Dict[str, deque] = {
            inst: deque(maxlen=max(var_window, correlation_window))
            for inst in self.instruments
        }
        self.portfolio_returns = deque(maxlen=var_window)
        self.current_positions: Dict[str, float] = {}
        self.position_history = deque(maxlen=100)
        self.trade_count = 0
        
        self.performance_metrics = {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.5,
            "recent_pnl": 0.0,
            "total_pnl": 0.0,
        }
        
        # Risk adjustment factors
        self.risk_adjustment = 1.0  # Multiplier for dynamic adjustment
        self.min_risk_adjustment = 0.5
        self.max_risk_adjustment = 1.5
        
        # Risk limits
        self.position_limits: Dict[str, float] = {
            inst: self.max_position_pct for inst in self.instruments
        }
        
        # Bootstrap mode
        self.bootstrap_mode = True
        
        # VaR calculation
        self.current_var = 0.0
        self.var_confidence = 0.95
        
        # Correlation tracking
        self.correlation_matrix: Optional[np.ndarray] = None
        self.max_correlation = 0.0
        
        # Audit trail
        self.audit_trail: List[Dict[str, Any]] = []
        self.max_audit_size = 100
        
        # Setup logging
        os.makedirs(os.path.dirname(audit_log_path), exist_ok=True)
        self.logger = logging.getLogger(f"PortfolioRiskSystem_{id(self)}")
        self.logger.handlers.clear()
        
        fh = logging.FileHandler(audit_log_path.replace(".jsonl", ".log"))
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(fh)
        
        if self.debug:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("[%(name)s] %(message)s"))
            self.logger.addHandler(ch)
            
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.logger.propagate = False

    def reset(self):
        """Reset portfolio risk system"""
        for inst in self.instruments:
            self.returns_history[inst].clear()
        self.portfolio_returns.clear()
        self.current_positions.clear()
        self.position_history.clear()
        self.trade_count = 0
        self.bootstrap_mode = True
        
        self.performance_metrics = {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.5,
            "recent_pnl": 0.0,
            "total_pnl": 0.0,
        }
        
        self.risk_adjustment = 1.0
        self.current_var = 0.0
        self.correlation_matrix = None
        self.max_correlation = 0.0
        
        for inst in self.instruments:
            self.position_limits[inst] = self.max_position_pct
            
        self.audit_trail.clear()
        
        if self.debug:
            self.logger.debug("Portfolio risk system reset")

    def update_position(self, trade: Dict[str, Any]):
        """Update position tracking after trade execution"""
        instrument = trade.get("instrument")
        if instrument not in self.instruments:
            return
            
        size = trade.get("size", 0)
        side = trade.get("side", "BUY")
        pnl = trade.get("pnl", 0)
        
        # Update position
        if instrument in self.current_positions:
            if side == "BUY":
                self.current_positions[instrument] += size
            else:
                self.current_positions[instrument] -= size
        else:
            self.current_positions[instrument] = size if side == "BUY" else -size
            
        # Track trade
        self.trade_count += 1
        self.position_history.append({
            "timestamp": utcnow(),
            "instrument": instrument,
            "size": size,
            "side": side,
            "positions": dict(self.current_positions),
        })
        
        # Update performance
        self.performance_metrics["recent_pnl"] = pnl
        self.performance_metrics["total_pnl"] += pnl
        
        # Check bootstrap mode
        if self.bootstrap_mode and self.trade_count >= self.bootstrap_trades:
            self.bootstrap_mode = False
            if self.debug:
                self.logger.info(f"Exiting bootstrap mode after {self.trade_count} trades")

    def prime_returns_with_history(self, price_history: Dict[str, np.ndarray]):
        """
        Prime the returns history with historical price data.
        
        Args:
            price_history: Dict mapping instrument to price array
        """
        for inst, prices in price_history.items():
            if inst not in self.instruments or len(prices) < 2:
                continue
                
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Add to history
            for ret in returns[-self.correlation_window:]:
                self.returns_history[inst].append(float(ret))
                
        if self.debug:
            filled = sum(1 for hist in self.returns_history.values() if len(hist) > 0)
            self.logger.debug(f"Primed returns for {filled}/{len(self.instruments)} instruments")

    def prime_returns_with_random(self):
        """Prime returns with realistic random data for bootstrap"""
        for inst in self.instruments:
            # Generate realistic returns based on instrument type
            if "XAU" in inst or "GOLD" in inst:
                # Gold: higher volatility
                std = 0.015
            else:
                # Forex: lower volatility
                std = 0.008
                
            # Generate returns
            n_samples = min(20, self.var_window)
            returns = np.random.normal(0, std, n_samples)
            
            for ret in returns:
                self.returns_history[inst].append(float(ret))
                
        if self.debug:
            self.logger.debug("Primed returns with random data")

    def calculate_var(self, returns: Optional[Dict[str, np.ndarray]] = None) -> float:
        """
        Calculate portfolio Value at Risk.
        
        Args:
            returns: Optional dict of returns by instrument
            
        Returns:
            VaR as percentage of portfolio value
        """
        # Use provided returns or history
        if returns is None:
            returns = {}
            for inst in self.instruments:
                if len(self.returns_history[inst]) >= 5:
                    returns[inst] = np.array(list(self.returns_history[inst]))
                    
        if not returns:
            return 0.0
            
        # Simple VaR calculation - can be enhanced
        all_returns = []
        for inst, ret_array in returns.items():
            if len(ret_array) > 0:
                all_returns.extend(ret_array)
                
        if len(all_returns) < 10:
            return 0.0
            
        # Calculate percentile VaR
        all_returns = np.array(all_returns)
        var_percentile = (1 - self.var_confidence) * 100
        self.current_var = abs(np.percentile(all_returns, var_percentile))
        
        return float(self.current_var)

    def calculate_correlations(self) -> np.ndarray:
        """Calculate correlation matrix for instruments"""
        n_inst = len(self.instruments)
        corr_matrix = np.eye(n_inst)
        
        # Need enough data
        min_len = min(len(self.returns_history[inst]) for inst in self.instruments)
        if min_len < 10:
            return corr_matrix
            
        # Build returns matrix
        returns_matrix = []
        for inst in self.instruments:
            ret = list(self.returns_history[inst])[-min_len:]
            returns_matrix.append(ret)
            
        returns_matrix = np.array(returns_matrix)
        
        # Calculate correlations
        for i in range(n_inst):
            for j in range(i+1, n_inst):
                corr = np.corrcoef(returns_matrix[i], returns_matrix[j])[0, 1]
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
                
        self.correlation_matrix = corr_matrix
        self.max_correlation = np.max(np.abs(corr_matrix[np.triu_indices(n_inst, k=1)]))
        
        return corr_matrix

    def get_position_limits(self) -> Dict[str, float]:
        """
        Get risk-adjusted position limits for each instrument.
        
        Returns:
            Dict mapping instrument to max position percentage
        """
        # Update risk adjustment based on performance
        self._update_risk_adjustment()
        
        # Calculate base limits
        limits = {}
        
        for inst in self.instruments:
            # Base limit
            base_limit = self.max_position_pct
            
            # Adjust for instrument volatility
            if len(self.returns_history[inst]) >= 10:
                inst_vol = np.std(list(self.returns_history[inst])[-20:])
                # Higher volatility = lower limit
                vol_adjustment = min(0.01 / (inst_vol + 0.001), 1.5)
                base_limit *= vol_adjustment
                
            # Apply risk adjustment
            base_limit *= self.risk_adjustment
            
            # Apply bootstrap bonus
            if self.bootstrap_mode:
                base_limit *= 1.5  # Allow larger positions during bootstrap
                
            # Enforce min/max
            limits[inst] = np.clip(base_limit, self.min_position_pct, self.max_position_pct)
            
        # Reduce limits if high correlation
        if self.max_correlation > 0.7:
            correlation_penalty = 1.0 - (self.max_correlation - 0.7) * 2
            for inst in limits:
                limits[inst] *= max(0.5, correlation_penalty)
                
        # Store for reference
        self.position_limits = limits
        
        return limits

    def _update_risk_adjustment(self):
        """Update dynamic risk adjustment based on performance"""
        # Need enough history
        if len(self.portfolio_returns) < 10:
            return
            
        # Calculate recent Sharpe ratio
        returns = np.array(list(self.portfolio_returns)[-20:])
        if returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            self.performance_metrics["sharpe"] = float(sharpe)
        else:
            sharpe = 0
            
        # Calculate max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        self.performance_metrics["max_dd"] = float(np.min(drawdown))
        
        # Adjust risk based on performance
        if sharpe > 1.5 and self.performance_metrics["max_dd"] > -0.1:
            # Good performance, increase risk
            self.risk_adjustment = min(self.max_risk_adjustment, 1.0 + sharpe * 0.1)
        elif sharpe < 0 or self.performance_metrics["max_dd"] < -0.15:
            # Poor performance, reduce risk
            self.risk_adjustment = max(self.min_risk_adjustment, 1.0 + sharpe * 0.2)
        else:
            # Gradual return to baseline
            self.risk_adjustment = 0.9 * self.risk_adjustment + 0.1 * 1.0

    def check_risk_limits(self, proposed_positions: Dict[str, float]) -> Tuple[bool, str]:
        """
        Check if proposed positions violate risk limits.
        
        Returns:
            (passed, reason) tuple
        """
        # Calculate total exposure
        total_exposure = sum(abs(pos) for pos in proposed_positions.values())
        
        # Check total exposure limit
        if total_exposure > 1.0:  # 100% of capital
            return False, f"Total exposure {total_exposure:.1%} exceeds limit"
            
        # Check individual position limits
        limits = self.get_position_limits()
        for inst, pos in proposed_positions.items():
            if inst in limits and abs(pos) > limits[inst]:
                return False, f"{inst} position {abs(pos):.1%} exceeds limit {limits[inst]:.1%})"
                
        # Check VaR limit
        if self.current_var > 0.05:  # 5% VaR limit
            return False, f"Portfolio VaR {self.current_var:.1%} exceeds limit"
            
        # Check correlation concentration
        if self.max_correlation > 0.9 and len(proposed_positions) > 1:
            return False, f"High correlation {self.max_correlation:.2f} with multiple positions"
            
        return True, "All risk checks passed"

    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics"""
        metrics = {
            "var": float(self.current_var),
            "max_correlation": float(self.max_correlation),
            "risk_adjustment": float(self.risk_adjustment),
            "total_exposure": sum(abs(pos) for pos in self.current_positions.values()),
            "position_count": len([p for p in self.current_positions.values() if abs(p) > 0.001]),
            "bootstrap_mode": float(self.bootstrap_mode),
        }
        
        # Add performance metrics
        metrics.update({
            f"perf_{k}": float(v) for k, v in self.performance_metrics.items()
        })
        
        return metrics

    def step(self, returns: Optional[Dict[str, float]] = None, **kwargs):
        """Update portfolio risk metrics"""
        # Update returns history
        if returns:
            for inst, ret in returns.items():
                if inst in self.instruments:
                    self.returns_history[inst].append(float(ret))
                    
            # Update portfolio return
            if self.current_positions:
                portfolio_return = sum(
                    ret * self.current_positions.get(inst, 0)
                    for inst, ret in returns.items()
                )
                self.portfolio_returns.append(portfolio_return)
                
        # Recalculate risk metrics
        if not self.bootstrap_mode:
            self.calculate_var()
            self.calculate_correlations()
            
        # Audit
        if self.debug and random.random() < 0.1:  # Sample audits
            self._add_audit({
                "timestamp": utcnow(),
                "metrics": self.get_risk_metrics(),
                "position_limits": self.get_position_limits(),
            })

    def get_observation_components(self) -> np.ndarray:
        """Get risk system features for observation"""
        features = [
            float(self.current_var),
            float(self.max_correlation),
            float(self.risk_adjustment),
            float(self.bootstrap_mode),
            float(len(self.current_positions)),
            float(sum(abs(p) for p in self.current_positions.values())),
            float(self.performance_metrics["sharpe"]),
            float(self.performance_metrics["max_dd"]),
            float(self.performance_metrics["win_rate"]),
            float(self.trade_count / self.bootstrap_trades) if self.bootstrap_mode else 1.0,
        ]
        
        return np.array(features, dtype=np.float32)

    def _add_audit(self, entry: Dict[str, Any]):
        """Add entry to audit trail"""
        self.audit_trail.append(entry)
        
        # Trim to size
        if len(self.audit_trail) > self.max_audit_size:
            self.audit_trail = self.audit_trail[-self.max_audit_size:]
            
        # Write to file
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit: {e}")

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information"""
        return {
            "risk_metrics": self.get_risk_metrics(),
            "position_limits": self.get_position_limits(),
            "current_positions": dict(self.current_positions),
            "bootstrap_mode": self.bootstrap_mode,
            "trade_count": self.trade_count,
            "returns_history_lengths": {
                inst: len(hist) for inst, hist in self.returns_history.items()
            },
        }

    def save_state(self) -> Dict[str, Any]:
        """Save portfolio risk state"""
        return {
            "config": {
                "var_window": self.var_window,
                "dd_limit": self.dd_limit,
                "instruments": self.instruments,
                "risk_mult": self.risk_mult,
                "min_position_pct": self.min_position_pct,
                "max_position_pct": self.max_position_pct,
            },
            "state": {
                "current_positions": dict(self.current_positions),
                "performance_metrics": dict(self.performance_metrics),
                "risk_adjustment": float(self.risk_adjustment),
                "current_var": float(self.current_var),
                "max_correlation": float(self.max_correlation),
                "bootstrap_mode": self.bootstrap_mode,
                "trade_count": self.trade_count,
            },
            "history": {
                "returns": {
                    inst: list(hist)[-20:] for inst, hist in self.returns_history.items()
                },
                "portfolio_returns": list(self.portfolio_returns)[-20:],
            },
        }

    def load_state(self, state: Dict[str, Any]):
        """Load portfolio risk state"""
        # Load config
        if "config" in state:
            cfg = state["config"]
            self.var_window = cfg.get("var_window", self.var_window)
            self.dd_limit = cfg.get("dd_limit", self.dd_limit)
            self.instruments = cfg.get("instruments", self.instruments)
            self.risk_mult = cfg.get("risk_mult", self.risk_mult)
            self.min_position_pct = cfg.get("min_position_pct", self.min_position_pct)
            self.max_position_pct = cfg.get("max_position_pct", self.max_position_pct)
            
        # Load state
        if "state" in state:
            st = state["state"]
            self.current_positions = st.get("current_positions", {})
            self.performance_metrics = st.get("performance_metrics", self.performance_metrics)
            self.risk_adjustment = float(st.get("risk_adjustment", 1.0))
            self.current_var = float(st.get("current_var", 0.0))
            self.max_correlation = float(st.get("max_correlation", 0.0))
            self.bootstrap_mode = st.get("bootstrap_mode", True)
            self.trade_count = st.get("trade_count", 0)
            
        # Load history
        if "history" in state:
            hist = state["history"]
            
            # Load returns
            if "returns" in hist:
                for inst, returns in hist["returns"].items():
                    if inst in self.instruments:
                        self.returns_history[inst].clear()
                        for r in returns:
                            self.returns_history[inst].append(r)
                            
            # Load portfolio returns
            if "portfolio_returns" in hist:
                self.portfolio_returns.clear()
                for r in hist["portfolio_returns"]:
                    self.portfolio_returns.append(r)