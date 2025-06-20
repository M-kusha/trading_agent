import numpy as np
import logging
import random
import json
import os
import datetime
from typing import Dict, List, Union, Optional, Any, Tuple
from collections import deque
from modules.core.core import Module


class PortfolioRiskSystem(Module):
    """
    FIXED: Practical portfolio risk management with proper position limits.
    
    Key improvements:
    - Dynamic risk adjustment based on performance
    - Proper audit trail implementation
    - Better correlation handling
    - Integration with actual trading sizes
    - More realistic VaR calculations
    """

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

        # State tracking
        self.returns_history: Dict[str, deque] = {
            inst: deque(maxlen=max(var_window, correlation_window))
            for inst in self.instruments
        }
        self.portfolio_returns = deque(maxlen=var_window)
        self.current_positions: Dict[str, float] = {}
        self.performance_metrics = {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.5,
            "recent_pnl": 0.0,
        }
        
        # Risk adjustment factors
        self.risk_adjustment = 1.0  # Multiplier for dynamic adjustment
        self.min_risk_adjustment = 0.5
        self.max_risk_adjustment = 1.5
        
        # Audit trail
        self.last_audit: Dict[str, Any] = {}
        self.audit_buffer: List[Dict[str, Any]] = []
        self.audit_buffer_size = 100
        
        # Setup logging
        self._setup_logging()
        
        # Ensure audit directory exists
        os.makedirs(os.path.dirname(self.audit_log_path), exist_ok=True)
        
        self.logger.info(
            f"Initialized PortfolioRiskSystem | instruments={len(self.instruments)} "
            f"var_window={self.var_window} dd_limit={self.dd_limit:.2f}"
        )

    def _setup_logging(self):
        """Setup file and console logging"""
        self.logger = logging.getLogger(f"PortfolioRiskSystem_{id(self)}")
        self.logger.handlers.clear()
        
        # File handler
        log_dir = "logs/risk"
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(f"{log_dir}/portfolio_risk.log")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(fh)
        
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        self.logger.propagate = False

    def add_instrument(self, instrument: str):
        """Add a new instrument to track"""
        if instrument not in self.instruments:
            self.instruments.append(instrument)
            self.returns_history[instrument] = deque(
                maxlen=max(self.var_window, self.correlation_window)
            )
            self.logger.info(f"Added instrument: {instrument}")

    def remove_instrument(self, instrument: str):
        """Remove an instrument from tracking"""
        if instrument in self.instruments:
            self.instruments.remove(instrument)
            self.returns_history.pop(instrument, None)
            self.current_positions.pop(instrument, None)
            self.logger.info(f"Removed instrument: {instrument}")

    def update_returns(self, returns: Dict[str, float]):
        """Update returns history for all instruments"""
        for inst in self.instruments:
            ret = returns.get(inst, 0.0)
            self.returns_history[inst].append(float(ret))
            
        # Update portfolio return
        if self.current_positions:
            portfolio_ret = sum(
                self.current_positions.get(inst, 0.0) * returns.get(inst, 0.0)
                for inst in self.instruments
            )
            self.portfolio_returns.append(portfolio_ret)
        else:
            # Equal weight if no positions
            avg_ret = np.mean([returns.get(inst, 0.0) for inst in self.instruments])
            self.portfolio_returns.append(avg_ret)

    def update_performance_metrics(
        self,
        sharpe: Optional[float] = None,
        max_dd: Optional[float] = None,
        win_rate: Optional[float] = None,
        recent_pnl: Optional[float] = None,
    ):
        """Update performance metrics for dynamic risk adjustment"""
        if sharpe is not None:
            self.performance_metrics["sharpe"] = float(sharpe)
        if max_dd is not None:
            self.performance_metrics["max_dd"] = float(max_dd)
        if win_rate is not None:
            self.performance_metrics["win_rate"] = float(win_rate)
        if recent_pnl is not None:
            self.performance_metrics["recent_pnl"] = float(recent_pnl)
            
        # Update risk adjustment based on performance
        self._update_risk_adjustment()

    def _update_risk_adjustment(self):
        """Dynamically adjust risk based on recent performance"""
        metrics = self.performance_metrics
        
        # Start with base adjustment
        adjustment = 1.0
        
        # Adjust based on Sharpe ratio
        if metrics["sharpe"] > 1.5:
            adjustment *= 1.1
        elif metrics["sharpe"] < 0.5:
            adjustment *= 0.9
            
        # Adjust based on drawdown
        if metrics["max_dd"] > 0.15:
            adjustment *= 0.8
        elif metrics["max_dd"] < 0.05:
            adjustment *= 1.05
            
        # Adjust based on win rate
        if metrics["win_rate"] > 0.6:
            adjustment *= 1.05
        elif metrics["win_rate"] < 0.4:
            adjustment *= 0.95
            
        # Adjust based on recent P&L
        if metrics["recent_pnl"] < -100:  # Recent losses
            adjustment *= 0.9
        elif metrics["recent_pnl"] > 200:  # Recent gains
            adjustment *= 1.05
            
        # Apply bounds
        self.risk_adjustment = np.clip(
            adjustment,
            self.min_risk_adjustment,
            self.max_risk_adjustment
        )
        
        if self.debug:
            self.logger.debug(f"Risk adjustment updated to {self.risk_adjustment:.2f}")

    def calculate_var(self, confidence: float = 0.95) -> Dict[str, float]:
        """Calculate Value at Risk for each instrument"""
        var_dict = {}
        
        for inst in self.instruments:
            returns = list(self.returns_history[inst])
            
            if len(returns) < 5:  # Not enough data
                var_dict[inst] = 0.01  # Default 1% VaR
                continue
                
            # Calculate VaR using historical simulation
            sorted_returns = sorted(returns)
            index = int((1 - confidence) * len(sorted_returns))
            var = -sorted_returns[index] if sorted_returns[index] < 0 else 0.01
            
            # Ensure reasonable bounds
            var = np.clip(var, 0.001, 0.05)  # Between 0.1% and 5%
            var_dict[inst] = float(var)
            
        return var_dict

    def calculate_correlation_matrix(self) -> np.ndarray:
        """Calculate correlation matrix with proper error handling"""
        n = len(self.instruments)
        corr_matrix = np.eye(n)  # Start with identity matrix
        
        # Need at least 2 observations for correlation
        min_obs = 10
        
        for i, inst1 in enumerate(self.instruments):
            for j, inst2 in enumerate(self.instruments):
                if i >= j:  # Skip diagonal and lower triangle
                    continue
                    
                ret1 = list(self.returns_history[inst1])
                ret2 = list(self.returns_history[inst2])
                
                # Ensure same length
                min_len = min(len(ret1), len(ret2))
                if min_len < min_obs:
                    # Use default low correlation
                    corr_matrix[i, j] = 0.3
                    corr_matrix[j, i] = 0.3
                    continue
                
                # Calculate correlation
                try:
                    ret1_array = np.array(ret1[-min_len:])
                    ret2_array = np.array(ret2[-min_len:])
                    
                    # Check for zero variance
                    if np.std(ret1_array) < 1e-8 or np.std(ret2_array) < 1e-8:
                        corr = 0.0
                    else:
                        corr = np.corrcoef(ret1_array, ret2_array)[0, 1]
                        
                    # Ensure valid correlation
                    if not np.isfinite(corr):
                        corr = 0.0
                        
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    
                except Exception as e:
                    self.logger.warning(f"Correlation calc failed for {inst1}/{inst2}: {e}")
                    corr_matrix[i, j] = 0.3
                    corr_matrix[j, i] = 0.3
                    
        return corr_matrix

    def get_position_limits(self, balance: float) -> Dict[str, float]:
        """
        Calculate position limits for each instrument with proper audit trail.
        
        Returns dict of instrument -> maximum position size in base currency.
        """
        # Create audit record
        audit = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "balance": balance,
            "risk_adjustment": self.risk_adjustment,
            "method": "portfolio_risk_limits",
            "inputs": {
                "dd_limit": self.dd_limit,
                "risk_mult": self.risk_mult,
                "var_window": self.var_window,
                "performance_metrics": self.performance_metrics.copy(),
            },
            "calculations": {},
            "limits": {},
        }
        
        try:
            # Calculate VaR for each instrument
            var_dict = self.calculate_var()
            audit["calculations"]["var"] = var_dict.copy()
            
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix()
            audit["calculations"]["correlation_matrix"] = corr_matrix.tolist()
            
            # Portfolio optimization approach
            n = len(self.instruments)
            
            if n == 0:
                self.logger.warning("No instruments configured")
                audit["error"] = "No instruments"
                self._write_audit(audit)
                return {}
            
            # Calculate risk budget
            total_risk_budget = balance * self.dd_limit * self.risk_adjustment
            audit["calculations"]["total_risk_budget"] = total_risk_budget
            
            # Equal risk contribution as baseline
            equal_risk = total_risk_budget / n
            
            # Adjust based on VaR and correlations
            limits = {}
            
            for i, inst in enumerate(self.instruments):
                # Base allocation
                inst_var = var_dict[inst]
                
                # Adjust for correlations (reduce allocation for highly correlated assets)
                avg_corr = np.mean([
                    abs(corr_matrix[i, j]) 
                    for j in range(n) if j != i
                ]) if n > 1 else 0.0
                
                corr_adjustment = 1.0 - (avg_corr * 0.3)  # Reduce by up to 30% for high correlation
                
                # Calculate position limit
                position_limit = (equal_risk * corr_adjustment) / (inst_var * self.risk_mult)
                
                # Apply percentage bounds
                min_limit = balance * self.min_position_pct
                max_limit = balance * self.max_position_pct
                
                position_limit = np.clip(position_limit, min_limit, max_limit)
                
                limits[inst] = float(position_limit)
                
                # Add to audit
                audit["calculations"][f"{inst}_details"] = {
                    "var": inst_var,
                    "avg_correlation": avg_corr,
                    "corr_adjustment": corr_adjustment,
                    "raw_limit": position_limit,
                }
            
            audit["limits"] = limits.copy()
            audit["success"] = True
            
        except Exception as e:
            self.logger.error(f"Error calculating position limits: {e}")
            audit["error"] = str(e)
            audit["success"] = False
            
            # Fallback limits
            limits = {
                inst: balance * 0.05  # 5% fallback
                for inst in self.instruments
            }
            audit["limits"] = limits.copy()
            
        # Write audit
        self._write_audit(audit)
        self.last_audit = audit
        
        return limits

    def _write_audit(self, audit: Dict[str, Any]):
        """Write audit record to file and buffer"""
        # Add to buffer
        self.audit_buffer.append(audit)
        if len(self.audit_buffer) > self.audit_buffer_size:
            self.audit_buffer.pop(0)
            
        # Write to file
        try:
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(audit) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write audit: {e}")

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Get current risk metrics for monitoring"""
        metrics = {
            "risk_adjustment": self.risk_adjustment,
            "instruments": len(self.instruments),
            "performance": self.performance_metrics.copy(),
        }
        
        # Add VaR summary
        if self.instruments:
            var_dict = self.calculate_var()
            metrics["var_summary"] = {
                "min": min(var_dict.values()),
                "max": max(var_dict.values()),
                "avg": np.mean(list(var_dict.values())),
            }
            
        # Add correlation summary
        if len(self.instruments) > 1:
            corr_matrix = self.calculate_correlation_matrix()
            # Get upper triangle (excluding diagonal)
            upper_triangle = []
            n = len(self.instruments)
            for i in range(n):
                for j in range(i+1, n):
                    upper_triangle.append(corr_matrix[i, j])
                    
            if upper_triangle:
                metrics["correlation_summary"] = {
                    "min": float(np.min(upper_triangle)),
                    "max": float(np.max(upper_triangle)),
                    "avg": float(np.mean(upper_triangle)),
                }
                
        return metrics

    # ================= Module Interface Methods ==================== #

    def reset(self):
        """Reset state for new episode"""
        for inst in self.instruments:
            self.returns_history[inst].clear()
        self.portfolio_returns.clear()
        self.current_positions.clear()
        self.risk_adjustment = 1.0
        self.performance_metrics = {
            "sharpe": 0.0,
            "max_dd": 0.0,
            "win_rate": 0.5,
            "recent_pnl": 0.0,
        }
        self.logger.debug("Reset portfolio risk system")

    def step(self, **kwargs):
        """Update system with new data"""
        # Update returns if provided
        returns = kwargs.get("returns")
        if returns:
            self.update_returns(returns)
            
        # Update performance metrics if provided
        for key in ["sharpe", "max_dd", "win_rate", "recent_pnl"]:
            if key in kwargs:
                self.performance_metrics[key] = float(kwargs[key])
                
        # Update current positions if provided
        positions = kwargs.get("positions")
        if positions:
            self.current_positions = positions.copy()
            
        # Update risk adjustment
        self._update_risk_adjustment()

    def get_observation_components(self) -> np.ndarray:
        """Return risk metrics as observation components"""
        components = [
            self.risk_adjustment,
            self.performance_metrics["sharpe"],
            self.performance_metrics["max_dd"],
            self.performance_metrics["win_rate"],
        ]
        
        # Add VaR for each instrument
        var_dict = self.calculate_var()
        for inst in self.instruments:
            components.append(var_dict.get(inst, 0.01))
            
        return np.array(components, dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Get state for serialization"""
        return {
            "config": {
                "var_window": self.var_window,
                "dd_limit": self.dd_limit,
                "risk_mult": self.risk_mult,
                "instruments": self.instruments.copy(),
                "min_position_pct": self.min_position_pct,
                "max_position_pct": self.max_position_pct,
            },
            "returns_history": {
                inst: list(self.returns_history[inst])
                for inst in self.instruments
            },
            "portfolio_returns": list(self.portfolio_returns),
            "current_positions": self.current_positions.copy(),
            "performance_metrics": self.performance_metrics.copy(),
            "risk_adjustment": self.risk_adjustment,
        }

    def set_state(self, state: Dict[str, Any]):
        """Restore state from serialization"""
        # Restore config
        config = state.get("config", {})
        self.var_window = config.get("var_window", self.var_window)
        self.dd_limit = config.get("dd_limit", self.dd_limit)
        self.risk_mult = config.get("risk_mult", self.risk_mult)
        self.min_position_pct = config.get("min_position_pct", self.min_position_pct)
        self.max_position_pct = config.get("max_position_pct", self.max_position_pct)
        
        # Restore instruments and returns
        self.instruments = config.get("instruments", self.instruments)
        returns_hist = state.get("returns_history", {})
        for inst in self.instruments:
            if inst in returns_hist:
                self.returns_history[inst] = deque(
                    returns_hist[inst],
                    maxlen=max(self.var_window, self.correlation_window)
                )
                
        # Restore other state
        self.portfolio_returns = deque(
            state.get("portfolio_returns", []),
            maxlen=self.var_window
        )
        self.current_positions = state.get("current_positions", {}).copy()
        self.performance_metrics = state.get("performance_metrics", self.performance_metrics).copy()
        self.risk_adjustment = state.get("risk_adjustment", 1.0)

    # ================= Evolutionary Methods ==================== #

    def mutate(self, std=0.1):
        """Mutate risk parameters for evolution"""
        # Mutate window size
        self.var_window = int(np.clip(
            self.var_window + np.random.normal(0, std * 10),
            10, 100
        ))
        
        # Mutate risk limits
        self.dd_limit = float(np.clip(
            self.dd_limit + np.random.normal(0, std * 0.05),
            0.05, 0.40
        ))
        
        self.risk_mult = float(np.clip(
            self.risk_mult + np.random.normal(0, std),
            1.0, 5.0
        ))
        
        # Mutate position bounds
        self.min_position_pct = float(np.clip(
            self.min_position_pct + np.random.normal(0, std * 0.02),
            0.005, 0.05
        ))
        
        self.max_position_pct = float(np.clip(
            self.max_position_pct + np.random.normal(0, std * 0.05),
            0.1, 0.5
        ))
        
        self.logger.info(f"Mutated: var_window={self.var_window}, dd_limit={self.dd_limit:.2f}")

    def crossover(self, other: "PortfolioRiskSystem") -> "PortfolioRiskSystem":
        """Create offspring via crossover"""
        child = PortfolioRiskSystem(
            var_window=self.var_window if np.random.rand() > 0.5 else other.var_window,
            dd_limit=self.dd_limit if np.random.rand() > 0.5 else other.dd_limit,
            risk_mult=self.risk_mult if np.random.rand() > 0.5 else other.risk_mult,
            instruments=list(set(self.instruments) | set(other.instruments)),
            min_position_pct=self.min_position_pct if np.random.rand() > 0.5 else other.min_position_pct,
            max_position_pct=self.max_position_pct if np.random.rand() > 0.5 else other.max_position_pct,
            debug=self.debug or other.debug,
            audit_log_path=self.audit_log_path,
        )
        return child

    def get_last_audit(self) -> Dict[str, Any]:
        """Get the last audit record"""
        return self.last_audit.copy()

    def get_audit_log(self, n: int = 50) -> List[Dict[str, Any]]:
        """Get recent audit records from buffer"""
        return self.audit_buffer[-n:] if n < len(self.audit_buffer) else self.audit_buffer.copy()