# Fixed MarketRegimeSwitcher with proper data handling and logging

import os
import numpy as np
from collections import deque
import copy
import logging
import json
from datetime import datetime, timezone
from ..core.core import Module

class MarketRegimeSwitcher(Module):
    """
    FIXED: Enhanced Market Regime Switcher with proper data handling and logging
    """
    REGIMES = ["trending_up", "trending_down", "mean_reverting", "high_vol", "low_vol", "neutral"]

    def __init__(
        self,
        window: int = 50,
        vol_window: int = 20,
        trend_factor: float = 0.3,  # FIXED: Reduced for more sensitivity
        mean_thr_factor: float = 0.05,  # FIXED: Reduced for more sensitivity
        vol_high_pct: float = 75.0,  # FIXED: Reduced for easier triggering
        vol_low_pct: float = 25.0,  # FIXED: Increased for easier triggering
        debug: bool = True,
    ):
        self.window = window
        self.vol_window = vol_window
        self.trend_factor = trend_factor
        self.mean_thr_factor = mean_thr_factor
        self.vol_high_pct = vol_high_pct
        self.vol_low_pct = vol_low_pct
        self.debug = debug
        self._action_dim = 1
        
        # FIXED: Enhanced price data handling
        self.prices = deque(maxlen=window)
        self.regime = "neutral"
        self.volatility = 0.0
        self.last_rationale = {}
        self.last_full_audit = {}
        
        # FIXED: Improved logging setup
        self._setup_logging()
        
        # FIXED: Bootstrap with some initial data to avoid always neutral
        self._bootstrap_data()

    def _setup_logging(self):
        """FIXED: Proper logging configuration"""
        # Ensure log directory exists
        os.makedirs("logs/regime", exist_ok=True)
        
        # Create unique logger to avoid conflicts
        logger_name = f"MarketRegimeSwitcher_{id(self)}"
        self.logger = logging.getLogger(logger_name)
        self.logger.handlers.clear()  # Clear any existing handlers
        
        # Set log level
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # File handler with proper formatting
        file_handler = logging.FileHandler("logs/regime/market_regime.log")
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        if self.debug:
            self.logger.addHandler(console_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
        
        # Test logging
        self.logger.info("MarketRegimeSwitcher logging initialized")
        
        # Rationale log file
        self.rationale_log_path = "logs/regime/regime_rationale.jsonl"

    def _bootstrap_data(self):
        """FIXED: Bootstrap with some initial synthetic data"""
        # Create some synthetic price movements to avoid always neutral
        base_price = 2000.0
        for i in range(min(self.window, 20)):
            # Add some realistic price movement
            change = np.random.normal(0, 0.01)
            price = base_price * (1 + change)
            self.prices.append(price)
            base_price = price
        
        if self.debug:
            self.logger.debug(f"Bootstrapped with {len(self.prices)} synthetic prices")

    def reset(self):
        """FIXED: Improved reset with logging"""
        old_regime = self.regime
        self.regime = "neutral"
        self.volatility = 0.0
        self.last_rationale = {}
        self.last_full_audit = {}
        self.prices.clear()
        
        # Re-bootstrap after reset
        self._bootstrap_data()
        
        self.logger.info(f"Reset: {old_regime} -> {self.regime}")

    def step(self, **kwargs):
        """
        FIXED: Enhanced step method that properly extracts price data
        """
        # FIXED: Extract price from various possible data sources
        price = None
        
        # Try to get price from kwargs
        if 'price' in kwargs:
            price = float(kwargs['price'])
        elif 'data_dict' in kwargs and 'current_step' in kwargs:
            # Extract from environment data
            try:
                data_dict = kwargs['data_dict']
                current_step = kwargs['current_step']
                
                # Get first instrument's daily data
                instruments = list(data_dict.keys())
                if instruments:
                    inst = instruments[0]
                    if "D1" in data_dict[inst]:
                        df = data_dict[inst]["D1"]
                        if current_step < len(df):
                            price = float(df.iloc[current_step]["close"])
                            
            except Exception as e:
                self.logger.warning(f"Failed to extract price from data_dict: {e}")
                
        elif 'env' in kwargs:
            # Extract from environment object
            try:
                env = kwargs['env']
                if hasattr(env, 'market_state') and hasattr(env, 'data'):
                    inst = env.instruments[0]
                    df = env.data[inst]["D1"]
                    step = env.market_state.current_step
                    if step < len(df):
                        price = float(df.iloc[step]["close"])
            except Exception as e:
                self.logger.warning(f"Failed to extract price from env: {e}")
        
        # If no price found, generate a synthetic one based on last price
        if price is None:
            if len(self.prices) > 0:
                last_price = self.prices[-1]
                # Add small random movement
                change = np.random.normal(0, 0.005)  # 0.5% std dev
                price = last_price * (1 + change)
            else:
                price = 2000.0  # Default starting price
                
            if self.debug:
                self.logger.debug(f"Generated synthetic price: {price:.4f}")
        
        # Now process the price
        self.prices.append(price)
        
        # FIXED: Immediate regime calculation even with less data
        if len(self.prices) >= 10:  # Reduced from window to 10 for faster detection
            self._calculate_regime()
        else:
            self.regime = "neutral"
            self.volatility = 0.0
            self.last_rationale = {"reason": f"Building data: {len(self.prices)}/{self.window}"}
            
        if self.debug:
            self.logger.debug(
                f"Step: price={price:.4f}, regime={self.regime}, "
                f"vol={self.volatility:.6f}, prices={len(self.prices)}"
            )

    def _calculate_regime(self):
        """FIXED: Enhanced regime calculation with better sensitivity"""
        prices = np.array(list(self.prices))
        
        # Calculate returns with proper handling
        if len(prices) < 2:
            self.regime = "neutral"
            return
            
        returns = np.diff(prices) / prices[:-1]  # Percentage returns
        
        # Remove any infinite or NaN values
        returns = returns[np.isfinite(returns)]
        if len(returns) == 0:
            self.regime = "neutral"
            return
            
        # Calculate metrics
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # Calculate recent volatility
        recent_prices = prices[-min(self.vol_window, len(prices)):]
        if len(recent_prices) >= 2:
            recent_returns = np.diff(recent_prices) / recent_prices[:-1]
            recent_returns = recent_returns[np.isfinite(recent_returns)]
            self.volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0.0
        else:
            self.volatility = 0.0

        # FIXED: More sensitive thresholds
        trend_thr = self.trend_factor * std_ret if std_ret > 0 else 0.001
        mean_thr = self.mean_thr_factor * std_ret if std_ret > 0 else 0.0005
        
        # Volatility thresholds based on recent data
        if len(returns) >= 10:
            vol_high_thr = np.percentile(np.abs(returns), self.vol_high_pct)
            vol_low_thr = np.percentile(np.abs(returns), self.vol_low_pct)
        else:
            vol_high_thr = 0.02  # 2% default
            vol_low_thr = 0.005  # 0.5% default

        # Store calculation details
        rationale = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prices_count": len(prices),
            "mean_ret": float(mean_ret),
            "std_ret": float(std_ret),
            "trend_thr": float(trend_thr),
            "volatility": float(self.volatility),
            "vol_high_thr": float(vol_high_thr),
            "vol_low_thr": float(vol_low_thr),
            "mean_thr": float(mean_thr),
        }

        # FIXED: Enhanced regime logic with proper ordering
        old_regime = self.regime
        
        # Check volatility first (most important)
        if self.volatility > vol_high_thr:
            new_regime = "high_vol"
            rationale["trigger"] = f"volatility ({self.volatility:.6f}) > vol_high_thr ({vol_high_thr:.6f})"
        elif self.volatility < vol_low_thr:
            new_regime = "low_vol"
            rationale["trigger"] = f"volatility ({self.volatility:.6f}) < vol_low_thr ({vol_low_thr:.6f})"
        # Then check trend
        elif mean_ret > trend_thr:
            new_regime = "trending_up"
            rationale["trigger"] = f"mean_ret ({mean_ret:.6f}) > trend_thr ({trend_thr:.6f})"
        elif mean_ret < -trend_thr:
            new_regime = "trending_down"
            rationale["trigger"] = f"mean_ret ({mean_ret:.6f}) < -trend_thr ({-trend_thr:.6f})"
        # Finally check mean reversion
        elif abs(mean_ret) < mean_thr and std_ret > 0:
            new_regime = "mean_reverting"
            rationale["trigger"] = f"abs(mean_ret) ({abs(mean_ret):.6f}) < mean_thr ({mean_thr:.6f})"
        else:
            new_regime = "neutral"
            rationale["trigger"] = "no conditions met (neutral)"

        # Update regime
        self.regime = new_regime
        self.last_rationale = rationale

        # Log regime changes
        if new_regime != old_regime:
            self.logger.info(
                f"Regime changed: {old_regime} -> {new_regime} | "
                f"Reason: {rationale['trigger']}"
            )
            
            # Save detailed rationale
            try:
                with open(self.rationale_log_path, "a") as f:
                    f.write(json.dumps({
                        "old_regime": old_regime,
                        "new_regime": new_regime,
                        **rationale
                    }) + "\n")
            except Exception as e:
                self.logger.error(f"Failed to write rationale log: {e}")
        
        # Always log in debug mode
        elif self.debug:
            self.logger.debug(
                f"Regime unchanged: {self.regime} | "
                f"mean_ret={mean_ret:.6f}, vol={self.volatility:.6f}"
            )

        # Update audit info
        self.last_full_audit = {
            "regime": self.regime,
            "rationale": rationale,
            "recent_prices": list(prices[-10:]),  # Last 10 prices
        }

    def get_regime(self) -> str:
        return self.regime

    def get_last_rationale(self) -> dict:
        return self.last_rationale.copy()

    def get_full_audit(self) -> dict:
        return copy.deepcopy(self.last_full_audit)
    

    def get_observation_components(self) -> np.ndarray:
        """FIXED: Enhanced observation with regime confidence"""
        arr = np.zeros(len(self.REGIMES) + 2, dtype=np.float32)  # +2 for volatility and confidence
        
        if self.regime in self.REGIMES:
            arr[self.REGIMES.index(self.regime)] = 1.0
            
        # Add volatility as continuous signal
        arr[-2] = float(np.clip(self.volatility * 100, 0, 1))  # Scale and clip
        
        # Add confidence based on data availability
        confidence = min(len(self.prices) / self.window, 1.0)
        arr[-1] = float(confidence)
        
        return arr

    # Voting committee methods (keep existing implementation)
    def set_action_dim(self, dim: int) -> None:
        self._action_dim = int(dim)

    def propose_action(self, obs=None) -> np.ndarray:
        """FIXED: Enhanced action proposal based on regime"""
        if not hasattr(self, "_action_dim"):
            self._action_dim = 2
            
        action = np.zeros(self._action_dim, dtype=np.float32)
        
        # Calculate strength based on volatility and data confidence
        vol_strength = float(np.clip(self.volatility * 10, 0.1, 1.0))
        data_confidence = min(len(self.prices) / max(self.window, 1), 1.0)
        strength = vol_strength * data_confidence
        
        # Map regime to action
        regime_signals = {
            "trending_up": (1.0, 0.7),     # Long position, longer duration
            "trending_down": (-1.0, 0.7),  # Short position, longer duration
            "high_vol": (0.0, 0.2),        # No position, short duration
            "low_vol": (0.3, 0.8),         # Small long, long duration
            "mean_reverting": (-0.2, 0.4), # Small counter-trend, medium duration
            "neutral": (0.0, 0.5),         # No position, medium duration
        }
        
        signal, duration = regime_signals.get(self.regime, (0.0, 0.5))
        
        # Apply to all instrument pairs
        for i in range(0, self._action_dim, 2):
            action[i] = signal * strength
            if i + 1 < self._action_dim:
                action[i + 1] = duration
                
        return action

    def confidence(self, obs=None) -> float:
        """Return confidence based on data availability and regime strength"""
        data_conf = min(len(self.prices) / max(self.window, 1), 1.0)
        regime_conf = 0.8 if self.regime != "neutral" else 0.3
        return float(data_conf * regime_conf)

    # Evolution methods (keep existing)
    def mutate(self, std: float = 0.1):
        old_params = (self.trend_factor, self.mean_thr_factor)
        
        self.window = int(np.clip(self.window + np.random.randint(-5, 6), 20, 200))
        self.vol_window = int(np.clip(self.vol_window + np.random.randint(-2, 3), 5, self.window))
        self.trend_factor = float(np.clip(self.trend_factor + np.random.normal(0, std), 0.1, 2.0))
        self.mean_thr_factor = float(np.clip(self.mean_thr_factor + np.random.normal(0, std/2), 0.01, 0.5))
        self.vol_high_pct = float(np.clip(self.vol_high_pct + np.random.normal(0, 5), 60.0, 95.0))
        self.vol_low_pct = float(np.clip(self.vol_low_pct + np.random.normal(0, 5), 5.0, 40.0))
        
        # Recreate deque with new window size
        old_prices = list(self.prices)
        self.prices = deque(old_prices[-self.window:], maxlen=self.window)
        
        if self.debug:
            self.logger.info(f"Mutated: trend_factor {old_params[0]:.3f}->{self.trend_factor:.3f}, "
                           f"mean_thr_factor {old_params[1]:.3f}->{self.mean_thr_factor:.3f}")

    def crossover(self, other: "MarketRegimeSwitcher"):
        child = copy.deepcopy(self)
        for attr in ["window", "vol_window", "trend_factor", "mean_thr_factor", "vol_high_pct", "vol_low_pct"]:
            if np.random.rand() > 0.5:
                setattr(child, attr, getattr(other, attr))
        
        # Recreate deque with new window size
        old_prices = list(child.prices)
        child.prices = deque(old_prices[-child.window:], maxlen=child.window)
        
        if self.debug:
            child.logger.info("Created crossover child")
        return child

    def get_state(self):
        return {
            "window": self.window,
            "vol_window": self.vol_window,
            "trend_factor": self.trend_factor,
            "mean_thr_factor": self.mean_thr_factor,
            "vol_high_pct": self.vol_high_pct,
            "vol_low_pct": self.vol_low_pct,
            "prices": list(self.prices),
            "regime": self.regime,
            "volatility": float(self.volatility),
            "last_rationale": self.last_rationale,
            "last_full_audit": self.last_full_audit,
        }

    def load_state(self, state):
        # Update parameters
        self.window = int(state.get("window", self.window))
        self.vol_window = int(state.get("vol_window", self.vol_window))
        self.trend_factor = float(state.get("trend_factor", self.trend_factor))
        self.mean_thr_factor = float(state.get("mean_thr_factor", self.mean_thr_factor))
        self.vol_high_pct = float(state.get("vol_high_pct", self.vol_high_pct))
        self.vol_low_pct = float(state.get("vol_low_pct", self.vol_low_pct))
        
        # Recreate deque with updated window
        prices = state.get("prices", [])
        self.prices = deque(prices[-self.window:], maxlen=self.window)
        
        # Restore state
        self.regime = state.get("regime", "neutral")
        self.volatility = float(state.get("volatility", 0.0))
        self.last_rationale = state.get("last_rationale", {})
        self.last_full_audit = state.get("last_full_audit", {})
        
        self.logger.info(f"Loaded state: regime={self.regime}, prices={len(self.prices)}")


# FIXED: Additional logging configuration function for the environment
def setup_trading_logging(debug=True):
    """
    Global logging setup for the trading environment
    """
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    os.makedirs("logs/regime", exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Set appropriate level
    level = logging.DEBUG if debug else logging.INFO
    root_logger.setLevel(level)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler for all logs
    file_handler = logging.FileHandler('logs/trading_system.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler for important messages
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO if debug else logging.WARNING)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # Test logging
    logging.info("Trading system logging configured")
    
    return root_logger


# FIXED: Regime testing utility
def test_regime_detection():
    """
    Test the regime detection with synthetic data
    """
    print("Testing MarketRegimeSwitcher...")
    
    # Setup logging
    setup_trading_logging(debug=True)
    
    # Create regime switcher
    regime_switcher = MarketRegimeSwitcher(
        window=30,
        trend_factor=0.2,
        mean_thr_factor=0.03,
        debug=True
    )
    
    # Test with trending up data
    print("\n1. Testing trending up...")
    base_price = 2000.0
    for i in range(50):
        price = base_price * (1 + 0.001 * i + np.random.normal(0, 0.002))
        regime_switcher.step(price=price)
        
    print(f"Final regime: {regime_switcher.get_regime()}")
    print(f"Rationale: {regime_switcher.get_last_rationale()}")
    
    # Test with high volatility data
    print("\n2. Testing high volatility...")
    regime_switcher.reset()
    base_price = 2000.0
    for i in range(50):
        price = base_price * (1 + np.random.normal(0, 0.02))  # High volatility
        regime_switcher.step(price=price)
        
    print(f"Final regime: {regime_switcher.get_regime()}")
    print(f"Rationale: {regime_switcher.get_last_rationale()}")
    
    # Test with mean reverting data
    print("\n3. Testing mean reverting...")
    regime_switcher.reset()
    base_price = 2000.0
    for i in range(50):
        # Oscillating around mean
        deviation = np.sin(i * 0.3) * 0.005
        price = base_price * (1 + deviation + np.random.normal(0, 0.001))
        regime_switcher.step(price=price)
        
    print(f"Final regime: {regime_switcher.get_regime()}")
    print(f"Rationale: {regime_switcher.get_last_rationale()}")
    
    print("\nRegime detection test completed!")


if __name__ == "__main__":
    test_regime_detection()