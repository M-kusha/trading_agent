import logging
import pytest
import pandas as pd
from unittest.mock import patch
from envs.ppo_env import EnhancedTradingEnv
import os

@pytest.fixture
def env():
    """Setup the environment with mock data."""
    # Mock data structure to match what the environment expects
    data_dict = {
        "XAU/USD": {
            "D1": pd.DataFrame({
                "close": [1800, 1810, 1820, 1830, 1840, 1850, 1860],
                "volatility": [0.02, 0.02, 0.03, 0.03, 0.04, 0.02, 0.01],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600],
                "high": [1805, 1815, 1825, 1835, 1845, 1855, 1865],
                "low": [1795, 1805, 1815, 1825, 1835, 1845, 1855],
            }),
            "H1": pd.DataFrame({
                "close": [1805, 1815, 1825, 1835, 1845, 1855, 1865],
                "volatility": [0.01, 0.01, 0.015, 0.02, 0.025, 0.02, 0.02],
                "volume": [200, 220, 240, 260, 280, 300, 320],
                "high": [1810, 1820, 1830, 1840, 1850, 1860, 1870],
                "low": [1800, 1810, 1820, 1830, 1840, 1850, 1860],
            }),
            "H4": pd.DataFrame({
                "close": [1802, 1812, 1822, 1832, 1842, 1852, 1862],
                "volatility": [0.015, 0.015, 0.02, 0.025, 0.03, 0.02, 0.015],
                "volume": [500, 550, 600, 650, 700, 750, 800],
                "high": [1807, 1817, 1827, 1837, 1847, 1857, 1867],
                "low": [1797, 1807, 1817, 1827, 1837, 1847, 1857],
            }),
        },
        "EUR/USD": {
            "D1": pd.DataFrame({
                "close": [1.10, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17],
                "volatility": [0.01, 0.01, 0.02, 0.03, 0.04, 0.02, 0.01],
                "volume": [5000, 5200, 5400, 5600, 5800, 6000, 6200],
                "high": [1.11, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18],
                "low": [1.09, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16],
            }),
            "H1": pd.DataFrame({
                "close": [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17],
                "volatility": [0.005, 0.005, 0.01, 0.015, 0.02, 0.02, 0.025],
                "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600],
                "high": [1.12, 1.13, 1.14, 1.15, 1.16, 1.17, 1.18],
                "low": [1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16],
            }),
            "H4": pd.DataFrame({
                "close": [1.105, 1.115, 1.125, 1.135, 1.145, 1.155, 1.165],
                "volatility": [0.01, 0.01, 0.015, 0.02, 0.025, 0.02, 0.015],
                "volume": [2000, 2100, 2200, 2300, 2400, 2500, 2600],
                "high": [1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.17],
                "low": [1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16],
            }),
        }
    }

    # Initialize the environment with the mock data
    env = EnhancedTradingEnv(data_dict, debug=True)
    env.current_step = 6  # Ensure we use a valid index within the range of the data (0 to 6)
    yield env

    # Cleanup after the test
    if os.path.exists("env.log"):
        os.remove("env.log")


def test_logger_initialized(env):
    """Test if the logger is initialized and the file handler is set up correctly"""
    env.reset()  # Explicitly reset the environment to trigger the reset log
    assert os.path.exists("env.log"), "Log file not created"

    with open("env.log", "r") as log_file:
        log_contents = log_file.read()

    # Check if the log file contains an expected log message
    assert "Environment reset:" in log_contents, "Initial reset log message not found"
    assert "Full observation:" in log_contents, "Full observation log message not found"



def test_logging_in_step(env):
    """Test if logging occurs during the `step()` method"""
    with patch.object(env.logger, 'info') as mock_info, patch.object(env.logger, 'debug') as mock_debug:
        actions = [0.1, 0.2, -0.1, -0.2]  # Sample actions
        env.step(actions)  # Run one step in the environment

        # Check that the logger was called at least once
        mock_info.assert_called()
        mock_debug.assert_called()

        # Check if certain expected logs were recorded
        mock_info.assert_any_call("Calculated reward:")
        mock_info.assert_any_call("SL Multiplier:")



def test_trade_execution_logging(env):
    """Test if trade execution logs are correctly captured"""
    with patch.object(env.logger, 'info') as mock_info:
        actions = [0.5, 0.2, -0.3, -0.4]  # Sample actions
        trades = env.step(actions)  # Perform a step and execute trades

        # Check that trade execution logs are captured
        mock_info.assert_any_call("Proposed trades:")
        if trades:
            mock_info.assert_any_call("Executing trade for XAU/USD")
        mock_info.assert_any_call(f"Executed trade for XAU/USD: {trades}")



def test_logger_formatting(env):
    """Test if the logger's format is correct"""
    with patch.object(env.logger, 'info') as mock_info:
        actions = [0.5, 0.3, -0.2, -0.1]  # Sample actions
        env.step(actions)  # Execute a step

        # Check the logger formatting
        mock_info.assert_any_call("Calculated reward:")
        log_message = mock_info.call_args[0][0]  # Get the actual log message
        assert "INFO" in log_message, "Log message does not have the correct log level"
        assert len(log_message) > 0, "Log message is empty"


import logging
from unittest.mock import patch

@pytest.mark.parametrize("log_level", [logging.DEBUG, logging.INFO, logging.WARNING])
def test_log_level_logging(env, log_level):
    """Test if different log levels are being captured"""
    env.logger.setLevel(log_level)  # Set the logger to a specific level
    
    with patch.object(env.logger, 'log') as mock_log:
        actions = [0.5, -0.3, 0.2, -0.4]  # Sample actions
        env.step(actions)  # Perform step to trigger logs
    
        # Check if the mock_log was called with the correct level
        mock_log.assert_any_call(log_level, "Calculated reward:")

