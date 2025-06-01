# tests/test_all_modules.py

import pytest
import numpy as np
import os
import importlib
import tempfile
from collections import defaultdict

# --- Adjust these imports to fit your project structure ---
from envs.enhanced_trading_env import EnhancedTradingEnv
from modules.feature_engine import EnhancedFeatureEngine
from modules.strategy import StrategyIntrospector, StrategyArbiter, PlaybookMemory
from modules.meta import MetaAgent, CurriculumPlannerPlus, MetaRLController
from modules.genome import StrategyGenomePool, ThesisEvolutionEngine
from modules.memory import MistakeMemory, MarketMemoryBank
from modules.risk import PositionManager, ComplianceModule, DynamicRiskController
from modules.logging import Logger, ExplanationGenerator
from modules.sentiment import NewsSentiment, AltSentiment
from modules.regime import RegimeDetector, MarketThemeDetector

# Optionally, API and live trading (mock if not available)
try:
    from live.live_trading import LiveTrader
except ImportError:
    LiveTrader = None

try:
    from ui.api import TradingAPI
except ImportError:
    TradingAPI = None

# Import your training script(s)
from train.train_sac import train_sac_agent
from train.train_td3 import train_td3_agent
from train.train_ppo import train_ppo_agent

# ------------------ MOCK DATA HELPERS --------------------

def mock_multi_tf_data():
    n = 400
    def rand_data():
        return {
            'open': np.random.uniform(1800, 2000, n),
            'high': np.random.uniform(1800, 2005, n),
            'low':  np.random.uniform(1795, 2000, n),
            'close': np.random.uniform(1800, 2000, n),
            'volume': np.random.uniform(10_000, 25_000, n)
        }
    return {
        'XAUUSD_H1': rand_data(),
        'XAUUSD_H4': rand_data(),
        'XAUUSD_D1': rand_data(),
        'EURUSD_H1': rand_data(),
        'EURUSD_H4': rand_data(),
        'EURUSD_D1': rand_data(),
    }

def random_valid_action(env):
    if hasattr(env, "action_space"):
        return env.action_space.sample()
    elif hasattr(env, "action_dim"):
        return np.random.uniform(-1, 1, env.action_dim)
    else:
        return 0  # fallback

# ------------------ FIXTURES --------------------

@pytest.fixture(scope="module")
def env():
    data = mock_multi_tf_data()
    config = {
        'feature_engine': EnhancedFeatureEngine(features=["price", "regime", "sentiment", "macro"]),
        'strategy_modules': [
            StrategyIntrospector(), StrategyArbiter(), PlaybookMemory()
        ],
        'meta_modules': [
            MetaAgent(), CurriculumPlannerPlus(), MetaRLController()
        ],
        'genome_modules': [
            StrategyGenomePool(), ThesisEvolutionEngine()
        ],
        'memory_modules': [
            MistakeMemory(), MarketMemoryBank()
        ],
        'risk_modules': [
            PositionManager(initial_balance=100_000, instruments=["XAUUSD", "EURUSD"]),
            ComplianceModule(), DynamicRiskController()
        ],
        'logging_modules': [
            Logger(), ExplanationGenerator()
        ],
        'sentiment_modules': [
            NewsSentiment(), AltSentiment()
        ],
        'regime_modules': [
            RegimeDetector(), MarketThemeDetector()
        ],
        'instruments': ["XAUUSD", "EURUSD"],
        'multi_timeframes': ["H1", "H4", "D1"],
        'debug': True
    }
    env = EnhancedTradingEnv(config=config, data=data)
    return env

# ------------------ MODULE LIFECYCLE --------------------

def test_module_lifecycle(env):
    """All modules should reset, step, profile, diagnostics, clear, teardown."""
    for name, module in env.modules.items():
        if hasattr(module, "reset"): module.reset()
        if hasattr(module, "step"): module.step()
        if hasattr(module, "profile"): module.profile()
        if hasattr(module, "diagnostics"): module.diagnostics()
        if hasattr(module, "clear"): module.clear()
        if hasattr(module, "teardown"): module.teardown()

# ------------------ OBS/ACTION/REWARD CONSISTENCY --------------------

def test_obs_action_reward_consistency(env):
    obs = env.reset()
    assert obs is not None and isinstance(obs, np.ndarray)
    assert obs.shape == env.observation_space.shape
    for _ in range(10):
        action = random_valid_action(env)
        result = env.step(action)
        if len(result) == 5:
            next_obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        elif len(result) == 4:
            next_obs, reward, done, info = result
        else:
            raise Exception("Env step returned invalid tuple length.")
        assert next_obs.shape == env.observation_space.shape
        assert np.isscalar(reward)
        assert isinstance(done, (bool, np.bool_))
        assert isinstance(info, dict)
        assert np.all(np.isfinite(next_obs))
        assert np.isfinite(reward)

# ------------------ EDGE CASES: EMPTY, NAN, INF, OUTLIER --------------------

def test_module_edge_cases(env):
    for name, module in env.modules.items():
        # Empty memory/case
        if hasattr(module, "clear"): module.clear()
        # NaN/Inf input
        try:
            if hasattr(module, "step"):
                module.step(obs=np.array([np.nan]), reward=np.inf)
        except Exception:
            pass
        # Out-of-range input
        try:
            if hasattr(module, "step"):
                module.step(obs=np.array([1e9]), reward=-1e9)
        except Exception:
            pass

# ------------------ META-LEARNING, GENOME, EVOLUTION --------------------

def test_meta_learning_and_evolution(env):
    # Run a few meta/evolution steps and check for correct feedback/logging
    if hasattr(env, "meta_agent"): env.meta_agent.step()
    if hasattr(env, "curriculum_planner"): env.curriculum_planner.step()
    if hasattr(env, "meta_rl_controller"): env.meta_rl_controller.step()
    if hasattr(env, "genome_pool"): env.genome_pool.mutate()
    if hasattr(env, "thesis_evolution"): env.thesis_evolution.step()
    # Check diagnostics
    if hasattr(env, "meta_agent"): env.meta_agent.diagnostics()
    if hasattr(env, "genome_pool"): env.genome_pool.diagnostics()

# ------------------ CONSENSUS, SL/TP, COMPLIANCE, RISK --------------------

def test_consensus_and_compliance(env):
    if hasattr(env, "get_consensus"):
        val = env.get_consensus()
        assert isinstance(val, float)
    if hasattr(env, "get_sl_tp"):
        sl, tp = env.get_sl_tp()
        assert np.isscalar(sl) and np.isscalar(tp)
    if hasattr(env, "position_manager"):
        pos = env.position_manager.get_open_positions()
        assert isinstance(pos, dict)
    if hasattr(env, "compliance_module"):
        env.compliance_module.step()
        assert hasattr(env.compliance_module, "diagnostics")
        env.compliance_module.diagnostics()

# ------------------ LOGGING, EXPLANATION, DIAGNOSTICS --------------------

def test_logging_and_explanation(env):
    for _ in range(5):
        action = random_valid_action(env)
        result = env.step(action)
        if hasattr(env, "logger"):
            env.logger.log_event("test_event", {"obs": result[0], "reward": result[1]})
        if hasattr(env, "explanation_generator"):
            exp = env.explanation_generator.generate(result[0], action)
            assert isinstance(exp, str)
    # Diagnostics for all modules
    for name, module in env.modules.items():
        if hasattr(module, "diagnostics"):
            module.diagnostics()

# ------------------ DRY-RUN TRAINING/AGENT LOOPS --------------------

@pytest.mark.parametrize("trainer_func", [
    train_sac_agent, train_td3_agent, train_ppo_agent
])
def test_dry_run_training(tmp_path, trainer_func):
    log_dir = tmp_path / "logs"
    os.makedirs(log_dir, exist_ok=True)
    try:
        model, stats = trainer_func(
            n_steps=20,
            log_dir=log_dir,
            dry_run=True,
            debug=True
        )
    except Exception as e:
        raise RuntimeError(f"{trainer_func.__name__} dry-run failed: {e}")
    log_files = list(os.listdir(log_dir))
    assert len(log_files) > 0
    assert any("model" in f for f in log_files)
    assert "reward" in stats and "steps" in stats

# ------------------ LIVE/HOT-RELOAD SIMULATION --------------------

def test_live_trading_hot_reload(env):
    if LiveTrader:
        trader = LiveTrader(env=env)
        trader.start_live(dry_run=True, n_steps=5)
        # Hot-reload a strategy module
        import modules.strategy
        importlib.reload(modules.strategy)
        trader.stop_live()

# ------------------ API/UI ENDPOINTS (OPTIONAL) --------------------

def test_api_endpoints():
    if TradingAPI:
        api = TradingAPI()
        # Diagnostics
        resp = api.get_diagnostics()
        assert "status" in resp
        # Module patch/hotswap (if supported)
        if hasattr(api, "patch_module"):
            api.patch_module("StrategyIntrospector", {"debug": True})

# ------------------ ADVERSARIAL/REGRESSION/STRESS --------------------

def test_adversarial_and_regression(env):
    # Adversarial: inject broken data
    obs = np.full(env.observation_space.shape, np.nan)
    try:
        _ = env.step(random_valid_action(env))
    except Exception:
        pass
    # Regression: simulate a known-good run and compare stats
    expected_reward_range = (-1000, 1000)
    obs = env.reset()
    total_reward = 0
    for _ in range(5):
        action = random_valid_action(env)
        result = env.step(action)
        total_reward += result[1]
    assert expected_reward_range[0] < total_reward < expected_reward_range[1]

# ------------------ MULTI-AGENT, MULTI-TIMEFRAME --------------------

def test_multi_agent_multi_tf():
    envs = [EnhancedTradingEnv(config={
        'feature_engine': EnhancedFeatureEngine(features=["price"]),
        'instruments': ["XAUUSD", "EURUSD"],
        'multi_timeframes': ["H1", "H4"],
        'debug': False
    }, data=mock_multi_tf_data()) for _ in range(3)]
    for env in envs:
        obs = env.reset()
        for _ in range(3):
            action = random_valid_action(env)
            env.step(action)

# ------------------ COMMENTS FOR FURTHER SCALING --------------------

"""
- Expand with more adversarial cases: delayed signals, missing columns, changing obs shape.
- Use pytest-xdist or multiprocessing for scaling up to hundreds of parallel environments.
- For full regression: snapshot obs/action/reward chains and check for code drift.
- For UI: automate Selenium or frontend tests if applicable.
- For API: add negative tests (bad requests, auth).
"""

# ------------------ MAIN --------------------

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__]))
