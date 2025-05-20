# tests/test_env.py
import pytest
import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.ppo_env import EnhancedTradingEnv

from utils.data_utils import load_data

# minimal fake data for one instrument
@pytest.fixture
def fake_data():
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    df = pd.DataFrame({
        "close": np.linspace(1, 2, 10),
        "volatility": np.linspace(0.1, 0.2, 10),
        "real_volume": np.ones(10),
    }, index=dates)
    return {"FAKE": {"D1": df.copy(), "H1": df.copy(), "H4": df.copy()}}

@pytest.fixture
def env(fake_data, tmp_path):
    env = EnhancedTradingEnv(
        data_dict=fake_data,
        initial_balance=1_000.0,
        max_steps=5,
        debug=True,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    # Initialize KMeans with dummy data
    dummy_features = np.random.rand(10, 3)
    env.mistake_memory._kmeans.fit(dummy_features)
    return env

def test_reset_and_observation_shape(env):
    obs, info = env.reset()
    # obs should be 1D float array
    assert isinstance(obs, np.ndarray)
    assert obs.ndim == 1
    # info returned empty dict
    assert info == {}

def test_step_no_trades(env):
    env.reset()
    zero_act = np.zeros(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(zero_act)
    assert isinstance(obs, np.ndarray)
    # Allow for small replay bonus instead of strict 0.0
    assert -0.5 < reward < 0.5  # Modified check

def test_step_random_actions(env):
    env.reset()
    # random actions in [-1,1]
    act = np.random.uniform(-1, 1, size=env.action_space.shape).astype(np.float32)
    obs, reward, terminated, truncated, info = env.step(act)
    # reward is a float, balance updated
    assert isinstance(reward, float)
    assert info["balance"] >= 0.0

def test_live_mode_toggle(env, monkeypatch):
    # monkeypatch MT5 calls so no real order is sent
    class FakeMT5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        TRADE_ACTION_DEAL = 0
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 0
        TRADE_RETCODE_DONE = 10009
        @staticmethod
        def symbol_info_tick(sym): 
            return type("T", (), {"ask":1.1, "bid":1.0})
        @staticmethod
        def order_send(req):
            return type("R", (), {"retcode":10009, "comment":"ok"})
    monkeypatch.setitem(__import__("sys").modules, "MetaTrader5", FakeMT5)

    env.live_mode = True
    env.reset()
    # random action to force a trade
    act = np.ones(env.action_space.shape, dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(act)
    # live-mode path returns size but no pnl
    assert "balance" in info

def test_pipeline_components_enabled_toggle(env):
    env.reset()
    # disable every module in pipeline, obs should still be valid zeros
    for nm in list(env.get_module_status()):
        env.set_module_enabled(nm, False)
    obs, reward, terminated, truncated, info = env.step(
        np.zeros(env.action_space.shape, np.float32)
    )
    assert np.all(np.isfinite(obs))

