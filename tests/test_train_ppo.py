# """
# End-to-end unit checks for the modular trading AI project.

# Run with:
#     pytest -q
# """
# from __future__ import annotations

# import importlib
# import types
# from pathlib import Path
# from typing import List
# from unittest import mock

# import numpy as np
# import pytest

# from modules.memory.memory import MistakeMemory
# from modules.strategy import StrategyGenomePool
# from modules.strategy.voting import CollusionAuditor

# # ---------------------------------------------------------------------------#
# # 1.  Dynamically stub MetaTrader5 before any app modules import it
# # ---------------------------------------------------------------------------#
# class _FakeMT5:
#     TIMEFRAME_H1 = 1
#     TIMEFRAME_H4 = 2
#     TIMEFRAME_D1 = 3

#     def initialize(self, *_, **__):
#         return True

#     def account_info(self):
#         return types.SimpleNamespace(balance=10_000.0, equity=10_000.0, margin=0.0)

#     def positions_get(self):
#         return []

#     def copy_rates_range(self, symbol, timeframe, _from, _to):
#         # 30 × 24 = 720 rows
#         n = 240 if timeframe == self.TIMEFRAME_H4 else 720
#         fake = np.zeros(
#             n,
#             dtype=[
#                 ("time", "<i8"),
#                 ("open", "<f8"),
#                 ("high", "<f8"),
#                 ("low", "<f8"),
#                 ("close", "<f8"),
#                 ("tick_volume", "<i8"),
#                 ("spread", "<i4"),
#                 ("real_volume", "<i8"),
#             ],
#         )
#         fake["high"] = 5.0
#         fake["low"] = 4.0
#         return fake


# mt5_stub = _FakeMT5()
# modules_to_patch = {
#     "MetaTrader5": mt5_stub,
#     # Some sub-modules do `import MetaTrader5 as mt5`
#     "MetaTrader5 as mt5": mt5_stub,
# }
# with mock.patch.dict("sys.modules", modules_to_patch):
#     # Import project packages AFTER stubbing MT5

#     from modules.compliance import ComplianceModule
 
#     from modules.utils.info_bus import InfoBus
#     from live.state_backend import StateBackend
#     from envs.ppo_env import EnhancedTradingEnv

# # ---------------------------------------------------------------------------#
# # 2.  Tests
# # ---------------------------------------------------------------------------#
# def test_mistake_memory_clustering():
#     mm = MistakeMemory(interval=1, n_clusters=2, max_records=10, debug=False)
#     losers = [
#         {"pnl": -5, "features": [1.0, 2.0, 3.0]},
#         {"pnl": -3, "features": [1.0, 2.0, 4.0]},
#         {"pnl": -1, "features": [1.1, 2.1, 3.1]},
#     ]
#     mm.step(trades=losers)
#     mm.step(episode_done=True)
#     pen = mm.cluster_match_penalty(np.array([1.05, 2.05, 3.05]))
#     assert 0.0 <= pen <= 1.0
#     assert len(mm._records) == len(losers)


# def test_strategy_genome_evolution():
#     sgp = StrategyGenomePool(population_size=10, debug=False)
#     sgp.evaluate_population(lambda g: float(np.sum(g)))
#     pre_pop = sgp.population.copy()
#     sgp.evolve_strategies()
#     assert sgp.population.shape == pre_pop.shape
#     # After evolution, fitness reset to zeros
#     assert np.allclose(sgp.fitness, 0.0)
#     # Diversity: mean pairwise distance > 0
#     div = sgp.get_observation_components()[-1]
#     assert div > 0.0


# def test_compliance_basic_flags():
#     bus: InfoBus = {
#         "current_price": 10.0,
#         "risk": {
#             "balance": 1000.0,
#             "equity": 1000.0,
#             "margin_used": 30.0,
#             "max_drawdown": 0.1,
#             "open_positions": [],
#         },
#         "raw_action": 1,
#         "extras": {"symbol": "RUB", "size": 1},
#     }
#     comp = ComplianceModule()
#     ok = comp.step(bus)
#     assert not ok
#     assert "prohibited" in comp.last_flags[0].lower()


# def test_collusion_detection():
#     aud = CollusionAuditor(window=5, corr_threshold=0.9)
#     for _ in range(5):
#         aud.add_votes(
#             [
#                 {"module": "A", "action": 1, "confidence": 0.9},
#                 {"module": "B", "action": 1, "confidence": 0.8},
#                 {"module": "C", "action": 0, "confidence": 0.5},
#             ]
#         )
#     suspects = aud.check()
#     assert set(suspects) == {"A", "B"}


# def test_state_backend_snapshot_and_vol():
#     sb = StateBackend()
#     snap = sb.get_account_snapshot()
#     assert {"balance", "equity", "drawdown", "positions"} <= snap.keys()

#     vol = sb.compute_vol_profile(symbol="EURUSD", tf="H4")
#     assert vol["timeframe"] == "H4"
#     assert len(vol["atr"]) > 0
#     # Ensure caching: second call should return same object (same id)
#     vol2 = sb.compute_vol_profile(symbol="EURUSD", tf="H4")
#     assert id(vol["atr"]) == id(vol2["atr"])


# def test_environment_reset_deterministic(tmp_path: Path):
#     env = EnhancedTradingEnv(
#         atr_period=14,
#         dynamic_max_steps=True,
#     )
#     obs1, _ = env.reset(seed=123)
#     obs2, _ = env.reset(seed=123)
#     obs3, _ = env.reset(seed=456)
#     assert np.array_equal(obs1, obs2)
#     assert not np.array_equal(obs1, obs3)


# # ---------------------------------------------------------------------------#
# # 3.  PyTest entry-point sanity
# # ---------------------------------------------------------------------------#
# if __name__ == "__main__":
#     import pytest as _pytest

#     _pytest.main([__file__])
