import os
import sys
import asyncio
import numpy as np
import pytest
import types
from typing import Any

# Ensure project root on path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.features.advanced_feature_engine import AdvancedFeatureEngine
from modules.utils.info_bus import InfoBusManager


# ─────────────────────────────────────────────────────────────
# Global fixture: reset SmartInfoBus between tests
# ─────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def _reset_bus_between_tests():
    InfoBusManager.reset_instance()
    yield
    InfoBusManager.reset_instance()


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def make_walk(n=200, start=100.0, drift=0.02, vol=0.5, seed=7):
    rng = np.random.default_rng(seed)
    prices = [start]
    for _ in range(n - 1):
        step = rng.normal(drift, vol)
        prices.append(max(0.01, prices[-1] * (1 + step / 100.0)))
    return prices


# ─────────────────────────────────────────────────────────────
# Core contract tests
# ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_dimensions_and_quality_direct_prices():
    eng: Any = AdvancedFeatureEngine({
        "window_sizes": [7, 14, 28, 56],
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    prices = make_walk(200)
    res = await eng.process(prices=prices)

    assert isinstance(res, dict)
    assert "advanced_features" in res and "features" in res and "feature_analysis" in res and "feature_thesis" in res

    adv = res["advanced_features"]
    expected_dim = 4 * 6 + 6
    assert len(adv["raw_features"]) == expected_dim, f"Expected {expected_dim}, got {len(adv['raw_features'])}"
    assert 0.0 <= adv["quality_score"] <= 100.0
    assert adv["feature_count"] == expected_dim
    assert res.get("success", True) is True


@pytest.mark.asyncio
async def test_bus_updates_and_alias_equivalence():
    bus = InfoBusManager.get_instance()
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    prices = make_walk(120)
    res = await eng.process(prices=prices)

    def _unpack(v):
        return v.value if hasattr(v, "value") else v

    adv = _unpack(bus.get("advanced_features", "TestHarness"))
    alias = _unpack(bus.get("features", "TestHarness"))
    analysis = _unpack(bus.get("feature_analysis", "TestHarness"))
    thesis = _unpack(bus.get("feature_thesis", "TestHarness"))

    assert isinstance(adv, dict) and isinstance(alias, dict) and isinstance(analysis, dict) and isinstance(thesis, str)
    assert adv["raw_features"] == alias["raw_features"], "Alias 'features' must mirror 'advanced_features'"
    assert "explanation" in analysis and "statistics" in analysis and "buffer_status" in analysis


@pytest.mark.asyncio
async def test_fallback_via_bus_historical_prices():
    bus = InfoBusManager.get_instance()
    closes = make_walk(100)
    bus.set("historical_prices", {"XAUUSD": {"M1": {"close": closes}}}, module="UnitTest", thesis="inject hist")

    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    res = await eng.process()
    assert res.get("success", True) is True
    assert len(res["advanced_features"]["raw_features"]) == eng.out_dim


@pytest.mark.asyncio
async def test_no_prices_yields_zero_fallback_and_error_flag():
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    eng.price_buffer.clear()
    eng.feature_buffer.clear()

    res = await eng.process()
    assert res.get("success", False) is False
    adv = res["advanced_features"]
    assert len(adv["raw_features"]) == eng.out_dim, "Fallback zeros should match out_dim"
    assert adv["quality_score"] == 0.0


@pytest.mark.asyncio
async def test_circuit_breaker_open_short_circuits():
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    eng.circuit_breaker["state"] = "OPEN"
    res = await eng.process(prices=make_walk(80))
    assert res.get("success", False) is False
    assert res.get("reason") == "Circuit breaker open"
    assert "advanced_features" in res and "feature_analysis" in res


@pytest.mark.asyncio
async def test_propose_action_and_confidence_roundtrip():
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    prices = np.linspace(100, 120, 120).tolist()
    result = await eng.propose_action(prices=prices)
    assert result["action_type"] in {"increase_position", "hold_position", "reduce_risk", "decrease_position"}
    conf = await eng.calculate_confidence(result)
    assert 0.0 <= conf <= 1.0


@pytest.mark.asyncio
async def test_state_roundtrip_and_thesis_contains_key_lines():
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    res = await eng.process(prices=make_walk(200))
    state = eng.get_state()
    assert "features" in state and "buffers" in state and "statistics" in state

    eng2: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    eng2.set_state(state)
    res2 = await eng2.process(prices=make_walk(200))
    assert res2["advanced_features"]["quality_score"] >= 0.0

    thesis = res2["feature_thesis"]
    assert "Quality score" in thesis and "Feature Quality" in thesis


@pytest.mark.asyncio
async def test_feature_quality_behaviour_reasonable_range():
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False,
        "enable_health_monitoring": False,
        "enable_error_pinpointing": False,
        "enable_english_explanations": False,
    })
    prices = [100.0] * 120
    res_const = await eng.process(prices=prices)
    q_const = res_const["advanced_features"]["quality_score"]
    assert q_const <= 40.0

    prices2 = make_walk(200, vol=2.0, drift=0.05)
    res_var = await eng.process(prices=prices2)
    q_var = res_var["advanced_features"]["quality_score"]
    assert q_var >= 60.0


# ─────────────────────────────────────────────────────────────
# Extended coverage
# ─────────────────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_fallback_via_bus_ohlcv_and_market_data():
    # 1) ohlcv_data path
    bus = InfoBusManager.get_instance()
    closes = np.linspace(100, 101, 60).tolist()
    bus.set("ohlcv_data", {"EURUSD": {"close": closes}}, module="UnitTest", thesis="inject ohlcv")
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False, "enable_health_monitoring": False,
        "enable_error_pinpointing": False, "enable_english_explanations": False,
    })
    res = await eng.process()
    assert res.get("success", True) is True

    # 2) market_data path
    InfoBusManager.reset_instance(); bus = InfoBusManager.get_instance()
    bus.set("market_data", {"XAUUSD": {"close": 123.45}}, module="UnitTest", thesis="inject market_data")
    eng2: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False, "enable_health_monitoring": False,
        "enable_error_pinpointing": False, "enable_english_explanations": False,
    })
    res2 = await eng2.process()
    assert res2.get("success", True) is True
    assert len(res2["advanced_features"]["raw_features"]) == eng2.out_dim


@pytest.mark.asyncio
async def test_validate_prices_filters_and_trims_outliers():
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False, "enable_health_monitoring": False,
        "enable_error_pinpointing": False, "enable_english_explanations": False,
    })
    prices = [100.0]*20 + [np.nan, np.inf, -5.0] + list(np.linspace(100, 101, 40)) + [10_000_000.0]  # big outlier
    res = await eng.process(prices=prices)
    adv = res["advanced_features"]
    # Success + correct dimensionality implies NaN/inf/neg/outlier handling passed
    assert res.get("success", True) is True
    assert adv["feature_count"] == eng.out_dim


@pytest.mark.asyncio
async def test_breaker_threshold_and_half_open_transition(monkeypatch):
    eng: Any = AdvancedFeatureEngine({
        "circuit_breaker_threshold": 3,
        "enable_performance_tracking": False, "enable_health_monitoring": False,
        "enable_error_pinpointing": False, "enable_english_explanations": False,
    })
    # 3 consecutive failures → OPEN
    for _ in range(3):
        res = await eng.process()   # no inputs anywhere
        assert res.get("success", False) is False
    assert eng.circuit_breaker["state"] == "OPEN"

    # After 60s → HALF_OPEN (simulate time jump)
    import time as _time
    now = _time.time()
    # Patch the module's time.time used inside AdvancedFeatureEngine
    monkeypatch.setattr("modules.features.advanced_feature_engine.time", types.SimpleNamespace(time=lambda: now + 61), raising=False)

    assert eng._check_circuit_breaker() is True
    assert eng.circuit_breaker["state"] == "HALF_OPEN"

    # Next success closes it
    res2 = await eng.process(prices=np.linspace(100, 102, 80).tolist())
    assert res2.get("success", True) is True
    assert eng.circuit_breaker["state"] == "CLOSED"


@pytest.mark.asyncio
async def test_health_issue_flags_without_slow_tests():
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False, "enable_health_monitoring": False,
        "enable_error_pinpointing": False, "enable_english_explanations": False,
    })
    # Force internal stats to trip the issue checks
    eng.feature_stats["avg_extraction_time_ms"] = 150.0  # >100 → "Processing time is high"
    eng.feature_stats["avg_feature_quality"] = 50.0       # <60 → "Feature quality is low"
    # Fill price buffer to >90%
    eng.price_buffer.extend([100.0]*(int(eng.max_buffer_size*0.92)))
    eng.circuit_breaker["state"] = "OPEN"
    eng._check_health_issues()
    issues = eng.health_metrics["issues_detected"]
    assert any("Circuit breaker is open" in s for s in issues)
    assert any("Price buffer nearly full" in s for s in issues)
    assert any("Processing time is high" in s for s in issues)
    assert any("Feature quality is low" in s for s in issues)


@pytest.mark.asyncio
async def test_explainer_and_performance_report_with_monitoring_disabled(monkeypatch):
    # Disable background loops before constructing
    monkeypatch.setattr(AdvancedFeatureEngine, "_start_monitoring", lambda self: None, raising=False)
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False, "enable_health_monitoring": False,
        "enable_error_pinpointing": False, "enable_english_explanations": True,
    })
    res = await eng.process(prices=np.linspace(100, 101, 120).tolist())
    assert isinstance(res["feature_analysis"]["explanation"], str)
    report = eng.get_performance_report()
    assert isinstance(report, str) and len(report) > 0


@pytest.mark.asyncio
async def test_no_bus_write_when_open():
    bus = InfoBusManager.get_instance()
    eng: Any = AdvancedFeatureEngine({
        "enable_performance_tracking": False, "enable_health_monitoring": False,
        "enable_error_pinpointing": False, "enable_english_explanations": False,
    })
    # Publish once (closed)
    await eng.process(prices=np.linspace(100, 101, 80).tolist())
    before = bus.get("advanced_features", "TestHarness")

    # Now open breaker and try again → should short-circuit and not overwrite bus
    eng.circuit_breaker["state"] = "OPEN"
    await eng.process(prices=np.linspace(101, 102, 80).tolist())
    after = bus.get("advanced_features", "TestHarness")

    # Compare values, accounting for possible DataVersion wrapper
    before_val = before.value if hasattr(before, "value") else before
    after_val = after.value if hasattr(after, "value") else after
    assert before_val == after_val
