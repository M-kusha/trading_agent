#!/usr/bin/env python3
"""
Test script for Risk Integration
Tests the risk API endpoints and data flow
"""

import json
import time
import requests
from modules.utils.info_bus import InfoBusManager

def setup_test_risk_data():
    """Set up sample risk data in InfoBus for testing"""
    print("Setting up test risk data...")

    bus = InfoBusManager.get_instance()

    # Risk overview data
    bus.set('risk_metrics', {
        'current_drawdown': 0.12,
        'max_drawdown': 0.18,
        'sharpe_ratio': 1.45,
        'win_rate': 0.68,
        'var_95': 0.05,
        'var_99': 0.08,
        'volatility_ratio': 1.25,
        'risk_budget_used': 0.42
    }, module='BackendAPI')

    bus.set('risk_level', 'ELEVATED', module='BackendAPI')
    bus.set('risk_scale', 0.85, module='BackendAPI')

    # Anomaly detector data
    bus.set('anomaly_detection', {
        'status': 'active',
        'processing_mode': 'ENHANCED'
    }, module='AnomalyDetector')

    bus.set('anomaly_alerts', [
        {
            'timestamp': time.time(),
            'severity': 'warning',
            'message': 'Unusual trading pattern detected',
            'type': 'pattern_anomaly'
        },
        {
            'timestamp': time.time() - 300,
            'severity': 'critical',
            'message': 'High volatility anomaly in EURUSD',
            'type': 'volatility_anomaly'
        }
    ], module='AnomalyDetector')

    bus.set('anomaly_score', 0.75, module='AnomalyDetector')
    bus.set('anomaly_threshold', 0.8, module='AnomalyDetector')
    bus.set('detection_mode', 'ACTIVE', module='AnomalyDetector')

    bus.set('anomaly_history', [
        {'timestamp': time.time() - i * 60, 'score': 0.3 + (i % 3) * 0.2}
        for i in range(10)
    ], module='AnomalyDetector')

    # Compliance data
    bus.set('compliance', {
        'status': 'active',
        'violations_count': 2
    }, module='BackendAPI')

    bus.set('trade_compliance', {
        'status': 'monitoring',
        'checks_passed': 95,
        'total_checks': 100
    }, module='Compliance')

    bus.set('compliance_violations', [
        {
            'timestamp': time.time(),
            'severity': 'warning',
            'message': 'Daily trade limit approaching: 87/100',
            'type': 'daily_limit'
        },
        {
            'timestamp': time.time() - 600,
            'severity': 'critical',
            'message': 'Position size exceeded max risk: 0.22 > 0.20',
            'type': 'position_risk'
        }
    ], module='Compliance')

    bus.set('risk_limits', {
        'max_leverage': 30.0,
        'max_position_risk': 0.20,
        'max_daily_trades': 100
    }, module='Compliance')

    bus.set('position_compliance', {
        'max_position_risk': 0.20,
        'current_exposure': 0.15,
        'compliance_score': 0.92
    }, module='Compliance')

    bus.set('daily_limits', {
        'max_daily_trades': 100,
        'current_trades': 87,
        'limit_utilization': 0.87
    }, module='Compliance')

    # Drawdown rescue data
    bus.set('drawdown_status', {
        'current_level': 0.12,
        'warning_level': 0.15,
        'critical_level': 0.25
    }, module='DrawdownRescue')

    bus.set('rescue_status', {
        'active': False,
        'triggered_count': 3,
        'last_trigger': time.time() - 3600
    }, module='DrawdownRescue')

    bus.set('drawdown_analysis', {
        'velocity': -0.02,
        'acceleration': 0.01,
        'trend': 'deteriorating'
    }, module='DrawdownRescue')

    bus.set('recovery_progress', {
        'recovery_ratio': 0.35,
        'time_to_recover': 1800
    }, module='DrawdownRescue')

    bus.set('rescue_triggers', [
        {
            'timestamp': time.time() - 3600,
            'trigger_type': 'velocity',
            'severity': 'warning'
        }
    ], module='DrawdownRescue')

    bus.set('velocity_analysis', {
        'velocity_trend': 'Improving',
        'current_velocity': -0.01
    }, module='DrawdownRescue')

    # Execution quality data
    bus.set('execution_quality', {
        'overall_score': 0.85,
        'recent_performance': 'good'
    }, module='ExecutionQualityMonitor')

    bus.set('execution_metrics', {
        'fill_rate': 0.97,
        'average_slippage': 0.0008,
        'average_latency': 45
    }, module='ExecutionQualityMonitor')

    bus.set('execution_alerts', [
        {
            'timestamp': time.time(),
            'message': 'Increased slippage detected on GBPUSD',
            'severity': 'warning'
        }
    ], module='ExecutionQualityMonitor')

    bus.set('slippage_analysis', {
        'average_slippage': 0.0008,
        'max_slippage': 0.0025,
        'slippage_trend': 'stable'
    }, module='ExecutionQualityMonitor')

    bus.set('latency_metrics', {
        'average_latency': 45,
        'max_latency': 120,
        'p95_latency': 78
    }, module='ExecutionQualityMonitor')

    bus.set('fill_rate_analysis', {
        'current_fill_rate': 0.97,
        'target_fill_rate': 0.95,
        'fill_rate_trend': 'stable'
    }, module='ExecutionQualityMonitor')

    bus.set('execution_vote', 'CAUTION', module='ExecutionQualityMonitor')
    bus.set('quality_score', 0.85, module='ExecutionQualityMonitor')

    # Portfolio risk data
    bus.set('portfolio_risk', {
        'total_var': 0.08,
        'concentration_risk': 0.15
    }, module='PortfolioRiskSystem')

    bus.set('correlation_matrix', {
        'EURUSD_GBPUSD': 0.75,
        'EURUSD_USDJPY': -0.45,
        'GBPUSD_USDJPY': -0.62
    }, module='BackendAPI')

    bus.set('correlation_risk', {
        'risk_score': 0.68,
        'high_correlation_pairs': 3
    }, module='BackendAPI')

    bus.set('position_risk', {
        'active_positions': 5,
        'total_exposure': 0.85
    }, module='PortfolioRiskSystem')

    bus.set('var_analysis', {
        'current_var': 0.08,
        'var_95': 0.05,
        'var_99': 0.08
    }, module='PortfolioRiskSystem')

    bus.set('exposure_analysis', {
        'total_exposure': 0.85,
        'max_exposure': 1.0
    }, module='PortfolioRiskSystem')

    bus.set('diversification_metrics', {
        'diversification_score': 0.72,
        'asset_class_count': 3
    }, module='PortfolioRiskSystem')

    # Dynamic risk controller data
    bus.set('dynamic_risk', {
        'scaling_active': True,
        'adjustments_today': 8
    }, module='DynamicRiskController')

    bus.set('risk_scaling', {
        'current_scale': 0.85,
        'target_scale': 0.90,
        'adjustment_reason': 'elevated_volatility'
    }, module='DynamicRiskController')

    bus.set('volatility_analysis', {
        'current_volatility': 1.25,
        'normal_volatility': 1.0,
        'volatility_trend': 'increasing'
    }, module='DynamicRiskController')

    bus.set('risk_adjustments', [
        {
            'timestamp': time.time(),
            'old_scale': 1.0,
            'new_scale': 0.85,
            'reason': 'High volatility detected in major pairs'
        },
        {
            'timestamp': time.time() - 1800,
            'old_scale': 0.85,
            'new_scale': 0.75,
            'reason': 'Drawdown threshold exceeded'
        }
    ], module='DynamicRiskController')

    bus.set('control_mode', 'PROTECTIVE', module='DynamicRiskController')

    bus.set('scaling_history', [
        {'timestamp': time.time() - i * 300, 'scale': 1.0 - (i % 4) * 0.05}
        for i in range(20)
    ], module='DynamicRiskController')

    print("Test risk data setup complete!")

def test_risk_endpoints():
    """Test the risk API endpoints"""
    print("\nTesting risk API endpoints...")

    base_url = "http://localhost:8000/api/risk"
    endpoints = [
        "/overview",
        "/anomalies",
        "/compliance",
        "/drawdown",
        "/execution",
        "/portfolio",
        "/dynamic",
        "/alerts"
    ]

    results = {}

    for endpoint in endpoints:
        try:
            url = base_url + endpoint
            print(f"Testing {url}...")

            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    print(f"  ✓ Success: {endpoint}")
                    results[endpoint] = "SUCCESS"

                    # Print sample data for verification
                    if endpoint == "/overview":
                        print(f"    Risk Level: {data.get('risk_level', 'N/A')}")
                        print(f"    Current DD: {(data.get('current_drawdown', 0) * 100):.2f}%")
                    elif endpoint == "/anomalies":
                        anomalies = data.get('anomalies', {})
                        print(f"    Anomaly Score: {anomalies.get('anomaly_score', 0):.3f}")
                        print(f"    Active Alerts: {len(anomalies.get('anomaly_alerts', []))}")
                    elif endpoint == "/compliance":
                        compliance = data.get('compliance', {})
                        violations = compliance.get('compliance_violations', [])
                        print(f"    Violations: {len(violations)}")

                else:
                    print(f"  ✗ Failed: {endpoint} - {data.get('error', 'Unknown error')}")
                    results[endpoint] = f"FAILED: {data.get('error', 'Unknown error')}"
            else:
                print(f"  ✗ HTTP Error: {endpoint} - Status {response.status_code}")
                results[endpoint] = f"HTTP_ERROR: {response.status_code}"

        except requests.exceptions.RequestException as e:
            print(f"  ✗ Connection Error: {endpoint} - {e}")
            results[endpoint] = f"CONNECTION_ERROR: {str(e)}"

    return results

def print_test_results(results):
    """Print formatted test results"""
    print("\n" + "="*50)
    print("RISK INTEGRATION TEST RESULTS")
    print("="*50)

    success_count = 0
    total_count = len(results)

    for endpoint, result in results.items():
        status = "✓ PASS" if result == "SUCCESS" else "✗ FAIL"
        print(f"{endpoint:15} | {status:8} | {result}")
        if result == "SUCCESS":
            success_count += 1

    print("-" * 50)
    print(f"SUMMARY: {success_count}/{total_count} endpoints passed")

    if success_count == total_count:
        print("🎉 ALL TESTS PASSED! Risk integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the backend server and endpoints.")

if __name__ == "__main__":
    print("Risk Integration Test")
    print("=" * 50)

    # Setup test data
    setup_test_risk_data()

    # Test endpoints (requires backend server to be running)
    print("\nNote: Make sure the backend server is running on localhost:8000")
    input("Press Enter to continue with endpoint testing, or Ctrl+C to exit...")

    # Test the endpoints
    results = test_risk_endpoints()

    # Print results
    print_test_results(results)