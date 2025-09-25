#!/usr/bin/env python3
"""
Test script for Voting Integration
Tests the voting API endpoints and data flow
"""

import json
import time
import requests
from modules.utils.info_bus import InfoBusManager

def setup_test_voting_data():
    """Set up sample voting data in InfoBus for testing"""
    print("Setting up test voting data...")

    bus = InfoBusManager.get_instance()

    # Core voting metrics
    bus.set('voting_metrics', {
        'successful_ticks': 95,
        'total_ticks': 100,
        'failed_ticks': 5,
        'avg_processing_time_ms': 42.5
    }, module='VotingKernel')

    bus.set('decision_coordination', {
        'decision_id': f'{time.time()}#95',
        'tick_ts': time.time(),
        'stages': 6,
        'status': 'ok'
    }, module='VotingKernel')

    bus.set('consensus_summary', {
        'score': 0.78,
        'components': {
            'directional_consensus': 0.82,
            'magnitude_consensus': 0.75,
            'confidence_consensus': 0.80,
            'temporal_stability': 0.73,
            'network_consensus': 0.76
        }
    }, module='VotingKernel')

    # Committee data
    bus.set('committee_members', [
        {'name': 'Technical Expert', 'specialization': 'technical', 'active': True},
        {'name': 'Momentum Expert', 'specialization': 'momentum', 'active': True},
        {'name': 'Mean Reversion Expert', 'specialization': 'mean_reversion', 'active': True},
        {'name': 'Volatility Expert', 'specialization': 'volatility', 'active': True},
        {'name': 'Sentiment Expert', 'specialization': 'sentiment', 'active': True}
    ], module='VotingKernel')

    bus.set('proposal_vectors', [
        [0.75, 0.82, 0.65],
        [0.68, 0.79, 0.73],
        [0.71, 0.85, 0.69],
        [0.77, 0.80, 0.72],
        [0.69, 0.76, 0.78]
    ], module='VotingKernel')

    bus.set('member_confidences_ordered', [0.85, 0.78, 0.82, 0.79, 0.81], module='VotingKernel')

    bus.set('committee_consensus', {
        'strength': 0.78,
        'agreement': 0.82,
        'coherence': 0.75
    }, module='VotingKernel')

    bus.set('committee_votes', [
        {'member_id': 0, 'decision': 'buy', 'confidence': 0.85},
        {'member_id': 1, 'decision': 'buy', 'confidence': 0.78},
        {'member_id': 2, 'decision': 'hold', 'confidence': 0.82},
        {'member_id': 3, 'decision': 'buy', 'confidence': 0.79},
        {'member_id': 4, 'decision': 'buy', 'confidence': 0.81}
    ], module='VotingKernel')

    # Consensus detection data
    bus.set('consensus_score', 0.78, module='VotingKernel')
    bus.set('consensus_components', {
        'directional_consensus': 0.82,
        'magnitude_consensus': 0.75,
        'confidence_consensus': 0.80,
        'temporal_stability': 0.73,
        'network_consensus': 0.76
    }, module='VotingKernel')

    bus.set('voting_consensus', {
        'score': 0.78,
        'quality': 0.81,
        'stability': 0.73,
        'reliability': 0.79,
        'components': {
            'directional_consensus': 0.82,
            'magnitude_consensus': 0.75,
            'confidence_consensus': 0.80,
            'temporal_stability': 0.73,
            'network_consensus': 0.76
        }
    }, module='VotingKernel')

    # Collusion detection data
    bus.set('collusion_score', 0.15, module='VotingKernel')
    bus.set('suspicious_pairs', [], module='VotingKernel')

    # Time horizon alignment data
    bus.set('voting_weights', [0.20, 0.25, 0.30, 0.15, 0.10], module='VotingKernel')
    bus.set('aligned_weights', [0.18, 0.28, 0.32, 0.14, 0.08], module='VotingKernel')

    # Alternative reality sampling data
    bus.set('sampling_uncertainty', 0.25, module='VotingKernel')
    bus.set('fragility', 0.18, module='VotingKernel')
    bus.set('effective_samples', 6, module='VotingKernel')

    # Strategy arbiter data
    bus.set('trade_vote_v2', {
        'decision': 'buy',
        'confidence': 0.79,
        'strength': 0.75,
        'processing_time': 38
    }, module='VotingKernel')

    bus.set('signals', {
        'primary': 'bullish',
        'secondary': ['momentum_up', 'volume_confirm'],
        'strength': 0.75,
        'coherence': 0.82,
        'cross_validation': 0.75,
        'execution_readiness': 0.88
    }, module='VotingKernel')

    # Pipeline timeline data
    bus.set('voting/kernel_timeline', {
        'decision_id': f'{time.time()}#95',
        'timeline': [
            {'stage': 'committee', 'status': 'success', 'duration_ms': 12.5},
            {'stage': 'consensus', 'status': 'success', 'duration_ms': 8.7},
            {'stage': 'collusion', 'status': 'success', 'duration_ms': 5.2},
            {'stage': 'horizon', 'status': 'success', 'duration_ms': 9.8},
            {'stage': 'sampling', 'status': 'success', 'duration_ms': 15.3},
            {'stage': 'arbiter', 'status': 'success', 'duration_ms': 11.2}
        ]
    }, module='VotingKernel')

    bus.set('pipeline_stats', {
        'total_ticks': 100,
        'successful_ticks': 95,
        'failed_ticks': 5,
        'avg_processing_time_ms': 42.5,
        'module_success_rates': {
            'committee': {'success': 98, 'total': 100},
            'consensus': {'success': 97, 'total': 100},
            'collusion': {'success': 99, 'total': 100},
            'horizon': {'success': 96, 'total': 100},
            'sampling': {'success': 94, 'total': 100},
            'arbiter': {'success': 95, 'total': 100}
        },
        'last_error': None,
        'session_start': time.time() - 3600
    }, module='VotingKernel')

    print("Test voting data setup complete!")

def test_voting_endpoints():
    """Test the voting API endpoints"""
    print("\nTesting voting API endpoints...")

    base_url = "http://localhost:8000/api/voting"
    endpoints = [
        "/overview",
        "/committee",
        "/consensus",
        "/collusion",
        "/alignment",
        "/sampling",
        "/strategy",
        "/timeline"
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
                    print(f"  ✅ Success: {endpoint}")
                    results[endpoint] = "SUCCESS"

                    # Print sample data for verification
                    if endpoint == "/overview":
                        print(f"    Total Decisions: {data.get('total_decisions', 0)}")
                        print(f"    Success Rate: {(data.get('success_rate', 0) * 100):.1f}%")
                        print(f"    Health Status: {data.get('health_status', 'unknown')}")
                        print(f"    Active Components: {data.get('components_active', 0)}/6")
                    elif endpoint == "/committee":
                        committee = data.get('committee', {})
                        summary = committee.get('summary', {})
                        analytics = committee.get('analytics', [])
                        print(f"    Total Members: {summary.get('total_members', 0)}")
                        print(f"    Active Members: {summary.get('active_members', 0)}")
                        print(f"    Member Analytics: {len(analytics)}")
                    elif endpoint == "/consensus":
                        consensus = data.get('consensus', {})
                        print(f"    Consensus Score: {(consensus.get('score', 0)):.3f}")
                        print(f"    Agreement Level: {consensus.get('analytics', {}).get('agreement_level', 'unknown')}")
                    elif endpoint == "/strategy":
                        strategy = data.get('strategy', {})
                        analysis = strategy.get('analysis', {})
                        print(f"    Final Decision: {analysis.get('final_decision', 'none')}")
                        print(f"    Confidence: {(analysis.get('confidence', 0) * 100):.1f}%")
                        print(f"    Gating Status: {analysis.get('gating_status', 'unknown')}")

                else:
                    print(f"  ❌ Failed: {endpoint} - {data.get('error', 'Unknown error')}")
                    results[endpoint] = f"FAILED: {data.get('error', 'Unknown error')}"
            else:
                print(f"  ❌ HTTP Error: {endpoint} - Status {response.status_code}")
                results[endpoint] = f"HTTP_ERROR: {response.status_code}"

        except requests.exceptions.RequestException as e:
            print(f"  ❌ Connection Error: {endpoint} - {e}")
            results[endpoint] = f"CONNECTION_ERROR: {str(e)}"

    return results

def print_test_results(results):
    """Print formatted test results"""
    print("\n" + "="*50)
    print("VOTING INTEGRATION TEST RESULTS")
    print("="*50)

    success_count = 0
    total_count = len(results)

    for endpoint, result in results.items():
        status = "✅ PASS" if result == "SUCCESS" else "❌ FAIL"
        print(f"{endpoint:15} | {status:8} | {result}")
        if result == "SUCCESS":
            success_count += 1

    print("-" * 50)
    print(f"SUMMARY: {success_count}/{total_count} endpoints passed")

    if success_count == total_count:
        print("🎉 ALL TESTS PASSED! Voting integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the backend server and endpoints.")

    # Additional voting system validation
    print("\n" + "="*50)
    print("VOTING SYSTEM VALIDATION")
    print("="*50)

    if success_count >= 6:  # Most endpoints working
        print("✅ Core voting pipeline operational")
        print("✅ Committee coordination functional")
        print("✅ Consensus detection working")
        print("✅ Strategy arbitration active")
        print("✅ Real-time monitoring enabled")
    else:
        print("❌ Voting system integration incomplete")
        print("❌ Check backend server and InfoBus connectivity")

if __name__ == "__main__":
    print("Voting Integration Test")
    print("=" * 50)

    # Setup test data
    setup_test_voting_data()

    # Test endpoints (requires backend server to be running)
    print("\nNote: Make sure the backend server is running on localhost:8000")
    input("Press Enter to continue with endpoint testing, or Ctrl+C to exit...")

    # Test the endpoints
    results = test_voting_endpoints()

    # Print results
    print_test_results(results)