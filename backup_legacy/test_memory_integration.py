#!/usr/bin/env python3
"""
Test script for Memory Integration
Tests the memory API endpoints and data flow
"""

import json
import time
import requests
from modules.utils.info_bus import InfoBusManager

def setup_test_data():
    """Set up sample memory data in InfoBus for testing"""
    print("Setting up test memory data...")

    bus = InfoBusManager.get_instance()

    # Overview data
    bus.set('unified_metrics', {
        'total_memories': 1250,
        'memory_utilization': 0.73,
        'components_active': 6,
        'processing_status': 'active',
        'health_status': 'healthy'
    }, module='UnifiedMemory')

    bus.set('unified_memory_status', {
        'components_enabled': 6,
        'memory_size': 1250,
        'status': 'initialized'
    }, module='UnifiedMemory')

    # Neural component data
    bus.set('neural_memory', {
        'buffer_size': 256,
        'memory_utilization': 0.68
    }, module='UnifiedMemory')

    bus.set('attention_retrieval', {
        'retrieved_count': 45,
        'similarity_scores': [0.95, 0.87, 0.92, 0.81, 0.89]
    }, module='UnifiedMemory')

    bus.set('memory_embedding', {
        'embedding_dim': 32,
        'total_embeddings': 256
    }, module='UnifiedMemory')

    bus.set('importance_scoring', {
        'average_importance': 0.734,
        'total_scored': 1180
    }, module='UnifiedMemory')

    # Playbook component data
    bus.set('playbook_recall', {
        'memory_entries': 342,
        'patterns_identified': 28
    }, module='UnifiedMemory')

    bus.set('pattern_memory', {
        'total_patterns': 28,
        'pattern_effectiveness': {'pattern_1': 0.85, 'pattern_2': 0.72}
    }, module='UnifiedMemory')

    bus.set('playbook_quality', {
        'quality_score': 0.81,
        'memory_utilization': 0.64
    }, module='UnifiedMemory')

    bus.set('memory_analytics', {
        'total_recalls': 892,
        'memory_health': 'good'
    }, module='UnifiedMemory')

    # Mistakes component data
    bus.set('mistake_memory', {
        'recent': [{'type': 'loss', 'amount': -15.5}, {'type': 'loss', 'amount': -8.2}],
        'stats': {'count': 47, 'last_ts': time.time()}
    }, module='UnifiedMemory')

    bus.set('mistake_avoidance', {
        'avoidance_signal': 0.234,
        'consecutive_losses': 2
    }, module='UnifiedMemory')

    bus.set('danger_zones', {
        'zones': [{'level': 0.8, 'area': 'high_volatility'}, {'level': 0.6, 'area': 'news_events'}],
        'zone_count': 2
    }, module='UnifiedMemory')

    bus.set('loss_prevention', {
        'avoidance_effectiveness': 0.67,
        'learning_samples': 234
    }, module='UnifiedMemory')

    bus.set('pattern_recognition', {
        'loss_patterns': {'pattern_A': 0.3, 'pattern_B': 0.45},
        'win_patterns': {'pattern_C': 0.82, 'pattern_D': 0.71}
    }, module='UnifiedMemory')

    print("Test data setup complete!")

def test_memory_endpoints():
    """Test the memory API endpoints"""
    print("\nTesting memory API endpoints...")

    base_url = "http://localhost:8000/api/memory"
    endpoints = [
        "/overview",
        "/components",
        "/patterns",
        "/mistakes",
        "/performance"
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
    print("MEMORY INTEGRATION TEST RESULTS")
    print("="*50)

    success_count = 0
    total_count = len(results)

    for endpoint, result in results.items():
        status = "✓ PASS" if result == "SUCCESS" else "✗ FAIL"
        print(f"{endpoint:20} | {status:8} | {result}")
        if result == "SUCCESS":
            success_count += 1

    print("-" * 50)
    print(f"SUMMARY: {success_count}/{total_count} endpoints passed")

    if success_count == total_count:
        print("🎉 ALL TESTS PASSED! Memory integration is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the backend server and endpoints.")

if __name__ == "__main__":
    print("Memory Integration Test")
    print("=" * 50)

    # Setup test data
    setup_test_data()

    # Test endpoints (requires backend server to be running)
    print("\nNote: Make sure the backend server is running on localhost:8000")
    input("Press Enter to continue with endpoint testing, or Ctrl+C to exit...")

    # Test the endpoints
    results = test_memory_endpoints()

    # Print results
    print_test_results(results)