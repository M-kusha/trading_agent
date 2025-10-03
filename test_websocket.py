#!/usr/bin/env python3
"""
WebSocket Diagnostic Tool
Tests if the WebSocket connection to the backend is working
"""

import asyncio
import sys
import json
import websockets
from datetime import datetime

async def test_websocket_connection(url="ws://localhost:8000/ws"):
    """Test WebSocket connection to backend"""
    print(f"\n{'='*60}")
    print(f"WebSocket Connection Test")
    print(f"{'='*60}")
    print(f"Target: {url}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    try:
        print(f"[1/4] Attempting to connect to {url}...")
        async with websockets.connect(url, ping_interval=None) as websocket:
            print(f"✅ [2/4] Connection established!")
            print(f"    Connection State: {websocket.state.name}")
            
            # Send a ping message
            print(f"\n[3/4] Sending ping message...")
            ping_message = json.dumps({"type": "ping", "timestamp": datetime.now().isoformat()})
            await websocket.send(ping_message)
            print(f"✅ Ping sent: {ping_message}")
            
            # Wait for response
            print(f"\n[4/4] Waiting for response (timeout: 10s)...")
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                print(f"✅ Response received:")
                
                # Try to parse as JSON
                try:
                    data = json.loads(response)
                    print(f"    Type: {data.get('type', 'unknown')}")
                    print(f"    Data keys: {list(data.keys())}")
                    if 'data' in data and isinstance(data['data'], dict):
                        print(f"    Data.data keys: {list(data['data'].keys())[:10]}")  # First 10 keys
                except json.JSONDecodeError:
                    print(f"    Raw: {response[:200]}")  # First 200 chars
                
                print(f"\n{'='*60}")
                print(f"✅ SUCCESS: WebSocket connection is working!")
                print(f"{'='*60}\n")
                return True
                
            except asyncio.TimeoutError:
                print(f"⚠️  Timeout: No response within 10 seconds")
                print(f"   This might be normal if backend is not sending data yet")
                print(f"   Connection itself seems to be working though")
                print(f"\n{'='*60}")
                print(f"⚠️  PARTIAL SUCCESS: Connection works but no data received")
                print(f"{'='*60}\n")
                return True
                
    except websockets.exceptions.WebSocketException as e:
        print(f"\n❌ WebSocket Error: {e}")
        print(f"   Error Type: {type(e).__name__}")
        print(f"\n{'='*60}")
        print(f"❌ FAILED: WebSocket connection error")
        print(f"{'='*60}\n")
        return False
        
    except ConnectionRefusedError:
        print(f"\n❌ Connection Refused!")
        print(f"   The backend server is not running or not accepting connections")
        print(f"   Make sure the backend is started on port 8000")
        print(f"\n{'='*60}")
        print(f"❌ FAILED: Backend not running")
        print(f"{'='*60}\n")
        return False
        
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        print(f"   Error Type: {type(e).__name__}")
        print(f"\n{'='*60}")
        print(f"❌ FAILED: Unexpected error")
        print(f"{'='*60}\n")
        return False

async def test_http_health():
    """Test HTTP health endpoint"""
    print(f"\nTesting HTTP Health Endpoint...")
    print(f"{'-'*60}")
    
    try:
        import requests
        response = requests.get("http://localhost:8000/health", timeout=5)
        
        if response.status_code == 200:
            print(f"✅ Backend HTTP health check: OK")
            print(f"   Status Code: {response.status_code}")
            print(f"   Response: {response.json()}")
            return True
        else:
            print(f"⚠️  Backend HTTP health check: Unexpected status")
            print(f"   Status Code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"❌ Backend HTTP health check: Connection refused")
        print(f"   The backend server is not running on port 8000")
        return False
        
    except Exception as e:
        print(f"❌ Backend HTTP health check: Error - {e}")
        return False
    
    finally:
        print(f"{'-'*60}\n")

async def main():
    """Run all diagnostic tests"""
    print("\n" + "="*60)
    print("AI Trading Dashboard - WebSocket Diagnostic Tool")
    print("="*60 + "\n")
    
    # Test 1: HTTP Health
    http_ok = await test_http_health()
    
    if not http_ok:
        print("\n⚠️  Backend is not running. Please start it first:")
        print("   python run_dashboard.py")
        print("   OR")
        print("   uvicorn backend.main:app --port 8000")
        return 1
    
    # Test 2: WebSocket Connection
    ws_ok = await test_websocket_connection()
    
    if ws_ok:
        print("\n🎉 All tests passed!")
        print("   The WebSocket connection is working correctly.")
        print("   If the dashboard still shows 'Disconnected', the issue is in the frontend.")
        return 0
    else:
        print("\n❌ WebSocket test failed!")
        print("   Possible causes:")
        print("   1. WebSocket endpoint is not configured correctly")
        print("   2. CORS or security policy blocking WebSocket")
        print("   3. Backend WebSocket handler has errors")
        print("\n   Check logs/backend.log for more details")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
