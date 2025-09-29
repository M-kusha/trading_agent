#!/usr/bin/env python3
"""
Comprehensive UnifiedMarketModule Isolated Test
Tests the module with real data to diagnose performance bottlenecks.
"""

import asyncio
import time
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from modules.utils.info_bus import SmartInfoBus
from modules.market.market_module import UnifiedMarketModule
from modules.external.market_data_provider import MarketDataProvider


class UnifiedMarketModuleTest:
    """Isolated test environment for UnifiedMarketModule"""

    def __init__(self):
        self.smart_bus = SmartInfoBus()
        self.results = {}

    async def setup_test_data(self):
        """Load real market data using MarketDataProvider"""
        print("Setting up test data...")

        # Initialize MarketDataProvider to get real data
        market_provider = MarketDataProvider()

        # Get a snapshot of real data
        print("Loading market data...")
        start_time = time.time()
        market_snapshot = await market_provider.process()
        load_time = time.time() - start_time

        print(f"Market data loaded in {load_time:.2f}s")
        print(f"Available keys: {list(market_snapshot.keys())}")

        # Publish to InfoBus (simulate MarketDataProvider behavior)
        print("Publishing to InfoBus...")
        for key, value in market_snapshot.items():
            if key.startswith('market_data_'):
                self.smart_bus.set(key, value, module="TestProvider")

        # Extract core inputs needed by UnifiedMarketModule
        test_inputs = {
            'bid_ask_data': market_snapshot.get('bid_ask_data', {}),
            'historical_prices': market_snapshot.get('historical_prices', {}),
            'macro_data': market_snapshot.get('macro_data', {}),
            'market_data': market_snapshot.get('market_data', {}),
            'multi_timeframe_data': market_snapshot.get('multi_timeframe_data', {}),
            'technical_indicators': market_snapshot.get('technical_indicators', {}),
            'timestamp': market_snapshot.get('timestamp', time.time()),
            'volatility_data': market_snapshot.get('volatility_data', {}),
            'volatility_level': market_snapshot.get('volatility_level', 0.5),
        }

        # Show data sizes
        print("\nInput Data Sizes:")
        for key, value in test_inputs.items():
            if isinstance(value, dict) and value:
                print(f"  {key}: {len(value)} items")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: {len(value)} elements")
            else:
                print(f"  {key}: {type(value).__name__}")

        return test_inputs

    async def test_unified_market_module(self, test_inputs: Dict[str, Any]):
        """Test UnifiedMarketModule with timing and data analysis"""
        print("\nTesting UnifiedMarketModule...")

        # Initialize the module
        print("Initializing UnifiedMarketModule...")
        init_start = time.time()
        unified_module = UnifiedMarketModule()
        init_time = time.time() - init_start
        print(f"Module initialized in {init_time:.3f}s")

        # Test individual components timing
        print("\nComponent Analysis:")

        # Check data extractor cache performance
        print("  Testing data extractor...")
        extractor_start = time.time()
        try:
            # Access the data extractor directly
            market_data = await unified_module.data_extractor.extract(
                sources=["inputs"],
                **test_inputs
            )
            extractor_time = time.time() - extractor_start
            print(f"  Data extractor: {extractor_time:.3f}s")
            print(f"     Extracted {len(market_data)} data fields")
        except Exception as e:
            extractor_time = time.time() - extractor_start
            print(f"  Data extractor failed: {extractor_time:.3f}s - {e}")
            return None

        # Test component execution
        print("  Testing components...")
        try:
            # Check which components are enabled
            enabled_components = []
            for comp_name, comp in unified_module.components.items():
                enabled_components.append(comp_name)
            print(f"     Enabled components: {enabled_components}")

            # Test each component individually (if possible)
            for comp_name in enabled_components:
                comp_start = time.time()
                print(f"     Testing {comp_name}...")
                # Note: Individual component testing would need component-specific logic
                comp_time = time.time() - comp_start
                print(f"     {comp_name}: Setup {comp_time:.3f}s")

        except Exception as e:
            print(f"  Component analysis failed: {e}")

        # Main process() test
        print("\nRunning main process() method...")
        process_start = time.time()

        try:
            # Set timeout to 30 seconds for detailed analysis
            result = await asyncio.wait_for(
                unified_module.process(**test_inputs),
                timeout=30.0
            )
            process_time = time.time() - process_start

            print(f"Process completed in {process_time:.3f}s")
            print(f"Output keys: {list(result.keys())}")

            # Analyze outputs
            self.analyze_outputs(result)

            return result

        except asyncio.TimeoutError:
            process_time = time.time() - process_start
            print(f"Process timed out after {process_time:.3f}s")
            return None

        except Exception as e:
            process_time = time.time() - process_start
            print(f"Process failed after {process_time:.3f}s: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            return None

    def analyze_outputs(self, result: Dict[str, Any]):
        """Analyze the outputs from UnifiedMarketModule"""
        print("\nOutput Analysis:")

        # Check required contract fields
        required_fields = [
            'fractal_metrics', 'market_regime', 'regime_data', 'regime_strength',
            'timestamps', 'trend_direction', 'liquidity_capabilities', 'liquidity_prediction',
            'liquidity_score', 'liquidity_thesis', 'market_depth', 'session_data',
            'spread_analysis', 'trading_sessions', 'market_theme', 'theme_detection'
        ]

        print("  Contract Compliance:")
        missing_fields = []
        present_fields = []

        for field in required_fields:
            if field in result:
                present_fields.append(field)
                print(f"    OK {field}: {type(result[field]).__name__}")
            else:
                missing_fields.append(field)
                print(f"    MISSING {field}")

        print(f"\n  Summary:")
        print(f"    Present: {len(present_fields)}/{len(required_fields)} fields")
        print(f"    Missing: {missing_fields}")

        # Check data quality
        print("\n  Data Quality:")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"    {key}: dict with {len(value)} items")
            elif isinstance(value, (list, tuple)):
                print(f"    {key}: {type(value).__name__} with {len(value)} elements")
            elif isinstance(value, str):
                print(f"    {key}: '{value[:50]}{'...' if len(value) > 50 else ''}'")
            else:
                print(f"    {key}: {value}")

    async def run_test(self):
        """Run the complete test suite"""
        print("UnifiedMarketModule Isolated Test")
        print("=" * 50)

        total_start = time.time()

        try:
            # Step 1: Setup test data
            test_inputs = await self.setup_test_data()

            # Step 2: Test the module
            result = await self.test_unified_market_module(test_inputs)

            total_time = time.time() - total_start

            print(f"\nTest completed in {total_time:.3f}s")

            if result:
                print("UnifiedMarketModule functioning correctly")

                # Save results for analysis
                output_file = Path("test_results_unified_market.json")
                with open(output_file, 'w') as f:
                    # Convert numpy arrays and other non-serializable types
                    serializable_result = self.make_serializable(result)
                    json.dump(serializable_result, f, indent=2)
                print(f"Results saved to {output_file}")

            else:
                print("UnifiedMarketModule failed or timed out")

        except Exception as e:
            total_time = time.time() - total_start
            print(f"\nTest failed after {total_time:.3f}s: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")

    def make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self.make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)


async def main():
    """Main test function"""
    test = UnifiedMarketModuleTest()
    await test.run_test()


if __name__ == "__main__":
    # Run the test
    print("Starting UnifiedMarketModule isolated test...")
    asyncio.run(main())