#!/usr/bin/env python3
"""
Shadow Test - Lightweight Model Testing (v2)
===========================================

Test a trained PPO model against live MT5 prices WITHOUT:
- Full module system
- Orchestrator
- SmartInfoBus
- Actual trade execution

Just: Model + MT5 Prices → Predicted Actions (display + optional JSON log)
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, TYPE_CHECKING

import numpy as np

# Add project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Type checking imports (won't run at runtime)
if TYPE_CHECKING:
    import MetaTrader5 as mt5  # type: ignore[import]

# Check MT5 availability
try:
    import MetaTrader5 as mt5  # type: ignore[import]
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None  # type: ignore[assignment]
    MT5_AVAILABLE = False

DEFAULT_BALANCE = 100_000.0
DEFAULT_SYMBOLS = ["EURUSD", "XAUUSD"]


# ═══════════════════════════════════════════════════════════════════
# MODEL HELPERS
# ═══════════════════════════════════════════════════════════════════

def load_model(model_path: Optional[str] = None):
    """Load the trained PPO model."""
    from stable_baselines3 import PPO

    # Find model
    if model_path:
        path = Path(model_path)
    else:
        # Try common locations
        candidates = [
            "models/ppo_final.zip",
        ]
        path = None
        for c in candidates:
            if Path(c).exists():
                path = Path(c)
                break

        if not path:
            print("❌ No model found! Checked:")
            for c in candidates:
                print(f"   - {c}")
            sys.exit(1)

    print(f"📦 Loading model: {path}")
    model = PPO.load(str(path))
    print("✅ Model loaded successfully")
    return model


def get_model_obs_dim(model) -> int:
    """Infer observation dimension from model; fall back to 64."""
    try:
        space = getattr(model, "observation_space", None)
        if space is not None and getattr(space, "shape", None) is not None:
            return int(np.prod(space.shape))
    except Exception:
        pass
    return 64


def get_action_layout(model, symbols: Optional[List[str]]) -> Tuple[int, List[str]]:
    """
    Infer action dimension and adjust symbol list accordingly.

    Assumes continuous Box action of size 2 * N:
      - For each instrument: (direction, size).
    """
    try:
        space = getattr(model, "action_space", None)
        if space is not None and getattr(space, "shape", None) is not None:
            action_dim = int(np.prod(space.shape))
        else:
            action_dim = 2
    except Exception:
        action_dim = 2

    if symbols is None or len(symbols) == 0:
        # Derive synthetic symbols from action dim
        if action_dim % 2 == 0:
            n_inst = action_dim // 2
        else:
            n_inst = 1
        symbols_used = [f"INST_{i+1}" for i in range(n_inst)]
    else:
        symbols_used = list(symbols)
        n_inst = len(symbols_used)
        expected_dim = n_inst * 2
        if action_dim != expected_dim:
            print(
                f"⚠️  Action dimension {action_dim} does not match 2 * instruments ({expected_dim}). "
                "Shadow interpretation may be approximate."
            )

    return action_dim, symbols_used


# ═══════════════════════════════════════════════════════════════════
# MT5 / MARKET DATA
# ═══════════════════════════════════════════════════════════════════

def connect_mt5() -> bool:
    """Connect to MetaTrader 5."""
    if not MT5_AVAILABLE:
        print("❌ MetaTrader5 not installed. Install with: pip install MetaTrader5")
        return False

    # Load credentials from existing credentials class
    try:
        from live.mt5_credentials import MT5Credentials
        credentials = MT5Credentials()
        account = credentials.ACCOUNT
        password = credentials.PASSWORD
        server = credentials.SERVER
        print(f"🔑 Using credentials for account {account}")
    except ImportError:
        account, password, server = None, None, None
        print("⚠️  No credentials file found, trying default MT5 connection")

    if not mt5.initialize():  # type: ignore[union-attr]
        print(f"❌ MT5 initialize failed: {mt5.last_error()}")  # type: ignore[union-attr]
        return False

    # Login if credentials available
    if account and password and server:
        if not mt5.login(account, password, server=server):  # type: ignore[union-attr]
            print(f"❌ MT5 login failed: {mt5.last_error()}")  # type: ignore[union-attr]
            mt5.shutdown()  # type: ignore[union-attr]
            return False

    account_info = mt5.account_info()  # type: ignore[union-attr]
    if account_info:
        print(f"✅ MT5 Connected: {account_info.login} | Balance: ${account_info.balance:,.2f}")
    else:
        print("✅ MT5 Connected (no account info)")

    return True


def get_live_prices(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Get current prices from MT5, ensuring symbols are selected."""
    prices: Dict[str, Dict[str, Any]] = {}

    for symbol in symbols:
        # Ensure symbol is visible in MarketWatch
        info = mt5.symbol_info(symbol)  # type: ignore[union-attr]
        if info is None:
            print(f"⚠️  Symbol {symbol} not found in MT5")
            continue
        if not info.visible:
            mt5.symbol_select(symbol, True)  # type: ignore[union-attr]

        tick = mt5.symbol_info_tick(symbol)  # type: ignore[union-attr]
        if not tick:
            print(f"⚠️  No tick data for {symbol}")
            continue

        # Spread in points, based on digits
        digits = info.digits
        multiplier = 10 ** digits if digits > 0 else 1
        spread_points = (tick.ask - tick.bid) * multiplier

        prices[symbol] = {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread_points": round(spread_points, 1),
            "time": datetime.fromtimestamp(tick.time).strftime("%H:%M:%S"),
        }

    return prices


# ═══════════════════════════════════════════════════════════════════
# OBSERVATION BUILDING
# ═══════════════════════════════════════════════════════════════════

def build_simple_observation(
    prices: Dict[str, Dict[str, Any]],
    history: List[Dict[str, Dict[str, Any]]],
    symbols: List[str],
    obs_size: int,
    balance: float = DEFAULT_BALANCE,
) -> np.ndarray:
    """
    Build a simplified observation for the model.

    NOTE: This is intentionally minimal, not a replica of your full observation
    builder. Treat this as a rough "probing input", not a definitive evaluation.
    """
    obs = np.zeros(obs_size, dtype=np.float32)

    # Market features: normalized bid price per symbol, sequentially
    idx = 0
    for symbol in symbols:
        if idx >= obs_size:
            break
        if symbol not in prices:
            continue
        price = float(prices[symbol]["bid"])
        if "XAU" in symbol:
            # Gold ~1500–2500
            obs[idx] = (price - 2000.0) / 500.0
        else:
            # Majors ~0.95–1.15 or similar
            obs[idx] = (price - 1.05) / 0.10
        idx += 1

    # Optional: 1-step return for primary instrument
    if history and obs_size > idx:
        primary = symbols[0]
        last = history[-1]
        if primary in last and primary in prices:
            prev_p = float(last[primary]["bid"])
            curr_p = float(prices[primary]["bid"])
            ret = (curr_p - prev_p) / max(1e-6, prev_p)
            obs[idx] = float(np.clip(ret * 100.0, -5.0, 5.0))  # % move clipped
            idx += 1

    # Account features (very rough)
    if obs_size > 16:
        obs[16] = (balance - DEFAULT_BALANCE) / 10_000.0  # Normalized equity change
    if obs_size > 17:
        obs[17] = 0.0  # Exposure placeholder
    if obs_size > 18:
        obs[18] = 0.0  # Position count placeholder

    # Leave remaining dims at 0.0 (no random noise; keep it stable)
    return obs


# ═══════════════════════════════════════════════════════════════════
# ACTION INTERPRETATION
# ═══════════════════════════════════════════════════════════════════

def interpret_single_action(direction: float, size: float) -> Tuple[str, float, float]:
    """Interpret a single (direction, size) pair."""
    # Convert to decision
    if direction > 0.3:
        decision = "🟢 BUY"
        confidence = min(1.0, (direction - 0.3) / 0.7)
    elif direction < -0.3:
        decision = "🔴 SELL"
        confidence = min(1.0, (-direction - 0.3) / 0.7)
    else:
        decision = "⚪ HOLD"
        confidence = 1.0 - min(1.0, abs(direction) / 0.3)

    return decision, confidence, float(abs(size))


def interpret_actions(
    action: np.ndarray,
    symbols: List[str],
) -> Dict[str, Dict[str, Any]]:
    """
    Interpret full continuous action array for all instruments.

    Expects (2 * len(symbols),) layout: [dir_0, size_0, dir_1, size_1, ...].
    If action is shorter, we re-use the first pair.
    """
    flat = np.array(action).flatten()
    n_sym = len(symbols)
    result: Dict[str, Dict[str, Any]] = {}

    if flat.size < 2:
        # Degenerate; treat as global direction only
        d = float(flat[0]) if flat.size == 1 else 0.0
        dec, conf, size = interpret_single_action(d, 0.5)
        for sym in symbols:
            result[sym] = {
                "decision": dec,
                "confidence": conf,
                "size": size,
                "direction_raw": d,
                "size_raw": 0.5,
            }
        return result

    # Use as many pairs as we can
    for i, sym in enumerate(symbols):
        base = 2 * i
        if base + 1 >= flat.size:
            # Re-use first pair as fallback
            d = float(flat[0])
            s = float(flat[1])
        else:
            d = float(flat[base])
            s = float(flat[base + 1])

        dec, conf, size = interpret_single_action(d, s)
        result[sym] = {
            "decision": dec,
            "confidence": conf,
            "size": size,
            "direction_raw": d,
            "size_raw": s,
        }

    return result


# ═══════════════════════════════════════════════════════════════════
# SHADOW TEST (LIVE) / OFFLINE TEST
# ═══════════════════════════════════════════════════════════════════

def run_shadow_test(
    model,
    symbols: List[str],
    interval: float = 5.0,
    duration_minutes: int = 60,
    balance: float = DEFAULT_BALANCE,
):
    """Run shadow test - display model decisions without executing."""
    obs_dim = get_model_obs_dim(model)
    _, symbols_used = get_action_layout(model, symbols)

    print("\n" + "=" * 70)
    print("🔮 SHADOW TEST MODE - Model predictions only (no real trades)")
    print("=" * 70)
    print(f"Symbols: {', '.join(symbols_used)}")
    print(f"Obs dim: {obs_dim}")
    print(f"Update interval: {interval}s")
    print(f"Duration: {duration_minutes} minutes")
    print("Press Ctrl+C to stop\n")

    price_history: List[Dict[str, Dict[str, Any]]] = []
    decisions_log: List[Dict[str, Any]] = []
    start_time = time.time()
    iteration = 0

    try:
        while True:
            iteration += 1
            elapsed_min = (time.time() - start_time) / 60.0

            if elapsed_min >= duration_minutes:
                print(f"\n⏱️  Duration reached ({duration_minutes} min)")
                break

            # Get prices
            prices = get_live_prices(symbols_used)
            if not prices:
                print("⚠️  No prices available")
                time.sleep(interval)
                continue

            price_history.append(prices)

            # Build observation
            obs = build_simple_observation(
                prices=prices,
                history=price_history,
                symbols=symbols_used,
                obs_size=obs_dim,
                balance=balance,
            )

            # Get model prediction
            action, _states = model.predict(obs, deterministic=True)
            decisions = interpret_actions(action, symbols_used)

            # Display
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"\n[{timestamp}] Iteration {iteration} | Elapsed: {elapsed_min:.1f}m")
            print("-" * 50)

            for symbol, data in prices.items():
                print(
                    f"  {symbol}: {data['bid']:.5f} / {data['ask']:.5f} "
                    f"(spread: {data['spread_points']} pts @ {data['time']})"
                )

            print("\n  📊 Model Decisions:")
            for sym in symbols_used:
                dec = decisions[sym]
                print(
                    f"    {sym}: {dec['decision']} | "
                    f"conf: {dec['confidence']:.1%} | "
                    f"size: {dec['size']:.2f} | "
                    f"raw: ({dec['direction_raw']:.3f}, {dec['size_raw']:.3f})"
                )

            # Log decision
            decisions_log.append(
                {
                    "time": timestamp,
                    "prices": prices,
                    "actions": np.array(action).flatten().tolist(),
                    "decisions": decisions,
                }
            )

            # Stats
            total_signals = max(1, len(decisions_log) * len(symbols_used))
            buy_count = sum(
                1
                for d in decisions_log
                for x in d["decisions"].values()
                if "BUY" in x["decision"]
            )
            sell_count = sum(
                1
                for d in decisions_log
                for x in d["decisions"].values()
                if "SELL" in x["decision"]
            )
            hold_count = sum(
                1
                for d in decisions_log
                for x in d["decisions"].values()
                if "HOLD" in x["decision"]
            )

            print(
                f"\n  📈 Stats across all instruments/signals:"
                f" BUY={buy_count} | SELL={sell_count} | HOLD={hold_count} "
                f"(total signals: {total_signals})"
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped by user")

    # Summary + JSON log
    print("\n" + "=" * 70)
    print("📊 SESSION SUMMARY")
    print("=" * 70)
    total_signals = max(1, len(decisions_log) * len(symbols_used))
    buy_count = sum(
        1
        for d in decisions_log
        for x in d["decisions"].values()
        if "BUY" in x["decision"]
    )
    sell_count = sum(
        1
        for d in decisions_log
        for x in d["decisions"].values()
        if "SELL" in x["decision"]
    )
    hold_count = sum(
        1
        for d in decisions_log
        for x in d["decisions"].values()
        if "HOLD" in x["decision"]
    )

    print(f"Total iterations: {len(decisions_log)}")
    print(f"Total signals:    {total_signals}")
    print(f"BUY signals:      {buy_count} ({buy_count / total_signals * 100:.1f}%)")
    print(f"SELL signals:     {sell_count} ({sell_count / total_signals * 100:.1f}%)")
    print(f"HOLD signals:     {hold_count} ({hold_count / total_signals * 100:.1f}%)")

    if decisions_log:
        avg_conf = np.mean(
            [x["confidence"] for d in decisions_log for x in d["decisions"].values()]
        )
        print(f"Avg confidence:   {avg_conf:.1%}")

        # Save JSON log
        log_dir = Path("logs/shadow_test")
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"shadow_test_{ts}.json"
        with log_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "created_at": datetime.now().isoformat(),
                    "symbols": symbols_used,
                    "decisions": decisions_log,
                },
                f,
                indent=2,
            )
        print(f"\n📝 Session log saved to: {log_path}")

    # Cleanup
    mt5.shutdown()  # type: ignore[union-attr]
    print("\n✅ MT5 disconnected")


def run_offline_test(model, symbols: Optional[List[str]] = None, num_steps: int = 100):
    """Run offline test with random observations (no MT5 needed)."""
    obs_dim = get_model_obs_dim(model)
    _, symbols_used = get_action_layout(model, symbols)

    print("\n" + "=" * 70)
    print("🧪 OFFLINE TEST MODE - Random observations (no MT5)")
    print("=" * 70)
    print(f"Obs dim:  {obs_dim}")
    print(f"Symbols:  {', '.join(symbols_used)}")
    print(f"Steps:    {num_steps}\n")

    decisions_count = {"BUY": 0, "SELL": 0, "HOLD": 0}
    confidences: List[float] = []

    for i in range(num_steps):
        # Random observation
        obs = (np.random.randn(obs_dim).astype(np.float32) * 0.5)

        # Get prediction
        action, _ = model.predict(obs, deterministic=True)
        decisions = interpret_actions(action, symbols_used)

        # Aggregate decisions
        for info in decisions.values():
            dec = str(info["decision"])
            conf = float(info["confidence"])
            confidences.append(conf)
            if "BUY" in dec:
                decisions_count["BUY"] += 1
            elif "SELL" in dec:
                decisions_count["SELL"] += 1
            else:
                decisions_count["HOLD"] += 1

        if (i + 1) % 20 == 0:
            # Show just the first instrument’s decision as a sample
            sample_sym = symbols_used[0]
            sample_dec = decisions[sample_sym]
            print(
                f"  Step {i+1}/{num_steps}: {sample_sym} "
                f"{sample_dec['decision']} (conf: {sample_dec['confidence']:.1%})"
            )

    total_signals = max(1, num_steps * len(symbols_used))
    print("\n" + "-" * 50)
    print("📊 RESULTS (all instruments combined):")
    print(
        f"  BUY:  {decisions_count['BUY']} "
        f"({decisions_count['BUY'] / total_signals * 100:.1f}%)"
    )
    print(
        f"  SELL: {decisions_count['SELL']} "
        f"({decisions_count['SELL'] / total_signals * 100:.1f}%)"
    )
    print(
        f"  HOLD: {decisions_count['HOLD']} "
        f"({decisions_count['HOLD'] / total_signals * 100:.1f}%)"
    )
    print(f"  Avg Confidence: {np.mean(confidences):.1%}")

    # Check for degenerate policy
    max_pct = max(decisions_count.values()) / total_signals
    if max_pct > 0.9:
        print("\n⚠️  WARNING: Policy seems degenerate (one action > 90%)")
    else:
        print("\n✅ Policy appears reasonably balanced")

    # Save JSON log
    log_dir = Path("logs/shadow_test")
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"offline_test_{ts}.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(),
                "mode": "offline",
                "symbols": symbols_used,
                "num_steps": num_steps,
                "summary": {
                    "total_signals": total_signals,
                    "buy_count": decisions_count["BUY"],
                    "sell_count": decisions_count["SELL"],
                    "hold_count": decisions_count["HOLD"],
                    "avg_confidence": float(np.mean(confidences)),
                },
            },
            f,
            indent=2,
        )
    print(f"\n📝 Session log saved to: {log_path}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Shadow Test - Test trained model against live prices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with live MT5 prices for 30 minutes
  python shadow_test.py --duration 30

  # Test with specific model
  python shadow_test.py --model checkpoints/ppo_exploration_final.zip

  # Offline test (no MT5 needed)
  python shadow_test.py --offline

  # Faster updates (every 2 seconds)
  python shadow_test.py --interval 2
        """,
    )
    parser.add_argument("--model", type=str, help="Path to trained model (.zip)")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help="Symbols to monitor (default: EURUSD XAUUSD)",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="Update interval in seconds")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in minutes")
    parser.add_argument("--offline", action="store_true", help="Run offline test (no MT5)")
    parser.add_argument(
        "--balance",
        type=float,
        default=DEFAULT_BALANCE,
        help="Virtual account balance used for normalization",
    )

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("🔮 SHADOW TEST - Lightweight Model Testing")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")

    # Load model
    model = load_model(args.model)

    if args.offline:
        run_offline_test(model, symbols=args.symbols, num_steps=100)
        return

    if not connect_mt5():
        print("\n💡 Tip: Use --offline to test without MT5")
        sys.exit(1)

    try:
        run_shadow_test(
            model,
            symbols=args.symbols,
            interval=args.interval,
            duration_minutes=args.duration,
            balance=args.balance,
        )
    finally:
        # In case something exploded before shutdown inside run_shadow_test
        try:
            if MT5_AVAILABLE:
                mt5.shutdown()  # type: ignore[union-attr]
        except Exception:
            pass


if __name__ == "__main__":
    main()
