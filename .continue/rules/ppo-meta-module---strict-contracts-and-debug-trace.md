---
globs: modules/meta/*.py
description: This rule ensures that the PPO meta module maintains strict
  contracts between training and live trading. The observation builder must only
  support XAUUSD in strict mode, with no silent defaults or cross-symbol
  leakage. Debug traces must be written to the dedicated JSONL file for
  end-to-end validation. The 84-dimensional observation schema (v5.6) must
  remain unchanged to preserve training/live parity.
alwaysApply: false
---

Always use the unified PPOObservationBuilder for strict XAUUSD-only observation construction. Enable debug traces in logs/ppo_obs_debug_xauusd.jsonl for full input/output validation. Never modify observation dimensions (84 features) or break training/live parity.