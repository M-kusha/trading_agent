---
globs: modules/meta/ppo_agent_shell.py
description: The PPOAgentShell is the only component that interacts with
  SmartInfoBus. It must implement warmup periods (default 30 cycles) to observe
  market conditions before making trading decisions. Position focus mode should
  prioritize managing existing positions over new entries. The trade_open_gate
  must integrate risk, memory, seasonality, and timing constraints to safely
  block new entries when conditions are unfavorable.
alwaysApply: false
---

Maintain single SmartInfoBus gateway in PPOAgentShell. Implement warmup periods (30+ cycles) for market observation before trading. Support position focus mode to manage existing positions. Use trade_open_gate to block new entries when risk/memory/seasonality conditions are violated.