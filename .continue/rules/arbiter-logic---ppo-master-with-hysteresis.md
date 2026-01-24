---
globs: modules/meta/arbiter_logic.py
description: ArbiterLogic must respect PPO as the primary decision maker, with
  experts/committee providing only advisory adjustments. Hysteresis should only
  smooth direction changes for existing positions, not influence initial entry
  decisions. Discrete HOLD actions must maintain current position direction.
  PPO's exit signals (explicit_close, explicit_reverse) must be preserved for
  downstream position management.
alwaysApply: false
---

PPO is always the master decision maker. Experts and committee only adjust confidence and size (advisory). Apply hysteresis only to open positions (not to flat state). Preserve discrete HOLD intent for position management. Never override PPO's exit/intent signals.