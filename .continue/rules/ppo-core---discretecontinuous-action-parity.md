---
globs: modules/meta/ppo_core.py
description: The PPOCore must support both continuous actions (2D [-1,1]
  vectors) and discrete MaskablePPO actions with identical semantics. Discrete
  actions must be decoded using decode_discrete_action() and converted to
  continuous format via to_continuous(). Size buckets must match
  prop_firm_env.py configuration. Action masking in live must replicate training
  environment logic.
alwaysApply: false
---

Maintain both continuous (direction_score, size_score) and discrete (MaskablePPO) action support with proper decoding. Ensure training/live parity by using the same action masking logic and size_buckets configuration. Always convert discrete actions to continuous format for arbiter compatibility.