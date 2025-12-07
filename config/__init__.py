"""Config package - YAML configuration files.

Available configs:
- system_config.yaml: Module timeouts, execution settings
- module_registry.yaml: Auto-generated module metadata
- risk_policy.yaml: Prop firm rules, lot sizing, risk limits
- explainability_standards.yaml: Thesis and explanation requirements

Note: For TradingConfig and ConfigFactory, import from envs.config
"""
# Config package exposes nothing - all config is in YAML files
# Use envs.config for TradingConfig/ConfigFactory
