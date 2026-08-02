from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class ModuleContract:
    name: str
    file: str = ""
    provides: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


CONTRACTS: Dict[str, ModuleContract] = {
    'LivePPOAgent': ModuleContract(
        name='LivePPOAgent',
        file='meta/live_ppo_agent.py',
        provides=[
            'ppo_final_decision',
            'ppo_gate_passed',
            'ppo_position_size',
            'action_mask',
        ],
        requires=[
            'market_data',
            'expert_signals',
            'risk_data',
            'account_state',
            'trading_mode_state',
            'governor_state',
        ],
        meta={'category': 'meta', 'version': '1.0.0'},
    ),

    'ActiveTradeMonitor': ModuleContract(
        name='ActiveTradeMonitor',
        file='risk/active_trade_monitor.py',
        provides=['duration_alerts', 'position_duration_risk', 'position_tracking', 'trade_monitor_status'],
        requires=['market_context', 'positions', 'step_idx'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'risk', 'version': '3.0.0'}
    ),

    'ComplianceModule': ModuleContract(
        name='ComplianceModule',
        file='risk/compliance.py',
        provides=['compliance', 'compliance_status', 'risk_limits', 'validation_results'],
        requires=['portfolio_metrics', 'market_context', 'order_data', 'positions'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'risk', 'version': '3.0.0'}
    ),

    'CorrelatedRiskController': ModuleContract(
        name='CorrelatedRiskController',
        file='risk/correlated_risk_controller.py',
        provides=['correlation_clusters', 'correlation_matrix', 'correlation_risk', 'diversification_score'],
        requires=['market_context', 'positions', 'prices'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'risk', 'version': '3.0.0'}
    ),

    'DrawdownRescue': ModuleContract(
        name='DrawdownRescue',
        file='risk/drawdown_rescue.py',
        provides=['drawdown_risk', 'rescue_status', 'risk_adjustment'],
        requires=['portfolio_metrics', 'market_context', 'positions'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'risk', 'version': '3.0.0'}
    ),

    'DynamicRiskController': ModuleContract(
        name='DynamicRiskController',
        file='risk/dynamic_risk_controller.py',
        provides=['risk_alerts', 'risk_analytics', 'risk_factors', 'risk_scaling',
                  'risk_level', 'risk_scale', 'risk_assessment',
                  'DynamicRiskController_voting_proposal', 'DynamicRiskController_confidence'],


        requires=['market_context', 'market_data', 'market_regime', 'position_data',
                  'memory_gate', 'danger_zones'],

        meta={'is_voting_member': False, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'risk', 'version': '4.2.0'}
    ),

    'EnhancedAnomalyDetector': ModuleContract(
        name='EnhancedAnomalyDetector',
        file='risk/anomaly_detector.py',
        provides=['anomaly_alerts', 'anomaly_detection', 'anomaly_score', 'detection_analytics', 'anomaly_detector',
                  'EnhancedAnomalyDetector_voting_proposal', 'EnhancedAnomalyDetector_confidence'],
        requires=['market_context', 'market_data', 'performance_data', 'risk_data', 'trading_data'],

        meta={'is_voting_member': False, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'risk', 'version': '4.0.0'}
    ),

    'ExecutionQualityMonitor': ModuleContract(
        name='ExecutionQualityMonitor',
        file='risk/execution_quality_monitor.py',
        provides=['execution_alerts', 'execution_analytics', 'execution_quality', 'quality_metrics',
                  'ExecutionQualityMonitor_voting_proposal', 'ExecutionQualityMonitor_confidence'],


        requires=['market_context', 'market_data'],

        meta={'is_voting_member': False, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'risk', 'version': '4.0.0'}
    ),

    'PortfolioRiskSystem': ModuleContract(
        name='PortfolioRiskSystem',
        file='risk/portfolio_risk_system.py',
        provides=['portfolio_risk', 'portfolio_risk_proposal', 'position_limits', 'risk_data', 'risk_metrics',
                  'risk_score', 'risk_signals', 'portfolio_trade_data', 'trading_data',
                  'PortfolioRiskSystem_voting_proposal', 'PortfolioRiskSystem_confidence'],


        requires=['market_context', 'market_data', 'positions',

                  'correlation_matrix', 'correlation_risk', 'diversification_score'],

        meta={'is_voting_member': False, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'risk', 'version': '4.1.0'}
    ),


    'MetaCognitivePlanner': ModuleContract(
        name='MetaCognitivePlanner',
        file='meta/metacognitive_planner.py',
        provides=['adaptation_metrics', 'planning_status', 'strategic_insights', 'tactical_recommendations'],
        requires=['actions', 'market_context', 'market_data', 'market_regime', 'performance_metrics',
                  'regime_data', 'trades', 'volatility_adjustment'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'meta', 'version': '3.0.1', 'disabled': True}
    ),


    'MetaRLController': ModuleContract(
        name='MetaRLController',
        file='meta/legacy/meta_rl_controller.py',
        provides=['agent_decisions', 'agents_performance', 'automation_status', 'controller_status',
                  'controller_training_overview', 'meta_signals', 'trading_signal', 'trading_signals'],
        requires=['actions', 'market_data', 'trades', 'training_signals'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'meta', 'version': '3.0.0', 'disabled': True}
    ),


    'MetaAgent': ModuleContract(
        name='MetaAgent',
        file='meta/legacy/meta_agent.py',
        provides=['automation_decisions', 'automation_metrics', 'meta_performance', 'system_mode',
                  'MetaAgent_voting_proposal', 'MetaAgent_confidence',

                  'active_strategy', 'auto_mode'],
        requires=['market_context', 'risk_signals', 'system_performance', 'time_risk_analysis', 'training_metrics'],


        meta={'is_voting_member': False, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'meta', 'version': '3.0.0', 'disabled': True}
    ),


    'PPOLagAgent': ModuleContract(
        name='PPOLagAgent',
        file='meta/ppo_lag_agent.py',
        provides=['agent_status', 'market_adaptation', 'position_metrics', 'ppo_lag_training_metrics'],
        requires=['actions', 'market_data', 'trades', 'training_signals'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'meta', 'version': '3.0.0', 'disabled': True}
    ),


    'BiasAuditor': ModuleContract(
        name='BiasAuditor',
        file='strategy/bias_auditor.py',
        provides=['bias_adjustments', 'bias_analysis', 'bias_auditor_initialization', 'bias_corrections',
                  'bias_recommendations', 'bias_report', 'psychological_state'],
        requires=['trading_result', 'market_regime', 'positions', 'recent_trades', 'risk_data', 'session_context',
                  'trading_session', 'volatility_level'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'strategy', 'version': '3.0.0'}
    ),


    'StrategyIntrospector': ModuleContract(
        name='StrategyIntrospector',
        file='strategy/strategy_introspector.py',
        provides=['adaptation_recommendations', 'behavior_patterns', 'introspection_metrics', 'module_data',
                  'strategy_analysis', 'strategy_introspector_initialization', 'strategy_performance',
                  'strategy_profiles', 'trading_performance',

                  'trade_performance'],
        requires=['market_context', 'market_regime', 'recent_trades', 'risk_data', 'volatility_data'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'strategy', 'version': '3.0.0'}
    ),

    'ThesisEvolutionEngine': ModuleContract(
        name='ThesisEvolutionEngine',
        file='strategy/thesis_evolution_engine.py',
        provides=['active_theses', 'best_thesis', 'evolution_history',
                  'thesis_diversity', 'thesis_evolution_initialization', 'thesis_performance', 'thesis_recommendations',
                  'market_thesis'],
        requires=['market_context', 'market_data', 'market_regime', 'recent_trades',
                  'risk_metrics', 'session_metrics', 'strategy_performance', 'trading_performance', 'volatility_data'],
        meta={'is_voting_member': False, 'thesis_required': True, 'explainable': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'strategy', 'version': '3.0.0'}
    ),

    'ExplanationGenerator': ModuleContract(
        name='ExplanationGenerator',
        file='strategy/explanation_generator.py',
        provides=['contextual_narratives', 'decision_rationales', 'explanation_generator_initialization',
                  'operator_updates', 'performance_insights', 'system_explanations', 'trading_explanations',
                  'market_overview'],
        requires=['bias_analysis', 'market_context', 'module_data', 'module_insights',
                  'performance_data', 'positions', 'recent_trades', 'risk_data', 'session_metrics',
                  'system_alerts'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'strategy', 'version': '3.0.0'}
    ),


    'CurriculumPlannerPlus': ModuleContract(
        name='CurriculumPlannerPlus',
        file='strategy/curriculum_planner_plus.py',
        provides=['competency_scores', 'curriculum_initialization', 'curriculum_stage', 'learning_constraints', 'learning_recommendations', 'mastery_assessment', 'stage_advancement', 'stage_progression'],
        requires=['episode_summary', 'market_context', 'performance_data', 'closed_positions', 'risk_metrics', 'trading_session'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),

    'EntryTimingController': ModuleContract(
        name='EntryTimingController',
        file='strategy/entry_timing_controller.py',
        provides=['entry_timing', 'entry_timing_array', 'entry_timing_allowed'],

        requires=['market_data_latest', 'multi_timeframe_data', 'atr_values', 'session_info'],
        meta={'thesis_required': False, 'health_monitoring': True, 'performance_tracking': False,
              'category': 'strategy', 'version': '1.0.0'}
    ),


    'MarketDataProvider': ModuleContract(
        name='MarketDataProvider',
        file='external/market_data_provider.py',
        provides=[
            "bid_ask_data", "historical_prices", "indicators",
            "market_data", "market_liquidity",
            "module_insights", "multi_timeframe_data", "ohlcv_data", "price_data", "prices",
            "session_type", "step_idx", "symbols",
            "technical_indicators", "timestamp", "trading_session",

            "volatility", "volatility_data", "volatility_level",
            "volume_data", "liquidity_data",
            "volatility_level_by_instrument", "volatility_by_instrument",
            "market_data_latest", "atr_values", "session_info",

            "market_data_XAUUSD_M15", "market_data_XAUUSD_H1", "market_data_XAUUSD_H4", "market_data_XAUUSD_D1",
            "universe", "watched_instruments",
        ],
        requires=[],
        meta={'is_voting_member': False, 'thesis_required': False, 'explainable': False,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'external', 'version': '1.0.1'}
    ),


    'SessionManager': ModuleContract(
        name='SessionManager',
        file='external/session_manager.py',
        provides=[
            'consensus_data', 'emergency_mode', 'episode_data', 'episode_summary',
            'market_open', 'memory_usage', 'mistakes', 'module_performance', 'performance_metrics',
            'performance_data',
            'playbook_entries', 'playbook_memory', 'session_pnl_data', 'session_context', 'session_metrics',
            'system_alerts', 'session_health', 'system_performance', 'system_health',
            'environment_config', 'execution_mode',

            'session_canonical_by_instrument',

            'daily_pnl',

            'prop_firm_status', 'prop_firm_state'
        ],
        requires=[],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'external', 'version': '3.0.0'}
    ),


    'PositionManager': ModuleContract(
        name='PositionManager',
        file='position/position_logic.py',


        provides=['position_decisions', 'position_health', 'portfolio_state', 'order_queue', 'position_manager_data'],
        requires=['instrument_signals', 'trade_vote_v2', 'kernel_decision', 'environment_config', 'indicators', 'liquidity_capabilities', 'liquidity_score',
                  'market_context', 'market_data', 'market_liquidity',
                  'market_regime', 'price_data', 'prices', 'technical_indicators',
                  'time_risk_analysis', 'volatility_data',
                  'memory_gate', 'playbook_recall', 'intuition_vector', 'danger_zones', 'mistake_avoidance',

                  'ppo_final_decision', 'ppo_gate_passed', 'ppo_position_size',

                  'position_limits'],
        meta={'is_voting_member': False, 'thesis_required': True, 'explainable': True,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'position', 'version': '3.1.3'}
    ),

    'Executor': ModuleContract(
        name='Executor',
        file='executor/executor.py',


        provides=['positions', 'trades', 'recent_trades',
                  'order_data', 'execution_data', 'execution_reports',
                  'portfolio_metrics', 'trading_result', 'current_pnl',
                  'trade_data', 'market_state', 'position_data',
                  'current_positions', 'pnl_data', 'closed_positions',
                  'live_adapter_status', 'pending_orders', 'account_state',
                  'position_focus_context'],


        requires=['prices', 'price_data', 'environment_config', 'step_idx', 'execution_mode'],
        meta={'is_voting_member': False, 'thesis_required': False, 'explainable': True,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'executor', 'version': '1.0.0'}
    ),

    'TradingModeManager': ModuleContract(
        name='TradingModeManager',
        file='trading_modes/trading_mode.py',


        provides=['decision_factors', 'mode_config', 'mode_effectiveness', 'mode_stats', 'mode_thresholds',
                  'trading_mode', 'trading_mode_manager_initialization'],
        requires=[

            'market_context', 'market_regime', 'positions', 'recent_trades', 'risk_metrics',
            'session_metrics', 'strategy_performance', 'trading_performance', 'volatility_data', 'votes',

            'closed_positions',

            'execution_quality', 'risk_alerts', 'anomaly_detection', 'portfolio_risk', 'drawdown_risk',
            'risk_scaling', 'anomaly_score', 'consensus_score', 'consensus_quality', 'committee_confidence',
            'committee_decision', 'collusion_score', 'member_confidences', 'market_predictions',

            'theme_detection', 'liquidity_score', 'regime_prediction',
            'prediction_confidence', 'bias_analysis', 'adaptation_recommendations', 'market_thesis'

        ],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'trading_modes', 'version': '3.1.0'}
    ),


    'RiskAdjustedReward': ModuleContract(
        name='RiskAdjustedReward',
        file='reward/risk_adjusted_reward.py',
        provides=['reward_analytics', 'reward_components', 'reward_performance', 'shaped_reward'],
        requires=['environment_config', 'market_context', 'mistake_memory', 'performance_data',
                  'risk_metrics', 'trade_data'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'reward', 'version': '4.0.0'}
    ),


    'VisualizationInterface': ModuleContract(
        name='VisualizationInterface',
        file='visualization/visualization_interface.py',
        provides=['alert_timeline', 'analytics_reports', 'dashboard_data', 'streaming_data',
                  'system_status', 'visualization_data'],
        requires=['consensus_data', 'market_data', 'module_performance', 'positions', 'recent_trades',
                  'risk_metrics', 'system_alerts', 'trading_performance'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'visualization', 'version': '3.0.0'}
    ),

    'TradeMapVisualizer': ModuleContract(
        name='TradeMapVisualizer',
        file='visualization/trade_map_visualizer.py',
        provides=['chart_cache', 'chart_history', 'chart_statistics', 'dashboard_charts',
                  'performance_charts', 'trade_charts', 'visualization_reports'],
        requires=['consensus_data', 'market_data', 'module_performance', 'positions', 'recent_trades',
                  'risk_metrics', 'trading_performance'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'visualization', 'version': '3.0.0'}
    ),


}


def contract_params(name: str) -> Dict[str, Any]:
    mc = CONTRACTS.get(name)
    if not mc:
        raise KeyError(f"No contract found for module: {name}")
    kwargs = {
        "name": mc.name,
        "provides": list(mc.provides),
        "requires": list(mc.requires),
    }
    kwargs.update(mc.meta)
    return kwargs


def _as_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in {"true", "1", "yes", "y", "on"}:
            return True
        if vv in {"false", "0", "no", "n", "off"}:
            return False
    return default


def _as_int(v, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def module_args(name: str, **overrides: Any) -> Dict[str, Any]:
    raw = contract_params(name)
    out: Dict[str, Any] = {
        "name": raw.get("name", name),
        "provides": list(raw.get("provides", [])),
        "requires": list(raw.get("requires", [])),
        "version": str(raw.get("version", "0.0.0")),
        "category": str(raw.get("category", "")),
        "thesis_required": _as_bool(raw.get("thesis_required", False), False),
        "health_monitoring": _as_bool(raw.get("health_monitoring", False), False),
        "performance_tracking": _as_bool(raw.get("performance_tracking", False), False),
        "is_voting_member": _as_bool(raw.get("is_voting_member", False), False),
    }
    if "timeout_ms" in raw:
        out["timeout_ms"] = _as_int(raw["timeout_ms"], 0)
    out.update(overrides)
    return out


def _indexes() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    providers: Dict[str, List[str]] = {}
    consumers: Dict[str, List[str]] = {}
    for mname, mc in CONTRACTS.items():
        for k in mc.provides:
            providers.setdefault(k, []).append(mname)
        for k in mc.requires:
            consumers.setdefault(k, []).append(mname)
    return providers, consumers


def audit_contracts() -> Dict[str, Any]:
    providers, consumers = _indexes()
    duplicates = {k: v for k, v in providers.items() if len(v) > 1}
    missing = {k: consumers[k] for k in consumers.keys() if k not in providers}
    unused = [k for k in providers.keys() if k not in consumers]
    return {
        "duplicate_providers": duplicates,
        "missing_providers": missing,
        "unused_provided_keys": sorted(unused),
    }
