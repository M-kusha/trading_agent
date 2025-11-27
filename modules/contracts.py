from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Set


@dataclass
class ModuleContract:
    name: str
    file: str = ""  # path relative to modules/
    provides: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE-WRITER OWNERSHIP (canonical selections)
# - trading_result      → Executor
# - performance_data    → SessionManager (aggregate), NOT TradingModeManager
# - market_regime       → UnifiedMarket
# - training_metrics    → PPOAgent
# - quality_metrics     → ExecutionQualityMonitor
# - consensus_quality_metrics → ConsensusDetector
# ──────────────────────────────────────────────────────────────────────────────

CONTRACTS: Dict[str, ModuleContract] = {
    # ═════════════════════════════════ RISK ══════════════════════════════════
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
        # NOTE: 'position_data' is provided by PositionManager
        # NOTE: Memory signals (memory_gate, danger_zones, etc.) used for risk factor adjustment
        requires=['anomaly_detection', 'compliance', 'execution_quality', 'market_context', 'market_data',
                  'market_regime', 'performance_data', 'portfolio_risk', 'position_data', 'risk_data',
                  'memory_gate', 'danger_zones', 'mistake_avoidance', 'intuition_vector'],
        meta={'is_voting_member': True, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'risk', 'version': '4.1.0'}
    ),

    'EnhancedAnomalyDetector': ModuleContract(
        name='EnhancedAnomalyDetector',
        file='risk/anomaly_detector.py',
        provides=['anomaly_alerts', 'anomaly_detection', 'anomaly_score', 'detection_analytics', 'anomaly_detector',
                  'EnhancedAnomalyDetector_voting_proposal', 'EnhancedAnomalyDetector_confidence'],
        requires=['market_context', 'market_data', 'performance_data', 'risk_data', 'trading_data'],
        meta={'is_voting_member': True, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'risk', 'version': '4.0.0'}
    ),

    'ExecutionQualityMonitor': ModuleContract(
        name='ExecutionQualityMonitor',
        file='risk/execution_quality_monitor.py',  # fixed path (was incorrectly under voting/)
        provides=['execution_alerts', 'execution_analytics', 'execution_quality', 'quality_metrics',
                  'ExecutionQualityMonitor_voting_proposal', 'ExecutionQualityMonitor_confidence'],
        requires=['execution_data', 'market_context', 'market_data', 'order_data', 'trade_data'],
        meta={'is_voting_member': True, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'risk', 'version': '4.0.0'}
    ),

    'PortfolioRiskSystem': ModuleContract(
        name='PortfolioRiskSystem',
        file='risk/portfolio_risk_system.py',  # fixed path spelling
        provides=['portfolio_risk', 'portfolio_risk_proposal', 'position_limits', 'risk_data', 'risk_metrics',
                  'risk_score', 'risk_signals', 'portfolio_trade_data', 'trading_data',
                  'PortfolioRiskSystem_voting_proposal', 'PortfolioRiskSystem_confidence'],
        requires=['market_context', 'market_data', 'positions'],
        meta={'is_voting_member': True, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'risk', 'version': '4.0.0'}
    ),

    # ═══════════════════════════════ FEATURES ════════════════════════════════
    'AdvancedFeatureEngine': ModuleContract(
        name='AdvancedFeatureEngine',
        file='features/advanced_feature_engine.py',
        provides=['advanced_features', 'feature_analysis', 'feature_engine_capabilities',
                  'feature_error', 'feature_health', 'feature_thesis',
                  'features', 'market_features', 'price_features',
                  'advanced_features_H1', 'advanced_features_H4', 'advanced_features_D1'],
        requires=['historical_prices', 'market_data', 'multi_timeframe_data', 'ohlcv_data', 'price_data'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'features', 'version': '3.0.0'}
    ),

    'MultiScaleFeatureEngine': ModuleContract(
        name='MultiScaleFeatureEngine',
        file='features/multiscale_feature_engine.py',
        provides=['attention_weights', 'feature_fusion', 'multiscale_features', 'neural_capabilities',
                  'neural_embeddings', 'neural_health'],
        requires=['advanced_features', 'market_data'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'features', 'version': '3.0.0'}
    ),

    # ═════════════════════════════════ META ══════════════════════════════════
    'MetaCognitivePlanner': ModuleContract(
        name='MetaCognitivePlanner',
        file='meta/metacognitive_planner.py',
        provides=['adaptation_metrics', 'planning_status', 'strategic_insights', 'tactical_recommendations'],
        requires=['actions', 'market_context', 'market_data', 'market_regime', 'performance_metrics',
                  'regime_data', 'trades', 'volatility_adjustment'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'meta', 'version': '3.0.1'}
    ),

    'MetaRLController': ModuleContract(
        name='MetaRLController',
        file='meta/meta_rl_controller.py',  # fixed path
        provides=['agent_decisions', 'agents_performance', 'automation_status', 'controller_status',
                  'controller_training_overview', 'meta_signals', 'trading_signal', 'trading_signals'],
        requires=['actions', 'market_data', 'trades', 'training_signals'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'meta', 'version': '3.0.0'}
    ),

    'MetaAgent': ModuleContract(
        name='MetaAgent',
        file='meta/meta_agent.py',
        provides=['automation_decisions', 'automation_metrics', 'meta_performance', 'system_mode',
                  'MetaAgent_voting_proposal', 'MetaAgent_confidence'],
        requires=['market_context', 'risk_signals', 'system_performance', 'time_risk_analysis', 'training_metrics'],
        meta={'is_voting_member': True, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'meta', 'version': '3.0.0'}
    ),

    'PPOAgent': ModuleContract(
        name='PPOAgent',
        file='meta/ppo_agent.py',
        provides=['actions', 'agent_performance', 'observations', 'policy_actions', 'policy_gradients', 'rewards',
                  'training_data', 'training_metrics', 'training_signals',
                  'PPOAgent_voting_proposal', 'PPOAgent_confidence'],
        requires=['market_data'],
        meta={'is_voting_member': True, 'thesis_required': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'meta', 'version': '3.0.0'}
    ),

    'PPOLagAgent': ModuleContract(
        name='PPOLagAgent',
        file='meta/ppo_lag_agent.py',
        provides=['agent_status', 'market_adaptation', 'position_metrics', 'ppo_lag_training_metrics'],
        requires=['actions', 'market_data', 'trades', 'training_signals'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'meta', 'version': '3.0.0'}
    ),

    # ═══════════════════════════════ STRATEGY ════════════════════════════════
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

    'OpponentModeEnhancer': ModuleContract(
        name='OpponentModeEnhancer',
        file='strategy/opponent_mode_enhancer.py',
        provides=['market_mode_detection', 'mode_analysis', 'mode_performance', 'mode_recommendations',
                  'mode_weights', 'opponent_mode_enhancer_initialization', 'strategy_adaptation'],
        requires=['market_context', 'market_data', 'market_regime', 'price_data', 'recent_trades', 'session_metrics',
                  'technical_indicators', 'trading_performance', 'volatility_data'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'strategy', 'version': '3.0.0'}
    ),

    'PlaybookClusterer': ModuleContract(
        name='PlaybookClusterer',
        file='strategy/playbook_clusterer.py',
        # FIX: Renamed pattern_analysis → playbook_patterns to avoid conflict with UnifiedMemory's canonical pattern_analysis
        provides=['cluster_analysis', 'cluster_effectiveness', 'cluster_recommendations', 'cluster_weights',
                  'clustering_health', 'clustering_thesis', 'playbook_clusterer_initialization', 'playbook_patterns'],
        requires=['market_context', 'market_data', 'market_regime', 'playbook_memory', 'recent_trades',
                  'session_metrics', 'trading_performance', 'volatility_data'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'strategy', 'version': '3.0.0'}
    ),

    'StrategyGenomePool': ModuleContract(
        name='StrategyGenomePool',
        file='strategy/strategy_genome_pool.py',
        provides=['best_genome', 'evolution_analytics', 'genome_analysis', 'genome_recommendations',
                  'genome_weights', 'population_metrics', 'strategy_genome_pool_initialization'],
        requires=['market_context', 'market_data', 'market_regime', 'recent_trades', 'risk_data',
                  'session_metrics', 'trading_performance', 'volatility_data'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'strategy', 'version': '3.0.0'}
    ),

    'StrategyIntrospector': ModuleContract(
        name='StrategyIntrospector',
        file='strategy/strategy_introspector.py',
        provides=['adaptation_recommendations', 'behavior_patterns', 'introspection_metrics', 'module_data',
                  'strategy_analysis', 'strategy_introspector_initialization', 'strategy_performance',
                  'strategy_profiles', 'trading_performance'],
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
        requires=['episode_summary', 'market_context', 'performance_data', 'recent_trades', 'risk_metrics', 'trading_session'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),

    # ═══════════════════════════════ VOTING ══════════════════════════════════
    # LEGACY VOTING MODULES - DEPRECATED (code in voting/legacy/, commented out)
    # These contracts are preserved for reference but modules are disabled.
    # Use the new UNIFIED VOTING (v5.0) modules below instead.
    
    # 'EnhancedThemeExpert': ModuleContract(
    #     name='EnhancedThemeExpert',
    #     file='voting/legacy/voting_wrappers.py',  # DEPRECATED
    #     provides=[
    #         'agreement_score', 'consensus_direction', 'member_confidences', 'raw_proposals',
    #         'theme_voting_proposal', 'theme_confidence',
    #         'EnhancedThemeExpert_voting_proposal', 'EnhancedThemeExpert_confidence'
    #     ],
    #     requires=['market_data', 'price_data', 'technical_indicators', 'market_regime', 'market_open'],
    #     meta={'is_voting_member': True, 'thesis_required': True, 'explainable': True, 'health_monitoring': True,
    #           'performance_tracking': True, 'category': 'voting', 'version': '4.0.0'}
    # ),

    # 'EnhancedSeasonalityRiskExpert': ModuleContract(
    #     name='EnhancedSeasonalityRiskExpert',
    #     file='voting/legacy/voting_wrappers.py',  # DEPRECATED
    #     provides=['seasonality_risk_analysis', 'seasonal_voting_proposal', 'seasonal_confidence',
    #               'seasonality_analysis', 'seasonality_voting_proposal', 'seasonality_confidence',
    #               'EnhancedSeasonalityRiskExpert_confidence', 'EnhancedSeasonalityRiskExpert_voting_proposal'],
    #     requires=['market_data', 'price_data', 'technical_indicators', 'market_regime', 'market_open'],
    #     meta={'is_voting_member': False, 'thesis_required': True, 'explainable': True, 'health_monitoring': True,
    #           'performance_tracking': True, 'category': 'voting', 'version': '4.0.0'}
    # ),

    # 'EnhancedVotingCommitteeCoordinator': ModuleContract(
    #     name='EnhancedVotingCommitteeCoordinator',
    #     file='voting/legacy/voting_wrappers.py',  # DEPRECATED
    #     provides=['committee_decision', 'committee_confidence', 'member_proposals', 'performance_feedback',
    #               'time_of_day', 'votes', 'committee_votes', 'voting_summary', 'voting_weights',
    #               'strategy_arbiter_weights', 'committee_consensus', 'member_confidences_ordered',
    #               'expert_votes', 'committee_members', 'proposal_vectors', 'signals', 'committee_decision_id'],
    #     requires=['emergency_mode', 'market_context', 'market_open', 'market_regime', 'portfolio_state',
    #               'recent_trades', 'risk_data', 'risk_score', 'session_type', 'system_health',
    #               'theme_detection', 'volatility_data',
    #               'memory_vote', 'playbook_recall', 'intuition_vector',
    #               'DynamicRiskController_voting_proposal', 'DynamicRiskController_confidence',
    #               'EnhancedAnomalyDetector_voting_proposal', 'EnhancedAnomalyDetector_confidence',
    #               'EnhancedSeasonalityRiskExpert_voting_proposal', 'EnhancedSeasonalityRiskExpert_confidence',
    #               'EnhancedThemeExpert_voting_proposal', 'EnhancedThemeExpert_confidence',
    #               'ExecutionQualityMonitor_voting_proposal', 'ExecutionQualityMonitor_confidence',
    #               'MetaAgent_voting_proposal', 'MetaAgent_confidence',
    #               'PortfolioRiskSystem_voting_proposal', 'PortfolioRiskSystem_confidence',
    #               'PPOAgent_voting_proposal', 'PPOAgent_confidence'],
    #     meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
    #           'category': 'voting', 'version': '4.0.0'}
    # ),

    # 'ConsensusDetector': ModuleContract(
    #     name='ConsensusDetector',
    #     file='voting/legacy/consensus_detector.py',  # DEPRECATED
    #     provides=['confidence_consensus', 'consensus_components', 'consensus_detector_initialization',
    #               'consensus_quality', 'consensus_recommendations', 'consensus_score', 'consensus_trends',
    #               'directional_consensus', 'magnitude_consensus', 'member_contributions',
    #               'consensus_quality_metrics', 'consensus_decision_id'],
    #     requires=['agreement_score', 'consensus_direction', 'market_context', 'market_regime', 'member_confidences',
    #               'raw_proposals', 'volatility_data'],
    #     meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
    #           'category': 'voting', 'version': '3.0.0'}
    # ),

    # 'AlternativeRealitySampler': ModuleContract(
    #     name='AlternativeRealitySampler',
    #     file='voting/legacy/alternative_reality_sampler.py',  # DEPRECATED
    #     provides=['alternative_reality_sampler_initialization', 'alternative_samples', 'confidence_bounds',
    #               'diversity_score', 'effective_samples', 'sampling_recommendations', 'sampling_stats',
    #               'sampling_uncertainty', 'sampling_decision_id', 'sampling_fragility'],
    #     requires=['agreement_score', 'consensus_direction', 'market_context', 'market_regime', 'recent_trades',
    #               'session_metrics', 'strategy_arbiter_weights', 'volatility_data', 'votes', 'voting_summary'],
    #     meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
    #           'category': 'voting', 'version': '3.0.0'}
    # ),

    # 'TimeHorizonAligner': ModuleContract(
    #     name='TimeHorizonAligner',
    #     file='voting/legacy/time_horizon_aligner.py',  # DEPRECATED
    #     provides=['aligned_weights', 'horizon_distances', 'horizon_multipliers', 'horizon_alignment',
    #               'alignment_quality', 'adaptation_status', 'horizon_decision_id', 'horizon_alignment_meta'],
    #     requires=['voting_weights', 'market_regime', 'session_type', 'volatility_data', 'market_context'],
    #     meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
    #           'category': 'voting', 'version': '3.0.0'}
    # ),

    # 'StrategyArbiter': ModuleContract(
    #     name='StrategyArbiter',
    #     file='voting/legacy/strategy_arbiter.py',  # DEPRECATED
    #     provides=['alpha_weights', 'arbiter_recommendations', 'decision_statistics', 'gate_decision',
    #               'instrument_signals', 'instruments',
    #               'member_performance', 'member_weights', 'proposal_analysis',
    #               'strategy_arbiter_initialization', 'strategy_weights', 'voting_quality',
    #               'expert_performance', 'arbiter_decision_id'],
    #     requires=['collusion_score', 'consensus_score', 'horizon_alignment', 'market_context', 'market_regime',
    #               'member_confidences', 'member_proposals', 'recent_trades', 'session_data', 'volatility_data',
    #               'universe', 'watched_instruments',
    #               'memory_gate', 'danger_zones', 'mistake_avoidance', 'playbook_recall'],
    #     meta={'category': 'voting', 'version': '3.0.0'}
    # ),

    # 'VotingKernel': ModuleContract(
    #     name='VotingKernel',
    #     file='voting/legacy/voting_kernel.py',  # DEPRECATED
    #     provides=['decision_coordination', 'voting_consensus', 'consensus_summary',
    #               'voting_metrics', 'decision_bundle', 'trade_vote_v2', 'fragility',
    #               'decision_id', 'tick_ts', 'kernel_decision_id', 'kernel_tick_ts'],
    #     requires=['market_data', 'price_data', 'technical_indicators', 'market_regime', 'portfolio_state'],
    #     meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
    #           'category': 'voting', 'version': '1.0.0'}
    # ),

    # 'CollusionAuditor': ModuleContract(
    #     name='CollusionAuditor',
    #     file='voting/legacy/collusion_auditor.py',  # DEPRECATED
    #     provides=['audit_recommendations', 'behavioral_profiles', 'collusion_alerts', 'collusion_auditor_initialization',
    #               'collusion_score', 'coordination_events', 'detection_statistics', 'member_independence_scores',
    #               'suspicious_pairs', 'collusion_decision_id'],
    #     requires=['agreement_score', 'consensus_direction', 'market_context', 'market_regime', 'member_confidences',
    #               'raw_proposals', 'recent_trades', 'strategy_arbiter_weights', 'volatility_data', 'votes',
    #               'voting_summary'],
    #     meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
    #           'category': 'voting', 'version': '3.0.0'}
    # ),

    # ════════════════════════ UNIFIED VOTING (v5.0) ══════════════════════════
    # NEW modular voting architecture - self-contained, replaces all legacy voting modules
    # See modules/voting/core/, experts/, stages/, pipeline/, utils/

    'ThemeExpert': ModuleContract(
        name='ThemeExpert',
        file='voting/experts/theme.py',
        provides=[
            'ThemeExpert_voting_proposal', 'ThemeExpert_confidence',
            'theme_voting_proposal', 'theme_confidence',
            'theme_analysis', 'agreement_score',
            'theme_volatility_regime', 'theme_trend_regime',
            'theme_risk_regime', 'theme_composite_score',
            'theme_expert_analysis', 'theme_expert_thesis'  # backward compat aliases
        ],
        requires=['market_data', 'features'],
        meta={'is_voting_member': True, 'thesis_required': True, 'explainable': True,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'voting', 'version': '5.1.0'}
    ),

    'SeasonalityRiskExpert': ModuleContract(
        name='SeasonalityRiskExpert',
        file='voting/experts/seasonality.py',
        provides=[
            'SeasonalityRiskExpert_voting_proposal', 'SeasonalityRiskExpert_confidence',
            'seasonality_voting_proposal', 'seasonality_confidence',
            'seasonal_voting_proposal', 'seasonal_confidence',
            'seasonal_session', 'seasonal_dow_bias', 'seasonal_monthly_pattern',
            'seasonal_composite_score', 'seasonal_rollover_risk', 'seasonal_weekend_risk',
            'seasonality_risk_analysis', 'seasonality_analysis',
            'seasonality_expert_analysis', 'seasonality_expert_thesis'  # backward compat aliases
        ],
        requires=['market_data', 'features'],
        meta={'is_voting_member': True, 'thesis_required': True, 'explainable': True,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'voting', 'version': '5.1.0'}
    ),

    'MomentumExpert': ModuleContract(
        name='MomentumExpert',
        file='voting/experts/momentum.py',
        provides=[
            'MomentumExpert_voting_proposal', 'MomentumExpert_confidence',
            'momentum_voting_proposal', 'momentum_confidence',
            'momentum_analysis'
        ],
        requires=['market_data', 'prices', 'technical_indicators'],
        meta={'is_voting_member': True, 'thesis_required': True, 'explainable': True,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'voting', 'version': '5.0.0'}
    ),

    'TrendExpert': ModuleContract(
        name='TrendExpert',
        file='voting/experts/trend.py',
        provides=[
            'TrendExpert_voting_proposal', 'TrendExpert_confidence',
            'trend_voting_proposal', 'trend_confidence',
            'trend_analysis'
        ],
        requires=['market_data', 'prices', 'technical_indicators'],
        meta={'is_voting_member': True, 'thesis_required': True, 'explainable': True,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'voting', 'version': '5.0.0'}
    ),

    'CommitteeCoordinator': ModuleContract(
        name='CommitteeCoordinator',
        file='voting/stages/committee.py',
        # Provides same keys as old EnhancedVotingCommitteeCoordinator for compatibility
        provides=[
            'committee_votes', 'committee_summary', 'committee_decision_id',
            'raw_proposals', 'member_confidences', 'voting_weights',
            # Backward compatibility
            'committee_decision', 'committee_confidence', 'votes', 'voting_summary',
            'strategy_arbiter_weights', 'committee_consensus'
        ],
        requires=[
            'ThemeExpert_voting_proposal', 'ThemeExpert_confidence',
            'SeasonalityRiskExpert_voting_proposal', 'SeasonalityRiskExpert_confidence',
            'MomentumExpert_voting_proposal', 'MomentumExpert_confidence',
            'TrendExpert_voting_proposal', 'TrendExpert_confidence',
            'DynamicRiskController_voting_proposal', 'DynamicRiskController_confidence',
            'PPOAgent_voting_proposal', 'PPOAgent_confidence',
            'market_regime', 'volatility_data'
        ],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'voting', 'version': '5.0.0'}
    ),

    'ConsensusAnalyzer': ModuleContract(
        name='ConsensusAnalyzer',
        file='voting/stages/consensus.py',
        # Provides same keys as old ConsensusDetector for compatibility
        provides=[
            'consensus_result', 'agreement_score', 'consensus_direction',
            'consensus_confidence', 'consensus_thesis',
            # Backward compatibility
            'consensus_score', 'consensus_components', 'consensus_quality',
            'directional_consensus', 'magnitude_consensus', 'consensus_decision_id'
        ],
        requires=['committee_votes', 'raw_proposals', 'member_confidences', 'voting_weights',
                  'market_regime', 'volatility_data'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'voting', 'version': '5.0.0'}
    ),

    'CollusionDetector': ModuleContract(
        name='CollusionDetector',
        file='voting/stages/collusion.py',
        # Provides same keys as old CollusionAuditor for compatibility
        provides=[
            'collusion_result', 'collusion_detected', 'collusion_score',
            'suspicious_pairs', 'collusion_thesis',
            # Backward compatibility
            'collusion_alerts', 'member_independence_scores', 'collusion_decision_id'
        ],
        requires=['committee_votes', 'raw_proposals', 'member_confidences',
                  'agreement_score', 'market_regime'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'voting', 'version': '5.0.0'}
    ),

    'HorizonAligner': ModuleContract(
        name='HorizonAligner',
        file='voting/stages/horizon.py',
        # Provides same keys as old TimeHorizonAligner for compatibility
        provides=[
            'horizon_weights', 'horizon_alignment', 'aligned_weights', 'horizon_thesis',
            # Backward compatibility
            'horizon_distances', 'horizon_multipliers', 'alignment_quality',
            'adaptation_status', 'horizon_decision_id'
        ],
        requires=['voting_weights', 'market_regime', 'session_type', 'volatility_data'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'voting', 'version': '5.0.0'}
    ),

    'UncertaintySampler': ModuleContract(
        name='UncertaintySampler',
        file='voting/stages/uncertainty.py',
        # Provides same keys as old AlternativeRealitySampler for compatibility
        provides=[
            'uncertainty_result', 'sampling_uncertainty', 'fragility_score',
            'effective_samples', 'uncertainty_thesis',
            # Backward compatibility
            'alternative_samples', 'confidence_bounds', 'diversity_score',
            'sampling_decision_id', 'sampling_fragility', 'fragility'
        ],
        requires=['committee_votes', 'consensus_result', 'agreement_score',
                  'market_regime', 'volatility_data'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'voting', 'version': '5.0.0'}
    ),

    'FinalArbiter': ModuleContract(
        name='FinalArbiter',
        file='voting/stages/arbiter.py',
        # Provides same keys as old StrategyArbiter for compatibility
        provides=[
            'final_decision', 'trade_vote', 'gate_decision', 'arbiter_thesis',
            'decision_confidence', 'decision_rationale',
            # Backward compatibility
            'arbiter_recommendations', 'instrument_signals', 'voting_quality',
            'member_weights', 'arbiter_decision_id'
        ],
        requires=['consensus_result', 'collusion_result', 'uncertainty_result',
                  'horizon_alignment', 'market_regime', 'volatility_data',
                  'memory_gate', 'danger_zones'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'voting', 'version': '5.0.0'}
    ),

    'SlimVotingKernel': ModuleContract(
        name='SlimVotingKernel',
        file='voting/pipeline/kernel.py',
        # Orchestrates the entire unified voting pipeline
        # Provides same keys as old VotingKernel for compatibility
        provides=[
            'kernel_decision', 'trade_vote_v2', 'decision_bundle',
            'voting_consensus', 'consensus_summary', 'voting_metrics',
            'decision_id', 'tick_ts', 'kernel_decision_id', 'kernel_tick_ts',
            'pipeline_status', 'pipeline_thesis',
            # Backward compatibility
            'decision_coordination', 'fragility'
        ],
        requires=[
            # Market data (portfolio_state removed to avoid circular dep with PositionManager)
            'market_data', 'price_data', 'technical_indicators', 'market_regime',
            'session_type', 'volatility_data', 'timestamp',
            # Voting stage outputs - kernel reads these from bus
            'committee_votes', 'committee_decision', 'committee_confidence', 'raw_proposals',
            'consensus_result', 'consensus_score', 'agreement_score',
            'collusion_result', 'collusion_detected', 'collusion_score',
            'horizon_alignment', 'aligned_weights',
            'uncertainty_result', 'fragility',
            'final_decision', 'gate_decision'
        ],
        meta={
            'thesis_required': True, 'explainable': True,
            'health_monitoring': True, 'performance_tracking': True,
            'category': 'voting', 'version': '5.0.0',
            'orchestrates': [
                'ThemeExpert', 'SeasonalityRiskExpert', 'CommitteeCoordinator',
                'ConsensusAnalyzer', 'CollusionDetector', 'HorizonAligner',
                'UncertaintySampler', 'FinalArbiter'
            ]
        }
    ),

    # ═══════════════════════════════ MARKET ══════════════════════════════════
    'UnifiedMarketModule': ModuleContract(
        name='UnifiedMarketModule',
        file='market/market_module.py',
        provides=[
            # Fractal / Regime
            'fractal_metrics', 'market_regime', 'regime_data', 'regime_strength', 'timestamps', 'trend_direction',
            # Liquidity
            'liquidity_capabilities', 'liquidity_prediction', 'liquidity_score', 'liquidity_thesis',
            'liquidity_score_by_instrument',  # FIX: Added for PositionManager instrument-level liquidity lookup
            'market_depth', 'session_data', 'spread_analysis', 'trading_sessions',
            # Theme
            'market_theme', 'theme_detection', 'theme_detector_health', 'theme_detector_status',
            'theme_strength', 'theme_transition',
            # Regime performance matrix
            'backtesting_data', 'regime_accuracy', 'regime_analysis', 'regime_matrix_analysis',
            'regime_matrix_health', 'regime_matrix_status', 'regime_performance', 'regime_prediction',
            'stress_test_results',
            # Time-aware risk scaling
            'risk_scaling_factor', 'session_risk', 'time_risk_health', 'time_risk_status',
            'time_risk_analysis', 'volatility_adjustment',
            # Unified extras
            'unified_market_analysis', 'market_analysis_thesis',
            # Market context (canonical owner - contains regime/volatility from analysis)
            'market_context'
        ],
        requires=['bid_ask_data', 'historical_prices', 'market_data',
                  'multi_timeframe_data', 'technical_indicators',
                  'timestamp', 'volatility_data', 'volatility_level'],
        meta={'is_voting_member': False, 'thesis_required': True, 'explainable': True, 'health_monitoring': True,
              'performance_tracking': True, 'category': 'market', 'version': '4.0.0'}
    ),

    # ═════════════════════════════ EXTERNAL / IO ═════════════════════════════
    'MarketDataProvider': ModuleContract(
        name='MarketDataProvider',
        file='external/market_data_provider.py',
        provides=[
            # NOTE: Removed stale placeholder keys (alerts, economic_calendar, environment,
            # input1, input2, learning_context, learning_status, macro_data, market_conditions,
            # step_data, strategy_status) - these were empty dicts/lists causing stale warnings.
            # Consumers handle missing keys with fallbacks or get data from other providers.
            'bid_ask_data', 'historical_prices', 'indicators',
            'market_data', 'market_liquidity',
            'module_insights', 'multi_timeframe_data', 'ohlcv_data', 'price_data', 'prices',
            'session_type', 'step_idx', 'symbols',
            'technical_indicators', 'timestamp', 'trading_session', 'volatility', 'volatility_data',
            'volatility_level', 'volume_data', 'liquidity_data',
            # Specific instrument data
            'market_data_EUR_USD_H1', 'market_data_EUR_USD_H4', 'market_data_EUR_USD_D1',
            'market_data_XAU_USD_H1', 'market_data_XAU_USD_H4', 'market_data_XAU_USD_D1',
            'universe', 'watched_instruments'
        ],
        requires=[],
        meta={'is_voting_member': False, 'thesis_required': False, 'explainable': False,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'external', 'version': '1.0.1'}
    ),

    # NOTE: NewsSentimentModule is currently DISABLED (entire file commented out).
    # Keep contract entry for future re-enablement but mark as disabled.
    'NewsSentimentModule': ModuleContract(
        name='NewsSentimentModule',
        file='external/news_sentiment.py',
        provides=['news_sentiment', 'news_summary', 'sentiment_confidence', 'sentiment_trend'],
        requires=['market_data', 'symbols', 'trading_session'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'external', 'version': '3.0.0', 'disabled': True}
    ),

    'SessionManager': ModuleContract(
        name='SessionManager',
        file='external/session_manager.py',
        provides=[
            'consensus_data', 'emergency_mode', 'episode_data', 'episode_summary',
            'market_open', 'memory_usage', 'mistakes', 'module_performance', 'performance_metrics',
            'performance_data',  # canonical owner selected
            'playbook_entries', 'playbook_memory', 'session_pnl_data', 'session_context', 'session_metrics',
            'system_alerts', 'session_health', 'system_performance', 'system_health',  # FIX #3: Added system_health
            'environment_config', 'execution_mode'    # fills gap for PM/Executor; may be moved to a dedicated Environment module
        ],
        requires=[],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'external', 'version': '3.0.0'}
    ),

    # ═════════════════════════════ EXECUTION / POSITION ══════════════════════
    'PositionManager': ModuleContract(
        name='PositionManager',
        file='position/position_logic.py',
        # FIX: Renamed position_data → position_manager_data to avoid conflict with Executor's canonical position_data
        # NOTE: Memory signals used for veto gate and position sizing intelligence
        # FIX: Uses trade_vote_v2 from SlimVotingKernel as primary signal source
        # NOTE: Removed market_conditions - uses market_context fallback
        provides=['position_decisions', 'position_health', 'portfolio_state', 'order_queue', 'position_manager_data'],
        requires=['trade_vote_v2', 'kernel_decision', 'environment_config', 'indicators', 'liquidity_capabilities', 'liquidity_score',
                  'market_context', 'market_data', 'market_liquidity',
                  'market_regime', 'price_data', 'prices', 'technical_indicators',
                  'time_risk_analysis', 'volatility_data',
                  'memory_gate', 'playbook_recall', 'intuition_vector', 'danger_zones', 'mistake_avoidance'],
        meta={'is_voting_member': False, 'thesis_required': True, 'explainable': True,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'position', 'version': '3.1.2'}
    ),

    'Executor': ModuleContract(
        name='Executor',
        file='executor/executor.py',
        # FIX: Added position_data as canonical provider (actual executed positions)
        # FIX: Added closed_positions (consumed by TrainingVisualizer for win rate tracking)
        # NOTE: Memory gate used for final safety veto on order execution
        # NOTE: order_queue is consumed but NOT required - Executor handles empty queue gracefully
        # This allows Executor to run in parallel with PositionManager (order_queue comes next cycle)
        provides=['positions', 'trades', 'recent_trades',
                  'order_data', 'execution_data', 'execution_reports',
                  'portfolio_metrics', 'trading_result', 'current_pnl',
                  'trade_data', 'market_state', 'position_data',
                  'current_positions', 'pnl_data', 'closed_positions',
                  'live_adapter_status', 'pending_orders', 'account_state'],
        requires=['prices', 'price_data', 'environment_config', 'step_idx', 'execution_mode'],
        meta={'is_voting_member': False, 'thesis_required': False, 'explainable': True,
              'health_monitoring': True, 'performance_tracking': True,
              'category': 'executor', 'version': '1.0.0'}
    ),

    'TradingModeManager': ModuleContract(
        name='TradingModeManager',
        file='trading_modes/trading_mode.py',
        # removed 'performance_data' to avoid duplicate writer with SessionManager
        # NOTE: Removed 'economic_calendar' - no longer provided by MarketDataProvider
        provides=['decision_factors', 'mode_config', 'mode_effectiveness', 'mode_stats', 'mode_thresholds',
                  'trading_mode', 'trading_mode_manager_initialization'],
        requires=[
            # Original required keys
            'market_context', 'market_regime', 'positions', 'recent_trades', 'risk_metrics',
            'session_metrics', 'strategy_performance', 'trading_performance', 'volatility_data', 'votes',
            # Enhanced integrations (optional but beneficial)
            'execution_quality', 'risk_alerts', 'anomaly_detection', 'portfolio_risk', 'drawdown_risk',
            'risk_scaling', 'anomaly_score', 'consensus_score', 'consensus_quality', 'committee_confidence',
            'committee_decision', 'collusion_score', 'member_confidences', 'market_predictions',
            # shadow_predictions is optional; defaults are set to avoid readiness failures
            'theme_detection', 'liquidity_score', 'regime_prediction',
            'prediction_confidence', 'bias_analysis', 'adaptation_recommendations', 'market_thesis'
            # Note: 'best_thesis' removed from required - it's optional with fallback logic in code
        ],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'trading_modes', 'version': '3.1.0'}
    ),

    # ═══════════════════════════════ REWARD ══════════════════════════════════
    'RiskAdjustedReward': ModuleContract(
        name='RiskAdjustedReward',
        file='reward/risk_adjusted_reward.py',
        provides=['reward_analytics', 'reward_components', 'reward_performance', 'shaped_reward'],
        requires=['environment_config', 'market_context', 'mistake_memory', 'performance_data',
                  'risk_metrics', 'trade_data'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'reward', 'version': '4.0.0'}
    ),

    # ═══════════════════════════════ AUDITING ════════════════════════════════
    'AuditingCoordinator': ModuleContract(
        name='AuditingCoordinator',
        file='auditing/auditing_coordinator.py',
        provides=['audit_metrics', 'audit_report', 'audit_status'],
        # FIX: Removed trading_signal - it's provided by MetaRLController which runs after this
        requires=['market_data', 'trades'],
        meta={'is_voting_member': False, 'explainable': True, 'category': 'auditing', 'version': '2.0.0'}
    ),

    'TradeExplanationAuditor': ModuleContract(
        name='TradeExplanationAuditor',
        file='auditing/trade_explanation_auditor.py',
        provides=['audit_alerts', 'explanation_metrics', 'trade_explanations'],
        # FIX: Removed trading_signal - it's provided by MetaRLController which runs after this
        requires=['market_data', 'trades'],
        meta={'is_voting_member': False, 'explainable': True, 'category': 'auditing', 'version': '2.0.0'}
    ),

    'TradeThesisTracker': ModuleContract(
        name='TradeThesisTracker',
        file='auditing/trade_thesis_tracker.py',
        provides=['thesis_alerts', 'thesis_analysis'],
        # FIX: Removed trading_signal - it's provided by MetaRLController which runs after this
        requires=['market_data', 'trades'],
        meta={'is_voting_member': False, 'explainable': True, 'category': 'auditing', 'version': '2.0.0'}
    ),

    # ═══════════════════════════ SIMULATION / MODELS ════════════════════════
    'OpponentSimulator': ModuleContract(
        name='OpponentSimulator',
        file='simulation/opponent_simulator.py',
        provides=['adversarial_scenarios', 'market_noise', 'market_perturbations', 'opponent_analysis',
                  'opponent_simulation', 'perturbation_history', 'simulated_prices', 'simulation_effects',
                  'simulation_statistics'],
        requires=['historical_prices', 'market_context', 'market_data', 'positions', 'prices',
                  'regime_data', 'session_data', 'volatility'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'simulation', 'version': '3.0.0'}
    ),

    'RoleCoach': ModuleContract(
        name='RoleCoach',
        file='simulation/role_coach.py',
        provides=['coaching_penalties', 'coaching_recommendations', 'coaching_results', 'coaching_statistics', 'compliance_tracking', 'discipline_assessment', 'discipline_penalty', 'performance_scoring', 'trade_limits'],
        requires=['market_context', 'pending_orders', 'positions', 'recent_trades', 'regime_data', 'risk_metrics', 'session_data', 'trading_performance'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'simulation', 'version': '3.0.0'}
    ),

    'ShadowSimulator': ModuleContract(
        name='ShadowSimulator',
        file='simulation/shadow_simulator.py',
        provides=['forward_projections', 'scenario_analysis', 'scenario_recommendations', 'shadow_predictions',
                  'shadow_simulation', 'simulation_confidence', 'simulation_predictions', 'strategy_simulations'],
        # NOTE: Removed 'environment' - no longer provided by MarketDataProvider
        requires=['votes', 'market_context', 'market_data', 'pending_orders', 'positions', 'prices',
                  'recent_trades', 'risk_metrics', 'trading_performance'],
        meta={'thesis_required': True, 'explainable': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'simulation', 'version': '3.0.0'}
    ),

    'EnhancedWorldModel': ModuleContract(
        name='EnhancedWorldModel',
        file='models/world_model.py',
        provides=['market_predictions', 'prediction_confidence', 'scenario_generation', 'world_model_analytics'],
        # FIX: Removed shadow_predictions - circular dependency with ShadowSimulator
        requires=['market_context', 'market_data', 'market_regime', 'performance_data',
                  'performance_metrics', 'regime_data', 'risk_data',
                  'time_risk_analysis', 'trading_data', 'volatility_adjustment'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'models', 'version': '4.0.1'}
    ),

    # ═══════════════════════════ VISUALIZATION ═══════════════════════════════
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

    # ═════════════════════════════ MEMORY (Unified) ══════════════════════════
    'UnifiedMemory': ModuleContract(
        name='UnifiedMemory',
        file='memory/unified_memory.py',
        provides=[
            # Overview metrics for frontend API
            'unified_metrics', 'unified_memory_status',
            # Replay
            'learning_progress', 'pattern_analysis', 'replay_sequences', 'sequence_quality',
            # Budget
            'allocation_strategy', 'budget_optimization', 'memory_allocation', 'memory_efficiency',
            # Compression
            'compressed_patterns', 'feature_importance', 'intuition_vector', 'memory_compression',
            # Mistakes
            'danger_zones', 'loss_prevention', 'mistake_avoidance', 'mistake_memory', 'pattern_recognition',
            # Neural
            'attention_retrieval', 'importance_scoring', 'memory_embedding', 'neural_memory',
            # Playbook
            'memory_analytics', 'pattern_memory', 'playbook_quality', 'playbook_recall',
            # NEW: Composite gate/vote signals for downstream consumers
            'memory_gate', 'memory_vote', 'memory_rationale',
            # NEW: Neural risk head output
            'neural_risk_hint'
        ],
        # FIX: Removed 'risk_data' from requires to break circular dependency:
        # PortfolioRiskSystem -> Executor -> PositionManager -> UnifiedMemory -> PortfolioRiskSystem
        # UnifiedMemory can get risk_data optionally from bus with fallback
        requires=['actions', 'episode_data', 'features', 'market_context', 'market_data',
                  'observations', 'prices', 'rewards', 'time_risk_analysis', 'trades'],
        meta={'thesis_required': True, 'health_monitoring': True, 'performance_tracking': True,
              'category': 'memory', 'version': '4.2.0'}
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Decorator helpers
# ──────────────────────────────────────────────────────────────────────────────

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
    """Return normalized kwargs for the @module decorator from registry."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Lightweight in-registry auditor
#   - duplicate providers per key
#   - missing providers for required keys
#   - provided but unused keys
# Returns a dict of lists so tests can assert.
# ──────────────────────────────────────────────────────────────────────────────

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
