from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class ModuleContract:
    name: str
    file: str = ""                      # relative to .../trading_agent/modules
    provides: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

CONTRACTS: Dict[str, ModuleContract] = {
    'ActiveTradeMonitor': ModuleContract(
        name='ActiveTradeMonitor',
        file='risk/active_trade_monitor.py',
        provides=['duration_alerts', 'position_duration_risk', 'position_tracking', 'trade_monitor_status'],
        requires=['market_context', 'positions', 'step_idx'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '3.0.0'}
    ),
    'AdvancedFeatureEngine': ModuleContract(
        name='AdvancedFeatureEngine',
        file='features/advanced_feature_engine.py',
        provides=['advanced_features', 'feature_analysis', 'feature_engine_capabilities', 'feature_error', 'feature_health', 'feature_thesis', 'features', 'market_features', 'price_features'],
        requires=['historical_prices', 'market_data', 'multi_timeframe_data', 'ohlcv_data', 'price_data'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'features', 'version': '3.0.0'}
    ),
    'AlternativeRealitySampler': ModuleContract(
        name='AlternativeRealitySampler',
        file='voting/alternative_reality_sampler.py',
        provides=['alternative_reality_sampler_initialization', 'alternative_samples', 'confidence_bounds', 'diversity_score', 'effective_samples', 'sampling_recommendations', 'sampling_stats', 'sampling_uncertainty'],
        requires=['agreement_score', 'consensus_direction', 'market_context', 'market_regime', 'recent_trades', 'session_metrics', 'strategy_arbiter_weights', 'volatility_data', 'votes', 'voting_summary'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'voting', 'version': '3.0.0'}
    ),
    'AuditingCoordinator': ModuleContract(
        name='AuditingCoordinator',
        file='auditing/auditing_coordinator.py',
        provides=['audit_metrics', 'audit_report', 'audit_status'],
        requires=['market_data', 'trades', 'trading_signal'],
        meta={'is_voting_member': 'False', 'explainable': 'True', 'category': 'auditing', 'version': '2.0.0'}
    ),
    'BiasAuditor': ModuleContract(
        name='BiasAuditor',
        file='strategy/bias_auditor.py',
        provides=['bias_adjustments', 'bias_analysis', 'bias_auditor_initialization', 'bias_corrections', 'bias_recommendations', 'bias_report', 'psychological_state'],
        requires=['bias_analysis', 'current_pnl', 'market_regime', 'positions', 'recent_trades', 'risk_data', 'session_context', 'trading_session', 'volatility_level'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),
    'CollusionAuditor': ModuleContract(
        name='CollusionAuditor',
        file='voting/collusion_auditor.py',
        provides=['audit_recommendations', 'behavioral_profiles', 'collusion_alerts', 'collusion_auditor_initialization', 'collusion_score', 'coordination_events', 'detection_statistics', 'member_independence_scores', 'suspicious_pairs'],
        requires=['agreement_score', 'consensus_direction', 'market_context', 'market_regime', 'member_confidences', 'raw_proposals', 'recent_trades', 'strategy_arbiter_weights', 'volatility_data', 'votes', 'voting_summary'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'voting', 'version': '3.0.0'}
    ),
    'ComplianceModule': ModuleContract(
        name='ComplianceModule',
        file='risk/compliance.py',
        provides=['compliance', 'compliance_status', 'risk_limits', 'validation_results'],
        requires=['balance', 'market_context', 'pending_orders', 'positions'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '3.0.0'}
    ),
    'ConsensusDetector': ModuleContract(
        name='ConsensusDetector',
        file='voting/consensus_detector.py',
        provides=['confidence_consensus', 'consensus_components', 'consensus_detector_initialization', 'consensus_quality', 'consensus_recommendations', 'consensus_score', 'consensus_trends', 'directional_consensus', 'magnitude_consensus', 'member_contributions', 'quality_metrics'],
        requires=['agreement_score', 'alpha_weights', 'blended_action', 'consensus_direction', 'market_context', 'market_regime', 'member_confidences', 'raw_proposals', 'volatility_data', 'votes', 'voting_summary'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'voting', 'version': '3.0.0'}
    ),
    'CorrelatedRiskController': ModuleContract(
        name='CorrelatedRiskController',
        file='risk/correlated_risk_controller.py',
        provides=['correlation_clusters', 'correlation_matrix', 'correlation_risk', 'diversification_score'],
        requires=['market_context', 'positions', 'prices'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '3.0.0'}
    ),
    'CurriculumPlannerPlus': ModuleContract(
        name='CurriculumPlannerPlus',
        file='strategy/curriculum_planner_plus.py',
        provides=['competency_scores', 'curriculum_initialization', 'curriculum_stage', 'learning_constraints', 'learning_recommendations', 'mastery_assessment', 'stage_advancement', 'stage_progression'],
        requires=['episode_summary', 'learning_context', 'market_conditions', 'performance_data', 'recent_trades', 'risk_metrics', 'trading_session'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),
    'DrawdownRescue': ModuleContract(
        name='DrawdownRescue',
        file='risk/drawdown_rescue.py',
        provides=['drawdown_risk', 'rescue_status', 'risk_adjustment'],
        requires=['balance', 'equity', 'market_context', 'positions'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '3.0.0'}
    ),
    'DynamicRiskController': ModuleContract(
        name='DynamicRiskController',
        file='risk/dynamic_risk_controller.py',
        provides=['risk_alerts', 'risk_analytics', 'risk_factors', 'risk_scaling'],
        requires=['anomaly_detector', 'compliance', 'execution_quality', 'market_context', 'market_data', 'market_regime', 'performance_data', 'portfolio_risk', 'position_data', 'risk_data'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '4.0.0'}
    ),
    'EnhancedAnomalyDetector': ModuleContract(
        name='EnhancedAnomalyDetector',
        file='risk/anomaly_detector.py',
        provides=['anomaly_alerts', 'anomaly_detection', 'anomaly_score', 'detection_analytics'],
        requires=['anomaly_detection', 'market_context', 'market_data', 'performance_data', 'risk_data', 'trading_data'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '4.0.0'}
    ),
    'EnhancedSeasonalityRiskExpert': ModuleContract(
        name='EnhancedSeasonalityRiskExpert',
        file='voting/voting_wrappers.py',
        provides=['committee_confidence', 'committee_decision', 'expert_performance', 'seasonality_analysis', 'seasonality_confidence', 'seasonality_voting_proposal', 'voting_consensus'],
        requires=['emergency_mode', 'expert_performance', 'market_data', 'market_open', 'market_regime', 'portfolio_state', 'recent_trades', 'risk_data', 'risk_score', 'session_type', 'system_health', 'theme_detection', 'volatility_data', 'voting_consensus'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'voting', 'version': '4.0.0'}
    ),
    'EnhancedThemeExpert': ModuleContract(
        name='EnhancedThemeExpert',
        file='voting/voting_wrappers.py',
        provides=['agreement_score', 'consensus_direction', 'member_confidences', 'raw_proposals', 'strategy_arbiter_weights', 'theme_analysis', 'theme_confidence', 'theme_voting_proposal'],
        requires=['emergency_mode', 'expert_performance', 'market_data', 'market_open', 'market_regime', 'portfolio_state', 'recent_trades', 'risk_data', 'risk_score', 'session_type', 'system_health', 'theme_detection', 'volatility_data', 'voting_consensus'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'voting', 'version': '4.0.0'}
    ),
    'EnhancedVotingCommitteeCoordinator': ModuleContract(
        name='EnhancedVotingCommitteeCoordinator',
        file='voting/voting_wrappers.py',
        provides=['horizon_alignment', 'member_proposals', 'performance_feedback', 'time_of_day', 'votes', 'voting_summary', 'voting_weights', 'trade_vote'],
        requires=['emergency_mode', 'expert_performance', 'expert_votes', 'market_context', 'market_open', 'market_regime', 'portfolio_state', 'recent_trades', 'risk_data', 'risk_score', 'session_type', 'system_health', 'theme_detection', 'volatility_data', 'voting_consensus'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'voting', 'version': '4.0.0'}
    ),
    'EnhancedWorldModel': ModuleContract(
        name='EnhancedWorldModel',
        file='models/world_model.py',
        provides=['market_predictions', 'prediction_confidence', 'scenario_generation', 'world_model_analytics'],
        requires=['market_conditions', 'market_context', 'market_data', 'market_predictions', 'market_regime', 'performance_data', 'performance_metrics', 'regime_data', 'risk_data', 'time_risk_analysis', 'trading_data', 'volatility_adjustment'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'models', 'version': '4.0.1'}
    ),
    'ExecutionQualityMonitor': ModuleContract(
        name='ExecutionQualityMonitor',
        file='voting/execution_quality_monitor.py',
        provides=['execution_alerts', 'execution_analytics', 'execution_quality'],
        requires=['execution_data', 'market_context', 'market_data', 'order_data', 'trade_data'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '4.0.0'}
    ),
    'ExplanationGenerator': ModuleContract(
        name='ExplanationGenerator',
        file='strategy/explanation_generator.py',
        provides=['contextual_narratives', 'decision_rationales', 'explanation_generator_initialization', 'operator_updates', 'performance_insights', 'system_explanations', 'trading_explanations'],
        requires=['bias_analysis', 'learning_status', 'market_context', 'module_insights', 'performance_data', 'positions', 'recent_trades', 'risk_data', 'session_metrics', 'strategy_status', 'system_alerts'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'utils', 'version': '3.0.0'}
    ),
    'FractalRegimeConfirmation': ModuleContract(
        name='FractalRegimeConfirmation',
        file='market/fractal_regime_confirmation.py',
        provides=['fractal_metrics', 'market_regime', 'regime_data', 'regime_strength', 'timestamps', 'trend_direction'],
        requires=['market_regime', 'volatility_level'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'market', 'version': '3.0.0'}
    ),
    'HistoricalReplayAnalyzer': ModuleContract(
        name='HistoricalReplayAnalyzer',
        file='memory/historical_replay_analyzer.py',
        provides=['learning_progress', 'pattern_analysis', 'replay_sequences', 'sequence_quality'],
        requires=['actions', 'episode_data', 'market_data', 'trades'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'memory', 'version': '3.0.0'}
    ),
    'LiquidityHeatmapLayer': ModuleContract(
        name='LiquidityHeatmapLayer',
        file='market/liquidity_heatmap_layer.py',
        provides=['liquidity_capabilities', 'liquidity_prediction', 'liquidity_score', 'liquidity_thesis', 'market_depth', 'session_data', 'spread_analysis', 'trading_sessions'],
        requires=['bid_ask_data', 'price_data', 'prices'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'market', 'version': '3.0.0'}
    ),
    'MarketDataProvider': ModuleContract(
        name='MarketDataProvider',
        file='external/market_data_provider.py',
        provides=['alerts', 'anomaly_detector', 'bid_ask_data', 'committee_votes', 'economic_calendar', 'environment', 'environment_config', 'historical_prices', 'indicators', 'input1', 'input2', 'learning_context', 'learning_status', 'macro_data', 'market_conditions', 'market_context', 'market_data', 'market_liquidity', 'module_insights', 'multi_timeframe_data', 'ohlcv_data', 'portfolio_metrics', 'price_data', 'prices', 'session_type', 'step_data', 'step_idx', 'strategy_status', 'symbols', 'technical_indicators', 'timestamp', 'trading_session', 'volatility', 'volatility_data', 'volatility_level'],
        requires=[],
        meta={'is_voting_member': 'False', 'thesis_required': 'False', 'explainable': 'False', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'external', 'version': '1.0.0'}
    ),
    'MarketThemeDetector': ModuleContract(
        name='MarketThemeDetector',
        file='market/market_theme_detector.py',
        provides=['market_theme', 'theme_detection', 'theme_detector_health', 'theme_detector_status', 'theme_strength', 'theme_transition'],
        requires=['historical_prices', 'macro_data', 'market_data', 'multi_timeframe_data', 'price_data', 'technical_indicators'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'market', 'version': '3.0.0'}
    ),
    'MemoryBudgetOptimizer': ModuleContract(
        name='MemoryBudgetOptimizer',
        file='memory/memory_budget_optimizer.py',
        provides=['allocation_strategy', 'budget_optimization', 'memory_allocation', 'memory_efficiency'],
        requires=['loss_prevention', 'memory_usage', 'mistakes', 'pattern_memory', 'performance_metrics', 'playbook_entries', 'trades'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'memory', 'version': '3.0.1'}
    ),
    'MemoryCompressor': ModuleContract(
        name='MemoryCompressor',
        file='memory/memory_compressor.py',
        provides=['compressed_patterns', 'feature_importance', 'intuition_vector', 'memory_compression'],
        requires=['episode_data', 'features', 'market_context', 'trades'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'memory', 'version': '3.0.0'}
    ),
    'MetaAgent': ModuleContract(
        name='MetaAgent',
        file='meta/meta_agent.py',
        provides=['automation_decisions', 'automation_metrics', 'meta_performance', 'system_mode'],
        requires=['market_conditions', 'risk_signals', 'system_performance', 'time_risk_analysis', 'training_metrics'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'meta', 'version': '3.0.0'}
    ),
    'MetaCognitivePlanner': ModuleContract(
        name='MetaCognitivePlanner',
        file='meta/metacognitive_planner.py',
        provides=['adaptation_metrics', 'planning_status', 'strategic_insights', 'tactical_recommendations'],
        requires=['actions', 'market_conditions', 'market_data', 'market_regime', 'performance_metrics', 'regime_data', 'trades', 'volatility_adjustment'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'meta', 'version': '3.0.1'}
    ),
    'MetaRLController': ModuleContract(
        name='MetaRLController',
        file='meta/metar_rl_controller.py',
        provides=['agent_decisions', 'agents_performance', 'automation_status', 'controller_status', 'controller_training_overview', 'meta_signals', 'trading_signal', 'trading_signals'],
        requires=['actions', 'market_data', 'trades', 'training_data'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'meta', 'version': '3.0.0'}
    ),
    'MistakeMemory': ModuleContract(
        name='MistakeMemory',
        file='memory/mistake_memory.py',
        provides=['danger_zones', 'loss_prevention', 'mistake_avoidance', 'mistake_memory', 'pattern_recognition'],
        requires=['features', 'market_context', 'risk_data', 'time_risk_analysis', 'trades'],
        meta={'category': 'memory', 'version': '3.0.2'}
    ),
    'MultiScaleFeatureEngine': ModuleContract(
        name='MultiScaleFeatureEngine',
        file='features/multiscale_feature_engine.py',
        provides=['attention_weights', 'feature_fusion', 'multiscale_features', 'neural_capabilities', 'neural_embeddings', 'neural_health'],
        requires=['advanced_features', 'market_data'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'features', 'version': '3.0.0'}
    ),
    'MyModule': ModuleContract(
        name='MyModule',
        file='monitoring/integration_validator.py',
        provides=['key', 'output1', 'output2'],
        requires=['input1', 'input2', 'key'],
        meta={'explainable': 'True', 'category': 'market'}
    ),
    'NeuralMemoryArchitect': ModuleContract(
        name='NeuralMemoryArchitect',
        file='memory/neural_memory_architect.py',
        provides=['attention_retrieval', 'importance_scoring', 'memory_embedding', 'neural_memory'],
        requires=['actions', 'market_context', 'observations', 'rewards'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'memory', 'version': '3.0.0'}
    ),
    'NewsSentimentModule': ModuleContract(
        name='NewsSentimentModule',
        file='external/news_sentiment.py',
        provides=['news_sentiment', 'news_summary', 'sentiment_confidence', 'sentiment_trend'],
        requires=['market_data', 'symbols', 'trading_session'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'external', 'version': '3.0.0'}
    ),
    'OpponentModeEnhancer': ModuleContract(
        name='OpponentModeEnhancer',
        file='strategy/opponent_mode_enhancer.py',
        provides=['market_mode_detection', 'mode_analysis', 'mode_performance', 'mode_recommendations', 'mode_weights', 'opponent_mode_enhancer_initialization', 'strategy_adaptation'],
        requires=['market_context', 'market_data', 'market_regime', 'price_data', 'recent_trades', 'session_metrics', 'technical_indicators', 'trading_performance', 'volatility_data'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),
    'OpponentSimulator': ModuleContract(
        name='OpponentSimulator',
        file='simulation/opponent_simulator.py',
        provides=['adversarial_scenarios', 'market_noise', 'market_perturbations', 'opponent_analysis', 'opponent_simulation', 'perturbation_history', 'simulated_prices', 'simulation_effects', 'simulation_statistics'],
        requires=['historical_prices', 'market_context', 'market_data', 'positions', 'prices', 'regime_data', 'session_data', 'volatility'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'simulation', 'version': '3.0.0'}
    ),
    'PlaybookClusterer': ModuleContract(
        name='PlaybookClusterer',
        file='strategy/playbook_clusterer.py',
        provides=['cluster_analysis', 'cluster_effectiveness', 'cluster_recommendations', 'cluster_weights', 'clustering_health', 'clustering_thesis', 'playbook_clusterer_initialization'],
        requires=['market_context', 'market_data', 'market_regime', 'playbook_memory', 'recent_trades', 'session_metrics', 'trading_performance', 'volatility_data'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),
    'PlaybookMemory': ModuleContract(
        name='PlaybookMemory',
        file='memory/playbook_memory.py',
        provides=['memory_analytics', 'pattern_memory', 'playbook_quality', 'playbook_recall'],
        requires=['actions', 'market_data', 'prices', 'trades'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'memory', 'version': '3.0.1'}
    ),
    'PortfolioRiskSystem': ModuleContract(
        name='PortfolioRiskSystem',
        file='risk/portofilio_risk_system.py',
        provides=['portfolio_risk', 'portfolio_risk_proposal', 'position_limits', 'risk_data', 'risk_metrics', 'risk_score', 'risk_signals', 'trade_data', 'trading_data'],
        requires=['market_context', 'market_data', 'position_data', 'risk_signals', 'trade_data'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '4.0.0'}
    ),
    'PositionManager': ModuleContract(
        name='PositionManager',
        file='position/position.py',
        provides=['balance', 'current_pnl', 'current_positions', 'equity', 'execution_data', 'order_data', 'pending_orders', 'portfolio_state', 'position_analysis', 'position_data', 'position_decisions', 'position_health', 'positions', 'recent_trades', 'trades'],
        requires=['correlation_matrix', 'environment_config', 'indicators', 'liquidity_capabilities', 'liquidity_score', 'market_conditions', 'market_context', 'market_data', 'market_liquidity', 'market_regime', 'market_state', 'portfolio_metrics', 'price_data', 'prices', 'risk_score', 'technical_indicators', 'time_risk_analysis', 'volatility_data'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'position', 'version': '3.1.0'}
    ),
    'PPOAgent': ModuleContract(
        name='PPOAgent',
        file='meta/ppo_agent.py',
        provides=['actions', 'agent_performance', 'observations', 'policy_actions', 'policy_gradients', 'rewards', 'training_data', 'training_metrics', 'training_signals'],
        requires=['market_data'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'meta', 'version': '3.0.0'}
    ),
    'PPOLagAgent': ModuleContract(
        name='PPOLagAgent',
        file='meta/ppo_lag_agent.py',
        provides=['agent_status', 'market_adaptation', 'position_metrics', 'ppo_lag_training_metrics'],
        requires=['actions', 'market_data', 'trades', 'training_signals'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'meta', 'version': '3.0.0'}
    ),
    'RegimePerformanceMatrix': ModuleContract(
        name='RegimePerformanceMatrix',
        file='market/regime_performance_matrix.py',
        provides=['backtesting_data', 'market_state', 'performance_metrics', 'regime_accuracy', 'regime_analysis', 'regime_matrix_analysis', 'regime_matrix_health', 'regime_matrix_status', 'regime_performance', 'regime_prediction', 'stress_test_results'],
        requires=['liquidity_score', 'market_data', 'market_regime', 'pnl_data', 'recent_trades', 'volatility_data'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'market', 'version': '3.0.0'}
    ),
    'RiskAdjustedReward': ModuleContract(
        name='RiskAdjustedReward',
        file='reward/risk_adjusted_reward.py',
        provides=['reward_analytics', 'reward_components', 'reward_performance', 'shaped_reward'],
        requires=['environment_config', 'market_context', 'mistake_memory', 'performance_data', 'risk_metrics', 'trade_data'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'reward', 'version': '4.0.0'}
    ),
    'RoleCoach': ModuleContract(
        name='RoleCoach',
        file='simulation/role_coach.py',
        provides=['coaching_penalties', 'coaching_recommendations', 'coaching_results', 'coaching_statistics', 'compliance_tracking', 'discipline_assessment', 'discipline_penalty', 'performance_scoring', 'trade_limits'],
        requires=['market_context', 'pending_orders', 'positions', 'recent_trades', 'regime_data', 'risk_metrics', 'session_data', 'trading_performance'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'simulation', 'version': '3.0.0'}
    ),
    'SessionManager': ModuleContract(
        name='SessionManager',
        file='external/session_manager.py',
        provides=['consensus_data', 'emergency_mode', 'episode_data', 'episode_summary', 'expert_votes', 'market_open', 'memory_usage', 'mistakes', 'module_performance', 'performance_data', 'playbook_entries', 'playbook_memory', 'pnl_data', 'session_context', 'session_metrics', 'system_alerts', 'system_health', 'system_performance'],
        requires=[],
        meta={'is_voting_member': 'False', 'thesis_required': 'False', 'explainable': 'False', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'external', 'version': '1.0.0'}
    ),
    'ShadowSimulator': ModuleContract(
        name='ShadowSimulator',
        file='simulation/shadow_simulator.py',
        provides=['forward_projections', 'scenario_analysis', 'scenario_recommendations', 'shadow_predictions', 'shadow_simulation', 'simulation_confidence', 'simulation_predictions', 'strategy_simulations'],
        requires=['committee_votes', 'environment', 'market_context', 'market_data', 'pending_orders', 'positions', 'prices', 'recent_trades', 'risk_metrics', 'trading_performance'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'simulation', 'version': '3.0.0'}
    ),
    'StrategyArbiter': ModuleContract(
        name='StrategyArbiter',
        file='voting/strategy_arbiter.py',
        provides=['alpha_weights', 'arbiter_recommendations', 'blended_action', 'decision_statistics', 'gate_decision', 'instrument_signals', 'instruments', 'member_performance', 'member_weights', 'proposal_analysis', 'strategy_arbiter_initialization', 'strategy_weights', 'voting_quality'],
        requires=['collusion_score', 'consensus_score', 'current_positions', 'horizon_alignment', 'market_context', 'market_regime', 'member_confidences', 'member_proposals', 'recent_trades', 'session_data', 'volatility_data'],
        meta={'category': 'voting', 'version': '3.0.0'}
    ),
    'StrategyGenomePool': ModuleContract(
        name='StrategyGenomePool',
        file='strategy/strategy_genome_pool.py',
        provides=['best_genome', 'evolution_analytics', 'genome_analysis', 'genome_recommendations', 'genome_weights', 'population_metrics', 'strategy_genome_pool_initialization'],
        requires=['market_context', 'market_data', 'market_regime', 'recent_trades', 'risk_data', 'session_metrics', 'trading_performance', 'volatility_data'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),
    'StrategyIntrospector': ModuleContract(
        name='StrategyIntrospector',
        file='strategy/strategy_introspector.py',
        provides=['adaptation_recommendations', 'behavior_patterns', 'introspection_metrics', 'module_data', 'strategy_analysis', 'strategy_introspector_initialization', 'strategy_performance', 'strategy_profiles', 'trading_performance'],
        requires=['market_context', 'market_regime', 'module_data', 'recent_trades', 'risk_data', 'strategy_performance', 'strategy_weights', 'trading_performance', 'volatility_data'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),
    'ThesisEvolutionEngine': ModuleContract(
        name='ThesisEvolutionEngine',
        file='strategy/thesis_evolution_engine.py',
        provides=['active_theses', 'best_thesis', 'evolution_history', 'thesis_diversity', 'thesis_evolution_initialization', 'thesis_performance', 'thesis_recommendations'],
        requires=['economic_calendar', 'market_context', 'market_data', 'market_regime', 'recent_trades', 'risk_metrics', 'session_metrics', 'strategy_performance', 'trading_performance', 'volatility_data'],
        meta={'is_voting_member': 'True', 'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'strategy', 'version': '3.0.0'}
    ),
    'TimeAwareRiskScaling': ModuleContract(
        name='TimeAwareRiskScaling',
        file='market/time_aware_risk_scaling.py',
        provides=['risk_scaling_factor', 'session_risk', 'time_risk_health', 'time_risk_status', 'time_risk_analysis', 'volatility_adjustment'],
        requires=['market_data', 'risk_data', 'timestamp', 'volatility_data'],
        meta={'thesis_required': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'risk', 'version': '3.0.0'}
    ),
    'TimeHorizonAligner': ModuleContract(
        name='TimeHorizonAligner',
        file='voting/time_horizon_aligner.py',
        provides=['adaptation_status', 'aligned_weights', 'alignment_quality', 'horizon_distances', 'horizon_multipliers', 'regime_adjustments', 'session_patterns', 'time_horizon_aligner_initialization'],
        requires=['expert_performance', 'market_context', 'market_regime', 'member_confidences', 'performance_feedback', 'recent_trades', 'session_type', 'time_of_day', 'volatility_data', 'voting_weights'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'voting', 'version': '3.0.0'}
    ),
    'TradeExplanationAuditor': ModuleContract(
        name='TradeExplanationAuditor',
        file='auditing/trade_explanation_auditor.py',
        provides=['audit_alerts', 'explanation_metrics', 'trade_explanations'],
        requires=['market_data', 'trades', 'trading_signal'],
        meta={'is_voting_member': 'False', 'explainable': 'True', 'category': 'auditing', 'version': '2.0.0'}
    ),
    'TradeMapVisualizer': ModuleContract(
        name='TradeMapVisualizer',
        file='visualization/trade_map_visualizer.py',
        provides=['chart_cache', 'chart_history', 'chart_statistics', 'dashboard_charts', 'performance_charts', 'trade_charts', 'visualization_reports'],
        requires=['consensus_data', 'market_data', 'module_performance', 'positions', 'recent_trades', 'risk_metrics', 'trading_performance'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'visualization', 'version': '3.0.0'}
    ),
    'TradeThesisTracker': ModuleContract(
        name='TradeThesisTracker',
        file='auditing/trade_thesis_tracker.py',
        provides=['thesis_alerts', 'thesis_analysis'],
        requires=['market_data', 'trades', 'trading_signal'],
        meta={'is_voting_member': 'False', 'explainable': 'True', 'category': 'auditing', 'version': '2.0.0'}
    ),
    'TradingModeManager': ModuleContract(
        name='TradingModeManager',
        file='trading_modes/trading_mode.py',
        provides=['decision_factors', 'mode_config', 'mode_effectiveness', 'mode_stats', 'mode_thresholds', 'trading_mode', 'trading_mode_manager_initialization'],
        requires=['economic_calendar', 'market_context', 'market_regime', 'positions', 'recent_trades', 'risk_metrics', 'session_metrics', 'strategy_performance', 'trading_performance', 'volatility_data', 'votes'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'trading_modes', 'version': '3.0.0'}
    ),
    'VisualizationInterface': ModuleContract(
        name='VisualizationInterface',
        file='visualization/visualization_interface.py',
        provides=['alert_timeline', 'analytics_reports', 'dashboard_data', 'streaming_data', 'system_status', 'visualization_data'],
        requires=['alerts', 'consensus_data', 'market_data', 'module_performance', 'positions', 'recent_trades', 'risk_metrics', 'step_data', 'system_alerts', 'trading_performance'],
        meta={'thesis_required': 'True', 'explainable': 'True', 'health_monitoring': 'True', 'performance_tracking': 'True', 'category': 'visualization', 'version': '3.0.0'}
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
