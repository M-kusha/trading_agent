# ─────────────────────────────────────────────────────────────
# File: train/enhanced_training_callback.py  
# InfoBus-Integrated Training Callback with Comprehensive Module Health Monitoring
# ─────────────────────────────────────────────────────────────

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
from collections import deque, defaultdict
from stable_baselines3.common.callbacks import BaseCallback

# InfoBus and audit infrastructure
from modules.utils.info_bus import InfoBus, create_info_bus, InfoBusExtractor, InfoBusUpdater
from modules.utils.audit_utils import RotatingLogger, AuditTracker, format_operator_message, system_audit
from modules.core.core import ModuleConfig
from envs.config import TradingConfig


class InfoBusTrainingCallback(BaseCallback):
    """
    Enhanced training callback with comprehensive InfoBus integration.
    Monitors all 50+ modules, tracks health, and provides operator-centric logging.
    """
    
    def __init__(self, total_timesteps: int, config: TradingConfig, 
                 metrics_broadcaster=None, verbose: int = 0):
        super().__init__(verbose)
        
        self.total_timesteps = total_timesteps
        self.config = config
        self.metrics_broadcaster = metrics_broadcaster
        self.start_time = datetime.now()
        
        # ══════════════════════════════════════════════════════════════
        # Enhanced InfoBus Infrastructure
        # ══════════════════════════════════════════════════════════════
        
        # 🔧 FIX: Use different name to avoid BaseCallback logger conflict
        self.training_logger = RotatingLogger(
            name="TrainingMonitor",
            log_path=f"logs/training/infobus_training_{datetime.now().strftime('%Y%m%d')}.log",
            max_lines=2000,
            operator_mode=True
        )
        
        # Audit tracker for training events
        self.audit_tracker = AuditTracker("TrainingSystem")
        
        # Module health monitoring
        self.module_health_tracker = ModuleHealthTracker()
        
        # ══════════════════════════════════════════════════════════════
        # Training Metrics with InfoBus Integration
        # ══════════════════════════════════════════════════════════════
        
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.training_metrics = {}
        self.performance_history = deque(maxlen=500)
        
        # InfoBus quality tracking
        self.info_bus_quality_history = deque(maxlen=100)
        self.module_performance_history = defaultdict(lambda: deque(maxlen=100))
        
        # Episode tracking
        self.best_reward = -float('inf')
        self.current_episode_reward = 0
        self.episode_count = 0
        self.consecutive_failures = 0
        
        # System health monitoring
        self.health_check_interval = 100  # Check every 100 steps
        self.last_health_check = 0
        self.health_alerts = deque(maxlen=50)
        
        # ══════════════════════════════════════════════════════════════
        # Module Integration Monitoring
        # ══════════════════════════════════════════════════════════════
        
        self.module_categories = {
            'auditing': ['TradeExplanationAuditor', 'TradeThesisTracker'],
            'core': ['Module', 'InfoBus'],
            'features': ['AdvancedFeatureEngine', 'MultiScaleFeatureEngine'],
            'market': ['MarketThemeDetector', 'FractalRegimeConfirmation', 'LiquidityHeatmapLayer'],
            'memory': ['NeuralMemoryArchitect', 'MistakeMemory', 'MemoryCompressor'],
            'position': ['PositionManager'],
            'reward': ['RiskAdjustedReward'],
            'risk': ['ActiveTradeMonitor', 'CorrelatedRiskController', 'DrawdownRescue'],
            'strategy': ['PlaybookClusterer', 'StrategyIntrospector', 'CurriculumPlannerPlus'],
            'voting': ['TimeHorizonAligner', 'ConsensusDetector', 'StrategyArbiter'],
            'meta': ['MetaAgent', 'MetaCognitivePlanner', 'MetaRLController']
        }
        
        self.training_logger.info(
            format_operator_message(
                "🚀", "INFOBUS_TRAINING_CALLBACK_INITIALIZED",
                details=f"Monitoring {sum(len(mods) for mods in self.module_categories.values())} modules",
                result=f"Mode: {'LIVE' if getattr(config, 'live_mode', False) else 'OFFLINE'}",
                context="training_startup"
            )
        )

    def _on_training_start(self) -> None:
        """Enhanced training start with InfoBus initialization"""
        
        self.start_time = datetime.now()
        
        # Initialize InfoBus connection with environment
        try:
            self._initialize_infobus_connection()
            self.training_logger.info(
                format_operator_message(
                    "🔗", "INFOBUS_CONNECTION_ESTABLISHED",
                    details="Training callback connected to environment InfoBus",
                    context="training_startup"
                )
            )
        except Exception as e:
            self.training_logger.error(
                format_operator_message(
                    "❌", "INFOBUS_CONNECTION_FAILED",
                    details=str(e),
                    context="training_startup"
                )
            )
        
        # Record training start event
        self.audit_tracker.record_event(
            "training_started",
            "TrainingCallback",
            {
                'total_timesteps': self.total_timesteps,
                'config_mode': 'live' if getattr(self.config, 'live_mode', False) else 'offline',
                'infobus_enabled': self.config.info_bus_enabled,
                'module_count': sum(len(mods) for mods in self.module_categories.values())
            },
            severity="info"
        )

    def _on_step(self) -> bool:
        """Enhanced step with comprehensive InfoBus monitoring"""
        
        try:
            # Collect comprehensive metrics every step
            if self.n_calls % 10 == 0:  # Every 10 steps
                metrics = self._collect_comprehensive_metrics()
                self._update_performance_tracking(metrics)
                
                # Broadcast metrics if available
                if self.metrics_broadcaster:
                    try:
                        self.metrics_broadcaster.send_metrics(metrics)
                    except Exception as e:
                        self.training_logger.warning(f"Metrics broadcast failed: {e}")
            
            # Health checks every N steps
            if self.n_calls - self.last_health_check >= self.health_check_interval:
                self._perform_comprehensive_health_check()
                self.last_health_check = self.n_calls
            
            # InfoBus quality monitoring
            if self.n_calls % 50 == 0:  # Every 50 steps
                self._monitor_infobus_quality()
            
            # Episode tracking
            self._track_episode_progress()
            
            # Emergency checks
            if self._check_emergency_conditions():
                return False
                
            return True
            
        except Exception as e:
            self.training_logger.error(
                format_operator_message(
                    "💥", "TRAINING_STEP_ERROR",
                    details=str(e),
                    context="training_monitoring"
                )
            )
            self.consecutive_failures += 1
            
            # Stop training if too many consecutive failures
            if self.consecutive_failures > 10:
                self.training_logger.critical(
                    format_operator_message(
                        "🚨", "TRAINING_STOPPED_EXCESSIVE_FAILURES",
                        details=f"{self.consecutive_failures} consecutive failures",
                        context="emergency_stop"
                    )
                )
                return False
            
            return True

    def _initialize_infobus_connection(self):
        """Initialize connection to environment InfoBus"""
        
        # Get environment from training context
        if hasattr(self.training_env, 'envs') and len(self.training_env.envs) > 0:
            env = self.training_env.envs[0]
            
            # Check if environment has InfoBus support
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'info_bus'):
                self.env_ref = env.unwrapped
                self.training_logger.info("InfoBus connection established with environment")
            else:
                self.training_logger.warning("Environment does not support InfoBus")
                self.env_ref = None
        else:
            self.training_logger.warning("Cannot access training environment")
            self.env_ref = None

    def _collect_comprehensive_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive training metrics with InfoBus data"""
        
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        progress = self.n_calls / self.total_timesteps
        
        # Base training metrics
        metrics = {
            # Training Progress
            'timestep': self.n_calls,
            'total_timesteps': self.total_timesteps,
            'progress_pct': progress * 100,
            'episodes': self.episode_count,
            'elapsed_time': elapsed_time,
            'steps_per_second': self.n_calls / elapsed_time if elapsed_time > 0 else 0,
            'estimated_time_remaining': (elapsed_time / progress - elapsed_time) if progress > 0 else 0,
            
            # Performance Metrics
            'episode_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'episode_reward_std': np.std(self.episode_rewards) if self.episode_rewards else 0,
            'episode_length_mean': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'best_episode_reward': self.best_reward,
            'current_episode_reward': self.current_episode_reward,
            'consecutive_failures': self.consecutive_failures,
            
            # System Status
            'training_mode': 'LIVE' if getattr(self.config, 'live_mode', False) else 'OFFLINE',
            'infobus_enabled': self.config.info_bus_enabled,
            'health_alerts_count': len(self.health_alerts),
        }
        
        # Add InfoBus-specific metrics
        if hasattr(self, 'env_ref') and self.env_ref and hasattr(self.env_ref, 'info_bus'):
            infobus_metrics = self._extract_infobus_metrics()
            metrics.update(infobus_metrics)
        
        # Add module health metrics
        module_health = self.module_health_tracker.get_health_summary()
        metrics.update(module_health)
        
        # Add model learning metrics if available
        if hasattr(self.model, 'logger') and self.model.logger:
            learning_metrics = self._extract_learning_metrics()
            metrics.update(learning_metrics)
        
        return metrics

    def _extract_infobus_metrics(self) -> Dict[str, Any]:
        """Extract metrics from environment InfoBus"""
        
        try:
            if not self.env_ref.info_bus:
                return {'infobus_status': 'no_data'}
            
            info_bus = self.env_ref.info_bus
            
            # Extract key InfoBus metrics
            return {
                'infobus_status': 'active',
                'infobus_step': InfoBusExtractor.get_safe_numeric(info_bus, 'step_idx', 0),
                'infobus_balance': InfoBusExtractor.get_safe_numeric(info_bus, 'balance', self.config.initial_balance),
                'infobus_positions': InfoBusExtractor.get_position_count(info_bus),
                'infobus_alerts': InfoBusExtractor.get_alert_count(info_bus),
                'infobus_consensus': InfoBusExtractor.get_safe_numeric(info_bus, 'consensus', 0.5),
                'infobus_risk_score': InfoBusExtractor.get_risk_score(info_bus),
                'infobus_drawdown': InfoBusExtractor.get_drawdown_pct(info_bus),
                'infobus_votes': len(info_bus.get('votes', [])),
                'infobus_market_regime': InfoBusExtractor.get_market_regime(info_bus),
                'infobus_volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
            }
            
        except Exception as e:
            self.training_logger.warning(f"Failed to extract InfoBus metrics: {e}")
            return {'infobus_status': 'error', 'infobus_error': str(e)}

    def _extract_learning_metrics(self) -> Dict[str, Any]:
        """Extract learning metrics from PPO model"""
        
        try:
            logger_data = self.model.logger.name_to_value
            
            return {
                'learning_rate': getattr(self.model, 'learning_rate', 0),
                'clip_fraction': logger_data.get('train/clip_fraction', 0),
                'explained_variance': logger_data.get('train/explained_variance', 0),
                'policy_loss': logger_data.get('train/policy_loss', 0),
                'value_loss': logger_data.get('train/value_loss', 0),
                'entropy_loss': logger_data.get('train/entropy_loss', 0),
                'kl_divergence': logger_data.get('train/kl_divergence', 0),
            }
            
        except Exception as e:
            return {'learning_metrics_error': str(e)}

    def _perform_comprehensive_health_check(self):
        """Perform comprehensive health check of all system components"""
        
        health_summary = {
            'timestamp': datetime.now().isoformat(),
            'step': self.n_calls,
            'checks_performed': [],
            'issues_found': [],
            'overall_health': 'OK'
        }
        
        try:
            # Environment health check
            env_health = self._check_environment_health()
            health_summary['checks_performed'].append('environment')
            if env_health['issues']:
                health_summary['issues_found'].extend(env_health['issues'])
            
            # InfoBus health check
            if self.config.info_bus_enabled:
                infobus_health = self._check_infobus_health()
                health_summary['checks_performed'].append('infobus')
                if infobus_health['issues']:
                    health_summary['issues_found'].extend(infobus_health['issues'])
            
            # Module health check
            if hasattr(self, 'env_ref') and self.env_ref:
                module_health = self.module_health_tracker.perform_health_check(self.env_ref)
                health_summary['checks_performed'].append('modules')
                if module_health['issues']:
                    health_summary['issues_found'].extend(module_health['issues'])
            
            # Model health check
            model_health = self._check_model_health()
            health_summary['checks_performed'].append('model')
            if model_health['issues']:
                health_summary['issues_found'].extend(model_health['issues'])
            
            # Determine overall health
            issue_count = len(health_summary['issues_found'])
            if issue_count == 0:
                health_summary['overall_health'] = 'OK'
            elif issue_count <= 3:
                health_summary['overall_health'] = 'WARNING'
            else:
                health_summary['overall_health'] = 'CRITICAL'
            
            # Log health status
            if health_summary['overall_health'] != 'OK':
                self.training_logger.warning(
                    format_operator_message(
                        "⚠️", "HEALTH_CHECK_ISSUES",
                        details=f"{issue_count} issues found",
                        result=health_summary['overall_health'],
                        context="health_monitoring"
                    )
                )
                self.health_alerts.append(health_summary)
            
            # Record health audit
            self.audit_tracker.record_event(
                "health_check",
                "TrainingCallback",
                health_summary,
                severity="warning" if issue_count > 0 else "info"
            )
            
        except Exception as e:
            self.training_logger.error(
                format_operator_message(
                    "💥", "HEALTH_CHECK_FAILED",
                    details=str(e),
                    context="health_monitoring"
                )
            )

    def _check_environment_health(self) -> Dict[str, Any]:
        """Check environment health"""
        
        issues = []
        
        try:
            if hasattr(self, 'env_ref') and self.env_ref:
                # Check if environment is responsive
                if not hasattr(self.env_ref, 'market_state'):
                    issues.append("Environment missing market_state")
                
                # Check balance sanity
                if hasattr(self.env_ref, 'market_state'):
                    balance = self.env_ref.market_state.balance
                    if balance <= 0:
                        issues.append(f"Balance is zero or negative: {balance}")
                    elif balance > self.config.initial_balance * 10:
                        issues.append(f"Balance suspiciously high: {balance}")
            else:
                issues.append("No environment reference available")
                
        except Exception as e:
            issues.append(f"Environment health check failed: {e}")
        
        return {'issues': issues}

    def _check_infobus_health(self) -> Dict[str, Any]:
        """Check InfoBus system health"""
        
        issues = []
        
        try:
            if hasattr(self, 'env_ref') and self.env_ref and hasattr(self.env_ref, 'info_bus'):
                if self.env_ref.info_bus is None:
                    issues.append("InfoBus is None")
                else:
                    # Validate InfoBus quality
                    from modules.utils.info_bus import validate_info_bus
                    quality = validate_info_bus(self.env_ref.info_bus)
                    
                    if not quality.is_valid:
                        issues.append(f"InfoBus quality issues: {quality.missing_fields}")
                    
                    if quality.score < 80:
                        issues.append(f"InfoBus quality score low: {quality.score}")
            else:
                issues.append("InfoBus not available in environment")
                
        except Exception as e:
            issues.append(f"InfoBus health check failed: {e}")
        
        return {'issues': issues}

    def _check_model_health(self) -> Dict[str, Any]:
        """Check PPO model health"""
        
        issues = []
        
        try:
            # Check model device
            if hasattr(self.model, 'device'):
                device_str = str(self.model.device)
                if 'cuda' in device_str.lower():
                    import torch
                    if not torch.cuda.is_available():
                        issues.append("Model on CUDA but CUDA not available")
            
            # Check learning rate
            if hasattr(self.model, 'learning_rate'):
                lr = self.model.learning_rate
                if lr <= 0:
                    issues.append(f"Learning rate is zero or negative: {lr}")
                elif lr > 0.1:
                    issues.append(f"Learning rate suspiciously high: {lr}")
            
            # Check model parameters
            if hasattr(self.model, 'policy'):
                try:
                    params = list(self.model.policy.parameters())
                    if not params:
                        issues.append("Model has no parameters")
                    else:
                        # Check for NaN parameters
                        for i, param in enumerate(params[:5]):  # Check first 5
                            if np.any(np.isnan(param.detach().cpu().numpy())):
                                issues.append(f"NaN detected in model parameters")
                                break
                except Exception as e:
                    issues.append(f"Cannot access model parameters: {e}")
            
        except Exception as e:
            issues.append(f"Model health check failed: {e}")
        
        return {'issues': issues}

    def _monitor_infobus_quality(self):
        """Monitor InfoBus data quality over time"""
        
        if not self.config.info_bus_enabled or not hasattr(self, 'env_ref') or not self.env_ref:
            return
        
        try:
            if hasattr(self.env_ref, 'info_bus') and self.env_ref.info_bus:
                from modules.utils.info_bus import validate_info_bus
                quality = validate_info_bus(self.env_ref.info_bus)
                
                self.info_bus_quality_history.append({
                    'step': self.n_calls,
                    'score': quality.score,
                    'is_valid': quality.is_valid,
                    'missing_fields': len(quality.missing_fields),
                    'warnings': len(quality.warnings)
                })
                
                # Alert on quality degradation
                if quality.score < 70:
                    self.training_logger.warning(
                        format_operator_message(
                            "⚠️", "INFOBUS_QUALITY_DEGRADED",
                            details=f"Score: {quality.score:.1f}",
                            result=f"Missing: {len(quality.missing_fields)}, Warnings: {len(quality.warnings)}",
                            context="infobus_monitoring"
                        )
                    )
            
        except Exception as e:
            self.training_logger.error(f"InfoBus quality monitoring failed: {e}")

    def _track_episode_progress(self):
        """Enhanced episode progress tracking"""
        
        # Track episode completion
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.locals.get('episode_length', 0))
            self.episode_count += 1
            
            # Check for new best episode
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
                self.training_logger.info(
                    format_operator_message(
                        "🏆", "NEW_BEST_EPISODE",
                        details=f"Episode {self.episode_count}",
                        result=f"Reward: {self.best_reward:.2f}",
                        context="performance_tracking"
                    )
                )
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.consecutive_failures = 0  # Reset on successful episode
        else:
            # Accumulate episode reward
            self.current_episode_reward += self.locals.get('rewards', [0])[0]

    def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions that require training stop"""
        
        emergency_conditions = []
        
        # Check for excessive failures
        if self.consecutive_failures > 10:
            emergency_conditions.append("Excessive consecutive failures")
        
        # Check health alerts
        if len(self.health_alerts) > 5:
            recent_critical = sum(1 for alert in list(self.health_alerts)[-5:] 
                                if alert.get('overall_health') == 'CRITICAL')
            if recent_critical >= 3:
                emergency_conditions.append("Multiple critical health alerts")
        
        # Check InfoBus quality
        if len(self.info_bus_quality_history) > 10:
            recent_scores = [q['score'] for q in list(self.info_bus_quality_history)[-10:]]
            if np.mean(recent_scores) < 50:
                emergency_conditions.append("InfoBus quality severely degraded")
        
        # Log and return emergency status
        if emergency_conditions:
            self.training_logger.critical(
                format_operator_message(
                    "🚨", "EMERGENCY_CONDITIONS_DETECTED",
                    details="; ".join(emergency_conditions),
                    context="emergency_monitoring"
                )
            )
            
            # Record emergency audit
            self.audit_tracker.record_event(
                "emergency_stop",
                "TrainingCallback",
                {"conditions": emergency_conditions, "step": self.n_calls},
                severity="critical"
            )
            
            return True
        
        return False

    def _update_performance_tracking(self, metrics: Dict[str, Any]):
        """Update performance tracking with enhanced metrics"""
        
        # Add to performance history
        performance_snapshot = {
            'step': self.n_calls,
            'timestamp': datetime.now().isoformat(),
            'episode_reward_mean': metrics.get('episode_reward_mean', 0),
            'steps_per_second': metrics.get('steps_per_second', 0),
            'health_score': 100 - len(self.health_alerts) * 5,  # Simple health score
            'infobus_score': metrics.get('infobus_score', 100),
        }
        
        self.performance_history.append(performance_snapshot)
        
        # Save metrics to file periodically
        if self.n_calls % 1000 == 0:
            self._save_metrics_to_file(metrics)

    def _save_metrics_to_file(self, metrics: Dict[str, Any]):
        """Save comprehensive metrics to file for persistence"""
        
        try:
            # Ensure directory exists
            os.makedirs("logs/training", exist_ok=True)
            
            metrics_file = f"logs/training/comprehensive_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            enhanced_metrics = {
                **metrics,
                'performance_history': list(self.performance_history)[-10:],  # Last 10 snapshots
                'health_alerts': list(self.health_alerts),
                'module_health': self.module_health_tracker.get_health_summary(),
                'infobus_quality': list(self.info_bus_quality_history)[-10:],
            }
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(enhanced_metrics, default=str) + '\n')
                
        except Exception as e:
            self.training_logger.error(f"Failed to save metrics: {e}")

    def _on_training_end(self) -> None:
        """Enhanced training end with comprehensive reporting"""
        
        end_time = datetime.now()
        training_duration = end_time - self.start_time
        
        # Generate comprehensive training report
        final_report = {
            'training_duration': str(training_duration),
            'total_episodes': self.episode_count,
            'total_steps': self.n_calls,
            'best_reward': self.best_reward,
            'final_reward_mean': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'health_alerts_total': len(self.health_alerts),
            'consecutive_failures': self.consecutive_failures,
            'infobus_enabled': self.config.info_bus_enabled,
            'final_module_health': self.module_health_tracker.get_health_summary(),
        }
        
        # Log training completion
        self.training_logger.info(
            format_operator_message(
                "✅", "TRAINING_COMPLETED",
                details=f"Duration: {training_duration}",
                result=f"Episodes: {self.episode_count}, Best: {self.best_reward:.2f}",
                context="training_completion"
            )
        )
        
        # Record final audit
        self.audit_tracker.record_event(
            "training_completed",
            "TrainingCallback",
            final_report,
            severity="info"
        )
        
        # Save final comprehensive report
        try:
            os.makedirs("logs/training", exist_ok=True)
            report_file = f"logs/training/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
        except Exception as e:
            self.training_logger.error(f"Failed to save final report: {e}")

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary for external monitoring"""
        
        return {
            'status': 'running',
            'progress': self.n_calls / self.total_timesteps,
            'episodes': self.episode_count,
            'best_reward': self.best_reward,
            'health_status': 'OK' if len(self.health_alerts) == 0 else 'ISSUES',
            'health_alerts': len(self.health_alerts),
            'consecutive_failures': self.consecutive_failures,
            'infobus_status': 'enabled' if self.config.info_bus_enabled else 'disabled',
            'module_health': self.module_health_tracker.get_health_summary(),
            'recent_performance': list(self.performance_history)[-5:] if self.performance_history else [],
        }


class ModuleHealthTracker:
    """Dedicated module health tracking for all integrated modules"""
    
    def __init__(self):
        self.module_health_history = defaultdict(lambda: deque(maxlen=100))
        self.last_health_check = {}
        
    def perform_health_check(self, env_ref) -> Dict[str, Any]:
        """Perform comprehensive health check of all modules"""
        
        issues = []
        module_statuses = {}
        
        try:
            if env_ref and hasattr(env_ref, 'pipeline'):
                # Check pipeline modules
                pipeline = env_ref.pipeline
                
                for module in pipeline.modules:
                    module_name = module.__class__.__name__
                    
                    try:
                        # Get module health status
                        if hasattr(module, 'get_health_status'):
                            health = module.get_health_status()
                            module_statuses[module_name] = health
                            
                            # Check for issues
                            if health.get('status') != 'OK':
                                issues.append(f"{module_name}: {health.get('status')}")
                            
                            # Track health history
                            self.module_health_history[module_name].append({
                                'timestamp': datetime.now().isoformat(),
                                'status': health.get('status', 'UNKNOWN'),
                                'step_count': health.get('step_count', 0)
                            })
                            
                    except Exception as e:
                        issues.append(f"{module_name}: Health check failed - {e}")
                        module_statuses[module_name] = {'status': 'ERROR', 'error': str(e)}
            
        except Exception as e:
            issues.append(f"Module health check system failed: {e}")
        
        return {
            'issues': issues,
            'module_statuses': module_statuses,
            'modules_checked': len(module_statuses)
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall module health summary"""
        
        total_modules = len(self.module_health_history)
        healthy_modules = 0
        degraded_modules = 0
        failed_modules = 0
        
        for module_name, history in self.module_health_history.items():
            if history:
                latest_status = history[-1].get('status', 'UNKNOWN')
                if latest_status == 'OK':
                    healthy_modules += 1
                elif latest_status in ['DEGRADED', 'WARNING']:
                    degraded_modules += 1
                else:
                    failed_modules += 1
        
        return {
            'total_modules': total_modules,
            'healthy_modules': healthy_modules,
            'degraded_modules': degraded_modules,
            'failed_modules': failed_modules,
            'health_percentage': (healthy_modules / max(total_modules, 1)) * 100
        }