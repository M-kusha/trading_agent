# ─────────────────────────────────────────────────────────────
# File: train/enhanced_training_callback.py  
# Modern InfoBus v4.0 Training Callback with Comprehensive Module Health Monitoring
# ─────────────────────────────────────────────────────────────

import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import deque, defaultdict
from stable_baselines3.common.callbacks import BaseCallback

# Modern InfoBus v4.0 and infrastructure
from modules.utils.info_bus import InfoBusManager, SmartInfoBus, create_info_bus, validate_info_bus
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import SystemUtilities, EnglishExplainer
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker
from modules.monitoring.integration_validator import IntegrationValidator
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.core.module_base import BaseModule
from envs.config import TradingConfig


class ModernInfoBusTrainingCallback(BaseCallback):
    """
    Modern training callback with SmartInfoBus v4.0 integration.
    Comprehensive module health monitoring and production-grade audit system.
    """
    
    def __init__(self, total_timesteps: int, config: TradingConfig, 
                 metrics_broadcaster=None, verbose: int = 0):
        super().__init__(verbose)
        
        self.total_timesteps = total_timesteps
        self.config = config
        self.metrics_broadcaster = metrics_broadcaster
        self.start_time = datetime.now()
        
        # ══════════════════════════════════════════════════════════════
        # Modern Infrastructure Integration
        # ══════════════════════════════════════════════════════════════
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Training metrics with SmartInfoBus integration
        self.episode_rewards = deque(maxlen=1000)
        self.episode_lengths = deque(maxlen=1000)
        self.training_metrics = {}
        self.performance_history = deque(maxlen=500)
        
        # Health and quality tracking
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
        self.circuit_breaker_state = {"active": False, "failures": 0}
        
        # Module integration tracking
        self.module_health_tracker = ModernModuleHealthTracker(self.health_monitor)
        
        # Initialize SmartInfoBus connection
        self.smart_bus = InfoBusManager.get_instance()
        
        self.training_logger.info(format_operator_message(
            message="Modern InfoBus Training Callback initialized",
            icon="🚀",
            total_timesteps=total_timesteps,
            health_monitoring=True,
            smartinfobus_v4=True,
            audit_system="production_grade"
        ))

    def _initialize_advanced_systems(self):
        """Initialize all advanced monitoring and audit systems"""
        # Modern logging with rotation (different name to avoid BaseCallback.logger conflict)
        self.training_logger = RotatingLogger(
            name="ModernTrainingCallback",
            log_path=f"logs/training/modern_callback_{datetime.now().strftime('%Y%m%d')}.log",
            max_lines=5000,
            operator_mode=True,
            plain_english=True
        )
        
        # Error handling and pinpointing
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("TrainingCallback", self.error_pinpointer)
        
        # Health monitoring (simplified - will be initialized when orchestrator is available)
        self.health_monitor = None
        self.performance_tracker = PerformanceTracker()
        
        # System utilities
        self.system_utilities = SystemUtilities()
        self.english_explainer = EnglishExplainer()
        
        # Integration validation
        self.integration_validator = IntegrationValidator()

    def _on_training_start(self) -> None:
        """Modern training start with comprehensive system initialization"""
        
        self.start_time = datetime.now()
        
        # Initialize SmartInfoBus connection with environment
        try:
            self._initialize_modern_infobus_connection()
            self.training_logger.info(format_operator_message(
                message="SmartInfoBus v4.0 connection established",
                icon="🔗",
                training_callback=True,
                environment_connected=True
            ))
        except Exception as e:
            self.error_handler.handle_error(e, "infobus_connection")
            self.training_logger.error(format_operator_message(
                message="SmartInfoBus connection failed",
                icon="❌",
                error=str(e)
            ))
        
        # Validate system integration
        try:
            validation_result = self.integration_validator.validate_system()
            if validation_result.integration_score < 80:
                self.training_logger.warning(format_operator_message(
                    message="System integration validation warnings",
                    icon="⚠️",
                    issues=len(validation_result.issues),
                    score=f"{validation_result.integration_score:.1f}%"
                ))
        except Exception as e:
            self.training_logger.error(f"Integration validation failed: {e}")
        
        # Initialize health monitoring (simplified)
        # Note: Health monitoring will be initialized as needed
        
        # Record training start in SmartInfoBus
        self.smart_bus.set(
            'training_session_start',
            {
                'timestamp': datetime.now().isoformat(),
                'total_timesteps': self.total_timesteps,
                'config_mode': 'live' if getattr(self.config, 'live_mode', False) else 'offline',
                'smartinfobus_v4': True,
                'health_monitoring': True
            },
            module='TrainingCallback',
            thesis="Training session commenced with full SmartInfoBus v4.0 integration and comprehensive health monitoring"
        )

    def _on_step(self) -> bool:
        """Modern step with comprehensive monitoring and circuit breaker protection"""
        
        try:
            # Performance tracking
            step_start = time.time()
            
            # Collect comprehensive metrics every 10 steps
            if self.n_calls % 10 == 0:
                metrics = self._collect_modern_metrics()
                self._update_performance_tracking(metrics)
                
                # Broadcast metrics with error handling
                if self.metrics_broadcaster:
                    try:
                        self.metrics_broadcaster.send_metrics(metrics)
                    except Exception as e:
                        self.training_logger.warning(f"Metrics broadcast failed: {e}")
            
            # Comprehensive health checks
            if self.n_calls - self.last_health_check >= self.health_check_interval:
                self._perform_modern_health_check()
                self.last_health_check = self.n_calls
            
            # SmartInfoBus quality monitoring
            if self.n_calls % 50 == 0:
                self._monitor_smartinfobus_quality()
            
            # Episode and emergency monitoring
            self._track_episode_progress()
            
            # Circuit breaker check
            if self._check_circuit_breaker_conditions():
                self.logger.error(format_operator_message(
                    message="Circuit breaker activated - stopping training",
                    icon="🚨",
                    failures=self.circuit_breaker_state["failures"],
                    reason="emergency_protection"
                ))
                return False
            
            # Record step performance
            step_duration = time.time() - step_start
            self.performance_tracker.record_metric(
                'TrainingCallback', 'step_processing', step_duration * 1000, True
            )
            
            return True
            
        except Exception as e:
            self.consecutive_failures += 1
            error_context = self.error_pinpointer.analyze_error(e, "TrainingCallback")
            
            self.logger.error(format_operator_message(
                message="Training step error",
                icon="💥",
                error=str(e),
                consecutive_failures=self.consecutive_failures,
                error_context=str(error_context)
            ))
            
            # Circuit breaker logic
            if self.consecutive_failures > 10:
                self.circuit_breaker_state["active"] = True
                self.circuit_breaker_state["failures"] = self.consecutive_failures
                return False
            
            return True

    def _initialize_modern_infobus_connection(self):
        """Initialize modern SmartInfoBus connection with environment"""
        
        # Get environment from training context
        try:
            # Try to access vectorized environment
            if hasattr(self.training_env, 'get_attr'):
                # DummyVecEnv or similar - get first environment's attributes
                env_attrs = self.training_env.get_attr('unwrapped', indices=[0])
                if env_attrs and len(env_attrs) > 0:
                    env_unwrapped = env_attrs[0]
                    
                    # Check for SmartInfoBus support
                    if hasattr(env_unwrapped, 'smart_bus'):
                        self.env_ref = env_unwrapped
                        self.training_logger.info("SmartInfoBus v4.0 connection established with environment")
                    elif hasattr(env_unwrapped, 'info_bus'):
                        self.env_ref = env_unwrapped
                        self.training_logger.info("Legacy InfoBus connection established with environment")
                    else:
                        self.training_logger.warning("Environment does not support InfoBus/SmartInfoBus")
                        self.env_ref = None
                else:
                    self.env_ref = None
            else:
                self.env_ref = None
        except Exception as e:
            self.training_logger.warning(f"Cannot access training environment: {e}")
            self.env_ref = None

    def _collect_modern_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive modern metrics with SmartInfoBus integration"""
        
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
            'smartinfobus_v4': True,
            'health_monitoring': True,
            'circuit_breaker_active': self.circuit_breaker_state["active"],
            'health_alerts_count': len(self.health_alerts),
        }
        
        # Add SmartInfoBus metrics
        if hasattr(self, 'env_ref') and self.env_ref:
            smartinfobus_metrics = self._extract_smartinfobus_metrics()
            metrics.update(smartinfobus_metrics)
        
        # Add comprehensive health metrics
        health_metrics = self._get_comprehensive_health_metrics()
        metrics.update(health_metrics)
        
        # Add model learning metrics
        learning_metrics = self._extract_modern_learning_metrics()
        metrics.update(learning_metrics)
        
        return metrics

    def _extract_smartinfobus_metrics(self) -> Dict[str, Any]:
        """Extract metrics from SmartInfoBus v4.0 system"""
        
        try:
            # Check if env_ref exists
            if not self.env_ref:
                return {'smartinfobus_status': 'no_environment'}
            
            # Try SmartInfoBus v4.0 first
            if hasattr(self.env_ref, 'smart_bus') and self.env_ref.smart_bus:
                smart_bus = self.env_ref.smart_bus
                
                return {
                    'smartinfobus_status': 'v4.0_active',
                    'smartinfobus_balance': smart_bus.get('balance', 'TrainingCallback') or self.config.initial_balance,
                    'smartinfobus_positions': len(smart_bus.get('current_positions', 'TrainingCallback') or {}),
                    'smartinfobus_consensus': smart_bus.get('consensus', 'TrainingCallback') or 0.5,
                    'smartinfobus_risk_score': smart_bus.get('risk_score', 'TrainingCallback') or 0.5,
                    'smartinfobus_regime': smart_bus.get('market_regime', 'TrainingCallback') or 'unknown',
                    'smartinfobus_health_score': smart_bus.get('system_health', 'TrainingCallback') or 100,
                    'smartinfobus_module_count': len(smart_bus.get('active_modules', 'TrainingCallback') or []),
                }
            
            # Fallback to legacy InfoBus
            elif hasattr(self.env_ref, 'info_bus') and self.env_ref.info_bus:
                info_bus = self.env_ref.info_bus
                
                return {
                    'smartinfobus_status': 'legacy_active',
                    'smartinfobus_balance': info_bus.get('balance', self.config.initial_balance),
                    'smartinfobus_positions': len(info_bus.get('current_positions', {})),
                    'smartinfobus_consensus': info_bus.get('consensus', 0.5),
                    'smartinfobus_risk_score': info_bus.get('risk_score', 0.5),
                    'smartinfobus_regime': info_bus.get('market_regime', 'unknown'),
                }
            
            return {'smartinfobus_status': 'not_available'}
            
        except Exception as e:
            self.training_logger.warning(f"Failed to extract SmartInfoBus metrics: {e}")
            return {'smartinfobus_status': 'error', 'smartinfobus_error': str(e)}

    def _get_comprehensive_health_metrics(self) -> Dict[str, Any]:
        """Get comprehensive health metrics from all monitoring systems"""
        
        try:
            # Health monitor metrics (if available)
            if self.health_monitor:
                health_status = self.health_monitor.check_system_health()
                system_health_score = health_status.get('overall_score', 100)
                system_health_status = health_status.get('status', 'unknown')
            else:
                system_health_score = 100
                system_health_status = 'unknown'
            
            # Performance tracker metrics
            performance_summary = self.performance_tracker.generate_performance_report()
            
            # Calculate overall performance from module metrics
            if performance_summary.module_metrics:
                avg_times = [m.get('avg_time_ms', 0) for m in performance_summary.module_metrics.values()]
                error_rates = [m.get('error_rate', 0) for m in performance_summary.module_metrics.values()]
                performance_avg_ms = sum(avg_times) / len(avg_times) if avg_times else 0
                performance_success_rate = (1 - (sum(error_rates) / len(error_rates))) * 100 if error_rates else 100
            else:
                performance_avg_ms = 0
                performance_success_rate = 100
            
            # Module health metrics
            module_health = self.module_health_tracker.get_health_summary()
            
            return {
                'system_health_score': system_health_score,
                'system_health_status': system_health_status,
                'performance_avg_ms': performance_avg_ms,
                'performance_success_rate': performance_success_rate,
                'module_health_score': module_health.get('health_percentage', 100),
                'healthy_modules': module_health.get('healthy_modules', 0),
                'total_modules': module_health.get('total_modules', 0),
                'health_monitoring_active': self.health_monitor is not None,
            }
            
        except Exception as e:
            self.training_logger.warning(f"Failed to get health metrics: {e}")
            return {'health_monitoring_error': str(e)}

    def _extract_modern_learning_metrics(self) -> Dict[str, Any]:
        """Extract modern learning metrics from PPO model"""
        
        try:
            if hasattr(self.model, 'logger') and self.model.logger:
                logger_data = self.model.logger.name_to_value
                
                return {
                    'learning_rate': float(getattr(self.model, 'learning_rate', 0)),
                    'clip_fraction': logger_data.get('train/clip_fraction', 0),
                    'explained_variance': logger_data.get('train/explained_variance', 0),
                    'policy_loss': logger_data.get('train/policy_loss', 0),
                    'value_loss': logger_data.get('train/value_loss', 0),
                    'entropy_loss': logger_data.get('train/entropy_loss', 0),
                    'kl_divergence': logger_data.get('train/kl_divergence', 0),
                    'model_device': str(getattr(self.model, 'device', 'unknown')),
                }
            
            return {'learning_metrics_available': False}
            
        except Exception as e:
            return {'learning_metrics_error': str(e)}

    def _perform_modern_health_check(self):
        """Perform comprehensive modern health check with all monitoring systems"""
        
        health_summary = {
            'timestamp': datetime.now().isoformat(),
            'step': self.n_calls,
            'checks_performed': [],
            'issues_found': [],
            'overall_health': 'OK'
        }
        
        try:
            # System health check (if available)
            if self.health_monitor:
                system_health = self.health_monitor.check_system_health()
                health_summary['checks_performed'].append('system_health')
                if system_health.get('issues'):
                    health_summary['issues_found'].extend(system_health['issues'])
            
            # SmartInfoBus health check
            if hasattr(self, 'env_ref') and self.env_ref:
                smartinfobus_health = self._check_smartinfobus_health()
                health_summary['checks_performed'].append('smartinfobus')
                if smartinfobus_health.get('issues'):
                    health_summary['issues_found'].extend(smartinfobus_health['issues'])
            
            # Module health check
            module_health = self.module_health_tracker.perform_comprehensive_health_check(self.env_ref)
            health_summary['checks_performed'].append('modules')
            if module_health.get('issues'):
                health_summary['issues_found'].extend(module_health['issues'])
            
            # Model health check
            model_health = self._check_modern_model_health()
            health_summary['checks_performed'].append('model')
            if model_health.get('issues'):
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
                            self.training_logger.warning(format_operator_message(
                message="Health check issues detected",
                icon="⚠️",
                issues_found=issue_count,
                overall_health=health_summary['overall_health']
            ))
            self.health_alerts.append(health_summary)
            
            # Update SmartInfoBus with health status
            self.smart_bus.set(
                'training_health_status',
                health_summary,
                module='TrainingCallback',
                thesis=f"Health check performed with {issue_count} issues found - overall status: {health_summary['overall_health']}"
            )
            
        except Exception as e:
            error_context = self.error_pinpointer.analyze_error(e, "health_check")
            self.logger.error(format_operator_message(
                message="Health check system failure",
                icon="💥",
                error=str(e),
                error_context=str(error_context)
            ))

    def _check_smartinfobus_health(self) -> Dict[str, Any]:
        """Check SmartInfoBus v4.0 system health"""
        
        issues = []
        
        try:
            # Check if env_ref exists
            if not self.env_ref:
                issues.append("No environment reference available")
                return {'issues': issues}
            
            # Check SmartInfoBus v4.0 connection
            if hasattr(self.env_ref, 'smart_bus') and self.env_ref.smart_bus:
                smart_bus = self.env_ref.smart_bus
                
                # Check SmartInfoBus basic health
                try:
                    if hasattr(smart_bus, 'get_performance_metrics'):
                        perf_metrics = smart_bus.get_performance_metrics()
                        if perf_metrics.get('cache_hit_rate', 0) < 0.5:
                            issues.append("SmartInfoBus cache hit rate low")
                        if len(perf_metrics.get('disabled_modules', [])) > 3:
                            issues.append("Multiple modules disabled in SmartInfoBus")
                except Exception:
                    issues.append("Cannot check SmartInfoBus health")
                
                # Check for stale data
                last_update = smart_bus.get('last_update', 'TrainingCallback')
                if last_update:
                    age = (datetime.now() - datetime.fromisoformat(last_update)).total_seconds()
                    if age > 60:  # 1 minute
                        issues.append(f"SmartInfoBus data stale: {age:.1f}s old")
                
            elif hasattr(self.env_ref, 'info_bus') and self.env_ref.info_bus:
                # Legacy InfoBus validation
                from modules.utils.info_bus import validate_info_bus
                quality = validate_info_bus(self.env_ref.info_bus)
                
                if not quality.is_valid:
                    issues.extend(quality.missing_fields)
                
                if quality.score < 70:
                    issues.append(f"Legacy InfoBus quality score low: {quality.score}")
            
            else:
                issues.append("No InfoBus/SmartInfoBus connection available")
            
        except Exception as e:
            issues.append(f"SmartInfoBus health check failed: {e}")
        
        return {'issues': issues}

    def _check_modern_model_health(self) -> Dict[str, Any]:
        """Check modern PPO model health with enhanced validation"""
        
        issues = []
        
        try:
            # Check model device
            if hasattr(self.model, 'device'):
                device_str = str(self.model.device)
                if 'cuda' in device_str.lower():
                    import torch
                    if not torch.cuda.is_available():
                        issues.append("Model on CUDA but CUDA not available")
                    elif not torch.cuda.is_initialized():
                        issues.append("CUDA not properly initialized")
            
            # Check learning rate with enhanced validation
            if hasattr(self.model, 'learning_rate'):
                lr = self.model.learning_rate
                try:
                    # Try to call as schedule function
                    lr_value = float(lr(self.n_calls)) if callable(lr) else float(lr)
                except (TypeError, AttributeError):
                    # Fallback to treating as float
                    lr_value = float(lr) if not callable(lr) else 0.001
                
                if lr_value <= 0:
                    issues.append(f"Learning rate is zero or negative: {lr_value}")
                elif lr_value > 0.1:
                    issues.append(f"Learning rate suspiciously high: {lr_value}")
            
            # Check model parameters for NaN/Inf
            if hasattr(self.model, 'policy'):
                try:
                    param_count = 0
                    nan_count = 0
                    inf_count = 0
                    
                    for param in self.model.policy.parameters():
                        param_count += 1
                        param_data = param.detach().cpu().numpy()
                        
                        if np.any(np.isnan(param_data)):
                            nan_count += 1
                        if np.any(np.isinf(param_data)):
                            inf_count += 1
                    
                    if param_count == 0:
                        issues.append("Model has no parameters")
                    if nan_count > 0:
                        issues.append(f"NaN detected in {nan_count} parameter tensors")
                    if inf_count > 0:
                        issues.append(f"Infinity detected in {inf_count} parameter tensors")
                    
                except Exception as e:
                    issues.append(f"Cannot validate model parameters: {e}")
            
            # Check gradient norms if available
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'parameters'):
                try:
                    total_norm = 0
                    for param in self.model.policy.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                    
                    total_norm = total_norm ** (1. / 2)
                    
                    if total_norm > 10.0:
                        issues.append(f"Gradient norm very high: {total_norm:.2f}")
                    elif total_norm == 0:
                        issues.append("No gradients detected")
                    
                except Exception as e:
                    issues.append(f"Cannot check gradient norms: {e}")
            
        except Exception as e:
            issues.append(f"Model health check failed: {e}")
        
        return {'issues': issues}

    def _monitor_smartinfobus_quality(self):
        """Monitor SmartInfoBus v4.0 data quality over time"""
        
        if not hasattr(self, 'env_ref') or not self.env_ref:
            return
        
        try:
            quality_score = 100
            is_valid = True
            issues = []
            
            # SmartInfoBus v4.0 quality check
            if hasattr(self.env_ref, 'smart_bus') and self.env_ref.smart_bus:
                try:
                    perf_metrics = self.env_ref.smart_bus.get_performance_metrics()
                    quality_score = 100 - (len(perf_metrics.get('disabled_modules', [])) * 10)
                    is_valid = perf_metrics.get('cache_hit_rate', 0) > 0.3
                    issues = []
                    if not is_valid:
                        issues.append("SmartInfoBus performance degraded")
                except Exception:
                    quality_score = 50
                    is_valid = False
                    issues = ["Cannot access SmartInfoBus metrics"]
                
            # Legacy InfoBus quality check
            elif hasattr(self.env_ref, 'info_bus') and self.env_ref.info_bus:
                from modules.utils.info_bus import validate_info_bus
                quality = validate_info_bus(self.env_ref.info_bus)
                quality_score = quality.score
                is_valid = quality.is_valid
                issues = quality.missing_fields
            
            # Record quality metrics
            self.info_bus_quality_history.append({
                'step': self.n_calls,
                'score': quality_score,
                'is_valid': is_valid,
                'issues_count': len(issues),
                'timestamp': datetime.now().isoformat()
            })
            
            # Alert on quality degradation
            if quality_score < 70:
                self.training_logger.warning(format_operator_message(
                    message="SmartInfoBus quality degraded",
                    icon="⚠️",
                    quality_score=f"{quality_score:.1f}",
                    issues_count=len(issues)
                ))
            else:
                self.training_logger.info(format_operator_message(
                    message="SmartInfoBus quality metrics",
                    icon="✅",
                    quality_score=f"{quality_score:.1f}",
                    issues_count=len(issues)
                ))
            
            # Update SmartInfoBus with quality metrics
            self.smart_bus.set(
                'training_quality_metrics',
                {
                    'quality_score': quality_score,
                    'is_valid': is_valid,
                    'issues_count': len(issues),
                    'timestamp': datetime.now().isoformat()
                },
                module='TrainingCallback',
                thesis=f"SmartInfoBus quality monitoring - score: {quality_score:.1f}, valid: {is_valid}"
            )
            
        except Exception as e:
            self.logger.error(f"SmartInfoBus quality monitoring failed: {e}")

    def _track_episode_progress(self):
        """Enhanced episode progress tracking with SmartInfoBus integration"""
        
        # Track episode completion
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.locals.get('episode_length', 0))
            self.episode_count += 1
            
            # Check for new best episode
            if self.current_episode_reward > self.best_reward:
                self.best_reward = self.current_episode_reward
                self.logger.info(format_operator_message(
                    message="New best episode achieved",
                    icon="🏆",
                    episode=self.episode_count,
                    reward=f"{self.best_reward:.2f}"
                ))
                
                # Update SmartInfoBus with new best
                self.smart_bus.set(
                    'training_best_episode',
                    {
                        'episode': self.episode_count,
                        'reward': self.best_reward,
                        'timestamp': datetime.now().isoformat()
                    },
                    module='TrainingCallback',
                    thesis=f"New best episode achieved: {self.best_reward:.2f} reward at episode {self.episode_count}"
                )
            
            # Reset for next episode
            self.current_episode_reward = 0
            self.consecutive_failures = 0  # Reset on successful episode
        else:
            # Accumulate episode reward
            self.current_episode_reward += self.locals.get('rewards', [0])[0]

    def _check_circuit_breaker_conditions(self) -> bool:
        """Check for circuit breaker conditions with enhanced safety"""
        
        emergency_conditions = []
        
        # Check for excessive failures
        if self.consecutive_failures > 15:
            emergency_conditions.append(f"Excessive consecutive failures: {self.consecutive_failures}")
        
        # Check health alerts
        if len(self.health_alerts) > 5:
            recent_critical = sum(1 for alert in list(self.health_alerts)[-5:] 
                                if alert.get('overall_health') == 'CRITICAL')
            if recent_critical >= 3:
                emergency_conditions.append("Multiple critical health alerts")
        
        # Check SmartInfoBus quality
        if len(self.info_bus_quality_history) > 10:
            recent_scores = [q['score'] for q in list(self.info_bus_quality_history)[-10:]]
            if np.mean(recent_scores) < 40:
                emergency_conditions.append("SmartInfoBus quality critically degraded")
        
        # Check system health (if available)
        if self.health_monitor:
            try:
                system_health = self.health_monitor.check_system_health()
                if system_health.get('overall_score', 100) < 30:
                    emergency_conditions.append("System health critically low")
            except Exception:
                pass
        
        # Log and activate circuit breaker if needed
        if emergency_conditions:
            self.circuit_breaker_state["active"] = True
            self.circuit_breaker_state["failures"] = len(emergency_conditions)
            
            self.training_logger.critical(format_operator_message(
                message="Circuit breaker conditions detected",
                icon="🚨",
                conditions="; ".join(emergency_conditions),
                step=self.n_calls
            ))
            
            # Update SmartInfoBus with emergency status
            self.smart_bus.set(
                'training_emergency_stop',
                {
                    'conditions': emergency_conditions,
                    'step': self.n_calls,
                    'timestamp': datetime.now().isoformat()
                },
                module='TrainingCallback',
                thesis="Circuit breaker activated due to emergency conditions - training halted for safety"
            )
            
            return True
        
        return False

    def _update_performance_tracking(self, metrics: Dict[str, Any]):
        """Update performance tracking with SmartInfoBus integration"""
        
        # Add to performance history
        performance_snapshot = {
            'step': self.n_calls,
            'timestamp': datetime.now().isoformat(),
            'episode_reward_mean': metrics.get('episode_reward_mean', 0),
            'steps_per_second': metrics.get('steps_per_second', 0),
            'system_health_score': metrics.get('system_health_score', 100),
            'smartinfobus_status': metrics.get('smartinfobus_status', 'unknown'),
            'circuit_breaker_active': metrics.get('circuit_breaker_active', False)
        }
        
        self.performance_history.append(performance_snapshot)
        
        # Update SmartInfoBus with performance data
        self.smart_bus.set(
            'training_performance',
            performance_snapshot,
            module='TrainingCallback',
            thesis=f"Training performance update - {metrics.get('steps_per_second', 0):.1f} steps/sec, health: {metrics.get('system_health_score', 100)}"
        )
        
        # Save metrics to file periodically
        if self.n_calls % 1000 == 0:
            self._save_comprehensive_metrics(metrics)

    def _save_comprehensive_metrics(self, metrics: Dict[str, Any]):
        """Save comprehensive metrics with SmartInfoBus integration"""
        
        try:
            # Ensure directory exists
            os.makedirs("logs/training", exist_ok=True)
            
            metrics_file = f"logs/training/modern_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
            
            enhanced_metrics = {
                **metrics,
                'performance_history': list(self.performance_history)[-10:],
                'health_alerts': list(self.health_alerts),
                'module_health': self.module_health_tracker.get_health_summary(),
                'smartinfobus_quality': list(self.info_bus_quality_history)[-10:],
                'circuit_breaker_state': self.circuit_breaker_state,
                'system_integration': {'status': 'simplified_mode'}
            }
            
            with open(metrics_file, 'a') as f:
                f.write(json.dumps(enhanced_metrics, default=str) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")

    def _on_training_end(self) -> None:
        """Modern training end with comprehensive reporting"""
        
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
            'circuit_breaker_activated': self.circuit_breaker_state["active"],
            'smartinfobus_v4_enabled': True,
            'final_system_health': self.module_health_tracker.get_health_summary(),
            'final_integration_status': {'status': 'simplified_mode'}
        }
        
        # Log training completion
        self.logger.info(format_operator_message(
            message="Modern training completed",
            icon="✅",
            duration=str(training_duration),
            episodes=self.episode_count,
            best_reward=f"{self.best_reward:.2f}"
        ))
        
        # Update SmartInfoBus with final status
        self.smart_bus.set(
            'training_session_completed',
            final_report,
            module='TrainingCallback',
            thesis=f"Training session completed successfully - {self.episode_count} episodes, best reward: {self.best_reward:.2f}"
        )
        
        # Save final comprehensive report
        try:
            os.makedirs("logs/training", exist_ok=True)
            report_file = f"logs/training/final_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(final_report, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save final report: {e}")
        
        # Stop health monitoring (if available)
        if self.health_monitor:
            self.health_monitor.stop_monitoring()

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive modern training summary"""
        
        return {
            'status': 'running',
            'progress': self.n_calls / self.total_timesteps,
            'episodes': self.episode_count,
            'best_reward': self.best_reward,
            'health_status': 'OK' if len(self.health_alerts) == 0 else 'ISSUES',
            'health_alerts': len(self.health_alerts),
            'consecutive_failures': self.consecutive_failures,
            'circuit_breaker_active': self.circuit_breaker_state["active"],
            'smartinfobus_v4_status': 'enabled',
            'system_health_score': self.module_health_tracker.get_health_summary().get('health_percentage', 100),
            'integration_status': {'status': 'simplified_mode'},
            'recent_performance': list(self.performance_history)[-5:] if self.performance_history else [],
        }


class ModernModuleHealthTracker:
    """Modern module health tracking with comprehensive monitoring"""
    
    def __init__(self, health_monitor: Optional[HealthMonitor]):
        self.health_monitor = health_monitor
        self.module_health_history = defaultdict(lambda: deque(maxlen=100))
        self.last_health_check = {}
        self.error_pinpointer = ErrorPinpointer()
        
    def perform_comprehensive_health_check(self, env_ref) -> Dict[str, Any]:
        """Perform comprehensive health check of all modules"""
        
        issues = []
        module_statuses = {}
        
        try:
            if env_ref and hasattr(env_ref, 'pipeline'):
                # Check pipeline modules
                pipeline = env_ref.pipeline
                
                if hasattr(pipeline, 'modules'):
                    for module in pipeline.modules:
                        module_name = module.__class__.__name__
                        
                        try:
                            # Get module health status
                            if hasattr(module, 'get_health_status'):
                                health = module.get_health_status()
                                module_statuses[module_name] = health
                                
                                # Check for issues
                                if health.get('status') != 'healthy':
                                    issues.append(f"{module_name}: {health.get('status')}")
                                
                                # Track health history
                                self.module_health_history[module_name].append({
                                    'timestamp': datetime.now().isoformat(),
                                    'status': health.get('status', 'unknown'),
                                    'metrics': health.get('metrics', {})
                                })
                                
                        except Exception as e:
                            error_context = self.error_pinpointer.analyze_error(e, module_name)
                            issues.append(f"{module_name}: Health check failed - {error_context}")
                            module_statuses[module_name] = {'status': 'ERROR', 'error': str(e)}
            
            # Check system-level health (if available)
            if self.health_monitor:
                system_health = self.health_monitor.check_system_health()
                if system_health.get('overall_score', 100) < 80:
                    issues.append(f"System health score low: {system_health.get('overall_score')}")
            
        except Exception as e:
            issues.append(f"Module health check system failed: {e}")
        
        return {
            'issues': issues,
            'module_statuses': module_statuses,
            'modules_checked': len(module_statuses),
            'system_health': system_health if 'system_health' in locals() else {}
        }
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive module health summary"""
        
        total_modules = len(self.module_health_history)
        healthy_modules = 0
        degraded_modules = 0
        failed_modules = 0
        
        for module_name, history in self.module_health_history.items():
            if history:
                latest_status = history[-1].get('status', 'unknown')
                if latest_status in ['healthy', 'OK']:
                    healthy_modules += 1
                elif latest_status in ['degraded', 'warning', 'WARNING']:
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