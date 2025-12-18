# ─────────────────────────────────────────────────────────────
# File: modules/models/world_model.py
# [ROCKET] PRODUCTION-READY Enhanced World Model
# Advanced market simulation with SmartInfoBus integration and intelligent automation
# ─────────────────────────────────────────────────────────────

import os
import asyncio
import time
import contextlib
import threading
import warnings
import copy
from modules.contracts import module_args
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import datetime
import random
from typing import Any, Dict, Optional, List, Tuple, Union, cast, TYPE_CHECKING
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from threading import Event, Lock
from contextlib import contextmanager

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusRiskMixin, SmartInfoBusStateMixin, SmartInfoBusTradingMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.performance_tracker import PerformanceTracker

# Optional health monitor import (fallback stub if missing)
try:
    from modules.monitoring.health_monitor import HealthMonitor  # type: ignore
except Exception:
    class HealthMonitor:  # minimal no-op fallback
        def record(self, name: str, status: str = "unknown", **kwargs):
            pass


class WorldModelMode(Enum):
    """World model operational modes"""
    INITIALIZATION = "initialization"
    DATA_COLLECTION = "data_collection"
    TRAINING = "training"
    CALIBRATION = "calibration"
    ACTIVE_PREDICTION = "active_prediction"
    SCENARIO_GENERATION = "scenario_generation"
    OPTIMIZATION = "optimization"
    MAINTENANCE = "maintenance"
    ERROR_RECOVERY = "error_recovery"


class PredictionConfidence(Enum):
    """Prediction confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class WorldModelConfig:
    """Configuration for Enhanced World Model"""
    # Neural architecture
    input_size: int = 16
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    attention_heads: int = 4
    
    # Training parameters
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    batch_size: int = 64
    
    # Sequence parameters
    sequence_length: int = 50
    prediction_horizon: int = 10
    scenario_steps: int = 20
    
    # Data management
    history_size: int = 1000
    min_training_samples: int = 100
    validation_split: float = 0.2
    
    # Performance thresholds
    max_processing_time_ms: float = 200
    circuit_breaker_threshold: int = 5
    min_prediction_quality: float = 0.6
    min_training_quality: float = 0.7
    
    # Device and optimization
    device: str = "auto"            # 'auto'|'cpu'|'cuda'|'mps'
    use_mixed_precision: bool = False
    compile_model: bool = False
    deterministic: bool = False      # reproducibility
    
    # Monitoring parameters
    health_check_interval: int = 60
    performance_window: int = 100
    confidence_threshold: float = 0.5

    # Live vs training mode controls
    live_mode: bool = False                    # True = live trading, False = backtest/training
    enable_online_training: bool = True        # Allow training during operation
    enable_scenario_generation: bool = True    # Allow scenario generation

    # Persistence
    save_dir: str = "artifacts/models/world_model"

    # Convenience helpers
    def to_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__.keys()}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "WorldModelConfig":
        defaults = {k: getattr(WorldModelConfig, k, None) for k in WorldModelConfig.__dataclass_fields__.keys()}
        merged = {**defaults, **(d or {})}
        return WorldModelConfig(**merged)

@module(**module_args(
    "EnhancedWorldModel",
    description="Deterministic multi-window feature extraction with circuit breaker, monitoring, and explainability.",
    error_handling=True,
    hot_reload=True,
    timeout_ms=15000,
))
class EnhancedWorldModel(BaseModule, SmartInfoBusRiskMixin, SmartInfoBusTradingMixin, SmartInfoBusStateMixin, nn.Module):
    """
    [ROCKET] Advanced world model for market simulation with SmartInfoBus integration.
    Provides intelligent market predictions, scenario generation, and comprehensive analytics.
    """

    # Help Pylance treat self.config like WorldModelConfig for attribute access during type checking
    if TYPE_CHECKING:
        config: WorldModelConfig

    def __init__(self, 
                 config: Optional[Union[WorldModelConfig, Dict[str, Any]]] = None,
                 genome: Optional[Dict[str, Any]] = None,
                 action_dim: int = 12,
                 **kwargs):
        
        # Store parameters for use in _initialize()
        if isinstance(config, dict):
            self._init_config = WorldModelConfig.from_dict(config)
        elif isinstance(config, WorldModelConfig):
            self._init_config = config
        else:
            self._init_config = WorldModelConfig()
        self._init_genome = genome
        self._init_action_dim = int(action_dim)
        self._init_kwargs = kwargs
        
        # Initialize nn.Module first (no parameters)  
        nn.Module.__init__(self)
        
        # Initialize BaseModule second (no parameters) - this will call _initialize()
        BaseModule.__init__(self)

    def _initialize(self):
        """Initialize module-specific state (called by BaseModule.__init__)"""
        # Apply stored initialization parameters
        # Keep a typed config alongside BaseModule's dict config
        self.wm_config: WorldModelConfig = self._init_config
        # Mirror into BaseModule.config (dict) for compatibility with the framework
        try:
            self.set_config(self.wm_config.to_dict())
        except Exception:
            # Fallback without logging failure
            self.config = self.wm_config.to_dict()  # type: ignore[assignment]
        self.action_dim = self._init_action_dim
        
        # Apply genome parameters early for mixin initialization
        if self._init_genome:
            self._apply_genome_early(self._init_genome)
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize world model state
        self._initialize_world_model_state(self._init_genome)
        
        # Initialize neural components
        self._initialize_neural_components_async()
        
        # Start monitoring after all initialization is complete
        self._start_monitoring()
        
        self.logger.info(format_operator_message(
            message="Enhanced world model ready",
            icon="🌍",
            input_size=self.wm_config.input_size,
            hidden_size=self.wm_config.hidden_size,
            sequence_length=self.wm_config.sequence_length,
            device=str(self.device),
            config_loaded=True
        ))

    def _apply_genome_early(self, genome: Dict[str, Any]):
        """Apply genome parameters that affect initialization"""
        self.wm_config.sequence_length = int(genome.get("sequence_length", self.wm_config.sequence_length))
        self.wm_config.hidden_size = int(genome.get("hidden_size", self.wm_config.hidden_size))
        self.wm_config.num_layers = int(genome.get("num_layers", self.wm_config.num_layers))
        # Mirror changes to BaseModule's dict config for framework compatibility
        try:
            self.set_config(self.wm_config.to_dict())
        except Exception:
            self.config = self.wm_config.to_dict()  # type: ignore[assignment]

    def _initialize_advanced_systems(self):
        """Initialize advanced systems for world model"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="EnhancedWorldModel", 
            log_path="logs/models/enhanced_world_model.log", 
            max_lines=5000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("EnhancedWorldModel", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        self.health_monitor = HealthMonitor()

        # Seed & determinism
        if getattr(self._init_config, "deterministic", False):
            np.random.seed(42)
            random.seed(42)
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                try:
                    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                    torch.backends.cudnn.benchmark = False     # type: ignore[attr-defined]
                except Exception:
                    pass
        
        # Circuit breaker for model operations (with HALF_OPEN)
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0.0,
            'state': 'CLOSED',  # CLOSED | OPEN | HALF_OPEN
            'threshold': self.wm_config.circuit_breaker_threshold,
            'reset_time': 300.0
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()

        # Concurrency
        self._monitoring_active: bool = False
        self._monitor_evt: Event = Event()
        self._monitor_evt.clear()
        self._monitor_thread: Optional[threading.Thread] = None
        self._train_lock: Lock = Lock()
        self._predict_lock: Lock = Lock()

    def _initialize_world_model_state(self, genome: Optional[Dict[str, Any]]):
        """Initialize world model state"""
        # Initialize mixin states
        self._initialize_risk_state()
        self._initialize_trading_state()
        self._initialize_state_management()

        # Current operational mode
        self.current_mode = WorldModelMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()

        # Genome and evolution
        self.genome = self._initialize_genome(genome)

        # Model state
        self.is_trained = False
        self.model_confidence = 0.0
        self.prediction_quality = 0.0
        self.training_quality = 0.0
        self.stability_score = 1.0
        self.last_training_time = None

        # Data management
        self.market_history = deque(maxlen=self.wm_config.history_size)
        self.feature_history = deque(maxlen=self.wm_config.history_size)
        self.prediction_history = deque(maxlen=self.wm_config.performance_window)
        self.training_history = deque(maxlen=50)

        # Performance tracking
        self.prediction_errors = deque(maxlen=self.wm_config.performance_window)
        self.confidence_history = deque(maxlen=self.wm_config.performance_window)
        self.training_curves = {
            'loss': deque(maxlen=100),
            'val_loss': deque(maxlen=100),
            'accuracy': deque(maxlen=100),
            'gradient_norm': deque(maxlen=100)
        }

        # Analytics and insights
        self.feature_importance = {}       # Context-dependent weights (regime/session/volatility)
        self.feature_stats = {}            # Per-feature variance/history statistics
        self.attention_patterns = deque(maxlen=50)
        self.scenario_cache = {}
        self.prediction_analytics = defaultdict(list)

        # External integrations
        self.external_model_sources = {}
        self.ensemble_weights = {}

        # Device management (auto)
        req = (self.wm_config.device or "auto").lower()
        if req == "auto":
            if torch.cuda.is_available():
                chosen = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():  # type: ignore[attr-defined]
                chosen = "mps"
            else:
                chosen = "cpu"
        else:
            chosen = req if req in ("cpu", "cuda", "mps") else "cpu"
        self.device = torch.device(chosen)

    def _initialize_genome(self, genome: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Initialize evolutionary genome"""
        default_genome = {
            "input_size": self.wm_config.input_size,
            "hidden_size": self.wm_config.hidden_size,
            "num_layers": self.wm_config.num_layers,
            "dropout": self.wm_config.dropout,
            "attention_heads": self.wm_config.attention_heads,
            "learning_rate": self.wm_config.learning_rate,
            "sequence_length": self.wm_config.sequence_length,
            "prediction_horizon": self.wm_config.prediction_horizon,
            "batch_size": self.wm_config.batch_size,
            "gradient_clip": self.wm_config.gradient_clip,
            "weight_decay": self.wm_config.weight_decay,
            "scenario_steps": self.wm_config.scenario_steps
        }
        
        if genome:
            # Validate and clip genome values
            for key, value in genome.items():
                if key in default_genome:
                    if key == "hidden_size":
                        default_genome[key] = int(np.clip(value, 32, 512))
                    elif key == "num_layers":
                        default_genome[key] = int(np.clip(value, 1, 6))
                    elif key == "learning_rate":
                        default_genome[key] = float(np.clip(value, 1e-5, 1e-2))
                    elif key == "dropout":
                        default_genome[key] = float(np.clip(value, 0.0, 0.8))
                    elif key == "sequence_length":
                        default_genome[key] = int(np.clip(value, 20, 200))
                    else:
                        default_genome[key] = value
            
            # Apply validated genome to config
            for key, value in default_genome.items():
                if hasattr(self.wm_config, key):
                    setattr(self.wm_config, key, value)
            # Keep BaseModule dict config in sync
            try:
                self.set_config(self.wm_config.to_dict())
            except Exception:
                self.config = self.wm_config.to_dict()  # type: ignore[assignment]
        
        return default_genome

    def _initialize_neural_components_async(self):
        """Initialize neural network components"""
        try:
            # Main LSTM backbone
            self.lstm = nn.LSTM(
                self.wm_config.input_size,
                self.wm_config.hidden_size,
                self.wm_config.num_layers,
                batch_first=True,
                dropout=self.wm_config.dropout if self.wm_config.num_layers > 1 else 0.0,
                bidirectional=False
            )
            
            # Multi-head attention for temporal patterns
            self.attention = nn.MultiheadAttention(
                self.wm_config.hidden_size, 
                num_heads=self.wm_config.attention_heads,
                dropout=self.wm_config.dropout,
                batch_first=True
            )
            
            # Prediction heads with enhanced architecture
            self.price_head = nn.Sequential(
                nn.Linear(self.wm_config.hidden_size, self.wm_config.hidden_size),
                nn.LayerNorm(self.wm_config.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.wm_config.dropout),
                nn.Linear(self.wm_config.hidden_size, self.wm_config.hidden_size // 2),
                nn.ReLU(),
                nn.Linear(self.wm_config.hidden_size // 2, 4)  # 4 price predictions
            )
            
            self.volatility_head = nn.Sequential(
                nn.Linear(self.wm_config.hidden_size, self.wm_config.hidden_size // 2),
                nn.LayerNorm(self.wm_config.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.wm_config.dropout),
                nn.Linear(self.wm_config.hidden_size // 2, 4)  # 4 volatility predictions
            )
            
            self.regime_head = nn.Sequential(
                nn.Linear(self.wm_config.hidden_size, self.wm_config.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.wm_config.dropout),
                nn.Linear(self.wm_config.hidden_size // 2, 4),  # 4 regime classes
                nn.Softmax(dim=-1)
            )
            
            # Confidence estimation head
            self.confidence_head = nn.Sequential(
                nn.Linear(self.wm_config.hidden_size, self.wm_config.hidden_size // 4),
                nn.ReLU(),
                nn.Linear(self.wm_config.hidden_size // 4, 1),
                nn.Sigmoid()
            )
            
            # Context integration network
            self.context_encoder = nn.Sequential(
                nn.Linear(16, self.wm_config.hidden_size // 2),  # 16 context features
                nn.LayerNorm(self.wm_config.hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(self.wm_config.dropout * 0.5),
                nn.Linear(self.wm_config.hidden_size // 2, self.wm_config.hidden_size // 2)
            )
            
            # Feature fusion layer
            self.fusion_layer = nn.Sequential(
                nn.Linear(self.wm_config.hidden_size + self.wm_config.hidden_size // 2, self.wm_config.hidden_size),
                nn.LayerNorm(self.wm_config.hidden_size),
                nn.ReLU(),
                nn.Dropout(self.wm_config.dropout)
            )
            
            # Move all components to device (avoid ambiguous self.to typing)
            try:
                self.lstm.to(self.device)
                self.attention.to(self.device)
                self.price_head.to(self.device)
                self.volatility_head.to(self.device)
                self.regime_head.to(self.device)
                self.confidence_head.to(self.device)
                self.context_encoder.to(self.device)
                self.fusion_layer.to(self.device)
            except Exception:
                pass
            
            # Initialize optimizer with modern settings
            self.optimizer = optim.AdamW(
                self.parameters(),
                lr=self.wm_config.learning_rate,
                weight_decay=self.wm_config.weight_decay,
                eps=1e-8,
                betas=(0.9, 0.999)
            )
            
            # Learning rate scheduler
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.8,
                patience=5
            )
            
            # Loss functions
            self.mse_loss = nn.MSELoss()
            self.huber_loss = nn.SmoothL1Loss()
            self.ce_loss = nn.CrossEntropyLoss()
            
            # AMP support (CUDA only for stability) with new API
            self._amp_enabled = bool(self.wm_config.use_mixed_precision) and (self.device.type == "cuda")
            try:
                if self._amp_enabled:
                    # Prefer new API when available
                    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
                        self._scaler = torch.amp.GradScaler("cuda", enabled=True)  # type: ignore[call-arg]
                    else:
                        self._scaler = torch.cuda.amp.GradScaler(enabled=True)  # type: ignore[attr-defined]
                else:
                    # Do not instantiate any scaler when AMP is disabled to avoid deprecation warnings
                    self._scaler = None
            except Exception:
                # Fallback: disable scaler on any error
                self._scaler = None

            # Initialize weights
            self._initialize_weights()
            
            # Model compilation for performance (if supported)
            if self.wm_config.compile_model and hasattr(torch, 'compile'):
                try:
                    self.lstm = torch.compile(self.lstm)           # type: ignore[attr-defined]
                    self.price_head = torch.compile(self.price_head)  # type: ignore[attr-defined]
                except Exception as ce:
                    warnings.warn(f"torch.compile failed: {ce}")
            
            self.logger.info("Neural components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Neural component initialization failed: {e}")
            self._trip_circuit_breaker()

    def _initialize_weights(self):
        """Initialize neural network weights with advanced techniques"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.kaiming_uniform_(param.data, nonlinearity='sigmoid')
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        nn.init.zeros_(param.data)
                        # Set forget gate bias to 1
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _start_monitoring(self):
        """Start background monitoring for world model (idempotent)."""
        if getattr(self, "_monitoring_active", False):
            return  # already running

        self._monitoring_active = True
        self._monitor_evt.clear()

        def monitoring_loop():
            try:
                while self._monitoring_active and not self._monitor_evt.is_set():
                    try:
                        self._update_model_health()
                        self._analyze_prediction_effectiveness()
                        self._adapt_model_parameters()
                        self._cleanup_old_data()
                        # Lightweight heartbeat
                        try:
                            self.health_monitor.record("EnhancedWorldModel", status=self._health_status)
                        except Exception:
                            pass
                    except Exception as inner_e:
                        self.logger.error(f"World model monitoring error: {inner_e}")
                    self._monitor_evt.wait(timeout=self.wm_config.health_check_interval)
            except Exception as e:
                self.logger.error(f"Monitoring loop failure: {e}")

        self._monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitor_thread.start()

    def _stop_monitoring_thread(self, timeout: float = 2.0):
        """Internal: stop monitor thread."""
        self._monitoring_active = False
        self._monitor_evt.set()
        try:
            if hasattr(self, "_monitor_thread") and self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=timeout)
        except Exception:
            pass


    async def _initialize_smartinfobus(self):
        """Initialize module with SmartInfoBus integration"""
        try:
            # Set initial world model status
            initial_status = {
                "current_mode": self.current_mode.value,
                "is_trained": self.is_trained,
                "model_confidence": self.model_confidence,
                "prediction_quality": self.prediction_quality,
                "device": str(self.device),
                "genome_config": self.genome.copy()
            }
            
            self.smart_bus.set(
                'market_predictions',
                initial_status,
                module='EnhancedWorldModel',
                thesis="Initial enhanced world model status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"World model initialization failed: {e}")
            return False

    def _maybe_half_open(self):
        """Move circuit breaker to HALF_OPEN after reset window."""
        if self.circuit_breaker['state'] == 'OPEN':
            if time.time() - self.circuit_breaker['last_failure'] > self.circuit_breaker.get('reset_time', 300.0):
                self.circuit_breaker['state'] = 'HALF_OPEN'
                self.circuit_breaker['failures'] = 0
                self.logger.info("[RELOAD] Circuit breaker moved to HALF_OPEN")

    def _trip_circuit_breaker(self):
        """Increment failures and open circuit if needed."""
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
            self._health_status = 'critical'
            self.current_mode = WorldModelMode.ERROR_RECOVERY

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process world model operations with enhanced analytics"""
        start_time = time.time()
        # Enforce an internal time budget to avoid stage-level timeouts
        try:
            cfg_budget_ms = float(getattr(self.wm_config, 'max_processing_time_ms', 200))
        except Exception:
            cfg_budget_ms = 200.0
        budget_s = max(0.1, cfg_budget_ms / 1000.0)
        end_time = start_time + budget_s

        def remaining() -> float:
            rem = end_time - time.time()
            return 0.0 if rem <= 0 else rem
        self._maybe_half_open()
        
        try:
            # Extract market data from SmartInfoBus
            market_data = await self._extract_market_data(**inputs)
            
            # Check if we actually got meaningful market data
            # The _extract_market_data method returns {} for market_data key if bus has no data
            actual_market_data = market_data.get('market_data', {}) if market_data else {}
            if not market_data or not actual_market_data:
                return await self._handle_no_data_fallback()
            
            # Track what operations we've successfully completed
            completed_operations = []
            
            # Update market context and features (budget-aware)
            context_result = {}
            if remaining() > 0:
                context_result = await self._update_market_context_async(market_data)
                completed_operations.append('context')
            
            # Process market data and update history
            processing_result = {}
            if remaining() > 0:
                processing_result = await self._process_market_data_async(market_data)
                completed_operations.append('processing')
            
            # Generate predictions if model is ready
            prediction_result = {}
            if remaining() > 0 and self.is_trained and len(self.market_history) >= self.wm_config.sequence_length:
                prediction_result = await self._generate_predictions_async(market_data)
                completed_operations.append('predictions')
            
            # Train model if enough data and training is needed
            # NOTE: Training takes longer than budget (seconds vs ms), so don't check remaining() AFTER training
            training_result = {}
            if remaining() > 0 and await self._should_trigger_training_async():
                training_result = await self._train_model_async()
                completed_operations.append('training')
            
            # Generate scenarios if requested or periodically
            scenario_result = {}
            if remaining() > 0 and (inputs.get('generate_scenarios', False) or await self._should_generate_scenarios_async()):
                scenario_result = await self._generate_scenarios_async(market_data)
                completed_operations.append('scenarios')
            
            # Update model performance metrics - skip if budget exceeded but don't fail
            performance_result = {}
            if remaining() > 0:
                performance_result = await self._update_performance_metrics_async(market_data)
                completed_operations.append('performance')
            
            # Update operational mode - skip if budget exceeded but don't fail
            mode_result = {}
            if remaining() > 0:
                mode_result = await self._update_operational_mode_async(market_data)
                completed_operations.append('mode')
            
            # Combine results
            result = {**context_result, **processing_result, **prediction_result,
                     **training_result, **scenario_result, **performance_result, **mode_result}
            
            # Generate thesis
            thesis = await self._generate_world_model_thesis(market_data, result)
            
            # Update SmartInfoBus
            await self._update_world_model_smart_bus(result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)

            # Build contract-compliant return payload (must include all provides)
            predictions_data: Dict[str, Any] = {
                'current_mode': self.current_mode.value,
                'is_trained': self.is_trained,
                'model_confidence': self.model_confidence,
                'prediction_quality': self.prediction_quality,
                'stability_score': self.stability_score,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'device': str(self.device),
                'timestamp': datetime.datetime.now().isoformat()
            }
            if result.get('predictions_generated') and 'predictions' in result:
                _p = result['predictions']
                try:
                    price_changes = _p['price_changes'].tolist() if hasattr(_p['price_changes'], 'tolist') else list(_p['price_changes'])
                    vol_preds = _p['volatility_predictions'].tolist() if hasattr(_p['volatility_predictions'], 'tolist') else list(_p['volatility_predictions'])
                    regime_probs = _p['regime_probabilities'].tolist() if hasattr(_p['regime_probabilities'], 'tolist') else list(_p['regime_probabilities'])
                except Exception:
                    price_changes, vol_preds, regime_probs = [], [], []
                predictions_data['latest_predictions'] = {
                    'price_changes': price_changes,
                    'volatility_predictions': vol_preds,
                    'regime_probabilities': regime_probs,
                    'confidence': _p.get('confidence', 0.0),
                    'predicted_regime': _p.get('predicted_regime', -1),
                    'confidence_level': _p.get('confidence_level', 'very_low'),
                    'timestamp': _p.get('timestamp')
                }

            scenario_data: Dict[str, Any] = {
                'scenarios_available': bool(self.scenario_cache),
                'scenario_count': len(self.scenario_cache.get('scenarios', [])) if self.scenario_cache else 0,
                'scenario_timestamp': self.scenario_cache.get('timestamp') if self.scenario_cache else None,
                'scenario_parameters': self.scenario_cache.get('parameters', {}) if self.scenario_cache else {},
                'scenarios': self.scenario_cache.get('scenarios', []) if self.scenario_cache else []
            }

            analytics_data: Dict[str, Any] = {
                'model_architecture': {
                    'input_size': self.wm_config.input_size,
                    'hidden_size': self.wm_config.hidden_size,
                    'num_layers': self.wm_config.num_layers,
                    'sequence_length': self.wm_config.sequence_length,
                    'attention_heads': self.wm_config.attention_heads
                },
                'performance_metrics': {
                    'training_quality': self.training_quality,
                    'prediction_quality': self.prediction_quality,
                    'stability_score': self.stability_score,
                    'model_confidence': self.model_confidence
                },
                'data_status': {
                    'market_history_size': len(self.market_history),
                    'prediction_history_size': len(self.prediction_history),
                    'training_sessions': len(self.training_history),
                    'feature_importance_count': len(self.feature_importance),
                    'feature_stats_count': len(self.feature_stats)
                },
                'training_curves': {
                    name: list(data)[-20:]
                    for name, data in self.training_curves.items() if data
                },
                'attention_patterns': len(self.attention_patterns),
                'genome_config': self.genome.copy()
            }

            confidence_data: Dict[str, Any] = {
                'current_confidence': self.model_confidence,
                'prediction_confidence': self.prediction_quality,
                'stability_confidence': self.stability_score,
                'training_confidence': self.training_quality,
                'confidence_history': list(self.confidence_history)[-20:] if self.confidence_history else []
            }
            # Enrich classification and recommendations
            try:
                confidence_data['confidence_classification'] = await self._classify_confidence_level_async(float(self.model_confidence))
                confidence_data['is_reliable'] = self.model_confidence > self.wm_config.confidence_threshold
                confidence_data['recommendations'] = await self._generate_confidence_recommendations_async()
            except Exception:
                pass

            return {
                # Preserve useful transient keys for downstreams
                **result,
                # Contract-required provides
                'market_predictions': predictions_data,
                'scenario_generation': scenario_data,
                'world_model_analytics': analytics_data,
                'prediction_confidence': confidence_data,
                # Thesis + status
                '_thesis': thesis,
                'success': True
            }
            
        except Exception as e:
            return await self._handle_world_model_error(e, start_time)

    async def _extract_market_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract comprehensive market data from SmartInfoBus (canonical-first, robust fallbacks)."""
        try:
            # Core snapshot (hard requirement)
            market_data = self.smart_bus.get('market_data', 'EnhancedWorldModel') or {}

            # Canonical risk/context (preferred) with legacy fallbacks
            time_risk       = self.smart_bus.get('time_risk_analysis', 'EnhancedWorldModel') or {}
            market_cond     = self.smart_bus.get('market_conditions', 'EnhancedWorldModel') or {}
            risk_data_legacy = self.smart_bus.get('risk_data', 'EnhancedWorldModel') or {}

            # Optional activity/performance (soft)
            trading_data    = self.smart_bus.get('trading_data', 'EnhancedWorldModel') or {}
            performance_data = (
                self.smart_bus.get('performance_metrics', 'EnhancedWorldModel')  # canonical
                or self.smart_bus.get('performance_data', 'EnhancedWorldModel')  # legacy if present
                or {}
            )

            # Direct inputs (compat)
            market_features = inputs.get('market_features', None)
            prices          = inputs.get('prices', {})
            training_data   = inputs.get('training_data', None)

            # Extract from market_data (prices inside snapshot)
            market_snapshot = market_data.get('market_snapshot', {})
            if not prices and 'prices' in market_snapshot:
                prices = market_snapshot['prices']

            # Snapshots for quick features (prefer canonical)
            risk_snapshot = (
                time_risk.get('risk_snapshot') or
                market_cond.get('risk_snapshot') or
                risk_data_legacy.get('risk_snapshot', {})
            )
            trading_snapshot = trading_data.get('trading_snapshot', {})

            return {
                'market_features': market_features,
                'prices': prices,
                'training_data': training_data,
                'market_data': market_data,
                'time_risk_analysis': time_risk,
                'market_conditions': market_cond,
                'risk_data': risk_data_legacy,     # keep for legacy downstreams
                'trading_data': trading_data,
                'performance_data': performance_data,
                'market_snapshot': market_snapshot,
                'risk_snapshot': risk_snapshot,
                'trading_snapshot': trading_snapshot,
                'timestamp': datetime.datetime.now().isoformat(),
                'step_count': getattr(self, '_step_count', 0)
            }

        except Exception as e:
            self.logger.error(f"Failed to extract market data: {e}")
            return None

    async def _update_market_context_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update market context awareness (canonical-first)."""
        try:
            # Canonical feeds
            regime_data_raw   = self.smart_bus.get('regime_data', 'EnhancedWorldModel')
            market_regime     = self.smart_bus.get('market_regime', 'EnhancedWorldModel')
            vol_adj_raw       = self.smart_bus.get('volatility_adjustment', 'EnhancedWorldModel')
            market_cond_raw   = self.smart_bus.get('market_conditions', 'EnhancedWorldModel')

            # Defensive normalization: some upstream modules may still publish scalar regime/volatility values
            regime_data = regime_data_raw if isinstance(regime_data_raw, dict) else {}
            # Accept both dict and scalar (float) for volatility_adjustment; normalize to dict
            if isinstance(vol_adj_raw, dict):
                vol_adj = vol_adj_raw
            elif isinstance(vol_adj_raw, (int, float)):
                vol_adj = {"scale": float(vol_adj_raw)}
                # Avoid spamming logs every tick; log once if seen scalar form
                if not getattr(self, "_vol_adj_scalar_logged", False):
                    try:
                        self.logger.debug(
                            f"[context_guard] volatility_adjustment scalar ({vol_adj_raw}); normalized -> {{'scale': {float(vol_adj_raw):.3f}}}"
                        )
                    except Exception:
                        pass
                    setattr(self, "_vol_adj_scalar_logged", True)
            else:
                vol_adj = {}
                if vol_adj_raw is not None:
                    self.logger.debug(
                        f"[context_guard] volatility_adjustment non-dict ({type(vol_adj_raw).__name__}); coerced -> {{}}"
                    )
            market_cond = market_cond_raw if isinstance(market_cond_raw, dict) else {}
            if regime_data_raw is not None and not isinstance(regime_data_raw, dict):
                self.logger.debug(f"[context_guard] regime_data non-dict ({type(regime_data_raw).__name__}); coerced -> {{}}")
            if market_cond_raw is not None and not isinstance(market_cond_raw, dict):
                self.logger.debug(f"[context_guard] market_conditions non-dict ({type(market_cond_raw).__name__}); coerced -> {{}}")

            # Legacy/compat (guard against non-dict writes)
            market_context_raw = self.smart_bus.get('market_context', 'EnhancedWorldModel')
            market_context = market_context_raw if isinstance(market_context_raw, dict) else {}
            if market_context_raw is not None and not isinstance(market_context_raw, dict):
                self.logger.debug(f"[context_guard] market_context non-dict ({type(market_context_raw).__name__}); coerced -> {{}}")

            # Resolve regime/session/volatility with fallbacks
            regime = (
                market_regime
                or regime_data.get('market_regime')
                or market_context.get('regime', 'unknown')
            )
            session = (
                market_cond.get('session')
                or market_context.get('session', 'unknown')
            )
            volatility_level = (
                str(vol_adj.get('volatility_regime')) if vol_adj.get('volatility_regime') is not None
                else market_context.get('volatility_level', 'medium')
            )
            stress_level = (
                market_cond.get('stress_level')
                or market_context.get('stress_level', 0.0)
            )

            # Update feature importance based on resolved context
            await self._update_context_feature_importance_async(regime, session, volatility_level)

            return {
                'market_context_updated': True,
                'regime': regime,
                'session': session,
                'volatility_level': volatility_level,
                'stress_level': stress_level
            }

        except Exception as e:
            self.logger.error(f"Market context update failed: {e}")
            return {'market_context_updated': False, 'error': str(e)}


    async def _update_context_feature_importance_async(self, regime: str, session: str, volatility_level: str):
        """Update feature importance based on market context"""
        try:
            context_key = f"{regime}_{session}_{volatility_level}"
            
            if context_key not in self.feature_importance:
                self.feature_importance[context_key] = {
                    'price_weight': 1.0,
                    'volume_weight': 1.0,
                    'volatility_weight': 1.0,
                    'trend_weight': 1.0,
                    'session_weight': 1.0,
                    'update_count': 0
                }
            
            # Adapt weights based on context
            importance = self.feature_importance[context_key]
            
            if volatility_level == 'high':
                importance['volatility_weight'] *= 1.1
                importance['price_weight'] *= 0.9
            elif volatility_level == 'low':
                importance['trend_weight'] *= 1.1
                importance['volatility_weight'] *= 0.9
            
            if regime == 'trending':
                importance['trend_weight'] *= 1.2
            elif regime == 'volatile':
                importance['volatility_weight'] *= 1.2
            
            importance['update_count'] += 1
            
        except Exception as e:
            self.logger.warning(f"Feature importance update failed: {e}")

    async def _process_market_data_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process market data and update history"""
        try:
            # Extract features
            features = await self._extract_comprehensive_features_async(market_data)
            
            if features is None:
                return {'market_data_processed': False, 'reason': 'feature_extraction_failed'}
            
            # Create market state record
            market_state = {
                'timestamp': market_data.get('timestamp'),
                'features': features,
                'prices': market_data.get('prices', {}),
                'market_snapshot': market_data.get('market_snapshot', {}),
                'risk_snapshot': market_data.get('risk_snapshot', {}),
                'trading_snapshot': market_data.get('trading_snapshot', {}),
                'step_count': market_data.get('step_count', 0)
            }
            
            # Add to history
            self.market_history.append(market_state)
            self.feature_history.append(features)
            
            # Update feature statistics
            await self._update_feature_statistics_async(features)
            
            return {
                'market_data_processed': True,
                'features_extracted': len(features),
                'history_size': len(self.market_history),
                'feature_quality': await self._assess_feature_quality_async(features)
            }
            
        except Exception as e:
            self.logger.error(f"Market data processing failed: {e}")
            return {'market_data_processed': False, 'error': str(e)}

    async def _extract_comprehensive_features_async(self, market_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract comprehensive features from market data"""
        try:
            features: List[float] = []
            
            # Price features (4 instruments)
            prices = market_data.get('prices', {})
            if prices:
                price_values = list(prices.values())[:4]
                # Normalize prices
                for price in price_values:
                    features.append(price / 2000.0)
                
                # Fill missing prices
                while len(price_values) < 4:
                    features.append(1.0)
                    price_values.append(2000.0)
                
                # Price changes if we have history
                if len(self.market_history) > 0:
                    prev_prices = list(self.market_history[-1]['prices'].values())[:4]
                    for i, (curr, prev) in enumerate(zip(price_values, prev_prices)):
                        if prev > 0:
                            features.append((curr - prev) / prev)
                        else:
                            features.append(0.0)
                else:
                    features.extend([0.0] * 4)
            else:
                # Default price features
                features.extend([1.0] * 4 + [0.0] * 4)  # 8 price features
            
            # Market regime features (4 classes)
            market_context = self.smart_bus.get('market_context', 'EnhancedWorldModel') or {}
            regime = market_context.get('regime', 'unknown')
            regime_encoding = {
                'trending': [1, 0, 0, 0],
                'volatile': [0, 1, 0, 0],
                'ranging': [0, 0, 1, 0],
                'unknown': [0, 0, 0, 1]
            }
            features.extend(regime_encoding.get(regime, [0.25] * 4))
            
            # Volatility features
            vol_level = market_context.get('volatility_level', 'medium')
            vol_encoding = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}
            features.append(vol_encoding.get(vol_level, 0.5))
            
            # Risk features
            risk_snapshot = market_data.get('risk_snapshot', {})
            features.extend([
                risk_snapshot.get('drawdown_pct', 0.0) / 100.0,
                risk_snapshot.get('exposure_pct', 0.0) / 100.0,
                min(1.0, risk_snapshot.get('position_count', 0) / 10.0)
            ])
            
            # Session features
            session = market_context.get('session', 'unknown')
            session_encoding = {
                'asian': [1, 0, 0, 0],
                'european': [0, 1, 0, 0],
                'american': [0, 0, 1, 0],
                'closed': [0, 0, 0, 1]
            }
            features.extend(session_encoding.get(session, [0.25] * 4))
            
            # Extend or truncate to target input size
            while len(features) < self.wm_config.input_size:
                features.append(0.0)
            
            return np.array(features[:self.wm_config.input_size], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return None

    async def _update_feature_statistics_async(self, features: np.ndarray):
        """Update feature statistics for monitoring"""
        try:
            feature_names = [
                f'price_{i}' for i in range(4)
            ] + [
                f'price_change_{i}' for i in range(4)
            ] + [
                'regime_trending', 'regime_volatile', 'regime_ranging', 'regime_unknown',
                'volatility_level', 'drawdown_pct', 'exposure_pct', 'position_count'
            ] + [
                'session_asian', 'session_european', 'session_american', 'session_closed'
            ]
            
            for i, (name, value) in enumerate(zip(feature_names[:len(features)], features)):
                if name not in self.feature_stats:
                    self.feature_stats[name] = {
                        'values': deque(maxlen=100),
                        'variance': 0.0,
                        'importance_score': 0.0
                    }
                
                self.feature_stats[name]['values'].append(value)
                
                # Update variance
                values = list(self.feature_stats[name]['values'])
                if len(values) > 1:
                    self.feature_stats[name]['variance'] = np.var(values)
                    self.feature_stats[name]['importance_score'] = min(1.0, self.feature_stats[name]['variance'] * 10)
            
        except Exception as e:
            self.logger.warning(f"Feature statistics update failed: {e}")

    async def _assess_feature_quality_async(self, features: np.ndarray) -> float:
        """Assess quality of extracted features"""
        try:
            # Check for invalid values
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                return 0.0
            
            # Check for reasonable value ranges
            out_of_range = np.sum((features < -10) | (features > 10))
            range_quality = 1.0 - (out_of_range / len(features))
            
            # Check for feature diversity
            diversity = 1.0 - np.exp(-np.var(features))
            
            # Combined quality score
            quality = (range_quality + diversity) / 2.0
            
            return float(np.clip(quality, 0.0, 1.0))
            
        except Exception:
            return 0.5

    async def _generate_predictions_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate model predictions with enhanced analytics"""
        try:
            if len(self.market_history) < self.wm_config.sequence_length:
                return {'predictions_generated': False, 'reason': 'insufficient_history'}

            if not self._predict_lock.acquire(blocking=False):
                return {'predictions_generated': False, 'reason': 'prediction_in_progress'}
            
            try:
                # Prepare input sequence
                sequence_features = [state['features'] for state in list(self.market_history)[-self.wm_config.sequence_length:]]
                X = np.vstack(sequence_features)
                X_tensor = torch.from_numpy(X).unsqueeze(0).to(self.device)
                
                # Extract context
                context_features = await self._encode_context_features_async(market_data)
                context_tensor = torch.from_numpy(context_features).unsqueeze(0).to(self.device)
                
                self.eval()
                with torch.no_grad():
                    # Use new autocast API when on CUDA; disabled on CPU
                    autocast_ctx = (
                        torch.amp.autocast(device_type="cuda", enabled=True)  # type: ignore[call-arg]
                        if (self._amp_enabled and hasattr(torch, "amp") and hasattr(torch.amp, "autocast"))
                        else contextlib.nullcontext()
                    )
                    with autocast_ctx:
                        # LSTM forward pass
                        lstm_out, (hidden, cell) = self.lstm(X_tensor)
                        
                        # Apply attention mechanism
                        attended_out, attention_weights = self.attention(
                            lstm_out, lstm_out, lstm_out
                        )
                        
                        # Use last attended output
                        last_attended = attended_out[:, -1, :]
                        
                        # Integrate context
                        context_encoded = self.context_encoder(context_tensor)
                        fused_features = self.fusion_layer(
                            torch.cat([last_attended, context_encoded], dim=1)
                        )
                        
                        # Generate predictions
                        price_pred = self.price_head(fused_features)
                        vol_pred = self.volatility_head(fused_features)
                        regime_pred = self.regime_head(fused_features)
                        confidence_pred = self.confidence_head(fused_features)
                    
                    # Calculate prediction confidence
                    prediction_confidence = float(confidence_pred.detach().cpu().numpy()[0, 0])
                    attention_confidence = await self._calculate_attention_confidence_async(attention_weights)
                    combined_confidence = (prediction_confidence + attention_confidence) / 2.0
                    
                    predictions = {
                        'price_changes': price_pred.detach().cpu().numpy()[0],
                        'volatility_predictions': vol_pred.detach().cpu().numpy()[0],
                        'regime_probabilities': regime_pred.detach().cpu().numpy()[0],
                        'confidence': combined_confidence,
                        'prediction_confidence': prediction_confidence,
                        'attention_confidence': attention_confidence,
                        'attention_weights': attention_weights.detach().cpu().numpy()[0],
                        'timestamp': market_data.get('timestamp'),
                        'predicted_regime': int(torch.argmax(regime_pred, dim=1).detach().cpu().numpy()[0]),
                        'confidence_level': await self._classify_confidence_level_async(combined_confidence)
                    }
                
                # Store prediction and confidence history
                self.prediction_history.append(predictions.copy())
                self.confidence_history.append(combined_confidence)  # FIX: was never updated
                
                # Track prediction performance
                await self._track_prediction_performance_async(predictions, market_data)
                
                # Store attention patterns
                self.attention_patterns.append(attention_weights.detach().cpu().numpy()[0])
                
                self.logger.info(format_operator_message(
                    message="Predictions generated",
                    icon="🔮",
                    confidence=f"{combined_confidence:.3f}",
                    regime=f"{predictions['predicted_regime']}",
                    price_trend=f"{np.mean(predictions['price_changes']):.4f}",
                    vol_avg=f"{np.mean(predictions['volatility_predictions']):.4f}"
                ))
                
                return {
                    'predictions_generated': True,
                    'predictions': predictions,
                    'sequence_length': self.wm_config.sequence_length,
                    'model_confidence': self.model_confidence
                }
            finally:
                self._predict_lock.release()
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            return {'predictions_generated': False, 'error': str(e)}

    async def _encode_context_features_async(self, market_data: Dict[str, Any]) -> np.ndarray:
        """Encode context into fixed-size (16) vector; canonical-first with legacy fallbacks."""
        try:
            feats: List[float] = []

            # Canonical context
            market_regime = self.smart_bus.get('market_regime', 'EnhancedWorldModel')
            regime_data   = self.smart_bus.get('regime_data', 'EnhancedWorldModel') or {}
            vol_adj       = self.smart_bus.get('volatility_adjustment', 'EnhancedWorldModel') or {}
            market_cond   = self.smart_bus.get('market_conditions', 'EnhancedWorldModel') or {}

            # Guard against scalar / non-dict values coming from the bus
            if not isinstance(regime_data, dict):
                regime_data = {}
            if not isinstance(vol_adj, dict):
                # Some pipelines publish volatility_adjustment as a scalar (e.g., 1.0);
                # treat that as "no structured volatility regime info" for encoding.
                vol_adj = {}
            if not isinstance(market_cond, dict):
                market_cond = {}

            # Legacy
            market_context = self.smart_bus.get('market_context', 'EnhancedWorldModel') or {}
            if not isinstance(market_context, dict):
                market_context = {}

            # Regime scalar (ordinal proxy)
            regime_val_map = {'trending': 1.0, 'volatile': 0.75, 'ranging': 0.5, 'unknown': 0.25}
            regime = market_regime or regime_data.get('market_regime') or market_context.get('regime', 'unknown')
            feats.append(regime_val_map.get(str(regime), 0.25))

            # Session scalar
            session_val_map = {'asian': 0.25, 'european': 0.5, 'american': 0.75, 'closed': 0.0}
            session = market_cond.get('session') or market_context.get('session', 'unknown')
            feats.append(session_val_map.get(str(session), 0.0))

            # Volatility scalar
            vol_val_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'extreme': 1.0}
            vol_regime = vol_adj.get('volatility_regime') or market_context.get('volatility_level', 'medium')
            feats.append(vol_val_map.get(str(vol_regime), 0.5))

            # Risk snapshot (prefer canonical analysis/conditions)
            risk_snapshot = (
                market_data.get('risk_snapshot')
                or market_cond.get('risk_snapshot', {})
            )
            if not isinstance(risk_snapshot, dict):
                risk_snapshot = {}
            feats.extend([
                risk_snapshot.get('drawdown_pct', 0.0) / 100.0,
                risk_snapshot.get('exposure_pct', 0.0) / 100.0,
                min(1.0, risk_snapshot.get('position_count', 0) / 20.0),
                risk_snapshot.get('risk_score', 0.0) / 100.0
            ])

            # Stress/correlation/liquidity (prefer canonical conditions)
            feats.extend([
                market_cond.get('stress_level', market_context.get('stress_level', 0.0)),
                market_cond.get('correlation_risk', market_context.get('correlation_risk', 0.0)),
                market_cond.get('liquidity_score', market_context.get('liquidity_score', 1.0))
            ])

            # Trading snapshot (optional)
            trading_snapshot = market_data.get('trading_snapshot', {})
            if not isinstance(trading_snapshot, dict):
                trading_snapshot = {}
            feats.extend([
                min(1.0, trading_snapshot.get('trade_count', 0) / 100.0),
                trading_snapshot.get('avg_trade_size', 0.0) / 1000.0,
                trading_snapshot.get('direction_bias', 0.0)
            ])

            # Performance (prefer canonical performance_metrics)
            perf = market_data.get('performance_data', {})
            if not isinstance(perf, dict):
                perf = {}
            feats.extend([
                perf.get('recent_pnl', 0.0) / 1000.0,
                perf.get('win_rate', 0.5),
                perf.get('sharpe_ratio', 0.0) / 3.0
            ])

            # Pad/trim to 16
            while len(feats) < 16:
                feats.append(0.0)
            return np.array(feats[:16], dtype=np.float32)

        except Exception as e:
            self.logger.warning(f"Context encoding failed: {e}")
            return np.zeros(16, dtype=np.float32)


    async def _calculate_attention_confidence_async(self, attention_weights: torch.Tensor) -> float:
        """Calculate confidence based on attention distribution"""
        try:
            attention_probs = attention_weights.squeeze().detach().cpu().numpy()
            
            if len(attention_probs.shape) > 1:
                attention_probs = attention_probs.mean(axis=0)
            
            if len(attention_probs) > 1:
                # Higher entropy = lower confidence (attention is spread out)
                entropy = -np.sum(attention_probs * np.log(attention_probs + 1e-8))
                max_entropy = np.log(len(attention_probs))
                confidence = 1.0 - (entropy / max_entropy)
            else:
                confidence = 0.5
            
            return float(np.clip(confidence, 0.0, 1.0))
            
        except Exception:
            return 0.5

    async def _classify_confidence_level_async(self, confidence: float) -> str:
        """Classify confidence into levels"""
        if confidence >= 0.9:
            return PredictionConfidence.VERY_HIGH.value
        elif confidence >= 0.75:
            return PredictionConfidence.HIGH.value
        elif confidence >= 0.5:
            return PredictionConfidence.MEDIUM.value
        elif confidence >= 0.25:
            return PredictionConfidence.LOW.value
        else:
            return PredictionConfidence.VERY_LOW.value

    async def _track_prediction_performance_async(self, predictions: Dict[str, Any], market_data: Dict[str, Any]):
        """Track prediction accuracy for performance monitoring"""
        try:
            if len(self.prediction_history) >= 2:
                # Get previous prediction
                prev_prediction = self.prediction_history[-2]
                current_prices = market_data.get('prices', {})
                
                if current_prices and len(self.market_history) >= 2:
                    prev_market_state = self.market_history[-2]
                    prev_prices = prev_market_state.get('prices', {})
                    
                    # Calculate actual price changes
                    actual_changes = []
                    predicted_changes = prev_prediction['price_changes']
                    
                    for i, symbol in enumerate(list(current_prices.keys())[:4]):
                        if symbol in prev_prices and prev_prices[symbol] > 0:
                            actual_change = (current_prices[symbol] - prev_prices[symbol]) / prev_prices[symbol]
                            actual_changes.append(actual_change)
                    
                    if len(actual_changes) >= 2:
                        # Calculate prediction errors
                        errors = []
                        for i in range(min(len(actual_changes), len(predicted_changes))):
                            error = abs(actual_changes[i] - predicted_changes[i])
                            errors.append(error)
                        
                        if errors:
                            avg_error = np.mean(errors)
                            self.prediction_errors.append(avg_error)
                            
                            # Update prediction quality
                            if len(self.prediction_errors) >= 20:
                                recent_errors = list(self.prediction_errors)[-20:]
                                avg_recent_error = np.mean(recent_errors)
                                self.prediction_quality = max(0.0, float(1.0 - avg_recent_error * 20))
            
        except Exception as e:
            self.logger.warning(f"Prediction performance tracking failed: {e}")

    async def _should_trigger_training_async(self) -> bool:
        """Determine if model training should be triggered
        
        PERFORMANCE FIX: Reduced training frequency to prevent blocking pipeline
        - Was: every 200 samples (too frequent, causing 3-16 second delays)
        - Now: every 1000 samples or when quality drops significantly
        
        LIVE MODE: Training disabled by default in live trading to prevent
        unexpected model changes during real trades.
        """
        try:
            # LIVE MODE GUARD: Respect enable_online_training config
            if self.wm_config.live_mode and not self.wm_config.enable_online_training:
                return False
            
            # Check if we have enough data
            if len(self.market_history) < self.wm_config.min_training_samples:
                return False
            
            # Check if model needs initial training
            if not self.is_trained:
                return True
            
            # Check prediction quality - only retrain if quality is VERY low
            if self.prediction_quality < self.wm_config.min_prediction_quality * 0.7:  # 30% below threshold
                return True
            
            # Check if enough time has passed since last training (reduced from 1 hour to 30 min)
            if self.last_training_time:
                time_since_training = datetime.datetime.now() - self.last_training_time
                # Only train if quality is also below threshold
                if time_since_training.total_seconds() > 1800 and self.prediction_quality < self.wm_config.min_prediction_quality:
                    return True
            
            # PERFORMANCE FIX: Increased from 200 to 1000 to reduce training frequency
            # This prevents the 3-16 second training delays every 200 steps
            if len(self.market_history) % 1000 == 0:
                return True
            
            return False
            
        except Exception:
            return False

    async def _train_model_async(self) -> Dict[str, Any]:
        """Train model asynchronously with enhanced monitoring"""
        if not self._train_lock.acquire(blocking=False):
            return {'training_completed': False, 'reason': 'training_in_progress'}

        try:
            self.current_mode = WorldModelMode.TRAINING
            
            # Prepare training data
            train_data = await self._prepare_training_data_async()
            
            if not train_data or len(train_data['X']) == 0:
                return {'training_completed': False, 'reason': 'insufficient_data'}
            
            # Split data
            split_idx = int(len(train_data['X']) * (1 - self.wm_config.validation_split))
            
            train_X = train_data['X'][:split_idx]
            val_X = train_data['X'][split_idx:]
            train_y_price = train_data['y_price'][:split_idx]
            val_y_price = train_data['y_price'][split_idx:]
            train_y_vol = train_data['y_vol'][:split_idx]
            val_y_vol = train_data['y_vol'][split_idx:]
            train_y_regime = train_data['y_regime'][:split_idx]
            val_y_regime = train_data['y_regime'][split_idx:]
            
            # Create data loaders
            train_dataset = TensorDataset(
                torch.stack(train_X),
                torch.stack(train_y_price),
                torch.stack(train_y_vol),
                torch.stack(train_y_regime)
            )
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.wm_config.batch_size, 
                shuffle=True,
                num_workers=0,
                pin_memory=True if self.device.type == 'cuda' else False
            )
            
            # Training parameters - PERFORMANCE FIX: Reduced max epochs from 20 to 5
            # This reduces training time from 3-16 seconds to ~1-3 seconds
            epochs = min(5, max(3, len(train_X) // 100))  # 3-5 epochs max
            best_val_loss = float('inf')
            patience_counter = 0
            patience = 2  # Reduced patience for faster early stopping
            
            self.train()
            training_losses: List[float] = []
            validation_losses: List[float] = []
            epochs_completed = 0
            
            for epoch in range(epochs):
                epochs_completed = epoch + 1
                # Training phase
                epoch_train_loss = 0.0
                num_batches = 0
                
                for batch_X, batch_y_price, batch_y_vol, batch_y_regime in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y_price = batch_y_price.to(self.device)
                    batch_y_vol = batch_y_vol.to(self.device)
                    batch_y_regime = batch_y_regime.to(self.device)

                    self.optimizer.zero_grad(set_to_none=True)

                    if self._amp_enabled:
                        # Use new autocast API if available
                        autocast_ctx = (
                            torch.amp.autocast(device_type="cuda", enabled=True)  # type: ignore[call-arg]
                            if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                            else contextlib.nullcontext()
                        )
                        with autocast_ctx:
                            lstm_out, _ = self.lstm(batch_X)
                            attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                            hidden_state = attended_out[:, -1, :]
                            
                            price_pred = self.price_head(hidden_state)
                            vol_pred = self.volatility_head(hidden_state)
                            regime_pred = self.regime_head(hidden_state)
                            
                            price_loss = self.huber_loss(price_pred, batch_y_price)
                            vol_loss = self.mse_loss(vol_pred, batch_y_vol)
                            regime_loss = self.ce_loss(regime_pred, batch_y_regime.long())
                            total_loss = 0.5 * price_loss + 0.3 * vol_loss + 0.2 * regime_loss
                        if self._scaler is not None:
                            self._scaler.scale(total_loss).backward()
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.wm_config.gradient_clip)
                            self._scaler.step(self.optimizer)
                            self._scaler.update()
                        else:
                            total_loss.backward()
                            torch.nn.utils.clip_grad_norm_(self.parameters(), self.wm_config.gradient_clip)
                            self.optimizer.step()
                    else:
                        lstm_out, _ = self.lstm(batch_X)
                        attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                        hidden_state = attended_out[:, -1, :]
                        
                        price_pred = self.price_head(hidden_state)
                        vol_pred = self.volatility_head(hidden_state)
                        regime_pred = self.regime_head(hidden_state)
                        
                        price_loss = self.huber_loss(price_pred, batch_y_price)
                        vol_loss = self.mse_loss(vol_pred, batch_y_vol)
                        regime_loss = self.ce_loss(regime_pred, batch_y_regime.long())
                        total_loss = 0.5 * price_loss + 0.3 * vol_loss + 0.2 * regime_loss
                        
                        total_loss.backward()
                        grad_norm = torch.nn.utils.clip_grad_norm_((self.parameters()), self.wm_config.gradient_clip)
                        self.optimizer.step()
                        self.training_curves['gradient_norm'].append(float(grad_norm))

                    epoch_train_loss += float(total_loss.detach().cpu())
                    num_batches += 1
                
                avg_train_loss = epoch_train_loss / num_batches if num_batches > 0 else float('inf')
                training_losses.append(avg_train_loss)
                
                # Validation phase
                val_loss = await self._validate_model_async(val_X, val_y_price, val_y_vol, val_y_regime)
                validation_losses.append(val_loss)
                
                # Learning rate scheduling
                self.scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                # Update training curves
                self.training_curves['loss'].append(avg_train_loss)
                self.training_curves['val_loss'].append(val_loss)
                self.training_curves['accuracy'].append(1.0 / (1.0 + val_loss))
                
                self.logger.info(format_operator_message(
                    message=f"Training epoch {epochs_completed}/{epochs}",
                    icon="[TARGET]",
                    train_loss=f"{avg_train_loss:.6f}",
                    val_loss=f"{val_loss:.6f}",
                    best_val=f"{best_val_loss:.6f}",
                    lr=f"{self.optimizer.param_groups[0]['lr']:.1e}"
                ))
                
                if patience_counter >= patience:
                    self.logger.info("Early stopping triggered")
                    break
            
            # Update model status
            self.is_trained = True
            self.last_training_time = datetime.datetime.now()
            self.training_quality = max(0.1, 1.0 / (1.0 + best_val_loss))
            self.model_confidence = self.training_quality
            
            # Record training session
            training_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'epochs_completed': epochs_completed,
                'final_train_loss': training_losses[-1] if training_losses else float('inf'),
                'best_val_loss': best_val_loss,
                'training_samples': len(train_X),
                'validation_samples': len(val_X),
                'training_quality': self.training_quality,
                'model_confidence': self.model_confidence
            }
            self.training_history.append(training_record)
            
            self.logger.info(format_operator_message(
                message="Model training completed",
                icon="[OK]",
                epochs=epochs_completed,
                final_loss=f"{training_losses[-1]:.6f}",
                val_loss=f"{best_val_loss:.6f}",
                quality=f"{self.training_quality:.3f}",
                samples=len(train_X)
            ))
            
            self.current_mode = WorldModelMode.ACTIVE_PREDICTION
            
            return {
                'training_completed': True,
                'epochs_completed': epochs_completed,
                'final_train_loss': training_losses[-1] if training_losses else float('inf'),
                'best_val_loss': best_val_loss,
                'training_quality': self.training_quality,
                'model_confidence': self.model_confidence,
                'training_samples': len(train_X)
            }
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            self.current_mode = WorldModelMode.ERROR_RECOVERY
            return {'training_completed': False, 'error': str(e)}
        finally:
            self._train_lock.release()

    async def _prepare_training_data_async(self) -> Dict[str, List]:
        """Prepare training data from market history"""
        try:
            X, y_price, y_vol, y_regime = [], [], [], []
            
            history_list = list(self.market_history)
            
            for i in range(self.wm_config.sequence_length, len(history_list)):
                # Input sequence
                sequence = [state['features'] for state in history_list[i-self.wm_config.sequence_length:i]]
                X.append(torch.from_numpy(np.vstack(sequence)).float())
                
                # Target values
                current_state = history_list[i]
                prev_state = history_list[i-1]
                
                # Price changes (4 instruments)
                current_prices = current_state.get('prices', {})
                prev_prices = prev_state.get('prices', {})
                
                price_changes = []
                for symbol in list(current_prices.keys())[:4]:
                    if symbol in prev_prices and prev_prices[symbol] > 0:
                        change = (current_prices[symbol] - prev_prices[symbol]) / prev_prices[symbol]
                        price_changes.append(change)
                    else:
                        price_changes.append(0.0)
                
                # Ensure 4 price changes
                while len(price_changes) < 4:
                    price_changes.append(0.0)
                
                y_price.append(torch.tensor(price_changes[:4], dtype=torch.float32))
                
                # Volatility estimates (simplified)
                market_snapshot = current_state.get('market_snapshot', {})
                vol_level = market_snapshot.get('volatility_level', 'medium')
                vol_values = {
                    'low': [0.2, 0.2, 0.2, 0.2],
                    'medium': [0.5, 0.5, 0.5, 0.5],
                    'high': [0.8, 0.8, 0.8, 0.8],
                    'extreme': [1.0, 1.0, 1.0, 1.0]
                }
                y_vol.append(torch.tensor(vol_values.get(vol_level, [0.5, 0.5, 0.5, 0.5]), dtype=torch.float32))
                
                # Regime classification
                regime = market_snapshot.get('regime', 'unknown')
                regime_idx = {'trending': 0, 'volatile': 1, 'ranging': 2, 'unknown': 3}
                y_regime.append(torch.tensor(regime_idx.get(regime, 3), dtype=torch.long))
            
            return {
                'X': X,
                'y_price': y_price,
                'y_vol': y_vol,
                'y_regime': y_regime
            }
            
        except Exception as e:
            self.logger.error(f"Training data preparation failed: {e}")
            return {'X': [], 'y_price': [], 'y_vol': [], 'y_regime': []}

    async def _validate_model_async(self, X_val: List[torch.Tensor], y_price_val: List[torch.Tensor],
                                  y_vol_val: List[torch.Tensor], y_regime_val: List[torch.Tensor]) -> float:
        """Validate model performance asynchronously"""
        try:
            if not X_val:
                return float('inf')
            
            self.eval()
            total_loss = 0.0
            num_samples = 0
            
            with torch.no_grad():
                for i in range(min(len(X_val), 50)):  # Limit validation samples for speed
                    X_batch = X_val[i].unsqueeze(0).to(self.device)
                    y_price_batch = y_price_val[i].unsqueeze(0).to(self.device)
                    y_vol_batch = y_vol_val[i].unsqueeze(0).to(self.device)
                    y_regime_batch = y_regime_val[i].unsqueeze(0).to(self.device)
                    
                    # Forward pass
                    lstm_out, _ = self.lstm(X_batch)
                    attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                    hidden_state = attended_out[:, -1, :]
                    
                    # Predictions
                    price_pred = self.price_head(hidden_state)
                    vol_pred = self.volatility_head(hidden_state)
                    regime_pred = self.regime_head(hidden_state)
                    
                    # Calculate losses
                    price_loss = self.huber_loss(price_pred, y_price_batch)
                    vol_loss = self.mse_loss(vol_pred, y_vol_batch)
                    regime_loss = self.ce_loss(regime_pred, y_regime_batch.long())
                    
                    total_loss += (0.5 * price_loss + 0.3 * vol_loss + 0.2 * regime_loss).item()
                    num_samples += 1
            
            return total_loss / num_samples if num_samples > 0 else float('inf')
            
        except Exception as e:
            self.logger.warning(f"Model validation failed: {e}")
            return float('inf')

    async def _should_generate_scenarios_async(self) -> bool:
        """Determine if scenarios should be generated
        
        LIVE MODE: Scenario generation disabled by default in live trading
        to reduce computational overhead during real trades.
        """
        try:
            # LIVE MODE GUARD: Respect enable_scenario_generation config
            if self.wm_config.live_mode and not self.wm_config.enable_scenario_generation:
                return False
            
            # Generate scenarios periodically
            if not hasattr(self, '_last_scenario_time'):
                self._last_scenario_time = datetime.datetime.now()
                return True
            
            time_since_scenarios = datetime.datetime.now() - self._last_scenario_time
            if time_since_scenarios.total_seconds() > 1800:  # 30 minutes
                return True
            
            # Generate if model confidence is high
            if self.model_confidence > 0.8:
                return True
            
            return False
            
        except Exception:
            return False

    async def _generate_scenarios_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market scenarios asynchronously"""
        try:
            if not self.is_trained or len(self.market_history) < self.wm_config.sequence_length:
                return {'scenarios_generated': False, 'reason': 'model_not_ready'}
            
            self.current_mode = WorldModelMode.SCENARIO_GENERATION
            
            num_scenarios = 5
            scenarios = []
            
            for scenario_id in range(num_scenarios):
                scenario = await self._generate_single_scenario_async(
                    self.wm_config.scenario_steps, scenario_id, num_scenarios, market_data
                )
                scenarios.append(scenario)
            
            # Cache scenarios
            self.scenario_cache = {
                'timestamp': datetime.datetime.now().isoformat(),
                'scenarios': scenarios,
                'parameters': {
                    'steps': self.wm_config.scenario_steps,
                    'num_scenarios': num_scenarios,
                    'model_confidence': self.model_confidence
                }
            }
            
            # Calculate scenario metrics
            scenario_metrics = await self._analyze_scenarios_async(scenarios)
            
            self._last_scenario_time = datetime.datetime.now()
            
            self.logger.info(format_operator_message(
                message=f"Generated {num_scenarios} scenarios",
                icon="🎭",
                steps=self.wm_config.scenario_steps,
                diversity=f"{scenario_metrics['diversity']:.3f}",
                avg_confidence=f"{scenario_metrics['avg_confidence']:.3f}"
            ))
            
            return {
                'scenarios_generated': True,
                'scenarios': scenarios,
                'scenario_metrics': scenario_metrics,
                'cache_updated': True
            }
            
        except Exception as e:
            self.logger.error(f"Scenario generation failed: {e}")
            return {'scenarios_generated': False, 'error': str(e)}

    async def _generate_single_scenario_async(self, steps: int, scenario_id: int, 
                                            num_scenarios: int, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single market scenario asynchronously"""
        try:
            # Start with current market state
            current_sequence = [state['features'] for state in list(self.market_history)[-self.wm_config.sequence_length:]]
            scenario_path = []
            
            # Add scenario-specific noise for diversity
            noise_scale = 0.1 * (scenario_id + 1) / num_scenarios
            
            self.eval()
            with torch.no_grad():
                for step in range(steps):
                    # Prepare input
                    X = torch.from_numpy(np.vstack(current_sequence)).unsqueeze(0).float().to(self.device)
                    
                    # Generate prediction
                    lstm_out, _ = self.lstm(X)
                    attended_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
                    hidden_state = attended_out[:, -1, :]
                    
                    price_pred = self.price_head(hidden_state)
                    vol_pred = self.volatility_head(hidden_state)
                    regime_pred = self.regime_head(hidden_state)
                    confidence_pred = self.confidence_head(hidden_state)
                    
                    # Add scenario-specific noise
                    price_pred = price_pred + torch.randn_like(price_pred) * noise_scale
                    vol_pred = vol_pred + torch.randn_like(vol_pred) * noise_scale * 0.5
                    
                    # Create step prediction
                    step_prediction = {
                        'step': step,
                        'price_changes': price_pred.detach().cpu().numpy()[0],
                        'volatility_predictions': vol_pred.detach().cpu().numpy()[0],
                        'regime_probabilities': regime_pred.detach().cpu().numpy()[0],
                        'confidence': float(confidence_pred.detach().cpu().numpy()[0, 0]),
                        'predicted_regime': int(torch.argmax(regime_pred, dim=1).detach().cpu().numpy()[0])
                    }
                    
                    scenario_path.append(step_prediction)
                    
                    # Update sequence for next prediction
                    new_features = await self._prediction_to_features_async(step_prediction, current_sequence[-1])
                    current_sequence = current_sequence[1:] + [new_features]
            
            # Calculate scenario summary
            scenario_summary = await self._summarize_scenario_async(scenario_path)
            
            return {
                'scenario_id': scenario_id,
                'steps': steps,
                'noise_scale': noise_scale,
                'path': scenario_path,
                'summary': scenario_summary
            }
            
        except Exception as e:
            self.logger.warning(f"Single scenario generation failed: {e}")
            return {'scenario_id': scenario_id, 'error': str(e)}

    async def _prediction_to_features_async(self, prediction: Dict[str, Any], prev_features: np.ndarray) -> np.ndarray:
        """Convert prediction back to feature representation"""
        try:
            new_features = prev_features.copy()
            
            # Update price features (first 4 elements for 4 instruments)
            price_changes = prediction['price_changes']
            for i in range(min(4, len(price_changes))):
                # Apply price changes to previous prices
                new_features[i] *= (1 + price_changes[i])
                # Update price change features
                if i + 4 < len(new_features):
                    new_features[i + 4] = price_changes[i]
            
            # Update regime features
            regime_probs = prediction['regime_probabilities']
            if len(regime_probs) >= 4:
                # Assuming regime features are at positions 8-11
                start_idx = 8
                for i in range(4):
                    if start_idx + i < len(new_features):
                        new_features[start_idx + i] = regime_probs[i]
            
            # Update volatility feature
            volatility_preds = prediction['volatility_predictions']
            if len(volatility_preds) >= 1:
                # Assuming volatility feature is at position 12
                if 12 < len(new_features):
                    new_features[12] = np.mean(volatility_preds)
            
            return new_features
            
        except Exception as e:
            self.logger.warning(f"Prediction to features conversion failed: {e}")
            return prev_features

    async def _summarize_scenario_async(self, scenario_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize scenario for analysis"""
        try:
            if not scenario_path:
                return {}
            
            price_changes = [step['price_changes'] for step in scenario_path]
            volatilities = [step['volatility_predictions'] for step in scenario_path]
            confidences = [step['confidence'] for step in scenario_path]
            regimes = [step['predicted_regime'] for step in scenario_path]
            
            # Calculate returns for each instrument
            total_returns = []
            for instrument in range(4):
                instrument_changes = [pc[instrument] for pc in price_changes if len(pc) > instrument]
                if instrument_changes:
                    total_return = np.sum(instrument_changes)
                    total_returns.append(total_return)
                else:
                    total_returns.append(0.0)
            
            # Calculate average volatilities
            avg_volatilities = []
            for instrument in range(4):
                instrument_vols = [v[instrument] for v in volatilities if len(v) > instrument]
                if instrument_vols:
                    avg_vol = np.mean(instrument_vols)
                    avg_volatilities.append(avg_vol)
                else:
                    avg_volatilities.append(0.5)
            
            # Regime distribution
            regime_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for regime in regimes:
                regime_counts[regime] += 1
            
            total_steps = len(regimes) if regimes else 1
            regime_distribution = {
                'trending': regime_counts[0] / total_steps,
                'volatile': regime_counts[1] / total_steps,
                'ranging': regime_counts[2] / total_steps,
                'unknown': regime_counts[3] / total_steps
            }
            
            # Calculate max drawdowns
            max_drawdowns = []
            for instrument in range(4):
                instrument_changes = [pc[instrument] for pc in price_changes if len(pc) > instrument]
                if instrument_changes:
                    max_dd = await self._calculate_max_drawdown_async(instrument_changes)
                    max_drawdowns.append(max_dd)
                else:
                    max_drawdowns.append(0.0)
            
            return {
                'total_returns': total_returns,
                'avg_volatilities': avg_volatilities,
                'max_drawdowns': max_drawdowns,
                'regime_distribution': regime_distribution,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'confidence_stability': 1.0 - np.std(confidences) if len(confidences) > 1 else 1.0,
                'dominant_regime': max(regime_distribution.items(), key=lambda x: x[1])[0],
                'scenario_quality': np.mean(confidences) * (1.0 - np.std(confidences)) if len(confidences) > 1 else 0.5
            }
            
        except Exception as e:
            self.logger.warning(f"Scenario summarization failed: {e}")
            return {}

    async def _calculate_max_drawdown_async(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        try:
            if not returns:
                return 0.0
            
            cumulative = np.cumprod(1 + np.array(returns))
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            
            return float(np.min(drawdown))
            
        except Exception:
            return 0.0

    async def _analyze_scenarios_async(self, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scenario collection for diversity and quality"""
        try:
            if not scenarios:
                return {}
            
            # Extract summaries
            summaries = [s.get('summary', {}) for s in scenarios if 'summary' in s]
            
            if not summaries:
                return {'diversity': 0.0, 'avg_confidence': 0.0, 'quality': 0.0}
            
            # Calculate diversity metrics
            returns_diversity = 0.0
            if all('total_returns' in s for s in summaries):
                all_returns = [s['total_returns'] for s in summaries]
                if all_returns:
                    returns_std = np.std([np.mean(returns) for returns in all_returns])
                    returns_diversity = min(1.0, float(returns_std) * 10)
            
            # Calculate average confidence
            avg_confidence = 0.0
            confidences = [s.get('avg_confidence', 0.0) for s in summaries]
            if confidences:
                avg_confidence = np.mean(confidences)
            
            # Calculate quality score
            qualities = [s.get('scenario_quality', 0.0) for s in summaries]
            avg_quality = np.mean(qualities) if qualities else 0.0
            
            # Regime diversity
            regime_diversity = 0.0
            if all('regime_distribution' in s for s in summaries):
                regime_entropies = []
                for summary in summaries:
                    dist = summary['regime_distribution']
                    probs = list(dist.values())
                    entropy = -np.sum([p * np.log(p + 1e-8) for p in probs])
                    regime_entropies.append(entropy)
                
                if regime_entropies:
                    regime_diversity = np.mean(regime_entropies) / np.log(4)  # Normalized by max entropy
            
            overall_diversity = (returns_diversity + regime_diversity) / 2.0
            
            return {
                'diversity': overall_diversity,
                'returns_diversity': returns_diversity,
                'regime_diversity': regime_diversity,
                'avg_confidence': avg_confidence,
                'avg_quality': avg_quality,
                'scenario_count': len(scenarios)
            }
            
        except Exception as e:
            self.logger.warning(f"Scenario analysis failed: {e}")
            return {'diversity': 0.0, 'avg_confidence': 0.0, 'quality': 0.0}

    # ================== PERFORMANCE AND MONITORING ==================

    async def _update_performance_metrics_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update comprehensive performance metrics"""
        try:
            # Update stability score
            if len(self.confidence_history) >= 10:
                recent_confidences = list(self.confidence_history)[-10:]
                self.stability_score = 1.0 - np.std(recent_confidences)
            
            # Update overall model confidence
            factors = [self.training_quality, self.prediction_quality, self.stability_score]
            self.model_confidence = np.mean([f for f in factors if f > 0])
            
            return {
                'performance_updated': True,
                'model_confidence': self.model_confidence,
                'training_quality': self.training_quality,
                'prediction_quality': self.prediction_quality,
                'stability_score': self.stability_score
            }
            
        except Exception as e:
            self.logger.warning(f"Performance metrics update failed: {e}")
            return {'performance_updated': False, 'error': str(e)}

    async def _update_operational_mode_async(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update operational mode based on model status"""
        try:
            old_mode = self.current_mode
            
            # Determine new mode based on state
            if not self.is_trained:
                if len(self.market_history) < self.wm_config.min_training_samples:
                    new_mode = WorldModelMode.DATA_COLLECTION
                else:
                    new_mode = WorldModelMode.TRAINING
            elif self.model_confidence < self.wm_config.min_prediction_quality:
                new_mode = WorldModelMode.CALIBRATION
            elif self.prediction_quality > 0.8:
                new_mode = WorldModelMode.OPTIMIZATION
            else:
                new_mode = WorldModelMode.ACTIVE_PREDICTION
            
            mode_changed = old_mode != new_mode
            
            if mode_changed:
                self.current_mode = new_mode
                self.mode_start_time = datetime.datetime.now()
                
                self.logger.info(format_operator_message(
                    message="World model mode changed",
                    icon="[RELOAD]",
                    old_mode=old_mode.value,
                    new_mode=new_mode.value,
                    confidence=f"{self.model_confidence:.3f}",
                    quality=f"{self.prediction_quality:.3f}"
                ))
            
            return {
                'mode_updated': True,
                'current_mode': self.current_mode.value,
                'mode_changed': mode_changed,
                'old_mode': old_mode.value if mode_changed else None,
                'mode_duration': (datetime.datetime.now() - self.mode_start_time).total_seconds()
            }
            
        except Exception as e:
            self.logger.warning(f"Mode update failed: {e}")
            return {'mode_updated': False, 'error': str(e)}

    def _update_model_health(self):
        """Update world model health metrics"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, 'is_trained') or not hasattr(self, 'training_quality') or not hasattr(self, 'prediction_quality'):
                return  # Skip if not fully initialized yet
                
            # Check model state
            if self.circuit_breaker['state'] == 'OPEN':
                self._health_status = 'warning'
                return
            
            # Check training quality
            if self.is_trained and self.training_quality < self.wm_config.min_training_quality:
                self._health_status = 'warning'
                return
            
            # Check prediction quality
            if self.prediction_quality < self.wm_config.min_prediction_quality:
                self._health_status = 'warning'
                return
            
            # Check for NaN parameters
            for param in self.parameters():
                if not torch.all(torch.isfinite(param.data)):
                    self._health_status = 'critical'
                    return
            
            self._health_status = 'healthy'
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Model health check failed: {e}")
            self._health_status = 'critical'

    def _analyze_prediction_effectiveness(self):
        """Analyze prediction effectiveness"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, 'prediction_errors'):
                return  # Skip if not fully initialized yet
                
            if len(self.prediction_errors) >= 20:
                recent_errors = list(self.prediction_errors)[-20:]
                avg_error = np.mean(recent_errors)
                
                if avg_error < 0.05:  # Very good predictions
                    self.logger.info(format_operator_message(
                        message="Excellent prediction accuracy achieved",
                        icon="[TARGET]",
                        avg_error=f"{avg_error:.4f}",
                        confidence=f"{self.model_confidence:.3f}"
                    ))
                elif avg_error > 0.2:  # Poor predictions
                    self.logger.warning(format_operator_message(
                        message="Poor prediction accuracy detected",
                        icon="[WARN]",
                        avg_error=f"{avg_error:.4f}",
                        mode=self.current_mode.value
                    ))
            
        except Exception as e:
            self.logger.error(f"Prediction effectiveness analysis failed: {e}")

    def _adapt_model_parameters(self):
        """Continuous model parameter adaptation"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, 'is_trained') or not hasattr(self, 'training_curves'):
                return  # Skip if not fully initialized yet
                
            # Adapt learning rate based on training progress
            if self.is_trained and len(self.training_curves['val_loss']) >= 10:
                recent_val_losses = list(self.training_curves['val_loss'])[-10:]
                try:
                    val_loss_trend = np.polyfit(range(len(recent_val_losses)), recent_val_losses, 1)[0]
                except Exception:
                    val_loss_trend = 0.0
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                if val_loss_trend > 0:  # Loss increasing - reduce LR
                    new_lr = current_lr * 0.95
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = max(new_lr, 1e-6)
                elif val_loss_trend < -0.01:  # Loss decreasing significantly - increase LR
                    new_lr = current_lr * 1.02
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = min(new_lr, 1e-2)
            
        except Exception as e:
            self.logger.warning(f"Model parameter adaptation failed: {e}")

    def _cleanup_old_data(self):
        """Cleanup old data to maintain performance"""
        try:
            # Check if all required attributes are initialized
            if not hasattr(self, 'attention_patterns'):
                return  # Skip if not fully initialized yet
                
            # Cleanup scenario cache if too old
            if (hasattr(self, '_last_scenario_time') and 
                (datetime.datetime.now() - self._last_scenario_time).total_seconds() > 7200):
                self.scenario_cache.clear()
            
            # Cleanup attention patterns
            if len(self.attention_patterns) > 100:
                # Keep only recent patterns
                recent_patterns = list(self.attention_patterns)[-50:]
                self.attention_patterns.clear()
                self.attention_patterns.extend(recent_patterns)
            
        except Exception as e:
            self.logger.warning(f"Data cleanup failed: {e}")

    # ================== THESIS AND SMARTINFOBUS METHODS ==================

    async def _generate_world_model_thesis(self, market_data: Dict[str, Any], 
                                         result: Dict[str, Any]) -> str:
        """Generate comprehensive world model thesis"""
        try:
            # Core metrics
            mode = self.current_mode.value
            confidence = self.model_confidence
            quality = self.prediction_quality
            
            thesis_parts = [
                f"World Model: {mode.upper()} mode with {confidence:.1%} confidence",
                f"Prediction Quality: {quality:.2f} accuracy score"
            ]
            
            # Model status
            if self.is_trained:
                if confidence > 0.8:
                    thesis_parts.append(f"EXCELLENT: High-quality predictions available")
                elif confidence > 0.6:
                    thesis_parts.append(f"GOOD: Reliable predictions generated")
                else:
                    thesis_parts.append(f"FAIR: Moderate prediction reliability")
            else:
                thesis_parts.append(f"TRAINING: Model learning from market data")
            
            # Recent activity
            if result.get('predictions_generated'):
                predictions = result.get('predictions', {})
                pred_confidence = predictions.get('confidence', 0.0)
                regime = predictions.get('predicted_regime', -1)
                thesis_parts.append(f"Latest prediction: {pred_confidence:.2f} confidence, regime {regime}")
            
            # Training status
            if result.get('training_completed'):
                epochs = result.get('epochs_completed', 0)
                thesis_parts.append(f"Training: {epochs} epochs completed")
            
            # Scenario generation
            if result.get('scenarios_generated'):
                scenario_count = len(result.get('scenarios', []))
                scenario_metrics = result.get('scenario_metrics', {})
                diversity = scenario_metrics.get('diversity', 0.0)
                thesis_parts.append(f"Scenarios: {scenario_count} generated, {diversity:.2f} diversity")
            
            # Data status
            data_sufficiency = len(self.market_history) / self.wm_config.history_size
            thesis_parts.append(f"Data: {data_sufficiency:.0%} capacity utilized")
            
            # Performance indicators
            if len(self.prediction_errors) >= 5:
                recent_error = np.mean(list(self.prediction_errors)[-5:])
                thesis_parts.append(f"Error rate: {recent_error:.3f} recent average")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"World model thesis generation failed: {str(e)} - Core modeling functional"

    async def _update_world_model_smart_bus(self, result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with world model results"""
        try:
            # Market predictions
            predictions_data = {
                'current_mode': self.current_mode.value,
                'is_trained': self.is_trained,
                'model_confidence': self.model_confidence,
                'prediction_quality': self.prediction_quality,
                'stability_score': self.stability_score,
                'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
                'device': str(self.device),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            # Add latest predictions if available
            if result.get('predictions_generated') and 'predictions' in result:
                predictions = result['predictions']
                predictions_data.update({
                    'latest_predictions': {
                        'price_changes': predictions['price_changes'].tolist() if hasattr(predictions['price_changes'], 'tolist') else list(predictions['price_changes']),
                        'volatility_predictions': predictions['volatility_predictions'].tolist() if hasattr(predictions['volatility_predictions'], 'tolist') else list(predictions['volatility_predictions']),
                        'regime_probabilities': predictions['regime_probabilities'].tolist() if hasattr(predictions['regime_probabilities'], 'tolist') else list(predictions['regime_probabilities']),
                        'confidence': predictions['confidence'],
                        'predicted_regime': predictions['predicted_regime'],
                        'confidence_level': predictions['confidence_level'],
                        'timestamp': predictions['timestamp']
                    }
                })
            
            self.smart_bus.set(
                'market_predictions',
                predictions_data,
                module='EnhancedWorldModel',
                thesis=thesis
            )
            
            # Scenario generation
            if self.scenario_cache:
                scenario_data = {
                    'scenarios_available': True,
                    'scenario_count': len(self.scenario_cache.get('scenarios', [])),
                    'scenario_timestamp': self.scenario_cache.get('timestamp'),
                    'scenario_parameters': self.scenario_cache.get('parameters', {}),
                    'scenarios': self.scenario_cache.get('scenarios', [])
                }
                
                self.smart_bus.set(
                    'scenario_generation',
                    scenario_data,
                    module='EnhancedWorldModel',
                    thesis="Market scenario generation and analysis"
                )
            
            # World model analytics
            analytics_data = {
                'model_architecture': {
                    'input_size': self.wm_config.input_size,
                    'hidden_size': self.wm_config.hidden_size,
                    'num_layers': self.wm_config.num_layers,
                    'sequence_length': self.wm_config.sequence_length,
                    'attention_heads': self.wm_config.attention_heads
                },
                'performance_metrics': {
                    'training_quality': self.training_quality,
                    'prediction_quality': self.prediction_quality,
                    'stability_score': self.stability_score,
                    'model_confidence': self.model_confidence
                },
                'data_status': {
                    'market_history_size': len(self.market_history),
                    'prediction_history_size': len(self.prediction_history),
                    'training_sessions': len(self.training_history),
                    'feature_importance_count': len(self.feature_importance)
                },
                'training_curves': {
                    curve_name: list(curve_data)[-20:]  # Recent training curve data
                    for curve_name, curve_data in self.training_curves.items()
                    if curve_data
                },
                'attention_patterns': len(self.attention_patterns),
                'genome_config': self.genome.copy()
            }
            
            self.smart_bus.set(
                'world_model_analytics',
                analytics_data,
                module='EnhancedWorldModel',
                thesis="Comprehensive world model analytics and performance tracking"
            )
            
            # Prediction confidence
            confidence_data = {
                'current_confidence': self.model_confidence,
                'prediction_confidence': self.prediction_quality,
                'stability_confidence': self.stability_score,
                'training_confidence': self.training_quality,
                'confidence_history': list(self.confidence_history)[-20:] if self.confidence_history else [],
                'confidence_classification': await self._classify_confidence_level_async(float(self.model_confidence)),
                'is_reliable': self.model_confidence > self.wm_config.confidence_threshold,
                'recommendations': await self._generate_confidence_recommendations_async()
            }
            
            self.smart_bus.set(
                'prediction_confidence',
                confidence_data,
                module='EnhancedWorldModel',
                thesis="World model prediction confidence and reliability assessment"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    async def _generate_confidence_recommendations_async(self) -> List[str]:
        """Generate confidence-based recommendations"""
        try:
            recommendations = []
            
            if self.model_confidence > 0.8:
                recommendations.append("High confidence: Suitable for active prediction and scenario planning")
            elif self.model_confidence > 0.6:
                recommendations.append("Moderate confidence: Good for trend analysis with caution")
            elif self.model_confidence > 0.4:
                recommendations.append("Low confidence: Use for general market insights only")
            else:
                recommendations.append("Very low confidence: Model requires retraining")
            
            if not self.is_trained:
                recommendations.append("Model training required before reliable predictions")
            
            if len(self.market_history) < self.wm_config.min_training_samples:
                recommendations.append("Collect more market data for improved accuracy")
            
            if self.prediction_quality < 0.5:
                recommendations.append("Consider adjusting model parameters or retraining")
            
            return recommendations
            
        except Exception:
            return ["Unable to generate recommendations"]

    # ================== FALLBACK AND ERROR HANDLING ==================

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no market data is available"""
        # FIXED: Reduce log noise - only warn every 100 calls during training
        # This is expected behavior when running in gym environment without SmartInfoBus
        if not hasattr(self, '_no_data_warn_count'):
            self._no_data_warn_count = 0
        self._no_data_warn_count += 1
        
        if self._no_data_warn_count == 1 or self._no_data_warn_count % 100 == 0:
            self.logger.warning(
                f"No market data available - maintaining current state "
                f"(occurrences: {self._no_data_warn_count}, expected during training)"
            )
        
        thesis = "No market data available - maintaining current state"

        # Minimal but contract-compliant payload
        predictions_data = {
            'current_mode': self.current_mode.value,
            'is_trained': self.is_trained,
            'model_confidence': self.model_confidence,
            'prediction_quality': self.prediction_quality,
            'stability_score': self.stability_score,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'device': str(self.device),
            'timestamp': datetime.datetime.now().isoformat()
        }
        scenario_data = {
            'scenarios_available': False,
            'scenario_count': 0,
            'scenario_timestamp': None,
            'scenario_parameters': {},
            'scenarios': []
        }
        analytics_data = {
            'model_architecture': {
                'input_size': self.wm_config.input_size,
                'hidden_size': self.wm_config.hidden_size,
                'num_layers': self.wm_config.num_layers,
                'sequence_length': self.wm_config.sequence_length,
                'attention_heads': self.wm_config.attention_heads
            },
            'performance_metrics': {
                'training_quality': self.training_quality,
                'prediction_quality': self.prediction_quality,
                'stability_score': self.stability_score,
                'model_confidence': self.model_confidence
            },
            'data_status': {
                'market_history_size': len(self.market_history),
                'prediction_history_size': len(self.prediction_history),
                'training_sessions': len(self.training_history),
                'feature_importance_count': len(self.feature_importance)
            },
            'training_curves': {},
            'attention_patterns': len(self.attention_patterns),
            'genome_config': self.genome.copy()
        }
        confidence_data = {
            'current_confidence': self.model_confidence,
            'prediction_confidence': self.prediction_quality,
            'stability_confidence': self.stability_score,
            'training_confidence': self.training_quality,
            'confidence_history': list(self.confidence_history)[-20:] if self.confidence_history else [],
            'confidence_classification': 'very_low',
            'is_reliable': self.model_confidence > self.wm_config.confidence_threshold,
            'recommendations': ["Collect more market data for improved accuracy"]
        }

        return {
            'market_predictions': predictions_data,
            'scenario_generation': scenario_data,
            'world_model_analytics': analytics_data,
            'prediction_confidence': confidence_data,
            '_thesis': thesis,
            'success': True,
            'fallback_reason': 'no_market_data'
        }

    async def _handle_world_model_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle world model errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self._trip_circuit_breaker()
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "EnhancedWorldModel")
        explanation = self.english_explainer.explain_error(
            "EnhancedWorldModel", str(error), "world model processing"
        )
        
        self.logger.error(format_operator_message(
            message="World model error",
            icon="[CRASH]",
            error=str(error),
            details=explanation,
            processing_time_ms=processing_time,
            circuit_breaker_state=self.circuit_breaker['state']
        ))
        
        # Record failure
        self._record_failure(error)
        
        return self._create_error_fallback_response(f"error: {str(error)}")

    def _create_error_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        thesis = f"World model error fallback: {reason}"

        predictions_data = {
            'current_mode': WorldModelMode.ERROR_RECOVERY.value,
            'is_trained': self.is_trained,
            'model_confidence': 0.1,
            'prediction_quality': 0.1,
            'stability_score': self.stability_score,
            'last_training_time': self.last_training_time.isoformat() if self.last_training_time else None,
            'device': str(self.device),
            'timestamp': datetime.datetime.now().isoformat(),
            'circuit_breaker_state': self.circuit_breaker['state']
        }
        scenario_data = {
            'scenarios_available': False,
            'scenario_count': 0,
            'scenario_timestamp': None,
            'scenario_parameters': {},
            'scenarios': []
        }
        analytics_data = {
            'model_architecture': {
                'input_size': self.wm_config.input_size,
                'hidden_size': self.wm_config.hidden_size,
                'num_layers': self.wm_config.num_layers,
                'sequence_length': self.wm_config.sequence_length,
                'attention_heads': self.wm_config.attention_heads
            },
            'performance_metrics': {
                'training_quality': self.training_quality,
                'prediction_quality': 0.1,
                'stability_score': self.stability_score,
                'model_confidence': 0.1
            },
            'data_status': {
                'market_history_size': len(self.market_history),
                'prediction_history_size': len(self.prediction_history),
                'training_sessions': len(self.training_history),
                'feature_importance_count': len(self.feature_importance)
            },
            'training_curves': {},
            'attention_patterns': len(self.attention_patterns),
            'genome_config': self.genome.copy()
        }
        confidence_data = {
            'current_confidence': 0.1,
            'prediction_confidence': 0.1,
            'stability_confidence': self.stability_score,
            'training_confidence': self.training_quality,
            'confidence_history': list(self.confidence_history)[-20:] if self.confidence_history else [],
            'confidence_classification': 'very_low',
            'is_reliable': False,
            'recommendations': ["Model requires error recovery and retraining"]
        }

        return {
            'market_predictions': predictions_data,
            'scenario_generation': scenario_data,
            'world_model_analytics': analytics_data,
            'prediction_confidence': confidence_data,
            '_thesis': thesis,
            'success': False,
            'fallback_reason': reason,
            'circuit_breaker_state': self.circuit_breaker['state']
        }

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        try:
            self.performance_tracker.record_metric(
                'EnhancedWorldModel', 'world_model_processing', processing_time, True
            )
        except Exception:
            pass
        
        # Reset circuit breaker on success (if HALF_OPEN -> CLOSED)
        if self.circuit_breaker['state'] in ('OPEN', 'HALF_OPEN'):
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        try:
            self.performance_tracker.record_metric(
                'EnhancedWorldModel', 'world_model_processing', 0, False
            )
        except Exception:
            pass

    # ================== PUBLIC INTERFACE METHODS ==================

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        try:
            # Core model status
            model_trained = float(self.is_trained)
            model_confidence = self.model_confidence
            prediction_quality = self.prediction_quality
            stability_score = self.stability_score
            
            # Training metrics
            training_quality = self.training_quality
            
            # Data availability
            data_sufficiency = min(1.0, len(self.market_history) / (self.wm_config.sequence_length * 2))
            
            # Recent prediction metrics
            latest_confidence = 0.0
            if self.prediction_history:
                latest_confidence = self.prediction_history[-1].get('confidence', 0.0)
            
            # Feature importance diversity (use feature_stats for per-feature statistics)
            feature_diversity = 0.0
            if self.feature_stats:
                variances = [info.get('variance', 0.0) for info in self.feature_stats.values() if isinstance(info, dict) and 'variance' in info]
                feature_diversity = np.mean(variances) if variances else 0.0
            
            # Circuit breaker status
            circuit_breaker_open = float(self.circuit_breaker['state'] == 'OPEN')
            
            # Mode indicator
            mode_indicator = {
                WorldModelMode.INITIALIZATION: 0.1,
                WorldModelMode.DATA_COLLECTION: 0.2,
                WorldModelMode.TRAINING: 0.3,
                WorldModelMode.CALIBRATION: 0.4,
                WorldModelMode.ACTIVE_PREDICTION: 0.8,
                WorldModelMode.SCENARIO_GENERATION: 0.9,
                WorldModelMode.OPTIMIZATION: 1.0,
                WorldModelMode.MAINTENANCE: 0.5,
                WorldModelMode.ERROR_RECOVERY: 0.0
            }.get(self.current_mode, 0.5)
            
            # Scenario availability
            scenarios_available = float(bool(self.scenario_cache))
            
            # Health status
            health_score = {'healthy': 1.0, 'warning': 0.5, 'critical': 0.0}.get(self._health_status, 0.5)
            
            return np.array([
                model_trained,
                model_confidence,
                prediction_quality,
                stability_score,
                training_quality,
                data_sufficiency,
                latest_confidence,
                feature_diversity,
                circuit_breaker_open,
                mode_indicator,
                scenarios_available,
                health_score
            ], dtype=np.float32)
            
        except Exception as e:
            self.logger.error(f"Observation generation failed: {e}")
            return np.zeros(12, dtype=np.float32)

    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose actions based on world model predictions"""
        try:
            if not self.is_trained or not self.prediction_history:
                return {
                    'action_type': 'no_action',
                    'action_values': [0.0, 0.0, 0.0, 0.0],
                    'confidence': 0.0,
                    'reasoning': 'Model not trained or no predictions available'
                }
            
            # Get latest prediction
            latest_prediction = self.prediction_history[-1]
            
            # Extract predicted price changes and confidence
            price_changes = latest_prediction['price_changes']
            confidence = latest_prediction['confidence']
            regime_probs = latest_prediction['regime_probabilities']
            
            # Scale actions by confidence and predicted magnitude
            action_scaling = confidence * 0.7  # Conservative scaling
            
            # Adjust scaling based on regime
            if len(regime_probs) >= 4:
                if regime_probs[1] > 0.5:  # Volatile regime
                    action_scaling *= 0.7
                elif regime_probs[0] > 0.5:  # Trending regime
                    action_scaling *= 1.2
            
            # Convert price changes to trading actions
            actions = price_changes * action_scaling
            
            # Apply additional risk constraints
            actions = np.clip(actions, -1.0, 1.0)
            
            # Ensure we return exactly 4 actions
            if len(actions) < 4:
                padded_actions = np.zeros(4)
                padded_actions[:len(actions)] = actions
                actions = padded_actions
            
            return {
                'action_type': 'world_model_prediction',
                'action_values': actions[:4].tolist(),
                'confidence': float(confidence),
                'reasoning': f'World model prediction with {confidence:.2%} confidence',
                'regime_probabilities': regime_probs.tolist() if hasattr(regime_probs, 'tolist') else list(regime_probs),
                'prediction_mode': self.current_mode.value,
                'model_trained': self.is_trained
            }
            
        except Exception as e:
            self.logger.error(f"Action proposal failed: {e}")
            return {
                'action_type': 'error',
                'action_values': [0.0, 0.0, 0.0, 0.0],
                'confidence': 0.0,
                'reasoning': f'Action proposal error: {str(e)}',
                'error': str(e)
            }

    def propose_action_legacy(self, obs: Any = None, info_bus: Optional[Any] = None) -> np.ndarray:
        """Legacy propose action method that returns numpy array"""
        try:
            if not self.is_trained or not self.prediction_history:
                return np.zeros(4, dtype=np.float32)
            
            # Get latest prediction
            latest_prediction = self.prediction_history[-1]
            
            # Extract predicted price changes and confidence
            price_changes = latest_prediction['price_changes']
            confidence = latest_prediction['confidence']
            regime_probs = latest_prediction['regime_probabilities']
            
            # Scale actions by confidence and predicted magnitude
            action_scaling = confidence * 0.7  # Conservative scaling
            
            # Adjust scaling based on regime
            if len(regime_probs) >= 4:
                if regime_probs[1] > 0.5:  # Volatile regime
                    action_scaling *= 0.7
                elif regime_probs[0] > 0.5:  # Trending regime
                    action_scaling *= 1.2
            
            # Convert price changes to trading actions
            actions = price_changes * action_scaling
            
            # Apply additional risk constraints
            actions = np.clip(actions, -1.0, 1.0)
            
            # Ensure we return exactly 4 actions
            if len(actions) < 4:
                padded_actions = np.zeros(4)
                padded_actions[:len(actions)] = actions
                actions = padded_actions
            
            return actions[:4].astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Legacy action proposal failed: {e}")
            return np.zeros(4, dtype=np.float32)

    def confidence(self, obs: Any = None, info_bus: Optional[Any] = None) -> float:
        """Return confidence in world model predictions"""
        try:
            base_confidence = 0.3
            
            # Confidence from model training
            if self.is_trained:
                base_confidence += self.model_confidence * 0.4
            
            # Confidence from prediction quality
            base_confidence += self.prediction_quality * 0.3
            
            # Confidence from stability
            base_confidence += self.stability_score * 0.2
            
            # Confidence from data sufficiency
            data_confidence = min(0.1, len(self.market_history) / (self.wm_config.sequence_length * 2) * 0.1)
            base_confidence += data_confidence
            
            # Penalty for circuit breaker issues
            if self.circuit_breaker['state'] == 'OPEN':
                base_confidence *= 0.3
            
            # Bonus for recent good predictions
            if self.prediction_history:
                recent_confidences = [p['confidence'] for p in list(self.prediction_history)[-5:]]
                if recent_confidences:
                    avg_recent_confidence = np.mean(recent_confidences)
                    base_confidence += avg_recent_confidence * 0.2
            
            return float(np.clip(base_confidence, 0.1, 1.0))
            
        except Exception:
            return 0.3

    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in world model predictions for given action"""
        try:
            # Base confidence from the existing confidence method
            base_confidence = self.confidence()
            
            # Action-specific confidence adjustments
            action_confidence = base_confidence
            
            # If action is provided, analyze its alignment with model predictions
            if action and isinstance(action, dict):
                # Get latest predictions from SmartInfoBus or internal state
                latest_predictions = self.smart_bus.get('market_predictions', 'EnhancedWorldModel')
                
                if latest_predictions and isinstance(latest_predictions, dict):
                    # Check alignment between action and predictions
                    prediction_alignment = self._calculate_prediction_alignment(action, latest_predictions)
                    action_confidence *= (0.7 + prediction_alignment * 0.3)  # 0.7 to 1.0 multiplier
                
                # Confidence boost for actions during high model certainty
                if 'confidence' in action:
                    model_certainty = min(1.0, float(action.get('confidence', 0.5)))
                    action_confidence *= (0.8 + model_certainty * 0.2)
            
            # Market context adjustments from inputs
            if inputs:
                market_data = inputs.get('market_data')
                if market_data and isinstance(market_data, dict):
                    # Higher confidence in familiar market conditions
                    market_familiarity = self._assess_market_familiarity(market_data)
                    action_confidence *= (0.85 + market_familiarity * 0.15)
            
            # Penalty for recent errors or low prediction quality
            if self.prediction_errors and len(self.prediction_errors) > 5:
                recent_errors = list(self.prediction_errors)[-5:]
                avg_error = np.mean(recent_errors)
                if avg_error > 0.3:  # High error threshold
                    action_confidence *= 0.8
            
            # Boost for stable prediction patterns
            if len(self.confidence_history) > 10:
                confidence_stability = 1.0 - np.std(list(self.confidence_history)[-10:])
                action_confidence += confidence_stability * 0.1
            
            return float(np.clip(action_confidence, 0.1, 1.0))
            
        except Exception as e:
            self.logger.warning(f"Calculate confidence failed: {e}")
            return 0.3

    def _calculate_prediction_alignment(self, action: Dict[str, Any], predictions: Dict[str, Any]) -> float:
        """Calculate how well an action aligns with model predictions.
        
        NOTE: Simplified implementation. The original checked for 'price_direction' 
        and 'volatility_prediction' keys which don't exist in the actual prediction 
        structure. Now uses 'latest_predictions' sub-dict with actual keys.
        """
        try:
            alignment_score = 0.5  # Neutral baseline
            
            # Get latest predictions from nested structure if available
            latest_preds = predictions.get('latest_predictions', predictions)
            
            # Check price change alignment (use predicted price_changes mean as direction proxy)
            price_changes = latest_preds.get('price_changes')
            if price_changes is not None and 'direction' in action:
                # Mean of price_changes indicates predicted direction
                predicted_direction = np.sign(np.mean(price_changes)) if hasattr(price_changes, '__iter__') else 0
                action_direction = action.get('direction', 0)
                
                # Convert string directions to numeric if needed
                if isinstance(action_direction, str):
                    action_direction = 1.0 if action_direction.lower() in ('long', 'buy') else (-1.0 if action_direction.lower() in ('short', 'sell') else 0.0)
                
                if predicted_direction == action_direction and predicted_direction != 0:
                    alignment_score += 0.3
                elif abs(predicted_direction - action_direction) < 0.5:
                    alignment_score += 0.1
            
            # Check confidence alignment
            pred_confidence = latest_preds.get('confidence', 0.5)
            if pred_confidence > 0.7:
                alignment_score += 0.1  # Boost when model is confident
            
            return float(np.clip(alignment_score, 0.0, 1.0))
            
        except Exception:
            return 0.5

    def _assess_market_familiarity(self, market_data: Dict[str, Any]) -> float:
        """Assess how familiar the model is with current market conditions"""
        try:
            familiarity = 0.5  # Baseline
            
            # Check if we have sufficient historical data for these conditions
            if len(self.market_history) > self.wm_config.sequence_length:
                # Simple heuristic: more data = more familiarity
                data_sufficiency = min(1.0, len(self.market_history) / self.wm_config.history_size)
                familiarity += data_sufficiency * 0.3
            
            # Check volatility familiarity (use feature_stats for per-feature data)
            if 'volatility' in market_data and self.feature_stats.get('volatility_level'):
                current_vol = market_data['volatility']
                vol_history = self.feature_stats['volatility_level']['values']
                if vol_history and len(vol_history) > 5:
                    vol_mean = np.mean(vol_history)
                    vol_std = np.std(vol_history)
                    if vol_std > 0:
                        # Familiarity based on how close current vol is to historical range
                        z_score = abs((current_vol - vol_mean) / vol_std)
                        if z_score < 1.0:  # Within 1 std dev
                            familiarity += 0.2
                        elif z_score < 2.0:  # Within 2 std dev  
                            familiarity += 0.1
            
            return np.clip(familiarity, 0.0, 1.0)
            
        except Exception:
            return 0.5

    # ================== EVOLUTIONARY METHODS ==================

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with network rebuilding if needed"""
        try:
            old_config = self.genome.copy()
            
            # Apply new genome
            self.genome = self._initialize_genome(genome)
            
            # Check if architecture changes require rebuilding
            architecture_changed = (
                old_config.get('hidden_size') != self.genome.get('hidden_size') or
                old_config.get('num_layers') != self.genome.get('num_layers') or
                old_config.get('attention_heads') != self.genome.get('attention_heads')
            )
            
            if architecture_changed:
                self.logger.info(format_operator_message(
                    message="Rebuilding neural networks due to genome changes",
                    icon="🧬",
                    hidden_size=self.wm_config.hidden_size,
                    num_layers=self.wm_config.num_layers,
                    attention_heads=self.wm_config.attention_heads
                ))
                
                # Reinitialize neural components
                self._initialize_neural_components_async()
                
                # Reset training status
                self.is_trained = False
                self.model_confidence = 0.0
                self.prediction_quality = 0.0
                self.training_quality = 0.0
                
                # Clear dependent data
                self.prediction_history.clear()
                self.training_history.clear()
                self.scenario_cache.clear()
            
            # Update optimizer parameters even without architecture changes
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.wm_config.learning_rate
                param_group['weight_decay'] = self.wm_config.weight_decay
            
            self.logger.info(f"Genome updated: {len([k for k, v in genome.items() if old_config.get(k) != v])} parameters changed")
            
        except Exception as e:
            self.logger.error(f"Genome setting failed: {e}")
        
    def mutate(self, mutation_rate: float = 0.3):
        """Enhanced mutation with intelligent parameter adjustment"""
        try:
            g = self.genome.copy()
            mutations = []
            
            # Architectural mutations with constraints
            if np.random.rand() < mutation_rate:
                old_val = g["hidden_size"]
                # Prefer common sizes for efficiency
                size_options = [32, 48, 64, 96, 128, 192, 256, 384, 512]
                current_idx = size_options.index(old_val) if old_val in size_options else 2
                new_idx = int(np.clip(current_idx + np.random.choice([-2, -1, 0, 1, 2]), 0, len(size_options) - 1))
                g["hidden_size"] = int(size_options[new_idx])
                mutations.append(f"hidden_size: {old_val} → {g['hidden_size']}")
                
            if np.random.rand() < mutation_rate:
                old_val = g["num_layers"]
                g["num_layers"] = int(np.clip(old_val + np.random.choice([-1, 0, 1]), 1, 6))
                mutations.append(f"num_layers: {old_val} → {g['num_layers']}")
                
            if np.random.rand() < mutation_rate:
                old_val = g["attention_heads"]
                head_options = [2, 4, 8, 16]
                g["attention_heads"] = int(np.random.choice(head_options))
                mutations.append(f"attention_heads: {old_val} → {g['attention_heads']}")
                
            if np.random.rand() < mutation_rate:
                old_val = g["sequence_length"]
                g["sequence_length"] = int(np.clip(old_val + np.random.randint(-15, 16), 20, 200))
                mutations.append(f"sequence_length: {old_val} → {g['sequence_length']}")
                
            # Hyperparameter mutations
            if np.random.rand() < mutation_rate:
                old_val = g["learning_rate"]
                multiplier = float(np.random.uniform(0.5, 2.0))
                g["learning_rate"] = float(np.clip(old_val * multiplier, 1e-5, 1e-2))
                mutations.append(f"learning_rate: {old_val:.1e} → {g['learning_rate']:.1e}")
                
            if np.random.rand() < mutation_rate:
                old_val = g["dropout"]
                g["dropout"] = float(np.clip(old_val + np.random.uniform(-0.1, 0.1), 0.0, 0.8))
                mutations.append(f"dropout: {old_val:.2f} → {g['dropout']:.2f}")
                
            if np.random.rand() < mutation_rate:
                old_val = g["batch_size"]
                batch_options = [16, 24, 32, 48, 64, 96, 128]
                g["batch_size"] = int(np.random.choice(batch_options))
                mutations.append(f"batch_size: {old_val} → {g['batch_size']}")
            
            if np.random.rand() < mutation_rate:
                old_val = g["weight_decay"]
                multiplier = float(np.random.uniform(0.1, 10.0))
                g["weight_decay"] = float(np.clip(old_val * multiplier, 1e-6, 1e-3))
                mutations.append(f"weight_decay: {old_val:.1e} → {g['weight_decay']:.1e}")
            
            if mutations:
                self.logger.info(format_operator_message(
                    message="World model mutation applied",
                    icon="🧬",
                    changes=", ".join(mutations[:3]) + (f" (+{len(mutations)-3} more)" if len(mutations) > 3 else "")
                ))
                
            # Neural weight mutation (more conservative)
            if np.random.rand() < mutation_rate * 0.2:
                noise_std = 0.01  # Smaller noise for stability
                with torch.no_grad():
                    for param in self.parameters():
                        if param.requires_grad and len(param.shape) > 1:  # Only weight matrices
                            param.data += noise_std * torch.randn_like(param.data)
                
                self.logger.info(f"Neural weights mutated with std={noise_std}")
            
            self.set_genome(g)
            
        except Exception as e:
            self.logger.error(f"Mutation failed: {e}")
        
    def crossover(self, other: "EnhancedWorldModel") -> "EnhancedWorldModel":
        """Enhanced crossover with performance-weighted combination"""
        try:
            if not isinstance(other, EnhancedWorldModel):
                self.logger.warning("Crossover with incompatible type")
                return self
            
            # Use getattr to safely access attributes
            self_confidence = getattr(self, 'model_confidence', 0.0)
            self_quality = getattr(self, 'prediction_quality', 0.0)
            other_confidence = getattr(other, 'model_confidence', 0.0)
            other_quality = getattr(other, 'prediction_quality', 0.0)
            
            # Performance-based weighting
            self_score = self_confidence * self_quality
            other_score = other_confidence * other_quality
            
            if self_score + other_score > 0:
                self_weight = self_score / (self_score + other_score)
            else:
                self_weight = 0.5
            
            # Weighted crossover for continuous parameters
            self_genome = getattr(self, 'genome', {})
            other_genome = getattr(other, 'genome', {})
            new_genome = {}
            
            for key in self_genome:
                if key in other_genome:
                    if isinstance(self_genome[key], (int, float)):
                        if np.random.rand() < 0.7:  # 70% chance of weighted average
                            new_genome[key] = (
                                self_weight * self_genome[key] + 
                                (1 - self_weight) * other_genome[key]
                            )
                            # Round integers
                            if isinstance(self_genome[key], int):
                                new_genome[key] = int(round(new_genome[key]))
                        else:  # 30% chance of discrete selection
                            new_genome[key] = self_genome[key] if np.random.rand() < self_weight else other_genome[key]
                    else:
                        # Discrete selection for non-numeric parameters
                        new_genome[key] = self_genome[key] if np.random.rand() < self_weight else other_genome[key]
                else:
                    new_genome[key] = self_genome[key]
            
            # Create child
            child = EnhancedWorldModel()
            # Use setattr to avoid linter issues with attribute assignment
            setattr(child, 'config', getattr(self, 'config', None) or WorldModelConfig())
            if hasattr(child, 'set_genome'):
                child.set_genome(new_genome)  # type: ignore
            
            # Neural weight crossover (if architectures match)
            self_config = getattr(self, 'config', None)
            other_config = getattr(other, 'config', None)
            if (self_config and other_config and
                self_config.hidden_size == other_config.hidden_size and 
                self_config.num_layers == other_config.num_layers and 
                self_config.attention_heads == other_config.attention_heads):
                
                try:
                    with torch.no_grad():
                        child_params = list(cast(nn.Module, child).parameters())
                        self_params = list(cast(nn.Module, self).parameters())
                        other_params = list(cast(nn.Module, other).parameters())
                        
                        for child_param, self_param, other_param in zip(
                            child_params, self_params, other_params
                        ):
                            if child_param.shape == self_param.shape == other_param.shape:
                                # Weighted combination instead of random selection
                                alpha = float(self_weight + np.random.normal(0, 0.1))
                                alpha = float(np.clip(alpha, 0, 1))
                                child_param.data = alpha * self_param.data + (1 - alpha) * other_param.data
                    
                    self.logger.info("Neural weight crossover completed with weighted blending")
                    
                except Exception as e:
                    self.logger.warning(f"Neural weight crossover failed: {e}")
            
            # Inherit best market data
            self_history = getattr(self, 'market_history', [])
            other_history = getattr(other, 'market_history', [])
            self_importance = getattr(self, 'feature_importance', {})
            other_importance = getattr(other, 'feature_importance', {})
            
            if self_score > other_score and len(self_history) > 0:
                setattr(child, 'market_history', self_history.copy())
                setattr(child, 'feature_importance', self_importance.copy())
            elif len(other_history) > 0:
                setattr(child, 'market_history', other_history.copy())
                setattr(child, 'feature_importance', other_importance.copy())
            
            self.logger.info(format_operator_message(
                message="World model crossover completed",
                icon="🧬",
                self_score=f"{self_score:.3f}",
                other_score=f"{other_score:.3f}",
                self_weight=f"{self_weight:.2f}"
            ))
            
            return child
            
        except Exception as e:
            self.logger.error(f"Crossover failed: {e}")
            return self

    # ================== STATE MANAGEMENT ==================

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()

        # Stop monitoring while resetting
        self._stop_monitoring_thread()
        
        # Reset mixin states (reset components directly)
        # Clear trading state
        if hasattr(self, '_trading_state'):
            self._trading_state = {}
        if hasattr(self, '_risk_metrics'):
            self._risk_metrics = {}
        if hasattr(self, '_analysis_state'):
            self._analysis_state = {}
        
        # Clear world model state
        self.market_history.clear()
        self.feature_history.clear()
        self.prediction_history.clear()
        self.training_history.clear()
        self.prediction_errors.clear()
        self.confidence_history.clear()
        self.attention_patterns.clear()
        self.scenario_cache.clear()
        
        # Reset performance metrics
        self.is_trained = False
        self.model_confidence = 0.0
        self.prediction_quality = 0.0
        self.training_quality = 0.0
        self.stability_score = 1.0
        self.last_training_time = None
        
        # Reset training curves
        for curve in self.training_curves.values():
            curve.clear()
        
        # Reset feature importance and stats
        self.feature_importance.clear()
        self.feature_stats.clear()
        
        # Reset circuit breaker
        self.circuit_breaker['failures'] = 0
        self.circuit_breaker['state'] = 'CLOSED'
        self._health_status = 'healthy'
        
        # Reset mode
        self.current_mode = WorldModelMode.INITIALIZATION
        self.mode_start_time = datetime.datetime.now()
        
        # Reset external integrations
        self.external_model_sources.clear()
        self.ensemble_weights.clear()
        
        # Restart monitoring
        self._start_monitoring()

        self.logger.info("[RELOAD] Enhanced World Model reset - all state cleared")

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring_thread()

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'current_mode': self.current_mode.value,
            'model_confidence': self.model_confidence,
            'prediction_quality': self.prediction_quality,
            'training_quality': self.training_quality,
            'stability_score': self.stability_score,
            'is_trained': self.is_trained,
            'data_sufficiency': len(self.market_history) / self.wm_config.history_size
        }

    def get_world_model_report(self) -> str:
        """Generate operator-friendly world model report"""
        
        # Model status indicators
        if self.is_trained:
            if self.model_confidence > 0.8:
                model_status = "[ROCKET] Excellent"
            elif self.model_confidence > 0.6:
                model_status = "[OK] Good"
            elif self.model_confidence > 0.4:
                model_status = "[FAST] Fair"
            else:
                model_status = "[WARN] Poor"
        else:
            model_status = "[FAIL] Untrained"
        
        # Mode status
        mode_emoji = {
            WorldModelMode.INITIALIZATION: "[RELOAD]",
            WorldModelMode.DATA_COLLECTION: "[STATS]",
            WorldModelMode.TRAINING: "[TARGET]",
            WorldModelMode.CALIBRATION: "[TOOL]",
            WorldModelMode.ACTIVE_PREDICTION: "🔮",
            WorldModelMode.SCENARIO_GENERATION: "🎭",
            WorldModelMode.OPTIMIZATION: "[FAST]",
            WorldModelMode.MAINTENANCE: "[TOOL]",
            WorldModelMode.ERROR_RECOVERY: "🆘"
        }
        
        mode_status = f"{mode_emoji.get(self.current_mode, '❓')} {self.current_mode.value.upper()}"
        
        # Health status
        health_emoji = "[OK]" if self._health_status == 'healthy' else "[WARN]" if self._health_status == 'warning' else "[ALERT]"
        cb_status = "[RED] OPEN" if self.circuit_breaker['state'] == 'OPEN' else "[GREEN] CLOSED"
        
        # Data sufficiency
        data_sufficiency = len(self.market_history) / self.wm_config.history_size
        if data_sufficiency > 0.8:
            data_status = "[OK] Excellent"
        elif data_sufficiency > 0.5:
            data_status = "[FAST] Good"
        elif data_sufficiency > 0.2:
            data_status = "[WARN] Limited"
        else:
            data_status = "[FAIL] Insufficient"
        
        # Training status
        training_status = "No training"
        if self.training_history:
            last_training = self.training_history[-1]
            training_time = datetime.datetime.fromisoformat(last_training['timestamp'])
            time_ago = datetime.datetime.now() - training_time
            if time_ago.total_seconds() < 3600:
                training_status = f"Recent ({time_ago.seconds//60}m ago)"
            elif time_ago.total_seconds() < 86400:
                training_status = f"Today ({time_ago.seconds//3600}h ago)"
            else:
                training_status = f"Stale ({time_ago.days}d ago)"
        
        # Prediction trend
        prediction_trend = "[STATS] No data"
        if len(self.prediction_history) >= 3:
            recent_confidences = [p['confidence'] for p in list(self.prediction_history)[-3:]]
            if len(recent_confidences) >= 2:
                trend = recent_confidences[-1] - recent_confidences[0]
                if trend > 0.1:
                    prediction_trend = "[CHART] Improving"
                elif trend < -0.1:
                    prediction_trend = "📉 Declining"
                else:
                    prediction_trend = "[STATS] Stable"
        
        return f"""
🌍 ENHANCED WORLD MODEL v4.1
═══════════════════════════════════════════════════
🧠 Model Status: {model_status} ({self.model_confidence:.3f})
[TOOL] Current Mode: {mode_status}
[CHART] Predictions: {prediction_trend}
[TARGET] Stability: {self.stability_score:.3f}
⏰ Training: {training_status}

[HEALTH] SYSTEM HEALTH
• Status: {health_emoji} {self._health_status.upper()}
• Circuit Breaker: {cb_status}
• Data Quality: {data_status} ({data_sufficiency:.1%})

🏗️ NEURAL ARCHITECTURE
• Input Features: {self.wm_config.input_size}
• Hidden Units: {self.wm_config.hidden_size}
• LSTM Layers: {self.wm_config.num_layers}
• Attention Heads: {self.wm_config.attention_heads}
• Sequence Length: {self.wm_config.sequence_length}
• Dropout Rate: {self.wm_config.dropout:.2f}

[STATS] PERFORMANCE METRICS
• Model Confidence: {self.model_confidence:.3f}
• Prediction Quality: {self.prediction_quality:.3f}
• Training Quality: {self.training_quality:.3f}
• Stability Score: {self.stability_score:.3f}

[SAVE] DATA STATUS
• Market History: {len(self.market_history)}/{self.wm_config.history_size}
• Feature History: {len(self.feature_history)}
• Prediction History: {len(self.prediction_history)}
• Training Sessions: {len(self.training_history)}
• Attention Patterns: {len(self.attention_patterns)}

[TOOL] TRAINING CONFIGURATION
• Learning Rate: {self.wm_config.learning_rate:.1e}
• Batch Size: {self.wm_config.batch_size}
• Weight Decay: {self.wm_config.weight_decay:.1e}
• Gradient Clip: {self.wm_config.gradient_clip}
• Device: {self.device}

[CHART] RECENT ACTIVITY
• Predictions (last hour): {len([p for p in self.prediction_history if ('timestamp' in p and (datetime.datetime.now() - datetime.datetime.fromisoformat(p['timestamp'])).total_seconds() < 3600)])}
• High-confidence predictions: {len([p for p in self.prediction_history if p.get('confidence', 0) > 0.7])}
• Scenario cache: {'Available' if self.scenario_cache else 'Empty'}
• Feature importance: {len(self.feature_importance)} tracked features

🧬 EVOLUTIONARY GENOME
• Hidden Size: {self.genome.get('hidden_size')}
• Layers: {self.genome.get('num_layers')}
• Attention Heads: {self.genome.get('attention_heads')}
• Sequence Length: {self.genome.get('sequence_length')}
• Learning Rate: {self.genome.get('learning_rate'):.1e}
• Dropout: {self.genome.get('dropout'):.2f}

🎭 SCENARIO GENERATION
• Last Generation: {self.scenario_cache.get('timestamp', 'Never') if self.scenario_cache else 'Never'}
• Scenarios Available: {len(self.scenario_cache.get('scenarios', [])) if self.scenario_cache else 0}
• Scenario Steps: {self.wm_config.scenario_steps}

🔗 EXTERNAL INTEGRATIONS
• Model Sources: {len(self.external_model_sources)}
• Ensemble Weights: {len(self.ensemble_weights)}
        """

    # ================== UTILITY METHODS ==================

    def force_training_mode(self, reason: str = "manual_override") -> None:
        """Force model into training mode"""
        old_mode = self.current_mode
        self.current_mode = WorldModelMode.TRAINING
        
        self.logger.warning(format_operator_message(
            message="Training mode forced",
            icon="[TARGET]",
            reason=reason,
            old_mode=old_mode.value
        ))

    def force_prediction_mode(self, reason: str = "manual_override") -> None:
        """Force model into prediction mode"""
        if self.is_trained:
            old_mode = self.current_mode
            self.current_mode = WorldModelMode.ACTIVE_PREDICTION
            
            self.logger.info(format_operator_message(
                message="Prediction mode forced",
                icon="🔮",
                reason=reason,
                old_mode=old_mode.value
            ))
        else:
            self.logger.warning("Cannot force prediction mode: model not trained")

    def clear_scenario_cache(self) -> None:
        """Clear scenario cache"""
        self.scenario_cache.clear()
        self.logger.info("🎭 Scenario cache cleared")

    def get_latest_predictions(self) -> Optional[Dict[str, Any]]:
        """Get latest predictions if available"""
        if self.prediction_history:
            return self.prediction_history[-1].copy()
        return None

    def get_training_progress(self) -> Dict[str, Any]:
        """Get training progress information"""
        if not self.training_history:
            return {'training_sessions': 0, 'progress': 'No training completed'}
        
        latest_training = self.training_history[-1]
        return {
            'training_sessions': len(self.training_history),
            'latest_session': latest_training,
            'training_quality': self.training_quality,
            'model_confidence': self.model_confidence,
            'data_sufficiency': len(self.market_history) / max(1, self.wm_config.min_training_samples)
        }

    def set_external_model_source(self, source_name: str, source_data: Dict[str, Any]) -> None:
        """Set external model source data"""
        self.external_model_sources[source_name] = {
            'data': source_data,
            'timestamp': datetime.datetime.now().isoformat()
        }

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, market_data: Optional[Dict[str, Any]] = None, **kwargs):
        """Legacy compatibility step method"""
        import asyncio
        
        # Create event loop for async operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Prepare inputs
            inputs = {}
            if market_data:
                inputs.update(market_data)
            inputs.update(kwargs)
            
            # Run async processing
            result = loop.run_until_complete(self.process(**inputs))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Legacy step operation failed: {e}")
            return {'error': str(e)}
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def fit_on_history(self, validation_split: float = 0.2, epochs: int = 10) -> Dict[str, float]:
        """Legacy training interface"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Trigger training
            result = loop.run_until_complete(self._train_model_async())
            
            if result.get('training_completed'):
                return {
                    'loss': result.get('final_train_loss', float('inf')),
                    'val_loss': result.get('best_val_loss', float('inf')),
                    'confidence': result.get('model_confidence', 0.0)
                }
            else:
                return {'loss': float('inf'), 'val_loss': float('inf'), 'confidence': 0.0}
                
        except Exception as e:
            self.logger.error(f"Legacy training failed: {e}")
            return {'loss': float('inf'), 'val_loss': float('inf'), 'confidence': 0.0}
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def simulate_scenarios(self, steps: int = 10, num_scenarios: int = 5) -> List[Dict[str, Any]]:
        """Legacy scenario generation interface"""
        import asyncio
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Set scenario parameters
            old_steps = self.wm_config.scenario_steps
            self.wm_config.scenario_steps = steps
            try:
                self.set_config(self.wm_config.to_dict())
            except Exception:
                self.config = self.wm_config.to_dict()  # type: ignore[assignment]
            
            # Generate scenarios
            result = loop.run_until_complete(self._generate_scenarios_async({}))
            
            # Restore original parameters
            self.wm_config.scenario_steps = old_steps
            try:
                self.set_config(self.wm_config.to_dict())
            except Exception:
                self.config = self.wm_config.to_dict()  # type: ignore[assignment]
            
            if result.get('scenarios_generated'):
                return result.get('scenarios', [])
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Legacy scenario generation failed: {e}")
            return []
        finally:
            try:
                loop.close()
            except Exception:
                pass

    def fit(self, feature1: np.ndarray, feature2: np.ndarray, seq_len: int = 50,
            batch_size: int = 64, epochs: int = 5) -> float:
        """Legacy training interface with features"""
        # Store features as market data
        for i, (f1, f2) in enumerate(zip(feature1, feature2)):
            market_state = {
                'timestamp': datetime.datetime.now().isoformat(),
                'features': np.array([f1, f2] + [0.0] * (self.wm_config.input_size - 2), dtype=np.float32),
                'prices': {'instrument_1': f1 * 2000, 'instrument_2': f2 * 2000},
                'market_snapshot': {},
                'risk_snapshot': {},
                'trading_snapshot': {},
                'step_count': i
            }
            self.market_history.append(market_state)
        
        # Train model
        result = self.fit_on_history(epochs=epochs)
        return result.get('loss', float('inf'))

    def simulate(self, init_returns: np.ndarray, init_vol: np.ndarray, steps: int = 10) -> np.ndarray:
        """Legacy simulation interface"""
        scenarios = self.simulate_scenarios(steps=steps, num_scenarios=1)
        
        if scenarios:
            scenario = scenarios[0]
            if 'path' in scenario:
                price_changes = []
                for step_data in scenario['path']:
                    changes = step_data.get('price_changes', [0.0, 0.0])
                    # Ensure we have at least 2 changes, pad with zeros if needed
                    if len(changes) < 2:
                        changes = list(changes) + [0.0] * (2 - len(changes))
                    price_changes.append(changes[:2])
                
                return np.array(price_changes, dtype=np.float32)
        
        return np.zeros((steps, 2), dtype=np.float32)

    def get_state(self) -> Dict[str, Any]:
        """Legacy state interface"""
        return super().get_state()

    def set_state(self, state: Dict[str, Any]):
        """Legacy state interface"""
        super().set_state(state)


# ================== SUPPORTING CLASSES AND FUNCTIONS ==================

def create_world_model_from_config(config_dict: Dict[str, Any]) -> EnhancedWorldModel:
    """Factory function to create world model from configuration"""
    model = EnhancedWorldModel(WorldModelConfig.from_dict(config_dict).to_dict())
    if hasattr(model, 'set_genome'):
        model.set_genome(config_dict)  # type: ignore
    return model

def create_world_model_ensemble(configs: List[Dict[str, Any]]) -> List[EnhancedWorldModel]:
    """Create ensemble of world models"""
    ensemble = []
    for i, config_dict in enumerate(configs):
        model = create_world_model_from_config(config_dict)
        model.logger.info(f"Ensemble model {i+1}/{len(configs)} created")
        ensemble.append(model)
    
    return ensemble

# Example usage:
if __name__ == "__main__":
    config = {
        "input_size": 10,
        "hidden_size": 20,
        "output_size": 2
    }
    model = create_world_model_from_config(config)
    print("World model created:", model)
