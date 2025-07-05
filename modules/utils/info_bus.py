# ─────────────────────────────────────────────────────────────
# File: modules/utils/info_bus.py
# 🚀 ENHANCED InfoBus with centralized management and validation
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal, Optional, Union, Callable, Set
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass, field
import logging
from functools import wraps
import hashlib
import json
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# CORE TYPE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════

class PositionInfo(TypedDict):
    """Information about a single position"""
    symbol: str
    side: Literal["long", "short"]
    size: float
    entry_price: float
    current_price: float
    unrealised_pnl: float
    realised_pnl: float
    duration: int  # bars held
    margin_used: float
    
class TradeInfo(TypedDict):
    """Information about executed trades"""
    symbol: str
    side: Literal["buy", "sell"]
    size: float
    price: float
    timestamp: str
    pnl: float
    reason: str
    confidence: float
    execution_time_ms: float

class RiskSnapshot(TypedDict, total=False):
    """Current risk metrics"""
    balance: float
    equity: float
    margin_used: float
    free_margin: float
    margin_level: float  # equity/margin_used * 100
    max_drawdown: float
    current_drawdown: float
    open_positions: List[PositionInfo]
    var_95: Optional[float]
    cvar_95: Optional[float]
    sharpe_ratio: Optional[float]
    sortino_ratio: Optional[float]
    risk_score: float  # 0-1 range (NOT percentage)
    
class Vote(TypedDict):
    """Individual module vote"""
    module: str
    instrument: str
    action: Union[float, np.ndarray]
    confidence: float
    reasoning: Optional[str]
    compute_time_ms: float
    
class MarketContext(TypedDict):
    """Market regime and conditions"""
    regime: Literal["trending", "volatile", "ranging", "unknown"]
    volatility: Dict[str, float]
    trend_strength: Dict[str, float]
    volume_profile: Optional[Dict[str, float]]
    news_sentiment: Optional[float]
    regime_confidence: float
    regime_duration: int
    
class ComputationCache(TypedDict):
    """Cached computation results"""
    features: Dict[str, np.ndarray]
    indicators: Dict[str, Dict[str, float]]
    regime_detection: Dict[str, Any]
    pattern_analysis: Dict[str, Any]
    correlation_matrix: Optional[np.ndarray]
    
class PerformanceMetrics(TypedDict):
    """System performance tracking"""
    step_compute_time_ms: float
    module_timings: Dict[str, float]
    cache_hits: int
    cache_misses: int
    data_quality_score: float
    
class InfoBus(TypedDict, total=False):
    """🚀 ENHANCED Central data container with single source of truth"""
    # Timing
    timestamp: str
    step_idx: int
    episode_idx: int
    
    # Market data (SINGLE SOURCE)
    prices: Dict[str, float]
    ohlcv: Dict[str, Dict[str, np.ndarray]]  # symbol -> timeframe -> OHLCV
    features: Dict[str, np.ndarray]
    market_context: MarketContext
    
    # Trading state
    positions: List[PositionInfo]
    recent_trades: List[TradeInfo]
    pending_orders: List[Dict[str, Any]]
    
    # Model outputs
    raw_actions: np.ndarray
    votes: List[Vote]
    consensus: float
    arbiter_weights: np.ndarray
    
    # Risk & compliance
    risk: RiskSnapshot
    compliance_flags: List[str]
    risk_limits: Dict[str, float]
    
    # Performance
    pnl_today: float
    trade_count: int
    win_rate: float
    sharpe_ratio: float
    
    # System state
    computation_cache: ComputationCache
    performance: PerformanceMetrics
    
    # Alerts & logging
    alerts: List[Dict[str, Any]]
    audit_log: List[str]
    errors: List[str]
    
    # Module data
    module_data: Dict[str, Any]
    
    # Metadata
    _version: int  # InfoBus version for compatibility
    _checksum: str  # Data integrity check

# ═══════════════════════════════════════════════════════════════════
# 🚀 INFOBUS MANAGER - Single Source of Truth
# ═══════════════════════════════════════════════════════════════════

class InfoBusManager:
    """
    Centralized InfoBus management ensuring single source of truth.
    All modules MUST go through this manager for data access/updates.
    """
    
    _instance: Optional['InfoBusManager'] = None
    _current_bus: Optional[InfoBus] = None
    
    def __init__(self):
        self.logger = logging.getLogger("InfoBusManager")
        self._computation_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._locked_fields: Set[str] = set()
        self._module_access_log = defaultdict(int)
        self._data_validators = {}
        
    @classmethod
    def get_instance(cls) -> 'InfoBusManager':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def create_info_bus(cls, env: Any, step: int = 0) -> InfoBus:
        """Create new InfoBus for step - SINGLE ENTRY POINT"""
        manager = cls.get_instance()
        
        # Clear previous bus
        cls._current_bus = None
        manager._computation_cache.clear()
        
        # Create new bus with all required fields
        info_bus: InfoBus = {
            'timestamp': now_utc(),
            'step_idx': step,
            'episode_idx': getattr(env, 'episode_count', 0),
            '_version': 2,
            '_checksum': '',
            
            # Initialize all collections
            'prices': {},
            'ohlcv': {},
            'features': {},
            'positions': [],
            'recent_trades': [],
            'pending_orders': [],
            'votes': [],
            'alerts': [],
            'errors': [],
            'audit_log': [],
            'compliance_flags': [],
            'module_data': {},
            
            # Initialize computation cache
            'computation_cache': {
                'features': {},
                'indicators': {},
                'regime_detection': {},
                'pattern_analysis': {},
                'correlation_matrix': None
            },
            
            # Initialize performance metrics
            'performance': {
                'step_compute_time_ms': 0.0,
                'module_timings': {},
                'cache_hits': 0,
                'cache_misses': 0,
                'data_quality_score': 100.0
            }
        }
        
        # Extract market data ONCE
        manager._extract_market_data(info_bus, env)
        
        # Extract risk data ONCE
        manager._extract_risk_data(info_bus, env)
        
        # Calculate checksum
        info_bus['_checksum'] = manager._calculate_checksum(info_bus)
        
        # Store as current
        cls._current_bus = info_bus
        
        manager.logger.debug(f"Created InfoBus for step {step}")
        return info_bus
    
    @classmethod
    def get_current(cls) -> Optional[InfoBus]:
        """Get current InfoBus - prevents multiple creation"""
        return cls._current_bus
    
    def _extract_market_data(self, info_bus: InfoBus, env: Any):
        """Extract market data ONCE from environment"""
        if not hasattr(env, 'data') or not hasattr(env, 'instruments'):
            self.logger.warning("Environment missing market data")
            return
            
        step = info_bus['step_idx']
        
        # Extract OHLCV data for all instruments and timeframes
        for instrument in env.instruments:
            if instrument not in env.data:
                continue
                
            info_bus['ohlcv'][instrument] = {}
            
            for timeframe in ['H1', 'H4', 'D1']:
                if timeframe not in env.data[instrument]:
                    continue
                    
                df = env.data[instrument][timeframe]
                if step >= len(df):
                    continue
                    
                # Get window of data
                window_size = 100
                start_idx = max(0, step - window_size + 1)
                end_idx = step + 1
                
                # Extract OHLCV arrays
                info_bus['ohlcv'][instrument][timeframe] = {
                    'open': df['open'].iloc[start_idx:end_idx].values.astype(np.float32),
                    'high': df['high'].iloc[start_idx:end_idx].values.astype(np.float32),
                    'low': df['low'].iloc[start_idx:end_idx].values.astype(np.float32),
                    'close': df['close'].iloc[start_idx:end_idx].values.astype(np.float32),
                    'volume': df['volume'].iloc[start_idx:end_idx].values.astype(np.float32) if 'volume' in df.columns else np.ones(end_idx - start_idx, dtype=np.float32)
                }
                
            # Set current price
            if 'D1' in info_bus['ohlcv'][instrument]:
                close_prices = info_bus['ohlcv'][instrument]['D1']['close']
                if len(close_prices) > 0:
                    info_bus['prices'][instrument] = float(close_prices[-1])
    
    def _extract_risk_data(self, info_bus: InfoBus, env: Any):
        """Extract risk data ONCE from environment"""
        if not hasattr(env, 'market_state'):
            return
            
        ms = env.market_state
        
        # Calculate risk metrics
        risk_snapshot: RiskSnapshot = {
            'balance': float(ms.balance),
            'equity': float(ms.balance),  # Will be updated with unrealized P&L
            'current_drawdown': float(ms.current_drawdown),
            'max_drawdown': float(getattr(ms, 'peak_drawdown', ms.current_drawdown)),
            'open_positions': [],
            'risk_score': 0.0  # Will be calculated
        }
        
        # Extract positions if available
        if hasattr(env, 'position_manager') and hasattr(env.position_manager, 'open_positions'):
            for symbol, pos_data in env.position_manager.open_positions.items():
                current_price = info_bus['prices'].get(symbol, pos_data.get('price_open', 0))
                entry_price = pos_data.get('price_open', current_price)
                size = pos_data.get('lots', 0)
                side = 'long' if pos_data.get('side', 1) > 0 else 'short'
                
                # Calculate P&L
                if side == 'long':
                    pnl = (current_price - entry_price) * size
                else:
                    pnl = (entry_price - current_price) * size
                
                pos_info: PositionInfo = {
                    'symbol': symbol,
                    'side': side,
                    'size': float(size),
                    'entry_price': float(entry_price),
                    'current_price': float(current_price),
                    'unrealised_pnl': float(pnl),
                    'realised_pnl': 0.0,
                    'duration': info_bus['step_idx'] - pos_data.get('entry_step', info_bus['step_idx']),
                    'margin_used': float(size * current_price * 0.01)  # 1% margin
                }
                
                risk_snapshot['open_positions'].append(pos_info)
        
        # Calculate aggregate risk metrics
        total_margin = sum(pos['margin_used'] for pos in risk_snapshot['open_positions'])
        total_pnl = sum(pos['unrealised_pnl'] for pos in risk_snapshot['open_positions'])
        
        risk_snapshot['margin_used'] = total_margin
        risk_snapshot['equity'] = risk_snapshot['balance'] + total_pnl
        risk_snapshot['free_margin'] = max(0, risk_snapshot['equity'] - total_margin)
        risk_snapshot['margin_level'] = (risk_snapshot['equity'] / max(total_margin, 1)) * 100 if total_margin > 0 else float('inf')
        
        # Calculate risk score (0-1 range)
        dd_score = min(0.5, risk_snapshot['current_drawdown'] / 0.2)  # 20% DD = 0.5
        margin_score = min(0.5, total_margin / risk_snapshot['equity']) if risk_snapshot['equity'] > 0 else 0.5
        risk_snapshot['risk_score'] = dd_score + margin_score  # 0-1 range
        
        info_bus['risk'] = risk_snapshot
        info_bus['positions'] = risk_snapshot['open_positions']
    
    def _calculate_checksum(self, info_bus: InfoBus) -> str:
        """Calculate checksum for data integrity"""
        # Create deterministic string representation
        checksum_data = {
            'step': info_bus['step_idx'],
            'prices': sorted(info_bus.get('prices', {}).items()),
            'positions': len(info_bus.get('positions', [])),
            'risk_score': info_bus.get('risk', {}).get('risk_score', 0)
        }
        
        checksum_str = json.dumps(checksum_data, sort_keys=True)
        return hashlib.md5(checksum_str.encode()).hexdigest()[:8]
    
    @classmethod
    def update_computation_cache(cls, key: str, value: Any, module: str = ""):
        """Update computation cache to prevent redundant calculations"""
        if cls._current_bus is None:
            return
            
        manager = cls.get_instance()
        cache = cls._current_bus['computation_cache']
        
        # Categorize cache entries
        if key.startswith('feature_'):
            cache['features'][key] = value
        elif key.startswith('regime_'):
            cache['regime_detection'][key] = value
        elif key.startswith('pattern_'):
            cache['pattern_analysis'][key] = value
        else:
            # Generic cache
            if 'generic' not in cache:
                cache['generic'] = {}
            cache['generic'][key] = value
        
        manager._computation_cache[key] = value
        manager.logger.debug(f"Cached {key} from {module}")
    
    @classmethod
    def get_cached_computation(cls, key: str, module: str = "") -> Optional[Any]:
        """Get cached computation to prevent redundant calculations"""
        manager = cls.get_instance()
        
        if key in manager._computation_cache:
            manager._cache_hits += 1
            manager.logger.debug(f"Cache hit for {key} requested by {module}")
            return manager._computation_cache[key]
        
        manager._cache_misses += 1
        return None
    
    @classmethod
    def lock_field(cls, field: str):
        """Lock a field to prevent modification"""
        manager = cls.get_instance()
        manager._locked_fields.add(field)
    
    @classmethod
    def register_validator(cls, field: str, validator: Callable[[Any], bool]):
        """Register a validator for a field"""
        manager = cls.get_instance()
        manager._data_validators[field] = validator
    
    @classmethod
    def validate_update(cls, field: str, value: Any) -> bool:
        """Validate field update"""
        manager = cls.get_instance()
        
        # Check if field is locked
        if field in manager._locked_fields:
            manager.logger.warning(f"Attempted to modify locked field: {field}")
            return False
        
        # Run validator if registered
        if field in manager._data_validators:
            if not manager._data_validators[field](value):
                manager.logger.warning(f"Validation failed for field: {field}")
                return False
        
        return True
    
    @classmethod
    def get_performance_summary(cls) -> Dict[str, Any]:
        """Get performance summary"""
        manager = cls.get_instance()
        
        return {
            'cache_hits': manager._cache_hits,
            'cache_misses': manager._cache_misses,
            'cache_hit_rate': manager._cache_hits / max(manager._cache_hits + manager._cache_misses, 1),
            'module_access_log': dict(manager._module_access_log),
            'locked_fields': list(manager._locked_fields)
        }

# ═══════════════════════════════════════════════════════════════════
# INFOBUS DECORATORS
# ═══════════════════════════════════════════════════════════════════

def require_info_bus(func):
    """Decorator to ensure function receives valid InfoBus"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Extract InfoBus from args/kwargs
        info_bus = kwargs.get('info_bus')
        if info_bus is None and len(args) > 1:
            # Check if second arg is InfoBus
            for arg in args[1:]:
                if isinstance(arg, dict) and '_version' in arg:
                    info_bus = arg
                    break
        
        if info_bus is None:
            # Try to get current InfoBus
            info_bus = InfoBusManager.get_current()
            if info_bus is None:
                raise ValueError(f"{func.__name__} requires InfoBus but none provided or available")
            kwargs['info_bus'] = info_bus
        
        # Validate InfoBus
        quality = validate_info_bus(info_bus)
        if not quality.is_valid and quality.score < 50:
            logging.getLogger("InfoBus").warning(
                f"{func.__name__} called with low quality InfoBus: {quality.score:.1f}%"
            )
        
        return func(*args, **kwargs)
    
    return wrapper

def cache_computation(key_prefix: str):
    """Decorator to cache computation results in InfoBus"""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Generate cache key
            module_name = self.__class__.__name__
            cache_key = f"{key_prefix}_{module_name}"
            
            # Check cache
            cached_result = InfoBusManager.get_cached_computation(cache_key, module_name)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(self, *args, **kwargs)
            
            # Cache result
            InfoBusManager.update_computation_cache(cache_key, result, module_name)
            
            return result
        
        return wrapper
    return decorator

# ═══════════════════════════════════════════════════════════════════
# INFOBUS QUALITY SYSTEM
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InfoBusQuality:
    """InfoBus data quality assessment"""
    score: float  # 0-100
    missing_fields: List[str] = field(default_factory=list)
    invalid_values: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_valid: bool = True
    completeness: float = 100.0
    freshness: float = 100.0  # Data freshness score
    
    @property
    def issues(self) -> List[str]:
        """All issues combined"""
        return self.missing_fields + self.invalid_values + self.warnings
    
    @property
    def issue_count(self) -> int:
        """Critical issue count"""
        return len(self.missing_fields) + len(self.invalid_values)
    
    @property
    def quality_score(self) -> float:
        """Legacy compatibility"""
        return self.score

class InfoBusValidator:
    """Enhanced InfoBus validation"""
    
    REQUIRED_FIELDS = ['timestamp', 'step_idx', 'prices', 'risk']
    CRITICAL_NUMERIC_FIELDS = {
        'consensus': (0.0, 1.0),
        'risk_score': (0.0, 1.0),  # Enforce 0-1 range
        'win_rate': (0.0, 1.0),
        'confidence': (0.0, 1.0)
    }
    
    @classmethod
    def validate(cls, info_bus: InfoBus) -> InfoBusQuality:
        """Comprehensive InfoBus validation"""
        missing_fields = []
        invalid_values = []
        warnings = []
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in info_bus or info_bus.get(field) is None:
                missing_fields.append(field)
        
        # Validate numeric fields
        for field, (min_val, max_val) in cls.CRITICAL_NUMERIC_FIELDS.items():
            if field in info_bus:
                value = info_bus[field]
                if isinstance(value, (int, float)):
                    if not np.isfinite(value):
                        invalid_values.append(f"{field}: non-finite")
                    elif not (min_val <= value <= max_val):
                        invalid_values.append(f"{field}: {value} outside [{min_val}, {max_val}]")
        
        # Check risk score specifically
        risk_data = info_bus.get('risk', {})
        if isinstance(risk_data, dict):
            risk_score = risk_data.get('risk_score', 0)
            if not (0 <= risk_score <= 1):
                invalid_values.append(f"risk_score: {risk_score} not in 0-1 range")
        
        # Check data freshness
        if 'timestamp' in info_bus:
            try:
                timestamp = datetime.fromisoformat(info_bus['timestamp'].replace('Z', '+00:00'))
                age_seconds = (datetime.now(timezone.utc) - timestamp).total_seconds()
                if age_seconds > 60:  # More than 1 minute old
                    warnings.append(f"Stale data: {age_seconds:.1f}s old")
            except:
                warnings.append("Invalid timestamp format")
        
        # Calculate scores
        base_score = 100.0
        penalty = len(missing_fields) * 20 + len(invalid_values) * 10 + len(warnings) * 5
        score = max(0, min(100, base_score - penalty))
        
        completeness = (1 - len(missing_fields) / max(len(cls.REQUIRED_FIELDS), 1)) * 100
        
        return InfoBusQuality(
            score=score,
            missing_fields=missing_fields,
            invalid_values=invalid_values,
            warnings=warnings,
            is_valid=score >= 70 and len(missing_fields) == 0,
            completeness=completeness
        )

# ═══════════════════════════════════════════════════════════════════
# ENHANCED INFOBUS EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

class InfoBusExtractor:
    """Enhanced extraction with caching and validation"""
    
    @staticmethod
    @cache_computation("market_regime")
    def get_market_regime(info_bus: InfoBus) -> str:
        """Extract market regime with caching"""
        return info_bus.get('market_context', {}).get('regime', 'unknown')
    
    @staticmethod
    def get_market_data(info_bus: InfoBus, instrument: str, timeframe: str = 'D1') -> Optional[Dict[str, np.ndarray]]:
        """Get market data for instrument/timeframe"""
        ohlcv = info_bus.get('ohlcv', {})
        if instrument in ohlcv and timeframe in ohlcv[instrument]:
            return ohlcv[instrument][timeframe]
        return None
    
    @staticmethod
    def get_latest_prices(info_bus: InfoBus) -> Dict[str, float]:
        """Get latest prices for all instruments"""
        return info_bus.get('prices', {}).copy()
    
    @staticmethod
    def get_cached_features(info_bus: InfoBus, feature_name: str) -> Optional[np.ndarray]:
        """Get cached feature computation"""
        cache = info_bus.get('computation_cache', {})
        return cache.get('features', {}).get(feature_name)
    
    @staticmethod
    def get_risk_score(info_bus: InfoBus) -> float:
        """Get risk score (0-1 range, NOT percentage)"""
        risk_data = info_bus.get('risk', {})
        risk_score = risk_data.get('risk_score', 0.0)
        
        # Ensure it's in 0-1 range
        if risk_score > 1.0:
            logging.getLogger("InfoBus").warning(
                f"Risk score {risk_score} > 1.0, clamping to 1.0"
            )
            risk_score = min(1.0, risk_score)
        
        return float(risk_score)
    
    @staticmethod
    def get_module_timing(info_bus: InfoBus, module: str) -> float:
        """Get module execution timing"""
        perf = info_bus.get('performance', {})
        return perf.get('module_timings', {}).get(module, 0.0)
    
    @staticmethod
    def has_fresh_data(info_bus: InfoBus, max_age_seconds: float = 1.0) -> bool:
        """Check if InfoBus data is fresh"""
        if 'timestamp' not in info_bus:
            return False
        
        try:
            timestamp = datetime.fromisoformat(info_bus['timestamp'].replace('Z', '+00:00'))
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()
            return age <= max_age_seconds
        except:
            return False

# ═══════════════════════════════════════════════════════════════════
# ENHANCED INFOBUS UPDATER
# ═══════════════════════════════════════════════════════════════════

class InfoBusUpdater:
    """Enhanced updater with validation and performance tracking"""
    
    @staticmethod
    def update_market_context(info_bus: InfoBus, regime: str, confidence: float = 0.5):
        """Update market context with validation"""
        if 'market_context' not in info_bus:
            info_bus['market_context'] = {}
        
        # Validate regime
        valid_regimes = ["trending", "volatile", "ranging", "unknown"]
        if regime not in valid_regimes:
            logging.getLogger("InfoBus").warning(f"Invalid regime: {regime}")
            regime = "unknown"
        
        # Validate confidence
        confidence = max(0.0, min(1.0, confidence))
        
        info_bus['market_context']['regime'] = regime
        info_bus['market_context']['regime_confidence'] = confidence
        
        # Track regime duration
        current_regime = info_bus['market_context'].get('regime')
        if current_regime == regime:
            info_bus['market_context']['regime_duration'] = info_bus['market_context'].get('regime_duration', 0) + 1
        else:
            info_bus['market_context']['regime_duration'] = 1
    
    @staticmethod
    def add_performance_timing(info_bus: InfoBus, module: str, timing_ms: float):
        """Add module performance timing"""
        if 'performance' not in info_bus:
            info_bus['performance'] = {
                'module_timings': {},
                'step_compute_time_ms': 0.0
            }
        
        info_bus['performance']['module_timings'][module] = timing_ms
        info_bus['performance']['step_compute_time_ms'] += timing_ms
    
    @staticmethod
    def update_feature(info_bus: InfoBus, feature_name: str, 
                      feature_data: np.ndarray, module: str = ""):
        """Update feature with caching"""
        if 'features' not in info_bus:
            info_bus['features'] = {}
        
        # Validate feature data
        if not isinstance(feature_data, np.ndarray):
            feature_data = np.array(feature_data, dtype=np.float32)
        
        # Ensure finite values
        if not np.all(np.isfinite(feature_data)):
            logging.getLogger("InfoBus").warning(
                f"Non-finite values in feature {feature_name} from {module}"
            )
            feature_data = np.nan_to_num(feature_data, nan=0.0)
        
        info_bus['features'][feature_name] = feature_data
        
        # Update cache
        InfoBusManager.update_computation_cache(f"feature_{feature_name}", feature_data, module)
    
    @staticmethod
    def add_vote(info_bus: InfoBus, vote: Vote):
        """Add vote with timing information"""
        if 'votes' not in info_bus:
            info_bus['votes'] = []
        
        # Ensure required fields
        if 'compute_time_ms' not in vote:
            vote['compute_time_ms'] = 0.0
        
        # Validate confidence
        if 'confidence' in vote:
            vote['confidence'] = max(0.0, min(1.0, float(vote['confidence'])))
        
        info_bus['votes'].append(vote)

# ═══════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def now_utc() -> str:
    """Current UTC ISO8601 timestamp"""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')

def create_info_bus(env: Any, step: int = 0) -> InfoBus:
    """Create InfoBus through centralized manager"""
    return InfoBusManager.create_info_bus(env, step)

def validate_info_bus(info_bus: InfoBus) -> InfoBusQuality:
    """Validate InfoBus quality"""
    return InfoBusValidator.validate(info_bus)

def extract_standard_context(info_bus: InfoBus) -> Dict[str, Any]:
    """Extract commonly used context from InfoBus"""
    return {
        'regime': InfoBusExtractor.get_market_regime(info_bus),
        'volatility_level': info_bus.get('market_context', {}).get('volatility_level', 'medium'),
        'risk_score': InfoBusExtractor.get_risk_score(info_bus),  # 0-1 range
        'position_count': len(info_bus.get('positions', [])),
        'has_fresh_data': InfoBusExtractor.has_fresh_data(info_bus),
        'cache_available': bool(info_bus.get('computation_cache', {}).get('features'))
    }

# ═══════════════════════════════════════════════════════════════════
# MIGRATION HELPERS
# ═══════════════════════════════════════════════════════════════════

def migrate_legacy_call(module_name: str, legacy_kwargs: Dict[str, Any]) -> InfoBus:
    """Helper to migrate legacy module calls to InfoBus"""
    # Get current InfoBus or create minimal one
    info_bus = InfoBusManager.get_current()
    
    if info_bus is None:
        # Create minimal InfoBus for legacy compatibility
        info_bus = {
            'timestamp': now_utc(),
            'step_idx': legacy_kwargs.get('step', 0),
            'prices': legacy_kwargs.get('prices', {}),
            'positions': [],
            'risk': {'risk_score': 0.0}
        }
        
        logging.getLogger("InfoBus").warning(
            f"Legacy call from {module_name} - created minimal InfoBus"
        )
    
    return info_bus