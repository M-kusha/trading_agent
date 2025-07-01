# ─────────────────────────────────────────────────────────────
# File: modules/utils/info_bus.py
# Enhanced InfoBus with utilities to eliminate repetitive code
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal, Optional, Union, Callable
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass
import logging

# ---- Core Per-Trade Info ----
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

class RiskSnapshot(TypedDict, total=False):
    """Current risk metrics"""
    balance: float
    equity: float
    margin_used: float
    max_drawdown: float
    current_drawdown: float
    open_positions: List[PositionInfo]
    var_95: Optional[float]  # Value at Risk
    dd_limit: Optional[float]
    max_exposure: Optional[float]
    correlation_matrix: Optional[Dict[str, float]]
    
# ---- Voting/Consensus Info ----
class Vote(TypedDict):
    """Individual module vote"""
    module: str
    instrument: str
    action: Union[float, np.ndarray]  # Continuous action
    confidence: float
    reasoning: Optional[str]
    
# ---- Market Context ----
class MarketContext(TypedDict):
    """Market regime and conditions"""
    regime: Literal["trending", "volatile", "ranging", "unknown"]
    volatility: Dict[str, float]  # Per instrument
    trend_strength: Dict[str, float]
    volume_profile: Optional[Dict[str, float]]
    news_sentiment: Optional[float]
    
# ---- Global Market Status ----
class MarketStatus(TypedDict, total=False):
    """Market session status"""
    is_open: bool
    session: Literal["asian", "european", "american", "closed"]
    next_open_time: Optional[str]
    next_close_time: Optional[str]
    holiday: Optional[bool]
    liquidity_score: Optional[float]
    
# ---- Central InfoBus Payload ----
class InfoBus(TypedDict, total=False):
    """Central data container for module communication"""
    # Timing
    timestamp: str
    step_idx: int
    episode_idx: int
    
    # Market data
    prices: Dict[str, float]  # Current prices by instrument
    features: Dict[str, np.ndarray]  # Technical indicators
    market_context: MarketContext
    market_status: MarketStatus
    
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
    
    # Alerts & logging
    alerts: List[Dict[str, Any]]
    audit_log: List[str]
    errors: List[str]
    
    # Module specific data
    module_data: Dict[str, Any]


# ═══════════════════════════════════════════════════════════════════
# INFOBUS UTILITIES - Eliminates repetitive code across modules
# ═══════════════════════════════════════════════════════════════════

@dataclass
class InfoBusQuality:
    """InfoBus data quality assessment"""
    score: float  # 0-100
    missing_fields: List[str]
    invalid_values: List[str]
    warnings: List[str]
    is_valid: bool

class InfoBusValidator:
    """Centralized InfoBus validation and quality checking"""
    
    REQUIRED_FIELDS = ['timestamp', 'step_idx', 'episode_idx']
    NUMERIC_FIELDS = ['consensus', 'pnl_today', 'trade_count', 'win_rate']
    
    @classmethod
    def validate(cls, info_bus: InfoBus) -> InfoBusQuality:
        """Comprehensive InfoBus validation"""
        score = 100.0
        missing_fields = []
        invalid_values = []
        warnings = []
        
        # Check required fields
        for field in cls.REQUIRED_FIELDS:
            if field not in info_bus:
                missing_fields.append(field)
                score -= 15
        
        # Check numeric field ranges
        for field in cls.NUMERIC_FIELDS:
            if field in info_bus:
                value = info_bus[field]
                if isinstance(value, (int, float)):
                    if not np.isfinite(value):
                        invalid_values.append(f"{field}: non-finite value")
                        score -= 10
                    elif field == 'consensus' and not (0 <= value <= 1):
                        invalid_values.append(f"{field}: out of range [0,1]")
                        score -= 5
        
        # Check data consistency
        if 'recent_trades' in info_bus and 'trade_count' in info_bus:
            actual_trades = len(info_bus['recent_trades'])
            reported_count = info_bus['trade_count']
            if actual_trades != reported_count:
                warnings.append(f"Trade count mismatch: actual={actual_trades}, reported={reported_count}")
                score -= 5
        
        # Check positions data integrity
        if 'positions' in info_bus:
            for i, pos in enumerate(info_bus['positions']):
                if not isinstance(pos.get('size', 0), (int, float)):
                    invalid_values.append(f"positions[{i}].size: non-numeric")
                    score -= 3
                    
        is_valid = score >= 70  # 70% threshold for validity
        
        return InfoBusQuality(
            score=score,
            missing_fields=missing_fields,
            invalid_values=invalid_values,
            warnings=warnings,
            is_valid=is_valid
        )


class InfoBusExtractor:
    """Standard extraction patterns used across modules"""
    
    @staticmethod
    def get_safe_numeric(info_bus: InfoBus, key: str, default: float = 0.0) -> float:
        """Safely extract numeric value with validation"""
        value = info_bus.get(key, default)
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
        return default
    
    @staticmethod
    def get_market_regime(info_bus: InfoBus) -> str:
        """Extract market regime with fallback"""
        return info_bus.get('market_context', {}).get('regime', 'unknown')
    
    @staticmethod
    def get_session(info_bus: InfoBus) -> str:
        """Extract trading session with fallback"""
        return info_bus.get('market_status', {}).get('session', 'unknown')
    
    @staticmethod
    def get_volatility_level(info_bus: InfoBus) -> str:
        """Categorize market volatility"""
        volatilities = info_bus.get('market_context', {}).get('volatility', {})
        if not volatilities:
            return "unknown"
            
        avg_vol = np.mean(list(volatilities.values()))
        if avg_vol > 0.03:
            return "high"
        elif avg_vol > 0.015:
            return "medium"
        else:
            return "low"
    
    @staticmethod
    def get_exposure_pct(info_bus: InfoBus) -> float:
        """Calculate exposure as percentage of equity"""
        risk_data = info_bus.get('risk', {})
        equity = risk_data.get('equity', 1)
        margin_used = risk_data.get('margin_used', 0)
        return (margin_used / max(equity, 1)) * 100
    
    @staticmethod
    def get_drawdown_pct(info_bus: InfoBus) -> float:
        """Get drawdown as percentage"""
        return info_bus.get('risk', {}).get('current_drawdown', 0) * 100
    
    @staticmethod
    def get_position_count(info_bus: InfoBus) -> int:
        """Get number of open positions"""
        return len(info_bus.get('positions', []))
    
    @staticmethod
    def get_recent_trades_count(info_bus: InfoBus) -> int:
        """Get number of recent trades"""
        return len(info_bus.get('recent_trades', []))
    
    @staticmethod
    def get_alert_count(info_bus: InfoBus) -> int:
        """Get number of active alerts"""
        return len(info_bus.get('alerts', []))
    
    @staticmethod
    def get_risk_score(info_bus: InfoBus) -> float:
        """Calculate overall risk score (0-100)"""
        dd_pct = InfoBusExtractor.get_drawdown_pct(info_bus)
        exposure_pct = InfoBusExtractor.get_exposure_pct(info_bus)
        
        # Simple risk scoring
        dd_score = min(50, dd_pct / 0.2 * 50)  # 20% dd = 50 points
        exposure_score = min(50, exposure_pct / 80 * 50)  # 80% exposure = 50 points
        
        return dd_score + exposure_score
    
    @staticmethod
    def get_votes_summary(info_bus: InfoBus) -> Dict[str, Any]:
        """Extract and summarize voting data"""
        votes = info_bus.get('votes', [])
        
        if not votes:
            return {
                'total_votes': 0,
                'avg_confidence': 0.0,
                'consensus_direction': 'neutral',
                'top_modules': []
            }
        
        # Calculate metrics
        confidences = [
            v.get('confidence', 0) for v in votes 
            if isinstance(v.get('confidence'), (int, float))
        ]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0
        
        # Direction analysis
        directions = []
        for vote in votes:
            action = vote.get('action', 0)
            if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                directions.append(np.sign(action[0]))
            else:
                directions.append(np.sign(float(action)) if action else 0)
        
        consensus_direction = "neutral"
        if directions:
            avg_dir = np.mean(directions)
            if avg_dir > 0.3:
                consensus_direction = "bullish"
            elif avg_dir < -0.3:
                consensus_direction = "bearish"
        
        # Top modules by confidence
        sorted_votes = sorted(votes, key=lambda v: v.get('confidence', 0), reverse=True)
        top_modules = [v.get('module', 'unknown') for v in sorted_votes[:3]]
        
        return {
            'total_votes': len(votes),
            'avg_confidence': avg_confidence,
            'consensus_direction': consensus_direction,
            'top_modules': top_modules,
            'agreement_score': InfoBusExtractor._calculate_agreement_score(votes)
        }
    
    @staticmethod
    def _calculate_agreement_score(votes: List[Dict]) -> float:
        """Calculate voting agreement (0-1 score)"""
        if len(votes) < 2:
            return 1.0
            
        actions = []
        for vote in votes:
            action = vote.get('action', 0)
            if isinstance(action, (list, np.ndarray)) and len(action) > 0:
                actions.append(action[0])
            else:
                actions.append(float(action) if action else 0)
        
        if not actions:
            return 0.0
            
        # Lower std dev = higher agreement
        std_dev = np.std(actions)
        agreement = max(0, 1 - std_dev)
        return float(agreement)


class InfoBusUpdater:
    """Standard patterns for updating InfoBus data"""
    
    @staticmethod
    def add_module_data(info_bus: InfoBus, module_name: str, data: Dict[str, Any]):
        """Add module-specific data to InfoBus"""
        if 'module_data' not in info_bus:
            info_bus['module_data'] = {}
        info_bus['module_data'][module_name] = data
    
    @staticmethod
    def add_alert(info_bus: InfoBus, message: str, severity: str = "info", 
                  module: str = "system"):
        """Add alert to InfoBus"""
        if 'alerts' not in info_bus:
            info_bus['alerts'] = []
            
        alert = {
            'timestamp': now_utc(),
            'message': message,
            'severity': severity,
            'module': module
        }
        info_bus['alerts'].append(alert)
    
    @staticmethod
    def add_compliance_flag(info_bus: InfoBus, flag: str):
        """Add compliance flag to InfoBus"""
        if 'compliance_flags' not in info_bus:
            info_bus['compliance_flags'] = []
        if flag not in info_bus['compliance_flags']:
            info_bus['compliance_flags'].append(flag)
    
    @staticmethod
    def update_performance_metrics(info_bus: InfoBus, metrics: Dict[str, float]):
        """Update performance metrics in InfoBus"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                info_bus[key] = float(value)
    
    @staticmethod
    def update_risk_snapshot(info_bus: InfoBus, risk_data: Dict[str, Any]):
        """Update risk snapshot in InfoBus"""
        if 'risk' not in info_bus:
            info_bus['risk'] = {}
        info_bus['risk'].update(risk_data)


class InfoBusBuilder:
    """Builder pattern for creating InfoBus from environment state"""
    
    def __init__(self, env):
        self.env = env
        self.info_bus: InfoBus = {}
    
    def add_timing(self, step: int = 0) -> 'InfoBusBuilder':
        """Add timing information"""
        self.info_bus.update({
            'timestamp': now_utc(),
            'step_idx': step,
            'episode_idx': getattr(self.env, 'episode_count', 0)
        })
        return self
    
    def add_market_data(self) -> 'InfoBusBuilder':
        """Add market prices and features"""
        prices = {}
        
        # Add current prices
        for inst in getattr(self.env, 'instruments', []):
            if hasattr(self.env, 'data') and inst in self.env.data:
                df = self.env.data[inst]["D1"]
                if self.env.market_state.current_step < len(df):
                    prices[inst] = float(df.iloc[self.env.market_state.current_step]['close'])
        
        self.info_bus['prices'] = prices
        self.info_bus['features'] = {}  # To be populated by feature modules
        return self
    
    def add_positions(self) -> 'InfoBusBuilder':
        """Add position information"""
        positions = []
        
        if hasattr(self.env, 'position_manager'):
            for inst, pos_data in self.env.position_manager.open_positions.items():
                pos_info: PositionInfo = {
                    'symbol': inst,
                    'side': 'long' if pos_data.get('side', 0) > 0 else 'short',
                    'size': float(pos_data.get('lots', 0)),
                    'entry_price': float(pos_data.get('price_open', 0)),
                    'current_price': self.info_bus.get('prices', {}).get(inst, 0),
                    'unrealised_pnl': 0.0,  # To be calculated
                    'realised_pnl': 0.0,
                    'duration': self.env.market_state.current_step - pos_data.get('entry_step', 0)
                }
                positions.append(pos_info)
        
        self.info_bus['positions'] = positions
        return self
    
    def add_risk_data(self) -> 'InfoBusBuilder':
        """Add risk snapshot"""
        risk_snapshot: RiskSnapshot = {
            'balance': self.env.market_state.balance,
            'equity': self.env.market_state.balance,
            'max_drawdown': self.env.market_state.current_drawdown,
            'current_drawdown': self.env.market_state.current_drawdown,
            'open_positions': self.info_bus.get('positions', [])
        }
        
        # Add margin calculation if position manager available
        if hasattr(self.env, 'position_manager'):
            margin_used = sum(
                pos.get('margin', 0) 
                for pos in self.env.position_manager.open_positions.values()
            )
            risk_snapshot['margin_used'] = margin_used
        
        self.info_bus['risk'] = risk_snapshot
        return self
    
    def add_defaults(self) -> 'InfoBusBuilder':
        """Add default empty values for optional fields"""
        defaults = {
            'recent_trades': [],
            'pending_orders': [],
            'votes': [],
            'alerts': [],
            'compliance_flags': [],
            'errors': [],
            'module_data': {}
        }
        
        for key, default_value in defaults.items():
            if key not in self.info_bus:
                self.info_bus[key] = default_value
        
        return self
    
    def build(self) -> InfoBus:
        """Build and validate the InfoBus"""
        quality = InfoBusValidator.validate(self.info_bus)
        
        if not quality.is_valid:
            logger = logging.getLogger("InfoBusBuilder")
            logger.warning(f"InfoBus quality issues: {quality.warnings}")
        
        return self.info_bus


# ---- Utility Functions ----
def now_utc() -> str:
    """Current UTC ISO8601 timestamp"""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')

def create_info_bus(env: Any, step: int = 0) -> InfoBus:
    """Enhanced InfoBus creation with full validation"""
    return (InfoBusBuilder(env)
            .add_timing(step)
            .add_market_data()
            .add_positions()
            .add_risk_data()
            .add_defaults()
            .build())

def validate_info_bus(info_bus: InfoBus) -> InfoBusQuality:
    """Validate InfoBus quality"""
    return InfoBusValidator.validate(info_bus)

def extract_standard_context(info_bus: InfoBus) -> Dict[str, Any]:
    """Extract commonly used context from InfoBus"""
    return {
        'regime': InfoBusExtractor.get_market_regime(info_bus),
        'session': InfoBusExtractor.get_session(info_bus),
        'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
        'exposure_pct': InfoBusExtractor.get_exposure_pct(info_bus),
        'drawdown_pct': InfoBusExtractor.get_drawdown_pct(info_bus),
        'position_count': InfoBusExtractor.get_position_count(info_bus),
        'risk_score': InfoBusExtractor.get_risk_score(info_bus),
        'consensus': InfoBusExtractor.get_safe_numeric(info_bus, 'consensus', 0.5),
        'votes_summary': InfoBusExtractor.get_votes_summary(info_bus)
    }

# ---- Common processing patterns ----
def process_recent_trades(info_bus: InfoBus, 
                         processor: Callable[[TradeInfo, Dict[str, Any]], Any]) -> List[Any]:
    """Standard pattern for processing recent trades"""
    context = extract_standard_context(info_bus)
    results = []
    
    for trade in info_bus.get('recent_trades', []):
        try:
            result = processor(trade, context)
            if result is not None:
                results.append(result)
        except Exception as e:
            logging.getLogger("InfoBus").warning(f"Trade processing failed: {e}")
    
    return results

def apply_info_bus_middleware(info_bus: InfoBus, 
                             middleware_funcs: List[Callable[[InfoBus], InfoBus]]) -> InfoBus:
    """Apply middleware functions to InfoBus"""
    for middleware in middleware_funcs:
        try:
            info_bus = middleware(info_bus)
        except Exception as e:
            logging.getLogger("InfoBus").error(f"Middleware failed: {e}")
    return info_bus