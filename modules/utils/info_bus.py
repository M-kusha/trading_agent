
# modules/utils/info_bus.py

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal, Optional, Union
from datetime import datetime, timezone
import numpy as np

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
    
# ---- Utility Functions ----
def now_utc() -> str:
    """Current UTC ISO8601 timestamp"""
    return datetime.now(timezone.utc).isoformat(timespec='seconds')
    
def create_info_bus(env: Any, step: int = 0) -> InfoBus:
    """Create InfoBus from environment state"""
    info_bus: InfoBus = {
        'timestamp': now_utc(),
        'step_idx': step,
        'episode_idx': getattr(env, 'episode_count', 0),
        'prices': {},
        'features': {},
        'positions': [],
        'risk': {
            'balance': env.market_state.balance,
            'equity': env.market_state.balance,
            'max_drawdown': env.market_state.current_drawdown,
            'current_drawdown': env.market_state.current_drawdown,
            'open_positions': []
        },
        'module_data': {}
    }
    
    # Add prices
    for inst in env.instruments:
        df = env.data[inst]["D1"]
        if env.market_state.current_step < len(df):
            info_bus['prices'][inst] = float(df.iloc[env.market_state.current_step]['close'])
            
    # Add positions
    for inst, pos_data in env.position_manager.open_positions.items():
        pos_info: PositionInfo = {
            'symbol': inst,
            'side': 'long' if pos_data.get('side', 0) > 0 else 'short',
            'size': float(pos_data.get('lots', 0)),
            'entry_price': float(pos_data.get('price_open', 0)),
            'current_price': info_bus['prices'].get(inst, 0),
            'unrealised_pnl': 0.0,  # Calculate based on current price
            'realised_pnl': 0.0,
            'duration': env.market_state.current_step - pos_data.get('entry_step', 0)
        }
        info_bus['positions'].append(pos_info)
        
    return info_bus