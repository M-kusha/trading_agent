"""
Position Manager Debugger
Production-ready debugging system with plain English explanations
"""

import json
import datetime
import time
import traceback
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import threading
from collections import deque, defaultdict
import numpy as np


# -------------------------------------------------------------
# Data Models
# -------------------------------------------------------------

class ActionType(Enum):
    """Clear action types for debugging"""
    BUY = "BUY 📈"
    SELL = "SELL 📉"
    HOLD = "HOLD ⏸️"
    SCALE_UP = "ADD_MORE 📊"
    SCALE_DOWN = "REDUCE 📉"
    CLOSE_POSITION = "CLOSE 🔒"
    EMERGENCY_EXIT = "EMERGENCY 🚨"


class DebugLevel(Enum):
    """Debug levels with visual indicators"""
    TRACE = ("TRACE", "🔍", 0)
    DEBUG = ("DEBUG", "🐛", 10)
    INFO = ("INFO", "ℹ️", 20)
    SUCCESS = ("SUCCESS", "✅", 25)
    WARNING = ("WARNING", "⚠️", 30)
    ERROR = ("ERROR", "❌", 40)
    CRITICAL = ("CRITICAL", "🚨", 50)
    
    def __init__(self, label: str, icon: str, priority: int):
        self.label = label
        self.icon = icon
        self.priority = priority


@dataclass
class DecisionSnapshot:
    """Complete snapshot of a position decision"""
    timestamp: str
    instrument: str
    action: ActionType
    is_buying: bool  # True for buy/long, False for sell/short
    direction: str  # "LONG" or "SHORT" or "NEUTRAL"
    size_eur: float
    confidence: float
    market_intensity: float
    current_price: float
    
    # Decision factors
    signal_strength: float
    trend_direction: str
    volatility: float
    portfolio_health: float
    risk_score: float
    
    # Rationale in plain English
    plain_english_reason: str
    technical_factors: List[str]
    risk_factors: List[str]
    
    # Execution details
    will_execute: bool
    execution_blocked_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['action'] = self.action.value
        return data
    
    def to_plain_english(self) -> str:
        """Generate plain English explanation"""
        if self.is_buying:
            action_word = "BUYING" if self.action == ActionType.BUY else "ADDING TO LONG"
        elif self.action == ActionType.SELL:
            action_word = "SELLING (SHORT)"
        elif self.action == ActionType.CLOSE_POSITION:
            action_word = "CLOSING POSITION"
        else:
            action_word = str(self.action.value)
        
        return f"""
📍 {self.instrument} - {action_word}
💰 Size: €{self.size_eur:,.2f}
📊 Confidence: {self.confidence:.1%}
📈 Signal: {self.signal_strength:.2f} ({self.trend_direction})
⚡ Volatility: {self.volatility:.3f}
🏥 Portfolio Health: {self.portfolio_health:.1%}
⚠️ Risk Score: {self.risk_score:.1%}

📝 Reason: {self.plain_english_reason}

Technical Factors:
{chr(10).join(f'  • {f}' for f in self.technical_factors)}

Risk Considerations:
{chr(10).join(f'  • {f}' for f in self.risk_factors)}

Status: {'✅ WILL EXECUTE' if self.will_execute else f'❌ BLOCKED: {self.execution_blocked_reason}'}
"""


@dataclass
class ErrorSnapshot:
    """Detailed error information"""
    timestamp: str
    error_type: str
    error_message: str
    component: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_action: Optional[str] = None
    
    def to_plain_english(self) -> str:
        return f"""
🚨 ERROR DETECTED
Time: {self.timestamp}
Component: {self.component}
Type: {self.error_type}

What went wrong:
{self.error_message}

Where it happened:
{self.stack_trace[:500]}...

Context:
{json.dumps(self.context, indent=2)[:500]}...

Recovery: {self.recovery_action or 'Manual intervention needed'}
"""


# -------------------------------------------------------------
# Main Debugger Class
# -------------------------------------------------------------

class PositionDebugger:
    """
    Comprehensive debugging system for Position Manager
    Tracks all decisions, errors, and provides plain English explanations
    """
    
    def __init__(self, 
                 log_dir: str = "logs/debug",
                 enable_console: bool = True,
                 enable_file: bool = True,
                 max_memory_items: int = 10000):
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.enable_console = enable_console
        self.enable_file = enable_file
        self.max_memory_items = max_memory_items
        
        # Thread safety
        self._lock = threading.RLock()
        
        # In-memory storage
        self.decision_history: deque = deque(maxlen=max_memory_items)
        self.error_history: deque = deque(maxlen=1000)
        self.performance_metrics: deque = deque(maxlen=5000)
        
        # Statistics
        self.stats = {
            "total_decisions": 0,
            "buy_decisions": 0,
            "sell_decisions": 0,
            "hold_decisions": 0,
            "successful_executions": 0,
            "blocked_executions": 0,
            "total_errors": 0,
            "start_time": datetime.datetime.utcnow(),
        }
        
        # File handles
        self._init_log_files()
        
    def _init_log_files(self):
        """Initialize log files with headers"""
        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Main debug log
        self.debug_file = self.log_dir / f"debug_{timestamp}.log"
        
        # Decision log (CSV-like for analysis)
        self.decision_file = self.log_dir / f"decisions_{timestamp}.csv"
        
        # Error log
        self.error_file = self.log_dir / f"errors_{timestamp}.log"
        
        # Plain English summary
        self.summary_file = self.log_dir / f"summary_{timestamp}.txt"
        
        # Initialize CSV header for decisions
        if self.enable_file:
            with open(self.decision_file, 'w') as f:
                f.write("timestamp,instrument,action,is_buying,direction,size_eur,confidence,"
                       "signal_strength,volatility,portfolio_health,risk_score,executed,reason\n")
    
    # -------------------------------------------------------------
    # Core Logging Methods
    # -------------------------------------------------------------
    
    def log_decision(self, 
                    instrument: str,
                    decision: str,
                    intensity: float,
                    size: float,
                    confidence: float,
                    context: Dict[str, Any],
                    rationale: Dict[str, Any]) -> DecisionSnapshot:
        """
        Log a position decision with full context
        
        Args:
            instrument: Trading pair (e.g., "EUR/USD")
            decision: Decision type (open_long, open_short, etc.)
            intensity: Signal intensity (-1 to 1)
            size: Position size in EUR
            confidence: Confidence level (0 to 1)
            context: Market context dictionary
            rationale: Decision rationale
        
        Returns:
            DecisionSnapshot with all details
        """
        
        # Determine action type and direction
        action, is_buying, direction = self._parse_decision(decision, intensity)
        
        # Extract context details
        signal_strength = abs(intensity)
        trend_direction = "BULLISH" if intensity > 0 else "BEARISH" if intensity < 0 else "NEUTRAL"
        volatility = context.get('volatility', 0.0)
        portfolio_health = context.get('portfolio_health', 1.0)
        risk_score = self._calculate_risk_score(context)
        current_price = context.get('current_price', 0.0)
        
        # Generate plain English explanation
        plain_english = self._generate_plain_english_reason(
            action, is_buying, instrument, signal_strength, 
            trend_direction, confidence, rationale
        )
        
        # Technical factors
        technical_factors = self._extract_technical_factors(context)
        
        # Risk factors
        risk_factors = self._extract_risk_factors(context, rationale.get('risk_factors', {}))
        
        # Execution check
        will_execute, blocked_reason = self._check_execution_viability(
            size, confidence, context
        )
        
        # Create snapshot
        snapshot = DecisionSnapshot(
            timestamp=datetime.datetime.utcnow().isoformat(),
            instrument=instrument,
            action=action,
            is_buying=is_buying,
            direction=direction,
            size_eur=size,
            confidence=confidence,
            market_intensity=intensity,
            current_price=current_price,
            signal_strength=signal_strength,
            trend_direction=trend_direction,
            volatility=volatility,
            portfolio_health=portfolio_health,
            risk_score=risk_score,
            plain_english_reason=plain_english,
            technical_factors=technical_factors,
            risk_factors=risk_factors,
            will_execute=will_execute,
            execution_blocked_reason=blocked_reason
        )
        
        # Store and log
        with self._lock:
            self.decision_history.append(snapshot)
            self._update_stats(snapshot)
            self._write_decision_to_file(snapshot)
            self._write_to_summary(snapshot)
            
            if self.enable_console:
                self._print_decision(snapshot)
        
        return snapshot
    
    def log_error(self,
                 error: Exception,
                 component: str,
                 context: Dict[str, Any],
                 recovery_action: Optional[str] = None) -> ErrorSnapshot:
        """
        Log an error with full context
        
        Args:
            error: The exception that occurred
            component: Component where error occurred
            context: Context dictionary
            recovery_action: Optional recovery action taken
        
        Returns:
            ErrorSnapshot with details
        """
        
        snapshot = ErrorSnapshot(
            timestamp=datetime.datetime.utcnow().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            component=component,
            stack_trace=traceback.format_exc(),
            context=context,
            recovery_action=recovery_action
        )
        
        with self._lock:
            self.error_history.append(snapshot)
            self.stats["total_errors"] += 1
            
            if self.enable_file:
                self._write_error_to_file(snapshot)
            
            if self.enable_console:
                self._print_error(snapshot)
        
        return snapshot
    
    def log_metric(self,
                  metric_name: str,
                  value: float,
                  unit: str = "",
                  context: Optional[Dict[str, Any]] = None):
        """Log a performance metric"""
        
        metric = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "name": metric_name,
            "value": value,
            "unit": unit,
            "context": context or {}
        }
        
        with self._lock:
            self.performance_metrics.append(metric)
            
            if self.enable_file:
                self._write_metric_to_file(metric)
    
    # -------------------------------------------------------------
    # Decision Analysis
    # -------------------------------------------------------------
    
    def _parse_decision(self, decision: str, intensity: float) -> Tuple[ActionType, bool, str]:
        """Parse decision into action type, buying flag, and direction"""
        
        decision_lower = decision.lower()
        
        if "open_long" in decision_lower:
            return ActionType.BUY, True, "LONG"
        elif "open_short" in decision_lower:
            return ActionType.SELL, False, "SHORT"
        elif "scale_up" in decision_lower:
            # Determine from intensity sign
            is_buying = intensity > 0
            return ActionType.SCALE_UP, is_buying, "LONG" if is_buying else "SHORT"
        elif "scale_down" in decision_lower:
            return ActionType.SCALE_DOWN, False, "REDUCING"
        elif "close" in decision_lower:
            if "emergency" in decision_lower:
                return ActionType.EMERGENCY_EXIT, False, "EXIT"
            return ActionType.CLOSE_POSITION, False, "CLOSING"
        else:
            return ActionType.HOLD, False, "NEUTRAL"
    
    def _generate_plain_english_reason(self,
                                      action: ActionType,
                                      is_buying: bool,
                                      instrument: str,
                                      signal_strength: float,
                                      trend: str,
                                      confidence: float,
                                      rationale: Dict[str, Any]) -> str:
        """Generate plain English explanation for the decision"""
        
        stage = rationale.get('stage', 'unknown')
        factors = rationale.get('factors', [])
        
        if action == ActionType.BUY:
            reason = f"Opening a new LONG position on {instrument} because "
            reason += f"the market shows a strong {trend} signal ({signal_strength:.2f}) "
            reason += f"with {confidence:.1%} confidence. "
        elif action == ActionType.SELL:
            reason = f"Opening a new SHORT position on {instrument} because "
            reason += f"the market shows a strong {trend} signal ({signal_strength:.2f}) "
            reason += f"with {confidence:.1%} confidence. "
        elif action == ActionType.SCALE_UP:
            reason = f"Adding to existing position on {instrument} because "
            reason += f"the trend continues to be {trend} and aligns with our position. "
        elif action == ActionType.SCALE_DOWN:
            reason = f"Reducing position on {instrument} to manage risk "
            reason += f"as market conditions have changed. "
        elif action == ActionType.CLOSE_POSITION:
            reason = f"Closing position on {instrument} "
            if "risk" in stage:
                reason += "due to risk management rules. "
            elif "reverse" in stage:
                reason += "because market has reversed against our position. "
            else:
                reason += "to lock in results. "
        elif action == ActionType.EMERGENCY_EXIT:
            reason = f"EMERGENCY EXIT from {instrument} position "
            reason += "due to critical risk conditions detected! "
        else:
            reason = f"Holding position on {instrument} - no action needed. "
            reason += "Market signals are not strong enough for a trade. "
        
        # Add specific factors
        if factors:
            reason += "Key factors: " + "; ".join(factors[:3])
        
        return reason
    
    def _extract_technical_factors(self, context: Dict[str, Any]) -> List[str]:
        """Extract technical factors in plain English"""
        
        factors = []
        
        # Trend
        trend_strength = context.get('trend_strength', 0)
        if abs(trend_strength) > 0.5:
            direction = "uptrend" if trend_strength > 0 else "downtrend"
            factors.append(f"Strong {direction} detected (strength: {abs(trend_strength):.2f})")
        
        # Momentum
        momentum = context.get('momentum', 0)
        if abs(momentum) > 0.3:
            momentum_dir = "positive" if momentum > 0 else "negative"
            factors.append(f"Momentum is {momentum_dir} ({momentum:.2f})")
        
        # RSI
        rsi = context.get('rsi', 50)
        if rsi > 70:
            factors.append(f"Market is overbought (RSI: {rsi:.0f})")
        elif rsi < 30:
            factors.append(f"Market is oversold (RSI: {rsi:.0f})")
        
        # Volatility
        volatility = context.get('volatility', 0.02)
        if volatility > 0.04:
            factors.append(f"High volatility detected ({volatility:.3f})")
        elif volatility < 0.01:
            factors.append(f"Low volatility environment ({volatility:.3f})")
        
        # Volume
        volume_profile = context.get('volume_profile', 1.0)
        if volume_profile > 1.5:
            factors.append(f"High trading volume ({volume_profile:.1f}x normal)")
        elif volume_profile < 0.5:
            factors.append(f"Low trading volume ({volume_profile:.1f}x normal)")
        
        return factors if factors else ["Normal market conditions"]
    
    def _extract_risk_factors(self, 
                             context: Dict[str, Any],
                             risk_factors: Dict[str, float]) -> List[str]:
        """Extract risk factors in plain English"""
        
        factors = []
        
        # Drawdown
        drawdown = context.get('drawdown', 0)
        if drawdown > 0.05:
            factors.append(f"Portfolio drawdown at {drawdown:.1%}")
        
        # Exposure
        exposure = context.get('current_exposure', 0)
        if exposure > 0.5:
            factors.append(f"High exposure level ({exposure:.1%} of capital)")
        
        # Correlation
        correlation = risk_factors.get('correlation', 0)
        if correlation > 0.5:
            factors.append(f"High correlation with other positions ({correlation:.1%})")
        
        # Session risk
        session = context.get('session', 'unknown')
        if session == 'closed':
            factors.append("Market is closed - higher spread risk")
        elif session == 'asian':
            factors.append("Asian session - lower liquidity")
        
        # Volatility risk
        vol_risk = risk_factors.get('volatility', 0)
        if vol_risk > 0.3:
            factors.append(f"Elevated volatility risk ({vol_risk:.1%})")
        
        return factors if factors else ["Risk levels are acceptable"]
    
    def _calculate_risk_score(self, context: Dict[str, Any]) -> float:
        """Calculate overall risk score"""
        
        drawdown = context.get('drawdown', 0)
        exposure = context.get('current_exposure', 0)
        volatility = context.get('volatility', 0.02)
        
        # Normalize components
        drawdown_risk = min(drawdown * 5, 1.0)  # 20% drawdown = max risk
        exposure_risk = min(exposure * 2, 1.0)   # 50% exposure = max risk
        vol_risk = min(volatility / 0.05, 1.0)   # 5% volatility = max risk
        
        # Weighted average
        risk_score = (drawdown_risk * 0.4 + exposure_risk * 0.3 + vol_risk * 0.3)
        
        return min(risk_score, 1.0)
    
    def _check_execution_viability(self,
                                  size: float,
                                  confidence: float,
                                  context: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check if execution should proceed"""
        
        # Size check
        if size <= 0:
            return False, "Position size is zero or negative"
        
        balance = context.get('balance', 0)
        if size > balance * 0.5:
            return False, f"Position size (€{size:.2f}) exceeds 50% of balance"
        
        # Confidence check
        if confidence < 0.3:
            return False, f"Confidence too low ({confidence:.1%})"
        
        # Risk check
        risk_score = self._calculate_risk_score(context)
        if risk_score > 0.8:
            return False, f"Risk score too high ({risk_score:.1%})"
        
        # Drawdown check
        drawdown = context.get('drawdown', 0)
        if drawdown > 0.15:
            return False, f"Drawdown limit exceeded ({drawdown:.1%})"
        
        return True, None
    
    # -------------------------------------------------------------
    # File Writing
    # -------------------------------------------------------------
    
    def _write_decision_to_file(self, snapshot: DecisionSnapshot):
        """Write decision to CSV file"""
        if not self.enable_file:
            return
        
        try:
            with open(self.decision_file, 'a') as f:
                f.write(f"{snapshot.timestamp},{snapshot.instrument},{snapshot.action.value},"
                       f"{snapshot.is_buying},{snapshot.direction},{snapshot.size_eur:.2f},"
                       f"{snapshot.confidence:.3f},{snapshot.signal_strength:.3f},"
                       f"{snapshot.volatility:.4f},{snapshot.portfolio_health:.3f},"
                       f"{snapshot.risk_score:.3f},{snapshot.will_execute},"
                       f'"{snapshot.plain_english_reason}"\n')
        except Exception as e:
            print(f"Error writing decision to file: {e}")
    
    def _write_to_summary(self, snapshot: DecisionSnapshot):
        """Write plain English summary"""
        if not self.enable_file:
            return
        
        try:
            with open(self.summary_file, 'a') as f:
                f.write("=" * 80 + "\n")
                f.write(snapshot.to_plain_english())
                f.write("\n")
        except Exception as e:
            print(f"Error writing summary: {e}")
    
    def _write_error_to_file(self, snapshot: ErrorSnapshot):
        """Write error to file"""
        if not self.enable_file:
            return
        
        try:
            with open(self.error_file, 'a') as f:
                f.write("=" * 80 + "\n")
                f.write(snapshot.to_plain_english())
                f.write("\n")
        except Exception as e:
            print(f"Error writing error log: {e}")
    
    def _write_metric_to_file(self, metric: Dict[str, Any]):
        """Write metric to debug log"""
        if not self.enable_file:
            return
        
        try:
            with open(self.debug_file, 'a') as f:
                f.write(f"[METRIC] {metric['timestamp']} - {metric['name']}: "
                       f"{metric['value']:.4f} {metric['unit']}\n")
        except Exception as e:
            print(f"Error writing metric: {e}")
    
    # -------------------------------------------------------------
    # Console Output
    # -------------------------------------------------------------
    
    def _print_decision(self, snapshot: DecisionSnapshot):
        """Print decision to console with colors"""
        
        # Color codes
        GREEN = '\033[92m'
        RED = '\033[91m'
        YELLOW = '\033[93m'
        BLUE = '\033[94m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        # Choose color based on action
        if snapshot.is_buying:
            color = GREEN
        elif snapshot.action in [ActionType.SELL, ActionType.CLOSE_POSITION]:
            color = RED
        elif snapshot.action == ActionType.EMERGENCY_EXIT:
            color = RED + BOLD
        else:
            color = YELLOW
        
        print(f"\n{color}{'='*80}{RESET}")
        print(f"{color}{BOLD}📍 DECISION: {snapshot.action.value} - {snapshot.instrument}{RESET}")
        print(f"{color}{'='*80}{RESET}")
        
        if snapshot.is_buying:
            print(f"{GREEN}🟢 BUYING/LONG POSITION{RESET}")
        elif snapshot.action == ActionType.SELL:
            print(f"{RED}🔴 SELLING/SHORT POSITION{RESET}")
        
        print(f"💰 Size: €{snapshot.size_eur:,.2f}")
        print(f"📊 Confidence: {snapshot.confidence:.1%}")
        print(f"📈 Signal Strength: {snapshot.signal_strength:.3f}")
        print(f"⚡ Volatility: {snapshot.volatility:.4f}")
        print(f"🏥 Portfolio Health: {snapshot.portfolio_health:.1%}")
        
        if snapshot.will_execute:
            print(f"{GREEN}✅ WILL EXECUTE{RESET}")
        else:
            print(f"{RED}❌ BLOCKED: {snapshot.execution_blocked_reason}{RESET}")
        
        print(f"\n📝 {BLUE}Reason:{RESET} {snapshot.plain_english_reason}")
        print(f"{color}{'='*80}{RESET}\n")
    
    def _print_error(self, snapshot: ErrorSnapshot):
        """Print error to console"""
        
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        print(f"\n{RED}{BOLD}{'!'*80}{RESET}")
        print(f"{RED}{BOLD}🚨 ERROR in {snapshot.component}{RESET}")
        print(f"{RED}{'!'*80}{RESET}")
        print(f"{RED}Type: {snapshot.error_type}{RESET}")
        print(f"{RED}Message: {snapshot.error_message}{RESET}")
        print(f"{RED}Recovery: {snapshot.recovery_action or 'Manual intervention needed'}{RESET}")
        print(f"{RED}{'!'*80}{RESET}\n")
    
    # -------------------------------------------------------------
    # Statistics and Analysis
    # -------------------------------------------------------------
    
    def _update_stats(self, snapshot: DecisionSnapshot):
        """Update statistics"""
        
        self.stats["total_decisions"] += 1
        
        if snapshot.action == ActionType.BUY:
            self.stats["buy_decisions"] += 1
        elif snapshot.action == ActionType.SELL:
            self.stats["sell_decisions"] += 1
        elif snapshot.action == ActionType.HOLD:
            self.stats["hold_decisions"] += 1
        
        if snapshot.will_execute:
            self.stats["successful_executions"] += 1
        else:
            self.stats["blocked_executions"] += 1
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics"""
        
        with self._lock:
            runtime = (datetime.datetime.utcnow() - self.stats["start_time"]).total_seconds()
            
            return {
                "runtime_hours": runtime / 3600,
                "total_decisions": self.stats["total_decisions"],
                "buy_decisions": self.stats["buy_decisions"],
                "sell_decisions": self.stats["sell_decisions"],
                "hold_decisions": self.stats["hold_decisions"],
                "execution_rate": (self.stats["successful_executions"] / 
                                 max(self.stats["total_decisions"], 1)) * 100,
                "block_rate": (self.stats["blocked_executions"] / 
                             max(self.stats["total_decisions"], 1)) * 100,
                "total_errors": self.stats["total_errors"],
                "decisions_per_hour": (self.stats["total_decisions"] / 
                                      max(runtime / 3600, 1)),
            }
    
    def print_summary(self):
        """Print summary statistics"""
        
        stats = self.get_summary_stats()
        
        print("\n" + "="*80)
        print("📊 POSITION DEBUGGER SUMMARY")
        print("="*80)
        print(f"Runtime: {stats['runtime_hours']:.2f} hours")
        print(f"Total Decisions: {stats['total_decisions']}")
        print(f"  • Buy Decisions: {stats['buy_decisions']}")
        print(f"  • Sell Decisions: {stats['sell_decisions']}")
        print(f"  • Hold Decisions: {stats['hold_decisions']}")
        print(f"Execution Rate: {stats['execution_rate']:.1f}%")
        print(f"Block Rate: {stats['block_rate']:.1f}%")
        print(f"Total Errors: {stats['total_errors']}")
        print(f"Decision Rate: {stats['decisions_per_hour']:.1f}/hour")
        print("="*80 + "\n")