# ─────────────────────────────────────────────────────────────
# File: modules/risk/compliance.py
# Enhanced with InfoBus integration & intelligent validation
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import datetime
import os
from typing import Any, List, Dict, Optional, Set, Tuple
from collections import deque, defaultdict

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import RiskMixin, TradingMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import AuditTracker, format_operator_message


class ComplianceModule(Module, RiskMixin, TradingMixin, StateManagementMixin):
    """
    Enhanced compliance validation system with InfoBus integration.
    Provides comprehensive trade validation with intelligent risk assessment.
    """

    DEFAULT_ALLOWED_SYMBOLS = {
        "EUR/USD", "EURUSD", "XAU/USD", "XAUUSD", "GBP/USD", "GBPUSD",
        "USD/JPY", "USDJPY", "AUD/USD", "AUDUSD", "USD/CHF", "USDCHF",
        "NZD/USD", "NZDUSD", "EUR/GBP", "EURGBP", "EUR/JPY", "EURJPY",
        "GBP/JPY", "GBPJPY", "CHF/JPY", "CHFJPY", "AUD/JPY", "AUDJPY"
    }

    def __init__(
        self,
        max_leverage: float = 30.0,
        max_single_position_risk: float = 0.20,
        max_total_risk: float = 0.50,
        max_daily_trades: int = 100,
        min_trade_size: float = 0.01,
        max_trade_size: float = 10.0,
        allowed_symbols: Optional[List[str]] = None,
        restricted_hours: Optional[List[int]] = None,
        dynamic_limits: bool = True,
        debug: bool = False,
        **kwargs
    ):
        # Initialize with enhanced config
        config = ModuleConfig(
            debug=debug,
            max_history=kwargs.get('max_history', 200),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_risk_state()
        self._initialize_trading_state()
        
        # Core configuration
        self.dynamic_limits = dynamic_limits
        
        # Risk parameters
        self.base_limits = {
            'max_leverage': float(max_leverage),
            'max_single_position_risk': float(max_single_position_risk),
            'max_total_risk': float(max_total_risk),
            'max_daily_trades': int(max_daily_trades),
            'min_trade_size': float(min_trade_size),
            'max_trade_size': float(max_trade_size)
        }
        
        # Current limits (may be adjusted dynamically)
        self.current_limits = self.base_limits.copy()
        
        # Symbol management
        self.allowed_symbols = self._normalize_symbols(
            allowed_symbols or self._get_default_symbols()
        )
        
        # Time restrictions
        self.restricted_hours = set(restricted_hours) if restricted_hours else set()
        
        # Enhanced state tracking
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.total_exposure = 0.0
        self.violations_count = 0
        self.violation_history = deque(maxlen=100)
        
        # Compliance tracking
        self.compliance_flags: List[str] = []
        self.validation_cache = {}
        self.risk_budget_used = 0.0
        
        # Performance metrics
        self.validation_stats = defaultdict(int)
        self.rejection_reasons = defaultdict(int)
        self.approval_rate = 1.0
        
        # Market context awareness
        self.context_adjustments = {
            'volatile': {'leverage': 0.8, 'position_risk': 0.8, 'daily_trades': 1.2},
            'ranging': {'leverage': 1.1, 'position_risk': 1.1, 'daily_trades': 0.9},
            'trending': {'leverage': 1.0, 'position_risk': 1.0, 'daily_trades': 1.0}
        }
        
        # Audit system
        self.audit_manager = AuditTracker("ComplianceModule")
        
        self.log_operator_info(
            "🛡️ Enhanced Compliance Module initialized",
            max_leverage=f"{max_leverage:.1f}x",
            position_risk_limit=f"{max_single_position_risk:.1%}",
            total_risk_limit=f"{max_total_risk:.1%}",
            allowed_symbols=len(self.allowed_symbols),
            dynamic_limits=dynamic_limits
        )

    def _get_default_symbols(self) -> List[str]:
        """Get default allowed symbols from environment or defaults"""
        env_symbols = os.getenv("COMPLIANCE_SYMBOLS")
        if env_symbols:
            return [s.strip() for s in env_symbols.split(",")]
        return list(self.DEFAULT_ALLOWED_SYMBOLS)

    def _normalize_symbols(self, symbols: List[str]) -> Set[str]:
        """Normalize symbols to handle both EUR/USD and EURUSD formats"""
        normalized = set()
        for sym in symbols:
            sym = sym.strip().upper()
            normalized.add(sym)
            # Add both formats
            if "/" in sym:
                normalized.add(sym.replace("/", ""))
            elif len(sym) == 6:  # Likely EURUSD format
                normalized.add(f"{sym[:3]}/{sym[3:]}")
        return normalized

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_trading_state()
        
        # Reset compliance state
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.total_exposure = 0.0
        self.violations_count = 0
        self.violation_history.clear()
        
        # Reset tracking
        self.compliance_flags.clear()
        self.validation_cache.clear()
        self.risk_budget_used = 0.0
        
        # Reset statistics
        self.validation_stats.clear()
        self.rejection_reasons.clear()
        self.approval_rate = 1.0
        
        # Reset limits to base values
        self.current_limits = self.base_limits.copy()
        
        self.log_operator_info("🔄 Compliance Module reset - all validation state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - compliance inactive")
            return
        
        # Extract context for dynamic limit adjustment
        context = extract_standard_context(info_bus)
        
        # Update daily trade tracking
        self._update_daily_tracking(info_bus)
        
        # Adjust limits dynamically if enabled
        if self.dynamic_limits:
            self._adjust_limits_for_context(context)
        
        # Validate pending trades
        self._validate_pending_trades(info_bus, context)
        
        # Update risk assessment
        self._assess_current_risk(info_bus, context)
        
        # Update InfoBus with compliance status
        self._update_info_bus(info_bus)
        
        # Update performance metrics
        self._update_compliance_metrics()

    def _update_daily_tracking(self, info_bus: InfoBus) -> None:
        """Update daily trade count tracking"""
        
        current_date = datetime.datetime.now().date()
        
        # Reset counter if new day
        if self.last_trade_date != current_date:
            if self.last_trade_date is not None and self.daily_trade_count > 0:
                self.log_operator_info(
                    f"📊 Daily trading summary",
                    date=str(self.last_trade_date),
                    trades=self.daily_trade_count,
                    limit=self.current_limits['max_daily_trades']
                )
            
            self.daily_trade_count = 0
            self.last_trade_date = current_date
        
        # Count new trades from InfoBus
        recent_trades = InfoBusExtractor.get_recent_trades(info_bus)
        if recent_trades:
            # Only count trades that happened today
            today_trades = [
                trade for trade in recent_trades
                if self._is_trade_today(trade)
            ]
            self.daily_trade_count += len(today_trades)

    def _is_trade_today(self, trade: Dict[str, Any]) -> bool:
        """Check if trade happened today"""
        try:
            trade_time = trade.get('timestamp')
            if isinstance(trade_time, str):
                trade_date = datetime.datetime.fromisoformat(
                    trade_time.replace('Z', '+00:00')
                ).date()
                return trade_date == datetime.datetime.now().date()
        except:
            pass
        return True  # Assume recent if can't parse

    def _adjust_limits_for_context(self, context: Dict[str, Any]) -> None:
        """Dynamically adjust limits based on market context"""
        
        regime = context.get('regime', 'unknown')
        volatility_level = context.get('volatility_level', 'medium')
        
        # Get adjustment factors
        adjustments = self.context_adjustments.get(regime, {'leverage': 1.0, 'position_risk': 1.0, 'daily_trades': 1.0})
        
        # Additional volatility adjustments
        vol_multiplier = 1.0
        if volatility_level == 'extreme':
            vol_multiplier = 0.7
        elif volatility_level == 'high':
            vol_multiplier = 0.85
        elif volatility_level == 'low':
            vol_multiplier = 1.15
        
        # Apply adjustments
        self.current_limits['max_leverage'] = (
            self.base_limits['max_leverage'] * 
            adjustments['leverage'] * vol_multiplier
        )
        
        self.current_limits['max_single_position_risk'] = (
            self.base_limits['max_single_position_risk'] * 
            adjustments['position_risk'] * vol_multiplier
        )
        
        self.current_limits['max_daily_trades'] = int(
            self.base_limits['max_daily_trades'] * 
            adjustments['daily_trades']
        )

    def _validate_pending_trades(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Validate any pending trades in InfoBus"""
        
        pending_orders = info_bus.get('pending_orders', [])
        
        for order in pending_orders:
            try:
                is_valid = self._validate_trade_from_info_bus(order, info_bus, context)
                
                # Add validation result to order
                order['compliance_validated'] = is_valid
                order['compliance_flags'] = self.compliance_flags.copy()
                
            except Exception as e:
                self.log_operator_error(f"Trade validation failed: {e}")
                order['compliance_validated'] = False
                order['compliance_flags'] = [f"validation_error: {str(e)}"]

    def _validate_trade_from_info_bus(self, order: Dict[str, Any], 
                                    info_bus: InfoBus, context: Dict[str, Any]) -> bool:
        """Validate a trade using InfoBus data"""
        
        # Clear previous flags
        self.compliance_flags.clear()
        
        # Extract trade parameters
        instrument = order.get('instrument', order.get('symbol', 'UNKNOWN'))
        size = abs(order.get('size', order.get('volume', 0)))
        price = order.get('price', self._get_current_price(instrument, info_bus))
        
        # Get current state from InfoBus
        risk_data = info_bus.get('risk', {})
        balance = risk_data.get('balance', risk_data.get('equity', 10000))
        positions = InfoBusExtractor.get_positions(info_bus)
        
        # Build validation context
        validation_context = {
            'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
            'step_idx': info_bus.get('step_idx', 0),
            'market_context': context,
            'instrument': instrument,
            'size': size,
            'price': price,
            'balance': balance,
            'positions': positions
        }
        
        # Perform comprehensive validation
        return self._comprehensive_trade_validation(validation_context)

    def _comprehensive_trade_validation(self, ctx: Dict[str, Any]) -> bool:
        """Perform comprehensive trade validation"""
        
        validation_passed = True
        checks_performed = []
        
        try:
            # 1. Daily trade limit check
            if not self._check_daily_trade_limit(ctx):
                validation_passed = False
            checks_performed.append(('daily_limit', validation_passed))
            
            # 2. Trading hours check
            if not self._check_trading_hours(ctx):
                validation_passed = False
            checks_performed.append(('trading_hours', validation_passed))
            
            # 3. Symbol allowlist check
            if not self._check_allowed_symbol(ctx):
                validation_passed = False
            checks_performed.append(('symbol_allowed', validation_passed))
            
            # 4. Trade size validation
            if not self._check_trade_size_limits(ctx):
                validation_passed = False
            checks_performed.append(('trade_size', validation_passed))
            
            # 5. Position risk validation
            if not self._check_position_risk(ctx):
                validation_passed = False
            checks_performed.append(('position_risk', validation_passed))
            
            # 6. Total exposure validation
            if not self._check_total_exposure(ctx):
                validation_passed = False
            checks_performed.append(('total_exposure', validation_passed))
            
            # 7. Leverage validation
            if not self._check_leverage_limit(ctx):
                validation_passed = False
            checks_performed.append(('leverage', validation_passed))
            
            # 8. Market context validation
            if not self._check_market_context_restrictions(ctx):
                validation_passed = False
            checks_performed.append(('market_context', validation_passed))
            
            # Update statistics
            self.validation_stats['total_validations'] += 1
            if validation_passed:
                self.validation_stats['approved'] += 1
            else:
                self.validation_stats['rejected'] += 1
                self.violations_count += 1
            
            # Record validation audit
            self._record_validation_audit(ctx, validation_passed, checks_performed)
            
        except Exception as e:
            self.log_operator_error(f"Validation process failed: {e}")
            self.compliance_flags.append(f"validation_error: {str(e)}")
            validation_passed = False
        
        return validation_passed

    def _check_daily_trade_limit(self, ctx: Dict[str, Any]) -> bool:
        """Check daily trade limit"""
        if self.daily_trade_count >= self.current_limits['max_daily_trades']:
            self.compliance_flags.append(
                f"Daily trade limit exceeded: {self.daily_trade_count}/{self.current_limits['max_daily_trades']}"
            )
            self.rejection_reasons['daily_limit'] += 1
            return False
        return True

    def _check_trading_hours(self, ctx: Dict[str, Any]) -> bool:
        """Check trading hours restrictions"""
        current_hour = datetime.datetime.now().hour
        if current_hour in self.restricted_hours:
            self.compliance_flags.append(f"Trading restricted at hour {current_hour}")
            self.rejection_reasons['restricted_hours'] += 1
            return False
        return True

    def _check_allowed_symbol(self, ctx: Dict[str, Any]) -> bool:
        """Check if symbol is in allowlist"""
        instrument = ctx['instrument'].upper()
        
        # Check both formats
        if instrument not in self.allowed_symbols:
            # Try alternative format
            if "/" in instrument:
                alt_format = instrument.replace("/", "")
            elif len(instrument) == 6:
                alt_format = f"{instrument[:3]}/{instrument[3:]}"
            else:
                alt_format = instrument
            
            if alt_format not in self.allowed_symbols:
                self.compliance_flags.append(f"Symbol {instrument} not in allowlist")
                self.rejection_reasons['symbol_not_allowed'] += 1
                return False
        
        return True

    def _check_trade_size_limits(self, ctx: Dict[str, Any]) -> bool:
        """Check trade size within limits"""
        size = ctx['size']
        
        if size < self.current_limits['min_trade_size']:
            self.compliance_flags.append(
                f"Trade size {size:.4f} below minimum {self.current_limits['min_trade_size']}"
            )
            self.rejection_reasons['size_too_small'] += 1
            return False
        
        if size > self.current_limits['max_trade_size']:
            self.compliance_flags.append(
                f"Trade size {size:.4f} exceeds maximum {self.current_limits['max_trade_size']}"
            )
            self.rejection_reasons['size_too_large'] += 1
            return False
        
        return True

    def _check_position_risk(self, ctx: Dict[str, Any]) -> bool:
        """Check single position risk limit"""
        size = ctx['size']
        price = ctx['price']
        balance = ctx['balance']
        
        # Calculate position value (assuming standard lot sizing)
        position_value = size * price * 100_000  # Standard forex lot
        position_risk = position_value / max(balance, 1.0)
        
        if position_risk > self.current_limits['max_single_position_risk']:
            self.compliance_flags.append(
                f"Position risk {position_risk:.1%} exceeds limit {self.current_limits['max_single_position_risk']:.1%}"
            )
            self.rejection_reasons['position_risk'] += 1
            return False
        
        return True

    def _check_total_exposure(self, ctx: Dict[str, Any]) -> bool:
        """Check total portfolio exposure"""
        size = ctx['size']
        price = ctx['price']
        balance = ctx['balance']
        positions = ctx['positions']
        
        # Calculate new position value
        new_position_value = size * price * 100_000
        
        # Calculate existing exposure
        existing_exposure = 0.0
        for pos in positions:
            pos_size = abs(pos.get('size', 0))
            pos_price = pos.get('current_price', pos.get('entry_price', price))
            existing_exposure += pos_size * pos_price * 100_000
        
        # Total exposure
        total_exposure = existing_exposure + new_position_value
        total_risk = total_exposure / max(balance, 1.0)
        
        if total_risk > self.current_limits['max_total_risk']:
            self.compliance_flags.append(
                f"Total risk {total_risk:.1%} exceeds limit {self.current_limits['max_total_risk']:.1%}"
            )
            self.rejection_reasons['total_exposure'] += 1
            return False
        
        # Update exposure tracking
        self.total_exposure = total_exposure
        
        return True

    def _check_leverage_limit(self, ctx: Dict[str, Any]) -> bool:
        """Check leverage limit"""
        leverage = self.total_exposure / max(ctx['balance'], 1.0)
        
        if leverage > self.current_limits['max_leverage']:
            self.compliance_flags.append(
                f"Leverage {leverage:.1f}x exceeds limit {self.current_limits['max_leverage']:.1f}x"
            )
            self.rejection_reasons['leverage'] += 1
            return False
        
        return True

    def _check_market_context_restrictions(self, ctx: Dict[str, Any]) -> bool:
        """Check market context-specific restrictions"""
        market_context = ctx.get('market_context', {})
        
        # Restrict large trades in extreme volatility
        volatility_level = market_context.get('volatility_level', 'medium')
        if volatility_level == 'extreme' and ctx['size'] > self.current_limits['max_trade_size'] * 0.5:
            self.compliance_flags.append(
                f"Large trade restricted in extreme volatility: {ctx['size']:.4f}"
            )
            self.rejection_reasons['extreme_volatility'] += 1
            return False
        
        # Additional regime-specific restrictions can be added here
        
        return True

    def _get_current_price(self, instrument: str, info_bus: InfoBus) -> float:
        """Get current price for instrument from InfoBus"""
        prices = info_bus.get('prices', {})
        
        # Try exact match first
        if instrument in prices:
            return float(prices[instrument])
        
        # Try alternative formats
        for symbol, price in prices.items():
            if (symbol.replace("/", "") == instrument.replace("/", "") or
                symbol.replace("/", "") == instrument or
                symbol == instrument.replace("/", "")):
                return float(price)
        
        # Default fallback
        return 1.0

    def _assess_current_risk(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Assess current risk status"""
        
        risk_data = info_bus.get('risk', {})
        balance = risk_data.get('balance', 10000)
        positions = InfoBusExtractor.get_positions(info_bus)
        
        # Calculate current exposure
        current_exposure = 0.0
        for pos in positions:
            pos_size = abs(pos.get('size', 0))
            pos_price = pos.get('current_price', 1.0)
            current_exposure += pos_size * pos_price * 100_000
        
        # Calculate risk metrics
        current_leverage = current_exposure / max(balance, 1.0)
        risk_utilization = current_leverage / self.current_limits['max_leverage']
        
        # Update risk budget tracking
        self.risk_budget_used = min(risk_utilization, 1.0)
        
        # Generate warnings for high risk utilization
        if risk_utilization > 0.8:
            self.log_operator_warning(
                f"⚠️ High risk utilization: {risk_utilization:.1%}",
                leverage=f"{current_leverage:.1f}x",
                limit=f"{self.current_limits['max_leverage']:.1f}x"
            )
        
        # Track in risk state
        self._update_risk_metrics({
            'current_leverage': current_leverage,
            'risk_utilization': risk_utilization,
            'total_exposure': current_exposure
        })

    def _update_info_bus(self, info_bus: InfoBus) -> None:
        """Update InfoBus with compliance status"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'compliance', {
            'daily_trade_count': self.daily_trade_count,
            'daily_limit': self.current_limits['max_daily_trades'],
            'violations_count': self.violations_count,
            'approval_rate': self.approval_rate,
            'risk_budget_used': self.risk_budget_used,
            'current_limits': self.current_limits.copy(),
            'allowed_symbols_count': len(self.allowed_symbols),
            'compliance_flags': self.compliance_flags.copy()
        })
        
        # Add compliance flags to InfoBus
        for flag in self.compliance_flags:
            InfoBusUpdater.add_compliance_flag(info_bus, flag)
        
        # Update risk snapshot
        InfoBusUpdater.update_risk_snapshot(info_bus, {
            'compliance_risk_score': self.risk_budget_used,
            'daily_trades_remaining': max(0, self.current_limits['max_daily_trades'] - self.daily_trade_count),
            'leverage_utilization': self.risk_budget_used
        })
        
        # Add alerts for compliance issues
        if self.compliance_flags:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Compliance violations detected: {len(self.compliance_flags)} issues",
                severity="warning",
                module="ComplianceModule"
            )

    def _record_validation_audit(self, ctx: Dict[str, Any], passed: bool, 
                               checks: List[Tuple[str, bool]]) -> None:
        """Record detailed validation audit"""
        
        audit_data = {
            'validation_result': passed,
            'trade_details': {
                'instrument': ctx['instrument'],
                'size': ctx['size'],
                'price': ctx['price']
            },
            'checks_performed': [
                {'check': check_name, 'passed': check_result}
                for check_name, check_result in checks
            ],
            'compliance_flags': self.compliance_flags.copy(),
            'current_limits': self.current_limits.copy(),
            'market_context': ctx.get('market_context', {}),
            'risk_metrics': {
                'daily_trades': self.daily_trade_count,
                'total_exposure': self.total_exposure,
                'risk_budget_used': self.risk_budget_used
            }
        }
        
        self.audit_manager.record_event(
            event_type="trade_validation",
            module="ComplianceModule",
            details=audit_data,
            severity="warning" if not passed else "info"
        )

    def _update_compliance_metrics(self) -> None:
        """Update performance and compliance metrics"""
        
        # Calculate approval rate
        total_validations = self.validation_stats.get('total_validations', 0)
        if total_validations > 0:
            self.approval_rate = self.validation_stats.get('approved', 0) / total_validations
        
        # Update performance metrics
        self._update_performance_metric('approval_rate', self.approval_rate)
        self._update_performance_metric('daily_trade_count', self.daily_trade_count)
        self._update_performance_metric('violations_count', self.violations_count)
        self._update_performance_metric('risk_budget_used', self.risk_budget_used)
        
        # Track violation frequency
        if self.compliance_flags:
            self.violation_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'flags': self.compliance_flags.copy(),
                'count': len(self.compliance_flags)
            })

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        
        try:
            # Daily trade utilization
            daily_utilization = min(
                self.daily_trade_count / max(self.current_limits['max_daily_trades'], 1),
                1.0
            )
            
            # Risk budget utilization
            risk_utilization = self.risk_budget_used
            
            # Violation indicators
            has_violations = float(len(self.compliance_flags) > 0)
            violation_frequency = min(
                self.violations_count / max(self.validation_stats.get('total_validations', 1), 1),
                1.0
            )
            
            # Approval rate
            approval_rate = self.approval_rate
            
            # Dynamic adjustment factor
            leverage_adjustment = (
                self.current_limits['max_leverage'] / self.base_limits['max_leverage']
            )
            
            # Recent violation trend
            if len(self.violation_history) >= 10:
                recent_violations = sum(1 for v in list(self.violation_history)[-10:] if v['count'] > 0)
                violation_trend = recent_violations / 10.0
            else:
                violation_trend = 0.0
            
            return np.array([
                daily_utilization,      # Daily trade utilization
                risk_utilization,       # Risk budget utilization
                has_violations,         # Current violations indicator
                violation_frequency,    # Historical violation frequency
                approval_rate,          # Trade approval rate
                leverage_adjustment,    # Dynamic limit adjustment
                violation_trend         # Recent violation trend
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Compliance observation generation failed: {e}")
            return np.zeros(7, dtype=np.float32)

    def get_compliance_report(self) -> str:
        """Generate operator-friendly compliance report"""
        
        # Status indicators
        if self.risk_budget_used > 0.8:
            risk_status = "🚨 Critical"
        elif self.risk_budget_used > 0.5:
            risk_status = "⚠️ Elevated"
        else:
            risk_status = "✅ Normal"
        
        # Approval rate status
        if self.approval_rate > 0.9:
            approval_status = "🎯 Excellent"
        elif self.approval_rate > 0.7:
            approval_status = "✅ Good"
        elif self.approval_rate > 0.5:
            approval_status = "⚡ Fair"
        else:
            approval_status = "⚠️ Poor"
        
        # Daily trading status
        daily_utilization = self.daily_trade_count / max(self.current_limits['max_daily_trades'], 1)
        if daily_utilization > 0.8:
            daily_status = "⚠️ High Usage"
        elif daily_utilization > 0.5:
            daily_status = "⚡ Moderate"
        else:
            daily_status = "✅ Available"
        
        # Dynamic limits indicator
        leverage_ratio = self.current_limits['max_leverage'] / self.base_limits['max_leverage']
        if leverage_ratio != 1.0:
            limits_status = f"🔧 Adjusted ({leverage_ratio:.2f}x)"
        else:
            limits_status = "📊 Base Limits"
        
        # Top rejection reasons
        top_rejections = sorted(self.rejection_reasons.items(), key=lambda x: x[1], reverse=True)[:3]
        rejection_lines = []
        for reason, count in top_rejections:
            if count > 0:
                rejection_lines.append(f"  🚫 {reason.replace('_', ' ').title()}: {count}")
        
        # Current flags
        flag_lines = []
        for flag in self.compliance_flags[-5:]:  # Show last 5 flags
            flag_lines.append(f"  ⚠️ {flag}")
        
        return f"""
🛡️ ENHANCED COMPLIANCE MODULE
═══════════════════════════════════════
🎯 Risk Status: {risk_status} ({self.risk_budget_used:.1%} budget used)
📊 Approval Rate: {approval_status} ({self.approval_rate:.1%})
📈 Daily Trading: {daily_status} ({self.daily_trade_count}/{self.current_limits['max_daily_trades']})
🔧 Limit Status: {limits_status}

⚖️ CURRENT LIMITS
• Leverage: {self.current_limits['max_leverage']:.1f}x (base: {self.base_limits['max_leverage']:.1f}x)
• Position Risk: {self.current_limits['max_single_position_risk']:.1%}
• Total Risk: {self.current_limits['max_total_risk']:.1%}
• Trade Size: {self.current_limits['min_trade_size']:.3f} - {self.current_limits['max_trade_size']:.3f}
• Dynamic Adjustments: {'✅ Enabled' if self.dynamic_limits else '❌ Disabled'}

🌍 SYMBOL MANAGEMENT
• Allowed Symbols: {len(self.allowed_symbols)}
• Sample Symbols: {', '.join(list(self.allowed_symbols)[:5])}{'...' if len(self.allowed_symbols) > 5 else ''}
• Restricted Hours: {list(self.restricted_hours) if self.restricted_hours else 'None'}

📊 VALIDATION PERFORMANCE
• Total Validations: {self.validation_stats.get('total_validations', 0)}
• Approved: {self.validation_stats.get('approved', 0)}
• Rejected: {self.validation_stats.get('rejected', 0)}
• Total Violations: {self.violations_count}

🚫 TOP REJECTION REASONS
{chr(10).join(rejection_lines) if rejection_lines else "  ✅ No rejections yet"}

⚠️ CURRENT COMPLIANCE FLAGS
{chr(10).join(flag_lines) if flag_lines else "  ✅ No active flags"}

💡 OPERATIONAL STATUS
• Risk Budget Used: {self.risk_budget_used:.1%}
• Total Exposure: €{self.total_exposure:,.0f}
• Violations History: {len(self.violation_history)} records
• Last Validation: {('Recent' if self.validation_stats.get('total_validations', 0) > 0 else 'None')}
        """

    def mutate(self, mutation_rate: float = 0.1) -> None:
        """Evolutionary parameter mutation"""
        
        import random
        
        if random.random() < mutation_rate:
            # Mutate leverage limit
            self.base_limits['max_leverage'] = max(
                5.0, 
                min(100.0, self.base_limits['max_leverage'] + random.gauss(0, 5.0))
            )
        
        if random.random() < mutation_rate:
            # Mutate position risk limit
            self.base_limits['max_single_position_risk'] = max(
                0.05,
                min(0.5, self.base_limits['max_single_position_risk'] + random.gauss(0, 0.02))
            )
        
        if random.random() < mutation_rate:
            # Mutate total risk limit
            self.base_limits['max_total_risk'] = max(
                0.2,
                min(1.0, self.base_limits['max_total_risk'] + random.gauss(0, 0.05))
            )
        
        # Update current limits
        self.current_limits = self.base_limits.copy()
        
        self.log_operator_info(
            "🧬 Compliance parameters mutated",
            leverage=f"{self.base_limits['max_leverage']:.1f}x",
            position_risk=f"{self.base_limits['max_single_position_risk']:.1%}",
            total_risk=f"{self.base_limits['max_total_risk']:.1%}"
        )

    def crossover(self, other: 'ComplianceModule') -> 'ComplianceModule':
        """Create offspring through parameter crossover"""
        
        import random
        
        # Combine parameters from both parents
        child_params = {}
        for param in ['max_leverage', 'max_single_position_risk', 'max_total_risk']:
            if random.random() > 0.5:
                child_params[param] = self.base_limits[param]
            else:
                child_params[param] = other.base_limits[param]
        
        # Create child with combined parameters
        child = ComplianceModule(
            max_leverage=child_params['max_leverage'],
            max_single_position_risk=child_params['max_single_position_risk'],
            max_total_risk=child_params['max_total_risk'],
            max_daily_trades=self.base_limits['max_daily_trades'],
            allowed_symbols=list(self.allowed_symbols),
            debug=self.config.debug
        )
        
        return child

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health reporting"""
        base_details = super()._get_health_details()
        
        compliance_details = {
            'compliance_status': {
                'daily_trades': f"{self.daily_trade_count}/{self.current_limits['max_daily_trades']}",
                'risk_budget_used': f"{self.risk_budget_used:.1%}",
                'total_violations': self.violations_count,
                'approval_rate': f"{self.approval_rate:.1%}"
            },
            'limit_adjustments': {
                'leverage_ratio': (
                    self.current_limits['max_leverage'] / self.base_limits['max_leverage']
                ),
                'position_risk_ratio': (
                    self.current_limits['max_single_position_risk'] / self.base_limits['max_single_position_risk']
                ),
                'dynamic_adjustments_active': self.dynamic_limits
            },
            'violation_analysis': {
                'recent_violations': len([
                    v for v in self.violation_history 
                    if v.get('count', 0) > 0
                ]),
                'top_rejection_reasons': dict(sorted(
                    self.rejection_reasons.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3])
            },
            'symbol_management': {
                'allowed_symbols': len(self.allowed_symbols),
                'restricted_hours': len(self.restricted_hours)
            }
        }
        
        if base_details:
            base_details.update(compliance_details)
            return base_details
        
        return compliance_details

    # ================== LEGACY COMPATIBILITY ==================

    def validate_trade(self, instrument: str, size: float, price: float, 
                      balance: float, current_positions: Optional[Dict[str, Any]] = None,
                      timestamp: Optional[datetime.datetime] = None) -> bool:
        """Legacy compatibility method"""
        
        # Convert to standard position format
        positions = []
        if current_positions:
            for symbol, pos_data in current_positions.items():
                positions.append({
                    'symbol': symbol,
                    'size': pos_data.get('size', pos_data.get('lots', 0)),
                    'current_price': price,  # Use provided price as default
                    'entry_price': pos_data.get('price_open', price)
                })
        
        # Create validation context
        ctx = {
            'timestamp': (timestamp or datetime.datetime.now()).isoformat(),
            'market_context': {'regime': 'unknown', 'volatility_level': 'medium'},
            'instrument': instrument,
            'size': size,
            'price': price,
            'balance': balance,
            'positions': positions
        }
        
        # Use enhanced validation
        return self._comprehensive_trade_validation(ctx)

    def step(self, **kwargs) -> bool:
        """Legacy step interface"""
        
        # Extract trade data from kwargs if available
        trade = kwargs.get("trade")
        if not trade:
            return True  # No trade to validate
        
        # Extract environment data
        env = kwargs.get("env")
        if env:
            balance = getattr(env, "balance", 10000.0)
            positions = getattr(env, "open_positions", {})
            
            # Get price from environment data
            instrument = trade.get("instrument", "")
            try:
                df = env.data.get(instrument, {}).get("D1")
                if df is not None and env.current_step < len(df):
                    price = float(df.iloc[env.current_step]["close"])
                else:
                    price = 1.0
            except:
                price = 1.0
            
            # Use enhanced validation
            return self.validate_trade(
                instrument=instrument,
                size=abs(trade.get("size", 0.0)),
                price=price,
                balance=balance,
                current_positions=positions
            )
        
        return True