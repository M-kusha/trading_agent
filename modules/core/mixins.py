# ─────────────────────────────────────────────────────────────
# File: modules/core/mixins.py
# 🚀 ENHANCED SmartInfoBus-aware mixins
# ─────────────────────────────────────────────────────────────

from abc import ABC
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque

from modules.utils.info_bus import (
    InfoBus, SmartInfoBus, InfoBusExtractor, InfoBusUpdater, InfoBusManager,
    cache_computation, extract_standard_context
)
from modules.utils.audit_utils import format_operator_message

# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS TRADING MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusTradingMixin:
    """
    Enhanced trading mixin with SmartInfoBus thesis generation.
    ALL trade data MUST flow through SmartInfoBus with explanations.
    """
    
    def _initialize_trading_state(self):
        """Initialize trading-specific state"""
        self._trades_processed = 0
        self._total_pnl = 0.0
        self._winning_trades = 0
        self._losing_trades = 0
        self._trade_history = deque(maxlen=self.config.max_history)
        self._trade_theses = deque(maxlen=self.config.max_history)
    
    def _process_trades_with_thesis(self, info_bus: InfoBus) -> List[Dict[str, Any]]:
        """Process trades with mandatory thesis generation"""
        trades = InfoBusExtractor.get_recent_trades(info_bus)
        smart_bus = InfoBusManager.get_instance()
        
        processed_trades = []
        for trade in trades:
            # Generate thesis for trade
            thesis = self._generate_trade_thesis(trade, info_bus)
            
            # Process trade
            processed = self._process_single_trade(trade, info_bus)
            if processed:
                processed['thesis'] = thesis
                processed_trades.append(processed)
                self._update_trading_metrics(processed)
                
                # Store in SmartInfoBus with thesis
                smart_bus.set(
                    f"trade_{processed['trade_id']}",
                    processed,
                    module=self.__class__.__name__,
                    thesis=thesis,
                    confidence=processed.get('confidence', 0.8)
                )
        
        # Update summary with thesis
        summary_thesis = self._generate_trading_summary_thesis(processed_trades)
        smart_bus.set(
            'trading_summary',
            self._get_trading_summary(),
            module=self.__class__.__name__,
            thesis=summary_thesis
        )
        
        return processed_trades
    
    def _generate_trade_thesis(self, trade: Dict[str, Any], info_bus: InfoBus) -> str:
        """Generate human-readable thesis for trade"""
        symbol = trade.get('symbol', 'UNKNOWN')
        side = trade.get('side', 'unknown')
        pnl = trade.get('pnl', 0)
        
        # Get context from SmartInfoBus
        smart_bus = InfoBusManager.get_instance()
        regime = smart_bus.get('market_regime', self.__class__.__name__) or 'unknown'
        risk_score = InfoBusExtractor.get_risk_score(info_bus)
        
        thesis = f"""
TRADE THESIS: {symbol} {side.upper()}
==================================
Market Regime: {regime}
Risk Level: {risk_score:.1%}
P&L Result: ${pnl:+.2f}

REASONING:
- Trade executed in {regime} market conditions
- Risk score of {risk_score:.1%} was {'acceptable' if risk_score < 0.5 else 'elevated'}
- Position aligned with current market dynamics
"""
        
        return thesis.strip()
    
    def _generate_trading_summary_thesis(self, trades: List[Dict]) -> str:
        """Generate summary thesis for trading session"""
        if not trades:
            return "No trades executed in this period"
        
        total_pnl = sum(t.get('pnl', 0) for t in trades)
        winning = sum(1 for t in trades if t.get('pnl', 0) > 0)
        
        return f"""
TRADING SESSION SUMMARY
======================
Total Trades: {len(trades)}
Winning Trades: {winning} ({winning/len(trades):.1%})
Total P&L: ${total_pnl:+.2f}

The session showed {'positive' if total_pnl > 0 else 'negative'} performance
with a win rate of {winning/len(trades):.1%}.
"""
    
    def _process_single_trade(self, trade: Dict[str, Any], info_bus: InfoBus) -> Optional[Dict[str, Any]]:
        """Process single trade with SmartInfoBus integration"""
        context = extract_standard_context(info_bus)
        risk_score = InfoBusExtractor.get_risk_score(info_bus)
        
        processed = {
            'trade_id': f"{trade.get('symbol')}_{info_bus.get('step_idx')}",
            'symbol': trade.get('symbol'),
            'side': trade.get('side'),
            'size': trade.get('size', 0),
            'pnl': trade.get('pnl', 0),
            'risk_at_entry': risk_score,
            'regime': context.get('regime'),
            'confidence': trade.get('confidence', 0.5),
            'processed_at': datetime.datetime.now().isoformat()
        }
        
        # Log significant trades
        if abs(processed['pnl']) > 100:
            self.logger.info(
                format_operator_message(
                    "💰" if processed['pnl'] > 0 else "💸",
                    "TRADE PROCESSED",
                    instrument=processed['symbol'],
                    details=f"P&L: ${processed['pnl']:+.2f}",
                    risk_score=risk_score
                ),
                info_bus
            )
        
        return processed
    
    # Other methods remain similar but use SmartInfoBus features
    def _update_trading_metrics(self, trade: Dict[str, Any]):
        pnl = trade.get('pnl', 0)
        
        self._trades_processed += 1
        self._total_pnl += pnl
        
        if pnl > 0:
            self._winning_trades += 1
        elif pnl < 0:
            self._losing_trades += 1
        
        self._trade_history.append(trade)
        if 'thesis' in trade:
            self._trade_theses.append(trade['thesis'])
        
        self._update_performance_metric('total_pnl', self._total_pnl)
        self._update_performance_metric('win_rate', self._get_win_rate())
    
    def _get_win_rate(self) -> float:
        total = self._winning_trades + self._losing_trades
        return self._winning_trades / max(total, 1)
    
    def _get_trading_summary(self) -> Dict[str, Any]:
        return {
            'trades_processed': self._trades_processed,
            'total_pnl': self._total_pnl,
            'winning_trades': self._winning_trades,
            'losing_trades': self._losing_trades,
            'win_rate': self._get_win_rate(),
            'avg_pnl': self._total_pnl / max(self._trades_processed, 1),
            'has_theses': len(self._trade_theses) > 0
        }

# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS RISK MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusRiskMixin:
    """
    Enhanced risk management with SmartInfoBus explanations.
    ALL risk assessments MUST include thesis.
    """
    
    def _initialize_risk_state(self):
        """Initialize risk-specific state"""
        self._risk_alerts = deque(maxlen=100)
        self._risk_violations = 0
        self._last_risk_check = None
        self._risk_theses = deque(maxlen=50)
    
    def _assess_risk_with_thesis(self, info_bus: InfoBus) -> Tuple[str, float, List[str], str]:
        """Assess risk level with mandatory thesis"""
        smart_bus = InfoBusManager.get_instance()
        risk_context = InfoBusExtractor.extract_risk_context(info_bus)
        
        alerts = []
        risk_score = risk_context['risk_score']
        
        # Determine level
        if risk_score >= 0.8:
            level = "CRITICAL"
        elif risk_score >= 0.6:
            level = "HIGH"
        elif risk_score >= 0.3:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        # Generate alerts
        if risk_context['drawdown_pct'] > 15:
            alerts.append(f"High drawdown: {risk_context['drawdown_pct']:.1f}%")
        
        if risk_context['exposure_pct'] > 80:
            alerts.append(f"High exposure: {risk_context['exposure_pct']:.1f}%")
        
        # Generate thesis
        thesis = self._generate_risk_thesis(level, risk_score, risk_context, alerts)
        
        # Store in SmartInfoBus with thesis
        smart_bus.set(
            'risk_assessment',
            {
                'level': level,
                'score': risk_score,
                'alerts': alerts,
                'context': risk_context
            },
            module=self.__class__.__name__,
            thesis=thesis,
            confidence=0.9
        )
        
        return level, risk_score, alerts, thesis
    
    def _generate_risk_thesis(self, level: str, score: float, 
                            context: Dict[str, Any], alerts: List[str]) -> str:
        """Generate risk assessment thesis"""
        thesis = f"""
RISK ASSESSMENT: {level}
======================
Risk Score: {score:.1%}
Drawdown: {context['drawdown_pct']:.1f}%
Exposure: {context['exposure_pct']:.1f}%
Open Positions: {context['position_count']}

ANALYSIS:
The current risk level is {level} based on:
"""
        
        if score >= 0.8:
            thesis += """
- CRITICAL risk conditions detected
- Immediate risk reduction required
- Trading should be restricted"""
        elif score >= 0.6:
            thesis += """
- Elevated risk conditions present
- Caution advised for new positions
- Consider reducing exposure"""
        else:
            thesis += """
- Risk levels within acceptable range
- Normal trading can continue
- Monitor for changes"""
        
        if alerts:
            thesis += f"\n\nSPECIFIC ALERTS:\n" + "\n".join(f"- {a}" for a in alerts)
        
        thesis += f"\n\nRECOMMENDATION: {self._get_risk_recommendation(level)}"
        
        return thesis.strip()
    
    def _get_risk_recommendation(self, level: str) -> str:
        """Get risk-based recommendation"""
        recommendations = {
            "CRITICAL": "Halt all trading and reduce positions immediately",
            "HIGH": "Avoid new positions and consider reducing exposure",
            "MEDIUM": "Trade with caution and monitor closely",
            "LOW": "Normal trading operations can proceed"
        }
        return recommendations.get(level, "Monitor risk levels")
    
    # Other risk methods updated for SmartInfoBus
    def _check_risk_limits_from_info_bus(self, info_bus: InfoBus) -> List[str]:
        violations = []
        risk_context = self._extract_risk_context(info_bus)
        limits = info_bus.get('risk_limits', {})
        
        # Check drawdown
        dd_limit = limits.get('max_drawdown', 0.2) * 100
        if risk_context['drawdown_pct'] > dd_limit:
            violations.append(
                f"Drawdown {risk_context['drawdown_pct']:.1f}% > {dd_limit:.1f}%"
            )
        
        # Check exposure
        exp_limit = limits.get('max_exposure', 0.8) * 100
        if risk_context['exposure_pct'] > exp_limit:
            violations.append(
                f"Exposure {risk_context['exposure_pct']:.1f}% > {exp_limit:.1f}%"
            )
        
        # Log violations with thesis
        if violations:
            self._risk_violations += 1
            
            violation_thesis = f"Risk limits breached: {', '.join(violations)}"
            smart_bus = InfoBusManager.get_instance()
            smart_bus.set(
                f'risk_violation_{self._risk_violations}',
                {
                    'violations': violations,
                    'timestamp': datetime.datetime.now().isoformat()
                },
                module=self.__class__.__name__,
                thesis=violation_thesis
            )
            
            for violation in violations:
                self.logger.warning(
                    format_operator_message(
                        "🚨", "RISK LIMIT BREACH",
                        details=violation,
                        risk_score=risk_context['risk_score']
                    ),
                    info_bus
                )
        
        return violations
    
    def _extract_risk_context(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract risk context from InfoBus"""
        return InfoBusExtractor.extract_risk_context(info_bus)

# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS VOTING MIXIN
# ═══════════════════════════════════════════════════════════════════

class SmartInfoBusVotingMixin:
    """
    Enhanced voting mixin with SmartInfoBus thesis requirements.
    ALL votes MUST include reasoning and confidence.
    """
    
    def _initialize_voting_state(self):
        """Initialize voting-specific state"""
        self._votes_cast = 0
        self._vote_history = deque(maxlen=self.config.max_history)
        self._confidence_history = deque(maxlen=100)
        self._vote_theses = deque(maxlen=50)
    
    def _prepare_vote_with_thesis(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Prepare vote with mandatory thesis"""
        smart_bus = InfoBusManager.get_instance()
        
        # Get action proposal
        action = self.propose_action(info_bus)
        confidence = self.confidence(info_bus)
        
        # Generate thesis
        thesis = self._generate_vote_thesis(action, confidence, info_bus)
        
        # Create vote
        vote = {
            'module': self.__class__.__name__,
            'action': action,
            'confidence': confidence,
            'reasoning': thesis,
            'timestamp': datetime.datetime.now().isoformat(),
            'step': info_bus.get('step_idx', 0),
            'risk_score': InfoBusExtractor.get_risk_score(info_bus)
        }
        
        # Record vote
        self._record_vote(vote)
        
        # Store in SmartInfoBus with thesis
        smart_bus.set(
            f'vote_{self.__class__.__name__}_{self._votes_cast}',
            vote,
            module=self.__class__.__name__,
            thesis=thesis,
            confidence=confidence
        )
        
        # Add to InfoBus votes
        InfoBusUpdater.add_vote(info_bus, vote)
        
        return vote
    
    def _generate_vote_thesis(self, action: Any, confidence: float, 
                            info_bus: InfoBus) -> str:
        """Generate voting thesis"""
        context = extract_standard_context(info_bus)
        
        # Determine action type
        if isinstance(action, np.ndarray):
            action_desc = f"Action vector with {len(action)} dimensions"
            max_idx = np.argmax(np.abs(action))
            action_desc += f", strongest signal at index {max_idx}"
        else:
            action_desc = str(action)
        
        thesis = f"""
VOTING DECISION
===============
Module: {self.__class__.__name__}
Confidence: {confidence:.1%}
Action: {action_desc}

CONTEXT:
- Market Regime: {context['regime']}
- Risk Score: {context['risk_score']:.1%}
- Open Positions: {context['position_count']}

REASONING:
"""
        
        if confidence > 0.8:
            thesis += "High confidence based on strong market signals and favorable conditions."
        elif confidence > 0.5:
            thesis += "Moderate confidence with mixed signals requiring careful execution."
        else:
            thesis += "Low confidence suggests caution or no action may be appropriate."
        
        return thesis.strip()
    
    def _record_vote(self, vote: Dict[str, Any]):
        """Record vote with thesis"""
        self._votes_cast += 1
        self._vote_history.append(vote)
        self._confidence_history.append(vote['confidence'])
        
        if 'reasoning' in vote:
            self._vote_theses.append(vote['reasoning'])
        
        self._update_performance_metric('votes_cast', self._votes_cast)
        self._update_performance_metric('avg_confidence', 
                                      np.mean(list(self._confidence_history)))

# ═══════════════════════════════════════════════════════════════════
# LEGACY COMPATIBILITY ALIASES
# ═══════════════════════════════════════════════════════════════════

# Map legacy mixins to SmartInfoBus versions
InfoBusTradingMixin = SmartInfoBusTradingMixin
InfoBusRiskMixin = SmartInfoBusRiskMixin
InfoBusVotingMixin = SmartInfoBusVotingMixin
InfoBusAnalysisMixin = SmartInfoBusTradingMixin  # Analysis integrated into trading
InfoBusComputationMixin = SmartInfoBusTradingMixin  # Computation integrated

# Legacy composite mixins
class InfoBusTradingAnalysisMixin(SmartInfoBusTradingMixin):
    """Legacy composite - maps to SmartInfoBus version"""
    pass

class InfoBusRiskAnalysisMixin(SmartInfoBusRiskMixin):
    """Legacy composite - maps to SmartInfoBus version"""
    pass

class InfoBusFullIntegrationMixin(
    SmartInfoBusTradingMixin, 
    SmartInfoBusRiskMixin, 
    SmartInfoBusVotingMixin
):
    """Full SmartInfoBus integration"""
    
    def _initialize_full_state(self):
        """Initialize all mixin states"""
        self._initialize_trading_state()
        self._initialize_risk_state()
        self._initialize_voting_state()