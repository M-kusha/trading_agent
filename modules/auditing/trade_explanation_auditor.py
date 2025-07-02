# ─────────────────────────────────────────────────────────────
# File: modules/auditing/enhanced_trade_auditor.py
# AFTER: Using enhanced infrastructure - 80% less code!
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import FullAuditMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, extract_standard_context


class TradeExplanationAuditor(Module, FullAuditMixin):
    """
    Trade auditor using enhanced infrastructure.
    90% less boilerplate code thanks to base class and mixins!
    """
    
    def _initialize_module_state(self):
        """Module-specific initialization"""
        self._initialize_full_audit_state()
        self.trade_explanations = []
        self.session_start = datetime.datetime.now()
        
        self.log_operator_info("Trade auditor initialized", mode="audit_mode")
    
    def reset(self) -> None:
        """Reset with automatic infrastructure cleanup"""
        super().reset()  # Handles all standard cleanup
        self._reset_full_audit_state()  # Handles mixin cleanup
        
        # Module-specific reset
        self.trade_explanations.clear()
        self.session_start = datetime.datetime.now()
    
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Core auditing logic - infrastructure handles the rest"""
        if not info_bus:
            return
        
        # Extract standard context (no repetitive code!)
        context = extract_standard_context(info_bus)
        
        # Process trades using mixin (no boilerplate!)
        processed_trades = self._process_trades_from_info_bus(info_bus)
        
        # Create step explanation
        explanation = {
            'timestamp': info_bus.get('timestamp'),
            'step': info_bus.get('step_idx', self._step_count),
            'context': context,
            'trade_count': len(processed_trades),
            'risk_score': context['risk_score']
        }
        
        self.trade_explanations.append(explanation)
        
        # Generate operator alerts automatically
        self._check_for_operator_alerts(explanation, context)
    
    def _process_single_trade(self, trade: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Override mixin method for trade-specific processing"""
        trade_explanation = {
            'trade_id': f"{trade.get('symbol', 'UNK')}_{trade.get('timestamp', '')}",
            'symbol': trade.get('symbol'),
            'pnl': trade.get('pnl', 0),
            'reason': trade.get('reason', 'unknown'),
            'confidence': trade.get('confidence', 0.5),
            'market_regime': context['regime'],
            'risk_level': 'high' if context['risk_score'] > 50 else 'normal',
            'processed_at': datetime.datetime.now().isoformat()
        }
        
        # Log significant trades automatically
        if abs(trade_explanation['pnl']) > 50:
            emoji = "🎉" if trade_explanation['pnl'] > 0 else "⚠️"
            self.log_operator_info(
                f"Significant trade: {trade_explanation['symbol']} ${trade_explanation['pnl']:.2f}",
                regime=context['regime'],
                confidence=f"{trade_explanation['confidence']:.1%}"
            )
        
        return trade_explanation
    
    def _check_for_operator_alerts(self, explanation: Dict[str, Any], context: Dict[str, Any]):
        """Generate operator alerts based on conditions"""
        
        # Risk alerts
        if context['risk_score'] > 70:
            self.log_operator_warning(
                "High risk conditions detected",
                risk_score=f"{context['risk_score']:.1f}",
                drawdown=f"{context['drawdown_pct']:.1f}%"
            )
        
        # Low consensus alert
        if context['consensus'] < 0.3:
            self.log_operator_info(
                "Low consensus detected - conflicting signals",
                consensus=f"{context['consensus']:.1%}"
            )
    
    def _get_observation_impl(self) -> np.ndarray:
        """Audit metrics for RL - infrastructure handles validation"""
        summary = self._get_comprehensive_summary()
        
        return np.array([
            float(len(self.trade_explanations)),
            summary['trading']['win_rate'],
            summary['trading']['total_pnl'] / 100.0,  # Normalized
            summary['risk']['avg_risk_score'] / 100.0,
            float(summary['trading']['trades_processed']),
            summary['overall_health']['overall_score'] / 100.0
        ], dtype=np.float32)
    
    def _get_module_state(self) -> Dict[str, Any]:
        """Module-specific state"""
        return {
            'trade_explanations': self.trade_explanations[-100:],  # Keep recent only
            'session_start': self.session_start.isoformat()
        }
    
    def _set_module_state(self, module_state: Dict[str, Any]):
        """Restore module-specific state"""
        self.trade_explanations = module_state.get('trade_explanations', [])
        session_start_str = module_state.get('session_start')
        if session_start_str:
            self.session_start = datetime.datetime.fromisoformat(session_start_str)
    
    def generate_operator_report(self) -> str:
        """Generate operator report using mixin data"""
        summary = self._get_comprehensive_summary()
        
        return f"""
🔍 TRADE AUDIT REPORT
═══════════════════════════════════════
📅 Session: {(datetime.datetime.now() - self.session_start).total_seconds()/3600:.1f} hours
📊 Health: {summary['overall_health']['overall_status']} ({summary['overall_health']['overall_score']:.0f}/100)

📈 TRADING PERFORMANCE
• Total Trades: {summary['trading']['trades_processed']}
• Win Rate: {summary['trading']['win_rate']:.1%}
• Total P&L: ${summary['trading']['total_pnl']:.2f}
• Avg P&L: ${summary['trading']['avg_pnl']:.2f}

⚠️ RISK METRICS
• Avg Risk Score: {summary['risk']['avg_risk_score']:.1f}/100
• Risk Violations: {summary['risk']['violations']}
• Recent Alerts: {summary['risk']['recent_critical_alerts']}

🎯 ANALYSIS INSIGHTS
• Pattern Types: {summary['analysis']['pattern_types']}
• Correlations: {summary['analysis']['correlations_tracked']}
        """

