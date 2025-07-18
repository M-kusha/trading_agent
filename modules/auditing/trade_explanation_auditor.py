# ─────────────────────────────────────────────────────────────
# File: modules/auditing/trade_explanation_auditor.py
# ENHANCED: Trade auditor using SmartInfoBus architecture
# ─────────────────────────────────────────────────────────────

import numpy as np
import datetime
from typing import Dict, Any, List, Optional
from collections import deque

# [OK] FIXED: Proper imports for SmartInfoBus system
from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import format_operator_message


@module(
    provides=['trade_explanations', 'audit_alerts', 'explanation_metrics'],
    requires=['trading_signal', 'market_data', 'trades'],
    category='auditing',
    is_voting_member=False,
    hot_reload=True,
    explainable=True,
    timeout_ms=100,
    priority=3,
    version="2.0.0"
)
class TradeExplanationAuditor(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin):
    """
    [SEARCH] PRODUCTION-GRADE Trade Explanation Auditor
    
    Advanced trade auditing with:
    - Real-time explanation validation
    - Confidence tracking and alerting
    - Pattern recognition for trade quality
    - NASA/MILITARY GRADE reliability patterns
    """
    
    def _initialize(self):
        """Initialize trade auditor with SmartInfoBus integration"""
        # Initialize base mixins
        self._initialize_trading_state()
        
        # Auditor-specific state
        self.trade_explanations = deque(maxlen=1000)
        self.session_start = datetime.datetime.now()
        self.explanation_cache = {}
        
        # Quality metrics
        self.quality_metrics = {
            'total_trades_audited': 0,
            'high_confidence_trades': 0,
            'low_confidence_trades': 0,
            'missing_explanations': 0,
            'pattern_violations': 0
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'low_confidence_rate': 0.3,  # Alert if >30% low confidence
            'missing_explanation_rate': 0.1,  # Alert if >10% missing explanations
            'pattern_violation_rate': 0.15  # Alert if >15% pattern violations
        }
        
        self.logger.info(f"[SEARCH] {self.__class__.__name__} initialized with explanation auditing")
    
    def reset(self) -> None:
        """Reset with automatic infrastructure cleanup"""
        super().reset()
        
        # Module-specific reset
        self.trade_explanations.clear()
        self.session_start = datetime.datetime.now()
        self.explanation_cache.clear()
        self.quality_metrics = {
            'total_trades_audited': 0,
            'high_confidence_trades': 0,
            'low_confidence_trades': 0,
            'missing_explanations': 0,
            'pattern_violations': 0
        }
        
        self.logger.info("[RELOAD] Trade explanation auditor reset complete")
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        [SEARCH] MAIN AUDITING PROCESS
        
        Audits trade explanations and generates quality metrics:
        1. Extract trade and explanation data
        2. Validate explanation quality and completeness
        3. Detect patterns and anomalies
        4. Generate alerts for quality issues
        """
        try:
            # Extract audit context
            context = self._extract_audit_context(inputs)
            
            # Process trades for explanation auditing
            processed_trades = self._process_trade_explanations(inputs, context)
            
            # Create explanation analysis
            explanation_analysis = self._analyze_explanations(processed_trades, context)
            
            # Check for alerts
            alerts = self._check_for_alerts(explanation_analysis, context)
            
            # Update quality metrics
            self._update_quality_metrics(explanation_analysis)
            
            # Generate thesis
            thesis = self._generate_explanation_thesis(explanation_analysis, alerts)
            
            # Store results in SmartInfoBus
            smart_bus = InfoBusManager.get_instance()
            smart_bus.set(
                'trade_explanations',
                explanation_analysis,
                module=self.__class__.__name__,
                thesis=thesis,
                confidence=explanation_analysis.get('overall_confidence', 0.7)
            )
            
            return {
                'trade_explanations': explanation_analysis,
                'audit_alerts': alerts,
                'explanation_metrics': self.quality_metrics,
                '_thesis': thesis
            }
            
        except Exception as e:
            self.logger.error(f"[FAIL] Explanation audit failed: {e}")
            return {
                'trade_explanations': {'error': str(e)},
                'audit_alerts': [{'type': 'audit_failure', 'message': str(e)}],
                'explanation_metrics': self.quality_metrics,
                '_thesis': f"Trade explanation audit encountered error: {str(e)}"
            }
    
    def _extract_audit_context(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context for explanation auditing"""
        return {
            'trading_signal': inputs.get('trading_signal', {}),
            'market_data': inputs.get('market_data', {}),
            'trades': inputs.get('trades', []),
            'timestamp': inputs.get('timestamp', datetime.datetime.now()),
            'step_idx': inputs.get('step_idx', 0),
            'risk_score': 0.5  # Default risk score - can be enhanced later
        }
    
    def _process_trade_explanations(self, inputs: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and validate trade explanations"""
        processed_trades = []
        trades = context.get('trades', [])
        
        for trade in trades:
            explanation = self._audit_single_trade_explanation(trade, context)
            if explanation:
                processed_trades.append(explanation)
                self.trade_explanations.append(explanation)
        
        return processed_trades
    
    def _audit_single_trade_explanation(self, trade: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Audit explanation for a single trade"""
        try:
            trade_explanation = {
                'trade_id': f"{trade.get('symbol', 'UNK')}_{trade.get('timestamp', '')}",
                'symbol': trade.get('symbol'),
                'action': trade.get('action', 'unknown'),
                'pnl': trade.get('pnl', 0),
                'explanation_quality': 0.0,
                'confidence': trade.get('confidence', 0.5),
                'has_thesis': bool(trade.get('reason') or trade.get('_thesis')),
                'market_regime': context.get('market_regime', 'unknown'),
                'risk_level': 'high' if context['risk_score'] > 0.7 else 'normal',
                'processed_at': datetime.datetime.now().isoformat(),
                'quality_issues': []
            }
            
            # Validate explanation quality
            self._validate_explanation_quality(trade_explanation, trade)
            
            # Log significant trades automatically
            if abs(trade_explanation['pnl']) > 50 or trade_explanation['explanation_quality'] < 0.5:
                emoji = "[PARTY]" if trade_explanation['pnl'] > 0 else "[WARN]"
                quality_emoji = "[OK]" if trade_explanation['explanation_quality'] > 0.7 else "[FAIL]"
                
                self.logger.info(
                    format_operator_message(
                        emoji,
                        f"Trade audit: {trade_explanation['symbol']} ${trade_explanation['pnl']:.2f}",
                        quality=f"{quality_emoji} {trade_explanation['explanation_quality']:.1%}",
                        confidence=f"{trade_explanation['confidence']:.1%}",
                        context="trade_audit"
                    )
                )
            
            return trade_explanation
            
        except Exception as e:
            self.logger.error(f"[FAIL] Failed to audit trade explanation: {e}")
            return None
    
    def _validate_explanation_quality(self, trade_explanation: Dict[str, Any], trade: Dict[str, Any]):
        """Validate and score explanation quality"""
        quality_score = 0.0
        issues = []
        
        # Check for thesis/reason
        if trade_explanation['has_thesis']:
            quality_score += 0.4
        else:
            issues.append('missing_thesis')
        
        # Check confidence level
        confidence = trade_explanation['confidence']
        if confidence > 0.7:
            quality_score += 0.3
        elif confidence < 0.3:
            issues.append('low_confidence')
            quality_score += 0.1
        else:
            quality_score += 0.2
        
        # Check for action clarity
        if trade_explanation['action'] in ['buy', 'sell', 'hold']:
            quality_score += 0.2
        else:
            issues.append('unclear_action')
        
        # Check for risk assessment
        if 'risk_assessment' in trade:
            quality_score += 0.1
        else:
            issues.append('missing_risk_assessment')
        
        trade_explanation['explanation_quality'] = quality_score
        trade_explanation['quality_issues'] = issues
    
    def _analyze_explanations(self, processed_trades: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall explanation quality and patterns"""
        if not processed_trades:
            return {
                'trade_count': 0,
                'avg_quality': 1.0,
                'avg_confidence': 1.0,
                'overall_confidence': 1.0,
                'pattern_analysis': 'No trades to analyze'
            }
        
        # Calculate averages
        total_quality = sum(t['explanation_quality'] for t in processed_trades)
        total_confidence = sum(t['confidence'] for t in processed_trades)
        
        avg_quality = total_quality / len(processed_trades)
        avg_confidence = total_confidence / len(processed_trades)
        
        # Analyze patterns
        missing_thesis_count = len([t for t in processed_trades if not t['has_thesis']])
        low_confidence_count = len([t for t in processed_trades if t['confidence'] < 0.5])
        
        # Pattern analysis
        pattern_analysis = f"Quality: {avg_quality:.1%}, Confidence: {avg_confidence:.1%}"
        if missing_thesis_count > 0:
            pattern_analysis += f", Missing thesis: {missing_thesis_count}"
        if low_confidence_count > 0:
            pattern_analysis += f", Low confidence: {low_confidence_count}"
        
        return {
            'trade_count': len(processed_trades),
            'avg_quality': avg_quality,
            'avg_confidence': avg_confidence,
            'overall_confidence': min(avg_quality, avg_confidence),
            'missing_thesis_count': missing_thesis_count,
            'low_confidence_count': low_confidence_count,
            'pattern_analysis': pattern_analysis,
            'detailed_trades': processed_trades[-10:]  # Keep last 10 for debugging
        }
    
    def _check_for_alerts(self, analysis: Dict[str, Any], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts based on explanation quality"""
        alerts = []
        
        if analysis['trade_count'] == 0:
            return alerts
        
        # Check explanation quality alerts
        if analysis['avg_quality'] < 0.5:
            alerts.append({
                'type': 'low_explanation_quality',
                'severity': 'high',
                'message': f"Low explanation quality detected: {analysis['avg_quality']:.1%}",
                'avg_quality': analysis['avg_quality'],
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        # Check confidence alerts
        if analysis['avg_confidence'] < 0.4:
            alerts.append({
                'type': 'low_confidence_pattern',
                'severity': 'medium',
                'message': f"Low confidence pattern detected: {analysis['avg_confidence']:.1%}",
                'avg_confidence': analysis['avg_confidence'],
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        # Check missing thesis rate
        missing_rate = analysis['missing_thesis_count'] / analysis['trade_count']
        if missing_rate > self.alert_thresholds['missing_explanation_rate']:
            alerts.append({
                'type': 'missing_explanations',
                'severity': 'medium',
                'message': f"High rate of missing explanations: {missing_rate:.1%}",
                'missing_count': analysis['missing_thesis_count'],
                'total_count': analysis['trade_count'],
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        # Risk-based alerts
        if context['risk_score'] > 0.8 and analysis['avg_confidence'] < 0.6:
            alerts.append({
                'type': 'high_risk_low_confidence',
                'severity': 'critical',
                'message': "High risk environment with low confidence trades",
                'risk_score': context['risk_score'],
                'confidence': analysis['avg_confidence'],
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        return alerts
    
    def _update_quality_metrics(self, analysis: Dict[str, Any]):
        """Update running quality metrics"""
        self.quality_metrics['total_trades_audited'] += analysis['trade_count']
        
        if analysis['trade_count'] > 0:
            high_conf_trades = analysis['trade_count'] - analysis['low_confidence_count']
            self.quality_metrics['high_confidence_trades'] += high_conf_trades
            self.quality_metrics['low_confidence_trades'] += analysis['low_confidence_count']
            self.quality_metrics['missing_explanations'] += analysis['missing_thesis_count']
            
            # Pattern violations (trades with multiple quality issues)
            pattern_violations = sum(1 for trade in analysis.get('detailed_trades', []) 
                                   if len(trade.get('quality_issues', [])) > 2)
            self.quality_metrics['pattern_violations'] += pattern_violations
    
    def _generate_explanation_thesis(self, analysis: Dict[str, Any], alerts: List[Dict[str, Any]]) -> str:
        """Generate explanation audit thesis"""
        if analysis['trade_count'] == 0:
            return "No trades processed for explanation auditing in this cycle."
        
        quality = analysis['avg_quality']
        confidence = analysis['avg_confidence']
        
        if quality > 0.8 and confidence > 0.7:
            thesis = f"Excellent explanation quality detected with {analysis['trade_count']} trades audited. "
            thesis += f"Average quality: {quality:.1%}, confidence: {confidence:.1%}."
        elif quality > 0.6 and confidence > 0.5:
            thesis = f"Good explanation quality with {analysis['trade_count']} trades audited. "
            thesis += f"Quality: {quality:.1%}, confidence: {confidence:.1%}."
        else:
            thesis = f"Explanation quality needs improvement. {analysis['trade_count']} trades audited. "
            thesis += f"Quality: {quality:.1%}, confidence: {confidence:.1%}."
        
        if alerts:
            thesis += f" Generated {len(alerts)} quality alerts requiring attention."
        
        return thesis
    
    def generate_detailed_report(self) -> str:
        """Generate comprehensive explanation audit report"""
        session_duration = (datetime.datetime.now() - self.session_start).total_seconds() / 3600
        total_audited = self.quality_metrics['total_trades_audited']
        
        if total_audited == 0:
            return "No trades have been audited yet."
        
        high_conf_rate = self.quality_metrics['high_confidence_trades'] / total_audited
        missing_rate = self.quality_metrics['missing_explanations'] / total_audited
        violation_rate = self.quality_metrics['pattern_violations'] / total_audited
        
        return f"""
[SEARCH] TRADE EXPLANATION AUDIT REPORT
═══════════════════════════════════════
📅 Session Duration: {session_duration:.1f} hours
[STATS] Total Trades Audited: {total_audited}

[CHART] EXPLANATION QUALITY METRICS
• High Confidence Rate: {high_conf_rate:.1%}
• Missing Explanations: {missing_rate:.1%}
• Pattern Violations: {violation_rate:.1%}

[WARN] QUALITY ALERTS
• Low Confidence: {self.quality_metrics['low_confidence_trades']}
• Missing Explanations: {self.quality_metrics['missing_explanations']}
• Pattern Violations: {self.quality_metrics['pattern_violations']}

[TARGET] RECOMMENDATIONS
• Target explanation completeness >90%
• Maintain confidence levels >70%
• Monitor pattern violations <10%
        """
    
    def get_explanation_statistics(self) -> Dict[str, Any]:
        """Get detailed explanation statistics"""
        if not self.trade_explanations:
            return {'error': 'No explanations available'}
        
        recent_explanations = list(self.trade_explanations)[-100:]  # Last 100
        
        avg_quality = sum(exp['explanation_quality'] for exp in recent_explanations) / len(recent_explanations)
        avg_confidence = sum(exp['confidence'] for exp in recent_explanations) / len(recent_explanations)
        
        return {
            'total_explanations': len(self.trade_explanations),
            'recent_explanations': len(recent_explanations),
            'avg_quality': avg_quality,
            'avg_confidence': avg_confidence,
            'quality_metrics': self.quality_metrics,
            'session_duration_hours': (datetime.datetime.now() - self.session_start).total_seconds() / 3600
        }
    
    # Required abstract methods for SmartInfoBusTradingMixin
    async def propose_action(self, **inputs) -> Dict[str, Any]:
        """Propose trade explanation audit action"""
        return {
            'action_type': 'explanation_audit',
            'priority': 'normal',
            'audit_focus': 'trade_explanations',
            'target_metrics': ['confidence', 'quality', 'completeness'],
            '_thesis': 'Auditing trade explanations for quality and completeness'
        }
    
    async def calculate_confidence(self, action: Dict[str, Any], **inputs) -> float:
        """Calculate confidence in explanation audit action"""
        # Base confidence on quality metrics
        total_trades = self.quality_metrics['total_trades_audited']
        if total_trades == 0:
            return 0.5
        
        # Calculate quality score
        high_confidence_rate = self.quality_metrics['high_confidence_trades'] / total_trades
        missing_explanation_rate = self.quality_metrics['missing_explanations'] / total_trades
        
        # Higher confidence if more high-confidence trades and fewer missing explanations
        quality_score = high_confidence_rate * (1 - missing_explanation_rate)
        return min(0.9, max(0.1, quality_score))


# ────────────────────────────────────────────────────────────────────────────────
# [SEARCH] MODULE REGISTRATION
# ────────────────────────────────────────────────────────────────────────────────
# Module is automatically discovered and registered via @module decorator
# No manual registration needed - SmartInfoBus handles everything!
