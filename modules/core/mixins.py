# ─────────────────────────────────────────────────────────────
# File: modules/core/mixins.py
# Module mixins to eliminate common functionality duplication
# ─────────────────────────────────────────────────────────────

from abc import ABC
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque

from modules.utils.info_bus import InfoBus, InfoBusExtractor, extract_standard_context


class TradingMixin:
    """Mixin for modules that process trading data"""
    
    def _initialize_trading_state(self):
        """Initialize trading-specific state"""
        self._trades_processed = 0
        self._total_pnl = 0.0
        self._winning_trades = 0
        self._losing_trades = 0
        self._trade_history = deque(maxlen=self.config.max_history)
    
    def _process_trades_from_info_bus(self, info_bus: InfoBus) -> List[Dict[str, Any]]:
        """Standard trade processing pattern"""
        processed_trades = []
        context = extract_standard_context(info_bus)
        
        for trade in info_bus.get('recent_trades', []):
            processed_trade = self._process_single_trade(trade, context)
            if processed_trade:
                processed_trades.append(processed_trade)
                self._update_trading_metrics(processed_trade)
        
        return processed_trades
    
    def _process_single_trade(self, trade: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Override this to implement trade-specific processing"""
        return {
            'trade_id': f"{trade.get('symbol', 'UNK')}_{trade.get('timestamp', '')}",
            'pnl': trade.get('pnl', 0),
            'symbol': trade.get('symbol', 'UNKNOWN'),
            'context': context,
            'processed_at': datetime.datetime.now().isoformat()
        }
    
    def _update_trading_metrics(self, processed_trade: Dict[str, Any]):
        """Update standard trading metrics"""
        pnl = processed_trade.get('pnl', 0)
        
        self._trades_processed += 1
        self._total_pnl += pnl
        
        if pnl > 0:
            self._winning_trades += 1
        elif pnl < 0:
            self._losing_trades += 1
        
        # Store in history
        self._trade_history.append(processed_trade)
        
        # Update performance metrics
        self._update_performance_metric('total_pnl', self._total_pnl)
        self._update_performance_metric('win_rate', self._get_win_rate())
        self._update_performance_metric('trade_count', self._trades_processed)
    
    def _get_win_rate(self) -> float:
        """Calculate current win rate"""
        total = self._winning_trades + self._losing_trades
        return self._winning_trades / max(total, 1)
    
    def _get_trading_summary(self) -> Dict[str, Any]:
        """Get trading performance summary"""
        return {
            'trades_processed': self._trades_processed,
            'total_pnl': self._total_pnl,
            'winning_trades': self._winning_trades,
            'losing_trades': self._losing_trades,
            'win_rate': self._get_win_rate(),
            'avg_pnl': self._total_pnl / max(self._trades_processed, 1),
            'recent_trades': len(self._trade_history)
        }
    
    def _reset_trading_state(self):
        """Reset trading state"""
        self._trades_processed = 0
        self._total_pnl = 0.0
        self._winning_trades = 0
        self._losing_trades = 0
        self._trade_history.clear()


class RiskMixin:
    """Mixin for modules that handle risk management"""
    
    def _initialize_risk_state(self):
        """Initialize risk-specific state"""
        self._risk_alerts = deque(maxlen=100)
        self._risk_violations = 0
        self._risk_score_history = deque(maxlen=50)
        self._last_risk_check = None
    
    def _extract_risk_context(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract comprehensive risk context"""
        return {
            'drawdown_pct': InfoBusExtractor.get_drawdown_pct(info_bus),
            'exposure_pct': InfoBusExtractor.get_exposure_pct(info_bus),
            'position_count': InfoBusExtractor.get_position_count(info_bus),
            'risk_score': InfoBusExtractor.get_risk_score(info_bus),
            'recent_trades': InfoBusExtractor.get_recent_trades_count(info_bus),
            'alert_count': InfoBusExtractor.get_alert_count(info_bus),
            'market_regime': InfoBusExtractor.get_market_regime(info_bus),
            'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
            'session': InfoBusExtractor.get_session(info_bus)
        }
    
    def _assess_risk_level(self, risk_context: Dict[str, Any]) -> Tuple[str, float, List[str]]:
        """Assess current risk level with alerts"""
        alerts = []
        risk_score = 0.0
        
        # Drawdown risk
        dd_pct = risk_context.get('drawdown_pct', 0)
        if dd_pct > 15:
            alerts.append(f"High drawdown: {dd_pct:.1f}%")
            risk_score += 30
        elif dd_pct > 10:
            alerts.append(f"Elevated drawdown: {dd_pct:.1f}%")
            risk_score += 15
        
        # Exposure risk
        exp_pct = risk_context.get('exposure_pct', 0)
        if exp_pct > 80:
            alerts.append(f"High exposure: {exp_pct:.1f}%")
            risk_score += 25
        elif exp_pct > 60:
            alerts.append(f"Elevated exposure: {exp_pct:.1f}%")
            risk_score += 10
        
        # Position concentration
        pos_count = risk_context.get('position_count', 0)
        if pos_count > 8:
            alerts.append(f"Too many positions: {pos_count}")
            risk_score += 15
        
        # Volatility risk
        vol_level = risk_context.get('volatility_level', 'medium')
        if vol_level == 'high':
            alerts.append("High market volatility")
            risk_score += 10
        
        # Determine overall level
        if risk_score >= 50:
            level = "CRITICAL"
        elif risk_score >= 25:
            level = "HIGH"
        elif risk_score >= 10:
            level = "MEDIUM"
        else:
            level = "LOW"
        
        return level, risk_score, alerts
    
    def _record_risk_event(self, event_type: str, severity: str, details: Dict[str, Any]):
        """Record risk event with automatic alerting"""
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'module': self.__class__.__name__
        }
        
        self._risk_alerts.append(event)
        
        if severity in ['HIGH', 'CRITICAL']:
            self._risk_violations += 1
            self.log_operator_warning(
                f"Risk event: {event_type}",
                severity=severity,
                **details
            )
        
        # Update metrics
        self._update_performance_metric('risk_violations', self._risk_violations)
        self._update_performance_metric('risk_alerts', len(self._risk_alerts))
    
    def _check_risk_limits(self, risk_context: Dict[str, Any], 
                          limits: Dict[str, float]) -> List[str]:
        """Check against risk limits"""
        violations = []
        
        for limit_name, limit_value in limits.items():
            current_value = risk_context.get(limit_name, 0)
            if current_value > limit_value:
                violations.append(f"{limit_name}: {current_value:.2f} > {limit_value:.2f}")
        
        return violations
    
    def _get_risk_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        return {
            'total_alerts': len(self._risk_alerts),
            'violations': self._risk_violations,
            'last_risk_check': self._last_risk_check,
            'avg_risk_score': np.mean(list(self._risk_score_history)) if self._risk_score_history else 0,
            'recent_critical_alerts': sum(1 for alert in list(self._risk_alerts)[-10:] 
                                        if alert.get('severity') == 'CRITICAL')
        }
    
    def _reset_risk_state(self):
        """Reset risk state"""
        self._risk_alerts.clear()
        self._risk_violations = 0
        self._risk_score_history.clear()
        self._last_risk_check = None


class AnalysisMixin:
    """Mixin for modules that perform data analysis"""
    
    def _initialize_analysis_state(self):
        """Initialize analysis-specific state"""
        self._analysis_cache = {}
        self._analysis_history = deque(maxlen=self.config.max_history)
        self._pattern_counts = defaultdict(int)
        self._correlation_matrix = {}
    
    def _analyze_patterns(self, data: List[Dict[str, Any]], 
                         pattern_key: str = 'pattern') -> Dict[str, Any]:
        """Analyze patterns in data"""
        if not data:
            return {}
        
        patterns = defaultdict(lambda: {'count': 0, 'values': []})
        
        for item in data:
            pattern = item.get(pattern_key, 'unknown')
            patterns[pattern]['count'] += 1
            
            # Store associated values for further analysis
            if 'pnl' in item:
                patterns[pattern]['values'].append(item['pnl'])
        
        # Calculate statistics for each pattern
        analysis = {}
        for pattern, stats in patterns.items():
            values = stats['values']
            analysis[pattern] = {
                'count': stats['count'],
                'frequency': stats['count'] / len(data),
                'avg_value': np.mean(values) if values else 0,
                'std_value': np.std(values) if len(values) > 1 else 0,
                'total_value': sum(values) if values else 0
            }
        
        return analysis
    
    def _calculate_correlations(self, data: List[Dict[str, Any]], 
                              fields: List[str]) -> Dict[str, float]:
        """Calculate correlations between fields"""
        if len(data) < 10:  # Need minimum data for meaningful correlation
            return {}
        
        correlations = {}
        
        # Extract field data
        field_data = {}
        for field in fields:
            field_data[field] = [item.get(field, 0) for item in data if field in item]
        
        # Calculate pairwise correlations
        for i, field1 in enumerate(fields):
            for field2 in fields[i+1:]:
                if (len(field_data[field1]) >= 10 and 
                    len(field_data[field2]) >= 10 and
                    len(field_data[field1]) == len(field_data[field2])):
                    
                    try:
                        corr = np.corrcoef(field_data[field1], field_data[field2])[0, 1]
                        if np.isfinite(corr):
                            correlations[f"{field1}_{field2}"] = float(corr)
                    except:
                        pass
        
        return correlations
    
    def _detect_anomalies(self, values: List[float], 
                         std_threshold: float = 3.0) -> List[int]:
        """Detect anomalies using standard deviation"""
        if len(values) < 10:
            return []
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        anomalies = []
        for i, value in enumerate(values):
            if abs(value - mean_val) > std_threshold * std_val:
                anomalies.append(i)
        
        return anomalies
    
    def _trend_analysis(self, values: List[float], window: int = 10) -> Dict[str, Any]:
        """Analyze trends in time series data"""
        if len(values) < window * 2:
            return {'trend': 'insufficient_data', 'strength': 0.0}
        
        recent = values[-window:]
        earlier = values[-window*2:-window]
        
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        if recent_avg > earlier_avg * 1.05:
            trend = 'improving'
            strength = (recent_avg - earlier_avg) / earlier_avg
        elif recent_avg < earlier_avg * 0.95:
            trend = 'declining'
            strength = (earlier_avg - recent_avg) / earlier_avg
        else:
            trend = 'stable'
            strength = abs(recent_avg - earlier_avg) / earlier_avg
        
        return {
            'trend': trend,
            'strength': float(strength),
            'recent_avg': float(recent_avg),
            'earlier_avg': float(earlier_avg),
            'change_pct': float((recent_avg - earlier_avg) / earlier_avg * 100)
        }
    
    def _get_analysis_summary(self) -> Dict[str, Any]:
        """Get analysis summary"""
        return {
            'cached_analyses': len(self._analysis_cache),
            'analysis_history': len(self._analysis_history),
            'pattern_types': len(self._pattern_counts),
            'correlations_tracked': len(self._correlation_matrix)
        }
    
    def _reset_analysis_state(self):
        """Reset analysis state"""
        self._analysis_cache.clear()
        self._analysis_history.clear()
        self._pattern_counts.clear()
        self._correlation_matrix.clear()


class VotingMixin:
    """Mixin for modules that participate in committee voting"""
    
    def _initialize_voting_state(self):
        """Initialize voting-specific state"""
        self._votes_cast = 0
        self._vote_history = deque(maxlen=self.config.max_history)
        self._confidence_history = deque(maxlen=100)
        self._voting_performance = {'correct': 0, 'incorrect': 0, 'pending': 0}
    
    def _prepare_vote(self, obs: Any, info_bus: InfoBus, instrument: str) -> Dict[str, Any]:
        """Prepare standardized vote structure"""
        action = self.propose_action(obs, info_bus)
        confidence = self.confidence(obs, info_bus)
        
        vote = {
            'module': self.__class__.__name__,
            'instrument': instrument,
            'action': action,
            'confidence': confidence,
            'reasoning': self._get_vote_reasoning(obs, info_bus),
            'timestamp': datetime.datetime.now().isoformat(),
            'context': extract_standard_context(info_bus)
        }
        
        self._record_vote(vote)
        return vote
    
    def _get_vote_reasoning(self, obs: Any, info_bus: InfoBus) -> str:
        """Override to provide vote reasoning"""
        return "Standard voting logic"
    
    def _record_vote(self, vote: Dict[str, Any]):
        """Record vote with performance tracking"""
        self._votes_cast += 1
        self._vote_history.append(vote)
        self._confidence_history.append(vote['confidence'])
        
        # Update metrics
        self._update_performance_metric('votes_cast', self._votes_cast)
        self._update_performance_metric('avg_confidence', np.mean(list(self._confidence_history)))
    
    def _evaluate_vote_performance(self, vote: Dict[str, Any], outcome: Dict[str, Any]):
        """Evaluate how well a vote performed"""
        # This is a simplified evaluation - override for specific logic
        vote_action = vote.get('action', 0)
        actual_pnl = outcome.get('pnl', 0)
        
        if isinstance(vote_action, (list, np.ndarray)):
            vote_direction = np.sign(vote_action[0]) if len(vote_action) > 0 else 0
        else:
            vote_direction = np.sign(float(vote_action)) if vote_action else 0
        
        pnl_direction = np.sign(actual_pnl)
        
        if vote_direction * pnl_direction > 0:
            self._voting_performance['correct'] += 1
            return 'correct'
        else:
            self._voting_performance['incorrect'] += 1
            return 'incorrect'
    
    def _get_voting_summary(self) -> Dict[str, Any]:
        """Get voting performance summary"""
        total_evaluated = self._voting_performance['correct'] + self._voting_performance['incorrect']
        accuracy = self._voting_performance['correct'] / max(total_evaluated, 1)
        
        return {
            'votes_cast': self._votes_cast,
            'accuracy': accuracy,
            'avg_confidence': np.mean(list(self._confidence_history)) if self._confidence_history else 0,
            'recent_votes': len(self._vote_history),
            'performance': self._voting_performance.copy()
        }
    
    def _reset_voting_state(self):
        """Reset voting state"""
        self._votes_cast = 0
        self._vote_history.clear()
        self._confidence_history.clear()
        self._voting_performance = {'correct': 0, 'incorrect': 0, 'pending': 0}


class StateManagementMixin:
    """Mixin for enhanced state management"""
    
    def _get_enhanced_state(self) -> Dict[str, Any]:
        """Get enhanced state including mixin data"""
        state = {}
        
        # Trading state
        if hasattr(self, '_trades_processed'):
            state['trading'] = self._get_trading_summary()
        
        # Risk state
        if hasattr(self, '_risk_alerts'):
            state['risk'] = self._get_risk_summary()
        
        # Analysis state
        if hasattr(self, '_analysis_cache'):
            state['analysis'] = self._get_analysis_summary()
        
        # Voting state
        if hasattr(self, '_votes_cast'):
            state['voting'] = self._get_voting_summary()
        
        return state
    
    def _set_enhanced_state(self, state: Dict[str, Any]):
        """Restore enhanced state including mixin data"""
        
        # Restore trading state
        if 'trading' in state:
            trading_state = state['trading']
            if hasattr(self, '_trades_processed'):
                self._trades_processed = trading_state.get('trades_processed', 0)
                self._total_pnl = trading_state.get('total_pnl', 0.0)
                self._winning_trades = trading_state.get('winning_trades', 0)
                self._losing_trades = trading_state.get('losing_trades', 0)
        
        # Restore risk state
        if 'risk' in state:
            risk_state = state['risk']
            if hasattr(self, '_risk_violations'):
                self._risk_violations = risk_state.get('violations', 0)
        
        # Restore voting state
        if 'voting' in state:
            voting_state = state['voting']
            if hasattr(self, '_votes_cast'):
                self._votes_cast = voting_state.get('votes_cast', 0)
                self._voting_performance = voting_state.get('performance', 
                    {'correct': 0, 'incorrect': 0, 'pending': 0})


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE MIXINS - Common combinations
# ═══════════════════════════════════════════════════════════════════

class TradingAnalysisMixin(TradingMixin, AnalysisMixin):
    """Combined trading and analysis functionality"""
    pass

class RiskAnalysisMixin(RiskMixin, AnalysisMixin):
    """Combined risk and analysis functionality"""
    pass

class VotingAnalysisMixin(VotingMixin, AnalysisMixin):
    """Combined voting and analysis functionality"""
    pass

class FullAuditMixin(TradingMixin, RiskMixin, AnalysisMixin, StateManagementMixin):
    """Full audit functionality for comprehensive modules"""
    
    def _initialize_full_audit_state(self):
        """Initialize all audit capabilities"""
        self._initialize_trading_state()
        self._initialize_risk_state()
        self._initialize_analysis_state()
    
    def _reset_full_audit_state(self):
        """Reset all audit state"""
        self._reset_trading_state()
        self._reset_risk_state()
        self._reset_analysis_state()
    
    def _get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary across all capabilities"""
        return {
            'trading': self._get_trading_summary(),
            'risk': self._get_risk_summary(),
            'analysis': self._get_analysis_summary(),
            'overall_health': self._calculate_overall_health()
        }
    
    def _calculate_overall_health(self) -> Dict[str, Any]:
        """Calculate overall module health across all dimensions"""
        health_scores = {}
        
        # Trading health
        if hasattr(self, '_trades_processed') and self._trades_processed > 0:
            win_rate = self._get_win_rate()
            health_scores['trading'] = min(100, win_rate * 100 + 50)  # 50-100 scale
        
        # Risk health (inverse of violations)
        if hasattr(self, '_risk_violations'):
            risk_health = max(0, 100 - self._risk_violations * 10)
            health_scores['risk'] = risk_health
        
        # Overall health is average of components
        if health_scores:
            overall_score = np.mean(list(health_scores.values()))
            overall_status = "HEALTHY" if overall_score >= 70 else "DEGRADED" if overall_score >= 40 else "UNHEALTHY"
        else:
            overall_score = 100
            overall_status = "HEALTHY"
        
        return {
            'overall_score': overall_score,
            'overall_status': overall_status,
            'component_scores': health_scores
        }