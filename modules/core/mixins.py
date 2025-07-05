# ─────────────────────────────────────────────────────────────
# File: modules/core/mixins.py
# 🚀 ENHANCED InfoBus-only mixins (NO legacy patterns)
# ─────────────────────────────────────────────────────────────

from abc import ABC
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque

from modules.utils.info_bus import (
    InfoBus, InfoBusExtractor, InfoBusUpdater, InfoBusManager,
    cache_computation, extract_standard_context
)
from modules.utils.audit_utils import format_operator_message


class InfoBusTradingMixin:
    """
    Mixin for modules that process trading data through InfoBus.
    ALL trade data MUST flow through InfoBus.
    """
    
    def _initialize_trading_state(self):
        """Initialize trading-specific state"""
        self._trades_processed = 0
        self._total_pnl = 0.0
        self._winning_trades = 0
        self._losing_trades = 0
        self._trade_history = deque(maxlen=self.config.max_history)
    
    def _process_trades_from_info_bus(self, info_bus: InfoBus) -> List[Dict[str, Any]]:
        """Process trades ONLY from InfoBus"""
        trades = InfoBusExtractor.get_recent_trades(info_bus)
        context = extract_standard_context(info_bus)
        
        processed_trades = []
        for trade in trades:
            processed = self._process_single_trade(trade, context, info_bus)
            if processed:
                processed_trades.append(processed)
                self._update_trading_metrics(processed)
        
        # Update InfoBus with processing results
        self._add_module_data(info_bus, {
            'trades_processed': len(processed_trades),
            'total_pnl': self._total_pnl,
            'win_rate': self._get_win_rate()
        })
        
        return processed_trades
    
    def _process_single_trade(self, trade: Dict[str, Any], 
                            context: Dict[str, Any],
                            info_bus: InfoBus) -> Optional[Dict[str, Any]]:
        """Process single trade with InfoBus context"""
        # Extract risk score from InfoBus
        risk_score = InfoBusExtractor.get_risk_score(info_bus)
        
        processed = {
            'trade_id': f"{trade.get('symbol')}_{info_bus.get('step_idx')}",
            'symbol': trade.get('symbol'),
            'pnl': trade.get('pnl', 0),
            'risk_at_entry': risk_score,
            'regime': context.get('regime'),
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
    
    def _update_trading_metrics(self, trade: Dict[str, Any]):
        """Update trading metrics"""
        pnl = trade.get('pnl', 0)
        
        self._trades_processed += 1
        self._total_pnl += pnl
        
        if pnl > 0:
            self._winning_trades += 1
        elif pnl < 0:
            self._losing_trades += 1
        
        self._trade_history.append(trade)
        
        # Update performance metrics
        self._update_performance_metric('total_pnl', self._total_pnl)
        self._update_performance_metric('win_rate', self._get_win_rate())
    
    def _get_win_rate(self) -> float:
        """Calculate win rate"""
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
            'avg_pnl': self._total_pnl / max(self._trades_processed, 1)
        }


class InfoBusRiskMixin:
    """
    Mixin for risk management through InfoBus.
    ALL risk data MUST flow through InfoBus.
    """
    
    def _initialize_risk_state(self):
        """Initialize risk-specific state"""
        self._risk_alerts = deque(maxlen=100)
        self._risk_violations = 0
        self._last_risk_check = None
    
    def _extract_risk_context(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Extract risk context ONLY from InfoBus"""
        return InfoBusExtractor.extract_risk_context(info_bus)
    
    def _assess_risk_level(self, info_bus: InfoBus) -> Tuple[str, float, List[str]]:
        """Assess risk level from InfoBus data"""
        risk_context = self._extract_risk_context(info_bus)
        alerts = []
        
        # Get risk score (0-1 range)
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
        
        if risk_context['position_count'] > 8:
            alerts.append(f"Too many positions: {risk_context['position_count']}")
        
        # Update InfoBus with assessment
        InfoBusUpdater.add_module_data(info_bus, self.__class__.__name__, {
            'risk_assessment': {
                'level': level,
                'score': risk_score,
                'alerts': alerts
            }
        })
        
        return level, risk_score, alerts
    
    def _check_risk_limits_from_info_bus(self, info_bus: InfoBus) -> List[str]:
        """Check risk limits using InfoBus data"""
        violations = []
        risk_context = self._extract_risk_context(info_bus)
        limits = info_bus.get('risk_limits', {})
        
        # Check drawdown
        dd_limit = limits.get('max_drawdown', 0.2) * 100  # Convert to percentage
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
        
        # Log violations
        if violations:
            self._risk_violations += 1
            
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
    
    def _record_risk_event(self, event_type: str, severity: str, 
                          details: Dict[str, Any], info_bus: InfoBus):
        """Record risk event to InfoBus"""
        event = {
            'timestamp': datetime.datetime.now().isoformat(),
            'event_type': event_type,
            'severity': severity,
            'details': details,
            'module': self.__class__.__name__,
            'step': info_bus.get('step_idx', 0)
        }
        
        self._risk_alerts.append(event)
        
        # Add to InfoBus alerts
        InfoBusUpdater.add_alert(
            info_bus,
            message=f"{event_type}: {details.get('summary', '')}",
            severity=severity,
            module=self.__class__.__name__
        )
        
        # Update metrics
        self._update_performance_metric('risk_violations', self._risk_violations)


class InfoBusAnalysisMixin:
    """
    Mixin for data analysis through InfoBus.
    ALL analysis MUST use InfoBus data and cache results.
    """
    
    def _initialize_analysis_state(self):
        """Initialize analysis-specific state"""
        self._analysis_history = deque(maxlen=self.config.max_history)
        self._pattern_counts = defaultdict(int)
    
    @cache_computation("pattern_analysis")
    def _analyze_patterns_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Analyze patterns using InfoBus data with caching"""
        # Check cache first
        cached = InfoBusExtractor.get_cached_features(info_bus, 'pattern_analysis')
        if cached is not None:
            return cached
        
        # Get trades from InfoBus
        trades = InfoBusExtractor.get_recent_trades(info_bus)
        
        patterns = defaultdict(lambda: {'count': 0, 'total_pnl': 0})
        
        for trade in trades:
            # Determine pattern based on InfoBus context at trade time
            regime = trade.get('regime', 'unknown')
            pattern = f"{regime}_{trade.get('side', 'unknown')}"
            
            patterns[pattern]['count'] += 1
            patterns[pattern]['total_pnl'] += trade.get('pnl', 0)
        
        # Calculate statistics
        analysis = {}
        for pattern, stats in patterns.items():
            analysis[pattern] = {
                'count': stats['count'],
                'frequency': stats['count'] / max(len(trades), 1),
                'total_pnl': stats['total_pnl'],
                'avg_pnl': stats['total_pnl'] / max(stats['count'], 1)
            }
        
        # Cache results
        InfoBusManager.update_computation_cache(
            'pattern_analysis', analysis, self.__class__.__name__
        )
        
        return analysis
    
    @cache_computation("correlation_analysis")
    def _analyze_correlations_from_info_bus(self, info_bus: InfoBus) -> Dict[str, float]:
        """Analyze correlations using InfoBus market data"""
        correlations = {}
        
        # Get market data from InfoBus
        instruments = list(info_bus.get('prices', {}).keys())
        
        for i, inst1 in enumerate(instruments):
            for inst2 in instruments[i+1:]:
                # Get price data
                data1 = InfoBusExtractor.get_market_data(info_bus, inst1)
                data2 = InfoBusExtractor.get_market_data(info_bus, inst2)
                
                if data1 and data2:
                    prices1 = data1.get('close', [])
                    prices2 = data2.get('close', [])
                    
                    if len(prices1) > 10 and len(prices2) > 10:
                        # Calculate returns
                        ret1 = np.diff(np.log(prices1[-20:]))
                        ret2 = np.diff(np.log(prices2[-20:]))
                        
                        if len(ret1) == len(ret2) and len(ret1) > 0:
                            corr = np.corrcoef(ret1, ret2)[0, 1]
                            correlations[f"{inst1}-{inst2}"] = float(corr)
        
        # Update InfoBus
        InfoBusUpdater.add_module_data(info_bus, self.__class__.__name__, {
            'correlation_analysis': correlations
        })
        
        return correlations


class InfoBusVotingMixin:
    """
    Mixin for committee voting through InfoBus.
    ALL voting MUST use InfoBus for data and coordination.
    """
    
    def _initialize_voting_state(self):
        """Initialize voting-specific state"""
        self._votes_cast = 0
        self._vote_history = deque(maxlen=self.config.max_history)
        self._confidence_history = deque(maxlen=100)
    
    def _prepare_vote_from_info_bus(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Prepare vote using ONLY InfoBus data"""
        # Get action proposal
        action = self.propose_action(info_bus)
        confidence = self.confidence(info_bus)
        
        # Create vote
        vote = {
            'module': self.__class__.__name__,
            'action': action,
            'confidence': confidence,
            'reasoning': self._get_vote_reasoning(info_bus),
            'timestamp': datetime.datetime.now().isoformat(),
            'step': info_bus.get('step_idx', 0),
            'risk_score': InfoBusExtractor.get_risk_score(info_bus)
        }
        
        # Record vote
        self._record_vote(vote)
        
        # Add to InfoBus
        InfoBusUpdater.add_vote(info_bus, vote)
        
        return vote
    
    def _get_vote_reasoning(self, info_bus: InfoBus) -> str:
        """Generate vote reasoning from InfoBus context"""
        context = extract_standard_context(info_bus)
        return (
            f"Regime: {context['regime']}, "
            f"Risk: {context['risk_score']:.2f}, "
            f"Positions: {context['position_count']}"
        )
    
    def _record_vote(self, vote: Dict[str, Any]):
        """Record vote internally"""
        self._votes_cast += 1
        self._vote_history.append(vote)
        self._confidence_history.append(vote['confidence'])
        
        # Update metrics
        self._update_performance_metric('votes_cast', self._votes_cast)
        self._update_performance_metric('avg_confidence', 
                                      np.mean(list(self._confidence_history)))


class InfoBusComputationMixin:
    """
    Mixin for shared computations through InfoBus.
    Ensures computations are done ONCE and shared.
    """
    
    def _compute_or_get_cached(self, info_bus: InfoBus, computation_name: str,
                              compute_func: callable) -> Any:
        """Get cached computation or compute and cache"""
        # Check InfoBus cache
        cached = InfoBusManager.get_cached_computation(
            computation_name, self.__class__.__name__
        )
        
        if cached is not None:
            self.logger.debug(f"Using cached {computation_name}")
            return cached
        
        # Compute
        self.logger.debug(f"Computing {computation_name}")
        result = compute_func(info_bus)
        
        # Cache
        InfoBusManager.update_computation_cache(
            computation_name, result, self.__class__.__name__
        )
        
        return result
    
    def _share_computation_result(self, info_bus: InfoBus, result_name: str,
                                 result_data: Any):
        """Share computation result through InfoBus"""
        # Add to features if array
        if isinstance(result_data, np.ndarray):
            InfoBusUpdater.update_feature(
                info_bus, result_name, result_data, self.__class__.__name__
            )
        else:
            # Add to module data
            self._add_module_data(info_bus, {result_name: result_data})


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE MIXINS for common patterns
# ═══════════════════════════════════════════════════════════════════

class InfoBusTradingAnalysisMixin(InfoBusTradingMixin, InfoBusAnalysisMixin):
    """Combined trading and analysis through InfoBus"""
    
    def _analyze_trading_patterns(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Analyze trading patterns combining both capabilities"""
        trades = self._process_trades_from_info_bus(info_bus)
        patterns = self._analyze_patterns_from_info_bus(info_bus)
        
        # Combine insights
        combined = {
            'trade_summary': self._get_trading_summary(),
            'pattern_analysis': patterns,
            'performance_by_pattern': self._calculate_pattern_performance(patterns)
        }
        
        # Share results
        self._share_computation_result(info_bus, 'trading_analysis', combined)
        
        return combined
    
    def _calculate_pattern_performance(self, patterns: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance by pattern"""
        perf = {}
        for pattern, stats in patterns.items():
            if stats['count'] > 0:
                perf[pattern] = stats['avg_pnl']
        return perf


class InfoBusRiskAnalysisMixin(InfoBusRiskMixin, InfoBusAnalysisMixin):
    """Combined risk and analysis through InfoBus"""
    
    def _analyze_risk_patterns(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Analyze risk patterns over time"""
        level, score, alerts = self._assess_risk_level(info_bus)
        correlations = self._analyze_correlations_from_info_bus(info_bus)
        
        # High correlation risk
        high_corr_pairs = [
            pair for pair, corr in correlations.items() 
            if abs(corr) > 0.8
        ]
        
        if high_corr_pairs:
            alerts.append(f"High correlations: {len(high_corr_pairs)} pairs")
        
        analysis = {
            'current_risk_level': level,
            'risk_score': score,
            'active_alerts': alerts,
            'correlation_risk': high_corr_pairs,
            'risk_trend': self._calculate_risk_trend()
        }
        
        # Share results
        self._share_computation_result(info_bus, 'risk_analysis', analysis)
        
        return analysis
    
    def _calculate_risk_trend(self) -> str:
        """Calculate risk trend from history"""
        if len(self._risk_alerts) < 10:
            return "insufficient_data"
        
        recent = list(self._risk_alerts)[-10:]
        critical_count = sum(1 for a in recent if a['severity'] == 'CRITICAL')
        
        if critical_count >= 5:
            return "deteriorating"
        elif critical_count >= 2:
            return "elevated"
        else:
            return "stable"


class InfoBusFullIntegrationMixin(
    InfoBusTradingMixin, 
    InfoBusRiskMixin, 
    InfoBusAnalysisMixin,
    InfoBusVotingMixin,
    InfoBusComputationMixin
):
    """
    Full InfoBus integration for comprehensive modules.
    Provides ALL InfoBus capabilities.
    """
    
    def _initialize_full_state(self):
        """Initialize all mixin states"""
        self._initialize_trading_state()
        self._initialize_risk_state()
        self._initialize_analysis_state()
        self._initialize_voting_state()
    
    def _get_comprehensive_analysis(self, info_bus: InfoBus) -> Dict[str, Any]:
        """Get comprehensive analysis using all capabilities"""
        return {
            'trading': self._get_trading_summary(),
            'risk': self._analyze_risk_patterns(info_bus),
            'patterns': self._analyze_patterns_from_info_bus(info_bus),
            'correlations': self._analyze_correlations_from_info_bus(info_bus),
            'vote_performance': {
                'votes_cast': self._votes_cast,
                'avg_confidence': np.mean(list(self._confidence_history)) 
                                 if self._confidence_history else 0.5
            }
        }