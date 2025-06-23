import logging
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Union
from modules.core.core import Module
from modules.utils.info_bus import InfoBus, TradeInfo

class TradeExplanationAuditor(Module):
    """
    Enhanced trade explanation auditor integrated with InfoBus.
    Provides comprehensive audit trail for every trading decision.
    """
    
    def __init__(self, history_len: int = 100, debug: bool = True):
        super().__init__()
        # immediately turn any floating‐point invalid operation into an exception
        np.seterr(all='raise')

        self.history_len = history_len
        self.debug = debug
        self.reset()

        # set up a dedicated logger
        self.logger = logging.getLogger(f"TradeExplanationAuditor_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        
    def reset(self) -> None:
        """Reset audit history"""
        self.explanations: List[Dict[str, Any]] = []
        self.step_counter = 0
        self.trade_summaries: Dict[str, Dict[str, Any]] = {}
        self.module_contributions: Dict[str, List[float]] = {}
        
    def step(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Process step from InfoBus or kwargs"""
        if info_bus:
            self._process_info_bus(info_bus)
        else:
            self._process_kwargs(**kwargs)
            
    def _process_info_bus(self, info_bus: InfoBus) -> None:
        """Extract explanation from InfoBus"""
        # Process any new trades
        for trade in info_bus.get('recent_trades', []):
            self._record_trade_explanation(trade, info_bus)
            
        # Record step explanation
        explanation = {
            'timestamp': info_bus.get('timestamp', datetime.datetime.now().isoformat()),
            'step': info_bus.get('step_idx', self.step_counter),
            'episode': info_bus.get('episode_idx', 0),
            'consensus': info_bus.get('consensus', 0.5),
            'market_regime': info_bus.get('market_context', {}).get('regime', 'unknown'),
            'positions': len(info_bus.get('positions', [])),
            'pnl_today': info_bus.get('pnl_today', 0.0),
            'alerts': info_bus.get('alerts', []),
            'votes': self._summarize_votes(info_bus.get('votes', [])),
            'risk_status': self._extract_risk_status(info_bus.get('risk', {}))
        }
        
        self._add_explanation(explanation)
        
    def _process_kwargs(self, **kwargs) -> None:
        """Legacy kwargs processing"""
        explanation = {
            'timestamp': datetime.datetime.now().isoformat(),
            'step': self.step_counter,
            'action': kwargs.get('action'),
            'order_status': kwargs.get('order_status'),
            'reasoning': kwargs.get('reasoning'),
            'confidence': kwargs.get('confidence'),
            'regime': kwargs.get('regime'),
            'voting': kwargs.get('voting'),
            'pnl': kwargs.get('pnl'),
            'modules_trace': kwargs.get('modules_trace'),
            'exception': str(kwargs.get('exception')) if kwargs.get('exception') else None
        }
        
        self._add_explanation(explanation)
        
    def _record_trade_explanation(self, trade: TradeInfo, info_bus: InfoBus) -> None:
        """Create detailed explanation for a specific trade"""
        trade_id = f"{trade['symbol']}_{trade['timestamp']}"
        
        explanation = {
            'trade_id': trade_id,
            'symbol': trade['symbol'],
            'side': trade['side'],
            'size': trade['size'],
            'price': trade['price'],
            'pnl': trade['pnl'],
            'reason': trade.get('reason', 'unknown'),
            'confidence': trade.get('confidence', 0.5),
            'market_context': {
                'regime': info_bus.get('market_context', {}).get('regime'),
                'volatility': info_bus.get('market_context', {}).get('volatility', {}).get(trade['symbol']),
                'session': info_bus.get('market_status', {}).get('session')
            },
            'risk_metrics': {
                'drawdown': info_bus.get('risk', {}).get('current_drawdown', 0),
                'exposure': info_bus.get('risk', {}).get('margin_used', 0),
                'var': info_bus.get('risk', {}).get('var_95', 0)
            },
            'module_votes': self._get_module_votes_for_trade(trade, info_bus.get('votes', []))
        }
        
        self.trade_summaries[trade_id] = explanation
        
        if self.debug:
            print(f"[TradeExplanationAuditor] Trade {trade_id}: {trade['side']} "
                  f"{trade['size']} @ {trade['price']}, PnL: {trade['pnl']:.2f}")
            
    def _summarize_votes(self, votes: List[Dict]) -> Dict[str, Any]:
        """Summarize voting patterns without ever producing a NaN."""
        # collect only real numeric confidences
        confidences = [
            v.get('confidence', 0.0)
            for v in votes
            if isinstance(v.get('confidence', None), (int, float))
        ]
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        summary = {
            'total_votes': len(votes),
            'avg_confidence': avg_confidence,
            'consensus_direction': self._calculate_consensus_direction(votes),
            'top_modules': self._get_top_voting_modules(votes)
        }
        return summary

        
    def _calculate_consensus_direction(self, votes: List[Dict]) -> str:
        """Determine overall voting direction"""
        directions = []
        for vote in votes:
            action = vote.get('action')
            if isinstance(action, (list, np.ndarray)):
                # Assume first element is direction
                directions.append(np.sign(action[0]) if len(action) > 0 else 0)
            else:
                directions.append(np.sign(float(action)) if action else 0)
                
        avg_direction = np.mean(directions) if directions else 0
        
        if avg_direction > 0.1:
            return "bullish"
        elif avg_direction < -0.1:
            return "bearish"
        else:
            return "neutral"
            
    def _get_top_voting_modules(self, votes: List[Dict]) -> List[str]:
        """Get modules with highest confidence"""
        sorted_votes = sorted(votes, key=lambda v: v.get('confidence', 0), reverse=True)
        return [v.get('module', 'unknown') for v in sorted_votes[:3]]
        
    def _extract_risk_status(self, risk: Dict) -> Dict[str, Any]:
        """Extract key risk metrics"""
        return {
            'drawdown': risk.get('current_drawdown', 0),
            'exposure': risk.get('margin_used', 0) / max(risk.get('equity', 1), 1),
            'position_count': len(risk.get('open_positions', [])),
            'at_risk_limit': risk.get('current_drawdown', 0) > 0.9 * risk.get('dd_limit', 0.3)
        }
        
    def _get_module_votes_for_trade(self, trade: TradeInfo, votes: List[Dict]) -> List[Dict]:
        """Extract votes relevant to this trade"""
        relevant_votes = []
        for vote in votes:
            if vote.get('instrument') == trade['symbol']:
                relevant_votes.append({
                    'module': vote.get('module'),
                    'confidence': vote.get('confidence'),
                    'reasoning': vote.get('reasoning')
                })
        return relevant_votes
        
    def _add_explanation(self, explanation: Dict[str, Any]) -> None:
        """Add explanation to history with size management"""
        self.explanations.append(explanation)
        self.step_counter += 1
        
        if len(self.explanations) > self.history_len:
            self.explanations.pop(0)
            
    def get_trade_analysis(self, n_recent: int = 10) -> Dict[str, Any]:
        """Analyze recent trades for patterns"""
        recent_trades = list(self.trade_summaries.values())[-n_recent:]
        
        if not recent_trades:
            return {}
            
        analysis = {
            'total_trades': len(recent_trades),
            'total_pnl': sum(t['pnl'] for t in recent_trades),
            'win_rate': sum(1 for t in recent_trades if t['pnl'] > 0) / len(recent_trades),
            'avg_confidence': np.mean([t['confidence'] for t in recent_trades]),
            'regime_breakdown': self._analyze_by_regime(recent_trades),
            'reason_breakdown': self._analyze_by_reason(recent_trades)
        }
        
        return analysis
        
    def _analyze_by_regime(self, trades: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by market regime"""
        regime_stats = {}
        for trade in trades:
            regime = trade.get('market_context', {}).get('regime', 'unknown')
            if regime not in regime_stats:
                regime_stats[regime] = {'count': 0, 'pnl': 0, 'wins': 0}
            regime_stats[regime]['count'] += 1
            regime_stats[regime]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                regime_stats[regime]['wins'] += 1
                
        # Calculate win rates
        for regime, stats in regime_stats.items():
            stats['win_rate'] = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            
        return regime_stats
        
    def _analyze_by_reason(self, trades: List[Dict]) -> Dict[str, int]:
        """Count trades by reason"""
        reason_counts = {}
        for trade in trades:
            reason = trade.get('reason', 'unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        return reason_counts
        
    def get_observation_components(self) -> np.ndarray:
        """
        Build a 6‐dim vector of audit metrics.
        If anything is ever NaN or infinite, we log the full vector and raise.
        """
        if not self.explanations:
            obs = np.zeros(6, dtype=np.float32)
        else:
            recent = self.explanations[-10:]
            # safe average confidence
            confs = [
                e['confidence']
                for e in recent
                if isinstance(e.get('confidence', None), (int, float))
            ]
            avg_conf = float(np.mean(confs)) if confs else 0.0

            ta = self.get_trade_analysis()
            obs = np.array([
                float(len(self.explanations)),        # total steps recorded
                avg_conf,                             # average confidence over last 10
                ta.get('win_rate', 0.5),              # win rate
                ta.get('total_pnl', 0.0) / 100.0,     # normalized PnL
                float(len(self.trade_summaries)),     # number of trades
                float(self.step_counter)              # steps since reset
            ], dtype=np.float32)

        # Fail fast if anything is non‐finite
        if not np.all(np.isfinite(obs)):
            self.logger.exception("🛑 Non‐finite in auditor obs: %r", obs)
            raise ValueError(f"Invalid values in audit observation: {obs!r}")

        return obs
        
    def generate_report(self) -> str:
        """Generate human-readable audit report"""
        analysis = self.get_trade_analysis()
        
        report = f"""
Trade Explanation Audit Report
==============================
Generated: {datetime.datetime.now().isoformat()}
Total Steps: {self.step_counter}
Total Trades: {len(self.trade_summaries)}

Performance Summary:
- Total P&L: ${analysis.get('total_pnl', 0):.2f}
- Win Rate: {analysis.get('win_rate', 0):.1%}
- Avg Confidence: {analysis.get('avg_confidence', 0):.2f}

Regime Analysis:
"""
        for regime, stats in analysis.get('regime_breakdown', {}).items():
            report += f"- {regime}: {stats['count']} trades, "
            report += f"${stats['pnl']:.2f} P&L, {stats['win_rate']:.1%} win rate\n"
            
        report += "\nTop Trade Reasons:\n"
        for reason, count in sorted(analysis.get('reason_breakdown', {}).items(), 
                                   key=lambda x: x[1], reverse=True)[:5]:
            report += f"- {reason}: {count} trades\n"
            
        return report
        
    def get_state(self) -> Dict[str, Any]:
        """Save complete audit state"""
        return {
            'explanations': self.explanations.copy(),
            'step_counter': self.step_counter,
            'trade_summaries': self.trade_summaries.copy(),
            'module_contributions': self.module_contributions.copy()
        }
        
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore audit state"""
        self.explanations = state.get('explanations', []).copy()
        self.step_counter = state.get('step_counter', 0)
        self.trade_summaries = state.get('trade_summaries', {}).copy()
        self.module_contributions = state.get('module_contributions', {}).copy()
        
    def export_as_jsonl(self, path: str) -> None:
        """Export audit trail as JSONL"""
        import json
        with open(path, 'w', encoding='utf-8') as f:
            for exp in self.explanations:
                f.write(json.dumps(exp, default=str) + '\n')
                
    def to_dataframe(self):
        """Convert to pandas DataFrame for analysis"""
        import pandas as pd
        return pd.DataFrame(self.explanations)