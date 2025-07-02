
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
from modules.core.core import Module
from modules.core.mixins import TradingAnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, extract_standard_context


class TradeThesisTracker(Module, TradingAnalysisMixin):
    """
    Thesis tracker using enhanced infrastructure.
    Focus only on thesis-specific logic!
    """
    
    def _initialize_module_state(self):
        """Initialize thesis-specific state"""
        self._initialize_trading_state()
        self._initialize_analysis_state()
        
        self.current_thesis = "unknown"
        self.thesis_changes = 0
        self.thesis_performance = defaultdict(lambda: {"trades": 0, "pnl": 0.0})
    
    def reset(self) -> None:
        """Reset with automatic cleanup"""
        super().reset()
        self._reset_trading_state()
        self._reset_analysis_state()
        
        self.current_thesis = "unknown"
        self.thesis_changes = 0
        self.thesis_performance.clear()
    
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Core thesis tracking logic"""
        if not info_bus:
            return
        
        # Detect thesis changes using simplified logic
        new_thesis = self._extract_thesis(info_bus)
        if new_thesis != self.current_thesis:
            self._handle_thesis_change(self.current_thesis, new_thesis)
            self.current_thesis = new_thesis
        
        # Process trades with thesis context
        self._process_trades_from_info_bus(info_bus)
    
    def _extract_thesis(self, info_bus: InfoBus) -> str:
        """Extract current trading thesis"""
        context = extract_standard_context(info_bus)
        votes_summary = context['votes_summary']
        
        # Simple thesis extraction
        if votes_summary['total_votes'] > 0:
            direction = votes_summary['consensus_direction']
            confidence = "high" if votes_summary['avg_confidence'] > 0.7 else "low"
            regime = context['regime']
            return f"{direction}_{regime}_{confidence}"
        else:
            return f"no_consensus_{context['regime']}"
    
    def _handle_thesis_change(self, old_thesis: str, new_thesis: str):
        """Handle thesis change with automatic logging"""
        self.thesis_changes += 1
        self.log_operator_info(
            f"Thesis change: {old_thesis} → {new_thesis}",
            change_count=self.thesis_changes
        )
    
    def _process_single_trade(self, trade: Dict[str, Any], context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process trade with thesis context"""
        processed_trade = super()._process_single_trade(trade, context)
        
        if processed_trade:
            # Add thesis information
            processed_trade['thesis'] = self.current_thesis
            
            # Update thesis performance
            thesis_stats = self.thesis_performance[self.current_thesis]
            thesis_stats['trades'] += 1
            thesis_stats['pnl'] += trade.get('pnl', 0)
        
        return processed_trade
    
    def _get_observation_impl(self) -> np.ndarray:
        """Thesis-specific metrics"""
        trading_summary = self._get_trading_summary()
        
        # Get best/worst thesis performance
        best_pnl = 0
        worst_pnl = 0
        if self.thesis_performance:
            performances = [stats['pnl'] for stats in self.thesis_performance.values()]
            best_pnl = max(performances) if performances else 0
            worst_pnl = min(performances) if performances else 0
        
        return np.array([
            float(trading_summary['trades_processed']),
            trading_summary['total_pnl'] / 100.0,
            trading_summary['win_rate'],
            float(len(self.thesis_performance)),  # Thesis diversity
            float(self.thesis_changes),
            best_pnl / 100.0,
            worst_pnl / 100.0,
            float(self._step_count)
        ], dtype=np.float32)
    
    def get_thesis_analysis(self) -> Dict[str, Any]:
        """Get comprehensive thesis analysis using mixins"""
        
        # Use analysis mixin for pattern analysis
        thesis_data = [
            {'thesis': thesis, 'pnl': stats['pnl'], 'trades': stats['trades']}
            for thesis, stats in self.thesis_performance.items()
        ]
        
        pattern_analysis = self._analyze_patterns(thesis_data, 'thesis')
        
        return {
            'current_thesis': self.current_thesis,
            'thesis_changes': self.thesis_changes,
            'pattern_analysis': pattern_analysis,
            'trading_summary': self._get_trading_summary(),
            'top_thesis': self._get_top_thesis(),
            'worst_thesis': self._get_worst_thesis()
        }
    
    def _get_top_thesis(self) -> Optional[str]:
        """Get best performing thesis"""
        if not self.thesis_performance:
            return None
        return max(self.thesis_performance.items(), key=lambda x: x[1]['pnl'])[0]
    
    def _get_worst_thesis(self) -> Optional[str]:
        """Get worst performing thesis"""
        if not self.thesis_performance:
            return None
        return min(self.thesis_performance.items(), key=lambda x: x[1]['pnl'])[0]