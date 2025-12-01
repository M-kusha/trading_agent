"""
Advanced ThemeExpert - Macro Theme and Risk Regime Voting Module.

This expert analyzes macro themes and market regimes including:
- Volatility regime analysis (ATR, historical vol, implied vol proxy)
- Correlation cluster analysis (cross-asset correlations)
- Trend strength aggregation (ADX across timeframes)
- Market breadth indicators
- Sentiment aggregation (composite scoring)
- Risk-on/Risk-off scoring

Per-Instrument Voting:
- Analyzes each instrument separately to produce per-instrument votes
- Different asset classes (FX, commodities) may have different regimes
- Each instrument gets its own action/confidence based on its own data

Actions: long, short, flat
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.utils.info_bus import InfoBusManager
from modules.voting.core.per_instrument import PerInstrumentVote, InstrumentProposal


def normalize_instrument(symbol: str) -> str:
    """Normalize instrument symbol to standard format."""
    if not symbol:
        return ""
    s = symbol.upper().replace("/", "").replace("_", "").replace("-", "").strip()
    # Map common variations
    mapping = {
        "EURUSD": "EURUSD",
        "XAUUSD": "XAUUSD",
        "GOLDUSD": "XAUUSD",
    }
    return mapping.get(s, s)


@module(**module_args("ThemeExpert"))
class ThemeExpert(BaseModule):
    """
    Advanced macro theme and regime analysis expert.
    
    Combines multiple regime detection methods to identify optimal
    trading themes and risk positioning.
    """
    
    def _initialize(self) -> None:
        """Initialize the theme expert with configuration."""
        self.smart_bus = InfoBusManager.get_instance()
        self.module_name = self.__class__.__name__
        
        # Instruments to analyze (from config or default)
        self.instruments = self.config.get('instruments', ['EURUSD', 'XAUUSD'])
        
        # Asset class mapping for different regime interpretations
        self.asset_classes = {
            'EURUSD': 'forex',
            'XAUUSD': 'commodity',
            'GBPUSD': 'forex',
            'USDJPY': 'forex',
        }
        
        # Volatility configuration
        self.atr_period = 20
        self.vol_lookback = 50
        self.vol_regime_threshold_low = 0.3
        self.vol_regime_threshold_high = 0.7
        
        # Correlation configuration
        self.corr_lookback = 30
        self.corr_cluster_threshold = 0.7
        
        # Trend configuration
        self.adx_period = 14
        self.trend_strength_threshold = 25.0
        self.strong_trend_threshold = 40.0
        
        # Risk scoring thresholds
        self.risk_on_threshold = 0.6
        self.risk_off_threshold = 0.4
        
        # Breadth configuration
        self.breadth_period = 20
        
        # Sentiment weights
        self.sentiment_weights = {
            'volatility': 0.25,
            'trend': 0.25,
            'correlation': 0.20,
            'breadth': 0.15,
            'momentum': 0.15
        }
        
        # Historical state for regime persistence
        self.regime_history: List[str] = []
        self.regime_persistence_count = 0
        self.min_regime_persistence = 3
        
        # Confidence calibration
        self.base_confidence = 0.5
        self.max_confidence = 0.95
        self.min_confidence = 0.15
        
        self.logger.info(f"ThemeExpert initialized with ATR period {self.atr_period}, instruments: {self.instruments}")
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Process market data to determine macro theme and regime PER INSTRUMENT.
        
        NEW: Analyzes each instrument separately to produce per-instrument votes.
        Different assets may have different volatility regimes and trends.
        
        Returns voting proposal with per-instrument actions and confidence.
        """
        try:
            market_data = self.smart_bus.get("market_data", self.module_name, default={})
            features = self.smart_bus.get("features", self.module_name, default={})
            
            if not market_data and not features:
                self.logger.warning("[THEME] No market data or features available")
                return self._neutral_output("No market data available")
            
            # ========== Per-Instrument Analysis ==========
            # Build a single PerInstrumentVote containing all instrument proposals
            per_instrument_vote = PerInstrumentVote(member=self.module_name)
            per_instrument_analysis: Dict[str, Dict] = {}
            
            for inst in self.instruments:
                inst_norm = normalize_instrument(inst)
                
                # Extract instrument-specific data
                inst_market = self._extract_instrument_data(market_data, inst)
                inst_features = self._extract_instrument_data(features, inst)
                
                # Get price arrays for this instrument
                close_prices = self._extract_prices(inst_market, inst_features, 'close', inst)
                high_prices = self._extract_prices(inst_market, inst_features, 'high', inst)
                low_prices = self._extract_prices(inst_market, inst_features, 'low', inst)
                
                if len(close_prices) < self.vol_lookback:
                    self.logger.debug(f"[THEME] Insufficient data for {inst}, using fallback")
                    # Create neutral proposal for this instrument
                    per_instrument_vote.set_proposal(InstrumentProposal(
                        instrument=inst_norm,
                        action='flat',
                        confidence=0.1,
                        magnitude=0.0,
                        rationale=f"Insufficient data for {inst}",
                    ))
                    per_instrument_analysis[inst_norm] = {
                        'volatility_regime': 'unknown',
                        'trend_regime': 'unknown',
                        'risk_regime': 'unknown',
                        'composite_score': 0.5,
                    }
                    continue
                
                # Calculate all regime components for THIS instrument
                vol_regime, vol_score = self._analyze_volatility_regime(
                    close_prices, high_prices, low_prices
                )
                
                trend_regime, trend_score = self._analyze_trend_regime(
                    close_prices, high_prices, low_prices
                )
                
                corr_regime, corr_score = self._analyze_correlation_regime(
                    close_prices, inst_market
                )
                
                breadth_score = self._calculate_market_breadth(
                    close_prices, high_prices, low_prices
                )
                
                momentum_score = self._calculate_momentum_score(close_prices)
                
                # Aggregate sentiment for this instrument
                composite_score = self._calculate_composite_sentiment(
                    vol_score, trend_score, corr_score, breadth_score, momentum_score
                )
                
                # Determine risk regime (with asset class consideration)
                asset_class = self.asset_classes.get(inst_norm, 'forex')
                risk_regime = self._determine_risk_regime(
                    vol_regime, trend_regime, composite_score, asset_class
                )
                
                # Select theme action FOR THIS INSTRUMENT
                action, confidence, thesis = self._select_theme_action(
                    vol_regime, trend_regime, risk_regime,
                    vol_score, trend_score, composite_score,
                    instrument=inst_norm
                )
                
                # Create per-instrument proposal
                per_instrument_vote.set_proposal(InstrumentProposal(
                    instrument=inst_norm,
                    action=action,
                    confidence=confidence,
                    magnitude=confidence,
                    rationale=thesis,
                ))
                
                # Store analysis for this instrument
                per_instrument_analysis[inst_norm] = {
                    'volatility_regime': vol_regime,
                    'trend_regime': trend_regime,
                    'risk_regime': risk_regime,
                    'vol_score': vol_score,
                    'trend_score': trend_score,
                    'composite_score': composite_score,
                    'action': action,
                    'confidence': confidence,
                }
                
                self.logger.debug(
                    f"[THEME] {inst}: vol={vol_regime}, trend={trend_regime}, "
                    f"action={action}, conf={confidence:.2f}"
                )
            
            # ========== Calculate Global Summary (backward compat) ==========
            # Use first instrument or average for global values
            if per_instrument_analysis:
                first_inst = list(per_instrument_analysis.keys())[0]
                global_analysis = per_instrument_analysis[first_inst]
            else:
                global_analysis = {
                    'volatility_regime': 'unknown',
                    'trend_regime': 'unknown',
                    'risk_regime': 'unknown',
                    'composite_score': 0.5,
                }
            
            # Calculate primary global vote (for backward compat)
            if per_instrument_vote.proposals:
                # Use the proposal with highest confidence as the global
                best_proposal = max(per_instrument_vote.proposals.values(), key=lambda p: p.confidence)
                global_action = best_proposal.action
                global_confidence = best_proposal.confidence
                global_thesis = best_proposal.rationale
            else:
                global_action = 'flat'
                global_confidence = 0.1
                global_thesis = "No instrument data available"
            
            # Build per-instrument proposals dict for CommitteeCoordinator
            proposals_dict = {}
            for inst, proposal in per_instrument_vote.proposals.items():
                proposals_dict[inst] = {
                    'action': proposal.action,
                    'confidence': proposal.confidence,
                    'magnitude': proposal.magnitude,
                    'rationale': proposal.rationale,
                }
            
            # Build proposal dict for voting (with per-instrument proposals)
            proposal = {
                "action": global_action,
                "signal_strength": global_confidence,
                "reason": global_thesis,
                "proposals": proposals_dict,  # NEW: Per-instrument proposals for committee
            }
            
            # Publish to SmartInfoBus for CommitteeCoordinator discovery
            name = self.__class__.__name__
            try:
                self.smart_bus.set('ThemeExpert_voting_proposal', proposal, module=name, thesis=global_thesis)
                self.smart_bus.set('ThemeExpert_confidence', global_confidence, module=name, thesis=f'Confidence: {global_confidence:.1%}')
                self.smart_bus.set('theme_voting_proposal', proposal, module=name, thesis=global_thesis)
                self.smart_bus.set('theme_confidence', global_confidence, module=name, thesis=f'Theme confidence: {global_confidence:.1%}')
                
                # NEW: Publish per-instrument votes for CommitteeCoordinator
                per_inst_votes_dict = {inst: prop.to_dict() for inst, prop in per_instrument_vote.proposals.items()}
                self.smart_bus.set(
                    'ThemeExpert_per_instrument_votes', 
                    per_inst_votes_dict, 
                    module=name, 
                    thesis=f'Per-instrument theme votes: {list(per_inst_votes_dict.keys())}'
                )
            except Exception:
                pass
            
            return {
                "ThemeExpert_voting_proposal": proposal,
                "ThemeExpert_confidence": global_confidence,
                "ThemeExpert_per_instrument_votes": {inst: prop.to_dict() for inst, prop in per_instrument_vote.proposals.items()},
                "per_instrument_votes": per_instrument_vote,  # PerInstrumentVote object with all proposals
                "theme_voting_proposal": proposal,  # Alias for contract compatibility
                "theme_confidence": global_confidence,   # Alias for contract compatibility
                "theme_analysis": {               # Required by contract
                    "volatility_regime": global_analysis.get('volatility_regime', 'unknown'),
                    "trend_regime": global_analysis.get('trend_regime', 'unknown'),
                    "risk_regime": global_analysis.get('risk_regime', 'unknown'),
                    "composite_score": global_analysis.get('composite_score', 0.5),
                    "action": global_action,
                    "confidence": global_confidence,
                    "per_instrument": per_instrument_analysis,  # NEW
                },
                "agreement_score": global_confidence,    # Required by contract
                "theme_expert_analysis": {        # Backward compat alias
                    "volatility_regime": global_analysis.get('volatility_regime', 'unknown'),
                    "trend_regime": global_analysis.get('trend_regime', 'unknown'),
                    "risk_regime": global_analysis.get('risk_regime', 'unknown'),
                    "composite_score": global_analysis.get('composite_score', 0.5),
                    "per_instrument": per_instrument_analysis,
                },
                "theme_expert_thesis": global_thesis,    # Backward compat alias
                "theme_volatility_regime": global_analysis.get('volatility_regime', 'unknown'),
                "theme_trend_regime": global_analysis.get('trend_regime', 'unknown'),
                "theme_risk_regime": global_analysis.get('risk_regime', 'unknown'),
                "theme_composite_score": global_analysis.get('composite_score', 0.5),
                "_thesis": global_thesis
            }
            
        except Exception as e:
            self.logger.error(f"ThemeExpert error: {e}")
            return self._neutral_output(f"Processing error: {str(e)}")
    
    def _extract_instrument_data(self, data: Dict, instrument: str) -> Dict:
        """Extract data for a specific instrument from nested market data."""
        if not isinstance(data, dict):
            return {}
        
        inst_norm = normalize_instrument(instrument)
        
        # Try direct instrument key
        for key in [instrument, inst_norm, instrument.upper(), instrument.lower()]:
            if key in data:
                return data[key] if isinstance(data[key], dict) else data
        
        # Try with separators
        for sep in ['_', '/', '-', '']:
            for pair in [f"EUR{sep}USD", f"XAU{sep}USD"]:
                norm_pair = normalize_instrument(pair)
                if norm_pair == inst_norm and pair in data:
                    return data[pair] if isinstance(data[pair], dict) else data
        
        # Return full data if no instrument-specific found (legacy format)
        return data
    
    def _extract_prices(
        self, 
        market_data: Dict, 
        features: Dict, 
        price_type: str,
        instrument: str = ''
    ) -> np.ndarray:
        """Extract price array from market data, features, or InfoBus.
        
        Args:
            market_data: Instrument-specific market data (may be pre-filtered)
            features: Instrument-specific features (may be pre-filtered)
            price_type: 'close', 'high', 'low', 'open'
            instrument: Target instrument (e.g., 'EURUSD', 'XAUUSD') for historical lookup
        """
        if isinstance(market_data, dict):
            if price_type in market_data:
                data = market_data[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)
            
            # Try nested structure
            for tf in ['H1', 'H4', 'D1']:
                if tf in market_data and isinstance(market_data[tf], dict):
                    if price_type in market_data[tf]:
                        data = market_data[tf][price_type]
                        if isinstance(data, (list, np.ndarray)):
                            return np.array(data, dtype=float)
        
        # Try features
        if isinstance(features, dict):
            if price_type in features:
                data = features[price_type]
                if isinstance(data, (list, np.ndarray)):
                    return np.array(data, dtype=float)

        try:
            historical = self.smart_bus.get("historical_prices", self.module_name, default=None)
        except Exception:
            historical = None

        if isinstance(historical, dict):
            # Map instrument to bus key aliases
            inst_aliases = {
                "EURUSD": ["EUR_USD", "EURUSD"],
                "XAUUSD": ["XAU_USD", "XAUUSD", "GOLDUSD"],
            }
            aliases = inst_aliases.get(instrument.upper(), [instrument, instrument.replace("USD", "_USD")])
            
            symbol = None
            for alias in aliases:
                if alias in historical:
                    symbol = alias
                    break
            
            # Fallback to first available if no instrument match
            if symbol is None and historical:
                symbol = next(iter(historical.keys()))

            if symbol is not None:
                sym_block = historical.get(symbol)
                if isinstance(sym_block, dict):
                    tf_rec = None
                    for tf in ("H4", "H1", "D1"):
                        candidate_rec = sym_block.get(tf)
                        if isinstance(candidate_rec, dict):
                            tf_rec = candidate_rec
                            break
                    if tf_rec is None and sym_block:
                        tf_rec = sym_block.get(next(iter(sym_block.keys())))
                    if isinstance(tf_rec, dict):
                        seq = tf_rec.get(price_type)
                        if isinstance(seq, (list, np.ndarray)):
                            return np.array(seq, dtype=float)
        
        return np.array([])
    
    def _analyze_volatility_regime(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray
    ) -> Tuple[str, float]:
        """
        Analyze volatility regime using ATR and historical volatility.
        
        Returns:
            regime: 'low_vol', 'normal_vol', 'high_vol', 'extreme_vol'
            score: 0-1 normalized volatility score
        """
        # Calculate ATR
        if len(high) >= self.atr_period and len(low) >= self.atr_period:
            tr_list = []
            for i in range(1, min(self.atr_period + 1, len(close))):
                tr_val = max(
                    high[-self.atr_period + i] - low[-self.atr_period + i],
                    abs(high[-self.atr_period + i] - close[-self.atr_period + i - 1]),
                    abs(low[-self.atr_period + i] - close[-self.atr_period + i - 1])
                )
                tr_list.append(tr_val)
            atr = np.mean(tr_list) if tr_list else np.std(close[-20:])
        else:
            atr = np.std(close[-20:]) if len(close) >= 20 else np.std(close)
        
        # Calculate historical volatility (annualized)
        if len(close) >= self.vol_lookback:
            returns = np.diff(np.log(close[-self.vol_lookback:]))
        else:
            returns = np.diff(np.log(close))
        hist_vol = np.std(returns) * np.sqrt(252)
        
        # Calculate realized volatility percentile
        if len(close) >= self.vol_lookback * 2:
            rolling_vols = []
            for i in range(self.vol_lookback, len(close)):
                window_returns = np.diff(np.log(close[i-self.vol_lookback:i]))
                rolling_vols.append(np.std(window_returns) * np.sqrt(252))
            
            if rolling_vols and max(rolling_vols) > min(rolling_vols):
                current_percentile = (hist_vol - min(rolling_vols)) / (max(rolling_vols) - min(rolling_vols) + 1e-8)
            else:
                current_percentile = 0.5
        else:
            current_percentile = 0.5
        
        # Normalize ATR as percentage of price
        atr_pct = atr / close[-1] if close[-1] > 0 else 0
        
        # Combine for final score
        vol_score = (current_percentile * 0.6 + min(atr_pct * 100, 1.0) * 0.4)
        vol_score = np.clip(vol_score, 0, 1)
        
        # Determine regime
        if vol_score < self.vol_regime_threshold_low:
            regime = "low_vol"
        elif vol_score > self.vol_regime_threshold_high:
            if vol_score > 0.9:
                regime = "extreme_vol"
            else:
                regime = "high_vol"
        else:
            regime = "normal_vol"
        
        return regime, float(vol_score)
    
    def _analyze_trend_regime(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray
    ) -> Tuple[str, float]:
        """
        Analyze trend regime using ADX and directional movement.
        
        Returns:
            regime: 'strong_uptrend', 'weak_uptrend', 'ranging', 
                    'weak_downtrend', 'strong_downtrend'
            score: -1 to 1 directional score
        """
        n = len(close)
        if n < self.adx_period + 1:
            return "ranging", 0.0
        
        # Calculate directional movement
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        tr = np.zeros(n)
        
        for i in range(1, n):
            up_move = high[i] - high[i-1]
            down_move = low[i-1] - low[i]
            
            if up_move > down_move and up_move > 0:
                plus_dm[i] = up_move
            if down_move > up_move and down_move > 0:
                minus_dm[i] = down_move
            
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        
        # Smooth with Wilder's method
        smoothed_plus_dm = self._wilder_smooth(plus_dm, self.adx_period)
        smoothed_minus_dm = self._wilder_smooth(minus_dm, self.adx_period)
        smoothed_tr = self._wilder_smooth(tr, self.adx_period)
        
        # Calculate +DI and -DI
        plus_di = 100 * smoothed_plus_dm / (smoothed_tr + 1e-8)
        minus_di = 100 * smoothed_minus_dm / (smoothed_tr + 1e-8)
        
        # Calculate DX and ADX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = self._wilder_smooth(dx, self.adx_period)
        
        current_adx = adx[-1]
        current_plus_di = plus_di[-1]
        current_minus_di = minus_di[-1]
        
        # Calculate directional score (-1 to 1)
        di_diff = current_plus_di - current_minus_di
        di_sum = current_plus_di + current_minus_di + 1e-8
        direction = di_diff / di_sum
        
        # Weight by trend strength
        trend_score = direction * (current_adx / 50.0)
        trend_score = np.clip(trend_score, -1, 1)
        
        # Determine regime
        if current_adx < self.trend_strength_threshold:
            regime = "ranging"
        elif current_adx >= self.strong_trend_threshold:
            regime = "strong_uptrend" if direction > 0 else "strong_downtrend"
        else:
            regime = "weak_uptrend" if direction > 0 else "weak_downtrend"
        
        return regime, float(trend_score)
    
    def _wilder_smooth(self, data: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing method."""
        result = np.zeros_like(data)
        result[:period] = np.cumsum(data[:period])
        if period > 0:
            result[period-1] = result[period-1] / period
        
        alpha = 1.0 / period if period > 0 else 1.0
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        
        return result
    
    def _analyze_correlation_regime(
        self,
        close: np.ndarray,
        market_data: Dict
    ) -> Tuple[str, float]:
        """
        Analyze correlation regime across assets/timeframes.
        
        Returns:
            regime: 'risk_on_correlation', 'risk_off_correlation', 'decorrelated'
            score: correlation clustering score
        """
        # Try to get multi-timeframe data for correlation
        prices_by_tf = {}
        
        if isinstance(market_data, dict):
            for tf in ['H1', 'H4', 'D1']:
                if tf in market_data and isinstance(market_data[tf], dict):
                    if 'close' in market_data[tf]:
                        tf_close = np.array(market_data[tf]['close'], dtype=float)
                        if len(tf_close) >= self.corr_lookback:
                            prices_by_tf[tf] = tf_close[-self.corr_lookback:]
        
        # If we have multiple timeframes, calculate cross-correlations
        if len(prices_by_tf) >= 2:
            correlations = []
            tfs = list(prices_by_tf.keys())
            
            for i in range(len(tfs)):
                for j in range(i+1, len(tfs)):
                    min_len = min(len(prices_by_tf[tfs[i]]), len(prices_by_tf[tfs[j]]))
                    r1 = np.diff(np.log(prices_by_tf[tfs[i]][-min_len:]))
                    r2 = np.diff(np.log(prices_by_tf[tfs[j]][-min_len:]))
                    
                    if len(r1) > 5 and len(r2) > 5:
                        corr = np.corrcoef(r1, r2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            if correlations:
                avg_corr = np.mean(correlations)
                
                if avg_corr > self.corr_cluster_threshold:
                    regime = "risk_on_correlation"
                    score = avg_corr
                elif avg_corr < -self.corr_cluster_threshold:
                    regime = "risk_off_correlation"
                    score = avg_corr
                else:
                    regime = "decorrelated"
                    score = avg_corr
                
                return regime, float(score)
        
        # Fallback: analyze autocorrelation of returns
        if len(close) >= self.corr_lookback:
            returns = np.diff(np.log(close[-self.corr_lookback:]))
            
            if len(returns) > 1:
                # Lag-1 autocorrelation
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                
                if not np.isnan(autocorr):
                    if autocorr > 0.3:
                        return "risk_on_correlation", float(autocorr)
                    elif autocorr < -0.3:
                        return "risk_off_correlation", float(autocorr)
        
        return "decorrelated", 0.0
    
    def _calculate_market_breadth(
        self,
        close: np.ndarray,
        high: np.ndarray,
        low: np.ndarray
    ) -> float:
        """
        Calculate market breadth score.
        
        Uses price position within range and trend consistency.
        """
        if len(close) < self.breadth_period:
            return 0.5
        
        period_close = close[-self.breadth_period:]
        if len(high) >= self.breadth_period:
            period_high = high[-self.breadth_period:]
        else:
            period_high = period_close
        if len(low) >= self.breadth_period:
            period_low = low[-self.breadth_period:]
        else:
            period_low = period_close
        
        # Price position within period range
        period_range = np.max(period_high) - np.min(period_low)
        if period_range > 0:
            position_score = (close[-1] - np.min(period_low)) / period_range
        else:
            position_score = 0.5
        
        # Count up days vs down days
        returns = np.diff(period_close)
        up_days = np.sum(returns > 0)
        down_days = np.sum(returns < 0)
        total_days = up_days + down_days
        
        if total_days > 0:
            breadth_ratio = up_days / total_days
        else:
            breadth_ratio = 0.5
        
        # New highs vs new lows (simplified)
        if len(close) >= 5:
            recent_high = np.max(close[-5:])
        else:
            recent_high = close[-1]
        period_high_max = np.max(period_high)
        new_high_score = 1.0 if recent_high >= period_high_max else 0.5
        
        if len(close) >= 5:
            recent_low = np.min(close[-5:])
        else:
            recent_low = close[-1]
        period_low_min = np.min(period_low)
        new_low_score = 0.0 if recent_low <= period_low_min else 0.5
        
        # Combine scores
        breadth_score = (
            position_score * 0.35 +
            breadth_ratio * 0.35 +
            new_high_score * 0.15 +
            new_low_score * 0.15
        )
        
        return float(np.clip(breadth_score, 0, 1))
    
    def _calculate_momentum_score(self, close: np.ndarray) -> float:
        """Calculate momentum score using rate of change."""
        if len(close) < 20:
            return 0.5
        
        # Multi-period ROC
        roc_5 = (close[-1] - close[-6]) / close[-6] if close[-6] > 0 else 0
        roc_10 = 0
        if len(close) > 10 and close[-11] > 0:
            roc_10 = (close[-1] - close[-11]) / close[-11]
        roc_20 = 0
        if len(close) > 20 and close[-21] > 0:
            roc_20 = (close[-1] - close[-21]) / close[-21]
        
        # Normalize to 0-1 range (assuming max 10% move)
        norm_roc_5 = np.clip(roc_5 * 10 + 0.5, 0, 1)
        norm_roc_10 = np.clip(roc_10 * 5 + 0.5, 0, 1)
        norm_roc_20 = np.clip(roc_20 * 2.5 + 0.5, 0, 1)
        
        # Weighted average
        momentum_score = norm_roc_5 * 0.5 + norm_roc_10 * 0.3 + norm_roc_20 * 0.2
        
        return float(momentum_score)
    
    def _calculate_composite_sentiment(
        self,
        vol_score: float,
        trend_score: float,
        corr_score: float,
        breadth_score: float,
        momentum_score: float
    ) -> float:
        """
        Calculate composite sentiment score.
        
        Combines all regime scores into single 0-1 sentiment value.
        """
        # Normalize trend score from -1,1 to 0,1
        norm_trend = (trend_score + 1) / 2
        
        # Invert volatility (high vol = lower sentiment)
        inv_vol = 1 - vol_score
        
        # Normalize correlation (positive = risk-on)
        norm_corr = (corr_score + 1) / 2
        
        composite = (
            inv_vol * self.sentiment_weights['volatility'] +
            norm_trend * self.sentiment_weights['trend'] +
            norm_corr * self.sentiment_weights['correlation'] +
            breadth_score * self.sentiment_weights['breadth'] +
            momentum_score * self.sentiment_weights['momentum']
        )
        
        return float(np.clip(composite, 0, 1))
    
    def _determine_risk_regime(
        self,
        vol_regime: str,
        trend_regime: str,
        composite_score: float,
        asset_class: str = 'forex'
    ) -> str:
        """
        Determine overall risk regime.
        
        For commodities (like Gold), risk-off can mean LONG (safe haven).
        For forex, risk-off typically means cautious/neutral.
        """
        if vol_regime == "extreme_vol":
            return "risk_off"
        
        if composite_score >= self.risk_on_threshold:
            if trend_regime in ["strong_uptrend", "weak_uptrend"]:
                return "risk_on"
            return "neutral"
        elif composite_score <= self.risk_off_threshold:
            if trend_regime in ["strong_downtrend", "weak_downtrend"]:
                return "risk_off"
            return "cautious"
        
        return "neutral"
    
    def _select_theme_action(
        self,
        vol_regime: str,
        trend_regime: str,
        risk_regime: str,
        vol_score: float,
        trend_score: float,
        composite_score: float,
        instrument: str = ''
    ) -> Tuple[str, float, str]:
        """
        Select trading theme action based on regime analysis.
        
        NEW: Now instrument-aware. Gold behaves as safe-haven in risk-off.
        
        Returns: (action, confidence, thesis) - action is 'long', 'short', or 'flat'
        """
        inst_norm = normalize_instrument(instrument) if instrument else ''
        is_safe_haven = inst_norm in ['XAUUSD', 'GOLD']  # Gold is safe haven
        
        # Extreme volatility: stay flat/hedge
        if vol_regime == "extreme_vol":
            # Exception: Gold can be LONG in extreme vol (safe haven flow)
            if is_safe_haven:
                confidence = 0.6 + (vol_score - 0.9) * 2
                return (
                    "long",
                    np.clip(confidence, 0.5, 0.75),
                    f"{inst_norm}: Safe haven bid during extreme volatility (vol: {vol_score:.2f})"
                )
            confidence = 0.7 + (vol_score - 0.9) * 3
            return (
                "flat",  # Standard neutral action
                np.clip(confidence, 0.6, 0.9),
                f"{inst_norm}: Extreme volatility (score: {vol_score:.2f}), defensive positioning"
            )
        
        # Strong uptrend: go long
        if trend_regime == "strong_uptrend":
            confidence = self.base_confidence + abs(trend_score) * 0.4
            return (
                "long",  # Standard bullish action
                np.clip(confidence, 0.5, 0.85),
                f"{inst_norm}: Strong uptrend (ADX: {abs(trend_score):.2f}), trend following"
            )
        
        # Strong downtrend: go short (but Gold may be long as safe haven)
        if trend_regime == "strong_downtrend":
            confidence = self.base_confidence + abs(trend_score) * 0.4
            # Safe haven exception: Gold in strong downtrend of risk assets
            if is_safe_haven and risk_regime in ["risk_off", "cautious"]:
                return (
                    "long",
                    np.clip(confidence * 0.8, 0.4, 0.7),
                    f"{inst_norm}: Safe haven bid during risk-off (trend: {trend_score:.2f})"
                )
            return (
                "short",  # Standard bearish action
                np.clip(confidence, 0.5, 0.85),
                f"{inst_norm}: Strong downtrend (ADX: {abs(trend_score):.2f}), bearish"
            )
        
        # Risk-on environment: go long
        if risk_regime == "risk_on" and composite_score > 0.55:  # Lowered threshold
            confidence = self.base_confidence + (composite_score - 0.5) * 0.6
            return (
                "long",  # Standard bullish action
                np.clip(confidence, 0.5, 0.8),
                f"{inst_norm}: Risk-on (composite: {composite_score:.2f}), bullish"
            )
        
        # Risk-off environment
        if risk_regime in ["risk_off", "cautious"] and composite_score < 0.45:
            confidence = self.base_confidence + (0.5 - composite_score) * 0.6
            # Safe haven: LONG in risk-off
            if is_safe_haven:
                return (
                    "long",
                    np.clip(confidence, 0.5, 0.75),
                    f"{inst_norm}: Safe haven bid in risk-off (composite: {composite_score:.2f})"
                )
            return (
                "short",  # Standard bearish action
                np.clip(confidence, 0.5, 0.8),
                f"{inst_norm}: Risk-off (composite: {composite_score:.2f}), bearish"
            )
        
        # Weak uptrend: cautious long
        if trend_regime == "weak_uptrend":
            return (
                "long",
                0.45,
                f"{inst_norm}: Weak uptrend, cautious bullish"
            )
        
        # Weak downtrend: cautious short
        if trend_regime == "weak_downtrend":
            if is_safe_haven:
                return (
                    "flat",
                    0.35,
                    f"{inst_norm}: Weak downtrend but safe haven - neutral"
                )
            return (
                "short",
                0.45,
                f"{inst_norm}: Weak downtrend, cautious bearish"
            )
        
        # Ranging market: use composite score for direction
        if trend_regime == "ranging":
            if composite_score > 0.5:
                return (
                    "long",
                    0.35,
                    f"{inst_norm}: Ranging with bullish bias (composite: {composite_score:.2f})"
                )
            elif composite_score < 0.5:
                return (
                    "short",
                    0.35,
                    f"{inst_norm}: Ranging with bearish bias (composite: {composite_score:.2f})"
                )
            else:
                return (
                    "flat",
                    0.3,
                    f"{inst_norm}: Ranging - neutral composite"
                )
        
        # Default: use composite score for direction
        if composite_score > 0.5:
            return (
                "long",
                0.3,
                f"{inst_norm}: Mixed signals favoring long (composite: {composite_score:.2f})"
            )
        elif composite_score < 0.5:
            return (
                "short",
                0.3,
                f"{inst_norm}: Mixed signals favoring short (composite: {composite_score:.2f})"
            )
        
        # Truly neutral - rare
        return (
            "flat",
            0.25,
            f"{inst_norm}: Neutral - vol: {vol_regime}, trend: {trend_regime}, risk: {risk_regime}"
        )
    
    def _apply_persistence_filter(
        self,
        action: str,
        confidence: float
    ) -> Tuple[str, float]:
        """
        Apply regime persistence filter to reduce whipsaws.
        
        Boosts confidence for persistent regimes, reduces for new ones.
        """
        if len(self.regime_history) > 0 and self.regime_history[-1] == action:
            self.regime_persistence_count += 1
            
            # Boost confidence for persistent regime
            if self.regime_persistence_count >= self.min_regime_persistence:
                boost = min(0.1, self.regime_persistence_count * 0.02)
                confidence = min(confidence + boost, self.max_confidence)
        else:
            # New regime - reduce confidence initially
            self.regime_persistence_count = 1
            confidence = max(confidence * 0.85, self.min_confidence)
        
        # Update history
        self.regime_history.append(action)
        if len(self.regime_history) > 20:
            self.regime_history = self.regime_history[-20:]
        
        return action, confidence
    
    def _neutral_output(self, reason: str) -> Dict[str, Any]:
        """Generate neutral output with explanation."""
        proposal = "flat"
        confidence = 0.1
        thesis = f"Theme flat: {reason}"
        
        # Publish to SmartInfoBus even when neutral (prevents stale keys)
        try:
            self.smart_bus.set('ThemeExpert_voting_proposal', proposal, 
                              module=self.module_name, thesis=thesis)
            self.smart_bus.set('ThemeExpert_confidence', confidence, 
                              module=self.module_name, thesis=f'Confidence: {confidence:.1%}')
            self.smart_bus.set('theme_voting_proposal', proposal, 
                              module=self.module_name, thesis=thesis)
            self.smart_bus.set('theme_confidence', confidence, 
                              module=self.module_name, thesis=f'Theme confidence: {confidence:.1%}')
        except Exception:
            pass
        
        return {
            "ThemeExpert_voting_proposal": proposal,
            "ThemeExpert_confidence": confidence,
            "theme_voting_proposal": proposal,  # Alias for contract compatibility
            "theme_confidence": confidence,                   # Alias for contract compatibility
            "theme_analysis": {                        # Required by contract
                "volatility_regime": "unknown",
                "trend_regime": "unknown",
                "risk_regime": "unknown",
                "composite_score": 0.5,
                "action": proposal,
                "confidence": confidence
            },
            "agreement_score": confidence,                    # Required by contract
            "theme_expert_analysis": {                 # Backward compat alias
                "volatility_regime": "unknown",
                "trend_regime": "unknown",
                "risk_regime": "unknown",
                "composite_score": 0.5
            },
            "theme_expert_thesis": thesis,  # Backward compat alias
            "theme_volatility_regime": "unknown",
            "theme_trend_regime": "unknown",
            "theme_risk_regime": "unknown",
            "theme_composite_score": 0.5,
            "_thesis": thesis
        }

