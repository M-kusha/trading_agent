"""
Advanced SeasonalityRiskExpert - Time-Based Pattern Voting Module.

This expert analyzes seasonal and temporal patterns including:
- Day-of-week patterns (Monday effect, Friday positioning)
- Hour-of-day patterns (session overlaps, volatility windows)
- Month-of-year patterns (January effect, summer doldrums)
- Session analysis (Asian/European/American sessions)
- Holiday calendar effects (reduced liquidity periods)
- Economic calendar awareness (high-impact event windows)
- Rollover and swap timing

Per-Instrument Voting:
- Different assets have different seasonal patterns
- Gold has different seasonality than EURUSD
- Each instrument gets its own vote based on asset-specific patterns

Actions: long, short, flat
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, time, timedelta
import calendar

from modules.contracts import module_args
from modules.core.module_base import BaseModule, module
from modules.utils.info_bus import InfoBusManager
from modules.voting.core.per_instrument import PerInstrumentVote, InstrumentProposal


def normalize_instrument(symbol: str) -> str:
    """Normalize instrument symbol to standard format."""
    if not symbol:
        return ""
    s = symbol.upper().replace("/", "").replace("_", "").replace("-", "").strip()
    mapping = {
        "EURUSD": "EURUSD",
        "XAUUSD": "XAUUSD",
        "GOLDUSD": "XAUUSD",
    }
    return mapping.get(s, s)


@module(**module_args("SeasonalityRiskExpert"))
class SeasonalityRiskExpert(BaseModule):
    """
    Advanced seasonality and time-pattern analysis expert.
    
    Combines multiple temporal factors to identify optimal/suboptimal
    trading windows and seasonal biases.
    """
    
    def _initialize(self) -> None:
        """Initialize the seasonality expert with configuration."""
        self.smart_bus = InfoBusManager.get_instance()
        self.module_name = self.__class__.__name__
        
        # Instruments to analyze (from config or default)
        self.instruments = self.config.get('instruments', ['EURUSD', 'XAUUSD'])
        
        # Asset class mapping for different seasonal patterns
        self.asset_classes = {
            'EURUSD': 'forex',
            'XAUUSD': 'commodity',
            'GBPUSD': 'forex',
            'USDJPY': 'forex',
        }
        
        # Session times (UTC)
        self.sessions = {
            'asian': {'start': time(0, 0), 'end': time(9, 0)},
            'european': {'start': time(7, 0), 'end': time(16, 0)},
            'american': {'start': time(13, 0), 'end': time(22, 0)},
            'overlap_eu_us': {'start': time(13, 0), 'end': time(16, 0)},
            'overlap_asia_eu': {'start': time(7, 0), 'end': time(9, 0)}
        }
        
        # Day-of-week biases (0=Monday, 4=Friday)
        # Based on historical FX patterns
        self.dow_biases = {
            0: {'name': 'Monday', 'volatility': 0.8, 'trend_continuation': 0.6, 'reversal_risk': 0.4},
            1: {'name': 'Tuesday', 'volatility': 1.0, 'trend_continuation': 0.7, 'reversal_risk': 0.3},
            2: {'name': 'Wednesday', 'volatility': 1.1, 'trend_continuation': 0.8, 'reversal_risk': 0.3},
            3: {'name': 'Thursday', 'volatility': 1.0, 'trend_continuation': 0.65, 'reversal_risk': 0.35},
            4: {'name': 'Friday', 'volatility': 0.9, 'trend_continuation': 0.5, 'reversal_risk': 0.5}
        }
        
        # Monthly patterns
        self.monthly_patterns = {
            1: {'name': 'January', 'trend_strength': 1.2, 'risk_on': True},
            2: {'name': 'February', 'trend_strength': 1.0, 'risk_on': True},
            3: {'name': 'March', 'trend_strength': 1.1, 'risk_on': True},
            4: {'name': 'April', 'trend_strength': 1.0, 'risk_on': True},
            5: {'name': 'May', 'trend_strength': 0.8, 'risk_on': False},
            6: {'name': 'June', 'trend_strength': 0.7, 'risk_on': False},
            7: {'name': 'July', 'trend_strength': 0.6, 'risk_on': False},
            8: {'name': 'August', 'trend_strength': 0.5, 'risk_on': False},
            9: {'name': 'September', 'trend_strength': 1.0, 'risk_on': False},
            10: {'name': 'October', 'trend_strength': 1.1, 'risk_on': True},
            11: {'name': 'November', 'trend_strength': 1.0, 'risk_on': True},
            12: {'name': 'December', 'trend_strength': 0.6, 'risk_on': False}
        }
        
        # High-volatility hours (UTC) - typically news releases
        self.high_impact_hours = [8, 12, 13, 14, 18]  # EU open, US data, US open, etc.
        
        # Rollover window (usually around 21:00-22:00 UTC)
        self.rollover_start = time(21, 0)
        self.rollover_end = time(22, 0)
        
        # Weekend gap risk window (Friday after 20:00 UTC)
        self.weekend_risk_start = time(20, 0)
        
        # Historical pattern tracking
        self.pattern_history: List[Dict] = []
        self.pattern_accuracy: Dict[str, float] = {}
        
        # Confidence parameters
        self.base_confidence = 0.4
        self.max_confidence = 0.85
        self.min_confidence = 0.15
        
        # Session quality weights
        self.session_weights = {
            'overlap_eu_us': 1.3,
            'european': 1.1,
            'american': 1.0,
            'overlap_asia_eu': 0.9,
            'asian': 0.8
        }
        
        self.logger.info(f"SeasonalityRiskExpert initialized with temporal analysis, instruments: {self.instruments}")
    
    async def process(self, **inputs) -> Dict[str, Any]:
        """
        Process temporal data to determine seasonal biases and timing PER INSTRUMENT.
        
        NEW: Analyzes each instrument separately since different assets have
        different seasonal patterns (e.g., Gold vs EURUSD).
        
        Returns voting proposal with per-instrument actions and confidence.
        """
        try:
            # Get current time (use system time or from market data)
            current_time = datetime.utcnow()
            
            market_data = self.smart_bus.get("market_data", self.module_name, default={})
            features = self.smart_bus.get("features", self.module_name, default={})
            
            # Analyze all temporal components (these are GLOBAL - time is the same for all instruments)
            session_analysis = self._analyze_session(current_time)
            dow_analysis = self._analyze_day_of_week(current_time)
            monthly_analysis = self._analyze_monthly_pattern(current_time)
            hour_analysis = self._analyze_hour_patterns(current_time)
            
            # Check special conditions (GLOBAL)
            rollover_risk = self._check_rollover_risk(current_time)
            weekend_risk = self._check_weekend_risk(current_time)
            high_impact_window = self._check_high_impact_window(current_time)
            
            # ========== Per-Instrument Analysis ==========
            per_instrument_vote = PerInstrumentVote(member=self.module_name)
            per_instrument_analysis: Dict[str, Dict] = {}
            
            for inst in self.instruments:
                inst_norm = normalize_instrument(inst)
                asset_class = self.asset_classes.get(inst_norm, 'forex')
                
                # Extract instrument-specific market data for historical pattern
                inst_market = self._extract_instrument_data(market_data, inst)
                inst_features = self._extract_instrument_data(features, inst)
                
                # Calculate historical pattern score for THIS instrument
                historical_score = self._calculate_historical_pattern_score(
                    current_time, inst_market, inst_features, inst
                )
                
                # Calculate composite temporal score
                composite_score = self._calculate_composite_score(
                    session_analysis,
                    dow_analysis,
                    monthly_analysis,
                    hour_analysis,
                    historical_score
                )
                
                # Select action based on analysis FOR THIS INSTRUMENT
                action, confidence, thesis = self._select_seasonal_action(
                    composite_score,
                    session_analysis,
                    dow_analysis,
                    monthly_analysis,
                    rollover_risk,
                    weekend_risk,
                    high_impact_window,
                    instrument=inst_norm,
                    asset_class=asset_class
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
                    'session': session_analysis['current_session'],
                    'dow_bias': dow_analysis['bias'],
                    'monthly_pattern': monthly_analysis['pattern'],
                    'composite_score': composite_score,
                    'historical_score': historical_score,
                    'action': action,
                    'confidence': confidence,
                }
                
                self.logger.debug(
                    f"[SEASONALITY] {inst}: session={session_analysis['current_session']}, "
                    f"action={action}, conf={confidence:.2f}"
                )
            
            # ========== Calculate Global Summary (backward compat) ==========
            # Use highest confidence vote as global
            if per_instrument_vote.proposals:
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
                self.smart_bus.set('SeasonalityRiskExpert_voting_proposal', proposal, module=name, thesis=global_thesis)
                self.smart_bus.set('SeasonalityRiskExpert_confidence', global_confidence, module=name, thesis=f'Confidence: {global_confidence:.1%}')
                self.smart_bus.set('seasonality_voting_proposal', proposal, module=name, thesis=global_thesis)
                self.smart_bus.set('seasonality_confidence', global_confidence, module=name, thesis=f'Seasonality confidence: {global_confidence:.1%}')
                
                # NEW: Publish per-instrument votes for CommitteeCoordinator
                per_inst_votes_dict = {inst: p.to_dict() for inst, p in per_instrument_vote.proposals.items()}
                self.smart_bus.set(
                    'SeasonalityRiskExpert_per_instrument_votes',
                    per_inst_votes_dict,
                    module=name,
                    thesis=f'Per-instrument seasonality votes: {list(per_inst_votes_dict.keys())}'
                )
            except Exception:
                pass
            
            return {
                "SeasonalityRiskExpert_voting_proposal": proposal,
                "SeasonalityRiskExpert_confidence": global_confidence,
                "SeasonalityRiskExpert_per_instrument_votes": {inst: p.to_dict() for inst, p in per_instrument_vote.proposals.items()},
                "per_instrument_votes": per_instrument_vote,  # PerInstrumentVote object
                "seasonality_voting_proposal": proposal,   # Alias for contract compatibility
                "seasonality_confidence": global_confidence,     # Alias for contract compatibility
                "seasonal_voting_proposal": proposal,       # Additional alias
                "seasonal_confidence": global_confidence,        # Additional alias
                "seasonality_risk_analysis": {            # Required by contract
                    "session": session_analysis['current_session'],
                    "dow_bias": dow_analysis['bias'],
                    "monthly_pattern": monthly_analysis['pattern'],
                    "composite_score": per_instrument_analysis.get(
                        self.instruments[0] if self.instruments else 'EURUSD', {}
                    ).get('composite_score', 0.5),
                    "rollover_risk": rollover_risk,
                    "weekend_risk": weekend_risk,
                    "action": global_action,
                    "confidence": global_confidence,
                    "per_instrument": per_instrument_analysis,  # NEW
                },
                "seasonality_analysis": {                 # Alias
                    "session": session_analysis['current_session'],
                    "dow_bias": dow_analysis['bias'],
                    "monthly_pattern": monthly_analysis['pattern'],
                    "composite_score": 0.5,
                    "per_instrument": per_instrument_analysis,
                },
                "seasonality_expert_analysis": {          # Backward compat alias
                    "session": session_analysis['current_session'],
                    "dow_bias": dow_analysis['bias'],
                    "monthly_pattern": monthly_analysis['pattern'],
                    "composite_score": 0.5,
                    "rollover_risk": rollover_risk,
                    "weekend_risk": weekend_risk,
                    "per_instrument": per_instrument_analysis,
                },
                "seasonality_expert_thesis": global_thesis,      # Backward compat alias
                "seasonal_session": session_analysis['current_session'],
                "seasonal_dow_bias": dow_analysis['bias'],
                "seasonal_monthly_pattern": monthly_analysis['pattern'],
                "seasonal_composite_score": 0.5,
                "seasonal_rollover_risk": rollover_risk,
                "seasonal_weekend_risk": weekend_risk,
                "_thesis": global_thesis
            }
            
        except Exception as e:
            self.logger.error(f"SeasonalityRiskExpert error: {e}")
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
        
        # Return full data if no instrument-specific found (legacy format)
        return data
    
    def _analyze_session(self, current_time: datetime) -> Dict[str, Any]:
        """
        Analyze current trading session and quality.
        
        Returns session info and quality score.
        """
        current_hour = current_time.time()
        active_sessions = []
        
        for session_name, times in self.sessions.items():
            if self._time_in_range(current_hour, times['start'], times['end']):
                active_sessions.append(session_name)
        
        # Determine primary session
        if 'overlap_eu_us' in active_sessions:
            primary_session = 'overlap_eu_us'
            session_quality = 1.0
        elif 'overlap_asia_eu' in active_sessions:
            primary_session = 'overlap_asia_eu'
            session_quality = 0.85
        elif 'european' in active_sessions:
            primary_session = 'european'
            session_quality = 0.9
        elif 'american' in active_sessions:
            primary_session = 'american'
            session_quality = 0.85
        elif 'asian' in active_sessions:
            primary_session = 'asian'
            session_quality = 0.7
        else:
            primary_session = 'off_hours'
            session_quality = 0.4
        
        # Calculate liquidity score
        liquidity_score = self.session_weights.get(primary_session, 0.5)
        
        # Determine session phase (early, mid, late)
        session_phase = self._get_session_phase(current_time, primary_session)
        
        return {
            'current_session': primary_session,
            'active_sessions': active_sessions,
            'session_quality': session_quality,
            'liquidity_score': liquidity_score,
            'session_phase': session_phase
        }
    
    def _time_in_range(self, current: time, start: time, end: time) -> bool:
        """Check if current time is in range (handles overnight sessions)."""
        if start <= end:
            return start <= current <= end
        else:
            return current >= start or current <= end
    
    def _get_session_phase(self, current_time: datetime, session: str) -> str:
        """Determine if we're in early, mid, or late session phase."""
        if session not in self.sessions:
            return 'unknown'
        
        session_times = self.sessions[session]
        start = session_times['start']
        end = session_times['end']
        
        current = current_time.time()
        
        # Calculate session duration in minutes
        start_mins = start.hour * 60 + start.minute
        end_mins = end.hour * 60 + end.minute
        current_mins = current.hour * 60 + current.minute
        
        if end_mins < start_mins:
            end_mins += 24 * 60
            if current_mins < start_mins:
                current_mins += 24 * 60
        
        duration = end_mins - start_mins
        elapsed = current_mins - start_mins
        
        if duration <= 0:
            return 'unknown'
        
        progress = elapsed / duration
        
        if progress < 0.33:
            return 'early'
        elif progress < 0.67:
            return 'mid'
        else:
            return 'late'
    
    def _analyze_day_of_week(self, current_time: datetime) -> Dict[str, Any]:
        """
        Analyze day-of-week patterns and biases.
        """
        dow = current_time.weekday()
        dow_info = self.dow_biases.get(dow, self.dow_biases[1])
        
        # Calculate directional bias based on typical patterns
        if dow == 0:  # Monday
            bias = 'cautious'
            continuation_prob = 0.6
        elif dow == 4:  # Friday
            bias = 'closing_bias'
            continuation_prob = 0.5
        elif dow in [1, 2]:  # Tuesday, Wednesday
            bias = 'trending'
            continuation_prob = 0.75
        else:  # Thursday
            bias = 'neutral'
            continuation_prob = 0.65
        
        return {
            'day_name': dow_info['name'],
            'day_number': dow,
            'bias': bias,
            'volatility_factor': dow_info['volatility'],
            'trend_continuation': dow_info['trend_continuation'],
            'reversal_risk': dow_info['reversal_risk'],
            'continuation_probability': continuation_prob
        }
    
    def _analyze_monthly_pattern(self, current_time: datetime) -> Dict[str, Any]:
        """
        Analyze monthly and seasonal patterns.
        """
        month = current_time.month
        day = current_time.day
        month_info = self.monthly_patterns.get(month, self.monthly_patterns[6])
        
        # End of month effects
        days_in_month = calendar.monthrange(current_time.year, month)[1]
        is_month_end = day >= days_in_month - 2
        is_month_start = day <= 3
        
        # Quarter effects
        quarter = (month - 1) // 3 + 1
        is_quarter_end = month in [3, 6, 9, 12] and is_month_end
        
        # Seasonal bias
        if month in [1, 2, 3, 4, 10, 11]:
            seasonal_bias = 'bullish'
        elif month in [5, 6, 7, 8, 9]:
            seasonal_bias = 'cautious'
        else:
            seasonal_bias = 'year_end_positioning'
        
        return {
            'month': month,
            'month_name': month_info['name'],
            'pattern': seasonal_bias,
            'trend_strength': month_info['trend_strength'],
            'risk_on': month_info['risk_on'],
            'is_month_end': is_month_end,
            'is_month_start': is_month_start,
            'is_quarter_end': is_quarter_end,
            'quarter': quarter
        }
    
    def _analyze_hour_patterns(self, current_time: datetime) -> Dict[str, Any]:
        """
        Analyze hourly patterns and volatility expectations.
        """
        hour = current_time.hour
        minute = current_time.minute
        
        # Pre-news caution (10 mins before typical news times)
        is_pre_news = hour in self.high_impact_hours and minute >= 50
        is_news_hour = hour in self.high_impact_hours and minute < 30
        
        # Calculate expected volatility based on hour
        if 13 <= hour <= 16:  # EU/US overlap
            expected_volatility = 1.3
        elif 7 <= hour <= 9:  # EU open / Asia-EU overlap
            expected_volatility = 1.1
        elif 13 <= hour <= 14:  # US open
            expected_volatility = 1.2
        elif 0 <= hour <= 6:  # Asian session
            expected_volatility = 0.7
        elif 22 <= hour <= 23:  # Late NY
            expected_volatility = 0.6
        else:
            expected_volatility = 0.9
        
        # Trading quality score
        if 8 <= hour <= 16:
            trading_quality = 0.9
        elif 13 <= hour <= 20:
            trading_quality = 0.85
        else:
            trading_quality = 0.6
        
        return {
            'hour': hour,
            'is_pre_news': is_pre_news,
            'is_news_hour': is_news_hour,
            'expected_volatility': expected_volatility,
            'trading_quality': trading_quality,
            'high_impact_window': hour in self.high_impact_hours
        }
    
    def _check_rollover_risk(self, current_time: datetime) -> bool:
        """Check if we're in the rollover window."""
        current = current_time.time()
        return self._time_in_range(current, self.rollover_start, self.rollover_end)
    
    def _check_weekend_risk(self, current_time: datetime) -> bool:
        """Check if we're in weekend gap risk window."""
        is_friday = current_time.weekday() == 4
        current = current_time.time()
        return is_friday and current >= self.weekend_risk_start
    
    def _check_high_impact_window(self, current_time: datetime) -> bool:
        """Check if we're in a high-impact event window."""
        hour = current_time.hour
        minute = current_time.minute
        
        # 30 minutes before and after typical news releases
        if hour in self.high_impact_hours:
            if minute <= 30 or minute >= 50:
                return True
        
        # Check hour before high impact
        if (hour + 1) % 24 in self.high_impact_hours and minute >= 50:
            return True
        
        return False
    
    def _calculate_historical_pattern_score(
        self,
        current_time: datetime,
        market_data: Dict,
        features: Dict,
        instrument: str = ''
    ) -> float:
        """
        Calculate score based on historical performance at similar times.
        """
        close_prices = self._extract_prices(market_data, features, 'close', instrument)
        
        if len(close_prices) < 50:
            return 0.5
        
        dow = current_time.weekday()
        hour = current_time.hour
        
        # Look at returns at similar time windows historically
        # This is simplified - in production would use actual timestamps
        recent_returns = np.diff(np.log(close_prices[-20:]))
        
        # Calculate recent momentum as proxy for pattern continuation
        if len(recent_returns) > 0:
            avg_return = np.mean(recent_returns)
            positive_returns = np.sum(recent_returns > 0) / len(recent_returns)
        else:
            avg_return = 0
            positive_returns = 0.5
        
        # Combine into historical score
        historical_score = 0.5 + (positive_returns - 0.5) * 0.6
        
        return float(np.clip(historical_score, 0, 1))
    
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
            
            for tf in ['H1', 'H4', 'D1']:
                if tf in market_data and isinstance(market_data[tf], dict):
                    if price_type in market_data[tf]:
                        data = market_data[tf][price_type]
                        if isinstance(data, (list, np.ndarray)):
                            return np.array(data, dtype=float)
        
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
    
    def _calculate_composite_score(
        self,
        session_analysis: Dict,
        dow_analysis: Dict,
        monthly_analysis: Dict,
        hour_analysis: Dict,
        historical_score: float
    ) -> float:
        """
        Calculate composite seasonal/temporal score.
        
        Higher scores = more favorable trading conditions.
        """
        # Session quality weight
        session_score = session_analysis['session_quality']
        
        # Day of week score (based on trend continuation)
        dow_score = dow_analysis['trend_continuation']
        
        # Monthly trend strength
        monthly_score = monthly_analysis['trend_strength'] / 1.2  # Normalize to 0-1
        monthly_score = np.clip(monthly_score, 0, 1)
        
        # Hour quality
        hour_score = hour_analysis['trading_quality']
        
        # Weighted combination
        composite = (
            session_score * 0.25 +
            dow_score * 0.20 +
            monthly_score * 0.20 +
            hour_score * 0.15 +
            historical_score * 0.20
        )
        
        return float(np.clip(composite, 0, 1))
    
    def _select_seasonal_action(
        self,
        composite_score: float,
        session_analysis: Dict,
        dow_analysis: Dict,
        monthly_analysis: Dict,
        rollover_risk: bool,
        weekend_risk: bool,
        high_impact_window: bool,
        instrument: str = '',
        asset_class: str = 'forex'
    ) -> Tuple[str, float, str]:
        """
        Select trading action based on seasonal analysis.
        
        NEW: Now instrument-aware. Gold has different seasonal patterns.
        
        Returns: (action, confidence, thesis) - action is 'long', 'short', or 'flat'
        """
        inst_norm = normalize_instrument(instrument) if instrument else ''
        is_gold = inst_norm in ['XAUUSD', 'GOLD']
        
        # High-impact caution: stay flat (but Gold may get safe-haven bid)
        if high_impact_window:
            if is_gold:
                return (
                    "long",
                    0.5,
                    f"{inst_norm}: Safe haven bid during high-impact window"
                )
            return (
                "flat",  # Standard neutral action
                0.7,
                f"{inst_norm}: High-impact event window, recommending caution"
            )
        
        # Weekend risk: stay flat
        if weekend_risk:
            return (
                "flat",  # Standard neutral action
                0.75,
                f"{inst_norm}: Weekend gap risk - Friday late session"
            )
        
        # Rollover caution: stay flat
        if rollover_risk:
            return (
                "flat",  # Standard neutral action
                0.6,
                f"{inst_norm}: Rollover window - wider spreads"
            )
        
        # Poor session quality: stay flat (but Gold has 24h market)
        if session_analysis['session_quality'] < 0.3:
            # Gold trades better in Asian session than forex
            if is_gold and session_analysis['current_session'] == 'asian':
                pass  # Don't return flat for Gold in Asian session
            else:
                return (
                    "flat",  # Standard neutral action
                    0.55,
                    f"{inst_norm}: Low session quality ({session_analysis['current_session']})"
                )
        
        # ========== Asset-specific seasonal patterns ==========
        
        # Gold seasonal patterns (typically strong in Jan, Aug-Sep, year-end)
        if is_gold:
            month = monthly_analysis.get('month', 0)
            if month in [1, 8, 9, 12]:  # Strong gold months
                confidence = self.base_confidence + 0.2
                return (
                    "long",
                    np.clip(confidence, 0.5, 0.75),
                    f"{inst_norm}: Favorable gold seasonality ({monthly_analysis['month_name']})"
                )
            elif month in [3, 4, 5]:  # Weak gold months
                confidence = self.base_confidence
                return (
                    "short",
                    np.clip(confidence, 0.4, 0.65),
                    f"{inst_norm}: Weak gold seasonality ({monthly_analysis['month_name']})"
                )
        
        # Strong seasonal long bias (forex)
        if monthly_analysis['risk_on'] and composite_score > 0.6:
            confidence = self.base_confidence + (composite_score - 0.5) * 0.6
            return (
                "long",  # Standard bullish action
                np.clip(confidence, 0.5, 0.8),
                f"{inst_norm}: Favorable seasonal ({monthly_analysis['month_name']}, risk-on)"
            )
        
        # Cautious seasonal short bias (forex)
        if not monthly_analysis['risk_on'] and composite_score < 0.45:
            confidence = self.base_confidence + (0.5 - composite_score) * 0.6
            # Gold in risk-off months tends to be LONG (safe haven)
            if is_gold:
                return (
                    "long",
                    np.clip(confidence, 0.5, 0.75),
                    f"{inst_norm}: Safe haven in risk-off season"
                )
            return (
                "short",  # Standard bearish action
                np.clip(confidence, 0.5, 0.75),
                f"{inst_norm}: Unfavorable seasonal ({monthly_analysis['month_name']}, risk-off)"
            )
        
        # Optimal session - use dow bias for direction
        if session_analysis['current_session'] in ['overlap_eu_us', 'european']:
            if dow_analysis['trend_continuation'] > 0.55:
                dow_bias = dow_analysis.get('bias', 'neutral')
                if dow_bias == 'bullish':
                    return (
                        "long",
                        0.55,
                        f"{inst_norm}: Optimal session + bullish {dow_analysis['day_name']}"
                    )
                elif dow_bias == 'bearish':
                    if is_gold:
                        return (
                            "flat",
                            0.4,
                            f"{inst_norm}: Bearish DOW but gold - neutral"
                        )
                    return (
                        "short",
                        0.55,
                        f"{inst_norm}: Optimal session + bearish {dow_analysis['day_name']}"
                    )
        
        # Month-end/quarter-end effects - stay flat
        if monthly_analysis['is_quarter_end']:
            return (
                "flat",  # Standard neutral action
                0.4,
                f"{inst_norm}: Quarter-end rebalancing"
            )
        
        if monthly_analysis['is_month_end']:
            return (
                "flat",  # Standard neutral action
                0.35,
                f"{inst_norm}: Month-end positioning"
            )
        
        # Default: use composite score for direction
        if composite_score > 0.52:
            return (
                "long",
                0.45,
                f"{inst_norm}: Slight bullish seasonal (composite: {composite_score:.2f})"
            )
        elif composite_score < 0.48:
            if is_gold:
                return (
                    "flat",
                    0.35,
                    f"{inst_norm}: Slight bearish but gold - neutral"
                )
            return (
                "short",
                0.45,
                f"{inst_norm}: Slight bearish seasonal (composite: {composite_score:.2f})"
            )
        
        # True neutral
        return (
            "flat",  # Standard neutral action
            0.3,
            f"{inst_norm}: Neutral seasonal - session: {session_analysis['current_session']}"
        )
    
    def _neutral_output(self, reason: str) -> Dict[str, Any]:
        """Generate neutral output with explanation."""
        return {
            "SeasonalityRiskExpert_voting_proposal": "flat",
            "SeasonalityRiskExpert_confidence": 0.1,
            "seasonality_voting_proposal": "flat",   # Alias for contract compatibility
            "seasonality_confidence": 0.1,                        # Alias for contract compatibility
            "seasonal_voting_proposal": "flat",       # Additional alias
            "seasonal_confidence": 0.1,                           # Additional alias
            "seasonality_risk_analysis": {                        # Required by contract
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "composite_score": 0.5,
                "rollover_risk": False,
                "weekend_risk": False,
                "action": "flat",
                "confidence": 0.1
            },
            "seasonality_analysis": {                             # Alias
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "composite_score": 0.5
            },
            "seasonality_expert_analysis": {                      # Backward compat alias
                "session": "unknown",
                "dow_bias": "unknown",
                "monthly_pattern": "unknown",
                "composite_score": 0.5,
                "rollover_risk": False,
                "weekend_risk": False
            },
            "seasonality_expert_thesis": f"Seasonal flat: {reason}",  # Backward compat alias
            "seasonal_session": "unknown",
            "seasonal_dow_bias": "unknown",
            "seasonal_monthly_pattern": "unknown",
            "seasonal_composite_score": 0.5,
            "seasonal_rollover_risk": False,
            "seasonal_weekend_risk": False,
            "_thesis": f"Seasonal neutral: {reason}"
        }


