# ─────────────────────────────────────────────────────────────
# File: modules/market/components/time_risk.py
# Time-Aware Risk Scaling Component — Production-Ready Upgrade
# ─────────────────────────────────────────────────────────────

import datetime
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from modules.utils.session_utils import classify_session

from ..shared.base_component import BaseMarketComponent


class TimeRiskComponent(BaseMarketComponent):
    """
    Time-aware risk scaling based on trading sessions, realized volatility, and
    session performance. Production upgrades vs. baseline:
      • All time logic in UTC + explicit weekend and rollover handling
      • EWMA volatility w/ z-score adjustment and regime classification
      • Hysteresis on scaling factor (caps step-up/down to prevent whipsaws)
      • Liquidity bias (optional) from shared context (e.g., LiquidityHeatmap)
      • Hourly risk scoring memory + session performance feedback loop
      • High/Critical risk events with automatic de-risking
      • Bug fix: duplicate `volatility_adjustment` key (keeps float for back-compat,
        adds `volatility_adjustment_detail` for the structured breakdown)
      • Safer math, NaN/inf scrubbing, and defensive fallbacks
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        default_config = {
            # Session boundaries in UTC (hours)
            'asian_end': 8,
            'euro_end': 16,
            'us_end': 22,

            # Rollover (thin liquidity) window in UTC
            'rollover_start': 21,          # inclusive
            'rollover_end': 23,            # exclusive
            'rollover_multiplier': 0.7,

            # General scaling controls
            'decay_factor': 0.9,
            'base_factor': 1.0,
            'min_scaling': 0.10,
            'max_scaling': 5.00,

            # Hysteresis / anti-whipsaw
            'hysteresis_up': 0.25,         # max +25% change per call
            'hysteresis_down': 0.30,       # max -30% change per call

            # Volatility + memory
            'vol_window': 100,
            'ewma_alpha': 0.20,            # EWMA smoothing for realized vol
            'session_memory': 24,

            # Session multipliers
            'asian_multiplier': 1.2,
            'european_multiplier': 1.0,
            'us_multiplier': 1.1,
            'closed_multiplier': 0.5,

            # Liquidity bias from shared_context.liquidity.liquidity_bias (0.3..1.0)
            'use_liquidity_bias': True,
            'liquidity_bias_weight': 0.15,  # blended into scaling

            # Risk thresholds
            'risk_threshold_high': 0.80,
            'risk_threshold_critical': 0.95,

            # Instruments used to derive realized volatility if not provided
            'instruments': ('XAUUSD',),

            # Misc
            'timezone': 'UTC',              # informational; we use UTC uniformly
        }
        default_config.update(config or {})

        super().__init__(
            name="TimeRisk",
            config=default_config,
            **kwargs
        )

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------
    def initialize(self):
        # Volatility & risk profiles by hour
        self.vol_profile = np.ones(24, np.float32)
        self.risk_profile = np.ones(24, np.float32)

        # Session state
        self._current_session = "unknown"
        self._session_changes = 0
        self._last_session_change = None

        # Risk tracking
        self._volatility_history = deque(maxlen=int(self.config['vol_window']))
        self._volatility_ewma = None  # float
        self._factor_history = deque(maxlen=200)
        self._risk_events = deque(maxlen=200)
        self._session_transitions = deque(maxlen=200)

        # Current metrics
        self.current_scaling_factor = float(self.config['base_factor'])
        self.current_volatility = 0.01
        self.current_risk_level = 0.0
        self.session_performance_score = 0.5

        # Session performance tracking
        self._session_risk_multipliers = {
            'asian': float(self.config['asian_multiplier']),
            'european': float(self.config['european_multiplier']),
            'american': float(self.config['us_multiplier']),
            'rollover': float(self.config['closed_multiplier']) * float(self.config['rollover_multiplier']),
        }
        self._session_performance: Dict[str, Dict[str, Any]] = {}
        for session in ('asian', 'european', 'american', 'rollover'):
            self._session_performance[session] = {
                'count': 0,
                'total_factor': 0.0,
                'avg_volatility': 0.0,
                'risk_events': 0,
                'success_rate': 1.0,
                'last_update': datetime.datetime.utcnow()
            }

        # Advanced session analytics
        self._hourly_risk_scores = np.zeros(24, dtype=np.float64)

        self.trace("Time risk component initialized", level="DEBUG")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    async def analyze_impl(self, **inputs) -> Dict[str, Any]:
        self.trace("Starting time risk analysis", level="TRACE")

        try:
            time_data = await self._extract_time_data(inputs)
            if not time_data:
                self.trace("No time data available", level="WARNING")
                return self.get_fallback_result("No time data")

            result = await self._process_time_aware_scaling(time_data)

            self.trace(
                f"Time risk: session={result['current_session']}, "
                f"scaling={result['scaling_factor']:.3f}, risk={result['risk_level']:.3f}",
                level="DEBUG"
            )
            return result

        except Exception as e:
            self.trace(f"Time risk analysis error: {e}", level="ERROR")
            return self.get_fallback_result(str(e))

    # -------------------------------------------------------------------------
    # Extraction
    # -------------------------------------------------------------------------
    async def _extract_time_data(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        market_data = inputs.get('market_data', {}) or {}
        shared_context = inputs.get('shared_context', {}) or {}

        # FIX: Use data timestamp for training consistency, fallback to UTC for live
        # This ensures session classification is correct during backtesting/training
        timestamp = self._extract_data_timestamp(inputs)
        if timestamp is None:
            timestamp = pd.Timestamp.utcnow()
        hour = int(timestamp.hour)
        weekday = int(timestamp.weekday())
        weekend = (weekday == 5) or (weekday == 6)

        # Extract / compute volatility (prefer shared_context → market_data → last known → default)
        volatility = await self._extract_volatility_data(market_data, shared_context)

        # Current session using UTC hour + rollover + weekend
        session = classify_session(hour=hour, weekend=weekend)

        return {
            'timestamp': timestamp,
            'hour': hour,
            'weekday': weekday,
            'weekend': weekend,
            'volatility': float(volatility),
            'market_data': market_data,
            'shared_context': shared_context,
            'session': session,
            'source': 'extracted'
        }

    async def _extract_volatility_data(self, market_data: Dict[str, Any], shared_context: Dict[str, Any]) -> float:
        # 1) Shared context (e.g., liquidity component may publish realized vol)
        if isinstance(shared_context, dict):
            liq = shared_context.get('liquidity', {})
            # support either explicit current_volatility, or derive from spread/depth proxies
            vol_ctx = liq.get('current_volatility') if isinstance(liq, dict) else None
            if vol_ctx is not None and np.isfinite(vol_ctx):
                return float(vol_ctx)

        # 2) Market data → realized volatility from close series
        vols: List[float] = []
        for instrument in self.config['instruments']:
            inst = market_data.get(instrument)
            if isinstance(inst, dict) and 'close' in inst:
                closes = np.asarray(inst['close'], dtype=np.float64)
                if closes.size > 10:
                    rets = np.diff(closes) / np.where(closes[:-1] == 0, 1.0, closes[:-1])
                    if rets.size > 1:
                        vols.append(float(np.std(rets[-min(50, rets.size):])))

        if vols:
            return float(np.nan_to_num(np.mean(vols), nan=0.01))

        # 3) Last known or default
        if self.current_volatility and np.isfinite(self.current_volatility):
            return float(self.current_volatility)

        return 0.01

    def _extract_data_timestamp(self, inputs: Dict[str, Any]) -> Optional[pd.Timestamp]:
        """
        Extract timestamp from data for training consistency.
        During training/backtesting, we must use the data's timestamp, not live time.
        """
        try:
            # 1) Check for explicit data_timestamp passed by market_module
            data_ts = inputs.get('data_timestamp')
            if data_ts is not None:
                if isinstance(data_ts, pd.Timestamp):
                    return data_ts
                if isinstance(data_ts, datetime.datetime):
                    return pd.Timestamp(data_ts)
                if isinstance(data_ts, str):
                    return pd.Timestamp(data_ts)
                if isinstance(data_ts, (int, float)):
                    return pd.Timestamp(datetime.datetime.utcfromtimestamp(data_ts))

            # 2) Check market_data for timestamp
            market_data = inputs.get('market_data', {}) or {}
            ts = market_data.get('timestamp') or market_data.get('data_timestamp')
            if ts is not None:
                if isinstance(ts, pd.Timestamp):
                    return ts
                if isinstance(ts, datetime.datetime):
                    return pd.Timestamp(ts)
                if isinstance(ts, str):
                    return pd.Timestamp(ts)
                if isinstance(ts, (int, float)):
                    return pd.Timestamp(datetime.datetime.utcfromtimestamp(ts))

            # 3) Check timestamps list (use last one)
            timestamps = market_data.get('timestamps', [])
            if isinstance(timestamps, (list, np.ndarray)) and len(timestamps) > 0:
                last_ts = timestamps[-1]
                if isinstance(last_ts, pd.Timestamp):
                    return last_ts
                if isinstance(last_ts, datetime.datetime):
                    return pd.Timestamp(last_ts)
                if isinstance(last_ts, str):
                    return pd.Timestamp(last_ts)
                if isinstance(last_ts, np.datetime64):
                    return pd.Timestamp(last_ts)
                if isinstance(last_ts, (int, float)):
                    return pd.Timestamp(datetime.datetime.utcfromtimestamp(last_ts))

            # 4) Check shared_context for timestamp
            shared = inputs.get('shared_context', {}) or {}
            ctx_ts = shared.get('timestamp') or shared.get('data_timestamp')
            if ctx_ts is not None:
                if isinstance(ctx_ts, pd.Timestamp):
                    return ctx_ts
                if isinstance(ctx_ts, datetime.datetime):
                    return pd.Timestamp(ctx_ts)
                if isinstance(ctx_ts, str):
                    return pd.Timestamp(ctx_ts)
        except Exception:
            pass

        return None  # Caller should fallback to UTC now

    # -------------------------------------------------------------------------
    # Sessions
    # -------------------------------------------------------------------------
    def _get_session(self, hour: int, weekend: bool) -> str:
        """Backward-compat wrapper; use classify_session."""
        return classify_session(hour=hour, weekend=weekend)

    @staticmethod
    def _in_window(h: int, start: int, end: int) -> bool:
        return (start <= h < end) if start < end else (h >= start or h < end)

    # -------------------------------------------------------------------------
    # Core logic
    # -------------------------------------------------------------------------
    async def _process_time_aware_scaling(self, time_data: Dict[str, Any]) -> Dict[str, Any]:
        hour = int(time_data['hour'])
        session = str(time_data['session'])
        volatility_raw = float(time_data['volatility'])
        shared_context = time_data.get('shared_context', {}) or {}

        # Update realized vol EWMA (safer than raw std for sudden spikes)
        self._update_volatility_ewma(volatility_raw)
        vol_for_decision = float(self._volatility_ewma if self._volatility_ewma is not None else volatility_raw)

        # Update histories
        self.current_volatility = vol_for_decision
        self._volatility_history.append(vol_for_decision)

        # Handle session transitions
        if session != self._current_session:
            await self._handle_session_transition(self._current_session, session, hour)

        # Base factor with mild decay toward recent vol trend & hourly profile
        base_factor = self._calculate_base_scaling_factor(hour, session, vol_for_decision)

        # Volatility z-score adjustment
        vol_adjustment, vol_regime = self._calculate_volatility_adjustment(vol_for_decision)

        # Session multiplier
        session_multiplier = float(self._session_risk_multipliers.get(session, 1.0))

        # Optional liquidity bias (from LiquidityHeatmapComponent)
        liquidity_bias = 1.0
        if self.config.get('use_liquidity_bias', True):
            liq = shared_context.get('liquidity', {})
            lb = liq.get('liquidity_bias') if isinstance(liq, dict) else None
            if lb is not None and np.isfinite(lb):
                # Blend bias toward 1.0 to avoid overreaction
                w = float(self.config.get('liquidity_bias_weight', 0.15))
                liquidity_bias = 1.0 * (1.0 - w) + float(lb) * w

        # Proposed scaling
        proposed = base_factor * vol_adjustment * session_multiplier * liquidity_bias

        # Hysteresis: cap delta vs last factor
        scaled = self._apply_hysteresis(proposed)

        # Clamp to global bounds
        scaling_factor = float(np.clip(scaled, float(self.config['min_scaling']), float(self.config['max_scaling'])))

        # Compute risk level (0..1)
        risk_level, vol_percentile = self._calculate_current_risk_level(scaling_factor, vol_for_decision, session)

        # High/critical risk handling (auto de-risk)
        scaling_factor, event = self._apply_risk_brakes(scaling_factor, risk_level)
        if event is not None:
            self._risk_events.append(event)

        # Update profiles and memory
        self.vol_profile[hour] = float(vol_for_decision)
        self.risk_profile[hour] = float(scaling_factor)
        self._factor_history.append(float(scaling_factor))
        self._update_session_performance(session, scaling_factor, vol_for_decision)
        self._update_hourly_risk_score(hour, risk_level)

        # Trends & efficiency
        risk_trend = self._calculate_risk_trend()
        volatility_trend = self._calculate_volatility_trend()
        session_efficiency = self._calculate_session_efficiency(session)

        # Build output (backward compatible)
        # NOTE: keep 'volatility_adjustment' as float (as in your original dict override),
        # and add a detailed dict under 'volatility_adjustment_detail'.
        return {
            'risk_scaling_factor': float(scaling_factor),

            'session_risk': {
                'current_session': session,
                'risk_level': float(risk_level),
                'session_multiplier': float(session_multiplier),
                'hour': int(hour)
            },

            # Back-compat scalar:
            'volatility_adjustment': float(vol_adjustment),

            # New: structured breakdown
            'volatility_adjustment_detail': {
                'adjustment_factor': float(vol_adjustment),
                'current_volatility': float(vol_for_decision),
                'volatility_percentile': float(vol_percentile),
                'volatility_regime': str(vol_regime),
                'volatility_trend': str(volatility_trend)
            },

            'scaling_factor': float(scaling_factor),  # same value, different key (kept)
            'risk_level': float(risk_level),
            'current_session': session,
            'hour': int(hour),
            'volatility': float(vol_for_decision),
            'session_multiplier': float(session_multiplier),
            'liquidity_bias_used': float(liquidity_bias),

            'risk_trend': str(risk_trend),
            'volatility_regime': str(vol_regime),
            'volatility_trend': str(volatility_trend),
            'session_efficiency': float(session_efficiency),
            'hourly_risk_score': float(self._hourly_risk_scores[hour]),
            'session_transitions': int(self._session_changes),

            'time_risk_analysis': {
                'risk_level': float(risk_level),
                'current_session': session,
                'hour': int(hour),
                'risk_trend': str(risk_trend),
                'volatility_trend': str(volatility_trend),
                'session_efficiency': float(session_efficiency),
                'hourly_risk_score': float(self._hourly_risk_scores[hour]),
                'session_transitions': int(self._session_changes),
                'processing_success': True
            },

            # Lightweight diagnostics
            'diagnostics': {
                'last_session_change': self._last_session_change.isoformat() if self._last_session_change else None,
                'recent_risk_events': list(self._risk_events)[-5:],
                'ewma_volatility': float(self._volatility_ewma) if self._volatility_ewma is not None else None
            },

            'processing_success': True
        }

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------
    def _update_volatility_ewma(self, vol: float):
        if vol is None or not np.isfinite(vol):
            return
        a = float(self.config['ewma_alpha'])
        if self._volatility_ewma is None:
            self._volatility_ewma = float(vol)
        else:
            self._volatility_ewma = float(a * vol + (1.0 - a) * self._volatility_ewma)

    async def _handle_session_transition(self, old_session: str, new_session: str, hour: int):
        self._session_changes += 1
        self._last_session_change = datetime.datetime.utcnow()
        self._current_session = new_session

        transition_data = {
            'from': old_session,
            'to': new_session,
            'hour': int(hour),
            'timestamp': self._last_session_change,
            'volatility': float(self.current_volatility)
        }
        self._session_transitions.append(transition_data)
        self.trace(f"Session transition: {old_session} -> {new_session}", level="INFO")

        # Optional: nudge multipliers based on performance
        self._adjust_session_multipliers(new_session)

    def _calculate_base_scaling_factor(self, hour: int, session: str, volatility: float) -> float:
        base = float(self.config['base_factor'])

        # Decay toward recent mean vol level (gentle)
        if len(self._volatility_history) >= 5:
            recent = np.asarray(list(self._volatility_history)[-10:], dtype=np.float64)
            mean_recent = float(np.mean(recent))
            mean_prev = float(np.mean(recent[:-1])) if recent.size > 1 else mean_recent
            ratio = mean_recent / (mean_prev + 1e-8)
            base *= float(self.config['decay_factor']) * float(np.clip(ratio, 0.8, 1.2))

        # Hourly pattern adjustment (normalize by mean vol_profile)
        if float(np.sum(self.vol_profile)) > 0.0:
            hourly_factor = float(self.vol_profile[hour]) / (float(np.mean(self.vol_profile)) + 1e-8)
            base *= float(np.clip(1.0 + 0.1 * (hourly_factor - 1.0), 0.8, 1.2))

        return float(base)

    def _calculate_volatility_adjustment(self, volatility: float) -> Tuple[float, str]:
        """Return (adjustment, regime_label)."""
        if len(self._volatility_history) < 10:
            return 1.0, "normal"

        hist = np.asarray(self._volatility_history, dtype=np.float64)
        mean_vol = float(np.mean(hist))
        std_vol = float(np.std(hist))

        if std_vol <= 1e-12:
            return 1.0, "normal"

        z = (volatility - mean_vol) / std_vol
        # Map z to discrete regimes & adjustment
        if z > 2.0:
            return 1.5, "high"
        elif z > 1.0:
            return 1.2, "elevated"
        elif z < -2.0:
            return 0.7, "low"
        elif z < -1.0:
            return 0.85, "subdued"
        else:
            return 1.0, "normal"

    def _apply_hysteresis(self, proposed: float) -> float:
        """Limit per-step change to avoid whipsaws."""
        last = float(self._factor_history[-1]) if self._factor_history else float(self.config['base_factor'])
        up_cap = 1.0 + float(self.config['hysteresis_up'])
        dn_cap = 1.0 - float(self.config['hysteresis_down'])
        # Bound relative to last
        upper = last * up_cap
        lower = last * dn_cap
        return float(np.clip(proposed, lower, upper))

    def _calculate_current_risk_level(self, scaling_factor: float, volatility: float, session: str):
        # Factor risk (normalized 0..1 by 2x)
        factor_risk = float(np.clip(scaling_factor / 2.0, 0.0, 1.0))

        # Volatility percentile → 0..1
        vol_percentile = self._get_volatility_percentile(volatility)
        vol_risk = float(vol_percentile) / 100.0

        # Session baseline risk
        session_risk_map = {'asian': 0.30, 'european': 0.20, 'american': 0.25, 'closed': 0.35}
        session_risk = float(session_risk_map.get(session, 0.20))

        combined = 0.40 * factor_risk + 0.40 * vol_risk + 0.20 * session_risk
        return float(np.clip(combined, 0.0, 1.0)), vol_percentile

    def _apply_risk_brakes(self, scaling_factor: float, risk_level: float):
        """If risk crosses thresholds, emit event and cap scaling."""
        high = float(self.config['risk_threshold_high'])
        critical = float(self.config['risk_threshold_critical'])

        event = None
        capped = scaling_factor

        if risk_level >= critical:
            capped = min(capped, max(float(self.config['min_scaling']), 0.5))  # hard de-risk
            event = {
                'severity': 'CRITICAL',
                'risk_level': float(risk_level),
                'scaling_capped_to': float(capped),
                'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
            }
        elif risk_level >= high:
            capped = min(capped, 1.0)  # soft cap
            event = {
                'severity': 'HIGH',
                'risk_level': float(risk_level),
                'scaling_capped_to': float(capped),
                'timestamp': datetime.datetime.utcnow().isoformat() + 'Z'
            }

        return float(capped), event

    def _get_volatility_percentile(self, volatility: float) -> float:
        if len(self._volatility_history) < 10:
            return 50.0
        hist = np.asarray(self._volatility_history, dtype=np.float64)
        # percentile of 'volatility' within hist
        percentile = float(np.sum(hist <= volatility)) / float(len(hist)) * 100.0
        return float(np.clip(percentile, 0.0, 100.0))

    def _calculate_risk_trend(self) -> str:
        if len(self._factor_history) < 5:
            return "stable"
        y = np.asarray(list(self._factor_history)[-8:], dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        try:
            if float(np.std(y)) == 0.0:
                return "stable"
            slope = float(np.polyfit(x, y, 1)[0])
            if slope > 0.02:
                return "increasing"
            elif slope < -0.02:
                return "decreasing"
            else:
                return "stable"
        except Exception:
            return "stable"

    def _calculate_volatility_trend(self) -> str:
        if len(self._volatility_history) < 10:
            return "stable"
        y = np.asarray(list(self._volatility_history)[-10:], dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        try:
            if float(np.std(y)) == 0.0:
                return "stable"
            slope = float(np.polyfit(x, y, 1)[0])
            if slope > 1e-3:
                return "increasing"
            elif slope < -1e-3:
                return "decreasing"
            else:
                return "stable"
        except Exception:
            return "stable"

    def _calculate_session_efficiency(self, session: str) -> float:
        perf = self._session_performance.get(session)
        if not perf:
            return 0.5
        cnt = int(perf.get('count', 0))
        if cnt == 0:
            return 0.5
        success_rate = float(perf.get('success_rate', 0.5))
        risk_events = int(perf.get('risk_events', 0))
        efficiency = success_rate * (1.0 - min(risk_events / max(cnt, 1), 0.5))
        return float(np.clip(efficiency, 0.0, 1.0))

    def _update_hourly_risk_score(self, hour: int, risk_level: float):
        # Simple exponential smoothing per hour
        prev = float(self._hourly_risk_scores[hour])
        self._hourly_risk_scores[hour] = 0.9 * prev + 0.1 * float(risk_level)

    def _update_session_performance(self, session: str, scaling_factor: float, volatility: float):
        perf = self._session_performance.get(session)
        if not perf:
            return

        perf['count'] = int(perf['count']) + 1
        perf['total_factor'] = float(perf['total_factor']) + float(scaling_factor)
        # Running average volatility
        cnt = perf['count']
        perf['avg_volatility'] = float((perf['avg_volatility'] * (cnt - 1) + volatility) / cnt)

        # Risk event heuristic
        if float(scaling_factor) > 2.0 or float(volatility) > 0.05:
            perf['risk_events'] = int(perf['risk_events']) + 1

        perf['success_rate'] = float(1.0 - (perf['risk_events'] / max(perf['count'], 1)))
        perf['last_update'] = datetime.datetime.utcnow()

    def _adjust_session_multipliers(self, session: str):
        """Light adaptive tuning of session multipliers based on success rate."""
        perf = self._session_performance.get(session)
        if not perf or int(perf['count']) < 10:
            return
        sr = float(perf['success_rate'])
        mult = float(self._session_risk_multipliers.get(session, 1.0))
        if sr > 0.80:
            mult *= 0.95
        elif sr < 0.60:
            mult *= 1.05

        # ═══════════════════════════════════════════════════════════════════
        # TRADING MODE MANAGER INTEGRATION
        # Blend mode's session_score into session multipliers
        # ═══════════════════════════════════════════════════════════════════
        try:
            from modules.utils.info_bus import InfoBusManager
            bus = InfoBusManager.get_instance()
            decision_factors = bus.get('decision_factors', 'TimeRiskComponent') or {}
            trading_mode = bus.get('trading_mode', 'TimeRiskComponent') or 'normal'
            session_score = float(decision_factors.get('session_score', 0.5))

            # If TradingModeManager gives low session score, reduce our multiplier
            if session_score < 0.5:
                mode_penalty = 0.7 + (session_score * 0.6)  # 0.7 to 1.0
                mult *= mode_penalty
        except Exception:
            pass  # Graceful fallback
        # ═══════════════════════════════════════════════════════════════════

        self._session_risk_multipliers[session] = float(np.clip(mult, 0.3, 2.0))

    # -------------------------------------------------------------------------
    # Fallback
    # -------------------------------------------------------------------------
    def get_fallback_result(self, error: str) -> Dict[str, Any]:
        now = pd.Timestamp.utcnow()
        fallback_session = self._get_session(int(now.hour), weekend=(now.weekday() in (5, 6)))

        return {
            'risk_scaling_factor': float(self.config['base_factor']),
            'session_risk': {
                'current_session': fallback_session,
                'risk_level': 0.5,
                'session_multiplier': float(self._session_risk_multipliers.get(fallback_session, 1.0)),
                'hour': int(now.hour)
            },
            'volatility_adjustment': 1.0,
            'volatility_adjustment_detail': {
                'adjustment_factor': 1.0,
                'current_volatility': float(self.current_volatility) if np.isfinite(self.current_volatility) else 0.01,
                'volatility_percentile': 50.0,
                'volatility_regime': 'unknown',
                'volatility_trend': 'unknown'
            },
            'scaling_factor': float(self.config['base_factor']),
            'risk_level': 0.5,
            'current_session': fallback_session,
            'hour': int(now.hour),
            'volatility': float(self.current_volatility) if np.isfinite(self.current_volatility) else 0.01,
            'session_multiplier': float(self._session_risk_multipliers.get(fallback_session, 1.0)),
            'liquidity_bias_used': 1.0,
            'risk_trend': 'unknown',
            'volatility_regime': 'unknown',
            'volatility_trend': 'unknown',
            'session_efficiency': 0.5,
            'hourly_risk_score': 0.5,
            'session_transitions': int(self._session_changes),
            'time_risk_analysis': {
                'risk_level': 0.5,
                'current_session': fallback_session,
                'hour': int(now.hour),
                'risk_trend': 'unknown',
                'volatility_trend': 'unknown',
                'session_efficiency': 0.5,
                'hourly_risk_score': 0.5,
                'session_transitions': int(self._session_changes),
                'processing_success': False
            },
            'diagnostics': {
                'last_session_change': self._last_session_change.isoformat() if self._last_session_change else None,
                'recent_risk_events': list(self._risk_events)[-5:],
                'ewma_volatility': float(self._volatility_ewma) if self._volatility_ewma is not None else None
            },
            'processing_success': False,
            'error': error
        }
