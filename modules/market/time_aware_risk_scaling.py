# ─────────────────────────────────────────────────────────────
# File: modules/market/time_aware_risk_scaling.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Union, Tuple
from collections import deque
import datetime

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import RiskMixin, AnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class TimeAwareRiskScaling(Module, RiskMixin, AnalysisMixin):
    def __init__(self, debug: bool = True, genome: Optional[Dict[str, Any]] = None, **kwargs):
        # 1) establish genome‐based attributes (including self.vol_window)
        self._initialize_genome_parameters(genome)

        # 2) now call base ctor, which will run _initialize_module_state()
        config = ModuleConfig(
            debug=debug,
            max_history=500,
            **kwargs
        )
        super().__init__(config)

        # 3) any further setup/logging
        self.log_operator_info(
            "Time-aware risk scaling initialized",
            asian_end=f"{self.asian_end}:00",
            euro_end=f"{self.euro_end}:00",
            decay_factor=f"{self.decay:.3f}",
            base_factor=f"{self.base_factor:.3f}"
        )

    def _initialize_genome_parameters(self, genome: Optional[Dict]):
        """Initialize genome-based parameters"""
        if genome:
            self.asian_end = int(genome.get("asian_end", 8))
            self.euro_end = int(genome.get("euro_end", 16))
            self.decay = float(genome.get("decay", 0.9))
            self.base_factor = float(genome.get("base_factor", 1.0))
            self.vol_window = int(genome.get("vol_window", 100))
            self.session_memory = int(genome.get("session_memory", 24))
        else:
            self.asian_end = 8
            self.euro_end = 16
            self.decay = 0.9
            self.base_factor = 1.0
            self.vol_window = 100
            self.session_memory = 24

        # Store genome for evolution
        self.genome = {
            "asian_end": self.asian_end,
            "euro_end": self.euro_end,
            "decay": self.decay,
            "base_factor": self.base_factor,
            "vol_window": self.vol_window,
            "session_memory": self.session_memory
        }
        
        # ✅ THESE ARE THE ONLY ADDITIONS NEEDED:
        self._risk_alerts = deque(maxlen=100)
        self._risk_score_history = deque(maxlen=100)

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_risk_state()
        self._initialize_analysis_state()
        
        # Time-aware specific state
        self.vol_profile = np.zeros(24, np.float32)
        self.seasonality_factor = 1.0
        
        # Enhanced tracking
        self._session_performance = {}  # Track performance by session
        self._current_session = "unknown"
        self._session_changes = 0
        self._volatility_history = deque(maxlen=self.vol_window)
        self._factor_history = deque(maxlen=100)
        self._session_transitions = deque(maxlen=50)
        
        # Risk assessment state
        self._risk_profile_by_hour = np.ones(24, np.float32)
        self._session_risk_multipliers = {
            "asian": 1.0,
            "european": 1.0, 
            "us": 1.0,
            "closed": 0.5
        }
        
        # Performance tracking by session
        for session in ["asian", "european", "us", "closed"]:
            self._session_performance[session] = {
                'count': 0,
                'total_factor': 0.0,
                'avg_volatility': 0.0,
                'risk_events': 0
            }

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_analysis_state()
        
        # Module-specific reset
        self.vol_profile.fill(0.0)
        self.seasonality_factor = 1.0
        self._current_session = "unknown"
        self._session_changes = 0
        self._volatility_history.clear()
        self._factor_history.clear()
        self._session_transitions.clear()
        self._risk_profile_by_hour.fill(1.0)
        
        # Reset session performance
        for session in self._session_performance:
            self._session_performance[session] = {
                'count': 0,
                'total_factor': 0.0,
                'avg_volatility': 0.0,
                'risk_events': 0
            }

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract time and market data
        time_data = self._extract_time_data(info_bus, kwargs)
        
        # Process risk scaling with enhanced analytics
        self._process_risk_scaling(time_data)
        
        # Update risk assessments
        self._update_risk_assessments(time_data)

    def _extract_time_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract time and market data with enhanced type safety"""
        
        # Try InfoBus first
        if info_bus:
            # Safe timestamp extraction
            timestamp = info_bus.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        ts = pd.Timestamp(timestamp)
                    elif hasattr(timestamp, 'hour'):
                        ts = pd.Timestamp(timestamp)
                    else:
                        ts = pd.Timestamp.now()
                except Exception:
                    ts = pd.Timestamp.now()
            else:
                ts = pd.Timestamp.now()
            
            # Safe hour extraction with validation
            try:
                hour = int(ts.hour) % 24
            except (ValueError, AttributeError):
                hour = 12  # Default to noon
            
            # Safe volatility extraction
            volatility = self._extract_volatility_safe(info_bus)
            
            return {
                'timestamp': ts,
                'hour': hour,
                'volatility': volatility,
                'market_context': info_bus.get('market_context', {}),
                'risk_data': info_bus.get('risk', {}),
                'session': InfoBusExtractor.get_session(info_bus),
                'volatility_level': InfoBusExtractor.get_volatility_level(info_bus),
                'source': 'info_bus'
            }
        
        # Kwargs fallback with safety
        if "data_dict" in kwargs:
            data_dict = kwargs["data_dict"]
            try:
                ts_raw = data_dict.get("timestamp", pd.Timestamp.now())
                ts = pd.Timestamp(ts_raw) if not isinstance(ts_raw, pd.Timestamp) else ts_raw
                hour = int(ts.hour) % 24
            except Exception:
                ts = pd.Timestamp.now()
                hour = 12
            
            # Safe returns extraction
            returns = np.asarray(data_dict.get("returns", []), dtype=np.float32)
            if len(returns) == 0 or not np.all(np.isfinite(returns)):
                returns = np.random.normal(0, 0.01, 100).astype(np.float32)
            
            volatility = float(np.nanstd(returns[-self.vol_window:]))
            if not np.isfinite(volatility) or volatility <= 0:
                volatility = 0.01
            
            return {
                'timestamp': ts,
                'hour': hour,
                'volatility': volatility,
                'returns': returns,
                'source': 'kwargs'
            }
        
        # Safe fallback
        ts = pd.Timestamp.now()
        return {
            'timestamp': ts,
            'hour': int(ts.hour) % 24,
            'volatility': 0.01,
            'source': 'simulation'
        }

    def _extract_volatility_safe(self, info_bus: InfoBus) -> float:
        """Safely extract volatility with multiple fallbacks"""
        try:
            # Try market context first
            market_context = info_bus.get('market_context', {})
            if 'volatility' in market_context:
                vol_data = market_context['volatility']
                
                if isinstance(vol_data, dict) and vol_data:
                    vol_values = []
                    for v in vol_data.values():
                        try:
                            val = float(v)
                            if np.isfinite(val) and val > 0:
                                vol_values.append(val)
                        except (ValueError, TypeError):
                            continue
                    
                    if vol_values:
                        return float(np.mean(vol_values))
                
                elif isinstance(vol_data, (int, float)):
                    val = float(vol_data)
                    if np.isfinite(val) and val > 0:
                        return val
            
            # Try volatility level mapping
            vol_level = InfoBusExtractor.get_volatility_level(info_bus)
            vol_mapping = {'low': 0.005, 'medium': 0.015, 'high': 0.03, 'extreme': 0.06}
            if vol_level in vol_mapping:
                return vol_mapping[vol_level]
            
            # Final fallback
            return 0.015
            
        except Exception:
            return 0.015

    def _process_risk_scaling(self, time_data: Dict[str, Any]):
        """Process risk scaling with enhanced session analytics"""
        
        try:
            # 🔧 FIX: Extract and validate data types
            hour = int(time_data['hour'])
            volatility = float(time_data['volatility'])
            
            # 🔧 FIX: Ensure volatility is finite and positive
            if not np.isfinite(volatility) or volatility < 0:
                volatility = 0.01
            
            # Update volatility profile with decay
            self.vol_profile = self.vol_profile * float(self.decay)
            self.vol_profile[hour] = float(volatility)
            
            # Store volatility history
            self._volatility_history.append(float(volatility))
            
            # Calculate dynamic base factor - 🔧 FIX: Safe division
            max_vol = float(self.vol_profile.max())
            if max_vol <= 0:
                max_vol = 0.01  # Avoid division by zero
                
            dynamic_factor = float(self.base_factor) - (float(volatility) / max_vol)
            dynamic_factor = float(np.clip(dynamic_factor, 0.0, 2.0))
            
            # Determine current session
            new_session = self._get_session(hour)
            
            # Check for session transition
            if new_session != self._current_session:
                self._handle_session_transition(self._current_session, new_session, hour)
                self._current_session = new_session
            
            # Calculate session-specific factor
            session_factor = self._calculate_session_factor(new_session, dynamic_factor, volatility)
            
            # Apply risk adjustments
            risk_adjusted_factor = self._apply_risk_adjustments(session_factor, time_data)
            
            # Update state
            self.seasonality_factor = float(risk_adjusted_factor)
            self._factor_history.append(self.seasonality_factor)
            
            # Update session performance tracking
            self._update_session_performance(new_session, self.seasonality_factor, volatility)
            
            # Update risk profile for this hour
            self._risk_profile_by_hour[hour] = self.seasonality_factor
            
            # Log significant changes
            if len(self._factor_history) > 1:
                factor_change = self.seasonality_factor - self._factor_history[-2]
                if abs(factor_change) > 0.2:
                    self.log_operator_info(
                        f"Significant risk factor change",
                        hour=f"{hour:02d}:00",
                        session=new_session,
                        old_factor=f"{self._factor_history[-2]:.3f}",
                        new_factor=f"{self.seasonality_factor:.3f}",
                        change=f"{factor_change:+.3f}",
                        volatility=f"{volatility:.5f}"
                    )
            
            # Update performance metrics
            self._update_performance_metric('seasonality_factor', self.seasonality_factor)
            self._update_performance_metric('current_volatility', volatility)
            self._update_performance_metric('session_changes', self._session_changes)
            
        except Exception as e:
            self.log_operator_error(f"Risk scaling processing failed: {e}")
            self._update_health_status("DEGRADED", f"Processing failed: {e}")

    def _get_session(self, hour: int) -> str:
        """Enhanced session determination with validation"""
        hour = int(hour % 24)  # Ensure valid hour
        
        if 0 <= hour < self.asian_end:
            return "asian"
        elif self.asian_end <= hour < self.euro_end:
            return "european"
        elif self.euro_end <= hour < 24:
            return "us"
        else:
            return "closed"

    def _handle_session_transition(self, old_session: str, new_session: str, hour: int):
        """Handle session transitions with logging and analytics"""
        
        self._session_changes += 1
        
        # Record transition
        transition = {
            'from': old_session,
            'to': new_session,
            'hour': int(hour),
            'timestamp': datetime.datetime.now().isoformat()
        }
        self._session_transitions.append(transition)
        
        # Log transition
        if old_session != "unknown":
            self.log_operator_info(
                f"Market session transition",
                from_session=old_session,
                to_session=new_session,
                hour=f"{hour:02d}:00",
                transition_count=self._session_changes
            )
        
        # Update risk multipliers based on session performance
        self._update_session_risk_multipliers()

    def _calculate_session_factor(self, session: str, dynamic_factor: float, volatility: float) -> float:
        """Calculate session-specific scaling factor"""
        
        # 🔧 FIX: Ensure all inputs are floats
        dynamic_factor = float(dynamic_factor)
        volatility = float(volatility)
        
        # Base session mapping with historical adjustments
        base_multipliers = {
            "asian": 1.0 + 0.3 * dynamic_factor,
            "european": dynamic_factor,
            "us": 1.0 - 0.4 * (1.0 - dynamic_factor),
            "closed": 0.5 * dynamic_factor
        }
        
        base_factor = float(base_multipliers.get(session, dynamic_factor))
        
        # Apply learned session risk multiplier
        session_multiplier = float(self._session_risk_multipliers.get(session, 1.0))
        
        # Volatility adjustment - 🔧 FIX: Safe volatility calculation
        vol_adjustment = 1.0 - min(0.3, float(volatility) * 20)  # Reduce factor for high volatility
        
        final_factor = base_factor * session_multiplier * vol_adjustment
        
        return float(np.clip(final_factor, 0.1, 2.0))

    def _apply_risk_adjustments(self, session_factor: float, time_data: Dict[str, Any]) -> float:
        """Apply additional risk adjustments based on market conditions"""
        
        # 🔧 FIX: Ensure session_factor is float
        adjusted_factor = float(session_factor)
        
        # Apply risk context from InfoBus
        if 'risk_data' in time_data:
            risk_data = time_data['risk_data']
            
            # Drawdown adjustment - 🔧 FIX: Safe extraction and conversion
            drawdown = float(risk_data.get('current_drawdown', 0.0))
            if drawdown > 0.1:  # 10% drawdown
                drawdown_penalty = 1.0 - min(0.5, drawdown * 2)
                adjusted_factor *= float(drawdown_penalty)
                
                if drawdown > 0.15:  # Significant drawdown
                    self._risk_alerts.append({
                        'type': 'high_drawdown',
                        'value': drawdown,
                        'adjustment': drawdown_penalty,
                        'timestamp': time_data.get('timestamp', pd.Timestamp.now()).isoformat()
                    })
            
            # Exposure adjustment - 🔧 FIX: Safe calculation
            margin_used = float(risk_data.get('margin_used', 0.0))
            equity = float(risk_data.get('equity', 1.0))
            if equity <= 0:
                equity = 1.0  # Avoid division by zero
                
            exposure_pct = (margin_used / equity) * 100
            
            if exposure_pct > 70:  # High exposure
                exposure_penalty = 1.0 - min(0.3, (exposure_pct - 70) / 100)
                adjusted_factor *= float(exposure_penalty)
        
        # Volatility level adjustment
        vol_level = time_data.get('volatility_level', 'medium')
        vol_adjustments = {
            'low': 1.1,      # Increase factor for low vol
            'medium': 1.0,   # No adjustment
            'high': 0.8,     # Reduce factor for high vol
            'extreme': 0.5   # Significantly reduce for extreme vol
        }
        vol_mult = float(vol_adjustments.get(vol_level, 1.0))
        adjusted_factor *= vol_mult
        
        return float(np.clip(adjusted_factor, 0.05, 2.5))

    def _update_session_performance(self, session: str, factor: float, volatility: float):
        """Update session performance tracking"""
        
        # 🔧 FIX: Ensure numeric types
        factor = float(factor)
        volatility = float(volatility)
        
        perf = self._session_performance[session]
        perf['count'] += 1
        perf['total_factor'] += factor
        perf['avg_volatility'] = ((perf['avg_volatility'] * (perf['count'] - 1)) + volatility) / perf['count']
        
        # Check for risk events
        if factor < 0.3 or volatility > 0.05:
            perf['risk_events'] += 1

    def _update_session_risk_multipliers(self):
        """Update session risk multipliers based on performance"""
        
        for session, perf in self._session_performance.items():
            if perf['count'] > 10:  # Sufficient data
                avg_factor = perf['total_factor'] / perf['count']
                risk_event_ratio = perf['risk_events'] / perf['count']
                
                # Adjust multiplier based on performance
                if avg_factor > 1.2 and risk_event_ratio < 0.1:
                    # Good performance, increase multiplier
                    self._session_risk_multipliers[session] = float(min(1.3, 
                        self._session_risk_multipliers[session] * 1.05))
                elif avg_factor < 0.5 or risk_event_ratio > 0.3:
                    # Poor performance, decrease multiplier
                    self._session_risk_multipliers[session] = float(max(0.5,
                        self._session_risk_multipliers[session] * 0.95))

    def _update_risk_assessments(self, time_data: Dict[str, Any]):
        """Update comprehensive risk assessments"""
        
        # Extract risk context
        risk_context = self._extract_risk_context_from_time_data(time_data)
        
        # Assess current risk level
        risk_level = self._assess_risk_level(risk_context)
        
        # Update risk history
        if len(self._risk_score_history) == 0 or abs(risk_level - self._risk_score_history[-1]) > 0.1:
            self._risk_score_history.append(float(risk_level))
            
            # Log significant risk changes
            if risk_level > 0.7:
                self.log_operator_warning(
                    f"High risk level detected",
                    risk_score=f"{risk_level:.3f}",
                    session=self._current_session,
                    factor=f"{self.seasonality_factor:.3f}"
                )

    def _extract_risk_context_from_time_data(self, time_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract risk context from time data"""
        
        risk_context = {
            'volatility': float(time_data.get('volatility', 0.01)),
            'session': self._current_session,
            'hour': int(time_data.get('hour', 12)),
            'factor': float(self.seasonality_factor)
        }
        
        # Add risk data if available
        if 'risk_data' in time_data:
            risk_data = time_data['risk_data']
            # 🔧 FIX: Safely convert risk data values
            for key, value in risk_data.items():
                try:
                    if isinstance(value, (int, float)):
                        risk_context[key] = float(value)
                    else:
                        risk_context[key] = value
                except (ValueError, TypeError):
                    risk_context[key] = value
        
        return risk_context

    def _assess_risk_level(self, risk_context: Dict[str, Any]) -> float:
        """Assess overall risk level based on context"""
        
        risk_score = 0.0
        
        # Volatility risk
        volatility = risk_context.get('volatility', 0.01)
        if volatility > 0.05:
            risk_score += 0.3
        elif volatility > 0.03:
            risk_score += 0.2
        elif volatility > 0.02:
            risk_score += 0.1
            
        # Factor risk
        factor = risk_context.get('factor', 1.0)
        if factor < 0.3:
            risk_score += 0.4
        elif factor < 0.5:
            risk_score += 0.2
            
        # Session risk
        session = risk_context.get('session', 'unknown')
        if session in ['closed', 'unknown']:
            risk_score += 0.1
            
        # Additional risk data
        if 'current_drawdown' in risk_context:
            drawdown = risk_context['current_drawdown']
            if drawdown > 0.15:
                risk_score += 0.3
            elif drawdown > 0.1:
                risk_score += 0.2
                
        return float(np.clip(risk_score, 0.0, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components"""
        
        # Base observation
        base_obs = np.array([self.seasonality_factor], np.float32)
        
        # Additional components
        current_hour = datetime.datetime.now().hour % 24
        session_encoding = self._encode_session(self._current_session)
        
        # Risk indicators
        avg_volatility = float(np.mean(list(self._volatility_history))) if self._volatility_history else 0.01
        factor_trend = self._calculate_factor_trend()
        risk_score = float(self._risk_score_history[-1]) if self._risk_score_history else 0.5
        
        # Session performance indicator
        session_performance = self._get_session_performance_score()
        
        enhanced_obs = np.concatenate([
            base_obs,
            [float(current_hour) / 24.0],  # Normalized hour
            session_encoding,
            [avg_volatility, factor_trend, risk_score, session_performance]
        ])
        
        return enhanced_obs.astype(np.float32)

    def _encode_session(self, session: str) -> np.ndarray:
        """Encode session as one-hot vector"""
        sessions = ["asian", "european", "us", "closed"]
        encoding = np.zeros(len(sessions), np.float32)
        if session in sessions:
            encoding[sessions.index(session)] = 1.0
        return encoding

    def _calculate_factor_trend(self) -> float:
        """Calculate recent factor trend"""
        if len(self._factor_history) < 5:
            return 0.0
            
        recent = list(self._factor_history)[-5:]
        if abs(recent[0]) < 1e-10:  # Avoid division by very small numbers
            return 0.0
            
        trend = (recent[-1] - recent[0]) / max(abs(recent[0]), 0.1)
        return float(np.clip(trend, -1.0, 1.0))

    def _get_session_performance_score(self) -> float:
        """Get current session performance score"""
        if self._current_session not in self._session_performance:
            return 0.5
            
        perf = self._session_performance[self._current_session]
        if perf['count'] == 0:
            return 0.5
            
        # Simple performance score based on average factor and risk events
        avg_factor = perf['total_factor'] / perf['count']
        risk_ratio = perf['risk_events'] / perf['count']
        
        # Normalize to 0-1 scale
        performance = (avg_factor - 0.5) * 0.5 + (1.0 - risk_ratio) * 0.5
        return float(np.clip(performance, 0.0, 1.0))

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose risk-adjusted action scaling"""
        
        # This module provides scaling factors, not direct actions
        # Return scaling recommendations
        if hasattr(obs, 'shape') and len(obs.shape) > 0:
            action_dim = obs.shape[0] if obs.shape[0] > 0 else 2
        else:
            action_dim = 2
            
        # Apply seasonality factor as action scaling
        scaling_factor = float(self.seasonality_factor)
        
        # Risk-based position size recommendations
        risk_level = float(self._risk_score_history[-1]) if self._risk_score_history else 0.5
        risk_adjustment = 1.0 - risk_level * 0.5  # Reduce size for high risk
        
        final_scaling = scaling_factor * risk_adjustment
        
        # Return scaling factors for all action dimensions
        return np.full(action_dim, final_scaling, dtype=np.float32)

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence in risk scaling assessment"""
        
        # Base confidence on data quality and session stability
        base_confidence = 0.7
        
        # Boost confidence with more data
        if len(self._volatility_history) > 50:
            base_confidence += 0.1
            
        if len(self._factor_history) > 20:
            base_confidence += 0.1
            
        # Reduce confidence for high volatility periods
        if self._volatility_history:
            recent_vol = float(np.mean(list(self._volatility_history)[-10:]))
            if recent_vol > 0.03:  # High volatility
                base_confidence -= 0.2
        
        # Session stability bonus
        if len(self._session_transitions) >= 3:
            recent_transitions = list(self._session_transitions)[-3:]
            if all(t['from'] == t['to'] for t in recent_transitions):
                base_confidence += 0.1
        
        return float(np.clip(base_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with validation"""
        self.asian_end = int(np.clip(genome.get("asian_end", self.asian_end), 4, 12))
        self.euro_end = int(np.clip(genome.get("euro_end", self.euro_end), 12, 20))
        self.decay = float(np.clip(genome.get("decay", self.decay), 0.8, 1.0))
        self.base_factor = float(np.clip(genome.get("base_factor", self.base_factor), 0.5, 1.5))
        self.vol_window = int(np.clip(genome.get("vol_window", self.vol_window), 20, 200))
        self.session_memory = int(np.clip(genome.get("session_memory", self.session_memory), 12, 48))
        
        self.genome = {
            "asian_end": self.asian_end,
            "euro_end": self.euro_end,
            "decay": self.decay,
            "base_factor": self.base_factor,
            "vol_window": self.vol_window,
            "session_memory": self.session_memory
        }
        
    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance tracking"""
        g = self.genome.copy()
        mutations = []
        
        if np.random.rand() < mutation_rate:
            old_val = g["asian_end"]
            g["asian_end"] = int(np.clip(self.asian_end + np.random.randint(-1, 2), 4, 12))
            mutations.append(f"asian_end: {old_val} → {g['asian_end']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["euro_end"]
            g["euro_end"] = int(np.clip(self.euro_end + np.random.randint(-1, 2), 12, 20))
            mutations.append(f"euro_end: {old_val} → {g['euro_end']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["decay"]
            g["decay"] = float(np.clip(self.decay + np.random.uniform(-0.05, 0.05), 0.8, 1.0))
            mutations.append(f"decay: {old_val:.3f} → {g['decay']:.3f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["base_factor"]
            g["base_factor"] = float(np.clip(self.base_factor + np.random.uniform(-0.2, 0.2), 0.5, 1.5))
            mutations.append(f"base_factor: {old_val:.3f} → {g['base_factor']:.3f}")
        
        if mutations:
            self.log_operator_info(f"Risk scaling mutation applied", changes=", ".join(mutations))
            
        self.set_genome(g)
        
    def crossover(self, other: "TimeAwareRiskScaling") -> "TimeAwareRiskScaling":
        """Enhanced crossover with compatibility checking"""
        if not isinstance(other, TimeAwareRiskScaling):
            self.log_operator_warning("Crossover with incompatible type")
            return self
            
        new_g = {k: np.random.choice([self.genome[k], other.genome[k]]) for k in self.genome}
        return TimeAwareRiskScaling(genome=new_g, debug=self.config.debug)

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check volatility profile validity
            if not np.all(np.isfinite(self.vol_profile)):
                return False
                
            # Check seasonality factor is reasonable
            if not (0.0 <= self.seasonality_factor <= 3.0):
                return False
                
            # Check session consistency
            if self._current_session not in ["asian", "european", "us", "closed", "unknown"]:
                return False
                
            # Check performance tracking integrity
            for session, perf in self._session_performance.items():
                if perf['count'] < 0 or not np.isfinite(perf['avg_volatility']):
                    return False
                    
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        risk_details = {
            'session_info': {
                'current_session': self._current_session,
                'session_changes': self._session_changes,
                'seasonality_factor': float(self.seasonality_factor),
                'session_performance': {k: v for k, v in self._session_performance.items() if v['count'] > 0}
            },
            'volatility_info': {
                'current_volatility': float(self._volatility_history[-1]) if self._volatility_history else 0.0,
                'avg_volatility': float(np.mean(list(self._volatility_history))) if self._volatility_history else 0.0,
                'vol_profile_max': float(self.vol_profile.max()),
                'vol_history_size': len(self._volatility_history)
            },
            'risk_info': {
                'risk_alerts': len(self._risk_alerts),
                'risk_score': float(self._risk_score_history[-1]) if self._risk_score_history else 0.5,
                'risk_multipliers': {k: float(v) for k, v in self._session_risk_multipliers.items()}
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(risk_details)
            return base_details
        
        return risk_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        return {
            "vol_profile": self.vol_profile.tolist(),
            "seasonality_factor": float(self.seasonality_factor),
            "genome": self.genome.copy(),
            "current_session": self._current_session,
            "session_changes": self._session_changes,
            "volatility_history": list(self._volatility_history)[-50:],  # Keep recent only
            "factor_history": list(self._factor_history)[-50:],
            "session_transitions": list(self._session_transitions)[-20:],
            "session_performance": self._session_performance.copy(),
            "risk_profile_by_hour": self._risk_profile_by_hour.tolist(),
            "session_risk_multipliers": {k: float(v) for k, v in self._session_risk_multipliers.items()}
        }
        
    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        self.vol_profile = np.array(module_state.get("vol_profile", [0.0]*24), dtype=np.float32)
        self.seasonality_factor = float(module_state.get("seasonality_factor", 1.0))
        self.set_genome(module_state.get("genome", self.genome))
        self._current_session = module_state.get("current_session", "unknown")
        self._session_changes = module_state.get("session_changes", 0)
        self._volatility_history = deque(module_state.get("volatility_history", []), maxlen=self.vol_window)
        self._factor_history = deque(module_state.get("factor_history", []), maxlen=100)
        self._session_transitions = deque(module_state.get("session_transitions", []), maxlen=50)
        self._session_performance = module_state.get("session_performance", {})
        self._risk_profile_by_hour = np.array(module_state.get("risk_profile_by_hour", [1.0]*24), dtype=np.float32)
        
        # Safely restore session risk multipliers
        risk_multipliers = module_state.get("session_risk_multipliers", {
            "asian": 1.0, "european": 1.0, "us": 1.0, "closed": 0.5
        })
        self._session_risk_multipliers = {k: float(v) for k, v in risk_multipliers.items()}

    def get_risk_scaling_report(self) -> str:
        """Generate operator-friendly risk scaling report"""
        
        # Current status
        current_hour = datetime.datetime.now().hour % 24
        
        # Session performance summary
        best_session = max(self._session_performance.keys(), 
                          key=lambda s: self._session_performance[s].get('total_factor', 0) / max(self._session_performance[s].get('count', 1), 1))
        
        # Risk level assessment
        if len(self._risk_score_history) > 0:
            current_risk = self._risk_score_history[-1]
            if current_risk > 0.7:
                risk_desc = "⚠️ High Risk"
            elif current_risk > 0.4:
                risk_desc = "⚡ Moderate Risk"
            else:
                risk_desc = "✅ Low Risk"
        else:
            risk_desc = "📊 Assessing"
            current_risk = 0.5
        
        return f"""
⏰ TIME-AWARE RISK SCALING
═══════════════════════════════════════
🕐 Current Time: {current_hour:02d}:00 ({self._current_session.title()} Session)
📊 Risk Factor: {self.seasonality_factor:.3f}
🎯 Risk Level: {risk_desc} ({current_risk:.3f})

📈 SESSION PERFORMANCE
• Best Session: {best_session.title()}
• Session Changes: {self._session_changes}
• Current Session Score: {self._get_session_performance_score():.3f}

⚡ VOLATILITY PROFILE
• Current Volatility: {float(self._volatility_history[-1]):.5f if self._volatility_history else 0:.5f}
• Average Volatility: {float(np.mean(list(self._volatility_history))):.5f if self._volatility_history else 0:.5f}
• Max Vol Hour: {int(np.argmax(self.vol_profile)):02d}:00
• Min Vol Hour: {int(np.argmin(self.vol_profile)):02d}:00

🎛️ CONFIGURATION
• Asian Session: 00:00 - {self.asian_end:02d}:00
• European Session: {self.asian_end:02d}:00 - {self.euro_end:02d}:00
• US Session: {self.euro_end:02d}:00 - 24:00
• Decay Factor: {self.decay:.3f}
• Base Factor: {self.base_factor:.3f}

🚨 RISK ALERTS
• Active Alerts: {len(self._risk_alerts)}
• Historical Risk Events: {sum(perf.get('risk_events', 0) for perf in self._session_performance.values())}
        """

    # Maintain backward compatibility
    def step(self, **kwargs):
        """Backward compatibility step method"""
        self._step_impl(None, **kwargs)

    def get_state(self):
        """Backward compatibility state method"""
        return super().get_state()

    def set_state(self, state):
        """Backward compatibility state method"""
        super().set_state(state)