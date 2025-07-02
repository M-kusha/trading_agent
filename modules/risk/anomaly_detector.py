# ─────────────────────────────────────────────────────────────
# File: modules/risk/anomaly_detector.py
# Enhanced with InfoBus integration & intelligent training mode
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque, defaultdict
from typing import Dict, Any, List, Optional, Tuple
import datetime

from modules.core.core import Module, ModuleConfig, audit_step
from modules.core.mixins import RiskMixin, AnalysisMixin, StateManagementMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor, InfoBusUpdater, extract_standard_context
from modules.utils.audit_utils import AuditTrailManager, format_operator_message


class AnomalyDetector(Module, RiskMixin, AnalysisMixin, StateManagementMixin):
    """
    Enhanced anomaly detection system with InfoBus integration.
    Intelligently detects trading anomalies with context-aware thresholds.
    """

    def __init__(
        self,
        pnl_limit: float = 1000.0,
        volume_zscore: float = 3.0,
        price_zscore: float = 3.0,
        observation_zscore: float = 4.0,
        enabled: bool = True,
        history_size: int = 100,
        training_mode: bool = True,
        adaptive_thresholds: bool = True,
        debug: bool = True,
        **kwargs
    ):
        # Initialize with enhanced config
        config = ModuleConfig(
            debug=debug,
            max_history=max(history_size, 50),
            audit_enabled=kwargs.get('audit_enabled', True),
            **kwargs
        )
        super().__init__(config)
        
        # Initialize mixins
        self._initialize_risk_state()
        self._initialize_analysis_state()
        
        # Core configuration
        self.enabled = enabled
        self.training_mode = training_mode
        self.adaptive_thresholds = adaptive_thresholds
        
        # Detection thresholds
        self.base_thresholds = {
            'pnl_limit': float(pnl_limit),
            'volume_zscore': float(volume_zscore),
            'price_zscore': float(price_zscore),
            'observation_zscore': float(observation_zscore)
        }
        
        # Adaptive threshold system
        self.current_thresholds = self.base_thresholds.copy()
        self.threshold_history = deque(maxlen=50)
        
        # Data history with intelligent sizing
        self.pnl_history = deque(maxlen=history_size)
        self.volume_history = deque(maxlen=history_size)
        self.price_history = deque(maxlen=history_size)
        self.observation_history = deque(maxlen=min(history_size, 50))  # Smaller for observations
        
        # Enhanced anomaly tracking
        self.anomalies: Dict[str, List[Dict[str, Any]]] = {
            "pnl": [],
            "volume": [],
            "price": [],
            "observation": [],
            "pattern": [],
            "correlation": [],
            "volatility": []
        }
        
        # State tracking
        self.anomaly_score = 0.0
        self.step_count = 0
        self.detection_stats = defaultdict(int)
        self.false_positive_tracker = deque(maxlen=100)
        
        # Context-aware detection
        self.regime_baselines = defaultdict(lambda: defaultdict(list))
        self.session_baselines = defaultdict(lambda: defaultdict(list))
        
        # Audit and performance tracking
        self.audit_manager = AuditTrailManager("AnomalyDetector")
        self._last_significant_anomaly = None
        self._anomaly_frequency = deque(maxlen=100)
        
        self.log_operator_info(
            "🔍 Enhanced Anomaly Detector initialized",
            pnl_limit=f"€{pnl_limit:,.0f}",
            training_mode=training_mode,
            adaptive_thresholds=adaptive_thresholds,
            history_size=history_size
        )

    def reset(self) -> None:
        """Enhanced reset with comprehensive state cleanup"""
        super().reset()
        self._reset_risk_state()
        self._reset_analysis_state()
        
        # Clear detection state
        for anomaly_type in self.anomalies:
            self.anomalies[anomaly_type].clear()
        
        self.anomaly_score = 0.0
        self.step_count = 0
        self.detection_stats.clear()
        self.false_positive_tracker.clear()
        
        # Clear history (but keep thresholds for continuity)
        self.pnl_history.clear()
        self.volume_history.clear()
        self.price_history.clear()
        self.observation_history.clear()
        self._anomaly_frequency.clear()
        
        # Reset baselines but keep structure
        self.regime_baselines.clear()
        self.session_baselines.clear()
        
        self.log_operator_info("🔄 Anomaly Detector reset - detection state cleared")

    @audit_step
    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not info_bus:
            self.log_operator_warning("No InfoBus provided - detector inactive")
            return
        
        if not self.enabled:
            self.anomaly_score = 0.0
            return
        
        self.step_count += 1
        
        # Clear previous anomalies
        for anomaly_type in self.anomalies:
            self.anomalies[anomaly_type].clear()
        
        # Extract context for intelligent detection
        context = extract_standard_context(info_bus)
        
        # Perform comprehensive anomaly detection
        critical_found = self._detect_comprehensive_anomalies(info_bus, context)
        
        # Update adaptive thresholds
        if self.adaptive_thresholds:
            self._update_adaptive_thresholds(context)
        
        # Calculate risk score
        self._calculate_anomaly_score(context)
        
        # Update InfoBus with results
        self._update_info_bus(info_bus, critical_found)
        
        # Record audit trail for significant events
        if critical_found or self.anomaly_score > 0.3:
            self._record_comprehensive_audit(info_bus, context)
        
        # Update performance metrics
        self._update_detection_metrics()

    def _detect_comprehensive_anomalies(self, info_bus: InfoBus, 
                                      context: Dict[str, Any]) -> bool:
        """Comprehensive anomaly detection across all data types"""
        
        critical_found = False
        
        try:
            # 1. PnL anomaly detection
            if self._detect_pnl_anomalies(info_bus, context):
                critical_found = True
            
            # 2. Volume anomaly detection
            self._detect_volume_anomalies(info_bus, context)
            
            # 3. Price anomaly detection
            self._detect_price_anomalies(info_bus, context)
            
            # 4. Observation anomaly detection
            if self._detect_observation_anomalies(info_bus, context):
                critical_found = True
            
            # 5. Pattern anomaly detection
            self._detect_pattern_anomalies(info_bus, context)
            
            # 6. Correlation anomaly detection
            self._detect_correlation_anomalies(info_bus, context)
            
            # 7. Volatility anomaly detection
            self._detect_volatility_anomalies(info_bus, context)
            
        except Exception as e:
            self.log_operator_error(f"Anomaly detection failed: {e}")
            
        return critical_found

    def _detect_pnl_anomalies(self, info_bus: InfoBus, context: Dict[str, Any]) -> bool:
        """Enhanced PnL anomaly detection with context awareness"""
        
        # Extract PnL from multiple sources
        pnl = self._extract_pnl_from_info_bus(info_bus)
        
        if pnl is None:
            # Generate synthetic PnL only in training mode when needed for bootstrapping
            if self.training_mode and len(self.pnl_history) < 10:
                pnl = self._generate_synthetic_pnl(context)
            else:
                return False
        
        self.pnl_history.append(pnl)
        critical_found = False
        
        # 1. Absolute threshold check with context adjustment
        adjusted_limit = self._get_context_adjusted_limit('pnl_limit', context)
        
        if abs(pnl) > adjusted_limit:
            self.anomalies["pnl"].append({
                "type": "absolute_limit",
                "value": pnl,
                "threshold": adjusted_limit,
                "base_threshold": self.current_thresholds['pnl_limit'],
                "context": context.copy(),
                "severity": "critical"
            })
            
            self.log_operator_error(
                f"🚨 CRITICAL PnL anomaly: €{pnl:,.2f}",
                limit=f"€{adjusted_limit:,.0f}",
                regime=context.get('regime', 'unknown'),
                session=context.get('session', 'unknown')
            )
            critical_found = True
        
        # 2. Statistical anomaly detection (if sufficient history)
        if len(self.pnl_history) >= 20:
            z_score = self._calculate_robust_zscore(pnl, list(self.pnl_history))
            
            if z_score > 4.0:  # Conservative threshold
                self.anomalies["pnl"].append({
                    "type": "statistical",
                    "value": pnl,
                    "z_score": float(z_score),
                    "context": context.copy(),
                    "severity": "critical" if z_score > 6.0 else "warning"
                })
                
                if z_score > 6.0:
                    self.log_operator_error(f"🚨 Statistical PnL anomaly: z-score {z_score:.2f}")
                    critical_found = True
                else:
                    self.log_operator_warning(f"⚠️ Statistical PnL outlier: z-score {z_score:.2f}")
        
        # 3. Regime-specific detection
        self._update_regime_baseline('pnl', context, pnl)
        
        return critical_found

    def _detect_observation_anomalies(self, info_bus: InfoBus, 
                                    context: Dict[str, Any]) -> bool:
        """Enhanced observation anomaly detection"""
        
        # Extract observation from InfoBus
        obs = self._extract_observation_from_info_bus(info_bus)
        
        if obs is None:
            return False
        
        try:
            obs = np.array(obs, dtype=np.float32)
        except (ValueError, TypeError):
            self.log_operator_warning("Invalid observation format detected")
            return False
        
        critical_found = False
        
        # 1. Check for invalid values (critical)
        if np.isnan(obs).any() or np.isinf(obs).any():
            nan_count = int(np.isnan(obs).sum())
            inf_count = int(np.isinf(obs).sum())
            
            self.anomalies["observation"].append({
                "type": "invalid_values",
                "nan_count": nan_count,
                "inf_count": inf_count,
                "obs_shape": obs.shape,
                "severity": "critical"
            })
            
            self.log_operator_error(
                f"🚨 CRITICAL: Invalid observation values",
                nan_count=nan_count,
                inf_count=inf_count,
                shape=str(obs.shape)
            )
            critical_found = True
        
        # 2. Store valid observations for statistical analysis
        if not critical_found:
            self.observation_history.append(obs)
            
            # Statistical analysis (if sufficient history)
            if len(self.observation_history) >= 10:
                z_scores = self._calculate_observation_zscores(obs)
                extreme_threshold = self.current_thresholds['observation_zscore']
                
                extreme_indices = np.where(z_scores > extreme_threshold)[0]
                
                if len(extreme_indices) > 0:
                    severity = "critical" if np.max(z_scores) > extreme_threshold * 1.5 else "warning"
                    
                    self.anomalies["observation"].append({
                        "type": "extreme_values",
                        "indices": extreme_indices.tolist(),
                        "z_scores": z_scores[extreme_indices].tolist(),
                        "max_z_score": float(np.max(z_scores)),
                        "threshold": extreme_threshold,
                        "severity": severity
                    })
                    
                    if severity == "critical":
                        self.log_operator_error(
                            f"🚨 CRITICAL: Extreme observation values",
                            max_z_score=f"{np.max(z_scores):.2f}",
                            indices_count=len(extreme_indices)
                        )
                        critical_found = True
                    else:
                        self.log_operator_warning(f"⚠️ Observation outliers detected")
        
        return critical_found

    def _detect_volume_anomalies(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Enhanced volume anomaly detection"""
        
        volume = self._extract_volume_from_info_bus(info_bus)
        
        if volume is None:
            if self.training_mode and len(self.volume_history) < 10:
                volume = self._generate_synthetic_volume(context)
            else:
                return
        
        self.volume_history.append(volume)
        
        # Statistical analysis
        if len(self.volume_history) >= 20:
            z_score = self._calculate_robust_zscore(volume, list(self.volume_history))
            threshold = self.current_thresholds['volume_zscore']
            
            if z_score > threshold:
                self.anomalies["volume"].append({
                    "type": "statistical",
                    "value": volume,
                    "z_score": float(z_score),
                    "threshold": threshold,
                    "context": context.copy(),
                    "severity": "warning"
                })
                
                self.log_operator_warning(
                    f"⚠️ Volume anomaly: {volume:,.0f}",
                    z_score=f"{z_score:.2f}",
                    regime=context.get('regime', 'unknown')
                )

    def _detect_price_anomalies(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Enhanced price anomaly detection"""
        
        price = self._extract_price_from_info_bus(info_bus)
        
        if price is None:
            if self.training_mode and len(self.price_history) < 10:
                price = self._generate_synthetic_price(context)
            else:
                return
        
        self.price_history.append(price)
        
        # Price jump detection
        if len(self.price_history) >= 2:
            prev_price = self.price_history[-2]
            
            if prev_price > 0:
                price_change = abs((price - prev_price) / prev_price)
                
                # Context-adjusted threshold
                volatility_level = context.get('volatility_level', 'medium')
                jump_threshold = {
                    'low': 0.05,      # 5%
                    'medium': 0.08,   # 8%
                    'high': 0.12,     # 12%
                    'extreme': 0.20   # 20%
                }.get(volatility_level, 0.08)
                
                if price_change > jump_threshold:
                    self.anomalies["price"].append({
                        "type": "price_jump",
                        "change_pct": float(price_change),
                        "prev_price": prev_price,
                        "current_price": price,
                        "threshold": jump_threshold,
                        "context": context.copy(),
                        "severity": "warning"
                    })
                    
                    self.log_operator_warning(
                        f"⚠️ Price jump: {price_change:.1%}",
                        from_price=f"{prev_price:.5f}",
                        to_price=f"{price:.5f}",
                        volatility=volatility_level
                    )

    def _detect_pattern_anomalies(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Enhanced pattern anomaly detection"""
        
        trades = InfoBusExtractor.get_recent_trades(info_bus)
        
        if not trades:
            return
        
        # Analyze trading patterns
        directions = []
        sizes = []
        
        for trade in trades:
            size = trade.get("size", trade.get("volume", 0))
            if size != 0:
                directions.append(np.sign(size))
                sizes.append(abs(size))
        
        if not directions:
            return
        
        # 1. Unidirectional trading detection
        if len(set(directions)) == 1 and len(trades) > 8:  # Increased threshold
            self.anomalies["pattern"].append({
                "type": "unidirectional",
                "trade_count": len(trades),
                "direction": directions[0],
                "context": context.copy(),
                "severity": "info"
            })
            
            direction_text = "BUY" if directions[0] > 0 else "SELL"
            self.log_operator_info(
                f"📊 Unidirectional pattern: {len(trades)} {direction_text} trades",
                regime=context.get('regime', 'unknown')
            )
        
        # 2. High frequency trading detection
        if len(trades) > 25:  # Increased threshold for training
            self.anomalies["pattern"].append({
                "type": "high_frequency",
                "trade_count": len(trades),
                "context": context.copy(),
                "severity": "warning"
            })
            
            self.log_operator_warning(f"⚠️ High frequency trading: {len(trades)} trades")
        
        # 3. Size anomaly detection
        if sizes:
            size_z_scores = [abs((s - np.mean(sizes)) / max(np.std(sizes), 0.01)) for s in sizes]
            extreme_sizes = [i for i, z in enumerate(size_z_scores) if z > 3.0]
            
            if extreme_sizes:
                self.anomalies["pattern"].append({
                    "type": "extreme_trade_sizes",
                    "extreme_indices": extreme_sizes,
                    "z_scores": [size_z_scores[i] for i in extreme_sizes],
                    "context": context.copy(),
                    "severity": "info"
                })

    def _detect_correlation_anomalies(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Detect correlation anomalies between instruments"""
        
        prices = info_bus.get('prices', {})
        
        if len(prices) < 2:
            return
        
        # Check for unusual correlation patterns
        instruments = list(prices.keys())
        
        # Store price data for correlation analysis
        for inst in instruments:
            if inst not in self.regime_baselines[context.get('regime', 'unknown')]:
                self.regime_baselines[context.get('regime', 'unknown')][inst] = deque(maxlen=50)
            
            self.regime_baselines[context.get('regime', 'unknown')][inst].append(prices[inst])
        
        # Analyze correlations if sufficient data
        regime = context.get('regime', 'unknown')
        if all(len(self.regime_baselines[regime].get(inst, [])) >= 10 for inst in instruments):
            correlations = self._calculate_cross_correlations(instruments, regime)
            
            # Detect unusual correlations
            for (inst1, inst2), corr in correlations.items():
                if abs(corr) > 0.95 and inst1 != inst2:  # Very high correlation
                    self.anomalies["correlation"].append({
                        "type": "high_correlation",
                        "instruments": [inst1, inst2],
                        "correlation": float(corr),
                        "context": context.copy(),
                        "severity": "info"
                    })

    def _detect_volatility_anomalies(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Detect volatility anomalies"""
        
        # Calculate current volatility from recent price history
        if len(self.price_history) >= 10:
            prices = np.array(list(self.price_history)[-10:])
            returns = np.diff(np.log(prices + 1e-8))  # Log returns
            current_vol = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Store volatility in history
            if not hasattr(self, '_volatility_history'):
                self._volatility_history = deque(maxlen=50)
            
            self._volatility_history.append(current_vol)
            
            # Detect volatility spikes
            if len(self._volatility_history) >= 10:
                vol_z_score = self._calculate_robust_zscore(current_vol, list(self._volatility_history))
                
                if vol_z_score > 3.0:
                    self.anomalies["volatility"].append({
                        "type": "volatility_spike",
                        "current_vol": float(current_vol),
                        "z_score": float(vol_z_score),
                        "context": context.copy(),
                        "severity": "warning" if vol_z_score > 4.0 else "info"
                    })
                    
                    if vol_z_score > 4.0:
                        self.log_operator_warning(
                            f"⚠️ Volatility spike: {current_vol:.1%} annualized",
                            z_score=f"{vol_z_score:.2f}"
                        )

    def _extract_pnl_from_info_bus(self, info_bus: InfoBus) -> Optional[float]:
        """Extract PnL from InfoBus with multiple fallback methods"""
        
        # Method 1: Direct PnL from trades
        trades = InfoBusExtractor.get_recent_trades(info_bus)
        if trades:
            total_pnl = sum(trade.get('pnl', 0) for trade in trades)
            if total_pnl != 0:
                return float(total_pnl)
        
        # Method 2: Balance change
        risk_data = info_bus.get('risk', {})
        current_balance = risk_data.get('balance', risk_data.get('equity', None))
        
        if current_balance is not None:
            if hasattr(self, '_last_balance') and self._last_balance is not None:
                pnl = current_balance - self._last_balance
                self._last_balance = current_balance
                return float(pnl)
            else:
                self._last_balance = current_balance
        
        # Method 3: From position unrealized PnL
        positions = InfoBusExtractor.get_positions(info_bus)
        if positions:
            unrealized_pnl = sum(pos.get('unrealised_pnl', 0) for pos in positions)
            if unrealized_pnl != 0:
                return float(unrealized_pnl)
        
        return None

    def _extract_volume_from_info_bus(self, info_bus: InfoBus) -> Optional[float]:
        """Extract volume from InfoBus"""
        
        # From recent trades
        trades = InfoBusExtractor.get_recent_trades(info_bus)
        if trades:
            total_volume = sum(abs(trade.get("size", trade.get("volume", 0))) for trade in trades)
            if total_volume > 0:
                return float(total_volume)
        
        # From market data
        market_data = info_bus.get('market_data', {})
        if 'volume' in market_data:
            return float(market_data['volume'])
        
        return None

    def _extract_price_from_info_bus(self, info_bus: InfoBus) -> Optional[float]:
        """Extract representative price from InfoBus"""
        
        prices = info_bus.get('prices', {})
        if prices:
            # Use first available price as representative
            return float(list(prices.values())[0])
        
        return None

    def _extract_observation_from_info_bus(self, info_bus: InfoBus) -> Optional[np.ndarray]:
        """Extract observation array from InfoBus"""
        
        # Try to get from module data
        module_data = info_bus.get('module_data', {})
        
        for module_name in ['features', 'feature_engine', 'observation']:
            if module_name in module_data:
                obs_data = module_data[module_name]
                if isinstance(obs_data, dict) and 'observation' in obs_data:
                    return obs_data['observation']
                elif isinstance(obs_data, (list, np.ndarray)):
                    return obs_data
        
        # Try direct observation field
        if 'observation' in info_bus:
            return info_bus['observation']
        
        return None

    def _generate_synthetic_pnl(self, context: Dict[str, Any]) -> float:
        """Generate realistic synthetic PnL for training mode"""
        
        # Base synthetic PnL
        base_pnl = np.random.normal(0, 50)
        
        # Adjust for market regime
        regime = context.get('regime', 'unknown')
        if regime == 'volatile':
            base_pnl *= 2.0  # Higher volatility
        elif regime == 'trending':
            base_pnl *= 1.5  # Moderate increase
        
        # Add occasional spikes for anomaly detection training
        if np.random.rand() < 0.05:  # 5% chance
            spike_pnl = np.random.choice([-1, 1]) * np.random.uniform(200, 800)
            base_pnl += spike_pnl
        
        return float(base_pnl)

    def _generate_synthetic_volume(self, context: Dict[str, Any]) -> float:
        """Generate realistic synthetic volume for training mode"""
        
        base_volume = abs(np.random.normal(5000, 2000))
        
        # Adjust for session
        session = context.get('session', 'unknown')
        if session == 'european':
            base_volume *= 1.5  # Higher European session volume
        elif session == 'american':
            base_volume *= 1.3  # Moderate increase
        
        return float(max(base_volume, 100))  # Minimum volume

    def _generate_synthetic_price(self, context: Dict[str, Any]) -> float:
        """Generate realistic synthetic price for training mode"""
        
        if self.price_history:
            last_price = self.price_history[-1]
            # Random walk with regime adjustment
            change_pct = np.random.normal(0, 0.002)  # 0.2% std dev
            
            regime = context.get('regime', 'unknown')
            if regime == 'volatile':
                change_pct *= 3.0  # More volatile
            elif regime == 'trending':
                change_pct += np.random.choice([-1, 1]) * 0.001  # Slight trend bias
            
            return float(last_price * (1 + change_pct))
        else:
            # Starting price around typical forex levels
            return float(np.random.uniform(1.0, 2.0))

    def _calculate_robust_zscore(self, value: float, history: List[float]) -> float:
        """Calculate robust z-score using median and MAD"""
        
        if len(history) < 3:
            return 0.0
        
        history_array = np.array(history)
        median = np.median(history_array)
        mad = np.median(np.abs(history_array - median))
        
        # Use MAD-based standard deviation estimate
        mad_std = mad * 1.4826  # Conversion factor for normal distribution
        
        if mad_std < 1e-8:  # Avoid division by zero
            return 0.0
        
        return abs((value - median) / mad_std)

    def _calculate_observation_zscores(self, obs: np.ndarray) -> np.ndarray:
        """Calculate z-scores for observation vector"""
        
        if len(self.observation_history) < 2:
            return np.zeros(len(obs))
        
        # Stack historical observations
        obs_stack = np.vstack(self.observation_history)
        
        # Calculate robust statistics
        medians = np.median(obs_stack, axis=0)
        mads = np.median(np.abs(obs_stack - medians), axis=0)
        
        # Convert to z-scores
        mad_stds = mads * 1.4826
        mad_stds[mad_stds < 1e-8] = 1.0  # Avoid division by zero
        
        z_scores = np.abs((obs - medians) / mad_stds)
        
        return z_scores

    def _calculate_cross_correlations(self, instruments: List[str], 
                                    regime: str) -> Dict[Tuple[str, str], float]:
        """Calculate cross-correlations between instruments"""
        
        correlations = {}
        
        for i, inst1 in enumerate(instruments):
            for j, inst2 in enumerate(instruments):
                if i <= j:  # Avoid duplicate calculations
                    data1 = list(self.regime_baselines[regime].get(inst1, []))
                    data2 = list(self.regime_baselines[regime].get(inst2, []))
                    
                    if len(data1) >= 10 and len(data2) >= 10:
                        corr = np.corrcoef(data1, data2)[0, 1]
                        if not np.isnan(corr):
                            correlations[(inst1, inst2)] = corr
        
        return correlations

    def _get_context_adjusted_limit(self, limit_name: str, context: Dict[str, Any]) -> float:
        """Get context-adjusted threshold"""
        
        base_limit = self.current_thresholds[limit_name]
        
        # Adjust for market regime
        regime = context.get('regime', 'unknown')
        volatility_level = context.get('volatility_level', 'medium')
        
        multiplier = 1.0
        
        if limit_name == 'pnl_limit':
            # More lenient in volatile markets
            if volatility_level == 'extreme':
                multiplier = 2.0
            elif volatility_level == 'high':
                multiplier = 1.5
            elif regime == 'volatile':
                multiplier = 1.3
        
        return base_limit * multiplier

    def _update_adaptive_thresholds(self, context: Dict[str, Any]) -> None:
        """Update thresholds based on recent performance"""
        
        if not self.adaptive_thresholds or self.step_count < 100:
            return
        
        try:
            # Adapt PnL threshold based on recent data
            if len(self.pnl_history) >= 50:
                recent_pnls = list(self.pnl_history)[-50:]
                pnl_array = np.array(recent_pnls)
                
                # Calculate adaptive threshold
                pnl_std = np.std(pnl_array)
                pnl_95th = np.percentile(np.abs(pnl_array), 95)
                
                # New threshold: higher of 95th percentile or 3 std devs
                adaptive_threshold = max(pnl_95th, 3 * pnl_std)
                adaptive_threshold = max(adaptive_threshold, self.base_thresholds['pnl_limit'] * 0.5)
                adaptive_threshold = min(adaptive_threshold, self.base_thresholds['pnl_limit'] * 3.0)
                
                # Smooth threshold changes
                old_threshold = self.current_thresholds['pnl_limit']
                self.current_thresholds['pnl_limit'] = (
                    0.8 * old_threshold + 0.2 * adaptive_threshold
                )
            
            # Store threshold history
            self.threshold_history.append({
                'timestamp': datetime.datetime.now().isoformat(),
                'thresholds': self.current_thresholds.copy(),
                'context': context.copy()
            })
            
        except Exception as e:
            self.log_operator_warning(f"Threshold adaptation failed: {e}")

    def _update_regime_baseline(self, data_type: str, context: Dict[str, Any], value: float) -> None:
        """Update regime-specific baselines"""
        
        regime = context.get('regime', 'unknown')
        session = context.get('session', 'unknown')
        
        # Store in regime baselines
        if data_type not in self.regime_baselines[regime]:
            self.regime_baselines[regime][data_type] = deque(maxlen=100)
        
        self.regime_baselines[regime][data_type].append(value)
        
        # Store in session baselines
        if data_type not in self.session_baselines[session]:
            self.session_baselines[session][data_type] = deque(maxlen=100)
        
        self.session_baselines[session][data_type].append(value)

    def _calculate_anomaly_score(self, context: Dict[str, Any]) -> None:
        """Calculate comprehensive anomaly score"""
        
        # Severity weights
        severity_weights = {
            "critical": 1.0,
            "warning": 0.5,
            "info": 0.1
        }
        
        total_score = 0.0
        total_anomalies = 0
        
        # Weight different anomaly types
        type_weights = {
            "pnl": 0.4,
            "observation": 0.3,
            "volume": 0.1,
            "price": 0.1,
            "pattern": 0.05,
            "correlation": 0.025,
            "volatility": 0.025
        }
        
        for anomaly_type, anomalies in self.anomalies.items():
            if not anomalies:
                continue
            
            type_weight = type_weights.get(anomaly_type, 0.1)
            
            for anomaly in anomalies:
                severity = anomaly.get("severity", "info")
                severity_weight = severity_weights.get(severity, 0.1)
                
                total_score += type_weight * severity_weight
                total_anomalies += 1
        
        # Normalize and apply context adjustments
        if total_anomalies > 0:
            base_score = min(total_score, 1.0)
            
            # Adjust for market context
            regime = context.get('regime', 'unknown')
            if regime == 'volatile':
                base_score *= 0.8  # More tolerance in volatile markets
            
            self.anomaly_score = base_score
        else:
            self.anomaly_score = 0.0
        
        # Track frequency
        self._anomaly_frequency.append(self.anomaly_score)

    def _update_info_bus(self, info_bus: InfoBus, critical_found: bool) -> None:
        """Update InfoBus with detection results"""
        
        # Add module data
        InfoBusUpdater.add_module_data(info_bus, 'anomaly_detector', {
            'anomaly_score': self.anomaly_score,
            'anomalies': {k: len(v) for k, v in self.anomalies.items() if v},
            'critical_found': critical_found,
            'thresholds': self.current_thresholds.copy(),
            'detection_stats': dict(self.detection_stats),
            'data_sufficiency': {
                'pnl_history': len(self.pnl_history),
                'volume_history': len(self.volume_history),
                'price_history': len(self.price_history),
                'observation_history': len(self.observation_history)
            }
        })
        
        # Add risk data
        InfoBusUpdater.update_risk_snapshot(info_bus, {
            'anomaly_risk_score': self.anomaly_score,
            'anomalies_detected': sum(len(v) for v in self.anomalies.values()),
            'critical_anomalies': critical_found
        })
        
        # Add alerts for critical situations
        if critical_found:
            InfoBusUpdater.add_alert(
                info_bus,
                "Critical anomalies detected - review system behavior",
                severity="critical",
                module="AnomalyDetector"
            )
        elif self.anomaly_score > 0.5:
            InfoBusUpdater.add_alert(
                info_bus,
                f"Elevated anomaly score: {self.anomaly_score:.1%}",
                severity="warning",
                module="AnomalyDetector"
            )

    def _record_comprehensive_audit(self, info_bus: InfoBus, context: Dict[str, Any]) -> None:
        """Record comprehensive audit trail"""
        
        audit_data = {
            'anomaly_score': self.anomaly_score,
            'context': context,
            'anomalies': {
                anomaly_type: anomalies
                for anomaly_type, anomalies in self.anomalies.items()
                if anomalies
            },
            'thresholds': self.current_thresholds.copy(),
            'data_status': {
                'pnl_history_size': len(self.pnl_history),
                'volume_history_size': len(self.volume_history),
                'price_history_size': len(self.price_history),
                'observation_history_size': len(self.observation_history)
            },
            'step_count': self.step_count,
            'training_mode': self.training_mode
        }
        
        self.audit_manager.record_event(
            event_type="anomaly_detection",
            module="AnomalyDetector",
            details=audit_data,
            severity="critical" if any(
                a.get("severity") == "critical"
                for anomalies in self.anomalies.values()
                for a in anomalies
            ) else "warning" if self.anomaly_score > 0.3 else "info"
        )

    def _update_detection_metrics(self) -> None:
        """Update performance and detection metrics"""
        
        # Update detection statistics
        total_anomalies = sum(len(v) for v in self.anomalies.values())
        self.detection_stats['total_anomalies'] += total_anomalies
        
        for anomaly_type, anomalies in self.anomalies.items():
            self.detection_stats[f'{anomaly_type}_count'] += len(anomalies)
        
        # Update performance metrics
        self._update_performance_metric('anomaly_score', self.anomaly_score)
        self._update_performance_metric('total_anomalies', total_anomalies)
        
        # Data sufficiency metrics
        self._update_performance_metric('pnl_data_sufficiency', 
                                      min(len(self.pnl_history) / 50.0, 1.0))
        self._update_performance_metric('volume_data_sufficiency',
                                      min(len(self.volume_history) / 20.0, 1.0))

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components for model integration"""
        
        try:
            # Core anomaly metrics
            anomaly_score = float(self.anomaly_score)
            
            # Critical anomaly indicator
            has_critical = float(any(
                a.get("severity") == "critical"
                for anomalies in self.anomalies.values()
                for a in anomalies
            ))
            
            # Anomaly frequency
            total_anomalies = sum(len(v) for v in self.anomalies.values())
            anomaly_frequency = min(float(total_anomalies) / 10.0, 1.0)
            
            # Data sufficiency indicators
            pnl_sufficiency = min(len(self.pnl_history) / 50.0, 1.0)
            volume_sufficiency = min(len(self.volume_history) / 30.0, 1.0)
            observation_sufficiency = min(len(self.observation_history) / 20.0, 1.0)
            
            # Trend indicators
            if len(self._anomaly_frequency) >= 10:
                recent_trend = np.mean(list(self._anomaly_frequency)[-10:])
                earlier_trend = np.mean(list(self._anomaly_frequency)[-20:-10]) if len(self._anomaly_frequency) >= 20 else recent_trend
                trend_direction = float(np.sign(recent_trend - earlier_trend))
            else:
                trend_direction = 0.0
            
            # Adaptive threshold status
            threshold_adaptation = float(
                self.current_thresholds['pnl_limit'] / self.base_thresholds['pnl_limit']
            ) if self.adaptive_thresholds else 1.0
            
            return np.array([
                anomaly_score,              # Current anomaly score
                has_critical,               # Critical anomaly indicator
                anomaly_frequency,          # Anomaly frequency
                pnl_sufficiency,           # PnL data sufficiency
                volume_sufficiency,        # Volume data sufficiency
                observation_sufficiency,   # Observation data sufficiency
                trend_direction,           # Anomaly trend direction
                threshold_adaptation       # Threshold adaptation ratio
            ], dtype=np.float32)
            
        except Exception as e:
            self.log_operator_error(f"Observation generation failed: {e}")
            return np.zeros(8, dtype=np.float32)

    def get_detection_report(self) -> str:
        """Generate operator-friendly detection report"""
        
        total_anomalies = sum(len(v) for v in self.anomalies.values())
        
        # Status indicators
        if self.anomaly_score > 0.7:
            anomaly_status = "🚨 Critical"
        elif self.anomaly_score > 0.3:
            anomaly_status = "⚠️ Elevated"
        else:
            anomaly_status = "✅ Normal"
        
        # Training mode status
        mode_status = "🎓 Training" if self.training_mode else "🚀 Live"
        
        # Data sufficiency status
        pnl_sufficiency = len(self.pnl_history) / (self.pnl_history.maxlen or 1)
        if pnl_sufficiency > 0.8:
            data_status = "✅ Sufficient"
        elif pnl_sufficiency > 0.5:
            data_status = "⚡ Partial"
        else:
            data_status = "❌ Limited"
        
        # Current anomalies breakdown
        anomaly_lines = []
        for anomaly_type, anomalies in self.anomalies.items():
            if anomalies:
                critical_count = sum(1 for a in anomalies if a.get('severity') == 'critical')
                warning_count = sum(1 for a in anomalies if a.get('severity') == 'warning')
                emoji = "🚨" if critical_count > 0 else "⚠️" if warning_count > 0 else "ℹ️"
                anomaly_lines.append(f"  {emoji} {anomaly_type.title()}: {len(anomalies)} ({critical_count} critical)")
        
        # Recent performance
        if len(self._anomaly_frequency) >= 10:
            recent_avg = np.mean(list(self._anomaly_frequency)[-10:])
            recent_trend = "📈 Rising" if recent_avg > self.anomaly_score else "📉 Falling"
        else:
            recent_trend = "📊 Stable"
        
        return f"""
🔍 ENHANCED ANOMALY DETECTOR
═══════════════════════════════════════
🎯 Anomaly Status: {anomaly_status} ({self.anomaly_score:.1%})
🔧 Detection Mode: {mode_status}
📊 Data Status: {data_status} ({pnl_sufficiency:.1%})
🔄 Detector Enabled: {'✅ Yes' if self.enabled else '❌ No'}

⚖️ DETECTION THRESHOLDS
• PnL Limit: €{self.current_thresholds['pnl_limit']:,.0f}
• Volume Z-Score: {self.current_thresholds['volume_zscore']:.1f}
• Price Z-Score: {self.current_thresholds['price_zscore']:.1f}
• Observation Z-Score: {self.current_thresholds['observation_zscore']:.1f}
• Adaptive Thresholds: {'✅ Enabled' if self.adaptive_thresholds else '❌ Disabled'}

📊 DATA COLLECTION STATUS
• PnL History: {len(self.pnl_history)}/{self.pnl_history.maxlen}
• Volume History: {len(self.volume_history)}/{self.volume_history.maxlen}
• Price History: {len(self.price_history)}/{self.price_history.maxlen}
• Observation History: {len(self.observation_history)}/{self.observation_history.maxlen}

🚨 CURRENT ANOMALIES ({total_anomalies} total)
{chr(10).join(anomaly_lines) if anomaly_lines else "  ✅ No anomalies detected"}

📈 DETECTION STATISTICS
• Total Detections: {self.detection_stats.get('total_anomalies', 0)}
• PnL Anomalies: {self.detection_stats.get('pnl_count', 0)}
• Volume Anomalies: {self.detection_stats.get('volume_count', 0)}
• Price Anomalies: {self.detection_stats.get('price_count', 0)}
• Observation Anomalies: {self.detection_stats.get('observation_count', 0)}

📊 PERFORMANCE TRENDS
• Recent Trend: {recent_trend}
• Detection Frequency: {(len([f for f in self._anomaly_frequency if f > 0.1]) / max(len(self._anomaly_frequency), 1)):.1%}
• Step Count: {self.step_count:,}

🔧 THRESHOLD ADAPTATION
• Current vs Base PnL: {(self.current_thresholds['pnl_limit'] / self.base_thresholds['pnl_limit']):.2f}x
• Threshold Updates: {len(self.threshold_history)}
        """

    # ================== LEGACY COMPATIBILITY ==================

    def step(self, pnl: Optional[float] = None, obs: Optional[np.ndarray] = None,
            volume: Optional[float] = None, price: Optional[float] = None,
            trades: Optional[List[Dict[str, Any]]] = None, **kwargs) -> bool:
        """Legacy compatibility method"""
        
        # Create mock InfoBus from legacy parameters
        mock_info_bus = {
            'step_idx': self.step_count,
            'timestamp': datetime.datetime.now().isoformat(),
            'recent_trades': trades or [],
            'prices': {'default': price} if price is not None else {},
            'market_data': {'volume': volume} if volume is not None else {},
            'observation': obs
        }
        
        # Add PnL info
        if pnl is not None:
            mock_info_bus['risk'] = {'balance': getattr(self, '_last_balance', 0) + pnl}
        
        # Extract context (will be mostly defaults)
        context = {
            'regime': kwargs.get('regime', 'unknown'),
            'session': kwargs.get('session', 'unknown'),
            'volatility_level': kwargs.get('volatility_level', 'medium')
        }
        
        # Use enhanced detection
        critical_found = self._detect_comprehensive_anomalies(mock_info_bus, context)
        self._calculate_anomaly_score(context)
        
        return critical_found