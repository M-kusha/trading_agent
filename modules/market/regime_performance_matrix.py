# ─────────────────────────────────────────────────────────────
# File: modules/market/regime_performance_matrix.py
# Enhanced with new infrastructure - InfoBus integration & mixins!
# ─────────────────────────────────────────────────────────────

import numpy as np
from collections import deque
from typing import Dict, Any, Optional, Tuple, List
import datetime

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin, RiskMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class RegimePerformanceMatrix(Module, AnalysisMixin, RiskMixin):
    def __init__(self,
                 n_regimes: int = 3,
                 decay: float = 0.95,
                 debug: bool = True,
                 genome: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 1) initialize genome‐based attrs first (so n_regimes exists)
        self._initialize_genome_parameters(genome, n_regimes, decay)

        # 2) now call base ctor (it will invoke _initialize_module_state)
        config = ModuleConfig(
            debug=debug,
            max_history=1000,
            **kwargs
        )
        super().__init__(config)

        # 3) any further logging or setup
        self.log_operator_info(
            "Regime performance matrix initialized",
            regimes=self.n_regimes,
            decay_factor=f"{self.decay:.3f}",
            matrix_size=f"{self.n_regimes}x{self.n_regimes}",
            stress_scenarios=len(self._stress_scenarios)
        )


    def _initialize_genome_parameters(self, genome: Optional[Dict], n_regimes: int, decay: float):
        """Initialize genome-based parameters"""
        if genome:
            self.n_regimes = int(genome.get("n_regimes", n_regimes))
            self.decay = float(genome.get("decay", decay))
            self.vol_history_size = int(genome.get("vol_history_size", 500))
            self.performance_window = int(genome.get("performance_window", 100))
            self.regime_sensitivity = float(genome.get("regime_sensitivity", 1.0))
        else:
            self.n_regimes = n_regimes
            self.decay = decay
            self.vol_history_size = 500
            self.performance_window = 100
            self.regime_sensitivity = 1.0

        # Store genome for evolution
        self.genome = {
            "n_regimes": self.n_regimes,
            "decay": self.decay,
            "vol_history_size": self.vol_history_size,
            "performance_window": self.performance_window,
            "regime_sensitivity": self.regime_sensitivity
        }

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        self._initialize_risk_state()
        
        # Performance matrix state
        self.matrix = np.zeros((self.n_regimes, self.n_regimes), np.float32)
        self.last_volatility = 0.0
        self.last_liquidity = 1.0
        
        # Enhanced tracking
        self.vol_history = deque(maxlen=self.vol_history_size)
        self.volatility_regimes = np.array([0.1, 0.3, 0.5], np.float32)
        self._performance_history = deque(maxlen=self.performance_window)
        self._regime_history = deque(maxlen=200)
        
        # Regime transition tracking
        self._regime_transitions = {}
        self._current_regime = 0
        self._predicted_regime_history = deque(maxlen=100)
        self._true_regime_history = deque(maxlen=100)
        
        # Enhanced analytics
        self._regime_accuracy_scores = np.zeros(self.n_regimes, np.float32)
        self._regime_pnl_tracking = {i: deque(maxlen=50) for i in range(self.n_regimes)}
        self._regime_characteristics = {}
        self._last_regime_update = None
        
        # Stress testing scenarios
        self._stress_scenarios = {
            "flash-crash": {"vol_mult": 3.0, "liq_mult": 0.2, "duration": 5},
            "rate-spike": {"vol_mult": 2.5, "liq_mult": 0.5, "duration": 10},
            "default-wave": {"vol_mult": 2.0, "liq_mult": 0.3, "duration": 15},
            "liquidity-crisis": {"vol_mult": 1.8, "liq_mult": 0.1, "duration": 20},
            "market-meltdown": {"vol_mult": 4.0, "liq_mult": 0.15, "duration": 8}
        }
        
        # Initialize regime characteristics
        for i in range(self.n_regimes):
            self._regime_characteristics[i] = {
                'avg_volatility': 0.0,
                'avg_pnl': 0.0,
                'count': 0,
                'best_predictions': 0,
                'worst_predictions': 0,
                'stability_score': 0.5
            }

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        self._reset_risk_state()
        
        # Module-specific reset
        self.matrix.fill(0.0)
        self.last_volatility = 0.0
        self.last_liquidity = 1.0
        self.vol_history.clear()
        self.volatility_regimes = np.array([0.1, 0.3, 0.5], np.float32)
        self._performance_history.clear()
        self._regime_history.clear()
        self._regime_transitions.clear()
        self._current_regime = 0
        self._predicted_regime_history.clear()
        self._true_regime_history.clear()
        self._regime_accuracy_scores.fill(0.0)
        
        # Reset regime tracking
        for i in range(self.n_regimes):
            self._regime_pnl_tracking[i].clear()
            self._regime_characteristics[i] = {
                'avg_volatility': 0.0,
                'avg_pnl': 0.0,
                'count': 0,
                'best_predictions': 0,
                'worst_predictions': 0,
                'stability_score': 0.5
            }

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        # Extract performance data
        performance_data = self._extract_performance_data(info_bus, kwargs)
        
        # Process regime performance with enhanced analytics
        self._process_regime_performance(performance_data)
        
        # Update regime characteristics
        self._update_regime_characteristics(performance_data)

    def _extract_performance_data(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract performance data from InfoBus or kwargs"""
        
        # Try InfoBus first
        if info_bus:
            # Extract regime and volatility data
            market_context = info_bus.get('market_context', {})
            risk_data = info_bus.get('risk', {})
            
            # Get regime information
            regime = InfoBusExtractor.get_market_regime(info_bus)
            volatility_level = InfoBusExtractor.get_volatility_level(info_bus)
            
            # Map regime to numeric
            regime_mapping = {'trending': 0, 'volatile': 1, 'ranging': 2, 'unknown': 0}
            predicted_regime = regime_mapping.get(regime, 0)
            
            # Extract volatility
            volatility = self._extract_volatility_from_info_bus(info_bus, market_context)
            
            # Extract PnL data
            recent_trades = info_bus.get('recent_trades', [])
            pnl = sum(trade.get('pnl', 0) for trade in recent_trades) if recent_trades else 0.0
            
            # Extract liquidity information
            liquidity_score = info_bus.get('market_status', {}).get('liquidity_score', 1.0)
            
            return {
                'pnl': pnl,
                'volatility': volatility,
                'predicted_regime': predicted_regime,
                'regime_name': regime,
                'volatility_level': volatility_level,
                'liquidity_score': liquidity_score,
                'market_context': market_context,
                'risk_data': risk_data,
                'recent_trades': recent_trades,
                'source': 'info_bus'
            }
        
        # Try kwargs (backward compatibility)
        required_keys = ["pnl", "volatility", "predicted_regime"]
        if all(k in kwargs for k in required_keys):
            return {
                'pnl': kwargs["pnl"],
                'volatility': kwargs["volatility"],
                'predicted_regime': kwargs["predicted_regime"],
                'liquidity_score': kwargs.get('liquidity_score', 1.0),
                'source': 'kwargs'
            }
        
        # Skip if insufficient data
        return {'source': 'insufficient_data'}

    def _extract_volatility_from_info_bus(self, info_bus: InfoBus, market_context: Dict) -> float:
        """Extract volatility from InfoBus data"""
        
        # Try market context volatility
        if 'volatility' in market_context:
            vol_data = market_context['volatility']
            if isinstance(vol_data, dict):
                vol_values = [float(v) for v in vol_data.values() if isinstance(v, (int, float))]
                return np.mean(vol_values) if vol_values else 0.01
            elif isinstance(vol_data, (int, float)):
                return float(vol_data)
        
        # Try volatility level conversion
        vol_level = InfoBusExtractor.get_volatility_level(info_bus)
        vol_mapping = {'low': 0.01, 'medium': 0.02, 'high': 0.04, 'extreme': 0.08}
        return vol_mapping.get(vol_level, 0.02)

    def _process_regime_performance(self, performance_data: Dict[str, Any]):
        """Process regime performance with enhanced analytics"""
        
        if performance_data.get('source') == 'insufficient_data':
            return
        
        try:
            pnl = performance_data['pnl']
            volatility = performance_data['volatility']
            predicted_regime = performance_data['predicted_regime']
            
            # Update volatility history and regimes
            self.vol_history.append(volatility)
            self._update_volatility_regimes()
            
            # Determine true regime based on current volatility
            true_regime = self._determine_true_regime(volatility)
            
            # Update performance matrix with decay
            self.matrix *= self.decay
            self.matrix[true_regime, predicted_regime] += pnl * self.regime_sensitivity
            
            # Track regime histories
            self._predicted_regime_history.append(predicted_regime)
            self._true_regime_history.append(true_regime)
            self._regime_history.append({'true': true_regime, 'predicted': predicted_regime, 'pnl': pnl})
            
            # Update current regime
            if true_regime != self._current_regime:
                self._handle_regime_transition(self._current_regime, true_regime, volatility)
                self._current_regime = true_regime
            
            # Track PnL by regime
            self._regime_pnl_tracking[true_regime].append(pnl)
            
            # Update performance history
            performance_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'true_regime': true_regime,
                'predicted_regime': predicted_regime,
                'pnl': pnl,
                'volatility': volatility,
                'accuracy': int(true_regime == predicted_regime)
            }
            self._performance_history.append(performance_record)
            
            # Update state
            self.last_volatility = volatility
            self.last_liquidity = performance_data.get('liquidity_score', 1.0)
            
            # Calculate and update accuracy scores
            self._update_accuracy_scores()
            
            # Log significant events
            if abs(pnl) > 0.1:  # Significant P&L
                self.log_operator_info(
                    f"Significant regime performance recorded",
                    true_regime=true_regime,
                    predicted_regime=predicted_regime,
                    pnl=f"{pnl:+.3f}",
                    volatility=f"{volatility:.5f}",
                    accuracy="✅" if true_regime == predicted_regime else "❌"
                )
            
            # Update performance metrics
            self._update_performance_metric('matrix_sum', float(self.matrix.sum()))
            self._update_performance_metric('current_regime', true_regime)
            self._update_performance_metric('regime_accuracy', self._calculate_overall_accuracy())
            
        except Exception as e:
            self.log_operator_error(f"Regime performance processing failed: {e}")
            self._update_health_status("DEGRADED", f"Processing failed: {e}")

    def _update_volatility_regimes(self):
        """Update volatility regime thresholds dynamically"""
        
        if len(self.vol_history) >= 20:
            # Update regime thresholds based on recent volatility distribution
            try:
                vol_array = np.array(list(self.vol_history))
                new_regimes = np.quantile(vol_array, [0.25, 0.5, 0.75]).astype(np.float32)
                
                # Check for significant change
                if np.max(np.abs(new_regimes - self.volatility_regimes)) > 0.01:
                    old_regimes = self.volatility_regimes.copy()
                    self.volatility_regimes = new_regimes
                    
                    self.log_operator_info(
                        f"Volatility regime thresholds updated",
                        old_thresholds=f"[{old_regimes[0]:.4f}, {old_regimes[1]:.4f}, {old_regimes[2]:.4f}]",
                        new_thresholds=f"[{new_regimes[0]:.4f}, {new_regimes[1]:.4f}, {new_regimes[2]:.4f}]"
                    )
                    
            except Exception as e:
                self.log_operator_warning(f"Failed to update volatility regimes: {e}")

    def _determine_true_regime(self, volatility: float) -> int:
        """Determine true regime based on volatility"""
        regime = min(int(np.digitize(volatility, self.volatility_regimes)), self.n_regimes - 1)
        return max(0, regime)  # Ensure non-negative

    def _handle_regime_transition(self, old_regime: int, new_regime: int, volatility: float):
        """Handle regime transitions with analytics"""
        
        transition_key = f"{old_regime}->{new_regime}"
        
        if transition_key not in self._regime_transitions:
            self._regime_transitions[transition_key] = {
                'count': 0,
                'avg_volatility': 0.0,
                'total_volatility': 0.0
            }
        
        # Update transition statistics
        trans = self._regime_transitions[transition_key]
        trans['count'] += 1
        trans['total_volatility'] += volatility
        trans['avg_volatility'] = trans['total_volatility'] / trans['count']
        
        # Log significant transitions
        self.log_operator_info(
            f"Market regime transition detected",
            from_regime=old_regime,
            to_regime=new_regime,
            volatility=f"{volatility:.5f}",
            transition_count=trans['count']
        )

    def _update_regime_characteristics(self, performance_data: Dict[str, Any]):
        """Update regime characteristics and analytics"""
        
        if performance_data.get('source') == 'insufficient_data':
            return
            
        true_regime = self._determine_true_regime(performance_data['volatility'])
        predicted_regime = performance_data['predicted_regime']
        pnl = performance_data['pnl']
        volatility = performance_data['volatility']
        
        # Update regime characteristics
        char = self._regime_characteristics[true_regime]
        char['count'] += 1
        
        # Update averages
        n = char['count']
        char['avg_volatility'] = ((char['avg_volatility'] * (n-1)) + volatility) / n
        char['avg_pnl'] = ((char['avg_pnl'] * (n-1)) + pnl) / n
        
        # Track prediction quality
        if true_regime == predicted_regime:
            if pnl > 0:
                char['best_predictions'] += 1
        else:
            if pnl < 0:
                char['worst_predictions'] += 1
        
        # Calculate stability score
        if len(self._regime_history) >= 10:
            recent_regimes = [r['true'] for r in list(self._regime_history)[-10:]]
            regime_consistency = recent_regimes.count(true_regime) / 10.0
            char['stability_score'] = regime_consistency

    def _update_accuracy_scores(self):
        """Update regime prediction accuracy scores"""
        
        if len(self._predicted_regime_history) < 10:
            return
            
        # Calculate accuracy for each regime
        for regime in range(self.n_regimes):
            regime_predictions = []
            regime_actuals = []
            
            for pred, actual in zip(list(self._predicted_regime_history)[-50:], 
                                   list(self._true_regime_history)[-50:]):
                if actual == regime:
                    regime_predictions.append(pred)
                    regime_actuals.append(actual)
            
            if len(regime_actuals) > 0:
                accuracy = sum(1 for p, a in zip(regime_predictions, regime_actuals) if p == a) / len(regime_actuals)
                self._regime_accuracy_scores[regime] = accuracy

    def _calculate_overall_accuracy(self) -> float:
        """Calculate overall regime prediction accuracy"""
        
        if len(self._predicted_regime_history) < 5:
            return 0.5
            
        # Compare recent predictions vs actuals
        recent_pred = list(self._predicted_regime_history)[-20:]
        recent_actual = list(self._true_regime_history)[-20:]
        
        if len(recent_pred) != len(recent_actual):
            return 0.5
            
        correct = sum(1 for p, a in zip(recent_pred, recent_actual) if p == a)
        return correct / len(recent_pred)

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STRESS TESTING
    # ═══════════════════════════════════════════════════════════════════

    def stress_test(self, scenario: str, volatility: Optional[float] = None, 
                   liquidity_score: Optional[float] = None) -> Dict[str, float]:
        """Enhanced stress testing with comprehensive scenarios"""
        
        if scenario not in self._stress_scenarios:
            available = list(self._stress_scenarios.keys())
            self.log_operator_warning(f"Unknown stress scenario: {scenario}. Available: {available}")
            scenario = "flash-crash"  # Default fallback
        
        crisis = self._stress_scenarios[scenario]
        
        # Use provided values or current state
        base_vol = volatility if volatility is not None else self.last_volatility
        base_liq = liquidity_score if liquidity_score is not None else self.last_liquidity
        
        # Apply stress multipliers
        stressed_vol = base_vol * crisis["vol_mult"]
        stressed_liq = base_liq * crisis["liq_mult"]
        
        # Determine stressed regime
        stressed_regime = self._determine_true_regime(stressed_vol)
        
        # Estimate performance impact based on matrix
        if self.matrix.sum() > 0:
            # Get average performance for this regime
            regime_performance = self.matrix[stressed_regime, :].mean()
            performance_impact = regime_performance * crisis.get("duration", 1)
        else:
            performance_impact = 0.0
        
        # Calculate risk metrics
        liquidity_impact = (1.0 - stressed_liq) * 100  # Percentage impact
        volatility_impact = (stressed_vol / max(base_vol, 1e-8) - 1.0) * 100
        
        stress_result = {
            "scenario": scenario,
            "volatility": float(stressed_vol),
            "liquidity": float(stressed_liq),
            "regime": int(stressed_regime),
            "performance_impact": float(performance_impact),
            "liquidity_impact_pct": float(liquidity_impact),
            "volatility_impact_pct": float(volatility_impact),
            "duration": crisis["duration"],
            "severity_score": float((volatility_impact + liquidity_impact) / 2)
        }
        
        # Log stress test
        self.log_operator_info(
            f"Stress test completed: {scenario}",
            regime=stressed_regime,
            vol_impact=f"{volatility_impact:+.1f}%",
            liq_impact=f"{liquidity_impact:+.1f}%",
            performance=f"{performance_impact:+.3f}",
            severity=f"{stress_result['severity_score']:.1f}"
        )
        
        # Add to risk alerts if severe
        if stress_result['severity_score'] > 50:
            self._risk_alerts.append({
                'type': 'stress_test_alert',
                'scenario': scenario,
                'severity': stress_result['severity_score'],
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        return stress_result

    def run_comprehensive_stress_test(self) -> Dict[str, Any]:
        """Run all stress scenarios and return comprehensive results"""
        
        results = {}
        
        for scenario in self._stress_scenarios:
            results[scenario] = self.stress_test(scenario)
        
        # Calculate overall stress resilience
        severity_scores = [r['severity_score'] for r in results.values()]
        avg_severity = np.mean(severity_scores)
        max_severity = np.max(severity_scores)
        
        # Determine resilience rating
        if max_severity < 30:
            resilience = "High"
        elif max_severity < 60:
            resilience = "Moderate"
        else:
            resilience = "Low"
        
        summary = {
            'overall_resilience': resilience,
            'avg_severity_score': float(avg_severity),
            'max_severity_score': float(max_severity),
            'worst_scenario': max(results.keys(), key=lambda k: results[k]['severity_score']),
            'test_results': results,
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        self.log_operator_info(
            f"Comprehensive stress test completed",
            resilience=resilience,
            avg_severity=f"{avg_severity:.1f}",
            worst_scenario=summary['worst_scenario']
        )
        
        return summary

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED OBSERVATION AND ACTION METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_observation_components(self) -> np.ndarray:
        """Enhanced observation components"""
        
        # Base matrix flattened
        flat_matrix = self.matrix.flatten()
        
        # Accuracy scores for each regime
        accuracy_scores = self._regime_accuracy_scores.copy()
        
        # Current regime information
        regime_info = np.array([
            float(self._current_regime),
            self.last_volatility,
            self.last_liquidity,
            self._calculate_overall_accuracy()
        ], dtype=np.float32)
        
        # Regime characteristics summary
        regime_summary = []
        for i in range(self.n_regimes):
            char = self._regime_characteristics[i]
            regime_summary.extend([
                char['avg_volatility'],
                char['avg_pnl'],
                char['stability_score']
            ])
        
        # Combine all components
        enhanced_obs = np.concatenate([
            flat_matrix,
            accuracy_scores,
            regime_info,
            np.array(regime_summary, dtype=np.float32)
        ])
        
        return enhanced_obs.astype(np.float32)

    def propose_action(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> np.ndarray:
        """Propose regime-aware action adjustments"""
        
        # Default action dimension
        action_dim = getattr(obs, 'shape', [2])[0] if hasattr(obs, 'shape') else 2
        action = np.zeros(action_dim, dtype=np.float32)
        
        if self.matrix.sum() == 0:
            return action  # No data yet
        
        # Get current regime performance
        current_performance = self.matrix[self._current_regime, :].mean()
        overall_performance = self.matrix.mean()
        
        # Calculate regime adjustment factor
        if overall_performance != 0:
            regime_factor = current_performance / overall_performance
        else:
            regime_factor = 1.0
        
        # Apply accuracy weighting
        current_accuracy = self._regime_accuracy_scores[self._current_regime]
        accuracy_weight = 0.5 + (current_accuracy * 0.5)  # Range: 0.5 to 1.0
        
        # Final adjustment
        final_adjustment = regime_factor * accuracy_weight
        final_adjustment = float(np.clip(final_adjustment, 0.1, 2.0))
        
        # Apply to all action dimensions
        action.fill(final_adjustment)
        
        return action

    def confidence(self, obs: Any = None, info_bus: Optional[InfoBus] = None) -> float:
        """Return confidence based on regime prediction accuracy and data quality"""
        
        # Base confidence on overall accuracy
        base_confidence = self._calculate_overall_accuracy()
        
        # Boost confidence with more data
        data_bonus = min(0.2, len(self._performance_history) / 500)
        
        # Current regime stability bonus
        stability_bonus = self._regime_characteristics[self._current_regime]['stability_score'] * 0.1
        
        # Matrix convergence bonus
        matrix_sum = self.matrix.sum()
        convergence_bonus = min(0.1, abs(matrix_sum) / 100) if matrix_sum != 0 else 0
        
        total_confidence = base_confidence + data_bonus + stability_bonus + convergence_bonus
        
        return float(np.clip(total_confidence, 0.1, 1.0))

    # ═══════════════════════════════════════════════════════════════════
    # EVOLUTIONARY METHODS
    # ═══════════════════════════════════════════════════════════════════

    def get_genome(self) -> Dict[str, Any]:
        """Get evolutionary genome"""
        return self.genome.copy()
        
    def set_genome(self, genome: Dict[str, Any]):
        """Set evolutionary genome with matrix rebuilding if needed"""
        old_regimes = self.n_regimes
        
        self.n_regimes = int(np.clip(genome.get("n_regimes", self.n_regimes), 2, 8))
        self.decay = float(np.clip(genome.get("decay", self.decay), 0.8, 0.99))
        self.vol_history_size = int(np.clip(genome.get("vol_history_size", self.vol_history_size), 100, 1000))
        self.performance_window = int(np.clip(genome.get("performance_window", self.performance_window), 50, 200))
        self.regime_sensitivity = float(np.clip(genome.get("regime_sensitivity", self.regime_sensitivity), 0.5, 2.0))
        
        self.genome = {
            "n_regimes": self.n_regimes,
            "decay": self.decay,
            "vol_history_size": self.vol_history_size,
            "performance_window": self.performance_window,
            "regime_sensitivity": self.regime_sensitivity
        }
        
        # Rebuild matrix if size changed
        if old_regimes != self.n_regimes:
            try:
                old_matrix = self.matrix.copy()
                self.matrix = np.zeros((self.n_regimes, self.n_regimes), np.float32)
                
                # Copy over compatible data
                min_size = min(old_regimes, self.n_regimes)
                self.matrix[:min_size, :min_size] = old_matrix[:min_size, :min_size]
                
                # Rebuild accuracy scores
                self._regime_accuracy_scores = np.zeros(self.n_regimes, np.float32)
                
                # Rebuild characteristics
                new_characteristics = {}
                for i in range(self.n_regimes):
                    if i in self._regime_characteristics:
                        new_characteristics[i] = self._regime_characteristics[i]
                    else:
                        new_characteristics[i] = {
                            'avg_volatility': 0.0,
                            'avg_pnl': 0.0,
                            'count': 0,
                            'best_predictions': 0,
                            'worst_predictions': 0,
                            'stability_score': 0.5
                        }
                self._regime_characteristics = new_characteristics
                
                # Rebuild PnL tracking
                new_pnl_tracking = {}
                for i in range(self.n_regimes):
                    if i in self._regime_pnl_tracking:
                        new_pnl_tracking[i] = self._regime_pnl_tracking[i]
                    else:
                        new_pnl_tracking[i] = deque(maxlen=50)
                self._regime_pnl_tracking = new_pnl_tracking
                
                self.log_operator_info(f"Matrix rebuilt for {self.n_regimes} regimes")
                
            except Exception as e:
                self.log_operator_error(f"Matrix rebuild failed: {e}")

    def mutate(self, mutation_rate: float = 0.2):
        """Enhanced mutation with performance tracking"""
        g = self.genome.copy()
        mutations = []
        
        if np.random.rand() < mutation_rate:
            old_val = g["n_regimes"]
            g["n_regimes"] = int(np.clip(self.n_regimes + np.random.choice([-1, 1]), 2, 8))
            mutations.append(f"n_regimes: {old_val} → {g['n_regimes']}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["decay"]
            g["decay"] = float(np.clip(self.decay + np.random.uniform(-0.02, 0.02), 0.8, 0.99))
            mutations.append(f"decay: {old_val:.3f} → {g['decay']:.3f}")
            
        if np.random.rand() < mutation_rate:
            old_val = g["regime_sensitivity"]
            g["regime_sensitivity"] = float(np.clip(self.regime_sensitivity + np.random.uniform(-0.2, 0.2), 0.5, 2.0))
            mutations.append(f"sensitivity: {old_val:.3f} → {g['regime_sensitivity']:.3f}")
        
        if mutations:
            self.log_operator_info(f"Regime matrix mutation applied", changes=", ".join(mutations))
            
        self.set_genome(g)
        
    def crossover(self, other: "RegimePerformanceMatrix") -> "RegimePerformanceMatrix":
        """Enhanced crossover with compatibility checking"""
        if not isinstance(other, RegimePerformanceMatrix):
            self.log_operator_warning("Crossover with incompatible type")
            return self
            
        new_g = {k: np.random.choice([self.genome[k], other.genome[k]]) for k in self.genome}
        return RegimePerformanceMatrix(genome=new_g, debug=self.config.debug)

    # ═══════════════════════════════════════════════════════════════════
    # ENHANCED STATE MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════

    def _check_state_integrity(self) -> bool:
        """Enhanced health check"""
        try:
            # Check matrix validity
            if not np.all(np.isfinite(self.matrix)):
                return False
                
            # Check dimensions
            if self.matrix.shape != (self.n_regimes, self.n_regimes):
                return False
                
            # Check regime bounds
            if not (0 <= self._current_regime < self.n_regimes):
                return False
                
            # Check volatility regimes
            if not np.all(np.isfinite(self.volatility_regimes)):
                return False
                
            # Check characteristics integrity
            if len(self._regime_characteristics) != self.n_regimes:
                return False
                
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details"""
        base_details = super()._get_health_details()
        
        regime_details = {
            'matrix_info': {
                'current_regime': self._current_regime,
                'matrix_sum': float(self.matrix.sum()),
                'matrix_shape': self.matrix.shape,
                'overall_accuracy': self._calculate_overall_accuracy(),
                'regime_transitions': len(self._regime_transitions)
            },
            'volatility_info': {
                'current_volatility': self.last_volatility,
                'volatility_regimes': self.volatility_regimes.tolist(),
                'vol_history_size': len(self.vol_history)
            },
            'performance_info': {
                'performance_records': len(self._performance_history),
                'accuracy_scores': self._regime_accuracy_scores.tolist(),
                'regime_characteristics': {k: v for k, v in self._regime_characteristics.items() if v['count'] > 0}
            },
            'genome_config': self.genome.copy()
        }
        
        if base_details:
            base_details.update(regime_details)
            return base_details
        
        return regime_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        return {
            "matrix": self.matrix.tolist(),
            "vol_history": list(self.vol_history)[-100:],  # Keep recent only
            "volatility_regimes": self.volatility_regimes.tolist(),
            "last_volatility": float(self.last_volatility),
            "last_liquidity": float(self.last_liquidity),
            "genome": self.genome.copy(),
            "current_regime": self._current_regime,
            "regime_transitions": dict(self._regime_transitions),
            "performance_history": list(self._performance_history)[-50:],  # Keep recent only
            "regime_characteristics": self._regime_characteristics.copy(),
            "accuracy_scores": self._regime_accuracy_scores.tolist(),
            "predicted_regime_history": list(self._predicted_regime_history)[-50:],
            "true_regime_history": list(self._true_regime_history)[-50:],
            "stress_scenarios": self._stress_scenarios.copy()
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        self.matrix = np.array(module_state.get("matrix", np.zeros((self.n_regimes, self.n_regimes))), dtype=np.float32)
        self.vol_history = deque(module_state.get("vol_history", []), maxlen=self.vol_history_size)
        self.volatility_regimes = np.array(module_state.get("volatility_regimes", [0.1, 0.3, 0.5]), dtype=np.float32)
        self.last_volatility = float(module_state.get("last_volatility", 0.0))
        self.last_liquidity = float(module_state.get("last_liquidity", 1.0))
        self.set_genome(module_state.get("genome", self.genome))
        self._current_regime = module_state.get("current_regime", 0)
        self._regime_transitions = module_state.get("regime_transitions", {})
        self._performance_history = deque(module_state.get("performance_history", []), maxlen=self.performance_window)
        self._regime_characteristics = module_state.get("regime_characteristics", {})
        self._regime_accuracy_scores = np.array(module_state.get("accuracy_scores", [0.5]*self.n_regimes), dtype=np.float32)
        self._predicted_regime_history = deque(module_state.get("predicted_regime_history", []), maxlen=100)
        self._true_regime_history = deque(module_state.get("true_regime_history", []), maxlen=100)
        self._stress_scenarios = module_state.get("stress_scenarios", self._stress_scenarios)
        
        # Rebuild PnL tracking
        self._regime_pnl_tracking = {i: deque(maxlen=50) for i in range(self.n_regimes)}

    def get_regime_performance_report(self) -> str:
        """Generate operator-friendly regime performance report"""
        
        # Current regime info
        regime_names = ["Low-Vol", "Med-Vol", "High-Vol"]
        current_regime_name = regime_names[self._current_regime] if self._current_regime < len(regime_names) else f"Regime-{self._current_regime}"
        
        # Best performing regime
        if self.matrix.sum() != 0:
            regime_performance = [self.matrix[i, :].mean() for i in range(self.n_regimes)]
            best_regime = np.argmax(regime_performance)
            best_regime_name = regime_names[best_regime] if best_regime < len(regime_names) else f"Regime-{best_regime}"
            best_performance = regime_performance[best_regime]
        else:
            best_regime_name = "Unknown"
            best_performance = 0.0
        
        # Accuracy summary
        overall_accuracy = self._calculate_overall_accuracy()
        if overall_accuracy > 0.8:
            accuracy_desc = "🎯 Excellent"
        elif overall_accuracy > 0.6:
            accuracy_desc = "✅ Good"
        elif overall_accuracy > 0.4:
            accuracy_desc = "⚡ Moderate"
        else:
            accuracy_desc = "⚠️ Poor"
        
        return f"""
📊 REGIME PERFORMANCE MATRIX
═══════════════════════════════════════
🎯 Current Regime: {current_regime_name} (ID: {self._current_regime})
🏆 Best Performer: {best_regime_name} ({best_performance:+.3f})
📈 Prediction Accuracy: {accuracy_desc} ({overall_accuracy:.1%})

📋 MATRIX STATUS
• Matrix Sum: {self.matrix.sum():+.3f}
• Data Points: {len(self._performance_history)}
• Regime Transitions: {len(self._regime_transitions)}
• Decay Factor: {self.decay:.3f}

⚡ VOLATILITY REGIMES
• Low-Vol Threshold: {self.volatility_regimes[0]:.4f}
• Med-Vol Threshold: {self.volatility_regimes[1]:.4f}
• High-Vol Threshold: {self.volatility_regimes[2]:.4f}
• Current Volatility: {self.last_volatility:.4f}

🎪 REGIME CHARACTERISTICS
""" + "\n".join([
    f"• {regime_names[i] if i < len(regime_names) else f'Regime-{i}'}: "
    f"Cnt={char['count']}, "
    f"Avg-PnL={char['avg_pnl']:+.3f}, "
    f"Stability={char['stability_score']:.2f}"
    for i, char in self._regime_characteristics.items() if char['count'] > 0
]) + f"""

🚨 STRESS TEST SCENARIOS
• Available Tests: {len(self._stress_scenarios)}
• Risk Alerts: {len(self._risk_alerts)}
• Last Liquidity: {self.last_liquidity:.3f}
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