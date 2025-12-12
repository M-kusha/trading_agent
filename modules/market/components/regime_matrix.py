# ─────────────────────────────────────────────────────────────
# File: modules/market/components/regime_matrix.py
# Regime Performance Matrix Component (Production-Ready)
# ─────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import numpy as np
import datetime

from ..shared.base_component import BaseMarketComponent


class RegimeMatrixComponent(BaseMarketComponent):
    """
    Tracks predicted vs. realized regimes with two matrices:
      • pnl_matrix[i,j]: decayed average P&L when predicted=i, actual=j
      • count_matrix[i,j]: decayed counts for predictions vs actuals (confusion)
    Determines "true" regime from EWMA volatility with adaptive quantile thresholds
    and hysteresis. Computes robust accuracy + per-regime characteristics.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        default_config = {
            'n_regimes': 3,

            # Decay & windows
            'decay_factor': 0.97,               # for matrices (closer to 1 => longer memory)
            'vol_history_size': 1000,
            'performance_window': 250,
            'dwell_window': 200,

            # Volatility estimation
            'ewma_lambda': 0.94,                # EWMA smoothing for volatility
            'quantile_low': 0.33,               # adaptive boundaries
            'quantile_high': 0.66,
            'threshold_damp': 0.10,             # EMA for thresholds
            'hysteresis_margin': 0.10,          # margin as fraction of threshold gap

            # Accuracy reporting
            'accuracy_threshold': 0.60,

            # Instruments to look for
            'instruments': ('XAUUSD', 'EURUSD'),

            # Determinism
            'seed': 1337,

            # PnL sourcing (set to False to avoid randomness)
            'simulate_pnl_if_missing': False,
        }
        default_config.update(config or {})

        super().__init__(
            name="RegimeMatrix",
            config=default_config,
            **kwargs
        )

    # ------------------------------
    # Lifecycle
    # ------------------------------
    def initialize(self):
        np.random.seed(int(self.config['seed']))

        n = int(self.config['n_regimes'])
        self.n = n

        # Matrices
        self.pnl_matrix = np.zeros((n, n), dtype=np.float64)    # decayed avg pnl by [pred, actual]
        self.count_matrix = np.zeros((n, n), dtype=np.float64)  # decayed counts (confusion)

        # Volatility thresholds (two boundaries for 3 regimes)
        # Start with reasonable spaced guesses; will adapt
        self._thr_low = 0.10
        self._thr_high = 0.30

        # Current state
        self._current_regime = 0          # realized (true)
        self._predicted_regime = 0
        self.last_volatility = 0.0
        self.last_liquidity = 1.0

        # Histories
        self.vol_history = deque(maxlen=int(self.config['vol_history_size']))
        self._performance_history = deque(maxlen=int(self.config['performance_window']))
        self._regime_history = deque(maxlen=200)
        self._predicted_regime_history = deque(maxlen=200)
        self._true_regime_history = deque(maxlen=200)
        self._dwell_history = deque(maxlen=int(self.config['dwell_window']))

        # Per-regime tracking
        self._regime_pnl_tracking = {i: deque(maxlen=100) for i in range(n)}
        self._regime_transitions: Dict[str, Dict[str, float]] = {}

        # Regime characteristics (live-updated)
        self._regime_characteristics: Dict[int, Dict[str, Any]] = {
            i: {
                "avg_volatility": 0.0,
                "avg_pnl": 0.0,
                "count": 0,
                "accuracy": 0.5,
                "stability_score": 0.5,
                "sharpe_like": 0.0,
                "avg_dwell": 0.0,
            } for i in range(n)
        }

        self.trace("Regime matrix component initialized", level="DEBUG")

    # ------------------------------
    # Public API
    # ------------------------------
    async def analyze_impl(self, **inputs) -> Dict[str, Any]:
        self.trace("Starting regime matrix analysis", level="TRACE")

        try:
            performance_data = await self._extract_performance_data(inputs)
            if not performance_data:
                self.trace("No performance data available", level="WARNING")
                return self.get_fallback_result("No performance data")

            matrix_result = await self._process_regime_matrix(performance_data)

            self.trace(
                f"Regime analysis: current={matrix_result['current_regime']}, "
                f"predicted={matrix_result['predicted_regime']}, "
                f"accuracy={matrix_result['overall_accuracy']:.3f}",
                level="DEBUG"
            )
            return matrix_result

        except Exception as e:
            self.trace(f"Regime matrix analysis error: {e}", level="ERROR")
            return self.get_fallback_result(str(e))

    # ------------------------------
    # Extraction
    # ------------------------------
    async def _extract_performance_data(self, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self.trace("Extracting performance data", level="TRACE")

        market_data = inputs.get('market_data', {}) or {}
        shared_context = inputs.get('shared_context', {}) or {}

        # Predicted regime from shared context (fractal/liquidity etc.)
        predicted_regime = 0
        try:
            # Prefer explicit id if provided by a component
            if 'fractal' in shared_context:
                fr = shared_context['fractal']
                if isinstance(fr, dict):
                    if 'regime_data' in fr and isinstance(fr['regime_data'], dict):
                        predicted_regime = int(fr['regime_data'].get('id', 0))
                    elif 'market_regime' in fr:
                        predicted_regime = self._map_label_to_regime(fr['market_regime'])
            elif 'regime_data' in shared_context:
                predicted_regime = int(shared_context['regime_data'].get('id', 0))
        except Exception:
            predicted_regime = 0

        # Volatility (realized) with robust estimators
        volatility = await self._calculate_volatility(market_data)

        # PnL: prefer real, otherwise 0 (or simulated if enabled)
        pnl = self._extract_pnl(market_data)

        # Liquidity score (optional)
        liquidity_score = 1.0
        if 'liquidity' in shared_context and isinstance(shared_context['liquidity'], dict):
            liquidity_score = float(shared_context['liquidity'].get('liquidity_score', 1.0))

        return {
            'predicted_regime': int(np.clip(predicted_regime, 0, self.n - 1)),
            'volatility': float(volatility),
            'pnl': float(pnl),
            'liquidity_score': float(liquidity_score),
            'timestamp': datetime.datetime.utcnow()
        }

    # ------------------------------
    # Volatility (robust) - MULTI-TIMEFRAME ENHANCED
    # ------------------------------
    async def _calculate_volatility(self, market_data: Dict[str, Any]) -> float:
        """
        Use best-available estimator across MULTIPLE TIMEFRAMES:
          1) Calculate volatility for H1, H4, D1 timeframes
          2) Weight them: H1=0.40 (short-term), H4=0.35 (medium), D1=0.25 (long-term)
          3) If OHLC available: Garman–Klass (less noisy than CC)
          4) Else: log return volatility (close/close)
        Then EWMA-smooth and return latest EWMA sigma.
        
        Multi-timeframe volatility gives better regime classification:
        - H1 high + D1 low = short-term spike (may be temporary)
        - H1 high + D1 high = true volatile regime
        - H1 low + D1 low = calm regime
        """
        # Timeframe weights for volatility aggregation (M15 for micro-volatility detection)
        tf_weights = {'M15': 0.20, 'H1': 0.35, 'H4': 0.28, 'D1': 0.17}
        timeframes = ['M15', 'H1', 'H4', 'D1']
        
        # Gather volatility estimates per instrument per timeframe
        all_sigmas: List[float] = []
        tf_sigmas: Dict[str, List[float]] = {tf: [] for tf in timeframes}

        instruments = self.config['instruments'] or ()
        for inst in instruments:
            inst_data = market_data.get(inst)
            if not isinstance(inst_data, dict):
                continue

            # Try each timeframe
            for tf in timeframes:
                tf_data = inst_data.get(tf)
                
                # Also try without timeframe nesting (flat structure)
                if tf_data is None and tf == 'H1':
                    tf_data = inst_data  # Fallback: assume flat structure is H1
                
                if not isinstance(tf_data, dict):
                    continue
                
                sigma = self._calculate_single_tf_volatility(tf_data)
                if sigma is not None and sigma > 0:
                    tf_sigmas[tf].append(sigma)

        # If no timeframe-nested data found, try flat structure
        if all(len(s) == 0 for s in tf_sigmas.values()):
            for inst in instruments:
                d = market_data.get(inst)
                if not isinstance(d, dict):
                    continue
                sigma = self._calculate_single_tf_volatility(d)
                if sigma is not None and sigma > 0:
                    all_sigmas.append(sigma)
        
        # Aggregate with timeframe weighting
        if any(len(s) > 0 for s in tf_sigmas.values()):
            weighted_vol = 0.0
            total_weight = 0.0
            
            for tf, sigmas in tf_sigmas.items():
                if sigmas:
                    tf_vol = float(np.median(sigmas))
                    weight = tf_weights.get(tf, 0.33)
                    weighted_vol += tf_vol * weight
                    total_weight += weight
            
            sigma_now = weighted_vol / total_weight if total_weight > 0 else 0.01
        elif all_sigmas:
            sigma_now = float(np.median(all_sigmas))
        else:
            sigma_now = 0.01

        # EWMA smooth with lambda
        lam = float(self.config['ewma_lambda'])
        if self.vol_history:
            prev = float(self.vol_history[-1])
            ewma = lam * prev + (1.0 - lam) * sigma_now
        else:
            ewma = sigma_now

        self.vol_history.append(float(ewma))
        self.last_volatility = float(ewma)
        return float(ewma)
    
    def _calculate_single_tf_volatility(self, tf_data: Dict[str, Any]) -> Optional[float]:
        """Calculate volatility for a single timeframe's data."""
        # Try OHLC arrays (Garman-Klass)
        O, H, L, C = (tf_data.get('open'), tf_data.get('high'), tf_data.get('low'), tf_data.get('close'))
        if all(isinstance(arr, (list, tuple, np.ndarray)) for arr in (O, H, L, C)):
            try:
                Oa = np.asarray(O, dtype=np.float64)
                Ha = np.asarray(H, dtype=np.float64)
                La = np.asarray(L, dtype=np.float64)
                Ca = np.asarray(C, dtype=np.float64)
                m = min(Oa.size, Ha.size, La.size, Ca.size)
                if m >= 10:
                    Oa, Ha, La, Ca = Oa[-m:], Ha[-m:], La[-m:], Ca[-m:]
                    # Garman–Klass variance
                    log_hl = np.log(np.maximum(Ha, 1e-12)) - np.log(np.maximum(La, 1e-12))
                    log_co = np.log(np.maximum(Ca, 1e-12)) - np.log(np.maximum(Oa, 1e-12))
                    var = 0.5 * (log_hl ** 2) - (2.0 * np.log(2.0) - 1.0) * (log_co ** 2)
                    var = np.clip(var, 0.0, None)
                    sigma = float(np.sqrt(np.mean(var[-20:])))
                    if np.isfinite(sigma) and sigma > 0:
                        return sigma
            except Exception:
                pass

        # Fallback: close/close log-return std
        C = tf_data.get('close')
        if isinstance(C, (list, tuple, np.ndarray)) and len(C) >= 10:
            ca = np.asarray(C, dtype=np.float64)
            ca = ca[-min(ca.size, 200):]
            rets = np.diff(np.log(np.maximum(ca, 1e-12)))
            if rets.size >= 2:
                return float(np.std(rets[-min(rets.size, 50):]))
        
        return None

    # ------------------------------
    # PnL sourcing
    # ------------------------------
    def _extract_pnl(self, market_data: Dict[str, Any]) -> float:
        # Plug real PnL if provided
        pnl = 0.0
        try:
            if 'pnl' in market_data:
                p = market_data['pnl']
                if isinstance(p, (int, float)) and np.isfinite(p):
                    return float(p)
        except Exception:
            pass

        # Optional simulation if enabled (kept deterministic around zero)
        if bool(self.config.get('simulate_pnl_if_missing', False)):
            # Small zero-mean noise scaled by volatility
            scale = max(self.last_volatility, 1e-4) * 10000.0
            rs = np.random.RandomState(int(self.config['seed']))
            return float(rs.normal(0.0, scale))
        return 0.0

    # ------------------------------
    # Core processing
    # ------------------------------
    async def _process_regime_matrix(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        pred_regime = int(performance_data['predicted_regime'])
        vol = float(performance_data['volatility'])
        pnl = float(performance_data['pnl'])

        # Update thresholds and derive true regime with hysteresis
        self._update_thresholds_if_ready()
        true_regime = self._determine_true_regime_with_hysteresis(vol)

        # Histories
        self._predicted_regime_history.append(pred_regime)
        self._true_regime_history.append(true_regime)
        self._performance_history.append(pnl)
        self._regime_history.append(true_regime)

        # Update matrices with decay
        i = int(np.clip(pred_regime, 0, self.n - 1))
        j = int(np.clip(true_regime, 0, self.n - 1))
        decay = float(self.config['decay_factor'])

        # PnL matrix (decayed mean-like)
        self.pnl_matrix[i, j] = self.pnl_matrix[i, j] * decay + pnl * (1.0 - decay)

        # Confusion counts (decayed)
        self.count_matrix *= decay
        self.count_matrix[i, j] += 1.0

        # Regime transition handling
        if true_regime != getattr(self, "_current_regime", true_regime):
            await self._handle_regime_transition(self._current_regime, true_regime, vol, pnl)
            # dwell update
            self._dwell_history.append((datetime.datetime.utcnow(), true_regime))

        # State update
        self._current_regime = true_regime
        self._predicted_regime = pred_regime
        self.last_volatility = vol

        # Per-regime characteristics
        self._update_regime_characteristics(true_regime, vol, pnl)

        # Metrics
        overall_accuracy, by_regime = self._compute_accuracies()
        regime_accuracy_current = by_regime.get(str(true_regime), 0.5)
        avg_performance = float(np.mean(self._performance_history)) if self._performance_history else 0.0
        volatility_trend = self._calculate_volatility_trend()

        regime_accuracy_details = {
            'value': overall_accuracy,
            'by_regime': by_regime,
            'current_regime_accuracy': float(regime_accuracy_current),
            'last_update': datetime.datetime.utcnow().isoformat() + 'Z'
        }

        # Build output (matrix key kept for backward-compat points at PnL matrix)
        return {
            'current_regime': int(true_regime),
            'predicted_regime': int(pred_regime),
            'matrix': self.pnl_matrix.tolist(),
            'overall_accuracy': float(overall_accuracy),
            'regime_accuracy': regime_accuracy_details,
            'regime_prediction': {
                'predicted': int(pred_regime),
                'actual': int(true_regime),
                'correct': bool(pred_regime == true_regime)
            },
            'avg_performance': float(avg_performance),
            'current_volatility': float(vol),
            'volatility_trend': str(volatility_trend),
            'regime_characteristics': self._regime_characteristics,
            'regime_performance': {
                'matrix': self.pnl_matrix.tolist(),
                'current_regime': int(true_regime),
                'predicted_regime': int(pred_regime),
                'avg_performance': float(avg_performance),
                # Extra diagnostics (non-breaking)
                'count_matrix': self.count_matrix.tolist(),
                'thresholds': {'low': float(self._thr_low), 'high': float(self._thr_high)}
            },
            'processing_success': True
        }

    # ------------------------------
    # Regime inference
    # ------------------------------
    def _update_thresholds_if_ready(self):
        """Adapt volatility thresholds using rolling quantiles with damping."""
        if len(self.vol_history) < 50:
            return
        vols = np.array(self.vol_history, dtype=np.float64)
        ql = float(np.quantile(vols, float(self.config['quantile_low'])))
        qh = float(np.quantile(vols, float(self.config['quantile_high'])))
        # Ensure ordering and minimum gap
        if qh <= ql:
            qh = ql + 1e-4
        alpha = float(self.config['threshold_damp'])
        self._thr_low = (1 - alpha) * self._thr_low + alpha * ql
        self._thr_high = (1 - alpha) * self._thr_high + alpha * qh

    def _determine_true_regime_with_hysteresis(self, vol: float) -> int:
        """
        3 regimes from two thresholds with margins:
          0: low
          1: medium
          2: high
        Hysteresis reduces flip-flop around boundaries.
        """
        low, high = float(self._thr_low), float(self._thr_high)
        margin = float(self.config['hysteresis_margin']) * max(high - low, 1e-8)

        prev = getattr(self, "_current_regime", 1)

        if prev == 0:
            # rise only after low + margin
            if vol > low + margin:
                return 1 if vol < high - margin else 2
            return 0
        elif prev == 1:
            # go low if vol < low - margin; high if vol > high + margin; else stay
            if vol < low - margin:
                return 0
            if vol > high + margin:
                return 2
            return 1
        else:  # prev == 2
            if vol < high - margin:
                return 1 if vol > low + margin else 0
            return 2

    def _map_label_to_regime(self, label: str) -> int:
        mapping = {
            'trend': 0, 'trending': 0, 'noise': 1, 'range': 1, 'ranging': 1, 'volatile': 2
        }
        try:
            return int(mapping.get(str(label).lower(), 1))
        except Exception:
            return 1

    # ------------------------------
    # Transitions & characteristics
    # ------------------------------
    async def _handle_regime_transition(self, old_regime: int, new_regime: int, volatility: float, pnl: float):
        key = f"{int(old_regime)}->{int(new_regime)}"
        rec = self._regime_transitions.get(key)
        if rec is None:
            rec = {'count': 0, 'avg_pnl': 0.0, 'avg_volatility': 0.0}
            self._regime_transitions[key] = rec

        new_count = int(rec['count']) + 1
        rec['avg_pnl'] = float((rec['avg_pnl'] * (new_count - 1) + pnl) / new_count)
        rec['avg_volatility'] = float((rec['avg_volatility'] * (new_count - 1) + volatility) / new_count)
        rec['count'] = new_count

        self.trace(f"Regime transition: {old_regime} -> {new_regime}", level="INFO")

    def _update_regime_characteristics(self, regime: int, volatility: float, pnl: float):
        regime = int(regime)
        ch = self._regime_characteristics.get(regime)
        if ch is None:
            return

        cnt = int(ch.get('count', 0)) + 1
        ch['count'] = cnt
        ch['avg_volatility'] = float((ch.get('avg_volatility', 0.0) * (cnt - 1) + volatility) / cnt)
        ch['avg_pnl'] = float((ch.get('avg_pnl', 0.0) * (cnt - 1) + pnl) / cnt)
        self._regime_pnl_tracking[regime].append(float(pnl))

        # Sharpe-like (mean/std of pnl) for regime
        pnl_arr = np.array(self._regime_pnl_tracking[regime], dtype=np.float64)
        if pnl_arr.size >= 5 and float(np.std(pnl_arr)) > 1e-12:
            ch['sharpe_like'] = float(np.mean(pnl_arr) / (np.std(pnl_arr) + 1e-12))
        else:
            ch['sharpe_like'] = 0.0

        # Stability estimate from dwell times
        ch['avg_dwell'] = float(self._estimate_avg_dwell(regime))
        ch['stability_score'] = float(np.clip(ch['avg_dwell'] / max(1.0, len(self._regime_history)), 0.0, 1.0))

        # Accuracy by regime (updated in _compute_accuracies)
        # we leave ch['accuracy'] to be set there; keep default if not computed yet

    def _estimate_avg_dwell(self, regime: int) -> float:
        """Average number of consecutive samples in this regime over dwell window."""
        if not self._regime_history:
            return 0.0
        # Count runs for target regime in recent history
        runs = []
        cur_len = 0
        for r in self._regime_history:
            if r == regime:
                cur_len += 1
            else:
                if cur_len > 0:
                    runs.append(cur_len)
                cur_len = 0
        if cur_len > 0:
            runs.append(cur_len)
        return float(np.mean(runs)) if runs else 0.0

    # ------------------------------
    # Accuracy & trends
    # ------------------------------
    def _compute_accuracies(self) -> Tuple[float, Dict[str, float]]:
        """
        Overall accuracy from decayed confusion counts; per-regime accuracy = TP / actual.
        Adds small Laplace prior for stability.
        """
        C = np.array(self.count_matrix, dtype=np.float64)
        if C.sum() <= 0:
            return 0.5, {str(i): 0.5 for i in range(self.n)}

        # Laplace smoothing
        C = C + 1e-6

        # Overall accuracy
        correct = float(np.trace(C))
        total = float(C.sum())
        overall = float(np.clip(correct / max(total, 1e-12), 0.0, 1.0))

        # Per-regime (by actual class)
        by_regime: Dict[str, float] = {}
        for j in range(self.n):
            actual_j = float(C[:, j].sum())
            tp = float(C[j, j])
            acc_j = tp / max(actual_j, 1e-12)
            by_regime[str(j)] = float(np.clip(acc_j, 0.0, 1.0))
            # write into characteristics
            if j in self._regime_characteristics:
                self._regime_characteristics[j]['accuracy'] = by_regime[str(j)]

        return overall, by_regime

    def _calculate_volatility_trend(self) -> str:
        if len(self.vol_history) < 10:
            return "stable"
        y = np.asarray(list(self.vol_history)[-30:], dtype=np.float64)
        x = np.arange(y.size, dtype=np.float64)
        try:
            if float(np.std(y)) == 0.0:
                return "stable"
            slope = float(np.polyfit(x, y, 1)[0])
            # relative slope threshold: ~1% of median per step
            med = float(np.median(y))
            rel = slope / max(med, 1e-8)
            if rel > 0.01:
                return "increasing"
            elif rel < -0.01:
                return "decreasing"
            else:
                return "stable"
        except Exception:
            return "stable"

    # ------------------------------
    # Fallback
    # ------------------------------
    def get_fallback_result(self, error: str) -> Dict[str, Any]:
        n = int(self.config['n_regimes'])
        return {
            'current_regime': getattr(self, "_current_regime", 0),
            'predicted_regime': getattr(self, "_predicted_regime", 0),
            'matrix': getattr(self, "pnl_matrix", np.zeros((n, n))).tolist(),
            'overall_accuracy': 0.5,
            'regime_accuracy': {
                'value': 0.5,
                'by_regime': {str(i): 0.5 for i in range(n)},
                'current_regime_accuracy': 0.5,
                'last_update': datetime.datetime.utcnow().isoformat() + 'Z'
            },
            'regime_prediction': {
                'predicted': getattr(self, "_predicted_regime", 0),
                'actual': getattr(self, "_current_regime", 0),
                'correct': False
            },
            'avg_performance': float(np.mean(self._performance_history)) if getattr(self, "_performance_history", None) else 0.0,
            'current_volatility': float(getattr(self, "last_volatility", 0.01)),
            'volatility_trend': 'unknown',
            'regime_characteristics': getattr(self, "_regime_characteristics", {i: {} for i in range(n)}),
            'regime_performance': {
                'matrix': getattr(self, "pnl_matrix", np.zeros((n, n))).tolist(),
                'current_regime': getattr(self, "_current_regime", 0),
                'predicted_regime': getattr(self, "_predicted_regime", 0),
                'avg_performance': 0.0,
                'count_matrix': getattr(self, "count_matrix", np.zeros((n, n))).tolist(),
                'thresholds': {'low': float(getattr(self, "_thr_low", 0.1)), 'high': float(getattr(self, "_thr_high", 0.3))}
            },
            'processing_success': False,
            'error': error
        }
