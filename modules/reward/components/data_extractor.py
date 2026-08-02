# modules/reward/components/data_extractor.py
"""
Data Extraction Component for Reward System (Hardened & Typed)
- Strong typing to satisfy Pylance/mypy
- Thread-safe counters & tracking (RLock)
- Defensive SmartInfoBus access (never raises)
- Category-specific validation & fallbacks
- Stable, contract-safe output with quality scoring
"""

from __future__ import annotations

import threading
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Optional

import numpy as np

_Report = Dict[str, Any]


class RewardDataExtractor:
    """
    Extracts and validates data from SmartInfoBus

    Features:
    - Comprehensive bus key extraction
    - Data validation and quality assessment
    - Missing key detection with frequency tracking
    - Robust fallbacks and environment mirrors
    - Clear debugging output (never throws)
    """

    # Known keys by category (used for reporting and quality scoring)
    _TRADING_KEYS = ('trade_data', 'trades', 'recent_trades')
    _RISK_KEYS = ('risk_metrics', 'account_state')
    _MARKET_KEYS = ('market_context', 'market_state', 'market_regime', 'regime_prediction')
    _PERF_KEYS = ('performance_data', 'environment_config')
    _MEMORY_KEYS = ('mistake_memory',)

    def __init__(
        self,
        smart_bus: Any,
        logger: Any,
        debug_manager: Any,
        env: Any = None
    ):
        """Initialize data extractor"""

        self.smart_bus = smart_bus
        self.logger = logger
        self.debug_manager = debug_manager
        self.env = env

        # Tracking (thread-safe)
        self._lock: threading.RLock = threading.RLock()
        self.extraction_count: int = 0
        self.missing_key_frequency: DefaultDict[str, int] = defaultdict(int)

    # ─────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────
    async def extract_reward_data(self, **inputs) -> Dict[str, Any]:
        """
        Extract all required data from bus with validation and fallbacks.

        Returns a complete data package with a 'data_quality' label and diagnostics.
        """

        with self._lock:
            self.extraction_count += 1
            step_idx = inputs.get('step_idx', self.extraction_count)

        # Track extraction process
        report: _Report = {
            'successful_keys': [],
            'missing_keys': [],
            'broken_keys': [],
            'validation_errors': {},
            'fallbacks_used': [],
        }

        # Category extracts (each tolerates bad types and returns {})
        trade_data = await self._extract_trading_data(report)
        risk_data = await self._extract_risk_data(report)
        market_data = await self._extract_market_data(report)
        performance_data = await self._extract_performance_data(report)
        memory_data = await self._extract_memory_data(report)

        # Resolutions
        trades = self._resolve_trades(trade_data)
        regime = self._resolve_regime(market_data)
        volatility_level = self._resolve_volatility(market_data, risk_data)
        consensus = self._resolve_consensus(market_data)
        balance_info = self._resolve_balance_info(risk_data, performance_data, market_data)

        # Quality
        data_quality = self._assess_data_quality(report)

        # Build output (stable schema)
        reward_data: Dict[str, Any] = {
            # Core data
            'trades': trades,
            'risk_metrics': risk_data.get('risk_metrics', {}) or {},
            'market_context': market_data.get('market_context', {}) or {},
            'performance_data': performance_data.get('performance_data', {}) or {},
            'env_config': performance_data.get('environment_config', {}) or {},
            'market_state': market_data.get('market_state', {}) or {},
            # Resolved values
            'regime': regime,
            'market_regime': regime,
            'volatility_level': volatility_level,
            'consensus': consensus,
            # Balance info
            'balance_now': balance_info['balance_now'],
            'baseline_balance': balance_info['baseline_balance'],
            # Metadata
            'timestamp': datetime.now().isoformat(),
            'step_idx': step_idx,
            'actions': inputs.get('actions'),
            'raw_inputs': inputs.get('reward_inputs', {}) or {},
            # Quality & diagnostics
            'data_quality': data_quality,
            'missing_keys': report['missing_keys'],
            'broken_keys': report['broken_keys'],
            'validation_errors': report['validation_errors'],
            'fallbacks_used': report['fallbacks_used'],
        }

        # Optional debug trace
        self._dbg_note(
            f"Extracted reward_data: regime={regime}, vol={volatility_level}, "
            f"trades={len(trades)}, quality={data_quality}"
        )

        return reward_data

    # ─────────────────────────────────────────────────────────────
    # Category Extractors (robust & typed)
    # ─────────────────────────────────────────────────────────────
    async def _extract_trading_data(self, report: _Report) -> Dict[str, Any]:
        """Extract trading-related data with fallbacks and validation."""
        data: Dict[str, Any] = {}

        trade_data = self._safe_bus_get('trade_data', report)
        if isinstance(trade_data, dict):
            data['trade_data'] = trade_data
            report['successful_keys'].append('trade_data')

        # Fallback to trades array (PositionManager style)
        if not isinstance(trade_data, dict) or not self._validate_trades(trade_data.get('recent_trades', [])):
            trades = self._safe_bus_get('trades', report)
            if isinstance(trades, (list, tuple)) and self._validate_trades(trades):
                data['trades'] = list(trades)
                report['successful_keys'].append('trades')
                report['fallbacks_used'].append('trades_from_position_manager')

        # Try recent_trades as a top-level bus key
        recent_trades = self._safe_bus_get('recent_trades', report)
        if isinstance(recent_trades, (list, tuple)) and self._validate_trades(recent_trades):
            data['recent_trades'] = list(recent_trades)
            report['successful_keys'].append('recent_trades')

        return data

    async def _extract_risk_data(self, report: _Report) -> Dict[str, Any]:
        """Extract risk-related data and provide a compatible fallback."""
        data: Dict[str, Any] = {}

        risk_metrics = self._safe_bus_get('risk_metrics', report)
        if isinstance(risk_metrics, dict) and self._validate_risk_metrics(risk_metrics):
            data['risk_metrics'] = risk_metrics
            report['successful_keys'].append('risk_metrics')

        account_state = self._safe_bus_get('account_state', report)
        if isinstance(account_state, dict):
            data['account_state'] = account_state
            report['successful_keys'].append('account_state')

            # Convert if risk_metrics absent or invalid
            if 'risk_metrics' not in data:
                converted = self._convert_account_to_risk(account_state)
                data['risk_metrics'] = converted
                report['fallbacks_used'].append('risk_from_account_state')

        return data

    async def _extract_market_data(self, report: _Report) -> Dict[str, Any]:
        """Extract market-related data safely."""
        data: Dict[str, Any] = {}

        market_context = self._safe_bus_get('market_context', report)
        if isinstance(market_context, dict):
            data['market_context'] = market_context
            report['successful_keys'].append('market_context')

        market_state = self._safe_bus_get('market_state', report)
        if isinstance(market_state, dict):
            data['market_state'] = market_state
            report['successful_keys'].append('market_state')

        market_regime = self._safe_bus_get('market_regime', report)
        if market_regime is not None:  # str or dict allowed
            data['market_regime'] = market_regime
            report['successful_keys'].append('market_regime')

        regime_prediction = self._safe_bus_get('regime_prediction', report)
        if isinstance(regime_prediction, dict):
            data['regime_prediction'] = regime_prediction
            report['successful_keys'].append('regime_prediction')

        return data

    async def _extract_performance_data(self, report: _Report) -> Dict[str, Any]:
        """Extract performance-related data safely."""
        data: Dict[str, Any] = {}

        performance_data = self._safe_bus_get('performance_data', report)
        if isinstance(performance_data, dict):
            data['performance_data'] = performance_data
            report['successful_keys'].append('performance_data')

        env_config = self._safe_bus_get('environment_config', report)
        if isinstance(env_config, dict):
            data['environment_config'] = env_config
            report['successful_keys'].append('environment_config')

        return data

    async def _extract_memory_data(self, report: _Report) -> Dict[str, Any]:
        """Extract optional memory-related data safely."""
        data: Dict[str, Any] = {}

        mistake_memory = self._safe_bus_get('mistake_memory', report)
        if isinstance(mistake_memory, dict):
            data['mistake_memory'] = mistake_memory
            report['successful_keys'].append('mistake_memory')

        return data

    # ─────────────────────────────────────────────────────────────
    # Resolution Methods (pure, side-effect free)
    # ─────────────────────────────────────────────────────────────
    def _resolve_trades(self, trade_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Resolve trades from various sources with clear precedence."""
        # Priority: trade_data.recent_trades → trades → recent_trades
        td = trade_data.get('trade_data', {})
        if isinstance(td, dict):
            rt = td.get('recent_trades', [])
            if self._validate_trades(rt):
                return list(rt)

        trades = trade_data.get('trades', [])
        if self._validate_trades(trades):
            return list(trades)

        rt2 = trade_data.get('recent_trades', [])
        if self._validate_trades(rt2):
            return list(rt2)

        return []

    def _resolve_regime(self, market_data: Dict[str, Any]) -> str:
        """
        Resolve market regime with robust precedence:
          1) market_regime (str or dict: 'regime'/'label'/'state')
          2) market_state.regime
          3) regime_prediction.predicted/label/regime
          4) market_context.regime
        """
        market_regime = market_data.get('market_regime')
        if isinstance(market_regime, str):
            val = market_regime.strip().lower()
            if val:
                return val
        elif isinstance(market_regime, dict):
            for key in ('regime', 'label', 'state'):
                val = market_regime.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip().lower()

        market_state = market_data.get('market_state', {}) or {}
        if isinstance(market_state, dict):
            val = market_state.get('regime')
            if isinstance(val, str) and val.strip():
                return val.strip().lower()

        regime_pred = market_data.get('regime_prediction', {}) or {}
        if isinstance(regime_pred, dict):
            for key in ('predicted', 'label', 'regime'):
                val = regime_pred.get(key)
                if isinstance(val, str) and val.strip():
                    return val.strip().lower()

        market_context = market_data.get('market_context', {}) or {}
        if isinstance(market_context, dict):
            val = market_context.get('regime')
            if isinstance(val, str) and val.strip():
                return val.strip().lower()

        return 'unknown'

    def _resolve_volatility(self, market_data: Dict[str, Any], risk_data: Dict[str, Any]) -> str:
        """Resolve volatility level from multiple sources with sane fallbacks."""
        market_context = market_data.get('market_context', {}) or {}
        if isinstance(market_context, dict):
            val = market_context.get('volatility_level')
            if isinstance(val, str) and val:
                return val

        market_state = market_data.get('market_state', {}) or {}
        if isinstance(market_state, dict):
            for key in ('volatility_level', 'volatility'):
                val = market_state.get(key)
                if isinstance(val, str) and val:
                    return val

        risk_metrics = risk_data.get('risk_metrics', {}) or {}
        if isinstance(risk_metrics, dict):
            val = risk_metrics.get('volatility_level')
            if isinstance(val, str) and val:
                return val

        if isinstance(market_context, dict):
            val = market_context.get('volatility_hint')
            if isinstance(val, str) and val:
                return val

        return 'medium'

    def _resolve_consensus(self, market_data: Dict[str, Any]) -> float:
        """Resolve consensus value from voting system.
        
        CRITICAL: Default of 0.5 means NO learning signal from voting alignment.
        We try multiple sources to find actual consensus data.
        """
        # Priority 1: kernel_consensus_score (canonical VotingKernel output)
        kernel_consensus = self._safe_bus_get_silent('kernel_consensus_score')
        if kernel_consensus is not None and self._is_valid_number(kernel_consensus):
            return float(np.clip(float(kernel_consensus), 0.0, 1.0))
        
        # Priority 2: Direct consensus_score 
        consensus_score = self._safe_bus_get_silent('consensus_score')
        if consensus_score is not None and self._is_valid_number(consensus_score):
            return float(np.clip(float(consensus_score), 0.0, 1.0))
        
        # Priority 3: voting_result.consensus_score
        voting_result = self._safe_bus_get_silent('voting_result')
        if isinstance(voting_result, dict):
            cs = voting_result.get('consensus_score')
            if cs is not None and self._is_valid_number(cs):
                return float(np.clip(float(cs), 0.0, 1.0))
        
        # Priority 4: arbiter_output.consensus
        arbiter_output = self._safe_bus_get_silent('arbiter_output')
        if isinstance(arbiter_output, dict):
            cs = arbiter_output.get('consensus')
            if cs is not None and self._is_valid_number(cs):
                return float(np.clip(float(cs), 0.0, 1.0))
        
        # Priority 5: market_context.consensus (legacy fallback)
        market_context = market_data.get('market_context', {}) or {}
        if isinstance(market_context, dict):
            val = market_context.get('consensus')
            if val is not None and self._is_valid_number(val):
                return float(np.clip(float(val), 0.0, 1.0))
        
        # No consensus found - log warning (important for training quality)
        self._dbg_note("WARNING: No consensus signal found - reward won't learn voting alignment")
        return 0.5
    
    def _safe_bus_get_silent(self, key: str) -> Optional[Any]:
        """Get from bus without tracking in report (for secondary lookups)."""
        try:
            return self.smart_bus.get(key, "RiskAdjustedReward")
        except Exception:
            return None

    def _resolve_balance_info(
        self,
        risk_data: Dict[str, Any],
        performance_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Resolve balance information from multiple sources.
        Never fabricates numbers beyond observed/declared values.
        """

        candidates: List[float] = []

        # 1) Risk metrics (authoritative at runtime)
        rm = risk_data.get('risk_metrics', {}) or {}
        if isinstance(rm, dict):
            for key in ('balance', 'equity', 'account_equity', 'cash', 'account_balance'):
                if key in rm and self._is_valid_number(rm[key]):
                    candidates.append(float(rm[key]))

        # 2) Account state (secondary)
        account = risk_data.get('account_state', {}) or {}
        if isinstance(account, dict):
            for key in ('balance', 'equity', 'cash'):
                if key in account and self._is_valid_number(account[key]):
                    candidates.append(float(account[key]))

        # 3) Performance data / env config (initials/baselines & snapshots)
        perf = performance_data.get('performance_data', {}) or {}
        if isinstance(perf, dict):
            for key in ('balance', 'equity', 'initial_balance', 'starting_balance'):
                if key in perf and self._is_valid_number(perf[key]):
                    candidates.append(float(perf[key]))

        env_cfg = performance_data.get('environment_config', {}) or {}
        if isinstance(env_cfg, dict):
            for key in ('initial_balance', 'starting_balance'):
                if key in env_cfg and self._is_valid_number(env_cfg[key]):
                    candidates.append(float(env_cfg[key]))

        # 4) Market state (as a last-ditch snapshot)
        ms = market_data.get('market_state', {}) or {}
        if isinstance(ms, dict) and 'balance' in ms and self._is_valid_number(ms['balance']):
            candidates.append(float(ms['balance']))

        # 5) Environment attributes (mirror if present)
        if self.env is not None:
            for attr in ('balance', 'equity', 'initial_balance'):
                try:
                    if hasattr(self.env, attr):
                        v = getattr(self.env, attr)
                        if self._is_valid_number(v):
                            candidates.append(float(v))
                except Exception:
                    pass

        balance_now = float(candidates[0]) if candidates else 0.0

        # Baseline discovery (explicit initials preferred)
        baseline_candidates: List[float] = []
        for src in (rm, account, perf, env_cfg):
            if isinstance(src, dict):
                for key in ('initial_balance', 'starting_balance'):
                    if key in src and self._is_valid_number(src[key]) and float(src[key]) > 0:
                        baseline_candidates.append(float(src[key]))

        if isinstance(ms, dict) and 'session_start_balance' in ms and self._is_valid_number(ms['session_start_balance']):
            if float(ms['session_start_balance']) > 0:
                baseline_candidates.append(float(ms['session_start_balance']))

        baseline_balance: Optional[float] = baseline_candidates[0] if baseline_candidates else None

        return {
            'balance_now': balance_now,
            'baseline_balance': baseline_balance,
        }

    # ─────────────────────────────────────────────────────────────
    # SmartInfoBus helpers & validators
    # ─────────────────────────────────────────────────────────────
    def _safe_bus_get(self, key: str, report: _Report) -> Optional[Any]:
        """Safely get value from bus with error tracking (never raises)."""
        try:
            value = self.smart_bus.get(key, "RiskAdjustedReward")  # type: ignore[attr-defined]
            if value is None:
                report['missing_keys'].append(key)
                with self._lock:
                    self.missing_key_frequency[key] += 1
                return None

            # Basic sanity check (numerics finite, containers allowed)
            if not self._is_valid_data(value):
                report['broken_keys'].append(key)
                return None

            return value

        except Exception as e:
            report['broken_keys'].append(key)
            report['validation_errors'][key] = str(e)
            return None

    def _is_valid_data(self, value: Any) -> bool:
        """Check if data is structurally usable."""
        if value is None:
            return False
        if isinstance(value, (int, float)):
            try:
                return bool(np.isfinite(float(value)))
            except Exception:
                return False
        # Accept dict/list/tuple—even if empty—as valid containers
        if isinstance(value, (list, tuple, dict)):
            return True
        # Everything else: accept, caller-specific logic will validate
        return True

    def _is_valid_number(self, value: Any) -> bool:
        """Check if value is a valid finite number."""
        try:
            return bool(np.isfinite(float(value)))
        except Exception:
            return False

    def _convert_account_to_risk(self, account_state: Dict[str, Any]) -> Dict[str, Any]:
        """Convert account state to risk metrics format."""
        bal = account_state.get('balance', 0.0)
        eq = account_state.get('equity', bal)
        return {
            'balance': float(bal) if self._is_valid_number(bal) else 0.0,
            'equity': float(eq) if self._is_valid_number(eq) else float(bal) if self._is_valid_number(bal) else 0.0,
            'drawdown': float(account_state.get('drawdown', 0.0) or 0.0),
            'exposure': float(account_state.get('exposure', 0.0) or 0.0),
            'margin_used': float(account_state.get('margin_used', 0.0) or 0.0),
        }

    # Category validators (lightweight, tolerant)
    def _validate_trades(self, trades: Any) -> bool:
        """Validate trade data structure."""
        if trades is None:
            return False
        if not isinstance(trades, (list, tuple)):
            return False
        for trade in trades:
            if not isinstance(trade, dict):
                return False
            # At least one of these commonly-present fields
            if not any(k in trade for k in ('pnl', 'side', 'size', 'price')):
                return False
        return True

    def _validate_risk_metrics(self, risk_metrics: Any) -> bool:
        """Validate risk metrics structure."""
        if not isinstance(risk_metrics, dict):
            return False
        balance_fields = ('balance', 'equity', 'account_equity', 'cash')
        return any(k in risk_metrics for k in balance_fields)

    def _validate_market_data(self, market_data: Any) -> bool:
        return isinstance(market_data, dict)

    def _validate_performance_data(self, perf_data: Any) -> bool:
        return isinstance(perf_data, dict)

    def _validate_memory_data(self, memory_data: Any) -> bool:
        return isinstance(memory_data, dict)

    # ─────────────────────────────────────────────────────────────
    # Quality scoring
    # ─────────────────────────────────────────────────────────────
    def _assess_data_quality(self, report: _Report) -> str:
        """Assess overall data quality from missing/broken stats."""
        missing = len(report['missing_keys'])
        broken = len(report['broken_keys'])

        if missing == 0 and broken == 0:
            return 'excellent'
        if missing <= 2 and broken == 0:
            return 'good'
        if missing <= 5 and broken <= 1:
            return 'acceptable'
        if missing <= 10 and broken <= 3:
            return 'poor'
        return 'invalid'

    # ─────────────────────────────────────────────────────────────
    # Debug helpers (never raise)
    # ─────────────────────────────────────────────────────────────
    def _dbg_note(self, msg: str) -> None:
        try:
            if getattr(self.debug_manager, "enabled", False):
                # Prefer debug_manager hook if available
                if hasattr(self.debug_manager, "_log"):
                    self.debug_manager._log("DEBUG", msg, "DATA_EXTRACTOR")  # type: ignore[attr-defined]
                else:
                    self.logger.debug(msg)
        except Exception:
            try:
                self.logger.debug(msg)
            except Exception:
                pass
