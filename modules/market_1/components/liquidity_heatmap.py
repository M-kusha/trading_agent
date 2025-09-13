# ─────────────────────────────────────────────────────────────
# File: modules/market/components/liquidity_heatmap.py
# Liquidity Heatmap Analysis Component (Production-Ready)
# ─────────────────────────────────────────────────────────────

from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..shared.base_component import BaseMarketComponent


# -----------------------------
# Utilities
# -----------------------------
def _best_num_heads(embed_dim: int, max_heads: int = 8) -> int:
    """Pick a number of attention heads that divides embed_dim."""
    for h in [8, 6, 5, 4, 3, 2]:
        if h <= max_heads and embed_dim % h == 0:
            return h
    return 1


class LiquidityLSTM(nn.Module):
    """
    LSTM + Self-Attention + MLP head.
    Outputs three values:
      0) liquidity score proxy in [0,1] via sigmoid
      1) depth proxy (>0) via softplus
      2) spread proxy (>0) via softplus
    """
    def __init__(
        self,
        input_dim: int,
        lstm_units: int,
        output_dim: int,
        num_layers: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )
        heads = _best_num_heads(lstm_units)
        self.attn = nn.MultiheadAttention(lstm_units, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(lstm_units)

        self.predictor = nn.Sequential(
            nn.Linear(lstm_units, max(64, lstm_units // 2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(max(64, lstm_units // 2), output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        o, _ = self.lstm(x)              # [B,T,H]
        a, _ = self.attn(o, o, o)        # [B,T,H]
        a = self.norm(a + o)             # residual + norm
        last = a[:, -1, :]               # [B,H]
        raw = self.predictor(last)       # [B,3]
        # map to stable ranges
        liq = torch.sigmoid(raw[:, 0:1]) # [0,1]
        depth = F.softplus(raw[:, 1:2])  # >=0
        spread = F.softplus(raw[:, 2:3]) # >=0
        out = torch.cat([liq, depth, spread], dim=1)
        return out


class LiquidityHeatmapComponent(BaseMarketComponent):
    """
    Liquidity analysis with neural predictions.
    Robust handling, online feature normalization, optional MC-Dropout confidence,
    and lightweight online training.
    """

    def __init__(self, config: Dict[str, Any], **kwargs):
        default_config = {
            # Model / training
            'lstm_units': 64,
            'num_layers': 1,
            'sequence_length': 24,
            'dropout_rate': 0.20,
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'grad_clip_norm': 1.0,
            'enable_gpu': True,
            'enable_amp': True,                 # autocast on GPU
            'training_enabled': True,
            'training_min_sequences': 64,       # start training after this many sequences
            'training_batch_size': 32,
            'training_steps_per_call': 2,       # tiny online updates
            'prediction_horizon': 5,

            # Data windows
            'depth_levels': 10,
            'spread_analysis_window': 50,
            'history_max_prices': 1000,
            'history_max_spread': 400,
            'history_max_depth': 400,
            'history_max_volume': 400,

            # Thresholds for descriptive analysis (heuristics, not strict)
            'high_liquidity_threshold': 0.80,
            'low_liquidity_threshold': 0.30,

            # Confidence via MC-Dropout (0 disables)
            'mc_dropout_samples': 0,

            # Scaling anchors (used in postprocessing)
            'depth_scale_hint': 10_000.0,       # typical depth scale
            'spread_scale_hint': 0.0010,        # typical spread (e.g., 0.1% for FX)

            # Determinism
            'torch_deterministic': False,
            'seed': 1337,
        }
        default_config.update(config or {})
        super().__init__(name="LiquidityHeatmap", config=default_config, **kwargs)

    # -----------------------------
    # Lifecycle
    # -----------------------------
    def initialize(self):
        # Seed & device
        seed = int(self.config.get('seed', 1337))
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if bool(self.config.get('torch_deterministic', False)):
            torch.backends.cudnn.deterministic = True  # type: ignore
            torch.backends.cudnn.benchmark = False     # type: ignore

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and self.config['enable_gpu'] else "cpu"
        )
        self._use_amp = bool(self.config.get('enable_amp', True) and self.device.type == "cuda")
        self.trace(f"Using device: {self.device} | AMP={self._use_amp}", level="INFO")

        # Initialize neural network
        self._init_neural_network()

        # Histories
        self.price_history = deque(maxlen=int(self.config['history_max_prices']))
        self.spread_history = deque(maxlen=int(self.config['history_max_spread']))
        self.depth_history = deque(maxlen=int(self.config['history_max_depth']))
        self.volume_history = deque(maxlen=int(self.config['history_max_volume']))

        # Sequences / training buffer
        self.sequence_length = int(self.config['sequence_length'])
        self.sequence_data = deque(maxlen=self.sequence_length + 256)  # keep a bit more to build samples
        self.training_enabled = bool(self.config['training_enabled'])
        self.training_data: deque = deque(maxlen=5000)  # (seq[T,C], target[3])

        # Current state
        self.current_liquidity_score = 0.5
        self.current_spread = 0.0
        self.current_depth = 0.0
        self.market_session = "unknown"

        # Stats
        self.liquidity_stats = {
            "predictions_made": 0,
            "successful_predictions": 0,
            "avg_prediction_accuracy": 0.0,
            "neural_forward_passes": 0,
            "train_steps": 0,
            "dataset_size": 0,
        }

        # Online normalization (Welford) for features
        self._feat_count = 0
        self._feat_mean = np.zeros(4, dtype=np.float64)
        self._feat_M2 = np.zeros(4, dtype=np.float64)

        # Last prediction for online scoring
        self._last_pred: Optional[Dict[str, float]] = None

        self.trace("Liquidity heatmap component initialized", level="DEBUG")

    def _init_neural_network(self):
        input_dim = 4  # [spread_rel, depth_norm, inv_vol, vol_volatility]
        output_dim = 3
        self.lstm_model = LiquidityLSTM(
            input_dim=input_dim,
            lstm_units=int(self.config['lstm_units']),
            output_dim=output_dim,
            num_layers=int(self.config.get('num_layers', 1)),
            dropout_rate=float(self.config['dropout_rate']),
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.lstm_model.parameters(),
            lr=float(self.config['learning_rate']),
            weight_decay=float(self.config.get('weight_decay', 0.0)),
        )
        self.criterion = nn.SmoothL1Loss(beta=0.5)  # robust to outliers

    # -----------------------------
    # Public analysis
    # -----------------------------
    async def analyze_impl(self, **inputs) -> Dict[str, Any]:
        self.trace("Starting liquidity analysis", level="TRACE")

        market_data = inputs.get('market_data', {})
        # shared_context = inputs.get('shared_context', {})  # unused for now

        # 1) Extract + update histories
        liquidity_data = self._extract_liquidity_data(market_data)

        # 2) Analyze heuristics
        liquidity_metrics = self._analyze_liquidity(liquidity_data)

        # 3) Neural prediction (with optional MC-Dropout)
        prediction_result = await self._neural_liquidity_prediction(liquidity_metrics)

        # 4) Sessions (unchanged shape)
        sessions_map, session_meta = self._compute_trading_sessions()

        self.trace(
            f"Liquidity analysis complete: score={liquidity_metrics['liquidity_score']:.3f}, "
            f"session={session_meta['active_session']}",
            level="DEBUG"
        )

        return {
            'liquidity_score': liquidity_metrics['liquidity_score'],
            'market_depth': liquidity_metrics['depth_analysis'],
            'spread_analysis': liquidity_metrics['spread_analysis'],
            'liquidity_prediction': prediction_result,
            'trading_sessions': sessions_map,
            'session_data': session_meta,
            'liquidity_capabilities': {
                'prediction_horizon': self.config['prediction_horizon'],
                'sequence_length': self.config['sequence_length'],
                'device': str(self.device),
                'depth_levels': self.config['depth_levels'],
                'neural_model': 'LSTM+SelfAttention',
                'mc_dropout_samples': int(self.config.get('mc_dropout_samples', 0)),
            },
            'current_spread': self.current_spread,
            'current_depth': self.current_depth,
            'processing_success': True
        }

    # -----------------------------
    # Data extraction / synthetic
    # -----------------------------
    def _extract_liquidity_data(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        self.trace("Extracting liquidity data", level="TRACE")

        prices: List[float] = []
        volumes: List[float] = []
        spreads: List[float] = []
        depth: Dict[str, List[Tuple[float, float]]] = {}

        # Prices
        if 'prices' in market_data and isinstance(market_data['prices'], list):
            prices = [float(p) for p in market_data['prices'] if np.isfinite(p)]
        elif 'EUR/USD' in market_data:
            d = market_data['EUR/USD']
            if isinstance(d, dict) and 'close' in d:
                prices = [float(p) for p in d['close'] if np.isfinite(p)]

        # Spreads
        if 'bid_ask_data' in market_data and isinstance(market_data['bid_ask_data'], dict):
            for _, data in market_data['bid_ask_data'].items():
                s = data.get('spread')
                if s is not None and np.isfinite(s):
                    spreads.append(float(s))

        # Depth
        if 'market_depth' in market_data and isinstance(market_data['market_depth'], dict):
            depth = market_data['market_depth']

        # Volumes (optional)
        if 'volumes' in market_data and isinstance(market_data['volumes'], list):
            volumes = [float(v) for v in market_data['volumes'] if np.isfinite(v)]

        # Synthetic fallback
        if not prices:
            self.trace("Generating synthetic liquidity data", level="WARNING")
            rng = np.random.default_rng(int(self.config.get('seed', 1337)))
            prices = rng.normal(1.1000, 0.001, 64).astype(float).tolist()
            if not volumes:
                volumes = rng.exponential(1000, 64).astype(float).tolist()
            if not spreads:
                spreads = rng.uniform(0.00005, 0.0005, 64).astype(float).tolist()
            if not depth:
                depth = {
                    'bids': [(1.0999, 1000.0), (1.0998, 1500.0)],
                    'asks': [(1.1001, 1200.0), (1.1002, 1800.0)],
                }

        return {
            'prices': prices,
            'volumes': volumes,
            'bid_ask_spreads': spreads,
            'market_depth': depth
        }

    # -----------------------------
    # Heuristic analytics
    # -----------------------------
    def _analyze_liquidity(self, liquidity_data: Dict[str, Any]) -> Dict[str, Any]:
        self.trace("Analyzing liquidity conditions", level="TRACE")

        # Update histories
        prices = liquidity_data['prices']
        spreads = liquidity_data['bid_ask_spreads']
        depth_info = liquidity_data.get('market_depth', {})
        volumes = liquidity_data.get('volumes', [])

        if prices:
            self.price_history.append(float(prices[-1]))
        if spreads:
            self.current_spread = float(spreads[-1])
            self.spread_history.append(self.current_spread)

        if depth_info:
            bid_depth = sum(float(v) for _, v in depth_info.get('bids', []))
            ask_depth = sum(float(v) for _, v in depth_info.get('asks', []))
            self.current_depth = float(max(0.0, bid_depth + ask_depth))
            self.depth_history.append(self.current_depth)

        if volumes:
            self.volume_history.append(float(volumes[-1]))

        # Liquidity score
        liquidity_score = self._calculate_liquidity_score()

        # Analyses
        spread_analysis = self._analyze_spread_patterns()
        depth_analysis = self._analyze_depth_patterns()

        return {
            'liquidity_score': liquidity_score,
            'spread_analysis': spread_analysis,
            'depth_analysis': depth_analysis,
            'current_spread': self.current_spread,
            'current_depth': self.current_depth,
        }

    def _calculate_liquidity_score(self) -> float:
        comps = []

        # Spread (lower better) — normalize by rolling typical spread
        if self.spread_history:
            tail = list(self.spread_history)[-min(50, len(self.spread_history)):]
            avg_spread = float(np.mean(tail))
            typical = float(np.median(tail)) if tail else 1e-6
            denom = max(typical, 1e-8)
            spread_norm = avg_spread / denom
            spread_score = 1.0 - float(np.clip((spread_norm - 1.0) / 3.0, 0.0, 1.0))
            comps.append(spread_score * 0.4)

        # Depth (higher better) — normalize vs rolling median
        if self.depth_history:
            tail = list(self.depth_history)[-min(100, len(self.depth_history)):]
            avg_depth = float(np.mean(tail))
            typical = float(np.median(tail)) if tail else self.config['depth_scale_hint']
            depth_score = float(np.clip(avg_depth / max(typical, 1.0), 0.0, 2.0))
            depth_score = min(depth_score / 1.5, 1.0)
            comps.append(depth_score * 0.4)

        # Price volatility (lower better)
        if len(self.price_history) > 10:
            tail = np.array(list(self.price_history)[-50:], dtype=float)
            if tail.size >= 3:
                base = np.where(tail[:-1] == 0.0, 1.0, tail[:-1])
                rets = (tail[1:] - tail[:-1]) / base
                rets = rets[np.isfinite(rets)]
                vol = float(np.std(rets)) if rets.size > 1 else 0.0
                # map: vol 0% -> 1.0, vol 2% -> ~0.0
                volatility_score = float(np.clip(1.0 - (vol / 0.02), 0.0, 1.0))
                comps.append(volatility_score * 0.2)

        self.current_liquidity_score = float(np.clip(sum(comps) if comps else 0.5, 0.0, 1.0))
        self.trace(f"Liquidity score calculated: {self.current_liquidity_score:.3f}", level="TRACE")
        return self.current_liquidity_score

    def _analyze_spread_patterns(self) -> Dict[str, Any]:
        if len(self.spread_history) < 5:
            return {"status": "insufficient_data", "current_spread": self.current_spread}

        spreads = list(self.spread_history)
        avg_spread = float(np.mean(spreads))
        spread_vol = float(np.std(spreads))
        low_th = float(self.config['low_liquidity_threshold']) * self.config['spread_scale_hint']
        high_th = float(self.config['high_liquidity_threshold']) * self.config['spread_scale_hint']

        if avg_spread < low_th:
            condition = "tight"
        elif avg_spread > high_th:
            condition = "wide"
        else:
            condition = "normal"

        return {
            'average_spread': avg_spread,
            'spread_volatility': spread_vol,
            'condition': condition,
            'trend': 'widening' if spreads[-1] > spreads[-5] else 'tightening',
            'status': 'analyzed',
            'current_spread': self.current_spread
        }

    def _analyze_depth_patterns(self) -> Dict[str, Any]:
        if len(self.depth_history) < 5:
            return {"status": "insufficient_data", "current_depth": self.current_depth}

        depths = list(self.depth_history)
        avg_depth = float(np.mean(depths))
        std_depth = float(np.std(depths))
        depth_stability = 1.0 - (std_depth / max(avg_depth, 1.0))

        if avg_depth > 5 * self.config['depth_scale_hint']:
            condition = "deep"
        elif avg_depth < 0.5 * self.config['depth_scale_hint']:
            condition = "shallow"
        else:
            condition = "moderate"

        return {
            'average_depth': avg_depth,
            'stability_score': float(np.clip(depth_stability, 0.0, 1.0)),
            'condition': condition,
            'trend': 'deepening' if depths[-1] > depths[-5] else 'shallowing',
            'status': 'analyzed',
            'current_depth': self.current_depth,
            'analysis': {
                'depth_percentile': self._get_depth_percentile(self.current_depth)
            }
        }

    def _get_depth_percentile(self, depth: float) -> float:
        if len(self.depth_history) < 10:
            return 50.0
        hist = np.array(list(self.depth_history), dtype=float)
        return float(np.sum(hist <= depth)) / float(len(hist)) * 100.0

    # -----------------------------
    # NN prediction
    # -----------------------------
    async def _neural_liquidity_prediction(self, liquidity_metrics: Dict[str, Any]) -> Dict[str, Any]:
        self.trace("Generating neural liquidity predictions", level="TRACE")

        try:
            # Build current features (robust, scale-invariant)
            features = self._build_feature_vector(liquidity_metrics)
            self._update_feature_stats(features)

            # Append normalized vector into sequence buffer
            norm_feat = self._normalize_features(features)
            self.sequence_data.append(norm_feat.astype(np.float32))

            # Create a training sample (t-horizon) if we have enough history
            self._maybe_buffer_training_sample()

            # Train lightly online
            if self.training_enabled:
                self._maybe_train_online()

            # Not enough sequence yet?
            if len(self.sequence_data) < self.sequence_length:
                return {
                    'predictions': {},
                    'confidence': 0.0,
                    'horizon_steps': int(self.config['prediction_horizon']),
                    'status': 'insufficient_sequence_data'
                }

            # Inference
            x = torch.tensor(
                np.array(list(self.sequence_data)[-self.sequence_length:], dtype=np.float32),
                dtype=torch.float32,
                device=self.device
            ).unsqueeze(0)  # [1,T,C]

            def _forward():
                self.lstm_model.eval()
                with torch.no_grad():
                    return self.lstm_model(x).squeeze(0)  # [3]

            if int(self.config.get('mc_dropout_samples', 0)) > 0:
                # MC-Dropout for uncertainty
                samples = int(self.config['mc_dropout_samples'])
                preds = []
                self.lstm_model.train()  # enable dropout
                with torch.no_grad():
                    for _ in range(samples):
                        preds.append(self.lstm_model(x).squeeze(0).detach().cpu().numpy())
                preds = np.stack(preds, axis=0)  # [S,3]
                mean_pred = preds.mean(axis=0)
                std_pred = preds.std(axis=0)
                confidence = float(np.clip(1.0 / (1.0 + float(np.mean(std_pred))), 0.0, 1.0))
                out = mean_pred
            else:
                if self._use_amp:
                    with torch.cuda.amp.autocast():  # type: ignore
                        out = _forward().detach().cpu().numpy()
                else:
                    out = _forward().detach().cpu().numpy()
                confidence = self._calculate_prediction_confidence()

            # Post-process to real ranges
            pred_liq = float(np.clip(out[0], 0.0, 1.0))
            # Rescale depth/spread using hints so magnitudes look realistic
            pred_depth = float(max(out[1] * self.config['depth_scale_hint'], 0.0))
            pred_spread = float(max(out[2] * self.config['spread_scale_hint'], 0.0))

            # Update stats with last prediction quality (next call will score it)
            self._update_online_accuracy(pred_liq, pred_depth, pred_spread)

            self.liquidity_stats['neural_forward_passes'] += 1

            self.trace(
                f"Neural prediction: liquidity={pred_liq:.3f}, depth={pred_depth:.0f}, spread={pred_spread:.6f}",
                level="DEBUG"
            )

            return {
                'predictions': {
                    'liquidity_score': pred_liq,
                    'depth': pred_depth,
                    'spread': pred_spread
                },
                'confidence': float(np.clip(confidence, 0.0, 1.0)),
                'horizon_steps': int(self.config['prediction_horizon']),
                'status': 'success'
            }

        except Exception as e:
            self.trace(f"Neural prediction failed: {e}", level="ERROR")
            return {
                'predictions': {},
                'confidence': 0.0,
                'horizon_steps': self.config['prediction_horizon'],
                'status': 'error',
                'error': str(e)
            }

    # -----------------------------
    # Feature engineering + online normalization
    # -----------------------------
    def _build_feature_vector(self, liquidity_metrics: Dict[str, Any]) -> np.ndarray:
        """Return [spread_rel, depth_norm, inv_vol, vol_volatility]."""
        price = float(self.price_history[-1]) if self.price_history else 1.0
        spread = float(liquidity_metrics.get('current_spread', 0.0))
        depth = float(liquidity_metrics.get('current_depth', 0.0))

        # Spread relative to price (scale-invariant)
        spread_rel = spread / max(abs(price), 1e-8)

        # Depth normalized by rolling median
        if self.depth_history:
            typical_depth = float(np.median(list(self.depth_history)))
        else:
            typical_depth = float(self.config['depth_scale_hint'])
        depth_norm = depth / max(typical_depth, 1.0)

        # Price volatility inverse (more volatile => lower value)
        inv_vol = 1.0
        if len(self.price_history) >= 10:
            arr = np.array(list(self.price_history)[-40:], dtype=float)
            base = np.where(arr[:-1] == 0.0, 1.0, arr[:-1])
            rets = (arr[1:] - arr[:-1]) / base
            rets = rets[np.isfinite(rets)]
            vol = float(np.std(rets)) if rets.size > 1 else 0.0
            inv_vol = 1.0 / (1.0 + vol)  # in (0,1]

        # Volume volatility
        vol_vol = 0.0
        if len(self.volume_history) >= 5:
            v = np.array(list(self.volume_history)[-40:], dtype=float)
            vol_vol = float(np.std(v)) / (float(np.mean(v)) + 1e-6)  # CV

        return np.array([spread_rel, depth_norm, inv_vol, vol_vol], dtype=np.float64)

    def _update_feature_stats(self, x: np.ndarray):
        """Welford update for per-feature mean/variance."""
        self._feat_count += 1
        delta = x - self._feat_mean
        self._feat_mean += delta / self._feat_count
        delta2 = x - self._feat_mean
        self._feat_M2 += delta * delta2

    def _normalize_features(self, x: np.ndarray) -> np.ndarray:
        if self._feat_count < 10:
            return x.astype(np.float32)
        var = self._feat_M2 / max(1, self._feat_count - 1)
        std = np.sqrt(np.clip(var, 1e-12, None))
        z = (x - self._feat_mean) / std
        # clip to avoid exploding inputs
        z = np.clip(z, -6.0, 6.0)
        return z.astype(np.float32)

    # -----------------------------
    # Online training
    # -----------------------------
    def _maybe_buffer_training_sample(self):
        """
        Build a (sequence, target) pair for next-step targets:
          target = [next_liquidity_score, next_depth_norm, next_spread_rel]
        We use our *heuristic* next observed metrics as supervision.
        """
        if len(self.sequence_data) < self.sequence_length + 1:
            return

        # Targets from the most recent observed point
        if not self.spread_history or not self.depth_history:
            return

        # next observed values (use last element)
        next_spread = float(self.spread_history[-1])
        next_depth = float(self.depth_history[-1])

        price = float(self.price_history[-1]) if self.price_history else 1.0
        next_spread_rel = next_spread / max(abs(price), 1e-8)

        if self.depth_history:
            typical_depth = float(np.median(list(self.depth_history)))
        else:
            typical_depth = float(self.config['depth_scale_hint'])
        next_depth_norm = next_depth / max(typical_depth, 1.0)

        next_liq = float(np.clip(self.current_liquidity_score, 0.0, 1.0))

        seq = np.array(list(self.sequence_data)[-self.sequence_length:], dtype=np.float32)
        targ = np.array([next_liq, next_depth_norm, next_spread_rel], dtype=np.float32)
        self.training_data.append((seq, targ))
        self.liquidity_stats['dataset_size'] = len(self.training_data)

    def _maybe_train_online(self):
        if len(self.training_data) < int(self.config['training_min_sequences']):
            return

        steps = int(self.config['training_steps_per_call'])
        batch_size = int(self.config['training_batch_size'])
        if steps <= 0 or batch_size <= 0:
            return

        self.lstm_model.train()
        for _ in range(steps):
            # Mini-batch sample
            idx = np.random.choice(len(self.training_data), size=min(batch_size, len(self.training_data)), replace=False)
            batch_seq = np.stack([self.training_data[i][0] for i in idx], axis=0)  # [B,T,C]
            batch_targ = np.stack([self.training_data[i][1] for i in idx], axis=0) # [B,3]

            x = torch.tensor(batch_seq, dtype=torch.float32, device=self.device)
            y = torch.tensor(batch_targ, dtype=torch.float32, device=self.device)

            self.optimizer.zero_grad(set_to_none=True)
            if self._use_amp:
                scaler = getattr(self, "_scaler", None)
                if scaler is None:
                    self._scaler = torch.cuda.amp.GradScaler()  # type: ignore
                    scaler = self._scaler
                with torch.cuda.amp.autocast():  # type: ignore
                    out = self.lstm_model(x)
                    loss = self.criterion(out, y)
                scaler.scale(loss).backward()
                # clip
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), float(self.config['grad_clip_norm']))
                scaler.step(self.optimizer)
                scaler.update()
            else:
                out = self.lstm_model(x)
                loss = self.criterion(out, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.lstm_model.parameters(), float(self.config['grad_clip_norm']))
                self.optimizer.step()

            self.liquidity_stats['train_steps'] += 1

        self.lstm_model.eval()

    # -----------------------------
    # Confidence / online accuracy
    # -----------------------------
    def _update_online_accuracy(self, pred_liq: float, pred_depth: float, pred_spread: float):
        """
        Compare previous predictions to newly observed values and update stats.
        This gives a rough rolling accuracy proxy without labels.
        """
        # Score last prediction if we can
        if self._last_pred is not None:
            # Observed now:
            obs_liq = float(np.clip(self.current_liquidity_score, 0.0, 1.0))
            obs_depth = float(self.current_depth)
            obs_spread = float(self.current_spread)

            # Directional correctness for depth & spread; absolute error for liq
            correct = 0
            total = 0

            # Liquidity: within 0.10 absolute error counts as success
            if abs(self._last_pred['liquidity'] - obs_liq) <= 0.10:
                correct += 1
            total += 1

            # Depth direction
            if (obs_depth - self._last_pred['depth_obs_base']) * (self._last_pred['depth'] - self._last_pred['depth_obs_base']) >= 0:
                correct += 1
            total += 1

            # Spread direction
            if (obs_spread - self._last_pred['spread_obs_base']) * (self._last_pred['spread'] - self._last_pred['spread_obs_base']) >= 0:
                correct += 1
            total += 1

            self.liquidity_stats['predictions_made'] += 1
            self.liquidity_stats['successful_predictions'] += correct / max(total, 1)

            n = self.liquidity_stats['predictions_made']
            acc = self.liquidity_stats['successful_predictions'] / max(n, 1)
            self.liquidity_stats['avg_prediction_accuracy'] = float(np.clip(acc, 0.0, 1.0))

        # Store this prediction with current obs for next-time scoring
        self._last_pred = {
            'liquidity': pred_liq,
            'depth': pred_depth,
            'spread': pred_spread,
            'depth_obs_base': float(self.current_depth),
            'spread_obs_base': float(self.current_spread),
        }

    def _calculate_prediction_confidence(self) -> float:
        """
        Heuristic confidence based on (a) online accuracy and (b) sequence coverage.
        """
        if self.liquidity_stats['predictions_made'] == 0:
            baseline = 0.5
        else:
            baseline = float(self.liquidity_stats['avg_prediction_accuracy'])

        data_quality = min(len(self.sequence_data) / max(1, self.sequence_length), 1.0)
        confidence = baseline * 0.7 + data_quality * 0.3
        return float(np.clip(confidence, 0.0, 1.0))

    # -----------------------------
    # Trading sessions (kept compatible)
    # -----------------------------
    def _compute_trading_sessions(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        import datetime

        now = datetime.datetime.utcnow()
        weekday = now.weekday()
        hour = now.hour + now.minute / 60.0

        weekend = (weekday == 5) or (weekday == 6)

        def in_window(h, start, end):
            return (h >= start and h < end) if start < end else (h >= start or h < end)

        asian = in_window(hour, 23.0, 7.0) and not weekend
        european = (7.0 <= hour < 16.0) and not weekend
        american = (12.0 <= hour < 21.0) and not weekend
        rollover = (21.0 <= hour < 23.0) and not weekend

        if weekend:
            active, bias = "weekend", 0.3
        elif rollover:
            active, bias = "rollover", 0.4
        elif american:
            active, bias = "american", 1.0
        elif european:
            active, bias = "european", 0.9
        elif asian:
            active, bias = "asian", 0.7
        else:
            active, bias = "unknown", 0.6

        sessions_map = {
            'asian': bool(asian),
            'european': bool(european),
            'american': bool(american),
            'rollover': bool(rollover),
            'weekend': bool(weekend),
            'active': active
        }

        session_meta = {
            'active_session': active,
            'utc_time': now.isoformat() + 'Z',
            'windows_utc': {
                'asian': {'start': '23:00', 'end': '07:00'},
                'european': {'start': '07:00', 'end': '16:00'},
                'american': {'start': '12:00', 'end': '21:00'},
                'rollover': {'start': '21:00', 'end': '23:00'},
            },
            'liquidity_bias': bias
        }

        return sessions_map, session_meta

    # -----------------------------
    # Fallback
    # -----------------------------
    def get_fallback_result(self, error: str) -> Dict[str, Any]:
        sessions_map, session_meta = self._compute_trading_sessions()

        return {
            'liquidity_score': float(getattr(self, "current_liquidity_score", 0.5)),
            'market_depth': {
                'status': 'fallback',
                'current_depth': float(getattr(self, "current_depth", 0.0)),
                'analysis': {},
                'condition': 'unknown'
            },
            'spread_analysis': {
                'status': 'fallback',
                'current_spread': float(getattr(self, "current_spread", 0.0)),
                'analysis': {},
                'condition': 'unknown'
            },
            'liquidity_prediction': {
                'predictions': {},
                'confidence': 0.0,
                'horizon_steps': int(self.config['prediction_horizon']),
                'status': 'unavailable'
            },
            'trading_sessions': sessions_map,
            'session_data': session_meta,
            'liquidity_capabilities': {
                'prediction_horizon': self.config['prediction_horizon'],
                'sequence_length': self.config['sequence_length'],
                'device': str(self.device),
                'depth_levels': self.config['depth_levels'],
                'neural_model': 'LSTM+SelfAttention',
                'mc_dropout_samples': int(self.config.get('mc_dropout_samples', 0)),
            },
            'processing_success': False,
            'error': error
        }
