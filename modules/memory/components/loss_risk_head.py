# modules/memory/components/loss_risk_head.py
"""
Loss Risk Head Component
Online calibrated head: (features, action, regime) → P(loss > τ)

Provides:
- loss_prob: Probability of significant loss
- uncertainty: Model uncertainty estimate
- calibration_ece: Expected Calibration Error metric

Integration: UnifiedMemory fuses loss_prob into risk_score = max(loss_prob, danger_similarity)
and weights memory_vote by (1 - uncertainty).
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.memory.shared.utils import safe_float

from .base import MemoryComponent


class LossRiskHead(nn.Module):
    """
    Neural network head for loss probability prediction.
    
    Architecture:
    - Input: concatenated [features, action, regime_encoding]
    - Hidden: 2-layer MLP with dropout
    - Output: (loss_prob, uncertainty) via mean/variance decomposition
    """
    
    def __init__(self, input_dim: int = 48, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Dual heads: mean (logit) and log-variance for uncertainty
        self.mean_head = nn.Linear(hidden_dim // 2, 1)
        self.var_head = nn.Linear(hidden_dim // 2, 1)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, input_dim]
            
        Returns:
            (loss_prob, uncertainty) each [B, 1]
        """
        h = self.net(x)
        
        # Mean head -> sigmoid for probability
        logit = self.mean_head(h)
        loss_prob = torch.sigmoid(logit)
        
        # Variance head -> softplus for positive variance, then convert to uncertainty
        log_var = self.var_head(h)
        variance = F.softplus(log_var)
        # Uncertainty: higher variance = higher uncertainty, normalized to [0, 1]
        uncertainty = torch.sigmoid(variance)
        
        return loss_prob, uncertainty


class LossRiskHeadComponent(MemoryComponent):
    """
    Loss Risk Head Component for online calibrated loss probability prediction.
    
    Provides pre-trade risk assessment:
    - loss_prob: Estimated probability of significant loss (> threshold)
    - uncertainty: Model confidence/uncertainty
    - calibration_ece: Expected Calibration Error for monitoring
    
    Features:
    - Online learning from trade outcomes
    - Regime-aware predictions
    - Calibration tracking
    - Soft labels from PnL magnitude
    """
    
    # Constants
    _BUFFER_FRACTION: float = 0.15
    _MIN_SAMPLES_FOR_TRAINING: int = 20
    _TRAINING_BATCH_SIZE: int = 16
    _LOSS_THRESHOLD: float = 5.0  # τ: PnL threshold for "significant loss"
    # Updated: features(32) + action(2) + regime(4) + vol(1) + session(4) + instrument(2) + extra(3)
    _INPUT_DIM: int = 48
    _REGIME_ENCODING: Dict[str, List[float]] = {
        "trending": [1.0, 0.0, 0.0, 0.0],
        "volatile": [0.0, 1.0, 0.0, 0.0],
        "ranging": [0.0, 0.0, 1.0, 0.0],
        "unknown": [0.0, 0.0, 0.0, 1.0],
    }
    _SESSION_ENCODING: Dict[str, List[float]] = {
        "asian": [1.0, 0.0, 0.0, 0.0],
        "european": [0.0, 1.0, 0.0, 0.0],
        "american": [0.0, 0.0, 1.0, 0.0],
        "us": [0.0, 0.0, 1.0, 0.0],  # alias for american
        "closed": [0.0, 0.0, 0.0, 1.0],
        "unknown": [0.25, 0.25, 0.25, 0.25],
    }
    # Instrument encoding (2 dims) for per-instrument risk learning
    _INSTRUMENT_ENCODING: Dict[str, List[float]] = {
        "EUR_USD": [1.0, 0.0],
        "EURUSD": [1.0, 0.0],  # alias
        "XAU_USD": [0.0, 1.0],
        "XAUUSD": [0.0, 1.0],  # alias
        "UNKNOWN": [0.5, 0.5],
    }
    
    def _initialize_component(self) -> None:
        """Initialize loss risk head resources."""
        cfg = self.config
        
        # Configuration
        self.max_memory_size: int = int(getattr(cfg, "max_memory_size", 10_000))
        self.embed_dim: int = int(getattr(cfg, "embed_dim", 32))
        self.loss_threshold: float = float(getattr(cfg, "loss_risk_threshold", self._LOSS_THRESHOLD))
        
        # P1 FIX: Use centralized config dimension if available, otherwise use class default
        input_dim = int(getattr(cfg, "loss_head_input_dim", self._INPUT_DIM))
        
        # Device detection
        self._device = self._infer_device()
        
        # Neural model - use configured input_dim
        self.risk_head = LossRiskHead(
            input_dim=input_dim,
            hidden_dim=64
        ).to(self._device)
        
        # Store actual input dim for feature building
        self._actual_input_dim = input_dim
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.risk_head.parameters(), lr=1e-3)
        
        # Training data buffer: (input_vec, label, soft_label)
        self.training_buffer: deque[Tuple[np.ndarray, float, float]] = deque(
            maxlen=int(self.max_memory_size * self._BUFFER_FRACTION)
        )
        
        # Calibration tracking (binned predictions vs actuals)
        self.calibration_bins: Dict[int, Dict[str, float]] = {
            i: {"predicted_sum": 0.0, "actual_sum": 0.0, "count": 0}
            for i in range(10)  # 10 bins: 0-0.1, 0.1-0.2, ..., 0.9-1.0
        }
        
        # State
        self.samples_trained: int = 0
        self.last_loss_prob: float = 0.0
        self.last_uncertainty: float = 0.5
        self.calibration_ece: float = 0.0
        
        # Metrics history
        self.prediction_history: deque[Dict[str, Any]] = deque(maxlen=100)
        self.training_loss_history: deque[float] = deque(maxlen=100)
        
        self._log_debug(
            "loss_risk_head_initialized",
            details={
                "input_dim": self._INPUT_DIM,
                "loss_threshold": self.loss_threshold,
                "device": str(self._device),
            },
        )
    
    def _infer_device(self) -> torch.device:
        """Infer execution device."""
        if self.encoder is not None:
            try:
                p = next(self.encoder.parameters(), None)
                if p is not None:
                    return p.device
            except Exception:
                pass
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process loss risk head operations."""
        try:
            # 1. Learn from completed trades
            learning_result = self._process_learning_data(context)
            
            # 2. Train if enough samples
            if len(self.training_buffer) >= self._MIN_SAMPLES_FOR_TRAINING:
                training_result = self._train_step()
                learning_result.update(training_result)
            
            # 3. Predict for current context
            prediction_result = self._predict_loss_risk(context)
            learning_result.update(prediction_result)
            
            # 4. Update calibration metrics
            self._update_calibration_ece()
            
            return self._format_output(learning_result)
            
        except Exception as e:
            self.log_error("Loss risk head processing failed", e)
            return self._get_fallback_output()
    
    def _process_learning_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process completed trades to update training buffer."""
        trades: List[Dict[str, Any]] = context.get("trades", []) or []
        market_context: Dict[str, Any] = context.get("market_context", {}) or {}
        
        samples_added = 0
        
        for trade in trades[-20:]:
            if not isinstance(trade, dict) or "pnl" not in trade:
                continue
            
            pnl = safe_float(trade.get("pnl", 0.0), 0.0)
            input_vec = self._build_input_vector(trade, market_context, context.get("features"))
            
            if input_vec is None:
                continue
            
            # Binary label: 1 if loss > threshold
            label = 1.0 if pnl < -self.loss_threshold else 0.0
            
            # Soft label: smooth based on loss magnitude
            # Maps PnL to [0, 1] where large losses → 1
            if pnl < 0:
                soft_label = float(np.clip(-pnl / (self.loss_threshold * 3), 0.0, 1.0))
            else:
                soft_label = 0.0
            
            self.training_buffer.append((input_vec, label, soft_label))
            samples_added += 1
            
            # Update calibration bins with actual outcome
            self._update_calibration_actual(label)
        
        return {
            "samples_added": samples_added,
            "buffer_size": len(self.training_buffer),
        }
    
    def _build_input_vector(
        self,
        trade: Dict[str, Any],
        market_context: Dict[str, Any],
        features: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """Build input vector for the risk head."""
        try:
            parts: List[float] = []
            
            # 1. Features (32 dims) - from context or extract from trade
            if features is not None:
                feat_arr = self._coerce_numeric_array(features, self.embed_dim)
                parts.extend(feat_arr.tolist())
            else:
                # Extract basic features from trade
                feat = self.extractor.extract_trade_features(trade, market_context)
                feat = self._pad_or_trim(np.asarray(feat, dtype=np.float32).reshape(-1), self.embed_dim)
                parts.extend(feat.tolist())
            
            # 2. Action (2 dims)
            action = trade.get("action", [0.0, 0.0])
            if isinstance(action, dict):
                action_values = [safe_float(v, 0.0) for v in action.values()]
                action_arr = np.asarray(action_values, dtype=np.float32).reshape(-1)
                parts.extend(self._pad_or_trim(action_arr, 2).tolist())
            elif isinstance(action, (list, tuple, np.ndarray)):
                action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                parts.extend(self._pad_or_trim(action_arr, 2).tolist())
            else:
                parts.extend([safe_float(action, 0.0), 0.5])
            
            # 3. Regime encoding (4 dims)
            regime = str(market_context.get("regime", "unknown")).lower()
            regime_enc = self._REGIME_ENCODING.get(regime, self._REGIME_ENCODING["unknown"])
            parts.extend(regime_enc)
            
            # 4. Volatility (1 dim)
            vol_raw = market_context.get("volatility", 0.5)
            if isinstance(vol_raw, dict):
                vol_values = list(vol_raw.values())
                vol_value = safe_float(vol_values[0] if vol_values else 0.5, 0.5)
            else:
                vol_value = safe_float(vol_raw, 0.5)
            parts.append(vol_value)
            
            # 5. Session encoding (4 dims)
            session = str(market_context.get("session", "unknown")).lower()
            session_enc = self._SESSION_ENCODING.get(session, self._SESSION_ENCODING["unknown"])
            parts.extend(session_enc)
            
            # 6. Instrument encoding (2 dims) - for per-instrument risk learning
            instrument = str(trade.get("instrument") or trade.get("symbol") or "UNKNOWN").upper().replace("/", "_")
            instrument_enc = self._INSTRUMENT_ENCODING.get(instrument, self._INSTRUMENT_ENCODING["UNKNOWN"])
            parts.extend(instrument_enc)
            
            # 7. Extra features (3 dims) - trade characteristics using safe_float from shared utils
            confidence = safe_float(trade.get("confidence", 0.5), 0.5)
            size = safe_float(trade.get("size", 0.0), 0.0)
            duration = safe_float(trade.get("duration", 1.0), 1.0)
            
            parts.append(confidence)
            parts.append(size / 10.0)
            parts.append(duration / 100.0)
            
            # Pad/trim to input dim - P1 FIX: use actual configured input dim
            target_dim = getattr(self, '_actual_input_dim', self._INPUT_DIM)
            arr = np.asarray(parts, dtype=np.float32)
            arr = self._pad_or_trim(arr, target_dim)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            
            return arr
            
        except Exception as e:
            self.log_error("Input vector build failed", e)
            return None
    
    def _train_step(self) -> Dict[str, Any]:
        """Perform one training step on buffered data."""
        try:
            if len(self.training_buffer) < self._TRAINING_BATCH_SIZE:
                return {"training_performed": False}
            
            # Sample a batch
            indices = np.random.choice(
                len(self.training_buffer),
                size=min(self._TRAINING_BATCH_SIZE, len(self.training_buffer)),
                replace=False
            )
            
            batch_inputs: List[np.ndarray] = []
            batch_labels: List[float] = []
            batch_soft: List[float] = []
            
            buffer_list = list(self.training_buffer)
            for idx in indices:
                inp, lbl, soft = buffer_list[idx]
                batch_inputs.append(inp)
                batch_labels.append(lbl)
                batch_soft.append(soft)
            
            # Convert to tensors
            X = torch.tensor(np.stack(batch_inputs), dtype=torch.float32, device=self._device)
            # Combined target: blend hard and soft labels
            y_hard = torch.tensor(batch_labels, dtype=torch.float32, device=self._device).unsqueeze(1)
            y_soft = torch.tensor(batch_soft, dtype=torch.float32, device=self._device).unsqueeze(1)
            y = 0.7 * y_hard + 0.3 * y_soft  # Weighted combination
            
            # Forward pass
            self.risk_head.train()
            self.optimizer.zero_grad()
            
            loss_prob, uncertainty = self.risk_head(X)
            
            # Loss: BCE for probability + regularization on uncertainty
            bce_loss = F.binary_cross_entropy(loss_prob, y)
            # Encourage uncertainty to be high when wrong, low when right
            pred_error = torch.abs(loss_prob - y)
            uncertainty_loss = torch.mean((uncertainty - pred_error) ** 2)
            
            total_loss = bce_loss + 0.1 * uncertainty_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.risk_head.parameters(), 1.0)
            self.optimizer.step()
            
            self.samples_trained += len(batch_inputs)
            loss_val = float(total_loss.item())
            self.training_loss_history.append(loss_val)
            
            return {
                "training_performed": True,
                "training_loss": loss_val,
                "batch_size": len(batch_inputs),
            }
            
        except Exception as e:
            self.log_error("Training step failed", e)
            return {"training_performed": False, "error": str(e)}
    
    def _predict_loss_risk(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Predict loss probability for current context."""
        try:
            market_context = context.get("market_context", {}) or {}
            features = context.get("features")
            
            # Build a "prospective" trade input (no actual trade yet)
            dummy_trade = {
                "confidence": 0.5,
                "size": 1.0,
                "duration": 1.0,
                "action": context.get("proposed_action", [0.0, 0.0]),
            }
            
            input_vec = self._build_input_vector(dummy_trade, market_context, features)
            
            if input_vec is None:
                self.last_loss_prob = 0.3
                self.last_uncertainty = 0.5
                return {
                    "loss_prob": 0.3,
                    "uncertainty": 0.5,
                    "prediction_valid": False,
                }
            
            # Forward pass
            self.risk_head.eval()
            with torch.no_grad():
                X = torch.tensor(input_vec, dtype=torch.float32, device=self._device).unsqueeze(0)
                loss_prob, uncertainty = self.risk_head(X)
                
                self.last_loss_prob = float(loss_prob.item())
                self.last_uncertainty = float(uncertainty.item())
            
            # Update calibration bins with prediction
            self._update_calibration_predicted(self.last_loss_prob)
            
            # Record for tracking
            self.prediction_history.append({
                "timestamp": time.time(),
                "loss_prob": self.last_loss_prob,
                "uncertainty": self.last_uncertainty,
                "regime": market_context.get("regime", "unknown"),
            })
            
            return {
                "loss_prob": self.last_loss_prob,
                "uncertainty": self.last_uncertainty,
                "prediction_valid": True,
            }
            
        except Exception as e:
            self.log_error("Prediction failed", e)
            return {
                "loss_prob": 0.3,
                "uncertainty": 0.5,
                "prediction_valid": False,
            }
    
    def _update_calibration_predicted(self, prob: float) -> None:
        """Update calibration bins with predicted probability."""
        bin_idx = min(9, int(prob * 10))
        self.calibration_bins[bin_idx]["predicted_sum"] += prob
        self.calibration_bins[bin_idx]["count"] += 1
    
    def _update_calibration_actual(self, actual: float) -> None:
        """Update calibration bins with actual outcome."""
        # Use last prediction's bin
        if self.prediction_history:
            last_prob = self.prediction_history[-1].get("loss_prob", 0.5)
            bin_idx = min(9, int(last_prob * 10))
            self.calibration_bins[bin_idx]["actual_sum"] += actual
    
    def _update_calibration_ece(self) -> None:
        """Calculate Expected Calibration Error."""
        total_samples = sum(b["count"] for b in self.calibration_bins.values())
        if total_samples == 0:
            self.calibration_ece = 0.0
            return
        
        ece = 0.0
        for bin_data in self.calibration_bins.values():
            n = bin_data["count"]
            if n == 0:
                continue
            
            avg_pred = bin_data["predicted_sum"] / n
            avg_actual = bin_data["actual_sum"] / max(1, n)
            
            ece += (n / total_samples) * abs(avg_pred - avg_actual)
        
        self.calibration_ece = float(ece)
    
    def _format_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Format output to match contract requirements."""
        avg_loss = float(np.mean(list(self.training_loss_history))) if self.training_loss_history else 0.0
        
        return {
            "loss_risk_assessment": {
                "loss_prob": self.last_loss_prob,
                "uncertainty": self.last_uncertainty,
                "calibration_ece": self.calibration_ece,
                "confidence": 1.0 - self.last_uncertainty,
            },
            "loss_risk_training": {
                "samples_trained": self.samples_trained,
                "buffer_size": len(self.training_buffer),
                "avg_training_loss": avg_loss,
                "training_performed": result.get("training_performed", False),
            },
            "loss_risk_metrics": {
                "prediction_count": len(self.prediction_history),
                "last_prediction_time": self.prediction_history[-1]["timestamp"] if self.prediction_history else 0.0,
                "model_ready": self.samples_trained >= self._MIN_SAMPLES_FOR_TRAINING,
            },
        }
    
    def _get_fallback_output(self) -> Dict[str, Any]:
        """Return fallback output on error."""
        return {
            "loss_risk_assessment": {
                "loss_prob": 0.3,
                "uncertainty": 0.5,
                "calibration_ece": 0.0,
                "confidence": 0.5,
            },
            "loss_risk_training": {
                "samples_trained": 0,
                "buffer_size": 0,
                "avg_training_loss": 0.0,
                "training_performed": False,
            },
            "loss_risk_metrics": {
                "prediction_count": 0,
                "last_prediction_time": 0.0,
            "model_ready": False,
            },
        }
    
    @staticmethod
    def _coerce_numeric_array(data: Any, target_size: int) -> np.ndarray:
        """
        Convert arbitrary input (array, list, dict) to a float32 array of target_size.
        """
        try:
            arr = np.asarray(data, dtype=np.float32).reshape(-1)
        except Exception:
            if isinstance(data, dict):
                arr = np.array([safe_float(v, 0.0) for v in data.values()], dtype=np.float32)
            elif isinstance(data, (list, tuple, set)):
                arr = np.array([safe_float(v, 0.0) for v in data], dtype=np.float32)
            else:
                arr = np.array([safe_float(data, 0.0)], dtype=np.float32)

        if arr.size == 0:
            arr = np.zeros(target_size, dtype=np.float32)

        return LossRiskHeadComponent._pad_or_trim(arr, target_size)

    @staticmethod
    def _pad_or_trim(arr: np.ndarray, size: int) -> np.ndarray:
        """Pad or trim array to exact size."""
        arr = arr.reshape(-1)
        if arr.size == size:
            return arr
        if arr.size < size:
            return np.pad(arr, (0, size - arr.size), mode="constant")
        return arr[:size]
