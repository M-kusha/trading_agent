# envs/prop_firm/signals/market_structure.py
# pyright: reportAttributeAccessIssue=false
"""
Market structure signal computation mixin for PropFirmTradingEnv.

Institutional-grade S/R detection, order blocks, liquidity pools.

Design goals:
- ATR-scaled thresholds for regime robustness
- Fractal pivots + clustering + rejection validation for high-signal S/R
- BOS/CHOCH via close breaks for stability
- Liquidity pools via equal-high/low clustering
- Order blocks via displacement + unmitigated zone detection
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class MarketStructureMixin:
    """Pure computation market-structure methods (only depend on passed arrays)."""

    # ---------------------------
    # ATR + Pivots + Clustering Helpers
    # ---------------------------

    def _structure_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
        """Robust ATR proxy for structure scaling."""
        n = int(len(close))
        if n < period + 2:
            return float(max(np.mean(high - low), 1e-8))

        h = np.asarray(high[-(period + 1):], dtype=np.float64)
        l = np.asarray(low[-(period + 1):], dtype=np.float64)
        c = np.asarray(close[-(period + 1):], dtype=np.float64)

        prev_c = c[:-1]
        tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - prev_c), np.abs(l[1:] - prev_c)))
        atr = float(np.mean(tr))
        return float(max(atr, 1e-8))

    def _find_fractal_pivots(
        self,
        high: np.ndarray,
        low: np.ndarray,
        left: int = 3,
        right: int = 3,
    ) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """
        Fractal pivots (strict uniqueness) - VECTORIZED for speed.
        Returns lists of (index, price) in ascending index order.
        """
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)

        n = int(len(h))
        if n < left + right + 3:
            return [], []

        # Vectorized: build rolling windows using stride tricks
        window_size = left + right + 1
        
        # Create sliding window views
        shape = (n - window_size + 1, window_size)
        strides = (h.strides[0], h.strides[0])
        
        h_windows = np.lib.stride_tricks.as_strided(h, shape=shape, strides=strides)
        l_windows = np.lib.stride_tricks.as_strided(l, shape=shape, strides=strides)
        
        # Center values (index 'left' within each window)
        center_h = h_windows[:, left]
        center_l = l_windows[:, left]
        
        # Check if center is max/min of window
        is_max = center_h == np.max(h_windows, axis=1)
        is_min = center_l == np.min(l_windows, axis=1)
        
        # Check uniqueness (only one occurrence of max/min value)
        unique_max = np.sum(h_windows == center_h[:, None], axis=1) == 1
        unique_min = np.sum(l_windows == center_l[:, None], axis=1) == 1
        
        # Combine conditions
        piv_hi_mask = is_max & unique_max
        piv_lo_mask = is_min & unique_min
        
        # Extract indices (offset by 'left' since windows start at 0)
        piv_hi_indices = np.where(piv_hi_mask)[0] + left
        piv_lo_indices = np.where(piv_lo_mask)[0] + left
        
        piv_hi = [(int(i), float(h[i])) for i in piv_hi_indices]
        piv_lo = [(int(i), float(l[i])) for i in piv_lo_indices]

        return piv_hi, piv_lo

    def _cluster_levels_1d(self, levels: List[float], eps: float) -> List[Tuple[float, int]]:
        """
        Centroid-based clustering for 1D levels.
        Returns list of (cluster_centroid, count), sorted by centroid ascending.
        """
        if not levels:
            return []

        xs = sorted(float(x) for x in levels if np.isfinite(x))
        if not xs:
            return []

        clusters: List[Tuple[float, int]] = []
        centroid = xs[0]
        count = 1

        for x in xs[1:]:
            if abs(x - centroid) <= eps:
                # Update centroid incrementally
                centroid = (centroid * count + x) / (count + 1)
                count += 1
            else:
                clusters.append((float(centroid), int(count)))
                centroid = x
                count = 1

        clusters.append((float(centroid), int(count)))
        return clusters

    def _count_rejections(
        self,
        level: float,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        *,
        side: str,                 # "support" or "resistance"
        eps_touch: float,
        eps_break: float,
        move_away: float,
        fwd: int = 3,
        cooldown: Optional[int] = None,
    ) -> int:
        """
        Count validated rejections - VECTORIZED for speed.
        """
        h = np.asarray(highs, dtype=np.float64)
        l = np.asarray(lows, dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)

        n = int(len(c))
        if n < fwd + 2:
            return 0

        cd = int(cooldown) if cooldown is not None else int(fwd)
        
        # Vectorized touch detection
        touched = ((l - eps_touch) <= level) & (level <= (h + eps_touch))
        
        # Pre-compute forward windows for all bars at once
        # Using stride tricks for rolling max/min
        if n <= fwd + 1:
            return 0
            
        # Create forward window views
        valid_range = n - fwd - 1
        if valid_range <= 0:
            return 0
        
        # Forward highs max, forward lows min, forward closes
        fwd_h_max = np.array([np.max(h[i+1:i+1+fwd]) for i in range(valid_range)])
        fwd_l_min = np.array([np.min(l[i+1:i+1+fwd]) for i in range(valid_range)])
        
        # Check break and move-away conditions vectorized
        touched_valid = touched[:valid_range]
        
        if side == "support":
            not_broken = fwd_l_min >= (level - eps_break)
            moved_away = np.array([np.any(c[i+1:i+1+fwd] > (level + move_away)) for i in range(valid_range)])
        else:
            not_broken = fwd_h_max <= (level + eps_break)
            moved_away = np.array([np.any(c[i+1:i+1+fwd] < (level - move_away)) for i in range(valid_range)])
        
        # Valid rejections
        valid = touched_valid & not_broken & moved_away
        
        # Apply cooldown by iterating through valid rejections
        if not np.any(valid):
            return 0
            
        valid_indices = np.where(valid)[0]
        rej = 0
        last_rej_idx = -cd
        for idx in valid_indices:
            if idx >= last_rej_idx + cd:
                rej += 1
                last_rej_idx = idx
        
        return int(rej)

    # ---------------------------
    # Institutional-Grade S/R Detection
    # ---------------------------

    def _compute_market_structure_signals(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> Tuple[float, float]:
        """
        Fast S/R proximity using fractal pivots with ATR-scaled distance scoring.
        Optimized for training speed while preserving signal quality.

        Returns: near_support, near_resistance in [0, 1]
        """
        if len(close) < 60:
            return 0.0, 0.0

        current_price = float(close[-1])
        if current_price <= 0 or not np.isfinite(current_price):
            return 0.0, 0.0

        lookback = min(120, len(close))  # Reduced from 180
        h = np.asarray(high[-lookback:], dtype=np.float64)
        l = np.asarray(low[-lookback:], dtype=np.float64)
        c = np.asarray(close[-lookback:], dtype=np.float64)

        atr = self._structure_atr(h, l, c, period=14)
        eps_prox = max(0.25 * atr, current_price * 0.0012)

        piv_hi, piv_lo = self._find_fractal_pivots(h, l, left=3, right=3)

        # Fast: just score based on distance to nearest pivots
        near_resistance = 0.0
        if piv_hi:
            # Find pivots ABOVE current price (resistance)
            above = [(i, p) for i, p in piv_hi if p > current_price]
            if above:
                # Weight by recency (newer pivots matter more)
                best_score = 0.0
                for idx, level in above[-4:]:  # Only last 4
                    dist = level - current_price
                    if dist <= 3.0 * eps_prox:
                        recency = (idx / lookback) ** 0.5  # sqrt decay
                        prox = float(np.exp(-dist / max(eps_prox, 1e-8)))
                        best_score = max(best_score, prox * (0.5 + 0.5 * recency))
                near_resistance = float(np.clip(best_score, 0.0, 1.0))

        near_support = 0.0
        if piv_lo:
            # Find pivots BELOW current price (support)
            below = [(i, p) for i, p in piv_lo if p < current_price]
            if below:
                best_score = 0.0
                for idx, level in below[-4:]:  # Only last 4
                    dist = current_price - level
                    if dist <= 3.0 * eps_prox:
                        recency = (idx / lookback) ** 0.5
                        prox = float(np.exp(-dist / max(eps_prox, 1e-8)))
                        best_score = max(best_score, prox * (0.5 + 0.5 * recency))
                near_support = float(np.clip(best_score, 0.0, 1.0))

        return float(near_support), float(near_resistance)

    # ---------------------------
    # Advanced Market Structure (BOS/CHOCH + Liquidity + Order Blocks)
    # ---------------------------

    def _compute_advanced_market_structure(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        open_: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Advanced structure:
        - structure_trend/strength from pivot sequences (ATR-aware)
        - BOS using close breaks of last pivots
        - liquidity pools as equal-high/low clusters
        - order blocks as displacement-based, unmitigated zones
        """
        out = {
            "structure_trend": 0.0,
            "structure_strength": 0.0,
            "bos_signal": 0.0,
            "liquidity_above": 0.0,
            "liquidity_below": 0.0,
            "order_block_bull": 0.0,
            "order_block_bear": 0.0,
        }

        if len(close) < 80:
            return out

        current_price = float(close[-1])
        if current_price <= 0 or not np.isfinite(current_price):
            return out

        lookback = min(220, len(close))
        h = np.asarray(high[-lookback:], dtype=np.float64)
        l = np.asarray(low[-lookback:], dtype=np.float64)
        c = np.asarray(close[-lookback:], dtype=np.float64)

        atr = self._structure_atr(h, l, c, period=14)

        eps_pivot_break = max(0.25 * atr, current_price * 0.0012)
        eps_liq = max(0.15 * atr, current_price * 0.0010)
        eps_ob_prox = max(0.30 * atr, current_price * 0.0015)

        piv_hi, piv_lo = self._find_fractal_pivots(h, l, left=3, right=3)

        # 1) Structure trend/strength
        if len(piv_hi) >= 2 and len(piv_lo) >= 2:
            (_, h1), (_, h2) = piv_hi[-2], piv_hi[-1]
            (_, l1), (_, l2) = piv_lo[-2], piv_lo[-1]

            hh = h2 > h1 + 0.05 * atr
            hl = l2 > l1 + 0.05 * atr
            ll = l2 < l1 - 0.05 * atr
            lh = h2 < h1 - 0.05 * atr

            if hh and hl:
                trend = 1.0
            elif ll and lh:
                trend = -1.0
            else:
                trend = 0.0

            # Strength: average pivot move in ATR units, scaled to [0,1]
            dh = abs(h2 - h1) / max(atr, 1e-8)
            dl = abs(l2 - l1) / max(atr, 1e-8)
            avg_move = 0.5 * (dh + dl)
            strength = float(np.clip(avg_move / 3.0, 0.0, 1.0))

            out["structure_trend"] = float(trend)
            out["structure_strength"] = float(strength)

        # 2) BOS via close break
        if len(piv_hi) >= 1 and len(piv_lo) >= 1:
            last_hi = float(piv_hi[-1][1])
            last_lo = float(piv_lo[-1][1])

            if c[-1] > last_hi + eps_pivot_break:
                mag = (c[-1] - (last_hi + eps_pivot_break)) / max(atr, 1e-8)
                out["bos_signal"] = float(np.clip(mag, 0.0, 1.0))
            elif c[-1] < last_lo - eps_pivot_break:
                mag = ((last_lo - eps_pivot_break) - c[-1]) / max(atr, 1e-8)
                out["bos_signal"] = float(-np.clip(mag, 0.0, 1.0))

        # 3) Liquidity pools: equal-high/low clusters
        hi_levels = [p for _, p in piv_hi]
        lo_levels = [p for _, p in piv_lo]
        hi_clusters = self._cluster_levels_1d(hi_levels, eps_liq)
        lo_clusters = self._cluster_levels_1d(lo_levels, eps_liq)

        def liq_score(level: float, count: int) -> float:
            if count < 2:
                return 0.0
            dist = abs(level - current_price)
            if dist > 3.0 * atr:
                return 0.0
            prox = float(np.exp(-dist / max(1.5 * atr, 1e-8)))
            depth = float(np.clip((count - 1) / 2.0, 0.0, 1.0))
            return float(np.clip(prox * depth, 0.0, 1.0))

        best_above = 0.0
        for lvl, ct in hi_clusters:
            if lvl > current_price:
                best_above = max(best_above, liq_score(float(lvl), int(ct)))
        out["liquidity_above"] = float(best_above)

        best_below = 0.0
        for lvl, ct in lo_clusters:
            if lvl < current_price:
                best_below = max(best_below, liq_score(float(lvl), int(ct)))
        out["liquidity_below"] = float(best_below)

        # 4) Order blocks: displacement + unmitigated zone - VECTORIZED
        if len(c) >= 20:
            if open_ is not None and len(open_) >= lookback:
                o = np.asarray(open_[-lookback:], dtype=np.float64)
            else:
                o = np.concatenate([[c[0]], c[:-1]])

            body = np.abs(c - o)
            disp_thresh = 0.9 * atr  # meaningful displacement

            # Vectorized: find all displacement candles at once
            n_bars = len(c)
            
            # Bullish displacement: c > o, body >= thresh, c > c[i-1], prior candle bearish
            bull_disp = np.zeros(n_bars, dtype=bool)
            bull_disp[2:-2] = (
                (c[2:-2] > o[2:-2]) & 
                (body[2:-2] >= disp_thresh) & 
                (c[2:-2] > c[1:-3]) &
                (c[1:-3] < o[1:-3])  # prior bearish
            )
            
            # Bearish displacement
            bear_disp = np.zeros(n_bars, dtype=bool)
            bear_disp[2:-2] = (
                (c[2:-2] < o[2:-2]) & 
                (body[2:-2] >= disp_thresh) & 
                (c[2:-2] < c[1:-3]) &
                (c[1:-3] > o[1:-3])  # prior bullish
            )
            
            # Only check mitigation for candidate OBs (much fewer iterations)
            bull_indices = np.where(bull_disp)[0]
            bear_indices = np.where(bear_disp)[0]
            
            best_bull = 0.0
            for i in bull_indices[-5:]:  # Only check last 5 candidates for speed
                ob_low, ob_high = float(l[i-1]), float(h[i-1])
                fut_l, fut_h = l[i+1:], h[i+1:]
                if len(fut_l) == 0 or not np.any((fut_l <= ob_high) & (fut_h >= ob_low)):
                    dist = max(0, ob_low - current_price) if current_price < ob_low else (
                           max(0, current_price - ob_high) if current_price > ob_high else 0)
                    prox = float(np.exp(-dist / max(eps_ob_prox, 1e-8)))
                    best_bull = max(best_bull, prox)
            
            best_bear = 0.0
            for i in bear_indices[-5:]:  # Only check last 5 candidates
                ob_low, ob_high = float(l[i-1]), float(h[i-1])
                fut_l, fut_h = l[i+1:], h[i+1:]
                if len(fut_l) == 0 or not np.any((fut_l <= ob_high) & (fut_h >= ob_low)):
                    dist = max(0, ob_low - current_price) if current_price < ob_low else (
                           max(0, current_price - ob_high) if current_price > ob_high else 0)
                    prox = float(np.exp(-dist / max(eps_ob_prox, 1e-8)))
                    best_bear = max(best_bear, prox)

            out["order_block_bull"] = float(np.clip(best_bull, 0.0, 1.0))
            out["order_block_bear"] = float(np.clip(best_bear, 0.0, 1.0))

        return out
