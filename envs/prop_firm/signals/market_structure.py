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
        Fractal pivots (strict uniqueness).
        Returns lists of (index, price) in ascending index order.
        """
        h = np.asarray(high, dtype=np.float64)
        l = np.asarray(low, dtype=np.float64)

        n = int(len(h))
        if n < left + right + 3:
            return [], []

        piv_hi: List[Tuple[int, float]] = []
        piv_lo: List[Tuple[int, float]] = []

        for i in range(left, n - right):
            window_h = h[i - left : i + right + 1]
            window_l = l[i - left : i + right + 1]

            hi = float(h[i])
            lo = float(l[i])

            if hi == float(np.max(window_h)) and int(np.sum(window_h == hi)) == 1:
                piv_hi.append((i, hi))
            if lo == float(np.min(window_l)) and int(np.sum(window_l == lo)) == 1:
                piv_lo.append((i, lo))

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
        Count validated rejections:
        - price touches the level (range overlap within eps_touch)
        - does not break beyond eps_break in the forward window
        - then moves away by move_away (close confirmation) within fwd bars

        cooldown: bars to skip after counting a rejection (prevents overcounting chop).
        """
        h = np.asarray(highs, dtype=np.float64)
        l = np.asarray(lows, dtype=np.float64)
        c = np.asarray(closes, dtype=np.float64)

        n = int(len(c))
        if n < fwd + 2:
            return 0

        cd = int(cooldown) if cooldown is not None else int(fwd)
        rej = 0
        i = 0

        while i < n - fwd - 1:
            # Touch defined as bar range overlapping the level band
            touched = (l[i] - eps_touch) <= level <= (h[i] + eps_touch)
            if not touched:
                i += 1
                continue

            f_hi = float(np.max(h[i + 1 : i + 1 + fwd]))
            f_lo = float(np.min(l[i + 1 : i + 1 + fwd]))
            f_cl = c[i + 1 : i + 1 + fwd]

            if side == "support":
                # Broken if forward lows pierce below level - eps_break
                if f_lo < (level - eps_break):
                    i += 1
                    continue
                # Valid rejection if any forward close is above level + move_away
                if np.any(f_cl > (level + move_away)):
                    rej += 1
                    i += cd
                    continue
            else:
                # Broken if forward highs pierce above level + eps_break
                if f_hi > (level + eps_break):
                    i += 1
                    continue
                # Valid rejection if any forward close is below level - move_away
                if np.any(f_cl < (level - move_away)):
                    rej += 1
                    i += cd
                    continue

            i += 1

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
        Institutional-grade S/R proximity with rejection validation.

        Returns: near_support, near_resistance in [0, 1]
        """
        if len(close) < 60:
            return 0.0, 0.0

        current_price = float(close[-1])
        if current_price <= 0 or not np.isfinite(current_price):
            return 0.0, 0.0

        lookback = min(180, len(close))
        h = np.asarray(high[-lookback:], dtype=np.float64)
        l = np.asarray(low[-lookback:], dtype=np.float64)
        c = np.asarray(close[-lookback:], dtype=np.float64)

        atr = self._structure_atr(h, l, c, period=14)

        # ATR-scaled epsilons
        eps_cluster = max(0.15 * atr, current_price * 0.0010)
        eps_touch   = max(0.10 * atr, current_price * 0.0008)
        eps_break   = max(0.20 * atr, current_price * 0.0012)
        move_away   = max(0.35 * atr, current_price * 0.0015)
        eps_prox    = max(0.25 * atr, current_price * 0.0012)

        piv_hi, piv_lo = self._find_fractal_pivots(h, l, left=3, right=3)

        res_levels = [p for _, p in piv_hi]
        sup_levels = [p for _, p in piv_lo]

        res_clusters = self._cluster_levels_1d(res_levels, eps_cluster)
        sup_clusters = self._cluster_levels_1d(sup_levels, eps_cluster)

        validated_res: List[Tuple[float, float]] = []
        for level, base_ct in res_clusters:
            rej = self._count_rejections(
                float(level), h, l, c,
                side="resistance",
                eps_touch=eps_touch, eps_break=eps_break,
                move_away=move_away, fwd=3, cooldown=3,
            )
            strength = 0.35 * min(base_ct / 4.0, 1.0) + 0.65 * min(rej / 3.0, 1.0)
            if rej >= 1 and (base_ct + rej) >= 3:
                validated_res.append((float(level), float(np.clip(strength, 0.0, 1.0))))

        validated_sup: List[Tuple[float, float]] = []
        for level, base_ct in sup_clusters:
            rej = self._count_rejections(
                float(level), h, l, c,
                side="support",
                eps_touch=eps_touch, eps_break=eps_break,
                move_away=move_away, fwd=3, cooldown=3,
            )
            strength = 0.35 * min(base_ct / 4.0, 1.0) + 0.65 * min(rej / 3.0, 1.0)
            if rej >= 1 and (base_ct + rej) >= 3:
                validated_sup.append((float(level), float(np.clip(strength, 0.0, 1.0))))

        validated_res.sort(key=lambda x: -x[1])
        validated_sup.sort(key=lambda x: -x[1])
        validated_res = validated_res[:4]
        validated_sup = validated_sup[:4]

        def prox_score(level: float, strength: float, side: str) -> float:
            dist = abs(current_price - level)
            if dist > 3.0 * eps_prox:
                return 0.0

            # Side constraint
            if side == "support" and current_price < (level - eps_touch):
                return 0.0
            if side == "resistance" and current_price > (level + eps_touch):
                return 0.0

            p = float(np.exp(-dist / max(eps_prox, 1e-8)))
            return float(np.clip(p * (0.5 + 0.5 * strength), 0.0, 1.0))

        near_support = 0.0
        for level, strength in validated_sup:
            near_support = max(near_support, prox_score(level, strength, "support"))

        near_resistance = 0.0
        for level, strength in validated_res:
            near_resistance = max(near_resistance, prox_score(level, strength, "resistance"))

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

        # 4) Order blocks: displacement + unmitigated zone overlap test
        if len(c) >= 20:
            if open_ is not None and len(open_) >= lookback:
                o = np.asarray(open_[-lookback:], dtype=np.float64)
            else:
                o = np.concatenate([[c[0]], c[:-1]])

            body = np.abs(c - o)
            disp_thresh = 0.9 * atr  # meaningful displacement

            bull_scores: List[float] = []
            bear_scores: List[float] = []

            for i in range(2, len(c) - 2):
                # Bullish displacement candle
                if (c[i] > o[i]) and (body[i] >= disp_thresh) and (c[i] > c[i - 1]):
                    if c[i - 1] < o[i - 1]:  # prior bearish candle = candidate bull OB
                        ob_low = float(l[i - 1])
                        ob_high = float(h[i - 1])

                        fut_l = l[i + 1 :]
                        fut_h = h[i + 1 :]

                        # Mitigated if any future candle overlaps zone
                        overlap = np.any((fut_l <= ob_high) & (fut_h >= ob_low))
                        if not overlap:
                            # Proximity score to zone
                            if current_price < ob_low:
                                dist = ob_low - current_price
                            elif current_price > ob_high:
                                dist = current_price - ob_high
                            else:
                                dist = 0.0
                            prox = float(np.exp(-dist / max(eps_ob_prox, 1e-8)))
                            bull_scores.append(prox)

                # Bearish displacement candle
                if (c[i] < o[i]) and (body[i] >= disp_thresh) and (c[i] < c[i - 1]):
                    if c[i - 1] > o[i - 1]:  # prior bullish candle = candidate bear OB
                        ob_low = float(l[i - 1])
                        ob_high = float(h[i - 1])

                        fut_l = l[i + 1 :]
                        fut_h = h[i + 1 :]

                        overlap = np.any((fut_l <= ob_high) & (fut_h >= ob_low))
                        if not overlap:
                            if current_price < ob_low:
                                dist = ob_low - current_price
                            elif current_price > ob_high:
                                dist = current_price - ob_high
                            else:
                                dist = 0.0
                            prox = float(np.exp(-dist / max(eps_ob_prox, 1e-8)))
                            bear_scores.append(prox)

            out["order_block_bull"] = float(np.clip(max(bull_scores) if bull_scores else 0.0, 0.0, 1.0))
            out["order_block_bear"] = float(np.clip(max(bear_scores) if bear_scores else 0.0, 0.0, 1.0))

        return out
