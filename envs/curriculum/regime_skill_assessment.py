

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List

import numpy as np

from envs.core.shared_utils import (
    clamp as _clamp,
)
from envs.core.shared_utils import (
    get_envs_logger,
)
from envs.core.shared_utils import (
    safe_float as _sf,
)

logger = get_envs_logger("regime_skill_assessment")


MIN_TRADES_PER_BUCKET = 5
MIN_TRADES_FOR_COVERAGE = 3


class VolatilityRegime(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TrendRegime(Enum):
    STRONG_TREND = "strong_trend"
    WEAK_TREND = "weak_trend"
    RANGING = "ranging"


class SessionRegime(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NY = "ny"
    LONDON_NY_OVERLAP = "overlap"
    OFF_HOURS = "off_hours"


class SpreadRegime(Enum):
    TIGHT = "tight"
    NORMAL = "normal"
    WIDE = "wide"


def _is_finite(x: float) -> bool:
    try:
        return bool(np.isfinite(float(x)))
    except Exception:
        return False


def _coerce_enum(enum_cls: Any, value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        s = value.strip().lower()

        for e in enum_cls:
            if str(e.value).lower() == s:
                return e

        for e in enum_cls:
            if str(e.name).lower() == s:
                return e
    return default


@dataclass
class TradeWithRegime:

    pnl: float = 0.0
    r_multiple: float = 0.0
    is_winner: bool = False
    bars_held: int = 0
    mae: float = 0.0
    mfe: float = 0.0


    entry_quality: float = 0.5
    exit_type: str = ""


    volatility_regime: VolatilityRegime = VolatilityRegime.MEDIUM
    trend_regime: TrendRegime = TrendRegime.RANGING
    session_regime: SessionRegime = SessionRegime.OFF_HOURS
    spread_regime: SpreadRegime = SpreadRegime.NORMAL


    atr_percentile: float = 0.5
    adx_value: float = 20.0
    spread_percentile: float = 0.5

    def __post_init__(self) -> None:

        self.pnl = float(_sf(self.pnl, 0.0))
        self.r_multiple = float(_sf(self.r_multiple, 0.0))
        self.mae = float(_sf(self.mae, 0.0))
        self.mfe = float(_sf(self.mfe, 0.0))
        self.entry_quality = float(_clamp(_sf(self.entry_quality, 0.5), 0.0, 1.0))
        self.bars_held = max(0, int(_sf(self.bars_held, 0)))

        self.atr_percentile = float(_clamp(_sf(self.atr_percentile, 0.5), 0.0, 1.0))
        self.adx_value = float(_sf(self.adx_value, 20.0))
        self.spread_percentile = float(_clamp(_sf(self.spread_percentile, 0.5), 0.0, 1.0))


        self.volatility_regime = _coerce_enum(VolatilityRegime, self.volatility_regime, VolatilityRegime.MEDIUM)
        self.trend_regime = _coerce_enum(TrendRegime, self.trend_regime, TrendRegime.RANGING)
        self.session_regime = _coerce_enum(SessionRegime, self.session_regime, SessionRegime.OFF_HOURS)
        self.spread_regime = _coerce_enum(SpreadRegime, self.spread_regime, SpreadRegime.NORMAL)

    @classmethod
    def classify_volatility(cls, atr_percentile: float) -> VolatilityRegime:
        if atr_percentile < 0.2:
            return VolatilityRegime.LOW
        elif atr_percentile > 0.8:
            return VolatilityRegime.HIGH
        return VolatilityRegime.MEDIUM

    @classmethod
    def classify_trend(cls, adx_value: float, slope: float = 0.0) -> TrendRegime:
        if adx_value > 30:
            return TrendRegime.STRONG_TREND
        elif adx_value > 20:
            return TrendRegime.WEAK_TREND
        return TrendRegime.RANGING

    @classmethod
    def classify_session(cls, hour_utc: int) -> SessionRegime:
        if 0 <= hour_utc < 7:
            return SessionRegime.ASIAN
        elif 7 <= hour_utc < 12:
            return SessionRegime.LONDON
        elif 12 <= hour_utc < 16:
            return SessionRegime.LONDON_NY_OVERLAP
        elif 16 <= hour_utc < 21:
            return SessionRegime.NY
        return SessionRegime.OFF_HOURS

    @classmethod
    def classify_spread(cls, spread_percentile: float) -> SpreadRegime:
        if spread_percentile < 0.3:
            return SpreadRegime.TIGHT
        elif spread_percentile > 0.7:
            return SpreadRegime.WIDE
        return SpreadRegime.NORMAL


@dataclass
class RegimePerformance:
    regime_name: str
    trade_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    total_r: float = 0.0
    coverage: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.win_count / max(self.trade_count, 1)

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / max(self.trade_count, 1)

    @property
    def avg_r(self) -> float:
        return self.total_r / max(self.trade_count, 1)

    def add_trade(self, trade: TradeWithRegime) -> None:

        pnl = float(_sf(trade.pnl, 0.0))
        r = float(_sf(trade.r_multiple, 0.0))
        if not _is_finite(pnl):
            pnl = 0.0
        if not _is_finite(r):
            r = 0.0

        self.trade_count += 1
        if trade.is_winner:
            self.win_count += 1
        self.total_pnl += pnl
        self.total_r += r

    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime_name,
            "trades": self.trade_count,
            "win_rate": self.win_rate,
            "avg_pnl": self.avg_pnl,
            "avg_r": self.avg_r,
            "coverage": self.coverage,
        }


@dataclass
class RegimeSkillAssessment:


    volatility_performance: Dict[VolatilityRegime, RegimePerformance] = field(default_factory=dict)
    trend_performance: Dict[TrendRegime, RegimePerformance] = field(default_factory=dict)
    session_performance: Dict[SessionRegime, RegimePerformance] = field(default_factory=dict)
    spread_performance: Dict[SpreadRegime, RegimePerformance] = field(default_factory=dict)


    adaptation_score: float = 0.5
    volatility_handling: float = 0.5
    trend_following: float = 0.5
    session_awareness: float = 0.5
    cost_resilience: float = 0.5


    regime_coverage: float = 0.0


    total_trades: int = 0
    confidence: float = 0.0

    @classmethod
    def from_trades(cls, trades: List[TradeWithRegime]) -> "RegimeSkillAssessment":
        assessment = cls()

        if not trades:
            return assessment

        assessment.total_trades = len(trades)

        base_confidence = min(1.0, len(trades) / 100.0)
        assessment.confidence = base_confidence


        for regime in VolatilityRegime:
            assessment.volatility_performance[regime] = RegimePerformance(regime.value)
        for regime in TrendRegime:
            assessment.trend_performance[regime] = RegimePerformance(regime.value)
        for regime in SessionRegime:
            assessment.session_performance[regime] = RegimePerformance(regime.value)
        for regime in SpreadRegime:
            assessment.spread_performance[regime] = RegimePerformance(regime.value)


        for trade in trades:

            v = _coerce_enum(VolatilityRegime, getattr(trade, "volatility_regime", None), VolatilityRegime.MEDIUM)
            t = _coerce_enum(TrendRegime, getattr(trade, "trend_regime", None), TrendRegime.RANGING)
            s = _coerce_enum(SessionRegime, getattr(trade, "session_regime", None), SessionRegime.OFF_HOURS)
            c = _coerce_enum(SpreadRegime, getattr(trade, "spread_regime", None), SpreadRegime.NORMAL)

            assessment.volatility_performance[v].add_trade(trade)
            assessment.trend_performance[t].add_trade(trade)
            assessment.session_performance[s].add_trade(trade)
            assessment.spread_performance[c].add_trade(trade)


        total = max(assessment.total_trades, 1)
        for perf_dict in [
            assessment.volatility_performance,
            assessment.trend_performance,
            assessment.session_performance,
            assessment.spread_performance,
        ]:
            for perf in perf_dict.values():
                perf.coverage = perf.trade_count / total


        assessment._compute_adaptation_score()
        assessment._compute_volatility_handling()
        assessment._compute_trend_following()
        assessment._compute_session_awareness()
        assessment._compute_cost_resilience()
        assessment._compute_regime_coverage()


        assessment.confidence = float(_clamp(base_confidence * (0.5 + 0.5 * assessment.regime_coverage), 0.0, 1.0))

        return assessment

    def _compute_adaptation_score(self) -> None:
        win_rates = []
        r_multiples = []
        weights = []

        for perf_dict in [
            self.volatility_performance,
            self.trend_performance,
            self.session_performance,
            self.spread_performance,
        ]:
            for perf in perf_dict.values():
                if perf.trade_count >= MIN_TRADES_PER_BUCKET:
                    win_rates.append(perf.win_rate)
                    r_multiples.append(perf.avg_r)
                    weights.append(perf.trade_count)

        if len(win_rates) < 3:
            self.adaptation_score = 0.5
            return


        wr_arr = np.array(win_rates)
        r_arr = np.array(r_multiples)
        w_arr = np.array(weights, dtype=float)
        w_arr = w_arr / w_arr.sum()


        wr_mean = float(np.average(wr_arr, weights=w_arr))
        wr_var = float(np.average((wr_arr - wr_mean) ** 2, weights=w_arr))
        wr_std = float(np.sqrt(wr_var))


        r_mean = float(np.average(r_arr, weights=w_arr))
        r_var = float(np.average((r_arr - r_mean) ** 2, weights=w_arr))
        r_std = float(np.sqrt(r_var))


        wr_cv = wr_std / max(wr_mean, 0.01)

        r_cv = r_std / max(abs(r_mean) + 0.1, 0.1)


        wr_score = _clamp(1.0 - wr_cv * 2, 0.0, 1.0)
        r_score = _clamp(1.0 - r_cv * 2, 0.0, 1.0)

        self.adaptation_score = 0.6 * wr_score + 0.4 * r_score

    def _compute_volatility_handling(self) -> None:
        high_vol = self.volatility_performance.get(VolatilityRegime.HIGH)
        low_vol = self.volatility_performance.get(VolatilityRegime.LOW)

        if not high_vol or high_vol.trade_count < MIN_TRADES_PER_BUCKET:
            self.volatility_handling = 0.5
            return


        wr_score = high_vol.win_rate

        r_score = _clamp((high_vol.avg_r + 0.5), 0.0, 1.0)

        base_score = 0.6 * wr_score + 0.4 * r_score


        if low_vol and low_vol.trade_count >= MIN_TRADES_PER_BUCKET:
            low_combined = 0.6 * low_vol.win_rate + 0.4 * _clamp((low_vol.avg_r + 0.5), 0.0, 1.0)
            if base_score >= low_combined * 0.9:
                base_score *= 1.1

        self.volatility_handling = _clamp(base_score, 0.0, 1.0)

    def _compute_trend_following(self) -> None:
        strong_trend = self.trend_performance.get(TrendRegime.STRONG_TREND)
        ranging = self.trend_performance.get(TrendRegime.RANGING)

        if not strong_trend or strong_trend.trade_count < MIN_TRADES_PER_BUCKET:
            self.trend_following = 0.5
            return


        wr_score = strong_trend.win_rate
        r_score = _clamp((strong_trend.avg_r + 0.5), 0.0, 1.0)
        base_score = 0.6 * wr_score + 0.4 * r_score

        if ranging and ranging.trade_count >= MIN_TRADES_PER_BUCKET:

            ranging_combined = 0.6 * ranging.win_rate + 0.4 * _clamp((ranging.avg_r + 0.5), 0.0, 1.0)

            if base_score > ranging_combined + 0.1:
                base_score *= 1.15

        self.trend_following = _clamp(base_score, 0.0, 1.0)

    def _compute_session_awareness(self) -> None:
        overlap = self.session_performance.get(SessionRegime.LONDON_NY_OVERLAP)
        off_hours = self.session_performance.get(SessionRegime.OFF_HOURS)

        if not overlap or overlap.trade_count < MIN_TRADES_PER_BUCKET:
            self.session_awareness = 0.5
            return


        wr_score = overlap.win_rate
        r_score = _clamp((overlap.avg_r + 0.5), 0.0, 1.0)
        base_score = 0.6 * wr_score + 0.4 * r_score


        if off_hours and off_hours.trade_count >= MIN_TRADES_PER_BUCKET:
            off_combined = 0.6 * off_hours.win_rate + 0.4 * _clamp((off_hours.avg_r + 0.5), 0.0, 1.0)
            if off_combined < 0.4 and off_hours.trade_count > overlap.trade_count:
                base_score *= 0.8

        self.session_awareness = _clamp(base_score, 0.0, 1.0)

    def _compute_cost_resilience(self) -> None:
        wide = self.spread_performance.get(SpreadRegime.WIDE)
        tight = self.spread_performance.get(SpreadRegime.TIGHT)

        if not wide or wide.trade_count < MIN_TRADES_PER_BUCKET:
            self.cost_resilience = 0.5
            return


        wr_score = wide.win_rate
        r_score = _clamp((wide.avg_r + 0.5), 0.0, 1.0)
        base_score = 0.6 * wr_score + 0.4 * r_score


        if tight and tight.trade_count >= MIN_TRADES_PER_BUCKET:
            tight_combined = 0.6 * tight.win_rate + 0.4 * _clamp((tight.avg_r + 0.5), 0.0, 1.0)
            ratio = base_score / max(tight_combined, 0.01)
            if ratio >= 0.8:
                base_score *= 1.1

        self.cost_resilience = _clamp(base_score, 0.0, 1.0)

    def _compute_regime_coverage(self) -> None:
        total_regimes = 0
        covered_regimes = 0

        for perf_dict in [
            self.volatility_performance,
            self.trend_performance,
            self.session_performance,
            self.spread_performance,
        ]:
            for perf in perf_dict.values():
                total_regimes += 1
                if perf.trade_count >= MIN_TRADES_FOR_COVERAGE:
                    covered_regimes += 1

        self.regime_coverage = covered_regimes / max(total_regimes, 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adaptation_score": self.adaptation_score,
            "volatility_handling": self.volatility_handling,
            "trend_following": self.trend_following,
            "session_awareness": self.session_awareness,
            "cost_resilience": self.cost_resilience,
            "regime_coverage": self.regime_coverage,
            "total_trades": self.total_trades,
            "confidence": self.confidence,
            "volatility_breakdown": {
                k.value: v.to_dict()
                for k, v in self.volatility_performance.items()
            },
            "trend_breakdown": {
                k.value: v.to_dict()
                for k, v in self.trend_performance.items()
            },
            "session_breakdown": {
                k.value: v.to_dict()
                for k, v in self.session_performance.items()
            },
            "spread_breakdown": {
                k.value: v.to_dict()
                for k, v in self.spread_performance.items()
            },
        }

    def get_skill_vector(self) -> Dict[str, float]:
        return {
            "adaptation": self.adaptation_score,
            "volatility_handling": self.volatility_handling,
            "trend_following": self.trend_following,
            "session_awareness": self.session_awareness,
            "cost_resilience": self.cost_resilience,
        }

    def get_weaknesses(self, threshold: float = 0.4) -> List[str]:
        weaknesses = []

        for skill, score in self.get_skill_vector().items():
            if score < threshold:
                weaknesses.append(f"Low {skill}: {score:.2f}")


        for regime, perf in self.volatility_performance.items():
            if perf.trade_count >= 5 and perf.win_rate < 0.35:
                weaknesses.append(f"Poor in {regime.value} volatility: {perf.win_rate:.0%} WR")

        for regime, perf in self.trend_performance.items():
            if perf.trade_count >= 5 and perf.win_rate < 0.35:
                weaknesses.append(f"Poor in {regime.value}: {perf.win_rate:.0%} WR")

        return weaknesses


def classify_bar_regime(
    atr: float,
    atr_history: List[float],
    adx: float,
    hour_utc: int,
    spread: float,
    spread_history: List[float],
) -> Dict[str, Any]:

    atr_hist = [float(x) for x in (atr_history or []) if _is_finite(x)]
    spr_hist = [float(x) for x in (spread_history or []) if _is_finite(x)]
    atr = float(_sf(atr, 0.0))
    adx = float(_sf(adx, 20.0))
    spread = float(_sf(spread, 0.0))


    if atr_hist and len(atr_hist) >= 10:
        atr_percentile = sum(1 for x in atr_hist if x <= atr) / len(atr_hist)
    else:
        atr_percentile = 0.5


    if spr_hist and len(spr_hist) >= 10:
        spread_percentile = sum(1 for x in spr_hist if x <= spread) / len(spr_hist)
    else:
        spread_percentile = 0.5

    return {
        "volatility": TradeWithRegime.classify_volatility(atr_percentile),
        "trend": TradeWithRegime.classify_trend(adx),
        "session": TradeWithRegime.classify_session(hour_utc),
        "spread": TradeWithRegime.classify_spread(spread_percentile),
        "atr_percentile": atr_percentile,
        "adx_value": adx,
        "spread_percentile": spread_percentile,
    }
