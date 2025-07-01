
# ─────────────────────────────────────────────────────────────
# modules/strategy/explanation_generator.py

from __future__ import annotations
import logging
import numpy as np
from modules.core.core import Module
from modules.market.fractal_regime_confirmation import FractalRegimeConfirmation
from modules.voting.strategy_arbiter import StrategyArbiter


class ExplanationGenerator(Module):
    def __init__(self, fractal_regime: FractalRegimeConfirmation, strategy_arbiter: StrategyArbiter, debug: bool = True):
        super().__init__()
        self.debug = debug
        self.last_explanation = ""
        self.trade_count = 0
        self.profit_today = 0.0
        self.fractal_regime = fractal_regime
        self.strategy_arbiter = strategy_arbiter
        self.arbiter = strategy_arbiter
        self._step_count = 0

        # Enhanced Logger Setup
        self.logger = logging.getLogger(f"ExplanationGenerator_{id(self)}")
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        
        fh = logging.FileHandler("logs/strategy/explanation_generator.log", mode='a')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        self.logger.info("ExplanationGenerator initialized")

    def reset(self) -> None:
        self.last_explanation = ""
        self.trade_count = 0
        self.profit_today = 0.0
        self._step_count = 0
        self.logger.info("ExplanationGenerator reset - all metrics cleared")

    def step(
        self,
        actions=None,
        arbiter_weights=None,
        member_names=None,
        votes=None,
        regime: str | None = None, # "unknown" for bootstrap
        volatility=None,
        drawdown=0.0,
        genome_metrics=None,
        pnl=0.0,
        target_achieved=False,
        *args, **kwargs
    ) -> None:
        """
        Robust to missing pipeline arguments and preserves full analytics with comprehensive logging.
        Accepts and safely ignores unused/extra parameters.
        """
        self._step_count += 1
        
        try:
            # Validate PnL
            if np.isnan(pnl):
                self.logger.error("NaN PnL received, setting to 0")
                pnl = 0.0

            if regime is None or str(regime).lower() == "unknown":
                self.fractal_regime.label
                self.logger.debug(f"Retrieved regime from switcher: {regime}")

            # Get expert names/weights from arbiter if not provided
            if not member_names:
                member_names = [m.__class__.__name__ for m in self.arbiter.members]
                self.logger.debug(f"Retrieved member names: {member_names}")
                
            if arbiter_weights is None or len(arbiter_weights) != len(member_names):
                if hasattr(self.arbiter, "last_alpha") and self.arbiter.last_alpha is not None:
                    arbiter_weights = np.array(self.arbiter.last_alpha, dtype=np.float32)
                else:
                    arbiter_weights = np.ones(len(member_names), dtype=np.float32)
                self.logger.debug(f"Retrieved/generated arbiter weights: {arbiter_weights}")

            # Defaults for safe operation with validation
            if actions is None or not hasattr(actions, "__len__"):
                actions = np.zeros(1, dtype=np.float32)
            if arbiter_weights is None or not hasattr(arbiter_weights, "__len__"):
                arbiter_weights = np.ones(1, dtype=np.float32)
            if not member_names or not isinstance(member_names, (list, tuple)):
                member_names = ["Unknown"]
            if votes is None or not isinstance(votes, dict):
                votes = {}
            if volatility is None or not isinstance(volatility, dict):
                volatility = {}
            if genome_metrics is None or not isinstance(genome_metrics, dict):
                genome_metrics = {}

            # Validate arbiter weights
            if np.any(np.isnan(arbiter_weights)):
                self.logger.error(f"NaN in arbiter weights: {arbiter_weights}")
                arbiter_weights = np.nan_to_num(arbiter_weights)

            self.trade_count += 1
            self.profit_today += pnl

            self.logger.debug(f"Step {self._step_count}: trade_count={self.trade_count}, profit_today=€{self.profit_today:.2f}")

            try:
                top_idx = int(np.argmax(arbiter_weights))
                top_name = member_names[top_idx] if top_idx < len(member_names) else "Unknown"
                top_w = float(arbiter_weights[top_idx]) * 100.0
            except Exception as e:
                self.logger.error(f"Error determining top strategy: {e}")
                top_idx, top_name, top_w = 0, "Unknown", 100.0

            # Aggregate votes with validation
            agg = {n: 0.0 for n in member_names}
            count = 0
            for vote_dict in votes.values():
                if isinstance(vote_dict, dict):
                    for n, w in vote_dict.items():
                        if not np.isnan(w):
                            agg[n] = agg.get(n, 0.0) + float(w)
                    count += 1
            if count:
                for n in agg:
                    agg[n] /= count
            votes_str = "; ".join(f"{n}: {agg[n] * 100.0:.1f}%" for n in list(member_names)[:3])

            # High volatility warning
            high_vol_instruments = [inst for inst, vol in volatility.items() if isinstance(vol, (float, int)) and vol > 0.02]
            vol_warning = f" ⚠️ HIGH VOL: {', '.join(high_vol_instruments)}" if high_vol_instruments else ""

            # Risk/reward calculation
            sl_base = float(genome_metrics.get("sl_base", 1.0))
            tp_base = float(genome_metrics.get("tp_base", 1.5))
            risk_reward = tp_base / sl_base if sl_base > 0 else 1.5

            # Validate drawdown
            if np.isnan(drawdown):
                self.logger.error("NaN drawdown received, setting to 0")
                drawdown = 0.0

            progress_pct = (self.profit_today / 150.0) * 100  # Against €150 target
            dd_pct = drawdown * 100.0

            self.last_explanation = (
                f"Day Progress: €{self.profit_today:.2f}/€150 ({progress_pct:.1f}%) | "
                f"Trades: {self.trade_count} | "
                f"Regime: {regime}{vol_warning} | "
                f"Strategy: {top_name} ({top_w:.0f}%) | "
                f"RR: {risk_reward:.1f}:1 | "
                f"DD: {dd_pct:.1f}%"
            )

            if target_achieved:
                self.last_explanation += "  TARGET ACHIEVED - Consider stopping"
                self.logger.info("Target achieved!")
            elif dd_pct > 5:
                self.last_explanation += "  High drawdown - Reduce position size"
                self.logger.warning(f"High drawdown detected: {dd_pct:.1f}%")
            elif progress_pct < 30 and self.trade_count > 20:
                self.last_explanation += "  Low progress - Review strategy"
                self.logger.warning(f"Low progress after {self.trade_count} trades: {progress_pct:.1f}%")

            self.logger.info(f"Generated explanation: {self.last_explanation}")

            if self.debug:
                print("[ExplanationGenerator]", self.last_explanation)

        except Exception as e:
            self.logger.error(f"Error in step: {e}")
            self.last_explanation = f"Error generating explanation: {str(e)}"

    def get_observation_components(self) -> np.ndarray:
        """Return profit metrics with validation"""
        try:
            avg_profit = self.profit_today / max(1, self.trade_count)
            
            # Validate components
            if np.isnan(self.profit_today):
                self.logger.error("NaN in profit_today")
                self.profit_today = 0.0
            if np.isnan(avg_profit):
                self.logger.error("NaN in avg_profit")
                avg_profit = 0.0
                
            observation = np.array([
                self.profit_today,
                float(self.trade_count),
                avg_profit
            ], dtype=np.float32)
            
            self.logger.debug(f"Observation: profit={self.profit_today:.2f}, trades={self.trade_count}, avg={avg_profit:.2f}")
            return observation
            
        except Exception as e:
            self.logger.error(f"Error getting observation components: {e}")
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
