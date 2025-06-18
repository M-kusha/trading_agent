# modules/position/position_manager.py
"""
Fully merged PositionManager + smart exit logic, now with tunable thresholds
moved into the constructor for easy tuning.
"""

import numpy as np
import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

# --- Optional live-broker connector -----------------------------------------
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # still works in back-test / unit-test mode

from modules.trading_modes.trading_mode import TradingModeManager


class PositionManager:
    """
    Advanced Position Manager with full audit, rationale logging, and explainability.
    Integrates live‐broker syncing, hard‐loss & trailing‐profit exits, and
    position sizing for both backtest and live.
    """

    def __init__(
        self,
        initial_balance: float,
        instruments: List[str],
        max_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        loss_reduction: float = 0.2,
        max_instrument_concentration: float = 0.25,
        min_volatility: float = 0.015,
        # Tunable exit thresholds:
        hard_loss_eur: float = 30.0,
        trail_pct: float = 0.10,
        trail_abs_eur: float = 10.0,
        pips_tolerance: int = 20,
        debug: bool = True,
    ):
        self.mode_manager = TradingModeManager(initial_mode="safe", window=50)
        self.initial_balance = float(initial_balance)
        self.instruments = instruments
        self.default_max_pct = float(max_pct)
        self.max_pct = float(max_pct)
        self.debug = debug

        # Loss‐streak breaker
        self.consecutive_losses = 0
        self.max_consecutive_losses = int(max_consecutive_losses)
        self.loss_reduction = float(loss_reduction)

        # Concentration & volatility floor
        self.max_instrument_concentration = float(max_instrument_concentration)
        self.min_volatility = float(min_volatility)

        # Exit thresholds (constructor parameters)
        self.hard_loss_eur = float(hard_loss_eur)
        self.trail_pct = float(trail_pct)
        self.trail_abs_eur = float(trail_abs_eur)
        self.pips_tolerance = int(pips_tolerance)

        self.open_positions: Dict[str, Dict[str, float]] = {}
        self.env: Optional[Any] = None  # to be set externally

        # Internals for audit/explanation
        self._forced_action = None
        self._forced_conf = None
        self.last_rationale: Dict[str, Any] = {}
        self.last_confidence_components: Dict[str, Any] = {}

        # Professional logger
        self.logger = logging.getLogger("PositionManager")
        if not self.logger.handlers:
            handler = logging.FileHandler("logs/position_manager.log")
            handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)

    # ------------- Evolutionary Logic ----------------------
    def mutate(self, std: float = 0.05):
        self.max_pct += np.random.normal(0, std)
        self.max_pct = np.clip(self.max_pct, 0.01, 0.25)
        self.max_instrument_concentration += np.random.normal(0, std)
        self.max_instrument_concentration = np.clip(
            self.max_instrument_concentration, 0.05, 0.5
        )
        self.loss_reduction += np.random.normal(0, std)
        self.loss_reduction = np.clip(self.loss_reduction, 0.05, 1.0)
        self.min_volatility += np.random.normal(0, std)
        self.min_volatility = np.clip(self.min_volatility, 0.001, 0.10)
        self.max_consecutive_losses += int(np.random.choice([-1, 0, 1]))
        self.max_consecutive_losses = int(np.clip(self.max_consecutive_losses, 1, 20))
        if self.debug:
            self.logger.info("[PositionManager] Mutated parameters.")

    def crossover(self, other: "PositionManager"):
        child = copy.deepcopy(self)
        for attr in [
            "max_pct",
            "max_instrument_concentration",
            "loss_reduction",
            "min_volatility",
            "max_consecutive_losses",
            # thresholds remain as in self by default
        ]:
            if np.random.rand() > 0.5:
                setattr(child, attr, getattr(other, attr))
        if self.debug:
            self.logger.info("[PositionManager] Crossover complete.")
        return child

    # ------------------------------------------------------

    def set_env(self, env: Any):
        self.env = env

    def reset(self):
        self.max_pct = self.default_max_pct
        self.consecutive_losses = 0
        self.open_positions.clear()
        self._forced_action = None
        self._forced_conf = None
        self.last_rationale.clear()
        self.last_confidence_components.clear()
        if self.max_pct < 1e-5:
            self.logger.warning(
                f"[PositionManager] max_pct was {self.max_pct:.6f} at reset, restoring to default {self.default_max_pct:.4f}"
            )
        self.max_pct = self.default_max_pct

    def step(self, **kwargs):
        """Call this once per bar/tick."""
        env = kwargs.get("env", None)
        if env:
            self.env = env
        
        min_cap = 0.01  # 1% minimal allocation; adjust as needed
        if self.max_pct < min_cap:
            self.logger.warning(
                f"[PositionManager] max_pct={self.max_pct:.6f} below min_cap={min_cap:.4f} – restoring"
            )
            self.max_pct = self.default_max_pct


        # Live‐mode: sync & apply exit rules
        if self.env and getattr(self.env, "live_mode", False):
            self._sync_live_positions()
            self._apply_exit_rules()

        if self.debug:
            self.logger.debug(
                f"Step | Open positions: {self.open_positions}, "
                f"Consecutive losses: {self.consecutive_losses}, Max %: {self.max_pct}"
            )

    # ────────────────────────────────────────────────────────
    #              Live‐sync positions from broker
    # ────────────────────────────────────────────────────────
    def _sync_live_positions(self):
        broker_positions: List[Dict[str, Any]] = []

        # 1) env.broker
        if self.env and hasattr(self.env, "broker") and self.env.broker is not None:
            try:
                broker_positions = self.env.broker.get_positions()
            except Exception as exc:
                self.logger.warning("env.broker.get_positions failed: %s", exc)

        # 2) MT5 fallback
        elif mt5 is not None:
            raw = mt5.positions_get() or []
            for p in raw:
                broker_positions.append(
                    dict(
                        instrument=f"{p.symbol[:3]}/{p.symbol[3:]}",
                        ticket=p.ticket,
                        side=1 if p.type == mt5.POSITION_TYPE_BUY else -1,
                        lots=p.volume,
                        price_open=p.price_open,
                    )
                )

        if not broker_positions:
            return

        new_positions: Dict[str, Dict[str, Any]] = {}
        for pos in broker_positions:
            inst = pos["instrument"]
            d = {
                "ticket": pos["ticket"],
                "side": pos["side"],
                "lots": pos["lots"],
                "price_open": pos["price_open"],
                "peak_profit": self.open_positions.get(inst, {}).get("peak_profit", 0.0),
                "size": pos["lots"],
            }
            new_positions[inst] = d

        self.open_positions = new_positions
        if self.debug:
            self.logger.debug(
                f"Synced live positions: {self.open_positions}"
            )

    # ──────────────────────────────────────────────────────────────
    #     Exit logic – hard‐loss & hybrid trailing‐profit
    # ──────────────────────────────────────────────────────────────
    def _apply_exit_rules(self):
        for inst, data in list(self.open_positions.items()):
            pnl_eur, _ = self._calc_unrealised_pnl(inst, data)
            # update peak
            if pnl_eur > data["peak_profit"]:
                data["peak_profit"] = pnl_eur

            # hard‐loss
            if pnl_eur <= -self.hard_loss_eur:
                self._close_position(inst, "hard_loss")
                continue

            # trailing‐profit
            if data["peak_profit"] > 0:
                drawdown_eur = data["peak_profit"] - pnl_eur
                trigger = max(
                    data["peak_profit"] * self.trail_pct,
                    self.trail_abs_eur
                )
                if drawdown_eur >= trigger:
                    self._close_position(inst, "trail_stop")

    def _close_position(self, inst: str, reason: str):
        # env.broker
        if self.env and getattr(self.env, "broker", None):
            ok = self.env.broker.close_position(inst, comment=reason)
            if ok:
                self.logger.info("Closed %s via env.broker (%s)", inst, reason)
                self.open_positions.pop(inst, None)
            else:
                self.logger.error("env.broker.close_position failed for %s (%s)", inst, reason)
            return

        # MT5 close
        if mt5 is not None:
            data = self.open_positions[inst]
            side = data["side"]
            lots = data["lots"]
            sym = inst.replace("/", "")
            tick = mt5.symbol_info_tick(sym)
            price = (tick.bid if side > 0 else tick.ask) if tick else 0.0

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       sym,
                "volume":       lots,
                "type":         (mt5.ORDER_TYPE_SELL if side > 0 else mt5.ORDER_TYPE_BUY),
                "price":        price,
                "deviation":    self.pips_tolerance,
                "position":     data["ticket"],
                "magic":        10001,
                "comment":      f"auto-exit:{reason}",
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_FOK,
            }
            res = mt5.order_send(request)
            if res.retcode == mt5.TRADE_RETCODE_DONE:
                self.logger.info("Closed %s ticket %d (%s)", inst, data["ticket"], reason)
                self.open_positions.pop(inst, None)
            else:
                self.logger.error("order_send close failed for %s: %s", inst, res)
            return

        # backtest fallback
        self.logger.info("Marked %s closed (sim) – %s", inst, reason)
        self.open_positions.pop(inst, None)

    def _calc_unrealised_pnl(
        self,
        inst: str,
        data: Dict[str, Any],
    ) -> Tuple[float, float]:
        sym = inst.replace("/", "")
        # price
        price = None
        if self.env and getattr(self.env, "broker", None):
            price = self.env.broker.get_price(sym, side=data["side"])
        elif mt5 is not None:
            tick = mt5.symbol_info_tick(sym)
            price = tick.bid if data["side"] > 0 else tick.ask
        if price is None or not np.isfinite(price):
            return 0.0, 0.0

        # contract size
        contract_size = 100_000
        if mt5 is not None:
            info = mt5.symbol_info(sym)
            if info and info.trade_contract_size:
                contract_size = info.trade_contract_size

        points = (price - data["price_open"]) * data["side"]
        pnl_eur = points * contract_size * data["lots"]
        pnl_pct = pnl_eur / (abs(data["price_open"]) * contract_size * data["lots"])
        return float(pnl_eur), float(pnl_pct)

    # ────────────────────────────────────────────────────────────────
    #  Sizing engine  —  called by env._execute_trade()
    # ────────────────────────────────────────────────────────────────
    def calculate_size(
        self,
        volatility: float,
        intensity: float,
        balance: float,
        drawdown: float,
        correlation: Optional[float] = None,
        current_exposure: Optional[float] = None,
    ) -> float:

        # ---------- 1) basic sanitising --------------------------------
        volatility = float(np.nan_to_num(volatility, nan=self.min_volatility))
        intensity  = float(np.nan_to_num(intensity,  nan=0.0))
        drawdown   = float(np.nan_to_num(drawdown,   nan=0.0))

        vol   = max(volatility, self.min_volatility)
        inten = np.clip(intensity, -1.0, 1.0)

        # ---------- 2) risk budget -------------------------------------
        pct        = max(self.max_pct, getattr(self, "min_risk", 0.05))
        risk_cap   = balance * pct
        vol_adj    = risk_cap / vol
        raw_size   = inten * vol_adj

        # ---------- 3) mode-dependent DD throttle ----------------------
        cur_mode  = getattr(self.mode_manager, "current_mode", "safe")
        dd_factor = 1.0 if cur_mode == "safe" else 1.0 - np.clip(drawdown, 0.0, 0.8)
        raw_size *= dd_factor

        # ---------- 4) correlation penalty -----------------------------
        corr      = correlation if correlation is not None \
                    else (self.env.get_current_correlation() if self.env else 0.0)
        corr_pen  = 1.0 - min(abs(corr) * 0.5, 1.0)
        raw_size *= corr_pen

        # ---------- 5) absolute cap ------------------------------------
        max_allow = balance * pct
        size      = float(np.clip(raw_size, -max_allow, max_allow))

        # ---------- 6) practical floor ---------------------------------
        min_size = 0.01 * balance
        if abs(size) < min_size and abs(inten) > 0.3:
            size = np.sign(size or inten) * min_size

        # ---------- 7) loss-streak haircut -----------------------------
        if self.consecutive_losses >= self.max_consecutive_losses:
            size *= self.loss_reduction
            self.logger.warning(
                f"Loss streak triggered reduction: {self.consecutive_losses} "
                f"losses. Size reduced to {size:.2f}"
            )

        # ---------- 8) portfolio concentration -------------------------
        if current_exposure is None:
            total_expo = 0.0
            for v in self.open_positions.values():
                if "size" in v:                       # sim / back-test
                    total_expo += abs(v["size"])
                else:                                 # live: lots × price × 100 k
                    total_expo += abs(v["lots"]) * v["price_open"] * 100_000
            expo = total_expo / max(balance, 1.0)
        else:
            expo = current_exposure

        if expo > self.max_instrument_concentration:
            self.logger.info(
                f"Concentration penalty: {expo:.2%} > cap  zeroing size."
            )
            size = 0.0

        return float(np.nan_to_num(size, nan=0.0, posinf=0.0, neginf=0.0))


    def propose_action(self, obs: Any) -> np.ndarray:
        if self._forced_action is not None:
            return np.array(
                [self._forced_action] * len(self.instruments) * 2, dtype=np.float32
            )
        signals: List[float] = []
        for inst in self.instruments:
            raw_inten = (
                self.env.meta_agent.get_intensity(inst)
                if self.env and hasattr(self.env, "meta_agent")
                else 0.0
            )
            inten = float(np.clip(raw_inten, -1.0, 1.0))
            duration = 1.0
            if self.debug:
                self.logger.debug(f"Propose action: {inst}: inten={inten:.3f}, dur={duration}")
            signals.extend([inten, duration])
        return np.array(signals, dtype=np.float32)

    def confidence(self, obs: Any) -> float:
        if self._forced_conf is not None:
            return float(self._forced_conf)

        dd = getattr(self.env, "current_drawdown", 0.0) if self.env else 0.0
        dd_pen = max(0.0, 1.0 - dd * 1.5)

        vols = [p.get("volatility", self.min_volatility)
        for p in self.open_positions.values()]

        avg_vol = float(np.mean(vols)) if vols else self.min_volatility
        vol_pen = 1.0 - np.clip(avg_vol / 0.05, 0.0, 1.0)

        tot_sz = sum(abs(p["size"]) for p in self.open_positions.values())
        conc = tot_sz / max(self.env.balance if self.env and hasattr(self.env, "balance") else 1.0, 1.0)
        conc_pen = 1.0 - np.clip(conc / 0.5, 0.0, 1.0)

        perf = (
            self.env.meta_agent.get_observation_components()[0]
            if self.env and hasattr(self.env, "meta_agent")
            else 1.0
        )
        perf_boost = np.clip(perf * 2.0, 0.5, 1.5)

        liq = (
            self.env.liquidity_layer.current_score()
            if self.env and hasattr(self.env, "liquidity_layer")
            else 1.0
        )

        score = dd_pen * vol_pen * perf_boost * liq * conc_pen
        self.last_confidence_components = {
            "drawdown_penalty": dd_pen,
            "vol_penalty": vol_pen,
            "conc_penalty": conc_pen,
            "perf_boost": perf_boost,
            "liquidity": liq,
            "score": score,
        }
        if self.debug:
            self.logger.debug(f"Confidence components: {self.last_confidence_components}")

        return float(np.clip(score, 0.1, 1.0))

    def force_action(self, value: float):
        self._forced_action = float(value)

    def force_confidence(self, value: float):
        self._forced_conf = float(value)

    def clear_forced(self):
        self._forced_action = None
        self._forced_conf = None

    def get_observation_components(self) -> np.ndarray:
        dd = getattr(self.env, "current_drawdown", 0.0) if self.env else 0.0
        conf = self.confidence(None)
        return np.array([float(dd), float(conf)], dtype=np.float32)

    def get_last_rationale(self) -> Dict[str, Any]:
        return self.last_rationale.copy()

    def get_audit_trail(self, n: int = 20) -> List[Dict[str, Any]]:
        return self.audit_trail[-n:]

    def get_last_confidence_components(self):
        return self.last_confidence_components.copy()

    def get_full_audit(self):
        return {
            "positions": copy.deepcopy(self.open_positions),
            "last_rationale": self.get_last_rationale(),
            "last_confidence_components": self.get_last_confidence_components(),
            "consecutive_losses": self.consecutive_losses,
            "max_pct": self.max_pct,
            "max_instrument_concentration": self.max_instrument_concentration,
        }

    def get_state(self) -> Dict[str, Any]:
        return {
            "positions": copy.deepcopy(self.open_positions),
            "max_pct": float(self.max_pct),
            "consecutive_losses": int(self.consecutive_losses),
        }

    def set_state(self, state: Dict[str, Any]):
        self.open_positions = copy.deepcopy(state.get("positions", {}))
        self.max_pct = float(state.get("max_pct", self.default_max_pct))
        self.consecutive_losses = int(state.get("consecutive_losses", 0))
        if self.debug:
            self.logger.info(
                f"Restored state: max_pct={self.max_pct}, "
                f"losses={self.consecutive_losses}, positions={self.open_positions}"
            )
