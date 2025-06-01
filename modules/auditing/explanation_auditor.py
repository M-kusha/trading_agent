# modules/explanation_auditor.py

import numpy as np
from typing import Dict, Any, List, Optional
from modules.core.core import Module

class TradeExplanationAuditor(Module):
    """
    Captures, stores, and provides a summary of the rationale for every trade.
    Supports export for logs, UI, and LLMs.
    """

    def __init__(self, history_len: int = 100, debug: bool = False):
        self.history_len = history_len
        self.debug = debug
        self.reset()

    def reset(self):
        self.explanations: List[Dict[str, Any]] = []

    def step(
        self,
        action: Optional[Any] = None,
        reasoning: Optional[str] = None,
        confidence: Optional[float] = None,
        regime: Optional[str] = None,
        voting: Optional[Dict[str, float]] = None,
        pnl: Optional[float] = None,
        **kwargs
    ):
        """Store explanation after every trade or at each episode step."""
        exp = {
            "action": action,
            "reasoning": reasoning,
            "confidence": confidence,
            "regime": regime,
            "voting": voting,
            "pnl": pnl
        }
        self.explanations.append(exp)
        if len(self.explanations) > self.history_len:
            self.explanations.pop(0)
        if self.debug:
            print(f"[ExplanationAuditor] Added explanation: {exp}")

    def get_last_explanation(self) -> Dict[str, Any]:
        """Get the latest trade's explanation for UI/LLM."""
        return self.explanations[-1] if self.explanations else {}

    def get_recent_explanations(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent n explanations."""
        return self.explanations[-n:]

    def get_observation_components(self) -> np.ndarray:
        # For obs, can return e.g. confidence of last trade, or zeros if not available
        if self.explanations and self.explanations[-1].get("confidence") is not None:
            return np.array([self.explanations[-1]["confidence"]], dtype=np.float32)
        return np.zeros(1, dtype=np.float32)
    
    def get_state(self):
        return {"explanations": self.explanations.copy()}

    def set_state(self, state):
        self.explanations = state.get("explanations", []).copy()
