# modules/news_sentiment.py

import numpy as np
from typing import Optional
from ..core.core import Module

class NewsSentimentModule(Module):
    """
    Toggleable module for external news/sentiment input.
    - Designed as a stub: starts as a dummy module, can later plug in real data/API.
    - When enabled, will fetch latest sentiment for the symbol(s) and push as obs.
    """

    def __init__(self, enabled: bool = False, default_sentiment: float = 0.0, debug: bool = False):
        self.enabled = enabled
        self.default_sentiment = default_sentiment
        self.latest_sentiment = default_sentiment
        self.debug = debug

    def reset(self):
        self.latest_sentiment = self.default_sentiment

    def step(self, symbol: Optional[str] = None, **kwargs):
        if not self.enabled:
            # Stub value, no data
            self.latest_sentiment = self.default_sentiment
            return

        # Here: insert your real news/sentiment API fetch logic
        # Example:
        # if symbol:
        #     self.latest_sentiment = get_sentiment_for(symbol)  # e.g., from API
        # else:
        #     self.latest_sentiment = ... (avg, or zeros)

        if self.debug:
            print(f"[NewsSentiment] (stub) Sentiment for {symbol}: {self.latest_sentiment}")

    def set_sentiment(self, value: float):
        """Manually set sentiment value, for tests or UI input."""
        self.latest_sentiment = float(value)

    def get_observation_components(self) -> np.ndarray:
        # Outputs a 1D float: sentiment score (-1 to +1 recommended, 0=neutral)
        return np.array([self.latest_sentiment], dtype=np.float32)
