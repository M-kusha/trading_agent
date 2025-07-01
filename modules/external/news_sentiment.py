# ─────────────────────────────────────────────────────────────
# File: modules/external/news_sentiment.py
# ─────────────────────────────────────────────────────────────
import os
import time
import requests
import numpy as np
from typing import Optional
from ..core.core import Module

class NewsSentimentModule(Module):

    def __init__(
        self,
        enabled: bool = False,
        default_sentiment: float = 0.0,
        cache_ttl: int = 60,
        debug: bool = False,
    ):
        self.enabled = enabled
        self.default_sentiment = float(default_sentiment)
        self.latest_sentiment = float(default_sentiment)
        self.cache_ttl = int(cache_ttl)
        self.debug = debug

        # In-memory cache for recent fetches: { symbol_str: (timestamp, sentiment_float) }
        self._cache: dict[str, tuple[float, float]] = {}

        # Expect an environment variable NEWS_API_KEY containing a valid NewsAPI.org key.
        # E.g. export NEWS_API_KEY="your_api_key_here"
        self.api_key = os.environ.get("NEWS_API_KEY", "")
        if self.enabled and not self.api_key and self.debug:
            print("[NewsSentimentModule] WARNING: enabled=True but NEWS_API_KEY not set in environment.")

    def reset(self):
        self.latest_sentiment = self.default_sentiment

    def step(self, symbol: Optional[str] = None, **kwargs):

        if not self.enabled:
            # In training/backtest mode, or user explicitly disabled, always return default (0.0).
            self.latest_sentiment = self.default_sentiment
            return

        if symbol is None:
            # If caller forgot to pass symbol, fallback.
            self.latest_sentiment = self.default_sentiment
            return

        now = time.time()
        # 1) Check cache
        cache_entry = self._cache.get(symbol)
        if cache_entry:
            fetched_at, cached_value = cache_entry
            if now - fetched_at < self.cache_ttl:
                # Use the cached sentiment
                self.latest_sentiment = cached_value
                if self.debug:
                    print(f"[NewsSentimentModule] Cache hit for {symbol}: {cached_value:.3f}")
                return

        # 2) Cache miss or expired → fetch new sentiment
        sentiment = self._fetch_sentiment_from_api(symbol)
        # Clamp to [-1.0, +1.0]
        sentiment = float(max(-1.0, min(1.0, sentiment)))
        # Store in cache
        self._cache[symbol] = (now, sentiment)
        self.latest_sentiment = sentiment

        if self.debug:
            print(f"[NewsSentimentModule] Fetched for {symbol}: {self.latest_sentiment:.3f}")

    def _fetch_sentiment_from_api(self, symbol: str) -> float:

        if not self.api_key:
            # No API key → cannot fetch → return default
            if self.debug:
                print(f"[NewsSentimentModule] No API key, returning default {self.default_sentiment}")
            return self.default_sentiment

        # Map symbol to a textual query
        # (“EUR/USD” → “EUR USD forex currency sentiment”, “XAU/USD” → “XAU USD gold sentiment”)
        query_map = {
            "EUR/USD": "EUR USD forex currency sentiment",
            "XAU/USD": "XAU USD gold spot price sentiment",
        }
        q = query_map.get(symbol.upper(), f"{symbol} forex sentiment")

        url = "https://newsapi.org/v2/everything"
        params = {
            "q":        q,
            "sortBy":   "publishedAt",
            "language": "en",
            "pageSize": 5,
            "apiKey":   self.api_key,
        }

        try:
            response = requests.get(url, params=params, timeout=5.0)
            response.raise_for_status()
            data = response.json()
            articles = data.get("articles", [])
        except Exception as e:
            if self.debug:
                print(f"[NewsSentimentModule] API request failed for {symbol}: {e}")
            return self.default_sentiment

        if not articles:
            if self.debug:
                print(f"[NewsSentimentModule] No articles returned for {symbol}")
            return self.default_sentiment

        # Simple keyword-based sentiment scoring
        pos_keywords = ["up", "rally", "gain", "rise", "bullish", "positive"]
        neg_keywords = ["down", "drop", "loss", "fall", "bearish", "negative"]

        total_score = 0.0
        count = 0
        for art in articles:
            title = (art.get("title") or "").lower()
            desc  = (art.get("description") or "").lower()
            text  = f"{title} {desc}"

            score = 0
            for w in pos_keywords:
                if w in text:
                    score += 1
            for w in neg_keywords:
                if w in text:
                    score -= 1
            total_score += score
            count += 1

        if count == 0:
            return self.default_sentiment

        avg_raw = total_score / float(count)
        normalized = max(-1.0, min(1.0, avg_raw / 5.0))
        return normalized

    def set_sentiment(self, value: float):
        """
        Manually override the sentiment (e.g., for testing or UI).
        """
        self.latest_sentiment = float(value)

    def get_observation_components(self) -> np.ndarray:
        """
        Return a 1D numpy array containing exactly one float: [ latest_sentiment ].
        """
        return np.array([self.latest_sentiment], dtype=np.float32)
