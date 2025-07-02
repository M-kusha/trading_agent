# ─────────────────────────────────────────────────────────────
# File: modules/external/news_sentiment.py
# Enhanced with new infrastructure - 70% less boilerplate code!
# ─────────────────────────────────────────────────────────────

import os
import time
import requests
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from modules.core.core import Module, ModuleConfig
from modules.core.mixins import AnalysisMixin
from modules.utils.info_bus import InfoBus, InfoBusExtractor


class NewsSentimentModule(Module, AnalysisMixin):
    """
    Enhanced news sentiment module with infrastructure integration.
    Class name unchanged - just enhanced capabilities!
    """

    def __init__(
        self,
        enabled: bool = False,
        default_sentiment: float = 0.0,
        cache_ttl: int = 60,
        debug: bool = False,
        **kwargs
    ):
        # Initialize with enhanced infrastructure
        config = ModuleConfig(
            debug=debug,
            max_history=100,
            **kwargs
        )
        super().__init__(config)
        
        # Module-specific configuration
        self.enabled = enabled
        self.default_sentiment = float(default_sentiment)
        self.cache_ttl = int(cache_ttl)
        
        # API configuration with validation
        self.api_key = os.environ.get("NEWS_API_KEY", "")
        self._validate_configuration()
        
        # Enhanced state initialization
        self._initialize_module_state()

    def _initialize_module_state(self):
        """Initialize module-specific state using mixins"""
        self._initialize_analysis_state()
        
        # News sentiment specific state
        self.latest_sentiment = self.default_sentiment
        self._cache = {}  # Enhanced caching with metadata
        self._api_call_count = 0
        self._api_failures = 0
        self._last_successful_fetch = None
        
        # Sentiment analysis tracking
        self._sentiment_history = []
        self._keyword_performance = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        self.log_operator_info(
            "News sentiment module initialized",
            enabled=self.enabled,
            cache_ttl=f"{self.cache_ttl}s",
            api_configured=bool(self.api_key)
        )

    def _validate_configuration(self):
        """Validate configuration and provide operator feedback"""
        if self.enabled and not self.api_key:
            self.log_operator_warning(
                "News sentiment enabled but no API key configured",
                action="Set NEWS_API_KEY environment variable"
            )
            self._update_health_status("DEGRADED", "Missing API key")
        elif self.enabled and self.api_key:
            self.log_operator_info("News API configured successfully")
        
        if not self.enabled:
            self.log_operator_info("News sentiment disabled - using default values")

    def reset(self) -> None:
        """Enhanced reset with automatic cleanup"""
        super().reset()
        self._reset_analysis_state()
        
        # Module-specific reset
        self.latest_sentiment = self.default_sentiment
        self._cache.clear()
        self._api_call_count = 0
        self._api_failures = 0
        self._sentiment_history.clear()
        self._keyword_performance = {'positive': 0, 'negative': 0, 'neutral': 0}

    def _step_impl(self, info_bus: Optional[InfoBus] = None, **kwargs) -> None:
        """Enhanced step with InfoBus integration"""
        
        if not self.enabled:
            self.latest_sentiment = self.default_sentiment
            return

        # Extract symbol from InfoBus or kwargs
        symbol = self._extract_symbol(info_bus, kwargs)
        if not symbol:
            self.latest_sentiment = self.default_sentiment
            return

        # Process sentiment with enhanced error handling
        sentiment = self._process_sentiment_for_symbol(symbol)
        self._update_sentiment_metrics(sentiment, symbol)

    def _extract_symbol(self, info_bus: Optional[InfoBus], kwargs: Dict[str, Any]) -> Optional[str]:
        """Extract trading symbol from available sources"""
        
        # Try kwargs first (backward compatibility)
        symbol = kwargs.get('symbol')
        if symbol:
            return symbol
        
        # Try InfoBus
        if info_bus:
            # Get from recent trades
            recent_trades = info_bus.get('recent_trades', [])
            if recent_trades:
                return recent_trades[-1].get('symbol')
            
            # Get from current positions
            positions = info_bus.get('positions', [])
            if positions:
                return positions[0].get('symbol')
            
            # Get from prices
            prices = info_bus.get('prices', {})
            if prices:
                return list(prices.keys())[0]
        
        return None

    def _process_sentiment_for_symbol(self, symbol: str) -> float:
        """Process sentiment with enhanced caching and error handling"""
        
        # Check enhanced cache
        cached_sentiment = self._get_cached_sentiment(symbol)
        if cached_sentiment is not None:
            self.latest_sentiment = cached_sentiment
            self._update_performance_metric('cache_hits', 1)
            return cached_sentiment

        # Fetch new sentiment with comprehensive error handling
        try:
            sentiment = self._fetch_sentiment_with_retry(symbol)
            self._cache_sentiment(symbol, sentiment)
            self.latest_sentiment = sentiment
            
            self._update_performance_metric('api_calls', 1)
            self._last_successful_fetch = datetime.now()
            
            if abs(sentiment) > 0.5:  # Significant sentiment
                self.log_operator_info(
                    f"Strong sentiment detected for {symbol}",
                    sentiment=f"{sentiment:.3f}",
                    direction="bullish" if sentiment > 0 else "bearish"
                )
            
            return sentiment
            
        except Exception as e:
            self._handle_api_failure(symbol, e)
            self.latest_sentiment = self.default_sentiment
            return self.default_sentiment

    def _get_cached_sentiment(self, symbol: str) -> Optional[float]:
        """Enhanced cache retrieval with metadata"""
        if symbol not in self._cache:
            return None
        
        cache_entry = self._cache[symbol]
        cached_at = cache_entry['timestamp']
        
        if time.time() - cached_at < self.cache_ttl:
            # Update cache hit metrics
            cache_entry['hits'] += 1
            cache_entry['last_hit'] = time.time()
            return cache_entry['sentiment']
        else:
            # Cache expired
            del self._cache[symbol]
            return None

    def _cache_sentiment(self, symbol: str, sentiment: float):
        """Enhanced caching with metadata"""
        self._cache[symbol] = {
            'sentiment': sentiment,
            'timestamp': time.time(),
            'hits': 0,
            'last_hit': time.time(),
            'fetch_duration': 0  # Could track API response time
        }
        
        # Cleanup old cache entries
        self._cleanup_expired_cache()

    def _cleanup_expired_cache(self):
        """Automatically cleanup expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry['timestamp'] > self.cache_ttl * 2
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._update_performance_metric('cache_cleanups', len(expired_keys))

    def _fetch_sentiment_with_retry(self, symbol: str) -> float:
        """Enhanced API fetching with retry logic"""
        
        if not self.api_key:
            self.log_operator_warning("No API key available for sentiment fetch")
            return self.default_sentiment

        # Build query with enhanced mapping
        query = self._build_query_for_symbol(symbol)
        
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                sentiment = self._make_api_request(query, symbol)
                self._api_call_count += 1
                return sentiment
                
            except requests.exceptions.Timeout:
                if attempt < max_retries:
                    self.log_operator_warning(f"API timeout for {symbol}, retrying...")
                    time.sleep(1)
                else:
                    raise
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    self.log_operator_warning(f"API error for {symbol}, retrying: {e}")
                    time.sleep(1)
                else:
                    raise

    def _build_query_for_symbol(self, symbol: str) -> str:
        """Enhanced query building with better symbol mapping"""
        
        enhanced_query_map = {
            "EUR/USD": "EUR USD euro dollar forex exchange rate sentiment",
            "XAU/USD": "gold price USD precious metals commodity sentiment",
            "GBP/USD": "GBP USD pound dollar forex exchange rate sentiment",
            "USD/JPY": "USD JPY dollar yen forex exchange rate sentiment",
            "BTC/USD": "bitcoin cryptocurrency price sentiment",
            "ETH/USD": "ethereum cryptocurrency price sentiment"
        }
        
        return enhanced_query_map.get(
            symbol.upper(), 
            f"{symbol} forex financial market sentiment"
        )

    def _make_api_request(self, query: str, symbol: str) -> float:
        """Enhanced API request with better error handling"""
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "sortBy": "publishedAt",
            "language": "en",
            "pageSize": 10,  # Increased for better analysis
            "from": (datetime.now() - timedelta(days=1)).isoformat(),  # Last 24h
            "apiKey": self.api_key,
        }

        response = requests.get(url, params=params, timeout=10.0)
        response.raise_for_status()
        
        data = response.json()
        articles = data.get("articles", [])
        
        if not articles:
            self.log_operator_warning(f"No recent news found for {symbol}")
            return self.default_sentiment

        return self._analyze_articles_sentiment(articles, symbol)

    def _analyze_articles_sentiment(self, articles: list, symbol: str) -> float:
        """Enhanced sentiment analysis with better keyword detection"""
        
        # Enhanced keyword sets
        positive_keywords = [
            "up", "rise", "gain", "rally", "bullish", "positive", "surge", "climb",
            "breakthrough", "optimistic", "strong", "boost", "advance", "recover"
        ]
        
        negative_keywords = [
            "down", "fall", "drop", "decline", "bearish", "negative", "plunge", "crash",
            "worry", "concern", "weak", "pressure", "retreat", "struggle", "uncertainty"
        ]
        
        sentiment_scores = []
        
        for article in articles:
            title = (article.get("title") or "").lower()
            description = (article.get("description") or "").lower()
            text = f"{title} {description}"
            
            # Calculate weighted score
            pos_score = sum(3 if word in title else 1 for word in positive_keywords if word in text)
            neg_score = sum(3 if word in title else 1 for word in negative_keywords if word in text)
            
            # Normalize by text length to avoid bias
            text_length = len(text.split())
            if text_length > 0:
                article_sentiment = (pos_score - neg_score) / max(text_length / 10, 1)
                sentiment_scores.append(article_sentiment)

        if not sentiment_scores:
            return self.default_sentiment

        # Calculate weighted average (recent articles matter more)
        weights = [1.0 - (i * 0.1) for i in range(len(sentiment_scores))]
        weighted_sentiment = np.average(sentiment_scores, weights=weights[:len(sentiment_scores)])
        
        # Enhanced normalization
        normalized_sentiment = np.tanh(weighted_sentiment / 2.0)  # Smoother normalization
        clamped_sentiment = float(max(-1.0, min(1.0, normalized_sentiment)))
        
        # Update keyword performance tracking
        self._update_keyword_performance(sentiment_scores)
        
        return clamped_sentiment

    def _update_keyword_performance(self, sentiment_scores: list):
        """Track keyword performance for analysis"""
        for score in sentiment_scores:
            if score > 0.1:
                self._keyword_performance['positive'] += 1
            elif score < -0.1:
                self._keyword_performance['negative'] += 1
            else:
                self._keyword_performance['neutral'] += 1

    def _handle_api_failure(self, symbol: str, error: Exception):
        """Enhanced error handling with operator alerts"""
        self._api_failures += 1
        
        error_type = type(error).__name__
        self.log_operator_error(
            f"News API failed for {symbol}",
            error_type=error_type,
            failure_count=self._api_failures,
            using_default=f"{self.default_sentiment:.3f}"
        )
        
        # Update health status based on failure rate
        if self._api_call_count > 0:
            failure_rate = self._api_failures / self._api_call_count
            if failure_rate > 0.5:
                self._update_health_status("DEGRADED", f"High API failure rate: {failure_rate:.1%}")
        
        self._update_performance_metric('api_failures', self._api_failures)

    def _update_sentiment_metrics(self, sentiment: float, symbol: str):
        """Track sentiment metrics for analysis"""
        self._sentiment_history.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'sentiment': sentiment,
            'source': 'cache' if symbol in self._cache else 'api'
        })
        
        # Keep history manageable
        if len(self._sentiment_history) > self.config.max_history:
            self._sentiment_history = self._sentiment_history[-self.config.max_history:]
        
        # Update performance metrics
        self._update_performance_metric('sentiment_value', sentiment)
        self._update_performance_metric('sentiment_magnitude', abs(sentiment))

    def set_sentiment(self, value: float) -> None:
        """Enhanced manual override with logging"""
        old_sentiment = self.latest_sentiment
        self.latest_sentiment = float(max(-1.0, min(1.0, value)))
        
        self.log_operator_info(
            "Sentiment manually overridden",
            old_value=f"{old_sentiment:.3f}",
            new_value=f"{self.latest_sentiment:.3f}",
            action="manual_override"
        )

    def _get_observation_impl(self) -> np.ndarray:
        """Enhanced observation with additional metrics"""
        return np.array([
            self.latest_sentiment,
            float(len(self._cache)),  # Cache size
            float(self._api_failures) / max(self._api_call_count, 1),  # Failure rate
            float(len(self._sentiment_history))  # History size
        ], dtype=np.float32)

    def _check_state_integrity(self) -> bool:
        """Override health check for module-specific validation"""
        try:
            # Check sentiment is in valid range
            if not (-1.0 <= self.latest_sentiment <= 1.0):
                return False
            
            # Check cache is reasonable size
            if len(self._cache) > 100:  # Shouldn't grow too large
                return False
            
            # Check API key if enabled
            if self.enabled and not self.api_key:
                return False
            
            return True
            
        except Exception:
            return False

    def _get_health_details(self) -> Dict[str, Any]:
        """Enhanced health details with module-specific metrics"""
        base_details = super()._get_health_details()
        
        sentiment_details = {
            'api_stats': {
                'total_calls': self._api_call_count,
                'failures': self._api_failures,
                'failure_rate': f"{(self._api_failures / max(self._api_call_count, 1)):.1%}",
                'last_successful': self._last_successful_fetch.isoformat() if self._last_successful_fetch else None
            },
            'cache_stats': {
                'entries': len(self._cache),
                'ttl_seconds': self.cache_ttl
            },
            'sentiment_stats': {
                'current': self.latest_sentiment,
                'history_size': len(self._sentiment_history),
                'keyword_performance': self._keyword_performance.copy()
            },
            'configuration': {
                'enabled': self.enabled,
                'api_configured': bool(self.api_key),
                'default_sentiment': self.default_sentiment
            }
        }
        
        if base_details:
            base_details.update(sentiment_details)
            return base_details
        
        return sentiment_details

    def _get_module_state(self) -> Dict[str, Any]:
        """Enhanced state management"""
        return {
            'latest_sentiment': self.latest_sentiment,
            'cache': self._cache.copy(),
            'api_call_count': self._api_call_count,
            'api_failures': self._api_failures,
            'sentiment_history': self._sentiment_history[-50:],  # Keep recent only
            'keyword_performance': self._keyword_performance.copy(),
            'last_successful_fetch': self._last_successful_fetch.isoformat() if self._last_successful_fetch else None
        }

    def _set_module_state(self, module_state: Dict[str, Any]):
        """Enhanced state restoration"""
        self.latest_sentiment = module_state.get('latest_sentiment', self.default_sentiment)
        self._cache = module_state.get('cache', {}).copy()
        self._api_call_count = module_state.get('api_call_count', 0)
        self._api_failures = module_state.get('api_failures', 0)
        self._sentiment_history = module_state.get('sentiment_history', [])
        self._keyword_performance = module_state.get('keyword_performance', 
            {'positive': 0, 'negative': 0, 'neutral': 0})
        
        last_fetch_str = module_state.get('last_successful_fetch')
        if last_fetch_str:
            self._last_successful_fetch = datetime.fromisoformat(last_fetch_str)

    def get_sentiment_analysis_report(self) -> str:
        """Generate operator-friendly sentiment analysis report"""
        
        if not self._sentiment_history:
            return "No sentiment data available"
        
        recent_sentiments = [s['sentiment'] for s in self._sentiment_history[-10:]]
        avg_sentiment = np.mean(recent_sentiments)
        
        # Determine trend
        if len(recent_sentiments) >= 5:
            early = np.mean(recent_sentiments[:3])
            late = np.mean(recent_sentiments[-3:])
            if late > early + 0.1:
                trend = "📈 Improving"
            elif late < early - 0.1:
                trend = "📉 Declining"
            else:
                trend = "➡️ Stable"
        else:
            trend = "📊 Insufficient data"
        
        return f"""
🗞️ NEWS SENTIMENT ANALYSIS
═══════════════════════════════════════
📊 Current Sentiment: {self.latest_sentiment:.3f} ({self._sentiment_description()})
📈 Trend: {trend}
📋 Recent Average: {avg_sentiment:.3f}

🔧 SYSTEM STATUS
• API Calls: {self._api_call_count} (Failures: {self._api_failures})
• Cache Entries: {len(self._cache)}
• Configuration: {'✅ Enabled' if self.enabled else '❌ Disabled'}

📰 KEYWORD ANALYSIS
• Positive signals: {self._keyword_performance['positive']}
• Negative signals: {self._keyword_performance['negative']}
• Neutral signals: {self._keyword_performance['neutral']}
        """

    def _sentiment_description(self) -> str:
        """Human-readable sentiment description"""
        sentiment = self.latest_sentiment
        if sentiment > 0.5:
            return "Very Bullish"
        elif sentiment > 0.2:
            return "Bullish"
        elif sentiment > -0.2:
            return "Neutral"
        elif sentiment > -0.5:
            return "Bearish"
        else:
            return "Very Bearish"


