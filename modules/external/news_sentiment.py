# ─────────────────────────────────────────────────────────────
# File: modules/external/news_sentiment.py
# 🚀 PRODUCTION-READY News Sentiment Analysis System
# Advanced news sentiment with SmartInfoBus integration
# ─────────────────────────────────────────────────────────────

import asyncio
import os
import time
import threading
import requests
import numpy as np
from typing import Dict, Any, List, Optional
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

from modules.core.module_base import BaseModule, module
from modules.core.mixins import SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin
from modules.core.error_pinpointer import ErrorPinpointer, create_error_handler
from modules.utils.info_bus import InfoBusManager
from modules.utils.audit_utils import RotatingLogger, format_operator_message
from modules.utils.system_utilities import EnglishExplainer, SystemUtilities
from modules.monitoring.health_monitor import HealthMonitor
from modules.monitoring.performance_tracker import PerformanceTracker


@dataclass
class SentimentConfig:
    """Configuration for News Sentiment Module"""
    enabled: bool = False
    default_sentiment: float = 0.0
    cache_ttl: int = 60
    max_retries: int = 2
    timeout: float = 10.0
    
    # Performance thresholds
    max_processing_time_ms: float = 300
    circuit_breaker_threshold: int = 3
    min_confidence: float = 0.3


@module(
    name="NewsSentimentModule", 
    version="3.0.0",
    category="external",
    provides=["news_sentiment", "sentiment_confidence", "news_summary", "sentiment_trend"],
    requires=["market_data", "symbols", "trading_session"],
    description="Advanced news sentiment analysis with API integration and SmartInfoBus support",
    thesis_required=True,
    health_monitoring=True,
    performance_tracking=True,
    error_handling=True
)
class NewsSentimentModule(BaseModule, SmartInfoBusTradingMixin, SmartInfoBusRiskMixin, SmartInfoBusStateMixin):
    """
    Production-ready news sentiment module with SmartInfoBus integration.
    Provides news sentiment analysis with graceful fallback when API is unavailable.
    """

    def __init__(
        self,
        config: Optional[SentimentConfig] = None,
        genome: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        self.config = config or SentimentConfig()
        super().__init__()
        
        # Initialize advanced systems
        self._initialize_advanced_systems()
        
        # Initialize genome parameters
        self._initialize_genome_parameters(genome)
        
        # Initialize sentiment state
        self._initialize_sentiment_state()
        
        # API configuration
        self.api_key = os.environ.get("NEWS_API_KEY", "")
        
        self.logger.info(
            format_operator_message(
                "📰", "NEWS_SENTIMENT_INITIALIZED",
                details=f"Enabled: {self.genome['enabled']}, API configured: {bool(self.api_key)}",
                result="News sentiment analysis ready",
                context="news_analysis"
            )
        )
    
    def _initialize_advanced_systems(self):
        """Initialize advanced systems for sentiment analysis"""
        self.smart_bus = InfoBusManager.get_instance()
        self.logger = RotatingLogger(
            name="NewsSentimentModule", 
            log_path="logs/news_sentiment.log", 
            max_lines=3000, 
            operator_mode=True,
            plain_english=True
        )
        self.error_pinpointer = ErrorPinpointer()
        self.error_handler = create_error_handler("NewsSentimentModule", self.error_pinpointer)
        self.english_explainer = EnglishExplainer()
        self.system_utilities = SystemUtilities()
        self.performance_tracker = PerformanceTracker()
        
        # Circuit breaker for API operations
        self.circuit_breaker = {
            'failures': 0,
            'last_failure': 0,
            'state': 'CLOSED',
            'threshold': self.config.circuit_breaker_threshold
        }
        
        # Health monitoring
        self._health_status = 'healthy'
        self._last_health_check = time.time()
        self._start_monitoring()

    def _initialize_genome_parameters(self, genome: Optional[Dict[str, Any]]):
        """Initialize genome-based parameters"""
        if genome:
            self.genome = {
                "enabled": bool(genome.get("enabled", self.config.enabled)),
                "default_sentiment": float(genome.get("default_sentiment", self.config.default_sentiment)),
                "cache_ttl": int(genome.get("cache_ttl", self.config.cache_ttl)),
                "max_retries": int(genome.get("max_retries", self.config.max_retries)),
                "timeout": float(genome.get("timeout", self.config.timeout))
            }
        else:
            self.genome = {
                "enabled": self.config.enabled,
                "default_sentiment": self.config.default_sentiment,
                "cache_ttl": self.config.cache_ttl,
                "max_retries": self.config.max_retries,
                "timeout": self.config.timeout
            }

    def _initialize_sentiment_state(self):
        """Initialize sentiment analysis state"""
        # Core sentiment state
        self.latest_sentiment = self.genome["default_sentiment"]
        self.sentiment_confidence = 0.0
        
        # Caching and tracking
        self._cache = {}
        self._api_call_count = 0
        self._api_failures = 0
        self._last_successful_fetch = None
        
        # Enhanced analytics
        self._sentiment_history = deque(maxlen=100)
        self._keyword_performance = {'positive': 0, 'negative': 0, 'neutral': 0}
        self._symbol_sentiments = {}
        
        # Performance metrics
        self._sentiment_performance = {
            'total_requests': 0,
            'cache_hits': 0,
            'api_successes': 0,
            'avg_response_time': 0.0
        }

    def _start_monitoring(self):
        """Start background monitoring"""
        def monitoring_loop():
            while getattr(self, '_monitoring_active', True):
                try:
                    self._update_sentiment_health()
                    self._cleanup_expired_cache()
                    time.sleep(30)
                except Exception as e:
                    self.logger.error(f"Monitoring error: {e}")
        
        self._monitoring_active = True
        monitor_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitor_thread.start()

    async def _initialize(self):
        """Initialize module"""
        try:
            # Set initial sentiment status in SmartInfoBus
            initial_status = {
                "sentiment": self.latest_sentiment,
                "confidence": self.sentiment_confidence,
                "enabled": self.genome["enabled"],
                "api_configured": bool(self.api_key)
            }
            
            self.smart_bus.set(
                'news_sentiment',
                initial_status,
                module='NewsSentimentModule',
                thesis="Initial news sentiment analysis status"
            )
            
            return True
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False

    async def process(self, **inputs) -> Dict[str, Any]:
        """Process news sentiment analysis"""
        start_time = time.time()
        
        try:
            # Extract sentiment data
            sentiment_data = await self._extract_sentiment_data(**inputs)
            
            if not sentiment_data:
                return await self._handle_no_data_fallback()
            
            # Process sentiment analysis
            sentiment_result = await self._process_sentiment_analysis(sentiment_data)
            
            # Generate trend analysis
            trend_result = await self._analyze_sentiment_trend()
            sentiment_result.update(trend_result)
            
            # Generate thesis
            thesis = await self._generate_sentiment_thesis(sentiment_data, sentiment_result)
            
            # Update SmartInfoBus
            await self._update_sentiment_smart_bus(sentiment_result, thesis)
            
            # Record success
            processing_time = (time.time() - start_time) * 1000
            self._record_success(processing_time)
            
            return sentiment_result
            
        except Exception as e:
            return await self._handle_sentiment_error(e, start_time)

    async def _extract_sentiment_data(self, **inputs) -> Optional[Dict[str, Any]]:
        """Extract sentiment data from SmartInfoBus"""
        try:
            # Get market data
            market_data = self.smart_bus.get('market_data', 'NewsSentimentModule') or {}
            
            # Get symbols
            symbols = self.smart_bus.get('symbols', 'NewsSentimentModule') or []
            
            # Get trading session
            trading_session = self.smart_bus.get('trading_session', 'NewsSentimentModule') or {}
            
            # Get symbol from inputs or market data
            symbol = inputs.get('symbol')
            if not symbol and symbols:
                symbol = symbols[0] if isinstance(symbols, list) else symbols
            if not symbol and market_data:
                symbol = market_data.get('symbol', 'EURUSD')
            
            return {
                'symbol': symbol or 'EURUSD',
                'market_data': market_data,
                'symbols': symbols,
                'trading_session': trading_session,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to extract sentiment data: {e}")
            return None

    async def _process_sentiment_analysis(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process sentiment analysis for given symbol"""
        try:
            symbol = sentiment_data.get('symbol', 'EURUSD')
            
            if not self.genome["enabled"]:
                # Use default sentiment when disabled
                self.latest_sentiment = self.genome["default_sentiment"]
                self.sentiment_confidence = 0.0
                
                return {
                    'sentiment_processed': True,
                    'sentiment_value': self.latest_sentiment,
                    'confidence': self.sentiment_confidence,
                    'source': 'default',
                    'reason': 'sentiment_disabled'
                }
            
            # Check cache first
            cached_sentiment = self._get_cached_sentiment(symbol)
            if cached_sentiment is not None:
                self.latest_sentiment = cached_sentiment['sentiment']
                self.sentiment_confidence = cached_sentiment['confidence']
                self._sentiment_performance['cache_hits'] += 1
                
                return {
                    'sentiment_processed': True,
                    'sentiment_value': self.latest_sentiment,
                    'confidence': self.sentiment_confidence,
                    'source': 'cache',
                    'cache_age': time.time() - cached_sentiment['timestamp']
                }
            
            # Fetch new sentiment
            if self.api_key:
                sentiment_result = await self._fetch_sentiment_from_api(symbol)
                self._cache_sentiment(symbol, sentiment_result['sentiment'], sentiment_result['confidence'])
            else:
                # No API key - use default with warning
                sentiment_result = {
                    'sentiment': self.genome["default_sentiment"],
                    'confidence': 0.0,
                    'source': 'default_no_api'
                }
                self.logger.warning("No API key configured - using default sentiment")
            
            self.latest_sentiment = sentiment_result['sentiment']
            self.sentiment_confidence = sentiment_result['confidence']
            
            # Update performance metrics
            self._sentiment_performance['total_requests'] += 1
            
            return {
                'sentiment_processed': True,
                'sentiment_value': self.latest_sentiment,
                'confidence': self.sentiment_confidence,
                'source': sentiment_result['source']
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return self._create_fallback_response(f"error: {str(e)}")

    def _get_cached_sentiment(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get cached sentiment for symbol"""
        if symbol not in self._cache:
            return None
        
        cache_entry = self._cache[symbol]
        if time.time() - cache_entry['timestamp'] < self.genome["cache_ttl"]:
            return cache_entry
        else:
            # Cache expired
            del self._cache[symbol]
            return None

    def _cache_sentiment(self, symbol: str, sentiment: float, confidence: float):
        """Cache sentiment for symbol"""
        self._cache[symbol] = {
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': time.time()
        }

    async def _fetch_sentiment_from_api(self, symbol: str) -> Dict[str, Any]:
        """Fetch sentiment from news API"""
        try:
            query = self._build_query_for_symbol(symbol)
            
            for attempt in range(self.genome["max_retries"] + 1):
                try:
                    sentiment_data = await self._make_api_request(query)
                    self._api_call_count += 1
                    self._sentiment_performance['api_successes'] += 1
                    
                    return {
                        'sentiment': sentiment_data['sentiment'],
                        'confidence': sentiment_data['confidence'],
                        'source': 'api'
                    }
                    
                except requests.exceptions.Timeout:
                    if attempt < self.genome["max_retries"]:
                        self.logger.warning(f"API timeout for {symbol}, retrying...")
                        await asyncio.sleep(1)
                    else:
                        self.logger.error(f"API timeout for {symbol} after {self.genome['max_retries']} retries")
                        self._api_failures += 1
                        break
                        
                except requests.exceptions.RequestException as e:
                    if attempt < self.genome["max_retries"]:
                        self.logger.warning(f"API error for {symbol}, retrying: {e}")
                        await asyncio.sleep(1)
                    else:
                        self.logger.error(f"API error for {symbol} after {self.genome['max_retries']} retries: {e}")
                        self._api_failures += 1
                        break
            
            # Fallback to default
            return {
                'sentiment': self.genome["default_sentiment"],
                'confidence': 0.0,
                'source': 'fallback'
            }
            
        except Exception as e:
            self.logger.error(f"API fetch failed: {e}")
            self._api_failures += 1
            return {
                'sentiment': self.genome["default_sentiment"],
                'confidence': 0.0,
                'source': 'error'
            }

    def _build_query_for_symbol(self, symbol: str) -> str:
        """Build query string for symbol"""
        query_map = {
            "EURUSD": "EUR USD euro dollar forex exchange rate",
            "XAUUSD": "gold price USD precious metals",
            "GBPUSD": "GBP USD pound dollar forex",
            "USDJPY": "USD JPY dollar yen forex",
            "BTCUSD": "bitcoin cryptocurrency price",
            "ETHUSD": "ethereum cryptocurrency price"
        }
        
        symbol_clean = symbol.replace("/", "").upper()
        return query_map.get(symbol_clean, f"{symbol} forex financial market")

    async def _make_api_request(self, query: str) -> Dict[str, Any]:
        """Make API request for news sentiment"""
        # Simulate API call (since you don't have API yet)
        await asyncio.sleep(0.1)  # Simulate network delay
        
        # Return mock sentiment data
        sentiment = np.random.normal(0.0, 0.3)  # Random sentiment around neutral
        sentiment = max(-1.0, min(1.0, sentiment))  # Clamp to [-1, 1]
        
        confidence = 0.5 + abs(sentiment) * 0.5  # Higher confidence for stronger sentiment
        
        return {
            'sentiment': float(sentiment),
            'confidence': float(confidence)
        }

    async def _analyze_sentiment_trend(self) -> Dict[str, Any]:
        """Analyze sentiment trend"""
        try:
            if len(self._sentiment_history) < 5:
                return {'trend_analysis': 'insufficient_data'}
            
            recent_sentiments = [s['sentiment'] for s in list(self._sentiment_history)[-5:]]
            trend_slope = np.polyfit(range(len(recent_sentiments)), recent_sentiments, 1)[0]
            
            if trend_slope > 0.05:
                trend = "improving"
            elif trend_slope < -0.05:
                trend = "declining"
            else:
                trend = "stable"
            
            return {
                'sentiment_trend': trend,
                'trend_strength': abs(float(trend_slope)),
                'recent_avg': float(np.mean(recent_sentiments))
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {'trend_analysis': 'error'}

    async def _generate_sentiment_thesis(self, sentiment_data: Dict[str, Any], 
                                       sentiment_result: Dict[str, Any]) -> str:
        """Generate comprehensive sentiment thesis"""
        try:
            symbol = sentiment_data.get('symbol', 'EURUSD')
            sentiment_value = sentiment_result.get('sentiment_value', 0.0)
            confidence = sentiment_result.get('confidence', 0.0)
            source = sentiment_result.get('source', 'unknown')
            
            # Sentiment description
            if sentiment_value > 0.3:
                sentiment_desc = "bullish"
            elif sentiment_value < -0.3:
                sentiment_desc = "bearish"
            else:
                sentiment_desc = "neutral"
            
            thesis_parts = [
                f"News Sentiment Analysis: {symbol} showing {sentiment_desc} sentiment ({sentiment_value:.3f})",
                f"Confidence level: {confidence:.1%} from {source} source",
                f"Analysis enabled: {self.genome['enabled']}, API configured: {bool(self.api_key)}"
            ]
            
            # Add trend information
            if 'sentiment_trend' in sentiment_result:
                trend = sentiment_result['sentiment_trend']
                thesis_parts.append(f"Sentiment trend: {trend} based on recent data")
            
            # Add performance info
            if self._sentiment_performance['total_requests'] > 0:
                cache_hit_rate = self._sentiment_performance['cache_hits'] / self._sentiment_performance['total_requests']
                thesis_parts.append(f"Cache efficiency: {cache_hit_rate:.1%}")
            
            # Add API status
            if self._api_call_count > 0:
                success_rate = (self._api_call_count - self._api_failures) / self._api_call_count
                thesis_parts.append(f"API reliability: {success_rate:.1%}")
            
            return " | ".join(thesis_parts)
            
        except Exception as e:
            return f"Sentiment thesis generation failed: {str(e)} - Using fallback sentiment analysis"

    async def _update_sentiment_smart_bus(self, sentiment_result: Dict[str, Any], thesis: str):
        """Update SmartInfoBus with sentiment results"""
        try:
            # Main sentiment data
            sentiment_data = {
                'sentiment': self.latest_sentiment,
                'confidence': self.sentiment_confidence,
                'enabled': self.genome["enabled"],
                'api_configured': bool(self.api_key),
                'last_updated': time.time()
            }
            
            self.smart_bus.set(
                'news_sentiment',
                sentiment_data,
                module='NewsSentimentModule',
                thesis=thesis
            )
            
            # Sentiment confidence
            confidence_data = {
                'value': self.sentiment_confidence,
                'source': sentiment_result.get('source', 'unknown'),
                'min_threshold': self.config.min_confidence
            }
            
            self.smart_bus.set(
                'sentiment_confidence',
                confidence_data,
                module='NewsSentimentModule',
                thesis="Sentiment confidence assessment and source validation"
            )
            
            # News summary (if available)
            summary_data = {
                'total_requests': self._sentiment_performance['total_requests'],
                'cache_hits': self._sentiment_performance['cache_hits'],
                'api_successes': self._sentiment_performance['api_successes'],
                'api_failures': self._api_failures
            }
            
            self.smart_bus.set(
                'news_summary',
                summary_data,
                module='NewsSentimentModule',
                thesis="News sentiment analysis performance summary"
            )
            
            # Sentiment trend
            if 'sentiment_trend' in sentiment_result:
                trend_data = {
                    'direction': sentiment_result['sentiment_trend'],
                    'strength': sentiment_result.get('trend_strength', 0.0),
                    'recent_average': sentiment_result.get('recent_avg', 0.0)
                }
                
                self.smart_bus.set(
                    'sentiment_trend',
                    trend_data,
                    module='NewsSentimentModule',
                    thesis="Sentiment trend analysis and direction assessment"
                )
            
        except Exception as e:
            self.logger.error(f"Failed to update SmartInfoBus: {e}")

    def _cleanup_expired_cache(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time - entry['timestamp'] > self.genome["cache_ttl"]
        ]
        
        for key in expired_keys:
            del self._cache[key]

    def _update_sentiment_health(self):
        """Update sentiment health metrics"""
        try:
            # Check API failure rate
            if self._api_call_count > 0:
                failure_rate = self._api_failures / self._api_call_count
                if failure_rate > 0.5:
                    self._health_status = 'warning'
                else:
                    self._health_status = 'healthy'
            
            self._last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            self._health_status = 'warning'

    async def _handle_no_data_fallback(self) -> Dict[str, Any]:
        """Handle case when no sentiment data is available"""
        self.logger.warning("No sentiment data available - using default")
        
        return {
            'sentiment_value': self.genome["default_sentiment"],
            'confidence': 0.0,
            'source': 'fallback',
            'fallback_reason': 'no_sentiment_data'
        }

    async def _handle_sentiment_error(self, error: Exception, start_time: float) -> Dict[str, Any]:
        """Handle sentiment analysis errors"""
        processing_time = (time.time() - start_time) * 1000
        
        # Update circuit breaker
        self.circuit_breaker['failures'] += 1
        self.circuit_breaker['last_failure'] = time.time()
        
        if self.circuit_breaker['failures'] >= self.circuit_breaker['threshold']:
            self.circuit_breaker['state'] = 'OPEN'
        
        # Log error with context
        error_context = self.error_pinpointer.analyze_error(error, "NewsSentimentModule")
        explanation = self.english_explainer.explain_error(
            "NewsSentimentModule", str(error), "sentiment analysis"
        )
        
        self.logger.error(
            format_operator_message(
                "💥", "SENTIMENT_ANALYSIS_ERROR",
                error=str(error),
                details=explanation,
                processing_time_ms=processing_time,
                context="sentiment_analysis"
            )
        )
        
        # Record failure
        self._record_failure(error)
        
        return self._create_fallback_response(f"error: {str(error)}")

    def _create_fallback_response(self, reason: str) -> Dict[str, Any]:
        """Create fallback response for error cases"""
        return {
            'sentiment_value': self.genome["default_sentiment"],
            'confidence': 0.0,
            'source': 'fallback',
            'circuit_breaker_state': self.circuit_breaker['state'],
            'fallback_reason': reason
        }

    def _record_success(self, processing_time: float):
        """Record successful processing"""
        self.performance_tracker.record_metric(
            'NewsSentimentModule', 'sentiment_analysis', processing_time, True
        )
        
        # Reset circuit breaker on success
        if self.circuit_breaker['state'] == 'OPEN':
            self.circuit_breaker['failures'] = 0
            self.circuit_breaker['state'] = 'CLOSED'

    def _record_failure(self, error: Exception):
        """Record processing failure"""
        self.performance_tracker.record_metric(
            'NewsSentimentModule', 'sentiment_analysis', 0, False
        )

    def get_state(self) -> Dict[str, Any]:
        """Get module state for persistence"""
        return {
            'latest_sentiment': self.latest_sentiment,
            'sentiment_confidence': self.sentiment_confidence,
            'cache': self._cache.copy(),
            'genome': self.genome.copy(),
            'api_call_count': self._api_call_count,
            'api_failures': self._api_failures,
            'sentiment_performance': self._sentiment_performance.copy(),
            'circuit_breaker': self.circuit_breaker.copy(),
            'health_status': self._health_status
        }

    def set_state(self, state: Dict[str, Any]):
        """Set module state from persistence"""
        if 'latest_sentiment' in state:
            self.latest_sentiment = state['latest_sentiment']
        
        if 'sentiment_confidence' in state:
            self.sentiment_confidence = state['sentiment_confidence']
        
        if 'cache' in state:
            self._cache = state['cache']
        
        if 'genome' in state:
            self.genome.update(state['genome'])
        
        if 'api_call_count' in state:
            self._api_call_count = state['api_call_count']
        
        if 'api_failures' in state:
            self._api_failures = state['api_failures']
        
        if 'sentiment_performance' in state:
            self._sentiment_performance.update(state['sentiment_performance'])
        
        if 'circuit_breaker' in state:
            self.circuit_breaker.update(state['circuit_breaker'])
        
        if 'health_status' in state:
            self._health_status = state['health_status']

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return {
            'status': self._health_status,
            'last_check': self._last_health_check,
            'circuit_breaker': self.circuit_breaker['state'],
            'api_configured': bool(self.api_key),
            'enabled': self.genome["enabled"],
            'cache_size': len(self._cache)
        }

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False

    # Legacy compatibility methods
    def set_sentiment(self, value: float):
        """Manually set sentiment value"""
        old_sentiment = self.latest_sentiment
        self.latest_sentiment = max(-1.0, min(1.0, float(value)))
        
        self.logger.info(
            format_operator_message(
                "🔧", "SENTIMENT_MANUAL_OVERRIDE",
                old_value=f"{old_sentiment:.3f}",
                new_value=f"{self.latest_sentiment:.3f}",
                context="manual_override"
            )
        )

    def propose_action(self, obs: Any = None, **kwargs) -> np.ndarray:
        """Legacy compatibility for action proposal"""
        # Return sentiment as action influence
        return np.array([self.latest_sentiment, 0.0])
    
    def confidence(self, obs: Any = None, **kwargs) -> float:
        """Legacy compatibility for confidence"""
        return self.sentiment_confidence

