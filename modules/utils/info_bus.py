# ─────────────────────────────────────────────────────────────
# File: modules/utils/info_bus.py
# 🚀 ENHANCED SmartInfoBus - Zero-wiring architecture with auto-discovery
# ─────────────────────────────────────────────────────────────

from __future__ import annotations
from typing import TypedDict, List, Dict, Any, Literal, Optional, Union, Callable, Set, Tuple
from datetime import datetime, timezone
import numpy as np
from dataclasses import dataclass, field
import logging
from functools import wraps
import hashlib
import json
import time
from collections import defaultdict, deque
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# SMARTINFOBUS DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DataVersion:
    """Versioned data with full tracking"""
    value: Any
    timestamp: float
    source_module: str
    version: int
    confidence: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    thesis: Optional[str] = None

class SmartInfoBus:
    """
    🚀 SmartInfoBus - Zero-wiring architecture with intelligent data flow
    ALL modules communicate ONLY through this bus
    """
    
    def __init__(self):
        # Core data storage with versioning
        self._data_store: Dict[str, DataVersion] = {}
        self._event_log: deque = deque(maxlen=10000)  # For replay
        
        # Module registry for zero-wiring
        self._providers: Dict[str, Set[str]] = defaultdict(set)
        self._consumers: Dict[str, Set[str]] = defaultdict(set)
        self._module_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self._access_patterns = defaultdict(lambda: defaultdict(int))
        self._latency_history = defaultdict(lambda: deque(maxlen=100))
        self._module_timings = defaultdict(list)
        
        # Event system
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Circuit breakers for reliability
        self._module_failures = defaultdict(int)
        self._module_disabled = set()
        self._failure_threshold = 3
        
        # Computation cache
        self._computation_cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Session replay support
        self._replay_mode = False
        self._replay_position = 0
        
        self.logger = logging.getLogger("SmartInfoBus")
    
    # ─────────────────────────────────────────────────────────────
    # Core Data Operations with Versioning
    # ─────────────────────────────────────────────────────────────
    
    def set(self, key: str, value: Any, module: str, 
            thesis: Optional[str] = None, confidence: float = 1.0,
            dependencies: Optional[List[str]] = None):
        """Set data with full tracking and thesis"""
        # Version management
        prev = self._data_store.get(key)
        version = prev.version + 1 if prev else 1
        
        # Create versioned data
        data = DataVersion(
            value=value,
            timestamp=time.time(),
            source_module=module,
            version=version,
            confidence=confidence,
            thesis=thesis,
            dependencies=dependencies or []
        )
        
        # Store
        self._data_store[key] = data
        self._providers[key].add(module)
        
        # Update dependency graph
        if dependencies:
            for dep in dependencies:
                self._module_graph[module].add(dep)
        
        # Log for replay
        self._event_log.append({
            'type': 'set',
            'key': key,
            'module': module,
            'timestamp': data.timestamp,
            'version': version,
            'has_thesis': thesis is not None
        })
        
        # Emit event
        self._emit('data_updated', {
            'key': key,
            'module': module,
            'version': version,
            'has_thesis': thesis is not None
        })
        
        self.logger.debug(f"{module} set {key} v{version}")
    
    def get(self, key: str, module: str, 
            max_age: Optional[float] = None,
            min_confidence: float = 0.0) -> Optional[Any]:
        """Get data with freshness and confidence checks"""
        self._consumers[key].add(module)
        self._access_patterns[module][f'read:{key}'] += 1
        
        data = self._data_store.get(key)
        if not data:
            self._cache_misses += 1
            return None
        
        # Check freshness
        age = time.time() - data.timestamp
        if max_age and age > max_age:
            self._emit('stale_data_warning', {
                'key': key,
                'age': age,
                'module': module
            })
            
        # Check confidence
        if data.confidence < min_confidence:
            return None
            
        self._cache_hits += 1
        return data.value
    
    def get_with_thesis(self, key: str, module: str) -> Optional[Tuple[Any, str]]:
        """Get data with its explanation"""
        data = self._data_store.get(key)
        if not data:
            return None
        
        self._consumers[key].add(module)
        return data.value, data.thesis or "No thesis provided"
    
    def get_versioned(self, key: str, module: str, version: Optional[int] = None) -> Optional[DataVersion]:
        """Get specific version or latest"""
        self._consumers[key].add(module)
        
        if version is None:
            return self._data_store.get(key)
        
        # For version history, check event log
        for event in reversed(self._event_log):
            if event['type'] == 'set' and event['key'] == key and event.get('version') == version:
                # Would need to store historical versions for full implementation
                self.logger.warning(f"Historical version {version} requested but not stored")
                return None
        
        return None
    
    # ─────────────────────────────────────────────────────────────
    # Module Registry & Discovery
    # ─────────────────────────────────────────────────────────────
    
    def register_provider(self, module: str, provides: List[str]):
        """Register what data a module provides"""
        for key in provides:
            self._providers[key].add(module)
        self.logger.info(f"Registered {module} providing: {provides}")
    
    def register_consumer(self, module: str, requires: List[str]):
        """Register what data a module requires"""
        for key in requires:
            self._consumers[key].add(module)
        self.logger.info(f"Registered {module} requiring: {requires}")
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get module dependency graph for visualization"""
        graph = {}
        
        # Build graph from providers/consumers
        for key, providers in self._providers.items():
            consumers = self._consumers.get(key, set())
            for provider in providers:
                if provider not in graph:
                    graph[provider] = []
                graph[provider].extend(list(consumers))
        
        # Add explicit dependencies
        for module, deps in self._module_graph.items():
            if module not in graph:
                graph[module] = []
            graph[module].extend(list(deps))
        
        # Remove duplicates
        for module in graph:
            graph[module] = list(set(graph[module]))
        
        return graph
    
    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order"""
        graph = self.get_dependency_graph()
        
        # Topological sort
        visited = set()
        order = []
        
        def visit(node):
            if node in visited:
                return
            visited.add(node)
            for dep in graph.get(node, []):
                visit(dep)
            order.append(node)
        
        # Visit all nodes
        all_modules = set(graph.keys()) | set(sum(graph.values(), []))
        for module in all_modules:
            visit(module)
        
        return order
    
    # ─────────────────────────────────────────────────────────────
    # Performance & Health Monitoring
    # ─────────────────────────────────────────────────────────────
    
    def record_module_timing(self, module: str, duration_ms: float):
        """Record module execution time"""
        self._latency_history[module].append(duration_ms)
        self._module_timings[module].append(duration_ms)
        
        # Check for performance degradation
        if len(self._latency_history[module]) >= 10:
            recent_avg = np.mean(list(self._latency_history[module])[-10:])
            if recent_avg > 100:  # 100ms threshold
                self._emit('performance_warning', {
                    'module': module,
                    'avg_latency_ms': recent_avg
                })
    
    def record_module_failure(self, module: str, error: str):
        """Record module failure for circuit breaker"""
        self._module_failures[module] += 1
        
        # Check circuit breaker
        if self._module_failures[module] >= self._failure_threshold:
            self._module_disabled.add(module)
            self._emit('module_disabled', {
                'module': module,
                'failures': self._module_failures[module],
                'error': error
            })
            self.logger.error(f"Module {module} disabled after {self._module_failures[module]} failures")
    
    def is_module_enabled(self, module: str) -> bool:
        """Check if module is enabled"""
        return module not in self._module_disabled
    
    def reset_module_failures(self, module: str):
        """Reset module failure count (for recovery)"""
        self._module_failures[module] = 0
        self._module_disabled.discard(module)
        self.logger.info(f"Module {module} re-enabled")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            'cache_hit_rate': self._cache_hits / max(self._cache_hits + self._cache_misses, 1),
            'total_events': len(self._event_log),
            'active_modules': len(set(self._providers.keys()) | set(self._consumers.keys())),
            'disabled_modules': list(self._module_disabled),
            'module_latencies': {}
        }
        
        # Module-specific metrics
        for module, timings in self._latency_history.items():
            if timings:
                metrics['module_latencies'][module] = {
                    'avg_ms': np.mean(timings),
                    'max_ms': max(timings),
                    'p95_ms': np.percentile(timings, 95) if len(timings) > 10 else max(timings)
                }
        
        return metrics
    
    # ─────────────────────────────────────────────────────────────
    # Event System
    # ─────────────────────────────────────────────────────────────
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to events"""
        self._subscribers[event_type].append(callback)
    
    def _emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to subscribers"""
        for callback in self._subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
    
    # ─────────────────────────────────────────────────────────────
    # Session Replay
    # ─────────────────────────────────────────────────────────────
    
    def start_replay(self, events: Optional[List[Dict]] = None):
        """Start replay mode"""
        self._replay_mode = True
        self._replay_position = 0
        if events:
            self._event_log = deque(events, maxlen=10000)
        self.logger.info(f"Started replay with {len(self._event_log)} events")
    
    def replay_next_event(self) -> Optional[Dict]:
        """Replay next event"""
        if not self._replay_mode or self._replay_position >= len(self._event_log):
            return None
        
        event = self._event_log[self._replay_position]
        self._replay_position += 1
        
        # Replay the event
        if event['type'] == 'set':
            # Simulate the set operation
            self._emit('replay_event', event)
        
        return event
    
    def stop_replay(self):
        """Stop replay mode"""
        self._replay_mode = False
        self._replay_position = 0
    
    # ─────────────────────────────────────────────────────────────
    # Plain English Explanations
    # ─────────────────────────────────────────────────────────────
    
    def explain_data_flow(self, key: str) -> str:
        """Explain data flow in plain English"""
        providers = list(self._providers.get(key, []))
        consumers = list(self._consumers.get(key, []))
        
        if not providers and not consumers:
            return f"No modules interact with '{key}'"
        
        explanation = f"DATA FLOW FOR '{key}':\n"
        
        if providers:
            explanation += f"Produced by: {', '.join(providers)}\n"
        
        if consumers:
            explanation += f"Consumed by: {', '.join(consumers)}\n"
        
        # Add data characteristics
        data = self._data_store.get(key)
        if data:
            age = time.time() - data.timestamp
            explanation += f"Last updated: {age:.1f}s ago by {data.source_module}\n"
            explanation += f"Confidence: {data.confidence:.1%}\n"
            if data.thesis:
                explanation += f"Reason: {data.thesis}\n"
        
        return explanation
    
    def explain_execution_order(self) -> str:
        """Explain execution order in plain English"""
        order = self.get_execution_order()
        
        explanation = "MODULE EXECUTION ORDER:\n"
        explanation += "=" * 50 + "\n"
        
        for i, module in enumerate(order, 1):
            deps = list(self._module_graph.get(module, []))
            if deps:
                explanation += f"{i}. {module} (depends on: {', '.join(deps)})\n"
            else:
                explanation += f"{i}. {module} (no dependencies)\n"
        
        return explanation

# ═══════════════════════════════════════════════════════════════════
# LEGACY INFOBUS COMPATIBILITY LAYER
# ═══════════════════════════════════════════════════════════════════

# Map legacy InfoBus types to SmartInfoBus
InfoBus = Dict[str, Any]  # Legacy type alias

class InfoBusManager:
    """Legacy compatibility wrapper around SmartInfoBus"""
    
    _instance: Optional[SmartInfoBus] = None
    _current_bus: Optional[SmartInfoBus] = None
    
    @classmethod
    def get_instance(cls) -> SmartInfoBus:
        """Get SmartInfoBus singleton"""
        if cls._instance is None:
            cls._instance = SmartInfoBus()
        return cls._instance
    
    @classmethod
    def create_info_bus(cls, env: Any, step: int = 0) -> InfoBus:
        """Create legacy InfoBus structure backed by SmartInfoBus"""
        smart_bus = cls.get_instance()
        cls._current_bus = smart_bus
        
        # Create legacy structure
        info_bus: InfoBus = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'step_idx': step,
            'episode_idx': getattr(env, 'episode_count', 0),
            '_smart_bus': smart_bus,  # Reference to SmartInfoBus
            'prices': {},
            'positions': [],
            'risk': {'risk_score': 0.0}
        }
        
        # Extract data from environment
        if hasattr(env, 'data'):
            for instrument in getattr(env, 'instruments', []):
                if instrument in env.data and 'D1' in env.data[instrument]:
                    df = env.data[instrument]['D1']
                    if step < len(df):
                        info_bus['prices'][instrument] = float(df['close'].iloc[step])
                        
                        # Store in SmartInfoBus
                        smart_bus.set(
                            f'price_{instrument}', 
                            float(df['close'].iloc[step]),
                            module='Environment',
                            thesis=f"Market price at step {step}"
                        )
        
        return info_bus
    
    @classmethod
    def get_current(cls) -> Optional[SmartInfoBus]:
        """Get current SmartInfoBus"""
        return cls._current_bus

# Legacy function mappings
def create_info_bus(env: Any, step: int = 0) -> InfoBus:
    """Legacy function - creates InfoBus backed by SmartInfoBus"""
    return InfoBusManager.create_info_bus(env, step)

def validate_info_bus(info_bus: InfoBus) -> 'InfoBusQuality':
    """Legacy validation"""
    from dataclasses import dataclass
    
    @dataclass
    class InfoBusQuality:
        score: float
        is_valid: bool
        missing_fields: List[str] = field(default_factory=list)
    
    # Basic validation
    required = ['timestamp', 'step_idx']
    missing = [f for f in required if f not in info_bus]
    
    score = 100.0 - len(missing) * 50
    return InfoBusQuality(
        score=score,
        is_valid=score >= 50,
        missing_fields=missing
    )

# Legacy extractor mappings
class InfoBusExtractor:
    """Legacy extractor - routes to SmartInfoBus"""
    
    @staticmethod
    def get_risk_score(info_bus: InfoBus) -> float:
        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            return smart_bus.get('risk_score', 'Environment') or 0.0
        return info_bus.get('risk', {}).get('risk_score', 0.0)
    
    @staticmethod 
    def get_market_regime(info_bus: InfoBus) -> str:
        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            return smart_bus.get('market_regime', 'Environment') or 'unknown'
        return 'unknown'
    
    @staticmethod
    def has_fresh_data(info_bus: InfoBus, max_age_seconds: float = 1.0) -> bool:
        return True  # Simplified for compatibility

class InfoBusUpdater:
    """Legacy updater - routes to SmartInfoBus"""
    
    @staticmethod
    def update_feature(info_bus: InfoBus, feature_name: str, 
                      feature_data: np.ndarray, module: str = ""):
        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            smart_bus.set(
                f'feature_{feature_name}',
                feature_data,
                module=module or 'Unknown',
                thesis=f"Feature computation for {feature_name}"
            )
    
    @staticmethod
    def add_module_data(info_bus: InfoBus, module: str, data: Dict[str, Any]):
        if '_smart_bus' in info_bus:
            smart_bus: SmartInfoBus = info_bus['_smart_bus']
            for key, value in data.items():
                smart_bus.set(
                    f'{module}_{key}',
                    value,
                    module=module
                )

# Decorators remain for compatibility
def require_info_bus(func):
    """Legacy decorator - kept for compatibility"""
    return func

def cache_computation(key: str):
    """Legacy decorator - kept for compatibility"""
    def decorator(func):
        return func
    return decorator

# Utility functions
def now_utc() -> str:
    """Current UTC timestamp"""
    return datetime.now(timezone.utc).isoformat()

def extract_standard_context(info_bus: InfoBus) -> Dict[str, Any]:
    """Extract standard context"""
    return {
        'regime': 'unknown',
        'risk_score': InfoBusExtractor.get_risk_score(info_bus),
        'position_count': len(info_bus.get('positions', [])),
        'has_fresh_data': True
    }