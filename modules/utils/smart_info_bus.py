# ─────────────────────────────────────────────────────────────
# File: modules/utils/smart_info_bus.py
# 🚀 Enhanced SmartInfoBus - Core of zero-wiring architecture
# ─────────────────────────────────────────────────────────────

import time
import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple, Callable
from collections import defaultdict, deque
from dataclasses import dataclass, field
import numpy as np
import logging
from datetime import datetime
import json
import pickle


@dataclass
class DataVersion:
    """Versioned data with complete tracking"""
    value: Any
    timestamp: float
    source_module: str
    version: int
    confidence: float = 1.0
    dependencies: List[str] = field(default_factory=list)
    thesis: Optional[str] = None
    processing_time_ms: float = 0.0
    
    def age_seconds(self) -> float:
        """Get age of data in seconds"""
        return time.time() - self.timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'value': self.value,
            'timestamp': self.timestamp,
            'source_module': self.source_module,
            'version': self.version,
            'confidence': self.confidence,
            'dependencies': self.dependencies,
            'thesis': self.thesis,
            'processing_time_ms': self.processing_time_ms,
            'age_seconds': self.age_seconds()
        }


@dataclass
class DataRequest:
    """Request for data from a module"""
    requesting_module: str
    requested_key: str
    timestamp: float
    max_age_seconds: Optional[float] = None
    min_confidence: Optional[float] = None


class SmartInfoBus:
    """
    Enhanced Information Bus for zero-wiring architecture.
    Central hub for all inter-module communication with:
    - Automatic dependency resolution
    - Data versioning and tracking
    - Performance monitoring
    - Circuit breakers
    - Plain English explanations
    """
    
    def __init__(self):
        # Core data storage
        self._data_store: Dict[str, DataVersion] = {}
        self._data_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # Event logging for replay
        self._event_log: deque = deque(maxlen=10000)
        self._replay_mode = False
        self._replay_position = 0
        
        # Module registry
        self._providers: Dict[str, Set[str]] = defaultdict(set)
        self._consumers: Dict[str, Set[str]] = defaultdict(set)
        self._module_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance tracking
        self._access_patterns = defaultdict(lambda: defaultdict(int))
        self._latency_history = defaultdict(lambda: deque(maxlen=100))
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Event subscriptions
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Circuit breakers
        self._module_failures = defaultdict(int)
        self._module_disabled: Set[str] = set()
        self._failure_threshold = 3
        self._recovery_time = 60  # seconds
        self._last_failure_time = {}
        
        # Request tracking
        self._pending_requests: List[DataRequest] = []
        
        # Setup logging
        self.logger = logging.getLogger("SmartInfoBus")
        
        # Initialize event
        self._emit('bus_initialized', {'timestamp': time.time()})
    
    # ═══════════════════════════════════════════════════════════
    # Core Data Operations
    # ═══════════════════════════════════════════════════════════
    
    def set(self, key: str, value: Any, module: str, 
            thesis: Optional[str] = None, confidence: float = 1.0,
            dependencies: Optional[List[str]] = None,
            processing_time_ms: float = 0.0):
        """
        Set data with full tracking and optional thesis.
        
        Args:
            key: Data key
            value: Data value
            module: Source module name
            thesis: Plain English explanation
            confidence: Confidence score (0-1)
            dependencies: List of data keys this depends on
            processing_time_ms: Time taken to compute
        """
        # Get previous version
        prev = self._data_store.get(key)
        version = prev.version + 1 if prev else 1
        
        # Create versioned data
        data = DataVersion(
            value=value,
            timestamp=time.time(),
            source_module=module,
            version=version,
            confidence=max(0.0, min(1.0, confidence)),
            thesis=thesis,
            dependencies=dependencies or [],
            processing_time_ms=processing_time_ms
        )
        
        # Store current version
        self._data_store[key] = data
        
        # Store in history
        self._data_history[key].append(data)
        
        # Update registry
        self._providers[key].add(module)
        
        # Update dependency graph
        if dependencies:
            for dep in dependencies:
                self._module_graph[module].add(self._get_provider(dep))
        
        # Log event
        self._log_event({
            'type': 'set',
            'key': key,
            'module': module,
            'timestamp': data.timestamp,
            'version': version,
            'has_thesis': thesis is not None,
            'confidence': confidence
        })
        
        # Emit event
        self._emit('data_updated', {
            'key': key,
            'module': module,
            'version': version,
            'confidence': confidence
        })
        
        # Check pending requests
        self._check_pending_requests(key)
        
        self.logger.debug(
            f"{module} set '{key}' v{version} "
            f"(confidence: {confidence:.2f})"
        )
    
    def get(self, key: str, module: str, 
            max_age: Optional[float] = None,
            min_confidence: float = 0.0) -> Optional[Any]:
        """
        Get data with optional freshness and confidence requirements.
        
        Args:
            key: Data key
            module: Requesting module
            max_age: Maximum age in seconds
            min_confidence: Minimum confidence required
            
        Returns:
            Data value or None if requirements not met
        """
        # Track access
        self._consumers[key].add(module)
        self._access_patterns[module][f'read:{key}'] += 1
        
        # Get data
        data = self._data_store.get(key)
        
        if not data:
            self._cache_misses += 1
            self._log_miss(key, module)
            return None
        
        # Check age
        if max_age and data.age_seconds() > max_age:
            self._emit('stale_data_warning', {
                'key': key,
                'age': data.age_seconds(),
                'module': module,
                'max_age': max_age
            })
            return None
        
        # Check confidence
        if data.confidence < min_confidence:
            return None
        
        self._cache_hits += 1
        return data.value
    
    def get_with_metadata(self, key: str, module: str) -> Optional[DataVersion]:
        """Get data with full metadata"""
        self._consumers[key].add(module)
        return self._data_store.get(key)
    
    def get_with_thesis(self, key: str, module: str) -> Optional[Tuple[Any, str]]:
        """Get data value and its explanation"""
        data = self.get_with_metadata(key, module)
        if not data:
            return None
        return data.value, data.thesis or "No explanation provided"
    
    def request_data(self, key: str, module: str, 
                    max_age: Optional[float] = None,
                    min_confidence: Optional[float] = None) -> None:
        """
        Request data that may not be available yet.
        Will be notified when data becomes available.
        """
        request = DataRequest(
            requesting_module=module,
            requested_key=key,
            timestamp=time.time(),
            max_age_seconds=max_age,
            min_confidence=min_confidence
        )
        
        self._pending_requests.append(request)
        self._consumers[key].add(module)
        
        self.logger.debug(f"{module} requested '{key}'")
    
    # ═══════════════════════════════════════════════════════════
    # Module Registry
    # ═══════════════════════════════════════════════════════════
    
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
    
    def get_providers(self, key: str) -> Set[str]:
        """Get all modules that can provide a data key"""
        return self._providers.get(key, set())
    
    def get_consumers(self, key: str) -> Set[str]:
        """Get all modules that consume a data key"""
        return self._consumers.get(key, set())
    
    def _get_provider(self, key: str) -> Optional[str]:
        """Get primary provider for a key"""
        providers = self.get_providers(key)
        if providers:
            # Return first non-disabled provider
            for provider in providers:
                if provider not in self._module_disabled:
                    return provider
        return None
    
    # ═══════════════════════════════════════════════════════════
    # Performance & Health
    # ═══════════════════════════════════════════════════════════
    
    def record_module_timing(self, module: str, duration_ms: float):
        """Record module execution time"""
        self._latency_history[module].append(duration_ms)
        
        # Check for performance issues
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
        self._last_failure_time[module] = time.time()
        
        # Check circuit breaker
        if self._module_failures[module] >= self._failure_threshold:
            self._module_disabled.add(module)
            self._emit('module_disabled', {
                'module': module,
                'failures': self._module_failures[module],
                'error': error
            })
            self.logger.error(
                f"Module {module} disabled after "
                f"{self._module_failures[module]} failures"
            )
    
    def is_module_enabled(self, module: str) -> bool:
        """Check if module is enabled"""
        # Check if disabled
        if module in self._module_disabled:
            # Check if recovery time has passed
            if module in self._last_failure_time:
                time_since_failure = time.time() - self._last_failure_time[module]
                if time_since_failure > self._recovery_time:
                    # Try to re-enable
                    self.reset_module_failures(module)
                    return True
            return False
        return True
    
    def reset_module_failures(self, module: str):
        """Reset module failure count"""
        self._module_failures[module] = 0
        self._module_disabled.discard(module)
        self._last_failure_time.pop(module, None)
        
        self._emit('module_enabled', {'module': module})
        self.logger.info(f"Module {module} re-enabled")
    
    def get_module_health(self, module: str) -> Dict[str, Any]:
        """Get module health information"""
        return {
            'enabled': self.is_module_enabled(module),
            'failures': self._module_failures.get(module, 0),
            'avg_latency_ms': np.mean(list(self._latency_history.get(module, [])))
                             if module in self._latency_history else 0,
            'provides': list(self._providers.get(module, [])),
            'consumes': list(self._consumers.get(module, []))
        }
    
    # ═══════════════════════════════════════════════════════════
    # Dependency Analysis
    # ═══════════════════════════════════════════════════════════
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get module dependency graph"""
        graph = {}
        
        # Build from data dependencies
        for key, consumers in self._consumers.items():
            providers = self._providers.get(key, set())
            
            for provider in providers:
                if provider not in graph:
                    graph[provider] = []
                graph[provider].extend(list(consumers))
        
        # Add explicit dependencies
        for module, deps in self._module_graph.items():
            if module not in graph:
                graph[module] = []
            graph[module].extend([d for d in deps if d])
        
        # Remove duplicates
        for module in graph:
            graph[module] = list(set(graph[module]))
        
        return graph
    
    def find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in module graph"""
        graph = self.get_dependency_graph()
        cycles = []
        
        def dfs(node: str, path: List[str], visited: Set[str]):
            if node in path:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor, path.copy(), visited.copy())
        
        # Start DFS from each node
        for node in graph:
            dfs(node, [], set())
        
        # Remove duplicate cycles
        unique_cycles = []
        for cycle in cycles:
            if not any(set(cycle) == set(c) for c in unique_cycles):
                unique_cycles.append(cycle)
        
        return unique_cycles
    
    # ═══════════════════════════════════════════════════════════
    # Event System
    # ═══════════════════════════════════════════════════════════
    
    def subscribe(self, event_type: str, callback: Callable):
        """Subscribe to bus events"""
        self._subscribers[event_type].append(callback)
    
    def _emit(self, event_type: str, data: Dict[str, Any]):
        """Emit event to subscribers"""
        for callback in self._subscribers[event_type]:
            try:
                callback(data)
            except Exception as e:
                self.logger.error(f"Event callback error: {e}")
    
    def _log_event(self, event: Dict[str, Any]):
        """Log event for replay"""
        event['timestamp'] = event.get('timestamp', time.time())
        self._event_log.append(event)
    
    def _log_miss(self, key: str, module: str):
        """Log data miss for analysis"""
        self._log_event({
            'type': 'miss',
            'key': key,
            'module': module,
            'providers': list(self._providers.get(key, [])),
            'timestamp': time.time()
        })
    
    def _check_pending_requests(self, key: str):
        """Check if any pending requests can be fulfilled"""
        fulfilled = []
        
        for i, request in enumerate(self._pending_requests):
            if request.requested_key == key:
                # Check if data meets requirements
                data = self._data_store.get(key)
                if data:
                    age_ok = (request.max_age_seconds is None or 
                             data.age_seconds() <= request.max_age_seconds)
                    conf_ok = (request.min_confidence is None or 
                              data.confidence >= request.min_confidence)
                    
                    if age_ok and conf_ok:
                        # Notify module
                        self._emit('data_available', {
                            'key': key,
                            'requesting_module': request.requesting_module,
                            'value': data.value
                        })
                        fulfilled.append(i)
        
        # Remove fulfilled requests
        for i in reversed(fulfilled):
            self._pending_requests.pop(i)
    
    # ═══════════════════════════════════════════════════════════
    # Analysis & Monitoring
    # ═══════════════════════════════════════════════════════════
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        total_requests = self._cache_hits + self._cache_misses
        
        return {
            'cache_hit_rate': self._cache_hits / max(total_requests, 1),
            'total_requests': total_requests,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'active_data_keys': len(self._data_store),
            'total_events': len(self._event_log),
            'disabled_modules': list(self._module_disabled),
            'pending_requests': len(self._pending_requests),
            'module_latencies': {
                module: {
                    'avg_ms': np.mean(timings),
                    'max_ms': max(timings),
                    'p95_ms': np.percentile(timings, 95) if len(timings) > 10 else max(timings)
                }
                for module, timings in self._latency_history.items()
                if timings
            }
        }
    
    def get_data_freshness_report(self) -> Dict[str, Dict[str, Any]]:
        """Get data freshness information"""
        report = {}
        
        for key, data in self._data_store.items():
            report[key] = {
                'age_seconds': data.age_seconds(),
                'version': data.version,
                'source': data.source_module,
                'confidence': data.confidence,
                'has_thesis': data.thesis is not None
            }
        
        return report
    
    def explain_data_flow(self, key: str) -> str:
        """Generate plain English explanation of data flow"""
        providers = list(self._providers.get(key, []))
        consumers = list(self._consumers.get(key, []))
        data = self._data_store.get(key)
        
        explanation = f"DATA FLOW ANALYSIS: '{key}'\n"
        explanation += "=" * 50 + "\n\n"
        
        if not providers and not consumers:
            explanation += f"No modules interact with '{key}'\n"
            return explanation
        
        if providers:
            explanation += f"PROVIDERS ({len(providers)}):\n"
            for provider in providers:
                enabled = "✓" if self.is_module_enabled(provider) else "✗"
                explanation += f"  {enabled} {provider}\n"
        
        if consumers:
            explanation += f"\nCONSUMERS ({len(consumers)}):\n"
            for consumer in consumers:
                explanation += f"  • {consumer}\n"
        
        if data:
            explanation += f"\nCURRENT STATE:\n"
            explanation += f"  Version: {data.version}\n"
            explanation += f"  Age: {data.age_seconds():.1f} seconds\n"
            explanation += f"  Source: {data.source_module}\n"
            explanation += f"  Confidence: {data.confidence:.1%}\n"
            
            if data.thesis:
                explanation += f"\nEXPLANATION:\n{data.thesis}\n"
        else:
            explanation += f"\nNO DATA AVAILABLE\n"
        
        # Access patterns
        access_count = sum(
            self._access_patterns[m].get(f'read:{key}', 0)
            for m in consumers
        )
        if access_count > 0:
            explanation += f"\nACCESS STATISTICS:\n"
            explanation += f"  Total reads: {access_count}\n"
        
        return explanation
    
    # ═══════════════════════════════════════════════════════════
    # Session Replay
    # ═══════════════════════════════════════════════════════════
    
    def start_replay(self, events: Optional[List[Dict]] = None):
        """Start replay mode"""
        self._replay_mode = True
        self._replay_position = 0
        
        if events:
            self._event_log = deque(events, maxlen=10000)
        
        self.logger.info(f"Started replay with {len(self._event_log)} events")
    
    def replay_next_event(self) -> Optional[Dict]:
        """Replay next event"""
        if not self._replay_mode:
            return None
        
        if self._replay_position >= len(self._event_log):
            self._replay_mode = False
            return None
        
        event = self._event_log[self._replay_position]
        self._replay_position += 1
        
        # Replay the event
        if event['type'] == 'set':
            # Simulate data set
            self._emit('replay_event', event)
        
        return event
    
    def stop_replay(self):
        """Stop replay mode"""
        self._replay_mode = False
        self._replay_position = 0
    
    def export_session(self, filepath: str):
        """Export session for replay"""
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'events': list(self._event_log),
            'final_state': {
                key: data.to_dict()
                for key, data in self._data_store.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2, default=str)
    
    def import_session(self, filepath: str):
        """Import session for replay"""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
        
        self._event_log = deque(session_data['events'], maxlen=10000)
        self.logger.info(f"Imported session with {len(self._event_log)} events")
    
    # ═══════════════════════════════════════════════════════════
    # Persistence
    # ═══════════════════════════════════════════════════════════
    
    def save_state(self, filepath: str):
        """Save complete bus state"""
        state = {
            'data_store': {
                key: data.to_dict()
                for key, data in self._data_store.items()
            },
            'providers': {k: list(v) for k, v in self._providers.items()},
            'consumers': {k: list(v) for k, v in self._consumers.items()},
            'module_failures': dict(self._module_failures),
            'disabled_modules': list(self._module_disabled),
            'performance_metrics': self.get_performance_metrics()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str):
        """Load bus state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        # Restore registrations
        self._providers = defaultdict(set)
        for k, v in state['providers'].items():
            self._providers[k] = set(v)
        
        self._consumers = defaultdict(set)
        for k, v in state['consumers'].items():
            self._consumers[k] = set(v)
        
        # Restore module state
        self._module_failures = defaultdict(int, state['module_failures'])
        self._module_disabled = set(state['disabled_modules'])
        
        self.logger.info("Loaded bus state")


# Global SmartInfoBus instance
_smart_bus_instance: Optional[SmartInfoBus] = None


def get_smart_bus() -> SmartInfoBus:
    """Get global SmartInfoBus instance"""
    global _smart_bus_instance
    if _smart_bus_instance is None:
        _smart_bus_instance = SmartInfoBus()
    return _smart_bus_instance