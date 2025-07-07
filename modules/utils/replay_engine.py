# ─────────────────────────────────────────────────────────────
# File: modules/utils/replay_engine.py
# 🚀 Session replay engine for SmartInfoBus
# ─────────────────────────────────────────────────────────────

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import asyncio
from collections import defaultdict, deque

from modules.utils.smart_info_bus import SmartInfoBus, get_smart_bus
from modules.core.module_orchestrator import ModuleOrchestrator


@dataclass
class ReplayEvent:
    """Single event in replay session"""
    timestamp: float
    event_type: str
    module: str
    data: Dict[str, Any]
    
    def age_at(self, current_time: float) -> float:
        """Get age of event at given time"""
        return current_time - self.timestamp


@dataclass
class ReplaySession:
    """Complete replay session data"""
    session_id: str
    start_time: datetime
    end_time: datetime
    events: List[ReplayEvent]
    initial_state: Dict[str, Any]
    final_state: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds"""
        if self.events:
            return self.events[-1].timestamp - self.events[0].timestamp
        return 0.0
    
    @property
    def event_count(self) -> int:
        """Get total event count"""
        return len(self.events)


class ReplayEngine:
    """
    Session replay engine for debugging and analysis.
    Allows replaying trading sessions with different parameters.
    """
    
    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None):
        self.orchestrator = orchestrator
        self.smart_bus = get_smart_bus()
        
        # Replay state
        self.current_session: Optional[ReplaySession] = None
        self.replay_position = 0
        self.replay_speed = 1.0  # 1.0 = real-time
        self.is_playing = False
        self.is_paused = False
        
        # Replay modifications
        self.event_filters: List[Callable[[ReplayEvent], bool]] = []
        self.event_modifiers: List[Callable[[ReplayEvent], ReplayEvent]] = []
        
        # Analysis collectors
        self.analysis_points: List[Dict[str, Any]] = []
        self.breakpoints: List[Tuple[str, Callable[[ReplayEvent], bool]]] = []
        
        # Callbacks
        self.event_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        
        # Session storage
        self.session_dir = Path("replay_sessions")
        self.session_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger("ReplayEngine")
    
    # ═══════════════════════════════════════════════════════════
    # Session Recording
    # ═══════════════════════════════════════════════════════════
    
    def start_recording(self, session_id: Optional[str] = None) -> str:
        """Start recording a new session"""
        if not session_id:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Subscribe to InfoBus events
        self.smart_bus.subscribe('data_updated', self._record_data_update)
        self.smart_bus.subscribe('module_disabled', self._record_module_event)
        self.smart_bus.subscribe('performance_warning', self._record_module_event)
        
        # Initialize session
        self.current_session = ReplaySession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=datetime.now(),
            events=[],
            initial_state=self._capture_system_state(),
            final_state={}
        )
        
        self.logger.info(f"Started recording session: {session_id}")
        return session_id
    
    def stop_recording(self) -> Optional[ReplaySession]:
        """Stop recording and save session"""
        if not self.current_session:
            return None
        
        # Capture final state
        self.current_session.end_time = datetime.now()
        self.current_session.final_state = self._capture_system_state()
        
        # Add metadata
        self.current_session.metadata = {
            'total_events': len(self.current_session.events),
            'duration_seconds': self.current_session.duration_seconds,
            'modules_involved': list(set(e.module for e in self.current_session.events)),
            'event_types': dict(self._count_event_types())
        }
        
        # Save session
        self._save_session(self.current_session)
        
        session = self.current_session
        self.current_session = None
        
        self.logger.info(f"Stopped recording session: {session.session_id}")
        return session
    
    def _record_data_update(self, event_data: Dict[str, Any]):
        """Record data update event"""
        if self.current_session:
            event = ReplayEvent(
                timestamp=time.time(),
                event_type='data_update',
                module=event_data.get('module', 'unknown'),
                data=event_data
            )
            self.current_session.events.append(event)
    
    def _record_module_event(self, event_data: Dict[str, Any]):
        """Record module-related event"""
        if self.current_session:
            event = ReplayEvent(
                timestamp=time.time(),
                event_type=event_data.get('type', 'module_event'),
                module=event_data.get('module', 'unknown'),
                data=event_data
            )
            self.current_session.events.append(event)
    
    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state"""
        state = {
            'timestamp': time.time(),
            'infobus_metrics': self.smart_bus.get_performance_metrics(),
            'data_freshness': self.smart_bus.get_data_freshness_report()
        }
        
        if self.orchestrator:
            state['module_states'] = {}
            for name, module in self.orchestrator.modules.items():
                try:
                    state['module_states'][name] = module.get_state()
                except:
                    state['module_states'][name] = {'error': 'Failed to get state'}
        
        return state
    
    def _count_event_types(self) -> Dict[str, int]:
        """Count events by type"""
        counts = defaultdict(int)
        for event in self.current_session.events:
            counts[event.event_type] += 1
        return counts
    
    # ═══════════════════════════════════════════════════════════
    # Session Replay
    # ═══════════════════════════════════════════════════════════
    
    def load_session(self, session_id: str) -> ReplaySession:
        """Load session for replay"""
        session_file = self.session_dir / f"{session_id}.replay"
        
        if not session_file.exists():
            raise ValueError(f"Session not found: {session_id}")
        
        with open(session_file, 'rb') as f:
            self.current_session = pickle.load(f)
        
        self.replay_position = 0
        self.logger.info(f"Loaded session: {session_id} with {self.current_session.event_count} events")
        
        return self.current_session
    
    async def play(self, start_position: int = 0, end_position: Optional[int] = None):
        """Play session from start to end position"""
        if not self.current_session:
            raise ValueError("No session loaded")
        
        self.replay_position = start_position
        end_pos = end_position or len(self.current_session.events)
        self.is_playing = True
        self.is_paused = False
        
        # Restore initial state if starting from beginning
        if start_position == 0 and self.orchestrator:
            await self._restore_system_state(self.current_session.initial_state)
        
        # Calculate time scaling
        if self.current_session.events:
            first_timestamp = self.current_session.events[0].timestamp
        else:
            first_timestamp = time.time()
        
        replay_start_time = time.time()
        
        while self.replay_position < end_pos and self.is_playing:
            if self.is_paused:
                await asyncio.sleep(0.1)
                continue
            
            event = self.current_session.events[self.replay_position]
            
            # Apply filters
            if not all(f(event) for f in self.event_filters):
                self.replay_position += 1
                continue
            
            # Apply modifiers
            for modifier in self.event_modifiers:
                event = modifier(event)
            
            # Check breakpoints
            for bp_name, bp_condition in self.breakpoints:
                if bp_condition(event):
                    self.logger.info(f"Breakpoint hit: {bp_name}")
                    await self.pause()
                    break
            
            # Calculate timing
            event_offset = event.timestamp - first_timestamp
            target_replay_time = replay_start_time + (event_offset / self.replay_speed)
            current_time = time.time()
            
            # Wait if needed (for real-time replay)
            if self.replay_speed > 0 and current_time < target_replay_time:
                await asyncio.sleep(target_replay_time - current_time)
            
            # Replay event
            await self._replay_event(event)
            
            # Trigger callbacks
            await self._trigger_event_callbacks(event)
            
            # Collect analysis if needed
            if self.analysis_points:
                self._collect_analysis(event)
            
            self.replay_position += 1
        
        self.is_playing = False
        self.logger.info(f"Replay completed at position {self.replay_position}")
    
    async def _replay_event(self, event: ReplayEvent):
        """Replay a single event"""
        if event.event_type == 'data_update':
            # Replay data update
            data = event.data
            self.smart_bus.set(
                key=data.get('key', 'unknown'),
                value=data.get('value'),
                module=event.module,
                thesis=data.get('thesis'),
                confidence=data.get('confidence', 1.0)
            )
        
        elif event.event_type == 'module_disabled':
            # Replay module disable
            if self.orchestrator:
                module_name = event.data.get('module')
                if module_name:
                    self.orchestrator.disable_module(module_name)
        
        # Add more event type handlers as needed
    
    async def _restore_system_state(self, state: Dict[str, Any]):
        """Restore system to saved state"""
        if 'module_states' in state and self.orchestrator:
            for name, module_state in state['module_states'].items():
                if name in self.orchestrator.modules:
                    try:
                        self.orchestrator.modules[name].set_state(module_state)
                    except Exception as e:
                        self.logger.error(f"Failed to restore {name}: {e}")
    
    async def pause(self):
        """Pause replay"""
        self.is_paused = True
        self.logger.info(f"Replay paused at position {self.replay_position}")
    
    async def resume(self):
        """Resume replay"""
        self.is_paused = False
        self.logger.info(f"Replay resumed at position {self.replay_position}")
    
    def stop(self):
        """Stop replay"""
        self.is_playing = False
        self.is_paused = False
        self.logger.info(f"Replay stopped at position {self.replay_position}")
    
    def seek(self, position: int):
        """Seek to specific position"""
        if not self.current_session:
            raise ValueError("No session loaded")
        
        self.replay_position = max(0, min(position, len(self.current_session.events) - 1))
        self.logger.info(f"Seeked to position {self.replay_position}")
    
    # ═══════════════════════════════════════════════════════════
    # Analysis & Modification
    # ═══════════════════════════════════════════════════════════
    
    def add_filter(self, filter_func: Callable[[ReplayEvent], bool]):
        """Add event filter"""
        self.event_filters.append(filter_func)
    
    def add_modifier(self, modifier_func: Callable[[ReplayEvent], ReplayEvent]):
        """Add event modifier for what-if analysis"""
        self.event_modifiers.append(modifier_func)
    
    def add_breakpoint(self, name: str, condition: Callable[[ReplayEvent], bool]):
        """Add breakpoint condition"""

        """Add breakpoint condition"""
        self.breakpoints.append((name, condition))
    
    def clear_breakpoint(self, name: str):
        """Remove breakpoint by name"""
        self.breakpoints = [(n, c) for n, c in self.breakpoints if n != name]
    
    def subscribe_to_event(self, event_type: str, callback: Callable):
        """Subscribe to specific event type during replay"""
        self.event_callbacks[event_type].append(callback)
    
    async def _trigger_event_callbacks(self, event: ReplayEvent):
        """Trigger callbacks for event"""
        # General callbacks
        for callback in self.event_callbacks.get('*', []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
        
        # Type-specific callbacks
        for callback in self.event_callbacks.get(event.event_type, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")
    
    def add_analysis_point(self, name: str, extractor: Callable[[ReplayEvent], Any]):
        """Add analysis data extractor"""
        self.analysis_points.append({
            'name': name,
            'extractor': extractor,
            'data': []
        })
    
    def _collect_analysis(self, event: ReplayEvent):
        """Collect analysis data from event"""
        for analysis in self.analysis_points:
            try:
                data = analysis['extractor'](event)
                if data is not None:
                    analysis['data'].append({
                        'timestamp': event.timestamp,
                        'position': self.replay_position,
                        'value': data
                    })
            except Exception as e:
                self.logger.error(f"Analysis error for {analysis['name']}: {e}")
    
    def get_analysis_results(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get collected analysis results"""
        return {
            analysis['name']: analysis['data']
            for analysis in self.analysis_points
        }
    
    # ═══════════════════════════════════════════════════════════
    # What-If Analysis
    # ═══════════════════════════════════════════════════════════
    
    async def run_what_if_scenario(self, scenario_name: str,
                                  modifications: List[Callable[[ReplayEvent], ReplayEvent]],
                                  start_position: int = 0) -> Dict[str, Any]:
        """Run what-if scenario with modifications"""
        self.logger.info(f"Running what-if scenario: {scenario_name}")
        
        # Save current state
        original_modifiers = self.event_modifiers.copy()
        original_position = self.replay_position
        
        # Apply scenario modifications
        self.event_modifiers = modifications
        
        # Collect results
        results = {
            'scenario_name': scenario_name,
            'start_position': start_position,
            'events_processed': 0,
            'analysis': {}
        }
        
        # Run replay
        try:
            await self.play(start_position)
            results['events_processed'] = self.replay_position - start_position
            results['analysis'] = self.get_analysis_results()
            results['final_state'] = self._capture_system_state()
        except Exception as e:
            results['error'] = str(e)
            self.logger.error(f"What-if scenario failed: {e}")
        finally:
            # Restore original state
            self.event_modifiers = original_modifiers
            self.replay_position = original_position
        
        return results
    
    def create_risk_modification(self, risk_multiplier: float) -> Callable:
        """Create modifier that adjusts risk scores"""
        def modify_risk(event: ReplayEvent) -> ReplayEvent:
            if event.event_type == 'data_update' and 'risk' in event.data.get('key', ''):
                # Modify risk value
                modified_event = ReplayEvent(
                    timestamp=event.timestamp,
                    event_type=event.event_type,
                    module=event.module,
                    data=event.data.copy()
                )
                
                if 'value' in modified_event.data and isinstance(modified_event.data['value'], (int, float)):
                    modified_event.data['value'] *= risk_multiplier
                
                return modified_event
            return event
        
        return modify_risk
    
    def create_latency_modification(self, latency_factor: float) -> Callable:
        """Create modifier that simulates increased latency"""
        def modify_latency(event: ReplayEvent) -> ReplayEvent:
            # Adjust timestamp to simulate latency
            modified_event = ReplayEvent(
                timestamp=event.timestamp * latency_factor,
                event_type=event.event_type,
                module=event.module,
                data=event.data.copy()
            )
            return modified_event
        
        return modify_latency
    
    def create_module_failure_modification(self, module_name: str, 
                                         failure_rate: float) -> Callable:
        """Create modifier that simulates module failures"""
        def modify_failure(event: ReplayEvent) -> ReplayEvent:
            if event.module == module_name and np.random.random() < failure_rate:
                # Convert to failure event
                return ReplayEvent(
                    timestamp=event.timestamp,
                    event_type='module_failure',
                    module=module_name,
                    data={'original_event': event.data, 'simulated': True}
                )
            return event
        
        return modify_failure
    
    # ═══════════════════════════════════════════════════════════
    # Session Management
    # ═══════════════════════════════════════════════════════════
    
    def _save_session(self, session: ReplaySession):
        """Save session to disk"""
        session_file = self.session_dir / f"{session.session_id}.replay"
        
        with open(session_file, 'wb') as f:
            pickle.dump(session, f)
        
        # Also save metadata as JSON for easy browsing
        metadata_file = self.session_dir / f"{session.session_id}.meta.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'session_id': session.session_id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat(),
                'duration_seconds': session.duration_seconds,
                'event_count': session.event_count,
                'metadata': session.metadata
            }, f, indent=2)
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List available replay sessions"""
        sessions = []
        
        for meta_file in self.session_dir.glob("*.meta.json"):
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                sessions.append(metadata)
            except Exception as e:
                self.logger.error(f"Failed to read {meta_file}: {e}")
        
        # Sort by start time
        sessions.sort(key=lambda x: x['start_time'], reverse=True)
        
        return sessions
    
    def delete_session(self, session_id: str):
        """Delete a replay session"""
        session_file = self.session_dir / f"{session_id}.replay"
        metadata_file = self.session_dir / f"{session_id}.meta.json"
        
        if session_file.exists():
            session_file.unlink()
        if metadata_file.exists():
            metadata_file.unlink()
        
        self.logger.info(f"Deleted session: {session_id}")
    
    # ═══════════════════════════════════════════════════════════
    # Analysis Helpers
    # ═══════════════════════════════════════════════════════════
    
    def analyze_module_performance(self, module_name: str) -> Dict[str, Any]:
        """Analyze performance of specific module during replay"""
        if not self.current_session:
            return {'error': 'No session loaded'}
        
        module_events = [e for e in self.current_session.events if e.module == module_name]
        
        if not module_events:
            return {'error': f'No events found for module {module_name}'}
        
        # Calculate statistics
        event_types = defaultdict(int)
        for event in module_events:
            event_types[event.event_type] += 1
        
        # Extract timings if available
        timings = []
        for event in module_events:
            if 'processing_time_ms' in event.data:
                timings.append(event.data['processing_time_ms'])
        
        analysis = {
            'module': module_name,
            'total_events': len(module_events),
            'event_types': dict(event_types),
            'first_event': module_events[0].timestamp,
            'last_event': module_events[-1].timestamp,
            'duration': module_events[-1].timestamp - module_events[0].timestamp
        }
        
        if timings:
            analysis['performance'] = {
                'avg_time_ms': np.mean(timings),
                'max_time_ms': max(timings),
                'min_time_ms': min(timings),
                'p95_time_ms': np.percentile(timings, 95)
            }
        
        return analysis
    
    def find_anomalies(self, threshold_std: float = 3.0) -> List[Dict[str, Any]]:
        """Find anomalous events in session"""
        if not self.current_session:
            return []
        
        anomalies = []
        
        # Group events by type and module
        event_groups = defaultdict(list)
        for i, event in enumerate(self.current_session.events):
            key = f"{event.module}:{event.event_type}"
            event_groups[key].append((i, event))
        
        # Find anomalies in each group
        for key, events in event_groups.items():
            if len(events) < 10:  # Need minimum data
                continue
            
            # Check timing anomalies
            if len(events) > 1:
                intervals = []
                for i in range(1, len(events)):
                    interval = events[i][1].timestamp - events[i-1][1].timestamp
                    intervals.append(interval)
                
                if intervals:
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                    
                    for i, interval in enumerate(intervals):
                        z_score = abs((interval - mean_interval) / max(std_interval, 0.001))
                        if z_score > threshold_std:
                            anomalies.append({
                                'type': 'timing_anomaly',
                                'position': events[i+1][0],
                                'event': events[i+1][1],
                                'z_score': z_score,
                                'expected_interval': mean_interval,
                                'actual_interval': interval
                            })
            
            # Check value anomalies
            values = []
            for pos, event in events:
                if 'value' in event.data and isinstance(event.data['value'], (int, float)):
                    values.append((pos, event, event.data['value']))
            
            if len(values) > 10:
                value_array = np.array([v[2] for v in values])
                mean_value = np.mean(value_array)
                std_value = np.std(value_array)
                
                for pos, event, value in values:
                    z_score = abs((value - mean_value) / max(std_value, 0.001))
                    if z_score > threshold_std:
                        anomalies.append({
                            'type': 'value_anomaly',
                            'position': pos,
                            'event': event,
                            'z_score': z_score,
                            'expected_value': mean_value,
                            'actual_value': value
                        })
        
        # Sort by position
        anomalies.sort(key=lambda x: x['position'])
        
        return anomalies
    
    def generate_replay_report(self) -> str:
        """Generate comprehensive replay analysis report"""
        if not self.current_session:
            return "No session loaded"
        
        report = f"""
REPLAY SESSION REPORT
====================
Session ID: {self.current_session.session_id}
Start Time: {self.current_session.start_time}
End Time: {self.current_session.end_time}
Duration: {self.current_session.duration_seconds:.1f} seconds
Total Events: {self.current_session.event_count}

EVENT BREAKDOWN:
"""
        
        # Event type breakdown
        event_types = defaultdict(int)
        for event in self.current_session.events:
            event_types[event.event_type] += 1
        
        for event_type, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True):
            report += f"  {event_type}: {count} ({count/self.current_session.event_count*100:.1f}%)\n"
        
        # Module activity
        module_events = defaultdict(int)
        for event in self.current_session.events:
            module_events[event.module] += 1
        
        report += "\nMODULE ACTIVITY:\n"
        for module, count in sorted(module_events.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"  {module}: {count} events\n"
        
        # Anomalies
        anomalies = self.find_anomalies()
        if anomalies:
            report += f"\nANOMALIES DETECTED: {len(anomalies)}\n"
            for i, anomaly in enumerate(anomalies[:5], 1):
                report += f"  {i}. {anomaly['type']} at position {anomaly['position']} (z-score: {anomaly['z_score']:.1f})\n"
        
        # Analysis results
        if self.analysis_points:
            report += "\nANALYSIS RESULTS:\n"
            results = self.get_analysis_results()
            for name, data in results.items():
                if data:
                    report += f"  {name}: {len(data)} data points collected\n"
        
        return report
    
    # ═══════════════════════════════════════════════════════════
    # Interactive Debugging
    # ═══════════════════════════════════════════════════════════
    
    def get_event_at_position(self, position: int) -> Optional[ReplayEvent]:
        """Get event at specific position"""
        if not self.current_session or position >= len(self.current_session.events):
            return None
        return self.current_session.events[position]
    
    def get_events_in_range(self, start: int, end: int) -> List[ReplayEvent]:
        """Get events in position range"""
        if not self.current_session:
            return []
        return self.current_session.events[start:end]
    
    def search_events(self, criteria: Dict[str, Any]) -> List[Tuple[int, ReplayEvent]]:
        """Search for events matching criteria"""
        if not self.current_session:
            return []
        
        matches = []
        
        for i, event in enumerate(self.current_session.events):
            match = True
            
            # Check criteria
            if 'module' in criteria and event.module != criteria['module']:
                match = False
            
            if 'event_type' in criteria and event.event_type != criteria['event_type']:
                match = False
            
            if 'after_time' in criteria and event.timestamp < criteria['after_time']:
                match = False
            
            if 'before_time' in criteria and event.timestamp > criteria['before_time']:
                match = False
            
            if 'data_contains' in criteria:
                data_str = json.dumps(event.data)
                if criteria['data_contains'] not in data_str:
                    match = False
            
            if match:
                matches.append((i, event))
        
        return matches
    
    def export_debug_info(self, position: int, window: int = 10) -> Dict[str, Any]:
        """Export debug information around specific position"""
        if not self.current_session:
            return {'error': 'No session loaded'}
        
        # Get events around position
        start = max(0, position - window)
        end = min(len(self.current_session.events), position + window + 1)
        
        events = []
        for i in range(start, end):
            event = self.current_session.events[i]
            events.append({
                'position': i,
                'timestamp': event.timestamp,
                'module': event.module,
                'type': event.event_type,
                'data': event.data,
                'is_target': i == position
            })
        
        # Get system state at position (approximate)
        state_snapshot = {}
        if self.orchestrator:
            # This would need to replay to position to get exact state
            state_snapshot = {'note': 'Exact state requires replay to position'}
        
        return {
            'target_position': position,
            'window_size': window,
            'events': events,
            'state_snapshot': state_snapshot
        }