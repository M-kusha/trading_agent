"""
Position Manager Debug Dashboard
Production-ready web-based monitoring and debugging interface
"""

import asyncio
import json
import datetime
import time
from typing import Dict, Any, List, Optional, Deque
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import traceback

from aiohttp import web
# Optional CORS dependency
try:
    import aiohttp_cors  # type: ignore
    _HAS_AIOHTTP_CORS = True
except ImportError:  # pragma: no cover
    aiohttp_cors = None  # type: ignore
    _HAS_AIOHTTP_CORS = False
import numpy as np

# -------------------------------------------------------------
# Data Models
# -------------------------------------------------------------

@dataclass
class DebugEvent:
    """Single debug event with full context"""
    timestamp: str
    level: str  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    category: str  # DECISION, EXECUTION, RISK, PORTFOLIO, ERROR
    instrument: Optional[str]
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DecisionTrace:
    """Complete trace of a position decision"""
    decision_id: str
    timestamp: str
    instrument: str
    market_data: Dict[str, Any]
    signal_context: Dict[str, Any]
    decision_stages: List[Dict[str, Any]]
    final_decision: str
    confidence: float
    size_eur: float
    risk_factors: Dict[str, float]
    rationale: Dict[str, Any]
    execution_result: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetric:
    """Performance tracking data point"""
    timestamp: str
    metric_name: str
    value: float
    unit: str
    context: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DebugLevel(Enum):
    TRACE = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


# -------------------------------------------------------------
# Debug Dashboard Server
# -------------------------------------------------------------

class DebugDashboard:
    """
    Web-based debug dashboard for Position Manager
    Provides real-time monitoring, decision tracing, and performance analysis
    """
    
    def __init__(self, port: int = 8888, host: str = "0.0.0.0"):
        self.port = port
        self.host = host
        self.app = web.Application()
        
        # Data storage (thread-safe with locks)
        self._lock = threading.RLock()
        self.events: Deque[DebugEvent] = deque(maxlen=10000)
        self.decision_traces: Deque[DecisionTrace] = deque(maxlen=1000)
        self.performance_metrics: Deque[PerformanceMetric] = deque(maxlen=5000)
        self.active_positions: Dict[str, Dict[str, Any]] = {}
        self.system_state: Dict[str, Any] = {}
        self.error_stats: Dict[str, int] = defaultdict(int)
        
        # Real-time subscribers
        self.websocket_clients: List[web.WebSocketResponse] = []
        
        # Statistics
        self.stats = {
            "start_time": datetime.datetime.utcnow().isoformat(),
            "total_events": 0,
            "total_decisions": 0,
            "total_errors": 0,
            "uptime_seconds": 0,
        }
        
        self._setup_routes()
        self._setup_cors()
        
        # Start background tasks
        self._running = True
        self._start_background_tasks()
    
    def _setup_routes(self):
        """Setup web server routes"""
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/api/events', self.handle_events)
        self.app.router.add_get('/api/decisions', self.handle_decisions)
        self.app.router.add_get('/api/performance', self.handle_performance)
        self.app.router.add_get('/api/positions', self.handle_positions)
        self.app.router.add_get('/api/system', self.handle_system)
        self.app.router.add_get('/api/stats', self.handle_stats)
        self.app.router.add_get('/ws', self.handle_websocket)
        self.app.router.add_post('/api/event', self.handle_post_event)
        self.app.router.add_post('/api/decision', self.handle_post_decision)
        self.app.router.add_post('/api/metric', self.handle_post_metric)
    
    def _setup_cors(self):
        """Setup CORS for cross-origin requests"""
        if not _HAS_AIOHTTP_CORS or aiohttp_cors is None:
            # Minimal fallback: allow all origins via simple middleware (development only)
            async def cors_middleware(app, handler):
                async def middleware_handler(request):
                    resp = await handler(request)
                    resp.headers["Access-Control-Allow-Origin"] = "*"
                    resp.headers["Access-Control-Allow-Headers"] = "*"
                    resp.headers["Access-Control-Allow-Methods"] = "*"
                    return resp
                return middleware_handler
            if not any(getattr(m, "__name__", "").startswith("cors_middleware") for m in self.app.middlewares):
                self.app.middlewares.append(cors_middleware)  # type: ignore
            print("[DebugDashboard] aiohttp-cors not installed; using permissive fallback CORS (install 'aiohttp-cors' for full control)")
        else:
            cors = aiohttp_cors.setup(self.app, defaults={  # type: ignore[attr-defined]
                "*": aiohttp_cors.ResourceOptions(  # type: ignore[attr-defined]
                    allow_credentials=True,
                    expose_headers="*",
                    allow_headers="*",
                    allow_methods="*"
                )
            })
            for route in list(self.app.router.routes()):
                cors.add(route)  # type: ignore[attr-defined]
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        def stats_updater():
            start = time.time()
            while self._running:
                try:
                    with self._lock:
                        self.stats["uptime_seconds"] = int(time.time() - start)
                        self.stats["total_events"] = len(self.events)
                        self.stats["total_decisions"] = len(self.decision_traces)
                        self.stats["total_errors"] = sum(self.error_stats.values())
                except Exception as e:
                    print(f"Stats update error: {e}")
                time.sleep(1)
        
        threading.Thread(target=stats_updater, daemon=True).start()
    
    # -------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------
    
    async def handle_index(self, request: web.Request) -> web.Response:
        """Serve dashboard HTML"""
        html = self._generate_dashboard_html()
        return web.Response(text=html, content_type='text/html')
    
    async def handle_events(self, request: web.Request) -> web.Response:
        """Get recent events"""
        limit = int(request.query.get('limit', 100))
        level = request.query.get('level', None)
        category = request.query.get('category', None)
        instrument = request.query.get('instrument', None)
        
        with self._lock:
            events = list(self.events)
        
        # Filter events
        if level:
            events = [e for e in events if e.level == level]
        if category:
            events = [e for e in events if e.category == category]
        if instrument:
            events = [e for e in events if e.instrument == instrument]
        
        # Limit and serialize
        events = events[-limit:]
        events_data = [e.to_dict() for e in events]
        
        return web.json_response(events_data)
    
    async def handle_decisions(self, request: web.Request) -> web.Response:
        """Get decision traces"""
        limit = int(request.query.get('limit', 50))
        instrument = request.query.get('instrument', None)
        
        with self._lock:
            traces = list(self.decision_traces)
        
        if instrument:
            traces = [t for t in traces if t.instrument == instrument]
        
        traces = traces[-limit:]
        traces_data = [t.to_dict() for t in traces]
        
        return web.json_response(traces_data)
    
    async def handle_performance(self, request: web.Request) -> web.Response:
        """Get performance metrics"""
        limit = int(request.query.get('limit', 500))
        metric = request.query.get('metric', None)
        
        with self._lock:
            metrics = list(self.performance_metrics)
        
        if metric:
            metrics = [m for m in metrics if m.metric_name == metric]
        
        metrics = metrics[-limit:]
        metrics_data = [m.to_dict() for m in metrics]
        
        return web.json_response(metrics_data)
    
    async def handle_positions(self, request: web.Request) -> web.Response:
        """Get active positions"""
        with self._lock:
            positions = dict(self.active_positions)
        return web.json_response(positions)
    
    async def handle_system(self, request: web.Request) -> web.Response:
        """Get system state"""
        with self._lock:
            state = dict(self.system_state)
        return web.json_response(state)
    
    async def handle_stats(self, request: web.Request) -> web.Response:
        """Get dashboard statistics"""
        with self._lock:
            stats = dict(self.stats)
            stats["error_breakdown"] = dict(self.error_stats)
        return web.json_response(stats)
    
    async def handle_websocket(self, request: web.Request) -> web.WebSocketResponse:
        """WebSocket for real-time updates"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websocket_clients.append(ws)
        
        try:
            async for msg in ws:
                if msg.type == web.WSMsgType.TEXT:
                    # Handle client messages if needed
                    pass
                elif msg.type == web.WSMsgType.ERROR:
                    print(f'WebSocket error: {ws.exception()}')
        finally:
            self.websocket_clients.remove(ws)
        
        return ws
    
    async def handle_post_event(self, request: web.Request) -> web.Response:
        """Receive new debug event"""
        try:
            data = await request.json()
            event = DebugEvent(**data)
            self.add_event(event)
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)
    
    async def handle_post_decision(self, request: web.Request) -> web.Response:
        """Receive decision trace"""
        try:
            data = await request.json()
            trace = DecisionTrace(**data)
            self.add_decision_trace(trace)
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)
    
    async def handle_post_metric(self, request: web.Request) -> web.Response:
        """Receive performance metric"""
        try:
            data = await request.json()
            metric = PerformanceMetric(**data)
            self.add_metric(metric)
            return web.json_response({"status": "ok"})
        except Exception as e:
            return web.json_response({"status": "error", "message": str(e)}, status=400)
    
    # -------------------------------------------------------------
    # Data Management
    # -------------------------------------------------------------
    
    def add_event(self, event: DebugEvent):
        """Add new debug event"""
        with self._lock:
            self.events.append(event)
            self.stats["total_events"] += 1
            
            if event.level in ["ERROR", "CRITICAL"]:
                self.error_stats[event.category] += 1
        
        # Broadcast to WebSocket clients
        asyncio.create_task(self._broadcast_event(event))
    
    def add_decision_trace(self, trace: DecisionTrace):
        """Add decision trace"""
        with self._lock:
            self.decision_traces.append(trace)
            self.stats["total_decisions"] += 1
        
        asyncio.create_task(self._broadcast_decision(trace))
    
    def add_metric(self, metric: PerformanceMetric):
        """Add performance metric"""
        with self._lock:
            self.performance_metrics.append(metric)
        
        asyncio.create_task(self._broadcast_metric(metric))
    
    def update_positions(self, positions: Dict[str, Dict[str, Any]]):
        """Update active positions"""
        with self._lock:
            self.active_positions = positions.copy()
    
    def update_system_state(self, state: Dict[str, Any]):
        """Update system state"""
        with self._lock:
            self.system_state.update(state)
    
    async def _broadcast_event(self, event: DebugEvent):
        """Broadcast event to WebSocket clients"""
        message = json.dumps({
            "type": "event",
            "data": event.to_dict()
        })
        
        for ws in self.websocket_clients[:]:
            try:
                await ws.send_str(message)
            except Exception:
                self.websocket_clients.remove(ws)
    
    async def _broadcast_decision(self, trace: DecisionTrace):
        """Broadcast decision to WebSocket clients"""
        message = json.dumps({
            "type": "decision",
            "data": trace.to_dict()
        })
        
        for ws in self.websocket_clients[:]:
            try:
                await ws.send_str(message)
            except Exception:
                self.websocket_clients.remove(ws)
    
    async def _broadcast_metric(self, metric: PerformanceMetric):
        """Broadcast metric to WebSocket clients"""
        message = json.dumps({
            "type": "metric",
            "data": metric.to_dict()
        })
        
        for ws in self.websocket_clients[:]:
            try:
                await ws.send_str(message)
            except Exception:
                self.websocket_clients.remove(ws)
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML with embedded JavaScript"""
        return '''<!DOCTYPE html>
<html>
<head>
    <title>Position Manager Debug Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1e 0%, #1a1a2e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1600px;
            margin: 0 auto;
        }
        .header {
            background: rgba(30, 30, 50, 0.8);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        h1 {
            font-size: 28px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }
        .status-bar {
            display: flex;
            gap: 20px;
            margin-top: 15px;
        }
        .status-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 8px 15px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .panel {
            background: rgba(30, 30, 50, 0.8);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            transition: transform 0.3s ease;
        }
        .panel:hover {
            transform: translateY(-2px);
            border-color: rgba(102, 126, 234, 0.3);
        }
        .panel-header {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 15px;
            color: #a78bfa;
        }
        .event-list {
            max-height: 400px;
            overflow-y: auto;
        }
        .event-item {
            background: rgba(255, 255, 255, 0.03);
            padding: 12px;
            margin-bottom: 8px;
            border-radius: 8px;
            border-left: 3px solid transparent;
            transition: all 0.2s ease;
        }
        .event-item:hover {
            background: rgba(255, 255, 255, 0.06);
        }
        .event-item.error {
            border-left-color: #ef4444;
            background: rgba(239, 68, 68, 0.05);
        }
        .event-item.warning {
            border-left-color: #f59e0b;
            background: rgba(245, 158, 11, 0.05);
        }
        .event-item.info {
            border-left-color: #3b82f6;
            background: rgba(59, 130, 246, 0.05);
        }
        .event-item.success {
            border-left-color: #10b981;
            background: rgba(16, 185, 129, 0.05);
        }
        .event-time {
            font-size: 12px;
            color: #6b7280;
            margin-bottom: 4px;
        }
        .event-message {
            font-size: 14px;
            line-height: 1.4;
        }
        .metric-value {
            font-size: 32px;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #9ca3af;
        }
        .chart-container {
            height: 250px;
            margin-top: 15px;
        }
        .position-grid {
            display: grid;
            gap: 10px;
        }
        .position-card {
            background: rgba(255, 255, 255, 0.03);
            padding: 15px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .position-instrument {
            font-weight: 600;
            color: #e0e6ed;
        }
        .position-details {
            text-align: right;
        }
        .position-pnl {
            font-size: 18px;
            font-weight: 600;
        }
        .position-pnl.positive {
            color: #10b981;
        }
        .position-pnl.negative {
            color: #ef4444;
        }
        .filters {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }
        .filter-btn {
            padding: 6px 12px;
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 6px;
            color: #e0e0e0;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .filter-btn:hover {
            background: rgba(255, 255, 255, 0.1);
        }
        .filter-btn.active {
            background: rgba(102, 126, 234, 0.2);
            border-color: #667eea;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #6b7280;
        }
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: rgba(255, 255, 255, 0.15);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 Position Manager Debug Dashboard</h1>
            <div class="status-bar">
                <div class="status-item">
                    <div class="status-dot"></div>
                    <span>Connected</span>
                </div>
                <div class="status-item">
                    <span id="uptime">Uptime: --:--:--</span>
                </div>
                <div class="status-item">
                    <span id="event-count">Events: 0</span>
                </div>
                <div class="status-item">
                    <span id="decision-count">Decisions: 0</span>
                </div>
            </div>
        </div>

        <div class="grid">
            <div class="panel">
                <div class="panel-header">📊 Portfolio Health</div>
                <div class="metric-value" id="health-score">--</div>
                <div class="metric-label">Health Score</div>
                <div class="chart-container" id="health-chart"></div>
            </div>

            <div class="panel">
                <div class="panel-header">💰 Active Positions</div>
                <div class="position-grid" id="positions">
                    <div class="loading">Loading positions...</div>
                </div>
            </div>

            <div class="panel">
                <div class="panel-header">⚡ Performance Metrics</div>
                <div class="metric-value" id="pnl">--</div>
                <div class="metric-label">Unrealized PnL (EUR)</div>
                <div class="chart-container" id="pnl-chart"></div>
            </div>

            <div class="panel" style="grid-column: 1 / -1;">
                <div class="panel-header">🧠 Recent Decisions</div>
                <div class="event-list" id="decisions"></div>
            </div>

            <div class="panel" style="grid-column: 1 / -1;">
                <div class="panel-header">📜 Events</div>
                <div class="event-list" id="events"></div>
            </div>
        </div>
    </div>

    <script>
    const ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
    const eventsEl = document.getElementById('events');
    const decisionsEl = document.getElementById('decisions');
    const positionsEl = document.getElementById('positions');
    const healthScoreEl = document.getElementById('health-score');
    const pnlEl = document.getElementById('pnl');
    const uptimeEl = document.getElementById('uptime');
    const eventCountEl = document.getElementById('event-count');
    const decisionCountEl = document.getElementById('decision-count');

    let startTime = Date.now();

    function pad(n){return n.toString().padStart(2,'0');}
    function fmtTime(sec){const h=Math.floor(sec/3600);const m=Math.floor((sec%3600)/60);const s=sec%60;return `${pad(h)}:${pad(m)}:${pad(s)}`;}

    function addEvent(item) {
        const div = document.createElement('div');
        div.className = 'event-item ' + (item.level ? item.level.toLowerCase() : '');
        div.innerHTML = `<div class="event-time">${item.timestamp}</div><div class="event-message">[${item.category}] ${item.message}</div>`;
        eventsEl.appendChild(div);
        while (eventsEl.children.length > 200) eventsEl.removeChild(eventsEl.firstChild);
    }

    function addDecision(item) {
        const div = document.createElement('div');
        div.className = 'event-item';
        div.innerHTML = `<div class="event-time">${item.timestamp}</div><div class="event-message">${item.instrument} → ${item.final_decision} (conf ${item.confidence?.toFixed ? item.confidence.toFixed(3) : item.confidence})</div>`;
        decisionsEl.appendChild(div);
        while (decisionsEl.children.length > 100) decisionsEl.removeChild(decisionsEl.firstChild);
    }

    ws.onmessage = (ev) => {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'event') addEvent(msg.data);
        else if (msg.type === 'decision') addDecision(msg.data);
        else if (msg.type === 'metric') {
            if (msg.data.metric_name === 'portfolio_health') {
                healthScoreEl.textContent = msg.data.value.toFixed(2);
            } else if (msg.data.metric_name === 'unrealized_pnl') {
                pnlEl.textContent = msg.data.value.toFixed(2);
            }
        }
    };

    function refreshStats() {
        fetch('/api/stats').then(r=>r.json()).then(stats => {
            uptimeEl.textContent = 'Uptime: ' + fmtTime(stats.uptime_seconds || 0);
            eventCountEl.textContent = 'Events: ' + stats.total_events;
            decisionCountEl.textContent = 'Decisions: ' + stats.total_decisions;
        }).catch(()=>{});
        fetch('/api/positions').then(r=>r.json()).then(p => {
            positionsEl.innerHTML='';
            Object.entries(p).forEach(([k,v])=>{
                const card=document.createElement('div');
                const pnl=(v.unrealized_pnl||0).toFixed(2);
                card.className='position-card';
                card.innerHTML=`<div class="position-instrument">${k}</div><div class="position-details"><div>${v.size||0} @ ${v.entry_price||'-'}</div><div class="position-pnl ${pnl>=0?'positive':'negative'}">${pnl}</div></div>`;
                positionsEl.appendChild(card);
            });
            if(!positionsEl.children.length){positionsEl.innerHTML='<div class="loading">No active positions</div>';}
        }).catch(()=>{});
    }
    setInterval(refreshStats, 2000);
    refreshStats();
    </script>
</body>
</html>'''