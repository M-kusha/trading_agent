# ─────────────────────────────────────────────────────────────
# File: dashboard/__init__.py
# Training Dashboard Package
# ─────────────────────────────────────────────────────────────

# Conditional imports - gracefully handle missing dependencies
try:
    from .server import start_dashboard_server, WEB_AVAILABLE
    __all__ = ['start_dashboard_server', 'WEB_AVAILABLE']
except ImportError:
    WEB_AVAILABLE = False
    start_dashboard_server = None
    __all__ = ['WEB_AVAILABLE']
