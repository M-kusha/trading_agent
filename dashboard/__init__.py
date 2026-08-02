

try:
    from .server import WEB_AVAILABLE, start_dashboard_server
    __all__ = ['WEB_AVAILABLE', 'start_dashboard_server']
except ImportError:
    WEB_AVAILABLE = False
    start_dashboard_server = None
    __all__ = ['WEB_AVAILABLE']
