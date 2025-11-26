import React from 'react';
import {
  Clock, Brain, TrendingUp, Pause, XCircle, AlertTriangle,
  Power, Bell, FileText, Radio, Heart, Activity, Zap, Eye,
  Users, Circle, ChevronRight, ArrowUp, ArrowDown
} from 'lucide-react';

// ─────────────────────────────────────────────────────────────────────────────────────
// Shared UI Components
// ─────────────────────────────────────────────────────────────────────────────────────

export const StatusIndicator = React.memo(function StatusIndicator({ status, className = '' }) {
  const configs = {
    IDLE: { color: 'text-gray-400', icon: Clock, pulse: false },
    TRAINING: { color: 'text-blue-400', icon: Brain, pulse: true },
    TRADING: { color: 'text-green-400', icon: TrendingUp, pulse: true },
    STOPPING: { color: 'text-yellow-400', icon: Pause, pulse: true },
    ERROR: { color: 'text-red-400', icon: XCircle, pulse: true },
    EMERGENCY_STOPPED: { color: 'text-red-600', icon: AlertTriangle, pulse: true }
  };
  const config = configs[status] || configs.IDLE;
  const Icon = config.icon;

  return (
    <span className={`flex items-center ${config.color} ${className}`}>
      <Icon className={`w-4 h-4 mr-1 ${config.pulse ? 'animate-pulse' : ''}`} />
      <span className="font-medium">{status}</span>
    </span>
  );
});

export const TabButton = React.memo(function TabButton({ icon: Icon, label, active, onClick, badge, disabled = false }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all duration-200 ${
        disabled
          ? 'opacity-50 cursor-not-allowed'
          : active
            ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white shadow-lg'
            : 'text-gray-300 hover:bg-gray-700 hover:text-white'
      }`}
    >
      <div className="flex items-center">
        <Icon className="w-5 h-5 mr-3" />
        <span className="font-medium">{label}</span>
      </div>
      {badge != null && (
        <span className={`text-xs px-2 py-1 rounded-full font-bold ${Number(badge) > 0 ? 'bg-red-500 text-white' : 'bg-blue-500 text-white'}`}>
          {badge}
        </span>
      )}
    </button>
  );
});

export const MetricCard = React.memo(function MetricCard({ title, value, icon: Icon, trend, color = 'blue', subtitle, onClick }) {
  return (
    <div
      onClick={onClick}
      className={`bg-gray-800 rounded-xl p-6 border border-gray-700 transition-all duration-200 ${
        onClick ? 'cursor-pointer hover:bg-gray-750 hover:border-gray-600' : ''
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-gray-400 text-sm font-medium">{title}</h3>
        <Icon className={`w-5 h-5 text-${color}-400`} />
      </div>
      <div className="flex items-end justify-between">
        <div>
          <div className="text-2xl font-bold text-white">{value}</div>
          {subtitle && <div className="text-sm text-gray-400 mt-1">{subtitle}</div>}
        </div>
        {trend !== undefined && (
          <div className={`flex items-center text-sm ${trend > 0 ? 'text-green-400' : trend < 0 ? 'text-red-400' : 'text-gray-400'}`}>
            {trend > 0 ? <ArrowUp className="w-4 h-4 mr-1" /> : trend < 0 ? <ArrowDown className="w-4 h-4 mr-1" /> : null}
            {trend !== 0 && `${Math.abs(trend).toFixed(2)}%`}
          </div>
        )}
      </div>
    </div>
  );
});

export const AlertBadge = React.memo(function AlertBadge({ alerts, readAlerts, setReadAlerts, onClick }) {
  const safeAlerts = Array.isArray(alerts) ? alerts : [];
  const unreadAlerts = safeAlerts.filter(alert => !readAlerts.has((alert.timestamp ?? alert.time) + (alert.module ?? '')));
  const criticalCount = unreadAlerts.filter(a => a.severity === 'critical').length;

  if (safeAlerts.length === 0) return null;

  const handleClick = () => {
    const newReadAlerts = new Set(readAlerts);
    safeAlerts.forEach(alert => {
      newReadAlerts.add((alert.timestamp ?? alert.time) + (alert.module ?? ''));
    });
    setReadAlerts(newReadAlerts);
    onClick();
  };

  return (
    <button
      onClick={handleClick}
      className="relative flex items-center px-3 py-1 bg-red-600 text-white text-sm rounded-full hover:bg-red-700 transition-colors"
    >
      <Bell className="w-4 h-4 mr-1" />
      <span>{unreadAlerts.length > 0 ? unreadAlerts.length : safeAlerts.length}</span>
      {criticalCount > 0 && (
        <div className="absolute -top-1 -right-1 w-2 h-2 bg-yellow-400 rounded-full animate-pulse" />
      )}
      {unreadAlerts.length > 0 && (
        <div className="absolute -top-2 -right-2 w-4 h-4 bg-blue-400 rounded-full flex items-center justify-center text-xs font-bold">
          {unreadAlerts.length > 9 ? '9+' : unreadAlerts.length}
        </div>
      )}
    </button>
  );
});

export const ProgressBar = React.memo(function ProgressBar({ value, max, label, color = 'blue' }) {
  const percentage = (value / max) * 100;
  return (
    <div className="w-full">
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-white font-medium">{percentage.toFixed(1)}%</span>
      </div>
      <div className="w-full bg-gray-700 rounded-full h-2">
        <div className={`bg-${color}-500 h-2 rounded-full transition-all duration-300`} style={{ width: `${percentage}%` }} />
      </div>
    </div>
  );
});

export const ModeSelector = React.memo(function ModeSelector({ value, onChange }) {
  return (
    <div className="flex items-center space-x-4 bg-gray-900 p-1 rounded-lg">
      <button
        onClick={() => onChange('offline')}
        className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
          value === 'offline'
            ? 'bg-blue-600 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-800'
        }`}
      >
        <FileText className="w-4 h-4" />
        <span>Offline (CSV)</span>
      </button>
      <button
        onClick={() => onChange('online')}
        className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all ${
          value === 'online'
            ? 'bg-green-600 text-white'
            : 'text-gray-400 hover:text-white hover:bg-gray-800'
        }`}
      >
        <Radio className="w-4 h-4" />
        <span>Online (MT5 Live)</span>
      </button>
    </div>
  );
});

// ─────────────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────────────

export const getCategoryIcon = (category) => {
  const icons = {
    'meta': Brain,
    'core': Power,
    'executor': Zap,
    'risk': AlertTriangle,
    'voting': Users,
    'strategy': Brain,
    'features': Activity,
    'auditing': Eye,
    'memory': Brain,
    'market': TrendingUp,
    'trading': TrendingUp,
    'monitoring': Eye,
    'reward': TrendingUp,
    'simulation': Activity,
    'external': Power,
    'visualization': Eye,
    'position': Activity,
    'other': Circle
  };
  return icons[category] || Circle;
};

export const getCategoryColor = (category) => {
  const colors = {
    'core': 'from-blue-500 to-blue-600',
    'executor': 'from-yellow-500 to-yellow-600',
    'position': 'from-cyan-500 to-cyan-600',
    'risk': 'from-red-500 to-red-600',
    'voting': 'from-blue-500 to-blue-600',
    'strategy': 'from-purple-500 to-purple-600',
    'features': 'from-green-500 to-green-600',
    'auditing': 'from-yellow-500 to-yellow-600',
    'memory': 'from-indigo-500 to-indigo-600',
    'market': 'from-orange-500 to-orange-600',
    'market_theme': 'from-pink-500 to-pink-600',
    'trading': 'from-cyan-500 to-cyan-600',
    'monitoring': 'from-teal-500 to-teal-600',
    'reward': 'from-amber-500 to-amber-600',
    'simulation': 'from-emerald-500 to-emerald-600',
    'meta': 'from-violet-500 to-violet-600',
    'external': 'from-slate-500 to-slate-600',
    'visualization': 'from-rose-500 to-rose-600',
    'other': 'from-gray-500 to-gray-600'
  };
  return colors[category] || 'from-gray-500 to-gray-600';
};

export const getStatusColor = (status) => {
  const key = String(status || '').toLowerCase();
  const colors = {
    'active': 'text-green-400',
    'monitoring': 'text-blue-400',
    'analyzing': 'text-purple-400',
    'voting': 'text-cyan-400',
    'learning': 'text-orange-400',
    'scanning': 'text-yellow-400',
    'idle': 'text-gray-400',
    'error': 'text-red-400',
    'disabled': 'text-gray-500'
  };
  return colors[key] || 'text-gray-400';
};

export const getHealthColor = (score) => {
  if (score >= 90) return 'text-green-400';
  if (score >= 75) return 'text-green-300';
  if (score >= 60) return 'text-yellow-400';
  if (score >= 40) return 'text-orange-400';
  return 'text-red-400';
};

export const getHealthBarColor = (score) => {
  if (score >= 90) return 'bg-gradient-to-r from-green-500 to-green-400';
  if (score >= 75) return 'bg-gradient-to-r from-green-400 to-lime-400';
  if (score >= 60) return 'bg-gradient-to-r from-yellow-500 to-yellow-400';
  if (score >= 40) return 'bg-gradient-to-r from-orange-500 to-orange-400';
  return 'bg-gradient-to-r from-red-500 to-red-400';
};

// Safely convert to array
export const toArray = (val) => {
  if (Array.isArray(val)) return val;
  if (val && typeof val === 'object') return Object.values(val);
  return [];
};

// Enhanced Module Card Component
export const EnhancedModuleCard = React.memo(function EnhancedModuleCard({ module, onToggle, onClick }) {
  const safeModule = {
    name: module?.name || 'Unknown Module',
    id: module?.id ?? module?.name,
    category: module?.category || 'other',
    status: module?.status || 'unknown',
    enabled: module?.enabled || false,
    health_score: module?.health_score || 0,
    health_status: module?.health_status || 'unknown',
    data_richness: module?.data_richness || 0,
    provides_count: module?.provides_count || 0,
    requires_count: module?.requires_count || 0,
    error_count: module?.error_count || 0,
    has_errors: module?.has_errors || false,
    insights: module?.insights || null,
    live_data: module?.live_data || {},
    last_update: module?.last_update || null,
    ...module
  };

  const IconComponent = getCategoryIcon(safeModule.category);
  const categoryGradient = getCategoryColor(safeModule.category);
  const dataUtilization = Math.min(100, (safeModule.data_richness / Math.max(1, safeModule.provides_count)) * 100);
  
  const getStatusIndicator = (status) => {
    const s = status?.toUpperCase();
    if (s === 'ACTIVE' || s === 'PROCESSING') return { icon: Zap, color: 'text-green-400', pulse: true };
    if (s === 'MONITORING') return { icon: Eye, color: 'text-blue-400', pulse: true };
    if (s === 'ANALYZING') return { icon: Brain, color: 'text-purple-400', pulse: true };
    if (s === 'VOTING') return { icon: Users, color: 'text-cyan-400', pulse: true };
    if (s === 'WARNING') return { icon: AlertTriangle, color: 'text-yellow-400', pulse: true };
    if (s === 'ERROR') return { icon: XCircle, color: 'text-red-400', pulse: false };
    if (s === 'DISABLED') return { icon: Power, color: 'text-gray-500', pulse: false };
    return { icon: Circle, color: 'text-gray-400', pulse: false };
  };
  
  const statusInfo = getStatusIndicator(safeModule.status);
  const StatusIcon = statusInfo.icon;

  const getLiveDataPreview = () => {
    if (!safeModule.live_data || Object.keys(safeModule.live_data).length === 0) return null;
    const entries = Object.entries(safeModule.live_data).slice(0, 2);
    return entries.map(([key, value]) => {
      const shortKey = key.length > 20 ? key.slice(0, 18) + '...' : key;
      let displayVal = '—';
      if (typeof value === 'number') displayVal = value.toFixed(2);
      else if (typeof value === 'string') displayVal = value.slice(0, 12);
      else if (typeof value === 'boolean') displayVal = value ? '✓' : '✗';
      else if (value !== null && typeof value === 'object') displayVal = `{${Object.keys(value).length}}`;
      return { key: shortKey, value: displayVal };
    });
  };

  const livePreview = getLiveDataPreview();

  return (
    <div 
      className={`relative bg-gradient-to-br from-gray-800 to-gray-850 rounded-xl border transition-all duration-300 cursor-pointer group overflow-hidden ${
        safeModule.has_errors ? 'border-red-500/50 hover:border-red-400' :
        safeModule.enabled ? 'border-gray-700 hover:border-blue-500/70 hover:shadow-lg hover:shadow-blue-500/10' :
        'border-gray-700/50 hover:border-gray-600'
      }`}
      onClick={() => onClick(safeModule)}
    >
      <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${categoryGradient}`} />
      
      {statusInfo.pulse && safeModule.enabled && (
        <div className="absolute top-3 right-3">
          <span className="relative flex h-2 w-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${statusInfo.color.replace('text-', 'bg-')} opacity-75`}></span>
            <span className={`relative inline-flex rounded-full h-2 w-2 ${statusInfo.color.replace('text-', 'bg-')}`}></span>
          </span>
        </div>
      )}

      <div className="p-4">
        <div className="flex items-start justify-between mb-3">
          <div className="flex items-center space-x-3">
            <div className={`p-2.5 rounded-xl bg-gradient-to-br ${categoryGradient} text-white shadow-lg`}>
              <IconComponent size={20} />
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-white font-semibold text-sm group-hover:text-blue-400 transition-colors truncate">
                {safeModule.name.replace(/([A-Z])/g, ' $1').trim()}
              </h3>
              <div className="flex items-center space-x-2 mt-0.5">
                <span className="text-gray-500 text-xs capitalize">
                  {safeModule.category.replace('_', ' ')}
                </span>
                <span className="text-gray-600">•</span>
                <div className="flex items-center space-x-1">
                  <StatusIcon size={10} className={statusInfo.color} />
                  <span className={`text-xs font-medium ${statusInfo.color}`}>
                    {safeModule.status}
                  </span>
                </div>
              </div>
            </div>
          </div>
          <button
            onClick={(e) => {
              e.stopPropagation();
              onToggle(safeModule.name);
            }}
            className={`p-1.5 rounded-lg transition-all transform hover:scale-110 ${
              safeModule.enabled
                ? 'bg-green-500/20 hover:bg-green-500/30 text-green-400'
                : 'bg-gray-700 hover:bg-gray-600 text-gray-400'
            }`}
          >
            <Power size={14} />
          </button>
        </div>

        {safeModule.health_score !== undefined && (
          <div className="mb-3">
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center space-x-1">
                <Heart size={10} className={getHealthColor(safeModule.health_score)} />
                <span className="text-gray-500 text-xs">Health</span>
              </div>
              <span className={`text-xs font-bold ${getHealthColor(safeModule.health_score)}`}>
                {safeModule.health_score}%
              </span>
            </div>
            <div className="w-full bg-gray-700/50 rounded-full h-1.5 overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-700 ease-out ${getHealthBarColor(safeModule.health_score)}`}
                style={{ width: `${safeModule.health_score}%` }}
              />
            </div>
          </div>
        )}

        <div className="grid grid-cols-3 gap-2 mb-3">
          <div className="bg-gray-900/50 rounded-lg p-2 text-center">
            <div className="text-blue-400 font-bold text-sm">{safeModule.provides_count}</div>
            <div className="text-gray-500 text-xs">Outputs</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-2 text-center">
            <div className="text-green-400 font-bold text-sm">{safeModule.data_richness}</div>
            <div className="text-gray-500 text-xs">Live</div>
          </div>
          <div className="bg-gray-900/50 rounded-lg p-2 text-center">
            <div className={`font-bold text-sm ${safeModule.has_errors ? 'text-red-400' : 'text-gray-400'}`}>
              {safeModule.error_count}
            </div>
            <div className="text-gray-500 text-xs">Errors</div>
          </div>
        </div>

        {livePreview && livePreview.length > 0 && (
          <div className="bg-gray-900/30 rounded-lg p-2 mb-3 border border-gray-700/50">
            <div className="flex items-center space-x-1 mb-1.5">
              <Activity size={10} className="text-green-400" />
              <span className="text-gray-400 text-xs font-medium">Live Data</span>
            </div>
            <div className="space-y-1">
              {livePreview.map(({ key, value }) => (
                <div key={key} className="flex items-center justify-between">
                  <span className="text-gray-500 text-xs truncate flex-1">{key}</span>
                  <span className="text-cyan-400 text-xs font-mono ml-2">{value}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {safeModule.insights && safeModule.insights.summary && safeModule.insights.summary !== "No data available" && (
          <div className="bg-gradient-to-r from-blue-900/20 to-purple-900/20 rounded-lg p-2 mb-3 border border-blue-500/20">
            <div className="flex items-start space-x-2">
              <Brain size={12} className="text-purple-400 mt-0.5 flex-shrink-0" />
              <p className="text-xs text-gray-300 leading-relaxed line-clamp-2">
                {safeModule.insights.summary}
              </p>
            </div>
          </div>
        )}

        <div className="mb-3">
          <div className="flex items-center justify-between mb-1">
            <span className="text-gray-500 text-xs">Data Flow</span>
            <span className="text-xs font-medium text-gray-400">
              {safeModule.data_richness}/{safeModule.provides_count} streams
            </span>
          </div>
          <div className="w-full bg-gray-700/30 rounded-full h-1">
            <div
              className="bg-gradient-to-r from-blue-500 via-cyan-500 to-green-500 h-1 rounded-full transition-all duration-500"
              style={{ width: `${dataUtilization}%` }}
            />
          </div>
        </div>

        <button
          onClick={(e) => {
            e.stopPropagation();
            onClick(safeModule);
          }}
          className="w-full py-2 bg-gray-700/50 hover:bg-blue-600/30 text-gray-300 hover:text-white text-xs rounded-lg transition-all duration-200 flex items-center justify-center space-x-1 group/btn"
        >
          <span>View Details</span>
          <ChevronRight size={12} className="transform group-hover/btn:translate-x-0.5 transition-transform" />
        </button>
      </div>
    </div>
  );
});
