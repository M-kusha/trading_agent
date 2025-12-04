import React, {
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
  startTransition,
  useTransition,
} from 'react';
import { createPortal } from 'react-dom';

import {
  Play, Pause, Square, AlertTriangle, TrendingUp, Activity, Database, Settings, LogOut, Brain, Heart,
  Shield, Target, BarChart3, Users, Zap, CheckCircle, XCircle, Clock, DollarSign, Vote,
  ArrowUp, ArrowDown, Wifi, WifiOff, Save, Upload, Download, RefreshCw, AlertCircle,
  Cpu, HardDrive, Network, Eye, BarChart2, PieChart as PieChartIcon, LineChart as LineChartIcon, Layers, Bell, X,
  TrendingDown, Percent, Timer, Gauge, Monitor, Server, CloudOff, Power, FileUp,
  Loader2, CheckSquare, Radio, UploadCloud, FolderOpen, FileText, ToggleLeft, ToggleRight,
  Search, Filter, Grid, List, Hash, Sparkles, Flame, Waves, Zap as Lightning, Crown,
  Circle, ChevronRight, BookOpen, Lightbulb, Swords, Dna, Minus
} from 'lucide-react';

import {
  LineChart as RechartsLineChart, Line, AreaChart, Area, BarChart, Bar,
  PieChart as RechartsPieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar,
  ComposedChart, Scatter
} from 'recharts';

// Tab components extracted to ./components/tabs/
import {
  AnalyticsTab,
  MemoryTab,
  RiskTab,
  StrategyTab,
  VotingTab,
  TradingTab,
  LogsTab,
  AlertsModal
} from './components/tabs';

// State management
import {
  useUIState,
  useUIDispatch,
  useTradingState,
  useTradingDispatch,
  useModulesState,
  useModulesDispatch,
  useDataState,
  useDataDispatch,
  UIActions,
  TradingActions,
  ModulesActions,
  DataActions,
  apiCall,
  useAppState,
} from './store';

const API_BASE = '/api';

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
// Error Boundary
// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  componentDidCatch(error, errorInfo) {
    console.error('Error caught by boundary:', error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center">
          <div className="bg-red-900/20 border border-red-500 rounded-lg p-6 max-w-md">
            <h2 className="text-red-400 text-xl font-bold mb-2">Application Error</h2>
            <p className="text-gray-300 mb-4">
              Something went wrong. Please refresh the page to continue.
            </p>
            <button
              onClick={() => window.location.reload()}
              className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-white"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}


const formatTime = (seconds) => {
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
};

const toArray = (v) =>
  Array.isArray(v) ? v : (v && typeof v === 'object') ? Object.values(v) : [];

// Normalize possibly structured values to a human-readable string
const toText = (v, fallback = '') => {
  try {
    if (v == null) return fallback;
    if (typeof v === 'string') return v;
    if (typeof v === 'number' || typeof v === 'boolean') return String(v);
    if (Array.isArray(v)) {
      if (v.length === 0) return fallback;
      return toText(v[0], fallback);
    }
    if (typeof v === 'object') {
      for (const k of ['label', 'name', 'regime', 'state', 'value']) {
        if (k in v && typeof v[k] === 'string') return v[k];
      }
      const keys = Object.keys(v);
      if (keys.length > 0 && typeof v[keys[0]] === 'string') return v[keys[0]];
    }
  } catch {}
  return fallback || String(v);
};


const getCategoryIcon = (category) => {
  const icons = {
    'risk': Shield,
    'voting': Users,
    'strategy': Brain,
    'features': Sparkles,
    'auditing': Eye,
    'memory': Database,
    'market': TrendingUp,
    // If a distinct thematic variant is ever introduced, use 'market_theme'
    'market_theme': Flame,
    'trading': Target,
    'monitoring': Monitor,
    'reward': Crown,
    'simulation': Waves,
    'meta': Lightning,
    'external': Hash,
    'visualization': BarChart3,
    'other': Settings
  };
  return icons[category] || Settings;
};

const getCategoryColor = (category) => {
  const colors = {
    'risk': 'from-red-500 to-red-600',
    'voting': 'from-blue-500 to-blue-600',
    'strategy': 'from-purple-500 to-purple-600',
    'features': 'from-green-500 to-green-600',
    'auditing': 'from-yellow-500 to-yellow-600',
    'memory': 'from-indigo-500 to-indigo-600',
    'market': 'from-orange-500 to-orange-600',
    // Optional thematic variant if needed later
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

const getStatusColor = (status) => {
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

const getHealthColor = (score) => {
  if (score >= 90) return 'text-green-400';
  if (score >= 75) return 'text-green-300';
  if (score >= 60) return 'text-yellow-400';
  if (score >= 40) return 'text-orange-400';
  return 'text-red-400';
};

const getHealthBarColor = (score) => {
  if (score >= 90) return 'bg-gradient-to-r from-green-500 to-green-400';
  if (score >= 75) return 'bg-gradient-to-r from-green-400 to-lime-400';
  if (score >= 60) return 'bg-gradient-to-r from-yellow-500 to-yellow-400';
  if (score >= 40) return 'bg-gradient-to-r from-orange-500 to-orange-400';
  return 'bg-gradient-to-r from-red-500 to-red-400';
};

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
/** UI Atoms (hoisted & memoized) */
// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

const StatusIndicator = React.memo(function StatusIndicator({ status, className = '' }) {
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

const TabButton = React.memo(function TabButton({ icon: Icon, label, active, onClick, badge, disabled = false }) {
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

const MetricCard = React.memo(function MetricCard({ title, value, icon: Icon, trend, color = 'blue', subtitle, onClick }) {
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

const EnhancedModuleCard = React.memo(function EnhancedModuleCard({ module, onToggle, onClick }) {
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
  const statusColor = getStatusColor(safeModule.status);
  const dataUtilization = Math.min(100, (safeModule.data_richness / Math.max(1, safeModule.provides_count)) * 100);
  
  // Get status icon and pulse animation
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

  // Format live data preview
  const getLiveDataPreview = () => {
    if (!safeModule.live_data || Object.keys(safeModule.live_data).length === 0) return null;
    const entries = Object.entries(safeModule.live_data).slice(0, 2);
    return entries.map(([key, value]) => {
      const shortKey = key.length > 20 ? key.slice(0, 18) + '...' : key;
      let displayVal = 'â€”';
      if (typeof value === 'number') displayVal = value.toFixed(2);
      else if (typeof value === 'string') displayVal = value.slice(0, 12);
      else if (typeof value === 'boolean') displayVal = value ? 'âœ“' : 'âœ—';
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
      {/* Top gradient accent based on category */}
      <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${categoryGradient}`} />
      
      {/* Status pulse indicator */}
      {statusInfo.pulse && safeModule.enabled && (
        <div className="absolute top-3 right-3">
          <span className="relative flex h-2 w-2">
            <span className={`animate-ping absolute inline-flex h-full w-full rounded-full ${statusInfo.color.replace('text-', 'bg-')} opacity-75`}></span>
            <span className={`relative inline-flex rounded-full h-2 w-2 ${statusInfo.color.replace('text-', 'bg-')}`}></span>
          </span>
        </div>
      )}

      <div className="p-4">
        {/* Header */}
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
                <span className="text-gray-600">â€¢</span>
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

        {/* Health Bar with gradient */}
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

        {/* Stats Grid */}
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

        {/* Live Data Preview */}
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

        {/* Insights Summary */}
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

        {/* Data Utilization */}
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

        {/* Footer with action button */}
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

const AlertBadge = React.memo(function AlertBadge({ alerts, readAlerts, setReadAlerts, onClick }) {
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

const ProgressBar = React.memo(function ProgressBar({ value, max, label, color = 'blue' }) {
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

const ModeSelector = React.memo(function ModeSelector({ value, onChange }) {
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

// Tabs (AnalyticsTab, MemoryTab, RiskTab, StrategyTab, VotingTab, TradingTab, LogsTab, AlertsModal)
// are imported from ./components/tabs/ - see imports at top of file


const OverviewTab = React.memo(function OverviewTab({
  performance,
  systemStatus,
  systemState,
  moduleStates,
  alerts,
  stopTrading,
  emergencyStop,
  readAlerts,
  setReadAlerts,
  accountInfo,
}) {
  const { state: appState, dispatch: appDispatch } = useAppState();
  const { selectedSymbol, mt5ChartData, recentTrades, mt5Symbols, selectedTimeframe } = appState;

  const [overviewAnalytics, setOverviewAnalytics] = useState({});
  const prevTfRef = useRef(appState.selectedTimeframe);

  const fetchMt5Data = useCallback(async (symbolParam = selectedSymbol) => {
    const tfMap = { '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30', '1h': 'H1', '4h': 'H4', '1d': 'D1', '1w': 'D1' };
    const tfParam = tfMap[selectedTimeframe] || 'M5';
    const symbol = (symbolParam === 'XAU_USD' ? 'XAUUSD' : symbolParam) || 'EURUSD';

    const now = Date.now();
    const tfChanged = prevTfRef.current !== selectedTimeframe;
    if (!tfChanged && (now - appState.lastUpdate.mt5) < 10000) return;

    try {
      const chartResponse = await fetch(`/api/mt5/chart-data/${symbol}?timeframe=${tfParam}&count=150`);
      if (chartResponse.ok) {
        const chartData = await chartResponse.json();
        if (chartData.success) {
          appDispatch({
            type: 'SET_MT5_DATA',
            payload: { chartData: chartData.data || [] }
          });
          prevTfRef.current = selectedTimeframe;
        }
      }

      const tradesResponse = await fetch('/api/mt5/deals/recent?limit=4');
      if (tradesResponse.ok) {
        const tradesData = await tradesResponse.json();
        if (tradesData.success) {
          appDispatch({
            type: 'SET_MT5_DATA',
            payload: { recentTrades: tradesData.deals || [] }
          });
        }
      }

      const symbolsResponse = await fetch('/api/trading/symbols');
      if (symbolsResponse.ok) {
        const symbolsData = await symbolsResponse.json();
        if (symbolsData.success) {
          appDispatch({
            type: 'SET_MT5_DATA',
            payload: { symbols: symbolsData.symbols || [] }
          });
        }
      }

      const analyticsResponse = await fetch('/api/visualization-data');
      if (analyticsResponse.ok) {
        const analyticsData = await analyticsResponse.json();
        if (analyticsData.success) {
          setOverviewAnalytics(prev => ({ ...prev, ...analyticsData }));
        }
      }

      const performanceResponse = await fetch('/api/performance/chart-data');
      if (performanceResponse.ok) {
        const performanceData = await performanceResponse.json();
        if (performanceData.success) {
          setOverviewAnalytics(prev => ({
            ...prev,
            performanceChart: performanceData.data,
            performanceMetadata: performanceData.metadata
          }));
        }
      }
    } catch (error) {
      console.error('Failed to fetch MT5 data:', error);
    }
  }, [selectedSymbol, selectedTimeframe, appState.lastUpdate.mt5, appDispatch]);

  useEffect(() => {
    fetchMt5Data(selectedSymbol);
  }, [selectedSymbol, selectedTimeframe, appState.dataLoaded.mt5Data, fetchMt5Data]);

  // Keep chart updating periodically (polling cadence based on timeframe)
  useEffect(() => {
    const cadence = { '1m': 5000, '5m': 10000, '15m': 30000, '30m': 60000, '1h': 120000, '4h': 300000, '1d': 600000 };
    const period = cadence[selectedTimeframe] || 15000;
    const id = setInterval(() => {
      fetchMt5Data(selectedSymbol);
    }, period);
    return () => clearInterval(id);
  }, [selectedSymbol, selectedTimeframe, fetchMt5Data]);

  const candlestickData = mt5ChartData;

  const enhancedPerformance = {
    current_balance: Number(performance?.current_balance) || 10000,
    start_balance: Number(performance?.start_balance) || 10000,
    total_pnl: Number(performance?.total_pnl) || 0,
    daily_pnl: Number(performance?.daily_pnl) || 0,
    win_rate: Number(performance?.win_rate) || 0,
    total_trades: Number(performance?.total_trades) || 0,
    winning_trades: Number(performance?.winning_trades) || 0,
    max_drawdown: Number(performance?.max_drawdown) || 0,
    current_drawdown: Number(performance?.current_drawdown) || 0,
    sharpe_ratio: Number(performance?.sharpe_ratio) || 1.2,
    profit_factor: Number(performance?.profit_factor) || 1.5
  };

  // Prefer live Trading Mode module data; fall back to analytics if missing
  const tradingModeState = (
    moduleStates?.trading_mode_manager ||
    moduleStates?.trading_modes ||
    moduleStates?.TradingModeManager ||
    moduleStates?.tradingModeManager ||
    null
  );

  const currentTradingMode = toText(
    tradingModeState?.trading_mode || tradingModeState?.current_mode || overviewAnalytics?.trading_analytics?.trading_mode,
    null
  )?.toLowerCase?.();

  const modeToRisk = {
    conservative: 'LOW',
    balanced: 'MODERATE',
    aggressive: 'HIGH',
    precision: 'LOW',
    swing: 'MODERATE',
    momentum: 'HIGH',
    scalping: 'HIGH',
    shelter: 'LOW',
    rescue: 'HIGH',
    normal: 'MODERATE'
  };

  const resolvedRiskLevel = currentTradingMode
    ? (modeToRisk[currentTradingMode] || currentTradingMode?.toUpperCase?.())
    : toText(overviewAnalytics?.trading_analytics?.risk_level, 'MODERATE');

  const resolvedAutoMode = (typeof tradingModeState?.auto_mode === 'boolean')
    ? tradingModeState.auto_mode
    : (overviewAnalytics?.trading_analytics?.auto_mode !== false);

  const tradingModes = {
    current: systemStatus === 'TRADING' ? 'ACTIVE' : systemStatus === 'TRAINING' ? 'LEARNING' : 'IDLE',
    strategy: toText(overviewAnalytics?.trading_analytics?.active_strategy, 'Adaptive Multi-Timeframe'),
    risk_level: resolvedRiskLevel,
    auto_mode: resolvedAutoMode,
    regime: toText(overviewAnalytics?.regime_analytics?.current_regime, 'TRENDING'),
    mode: (currentTradingMode || '').toUpperCase()
  };

  const moduleStatus = Object.entries(moduleStates).reduce((acc, [name, state]) => {
    acc[state.enabled ? 'active' : 'inactive'] = (acc[state.enabled ? 'active' : 'inactive'] || 0) + 1;
    return acc;
  }, { active: 0, inactive: 0, total: Object.keys(moduleStates).length });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white flex items-center">
            <Crown className="w-8 h-8 mr-3 text-yellow-400" />
            AI Trading Command Center
          </h2>
          <div className="flex items-center space-x-6 mt-2">
            <p className="text-gray-400 flex items-center">
              <span className="mr-1">Uptime:</span>
              <LiveUptimeDisplay fallbackUptime={systemState?.uptime || '0m'} />
            </p>
            <p className="text-gray-400">
              <Hash className="w-4 h-4 inline mr-1" />
              Session: {systemState?.session_id?.slice(-8) || 'Unknown'}
            </p>
            <div className={`flex items-center px-3 py-1 rounded-full text-xs font-medium ${
              systemStatus === 'TRADING' ? 'bg-green-500/20 text-green-400' :
              systemStatus === 'TRAINING' ? 'bg-blue-500/20 text-blue-400' :
              'bg-gray-500/20 text-gray-400'
            }`}>
              <div className={`w-2 h-2 rounded-full mr-2 ${
                systemStatus === 'TRADING' ? 'bg-green-400 animate-pulse' :
                systemStatus === 'TRAINING' ? 'bg-blue-400 animate-pulse' :
                'bg-gray-400'
              }`} />
              {tradingModes.current}
            </div>
          </div>
        </div>

        <div className="flex items-center space-x-4">
          <AlertBadge alerts={alerts} readAlerts={readAlerts} setReadAlerts={setReadAlerts} onClick={() => appDispatch({ type: 'SET_SHOW_ALERTS', payload: true })} />

          {systemStatus === 'TRADING' && (
            <>
              <button
                onClick={stopTrading}
                className="flex items-center space-x-2 bg-gradient-to-r from-yellow-600 to-yellow-700 hover:from-yellow-700 hover:to-yellow-800 px-4 py-2 rounded-lg transition-all duration-200 text-white font-medium shadow-lg hover:shadow-xl"
              >
                <Pause className="w-4 h-4" />
                <span>Stop Trading</span>
              </button>
              <button
                onClick={emergencyStop}
                className="flex items-center space-x-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 px-4 py-2 rounded-lg transition-all duration-200 text-white font-medium shadow-lg hover:shadow-xl"
              >
                <AlertTriangle className="w-4 h-4" />
                <span>Emergency Stop</span>
              </button>
            </>
          )}
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <div className="bg-gradient-to-br from-purple-600/20 to-purple-800/20 border border-purple-500/30 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            <span className="text-xs text-purple-300 font-medium">STRATEGY</span>
          </div>
          <div className="text-white font-bold text-sm">{tradingModes.strategy}</div>
          <div className="text-purple-300 text-xs mt-1">Regime: {tradingModes.regime}</div>
        </div>

        <div className="bg-gradient-to-br from-blue-600/20 to-blue-800/20 border border-blue-500/30 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <Shield className="w-5 h-5 text-blue-400" />
            <span className="text-xs text-blue-300 font-medium">RISK</span>
          </div>
          <div className="text-white font-bold text-sm">{tradingModes.risk_level}</div>
          <div className="text-blue-300 text-xs mt-1">Auto: {tradingModes.auto_mode ? 'ON' : 'OFF'}</div>
        </div>

        <div className="bg-gradient-to-br from-green-600/20 to-green-800/20 border border-green-500/30 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <Cpu className="w-5 h-5 text-green-400" />
            <span className="text-xs text-green-300 font-medium">MODULES</span>
          </div>
          <div className="text-white font-bold text-sm">{moduleStatus.active}/{moduleStatus.total}</div>
          <div className="text-green-300 text-xs mt-1">Active Modules</div>
        </div>

        <div className="bg-gradient-to-br from-yellow-600/20 to-yellow-800/20 border border-yellow-500/30 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <Wifi className="w-5 h-5 text-yellow-400" />
            <span className="text-xs text-yellow-300 font-medium">MT5</span>
          </div>
          <div className="text-white font-bold text-sm">
            {systemState?.mt5_connected ? 'CONNECTED' : 'OFFLINE'}
          </div>
          <div className="text-yellow-300 text-xs mt-1">
            {accountInfo ? `${accountInfo.company || 'MetaTrader 5'}` : 'MetaTrader 5'}
          </div>
          {accountInfo && (
            <div className="text-yellow-200 text-xs mt-1">
              ${accountInfo.balance?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}
            </div>
          )}
        </div>

        <div className="bg-gradient-to-br from-red-600/20 to-red-800/20 border border-red-500/30 rounded-xl p-4">
          <div className="flex items-center justify-between mb-2">
            <Gauge className="w-5 h-5 text-red-400" />
            <span className="text-xs text-red-300 font-medium">ALERTS</span>
          </div>
          <div className="text-white font-bold text-sm">{alerts.length}</div>
          <div className="text-red-300 text-xs mt-1">Active Alerts</div>
        </div>
      </div>

      {/* Live Prices ticker */}
      {mt5Symbols.length > 0 && (
        <div className="bg-gray-800/50 rounded-xl p-4 border border-gray-700/50">
          <div className="flex items-center space-x-4 overflow-x-auto scrollbar-hide">
            <div className="flex items-center space-x-2 flex-shrink-0">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
              <span className="text-gray-400 text-sm font-medium">LIVE PRICES</span>
            </div>
            {(() => {
              const visible = (Array.isArray(mt5Symbols) ? mt5Symbols : []).filter(s => s && ['EURUSD','XAUUSD'].includes(String(s.symbol || '').toUpperCase()));
              return visible.map((symbol) => (
              <div key={symbol.symbol} className="flex items-center space-x-3 px-4 py-2 bg-gray-700/30 rounded-lg flex-shrink-0">
                <div className="text-white font-bold text-sm">{String(symbol.symbol)}</div>
                <div className="text-gray-300 text-xs">
                  <span className="text-blue-400">{(symbol.bid || 0).toFixed(symbol.symbol.includes('JPY') ? 3 : 5)}</span>
                  <span className="mx-1 text-gray-500">/</span>
                  <span className="text-red-400">{(symbol.ask || 0).toFixed(symbol.symbol.includes('JPY') ? 3 : 5)}</span>
                </div>
                <div className="text-xs text-gray-400">
                  {symbol.spread || 0}pts
                </div>
              </div>
              ));
            })()}
          </div>
        </div>
      )}

      {/* Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Account Balance"
          value={`$${enhancedPerformance.current_balance?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}`}
          icon={DollarSign}
          trend={enhancedPerformance.total_pnl ? (enhancedPerformance.total_pnl / enhancedPerformance.start_balance * 100) : 0}
          color="green"
          subtitle={`Start: $${enhancedPerformance.start_balance?.toLocaleString() || '0.00'}`}
        />
        <MetricCard
          title="Total P&L"
          value={`$${enhancedPerformance.total_pnl?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}`}
          icon={TrendingUp}
          color={enhancedPerformance.total_pnl >= 0 ? 'green' : 'red'}
          subtitle={`Daily: $${enhancedPerformance.daily_pnl?.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }) || '0.00'}`}
          trend={enhancedPerformance.daily_pnl ? (enhancedPerformance.daily_pnl / enhancedPerformance.current_balance * 100) : 0}
        />
        <MetricCard
          title="Win Rate"
          value={`${(enhancedPerformance.win_rate * 100 || 0).toFixed(1)}%`}
          icon={Target}
          subtitle={`${enhancedPerformance.winning_trades || 0}/${enhancedPerformance.total_trades || 0} trades`}
          color="blue"
          trend={enhancedPerformance.win_rate ? (enhancedPerformance.win_rate - 0.5) * 200 : 0}
        />
        <MetricCard
          title="Drawdown"
          value={`${(enhancedPerformance.current_drawdown * 100 || 0).toFixed(1)}%`}
          icon={TrendingDown}
          color="purple"
          subtitle={`Max: ${(enhancedPerformance.max_drawdown * 100 || 0).toFixed(1)}%`}
          trend={-(enhancedPerformance.current_drawdown * 100 || 0)}
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Live Chart */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 shadow-2xl lg:col-span-2">
          <div className="mb-3">
            <h3 className="text-lg font-semibold flex items-center text-white mb-2">
              <BarChart3 className="w-5 h-5 mr-2 text-green-400" />
              {selectedSymbol} Live Chart ({(() => { const m = { '1m':'M1','5m':'M5','15m':'M15','30m':'M30','1h':'H1','4h':'H4','1d':'D1', '1w':'D1'}; return m[selectedTimeframe] || 'M5'; })()})
            </h3>
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="flex items-center space-x-3">
                <select
                  value={selectedSymbol}
                  onChange={(e) => appDispatch({ type: 'SET_SYMBOL', payload: e.target.value })}
                  className="bg-gray-700 text-white px-3 py-1 rounded text-sm border border-gray-600 focus:border-green-400 focus:outline-none"
                >
                  <option value="EURUSD">EUR_USD</option>
                  <option value="XAUUSD">XAU_USD</option>
                </select>
                <div className="flex items-center space-x-1">
                  {['1m','5m','15m','30m','1h','4h','1d'].map(tf => (
                    <button
                      key={tf}
                      onClick={() => appDispatch({ type: 'SET_TIMEFRAME', payload: tf })}
                      className={`px-2 py-1 rounded text-xs font-medium ${selectedTimeframe === tf ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
                    >
                      {tf}
                    </button>
                  ))}
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                <span className="text-xs text-gray-400">LIVE</span>
              </div>
            </div>
          </div>
          <div className="h-[500px]">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={candlestickData.map(item => ({
                  ...item,
                  // Fix timezone - display local time
                  time: new Date(item.timestamp_ms || Date.now()).toLocaleTimeString('en-US', {
                    hour12: false,
                    hour: '2-digit',
                    minute: '2-digit',
                    timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
                  })
                }))}
                margin={{ top: 30, right: 70, left: 20, bottom: 30 }}
              >
                <defs>
                  <linearGradient id="volumeGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#10B981" stopOpacity={0.4} />
                    <stop offset="100%" stopColor="#10B981" stopOpacity={0.1} />
                  </linearGradient>
                </defs>

                <CartesianGrid strokeDasharray="2 4" stroke="#374151" opacity={0.2} />

                <XAxis
                  dataKey="time"
                  stroke="#9CA3AF"
                  fontSize={12}
                  tick={{ fill: '#9CA3AF', fontSize: 11 }}
                  axisLine={{ stroke: '#4B5563' }}
                  tickLine={{ stroke: '#4B5563' }}
                  interval="preserveStartEnd"
                />

                <YAxis
                  yAxisId="price"
                  stroke="#9CA3AF"
                  fontSize={12}
                  tick={{ fill: '#9CA3AF', fontSize: 11 }}
                  axisLine={{ stroke: '#4B5563' }}
                  tickLine={{ stroke: '#4B5563' }}
                  domain={['dataMin - 0.0005', 'dataMax + 0.0005']}
                  orientation="right"
                  width={60}
                />

                <YAxis
                  yAxisId="volume"
                  stroke="#9CA3AF"
                  fontSize={11}
                  tick={{ fill: '#6B7280', fontSize: 10 }}
                  axisLine={{ stroke: '#4B5563' }}
                  tickLine={{ stroke: '#4B5563' }}
                  orientation="left"
                  domain={[0, 'dataMax * 2.5']}
                  width={50}
                />

                <Tooltip
                  contentStyle={{
                    backgroundColor: '#111827',
                    border: '1px solid #374151',
                    borderRadius: '12px',
                    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                    padding: '12px',
                    color: '#F9FAFB'
                  }}
                  content={({ active, payload, label }) => {
                    if (!active || !payload || !payload[0]) return null;
                    const data = payload[0].payload;
                    const isGold = String(selectedSymbol).toUpperCase().includes('XAU');
                    const precision = isGold ? 2 : 5;

                    // Fix timezone for tooltip
                    const localTime = new Date(data.timestamp_ms || Date.now()).toLocaleString('en-US', {
                      year: 'numeric',
                      month: 'short',
                      day: 'numeric',
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                      timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
                    });

                    return (
                      <div className="bg-gray-900 border border-gray-600 rounded-lg p-3 shadow-xl">
                        <p className="text-gray-300 text-sm mb-2">{localTime}</p>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between gap-4">
                            <span className="text-gray-400">Open:</span>
                            <span className="text-white font-mono">{data.open?.toFixed(precision)}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span className="text-gray-400">High:</span>
                            <span className="text-green-400 font-mono">{data.high?.toFixed(precision)}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span className="text-gray-400">Low:</span>
                            <span className="text-red-400 font-mono">{data.low?.toFixed(precision)}</span>
                          </div>
                          <div className="flex justify-between gap-4">
                            <span className="text-gray-400">Close:</span>
                            <span className={`font-mono ${data.close >= data.open ? 'text-green-400' : 'text-red-400'}`}>
                              {data.close?.toFixed(precision)}
                            </span>
                          </div>
                          <div className="flex justify-between gap-4 pt-1 border-t border-gray-600">
                            <span className="text-gray-400">Volume:</span>
                            <span className="text-blue-400 font-mono">{data.volume?.toLocaleString()}</span>
                          </div>
                        </div>
                      </div>
                    );
                  }}
                />

                {/* Volume bars */}
                <Bar
                  yAxisId="volume"
                  dataKey="volume"
                  fill="url(#volumeGradient)"
                  opacity={0.25}
                  isAnimationActive={false}
                  maxBarSize={3}
                />

                {/* Candlestick representation using Scatter with custom shape */}
                <Scatter
                  yAxisId="price"
                  dataKey="close"
                  shape={(props) => {
                    const { cx, cy, payload } = props;
                    if (!payload) return null;

                    const { open, high, low, close } = payload;
                    const isGreen = close >= open;

                    // Calculate Y-axis scale with larger chart height
                    const yAxisDomain = [Math.min(...candlestickData.map(d => Math.min(d.open, d.high, d.low, d.close))),
                                         Math.max(...candlestickData.map(d => Math.max(d.open, d.high, d.low, d.close)))];
                    const yAxisRange = yAxisDomain[1] - yAxisDomain[0];
                    const chartHeight = 400; // larger chart height
                    const pixelsPerUnit = chartHeight / yAxisRange;

                    // Calculate positions
                    const highY = cy - (high - close) * pixelsPerUnit;
                    const lowY = cy + (close - low) * pixelsPerUnit;
                    const openY = cy - (open - close) * pixelsPerUnit;

                    const bodyTop = Math.min(cy, openY);
                    const bodyBottom = Math.max(cy, openY);
                    const bodyHeight = Math.abs(bodyBottom - bodyTop);

                    return (
                      <g key={`candle-${cx}`}>
                        {/* High-Low Wick */}
                        <line
                          x1={cx}
                          y1={highY}
                          x2={cx}
                          y2={lowY}
                          stroke={isGreen ? '#10B981' : '#EF4444'}
                          strokeWidth={1.2}
                        />

                        {/* Open-Close Body */}
                        <rect
                          x={cx - 5}
                          y={bodyTop}
                          width={10}
                          height={Math.max(bodyHeight, 1.5)}
                          fill={isGreen ? 'transparent' : '#EF4444'}
                          stroke={isGreen ? '#10B981' : '#EF4444'}
                          strokeWidth={1.8}
                        />
                      </g>
                    );
                  }}
                  isAnimationActive={false}
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Trading Information - Real MT5 Data Only */}
      <div className="mt-8 grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Real Recent Trades */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center text-white">
            <Activity className="w-5 h-5 mr-2 text-yellow-400" />
            Recent Trades (MT5)
          </h3>
          <div className="space-y-3">
            {Array.isArray(recentTrades) && recentTrades.length > 0 ? recentTrades.map((trade, index) => (
              <div key={trade.ticket ?? index} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                <div className="flex items-center space-x-3">
                  <div className={`w-2 h-2 rounded-full ${trade.profit >= 0 ? 'bg-green-400' : 'bg-red-400'}`} />
                  <div>
                    <div className="text-white font-medium text-sm">{trade.symbol}</div>
                    <div className="text-gray-400 text-xs">
                      {new Date(trade.time * 1000).toLocaleString('en-US', {
                        month: 'short',
                        day: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                        timeZone: Intl.DateTimeFormat().resolvedOptions().timeZone
                      })}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className={`text-sm font-medium ${trade.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {trade.profit >= 0 ? '+' : ''}${trade.profit.toFixed(2)}
                  </div>
                  <div className="text-gray-400 text-xs">{trade.volume} lots</div>
                </div>
              </div>
            )) : (
              <div className="text-center py-8 text-gray-400">
                <Activity className="w-8 h-8 mx-auto mb-3 opacity-50" />
                <p className="text-sm">No recent trades</p>
                <p className="text-xs mt-1">Connect MT5 for live data</p>
              </div>
            )}
          </div>
        </div>

        {/* Live Market Analysis */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center text-white">
            <TrendingUp className="w-5 h-5 mr-2 text-blue-400" />
            Market Analysis
          </h3>
          <div className="space-y-4">
            <div className="bg-gray-700/50 p-3 rounded-lg">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-white">{selectedSymbol}</span>
                <span className={`text-sm font-bold ${candlestickData.length > 0 && candlestickData[candlestickData.length - 1]?.close >= candlestickData[candlestickData.length - 1]?.open ? 'text-green-400' : 'text-red-400'}`}>
                  {candlestickData.length > 0 && candlestickData[candlestickData.length - 1]?.close >= candlestickData[candlestickData.length - 1]?.open ? 'â†— BULLISH' : 'â†˜ BEARISH'}
                </span>
              </div>
              <div className="text-xs text-gray-400">
                {selectedTimeframe} â€¢ {candlestickData.length} candles â€¢ Live MT5 Data
              </div>
            </div>

            {candlestickData.length > 0 && (
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">Current:</span>
                  <span className="text-white font-mono text-sm">
                    {candlestickData[candlestickData.length - 1]?.close?.toFixed(String(selectedSymbol).includes('XAU') ? 2 : 5)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">High:</span>
                  <span className="text-green-400 font-mono text-sm">
                    {Math.max(...candlestickData.map(d => d.high)).toFixed(String(selectedSymbol).includes('XAU') ? 2 : 5)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400 text-sm">Low:</span>
                  <span className="text-red-400 font-mono text-sm">
                    {Math.min(...candlestickData.map(d => d.low)).toFixed(String(selectedSymbol).includes('XAU') ? 2 : 5)}
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Trading Sessions */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center text-white">
            <Clock className="w-5 h-5 mr-2 text-purple-400" />
            Trading Sessions
          </h3>
          <div className="space-y-3">
            {(() => {
              const now = new Date();
              const utcHour = now.getUTCHours();
              const sessions = [
                { name: 'Tokyo', start: 0, end: 9, active: utcHour >= 0 && utcHour < 9 },
                { name: 'London', start: 8, end: 17, active: utcHour >= 8 && utcHour < 17 },
                { name: 'New York', start: 13, end: 22, active: utcHour >= 13 && utcHour < 22 },
                { name: 'Sydney', start: 22, end: 7, active: utcHour >= 22 || utcHour < 7 }
              ];

              return sessions.map(session => (
                <div key={session.name} className={`flex justify-between items-center p-2 rounded ${session.active ? 'bg-gray-700/50' : ''}`}>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${session.active ? 'bg-green-400' : 'bg-gray-600'}`} />
                    <span className={`text-sm ${session.active ? 'text-white font-medium' : 'text-gray-400'}`}>
                      {session.name}
                    </span>
                  </div>
                  <span className={`text-xs ${session.active ? 'text-green-400' : 'text-gray-500'}`}>
                    {session.active ? 'ACTIVE' : 'CLOSED'}
                  </span>
                </div>
              ));
            })()}
          </div>
        </div>

      </div>
    </div>
  );
});

const ModulesTab = React.memo(function ModulesTab({
  modules,
  moduleCategories,
  moduleStats,
  onToggle,
  onRefresh,
  selectedModule,
  onSelectModule,
}) {
  const { state: appState, dispatch: appDispatch } = useAppState();
  const { moduleSearch, moduleFilter, moduleViewMode } = appState;
  const [portalRoot, setPortalRoot] = useState(null);
  useEffect(() => { setPortalRoot(document.body); }, []);

  const filteredModules = useMemo(() => {
    return (Array.isArray(modules) ? modules : []).filter(module => {
      const matchesSearch =
        module.name?.toLowerCase().includes(moduleSearch.toLowerCase()) ||
        module.category?.toLowerCase().includes(moduleSearch.toLowerCase());
      const matchesFilter =
        moduleFilter === 'all' ||
        (moduleFilter === 'enabled' && module.enabled) ||
        (moduleFilter === 'disabled' && !module.enabled) ||
        (moduleFilter === 'with-data' && module.data_richness > 0) ||
        (moduleFilter === 'with-errors' && module.has_errors) ||
        (moduleFilter === module.category);
      return matchesSearch && matchesFilter;
    });
  }, [modules, moduleSearch, moduleFilter]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
        <div>
          <h2 className="text-2xl font-bold text-white">AI Module Observatory</h2>
          <p className="text-gray-400 mt-1">Comprehensive overview of your {moduleStats.total} intelligent modules</p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            onClick={() => onRefresh('enable-all')}
            className="px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center space-x-2 text-sm"
          >
            <CheckCircle size={14} />
            <span>Enable All</span>
          </button>
          <button
            onClick={() => onRefresh('disable-all')}
            className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors flex items-center space-x-2 text-sm"
          >
            <XCircle size={14} />
            <span>Disable All</span>
          </button>
          <button
            onClick={() => onRefresh('refresh')}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors flex items-center space-x-2 text-sm"
          >
            <RefreshCw size={14} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gradient-to-r from-blue-600 to-blue-700 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm">Total Modules</p>
              <p className="text-2xl font-bold">{moduleStats.total}</p>
            </div>
            <Layers size={24} className="text-blue-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-600 to-green-700 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100 text-sm">Enabled</p>
              <p className="text-2xl font-bold">{moduleStats.enabled}</p>
            </div>
            <CheckCircle size={24} className="text-green-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-orange-600 to-orange-700 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-orange-100 text-sm">With Live Data</p>
              <p className="text-2xl font-bold">{moduleStats.withData}</p>
            </div>
            <Activity size={24} className="text-orange-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-red-600 to-red-700 rounded-xl p-4 text-white">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-red-100 text-sm">With Errors</p>
              <p className="text-2xl font-bold">{moduleStats.withErrors}</p>
            </div>
            <AlertTriangle size={24} className="text-red-200" />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" size={16} />
            <input
              type="text"
              placeholder="Search modules..."
              value={appState.moduleSearch}
              onChange={(e) => appDispatch({ type: 'SET_MODULE_SEARCH', payload: e.target.value })}
              className="w-full pl-9 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-blue-500 text-sm"
            />
          </div>

          <div className="flex items-center space-x-3">
            <select
              value={appState.moduleFilter}
              onChange={(e) => appDispatch({ type: 'SET_MODULE_FILTER', payload: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white focus:outline-none focus:border-blue-500 text-sm"
            >
              <option value="all">All Modules</option>
              <option value="enabled">Enabled Only</option>
              <option value="disabled">Disabled Only</option>
              <option value="with-data">With Live Data</option>
              <option value="with-errors">With Errors</option>
              {Object.keys(moduleCategories || {}).map(category => (
                <option key={category} value={category}>
                  {category.replace('_', ' ').toUpperCase()}
                </option>
              ))}
            </select>

            <div className="flex items-center bg-gray-700 rounded-lg p-1">
              <button
                onClick={() => appDispatch({ type: 'SET_MODULE_VIEW_MODE', payload: 'grid' })}
                className={`p-1.5 rounded ${appState.moduleViewMode === 'grid' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                <Grid size={14} />
              </button>
              <button
                onClick={() => appDispatch({ type: 'SET_MODULE_VIEW_MODE', payload: 'list' })}
                className={`p-1.5 rounded ${appState.moduleViewMode === 'list' ? 'bg-blue-600 text-white' : 'text-gray-400 hover:text-white'}`}
              >
                <List size={14} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* List */}
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">
            Modules ({filteredModules.length})
          </h3>
          <div className="text-sm text-gray-400">
            Showing {filteredModules.length} of {(Array.isArray(modules) ? modules.length : 0)} modules
          </div>
        </div>

        {moduleViewMode === 'grid' ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {filteredModules.map(module => (
              <div key={module.id ?? module.name} onClick={() => onSelectModule(module.id ?? module.name)}>
                <EnhancedModuleCard
                  module={module}
                  onToggle={onToggle}
                  onClick={() => onSelectModule(module.id ?? module.name)}
                />
              </div>
            ))}
          </div>
        ) : (
          <div className="space-y-2">
            {filteredModules.map(module => {
              const IconComponent = getCategoryIcon(module.category);
              const statusColor = getStatusColor(module.status);
              const categoryGradient = getCategoryColor(module.category);
              const healthColor = getHealthColor(module.health_score || 0);
              const dataUtilization = Math.min(100, ((module.data_richness || 0) / Math.max(1, module.provides_count || 1)) * 100);
              
              // Get key metric preview
              const keyMetric = module.insights?.key_metrics ? 
                Object.entries(module.insights.key_metrics)[0] : null;
              
              return (
                <div
                  key={module.id ?? module.name}
                  className={`bg-gradient-to-r from-gray-800 to-gray-750 rounded-xl p-4 transition-all duration-300 cursor-pointer border group ${
                    module.has_errors ? 'border-red-500/30 hover:border-red-400/50' :
                    module.enabled ? 'border-gray-700 hover:border-blue-500/50 hover:shadow-lg hover:shadow-blue-500/5' :
                    'border-gray-700/50 hover:border-gray-600'
                  }`}
                  onClick={() => onSelectModule(module.id ?? module.name)}
                >
                  <div className="flex items-center justify-between">
                    {/* Left section */}
                    <div className="flex items-center space-x-4 flex-1 min-w-0">
                      <div className={`p-2.5 rounded-xl bg-gradient-to-br ${categoryGradient} text-white shadow-md flex-shrink-0`}>
                        <IconComponent size={18} />
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-2">
                          <h3 className="text-white font-semibold text-sm truncate group-hover:text-blue-400 transition-colors">
                            {module.name.replace(/([A-Z])/g, ' $1').trim()}
                          </h3>
                          {module.enabled && module.data_richness > 0 && (
                            <span className="flex h-2 w-2 flex-shrink-0">
                              <span className="animate-ping absolute inline-flex h-2 w-2 rounded-full bg-green-400 opacity-75"></span>
                              <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
                            </span>
                          )}
                        </div>
                        <div className="flex items-center space-x-3 mt-1">
                          <span className="text-gray-500 text-xs capitalize">{module.category?.replace('_', ' ')}</span>
                          <span className="text-gray-600">â€¢</span>
                          <span className={`text-xs font-medium ${statusColor}`}>{module.status}</span>
                          {module.insights?.summary && module.insights.summary !== "No data available" && (
                            <>
                              <span className="text-gray-600">â€¢</span>
                              <span className="text-gray-400 text-xs truncate max-w-[200px]">
                                {module.insights.summary}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Stats section */}
                    <div className="hidden lg:flex items-center space-x-6 mx-4">
                      {/* Health */}
                      <div className="text-center">
                        <div className={`text-sm font-bold ${healthColor}`}>{module.health_score || 0}%</div>
                        <div className="text-gray-500 text-xs">Health</div>
                      </div>
                      {/* Data Flow */}
                      <div className="text-center">
                        <div className="text-sm font-bold text-blue-400">
                          {module.data_richness || 0}/{module.provides_count || 0}
                        </div>
                        <div className="text-gray-500 text-xs">Data</div>
                      </div>
                      {/* Key Metric */}
                      {keyMetric && (
                        <div className="text-center max-w-[100px]">
                          <div className="text-sm font-bold text-cyan-400 truncate">
                            {typeof keyMetric[1] === 'number' ? keyMetric[1].toFixed(2) : String(keyMetric[1]).slice(0, 8)}
                          </div>
                          <div className="text-gray-500 text-xs truncate">{keyMetric[0].replace(/_/g, ' ')}</div>
                        </div>
                      )}
                      {/* Utilization bar */}
                      <div className="w-20">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-xs text-gray-500">Flow</span>
                          <span className="text-xs text-gray-400">{dataUtilization.toFixed(0)}%</span>
                        </div>
                        <div className="w-full bg-gray-700 rounded-full h-1.5">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-green-500 h-1.5 rounded-full transition-all duration-500"
                            style={{ width: `${dataUtilization}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Right section */}
                    <div className="flex items-center space-x-3 flex-shrink-0">
                      {module.has_errors && (
                        <div className="flex items-center space-x-1 px-2 py-1 bg-red-500/20 rounded-lg">
                          <AlertTriangle size={12} className="text-red-400" />
                          <span className="text-red-400 text-xs font-medium">{module.error_count}</span>
                        </div>
                      )}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onToggle(module.name);
                        }}
                        className={`p-2 rounded-lg transition-all transform hover:scale-105 ${
                          module.enabled 
                            ? 'bg-green-500/20 text-green-400 hover:bg-green-500/30' 
                            : 'bg-gray-700 text-gray-500 hover:bg-gray-600'
                        }`}
                      >
                        <Power size={14} />
                      </button>
                      <ChevronRight size={16} className="text-gray-500 group-hover:text-blue-400 transition-colors" />
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Details Modal via portal (prevents reflow/flicker) */}
      {portalRoot && selectedModule && createPortal(
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 rounded-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between p-6 border-b border-gray-700">
              <div className="flex items-center space-x-4">
                <div className={`p-3 rounded-xl bg-gradient-to-r ${getCategoryColor(selectedModule.category)} text-white`}>
                  {React.createElement(getCategoryIcon(selectedModule.category), { size: 24 })}
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white">{selectedModule.name}</h2>
                  <p className="text-gray-400 capitalize">{selectedModule.category.replace('_', ' ')} Module</p>
                </div>
              </div>
              <button onClick={() => onSelectModule(null)} className="p-2 hover:bg-gray-700 rounded-lg transition-colors">
                <X size={20} className="text-gray-400" />
              </button>
            </div>

            <div className="p-6 space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-gray-900 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-2">Status</h3>
                  <p className={`text-lg font-medium ${getStatusColor(selectedModule.status)} capitalize`}>
                    {selectedModule.status}
                  </p>
                </div>
                <div className="bg-gray-900 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-2">State</h3>
                  <p className={`text-lg font-medium ${selectedModule.enabled ? 'text-green-400' : 'text-red-400'}`}>
                    {selectedModule.enabled ? 'Enabled' : 'Disabled'}
                  </p>
                </div>
                <div className="bg-gray-900 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-2">Last Update</h3>
                  <p className="text-gray-300 text-sm">{selectedModule.last_update ? new Date(selectedModule.last_update).toLocaleString() : 'Ã¢â‚¬â€'}</p>
                </div>
              </div>

              {selectedModule.health_score !== undefined && (
                <div className="bg-gray-900 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3 flex items-center">
                    <Heart size={16} className="mr-2 text-green-400" />
                    Module Health
                  </h3>
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <span className="text-gray-300">Health Score:</span>
                      <span className={`text-lg font-bold ${getHealthColor(selectedModule.health_score)}`}>
                        {selectedModule.health_score}% ({selectedModule.health_status})
                      </span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-3">
                      <div className={`h-3 rounded-full transition-all duration-500 ${getHealthBarColor(selectedModule.health_score)}`} style={{ width: `${selectedModule.health_score}%` }} />
                    </div>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <div className="text-center">
                        <div className="text-xl font-bold text-blue-400">{selectedModule.data_richness}</div>
                        <div className="text-xs text-gray-400">Active Data Streams</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xl font-bold text-purple-400">{selectedModule.provides_count}</div>
                        <div className="text-xs text-gray-400">Total Capabilities</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {selectedModule.insights && (
                <div className="bg-gray-900 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3 flex items-center">
                    <Brain size={16} className="mr-2 text-purple-400" />
                    Module Insights
                  </h3>
                  <div className="space-y-4">
                    <div className="bg-gray-800 rounded-lg p-3">
                      <h4 className="text-white font-medium mb-2">Current Activity</h4>
                      <p className="text-gray-300 text-sm">{selectedModule.insights.summary}</p>
                    </div>

                    {selectedModule.insights.key_metrics && Object.keys(selectedModule.insights.key_metrics).length > 0 && (
                      <div className="bg-gray-800 rounded-lg p-3">
                        <h4 className="text-white font-medium mb-3">Key Metrics</h4>
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                          {Object.entries(selectedModule.insights.key_metrics).map(([key, value]) => (
                            <div key={key} className="flex items-center justify-between bg-gray-700 rounded px-3 py-2">
                              <span className="text-gray-300 text-sm">{key.replace(/_/g, ' ')}</span>
                              <span className="text-cyan-400 font-medium text-sm">
                                {typeof value === 'number' ? value.toFixed(3) :
                                  typeof value === 'object' ? JSON.stringify(value).slice(0, 15) + '...' :
                                  String(value).slice(0, 20)}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <div className="grid grid-cols-3 gap-3">
                      <div className="bg-gray-800 rounded-lg p-3 text-center">
                        <div className="text-lg font-bold text-green-400">
                          {((selectedModule.data_richness / Math.max(1, selectedModule.provides_count)) * 100).toFixed(0)}%
                        </div>
                        <div className="text-xs text-gray-400">Data Utilization</div>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3 text-center">
                        <div className="text-lg font-bold text-blue-400">
                          {selectedModule.error_count === 0 ? '100%' : Math.max(0, 100 - selectedModule.error_count * 10).toFixed(0) + '%'}
                        </div>
                        <div className="text-xs text-gray-400">Reliability</div>
                      </div>
                      <div className="bg-gray-800 rounded-lg p-3 text-center">
                        <div className="text-lg font-bold text-purple-400">
                          {selectedModule.enabled ? 'ONLINE' : 'OFFLINE'}
                        </div>
                        <div className="text-xs text-gray-400">Availability</div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="bg-gray-900 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3 flex items-center">
                    <ArrowUp size={16} className="mr-2 text-green-400" />
                    Provides ({selectedModule.provides_count})
                  </h3>
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {(selectedModule.provides || []).map(key => (
                      <div key={key} className="flex items-center justify-between text-sm">
                        <span className="text-gray-300">{key}</span>
                        <span className={`text-xs px-2 py-1 rounded ${
                          selectedModule.live_data?.[key] !== undefined ? 'bg-green-600 text-white' : 'bg-gray-600 text-gray-300'
                        }`}>
                          {selectedModule.live_data?.[key] !== undefined ? 'LIVE' : 'NULL'}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="bg-gray-900 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3 flex items-center">
                    <ArrowDown size={16} className="mr-2 text-blue-400" />
                    Requires ({selectedModule.requires_count})
                  </h3>
                  <div className="max-h-40 overflow-y-auto space-y-1">
                    {(selectedModule.requires || []).map(key => (
                      <span key={key} className="inline-block text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded mr-1 mb-1">
                        {key}
                      </span>
                    ))}
                  </div>
                </div>
              </div>

              {selectedModule.live_data && Object.keys(selectedModule.live_data).length > 0 && (
                <div className="bg-gray-900 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3 flex items-center">
                    <Activity size={16} className="mr-2 text-green-400" />
                    Live Data ({Object.keys(selectedModule.live_data).length})
                  </h3>
                  <div className="max-h-60 overflow-y-auto">
                    <pre className="text-sm text-gray-300 whitespace-pre-wrap">
                      {JSON.stringify(selectedModule.live_data, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {selectedModule.errors && selectedModule.errors.length > 0 && (
                <div className="bg-red-900/20 border border-red-700 rounded-lg p-4">
                  <h3 className="text-white font-semibold mb-3 flex items-center">
                    <AlertTriangle size={16} className="mr-2 text-red-400" />
                    Recent Errors ({selectedModule.error_count})
                  </h3>
                  <div className="space-y-2">
                    {selectedModule.errors.map((err, idx) => (
                      <div key={idx} className="text-red-300 text-sm bg-red-900/30 p-2 rounded">
                        {err}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>,
        document.body
      )}
    </div>
  );
});

// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

// LiveUptime Component - Self-contained, only re-renders itself
const LiveUptimeDisplay = React.memo(function LiveUptimeDisplay({ fallbackUptime }) {
  const [uptime, setUptime] = useState(fallbackUptime || '');
  const sessionStartRef = useRef(null);
  
  useEffect(() => {
    fetch('/api/status')
      .then(res => res.json())
      .then(data => {
        const startTime = data?.performance?.session_start_time;
        if (startTime) {
          sessionStartRef.current = new Date(startTime);
        }
      })
      .catch(() => {});
  }, []);
  
  useEffect(() => {
    const updateUptime = () => {
      if (!sessionStartRef.current) return;
      const now = new Date();
      const diff = now - sessionStartRef.current;
      const days = Math.floor(diff / (1000 * 60 * 60 * 24));
      const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
      const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
      const seconds = Math.floor((diff % (1000 * 60)) / 1000);
      let newUptime;
      if (days > 0) newUptime = days + 'd ' + hours + 'h ' + minutes + 'm';
      else if (hours > 0) newUptime = hours + 'h ' + minutes + 'm ' + seconds + 's';
      else newUptime = minutes + 'm ' + seconds + 's';
      setUptime(newUptime);
    };
    updateUptime();
    const interval = setInterval(updateUptime, 1000);
    return () => clearInterval(interval);
  }, []);
  
  useEffect(() => {
    if (fallbackUptime && !sessionStartRef.current) setUptime(fallbackUptime);
  }, [fallbackUptime]);
  
  if (!uptime) return null;
  return (<span className="flex items-center"><Clock className="w-4 h-4 mr-1" />{uptime}</span>);
});


/** Main Dashboard (stateful container) */
// Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

const EnhancedTradingDashboard = () => {
  const { state: appState, dispatch: appDispatch } = useAppState();
  const [isPending, startUITransition] = useTransition();

  // Core state
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginForm, setLoginForm] = useState({ login: '', password: '', server: 'MetaQuotes-Demo' });

  // System state
  const [systemStatus, setSystemStatus] = useState('IDLE');
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  // Selection (stable by ID)
  const [selectedModuleId, setSelectedModuleId] = useState(null);

  // Comprehensive system data
  const [systemState, setSystemState] = useState(null);
  const [performance, setPerformance] = useState({});
  const [moduleStates, setModuleStates] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [accountInfo, setAccountInfo] = useState(null); // MT5 account data from login
  // Track alerts the user dismissed locally so they don't reappear
  const dismissedStorageKey = 'dismissed_alert_ids_v1';
  const [dismissedAlerts, setDismissedAlerts] = useState(() => {
    try {
      const raw = localStorage.getItem(dismissedStorageKey);
      if (!raw) return new Set();
      const arr = JSON.parse(raw);
      return new Set(Array.isArray(arr) ? arr : []);
    } catch {
      return new Set();
    }
  });
  const [systemMetrics, setSystemMetrics] = useState({});

  // Enhanced module data
  const [modulesData, setModulesData] = useState([]);
  const [modulesById, setModulesById] = useState({});
  const [moduleCategories, setModuleCategories] = useState({});
  const [moduleStats, setModuleStats] = useState({});

  // UI state
  const [logs, setLogs] = useState({});
  const [checkpoints, setCheckpoints] = useState([]);
  const [tensorboardUrl, setTensorboardUrl] = useState(null);
  const selectedLogCategory = appState.selectedLogCategory;
  const showAlerts = appState.showAlerts;
  const alertFilter = appState.alertFilter;
  const moduleSearch = appState.moduleSearch;
  const moduleFilter = appState.moduleFilter;
  const moduleViewMode = appState.moduleViewMode;
  const [readAlerts, setReadAlerts] = useState(new Set());

  // Persist dismissed alerts when they change
  useEffect(() => {
    try {
      localStorage.setItem(dismissedStorageKey, JSON.stringify([...dismissedAlerts]));
    } catch {}
  }, [dismissedAlerts]);

  // Helper to create a stable alert id (timestamp+module fallback)
  const alertId = useCallback((a) => (
    String(a?.timestamp ?? a?.time ?? '') + '|' + String(a?.module ?? '') + '|' + String(a?.alert?.message ?? a?.alert ?? '')
  ), []);

  // When alerts list changes, drop any that were dismissed
  useEffect(() => {
    if (!Array.isArray(alerts) || dismissedAlerts.size === 0) return;
    const filtered = alerts.filter(a => !dismissedAlerts.has(alertId(a)));
    if (filtered.length !== alerts.length) setAlerts(filtered);
  }, [alerts, dismissedAlerts, alertId]);

  // Trading config
  const [tradingConfig, setTradingConfig] = useState({
    instruments: ["EURUSD", "XAUUSD"],
    timeframes: ["M15", "H1", "H4", "D1"],
    update_interval: 5,
    max_position_size: 0.1,
    max_total_exposure: 0.3,
    min_trade_interval: 60,
    use_trailing_stop: true,
    emergency_drawdown_limit: 0.25,
    debug: false
  });

  // WS refs
  const ws = useRef(null);
  const reconnectTimer = useRef(null);
  const pingInterval = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  const lastMessageTime = useRef({});
  const updateQueue = useRef([]);
  const isProcessingQueue = useRef(false);

  // API cache
  const apiCache = useRef({});
  const apiCacheTTL = 5000;

  const apiCall = useCallback(async (endpoint, options = {}) => {
    const cacheKey = `${endpoint}_${JSON.stringify(options)}`;
    const cached = apiCache.current[cacheKey];
    if (cached && Date.now() - cached.timestamp < apiCacheTTL && !options.noCache) return cached.data;

    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      const data = await response.json();
      apiCache.current[cacheKey] = { data, timestamp: Date.now() };
      return data;
    } catch (err) {
      console.error(`API call failed for ${endpoint}:`, err);
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError('Connection failed. Please check your network.');
      } else {
        setError(`Request failed: ${err.message}`);
      }
      throw err;
    }
  }, []);

  // Load live configuration
  const loadSystemConfiguration = async () => {
    try {
      const config = await apiCall('/config/system');
      if (config.trading) setTradingConfig(prev => ({ ...prev, ...config.trading }));
      if (config.mt5?.server) setLoginForm(prev => ({ ...prev, server: config.mt5.server }));
    } catch (err) {
      console.error('Failed to load system configuration:', err);
    }
  };

  // Auth
  const handleLogin = async () => {
    try {
      const data = await apiCall('/login', {
        method: 'POST',
        body: JSON.stringify({
          login: parseInt(loginForm.login),
          password: loginForm.password,
          server: loginForm.server
        }),
        noCache: true
      });
      if (data.success) {
        setIsLoggedIn(true);
        setError('');
        // Store MT5 account info for display
        if (data.account) {
          setAccountInfo(data.account);
        }
      }
      else setError(data.error || 'Login failed');
    } catch (err) {
      setError(`Login failed: ${err.message}`);
    }
  };

  const handleLogout = async () => {
    try {
      await apiCall('/logout', { method: 'POST', noCache: true });
      setIsLoggedIn(false);
      setSystemState(null);
      setModuleStates({});
      setPerformance({});
      setAlerts([]);
      setAccountInfo(null); // Clear MT5 account info
    } catch (err) {
      console.error('Logout error:', err);
    }
  };

  // Trading actions
  const startTrading = async () => {
    try {
      await apiCall('/trading/start', { method: 'POST', body: JSON.stringify(tradingConfig), noCache: true });
    } catch (err) {
      setError(`Failed to start trading: ${err.message}`);
    }
  };
  const stopTrading = async () => {
    try {
      await apiCall('/trading/stop', { method: 'POST', noCache: true });
    } catch (err) {
      setError(`Failed to stop trading: ${err.message}`);
    }
  };

  const emergencyStop = async () => {
    if (!confirm(' EMERGENCY STOP: This will close all positions immediately. Are you sure?')) return;
    try {
      await apiCall('/trading/emergency-stop', { method: 'POST', noCache: true });
    } catch (err) {
      setError(`Emergency stop failed: ${err.message}`);
    }
  };

  const toggleModule = async (moduleName) => {
    try {
      await apiCall(`/modules/${moduleName}/toggle`, { method: 'POST', noCache: true });
    } catch (err) {
      setError(`Failed to toggle module ${moduleName}: ${err.message}`);
    }
  };

  const saveCheckpoint = async () => {
    try {
      await apiCall('/checkpoints/save', { method: 'POST', noCache: true });
      fetchCheckpoints();
    } catch (err) {
      setError(`Failed to save checkpoint: ${err.message}`);
    }
  };

  const fetchLogs = useCallback(async (category) => {
    try {
      const data = await apiCall(`/logs/${category}`);
      setLogs(prev => ({ ...prev, [category]: data }));
      appDispatch({ type: 'MARK_DATA_LOADED', payload: 'logs' });
    } catch (err) {
      console.error(`Failed to fetch ${category} logs:`, err);
    }
  }, [apiCall, appDispatch]);

  const fetchCheckpoints = useCallback(async () => {
    try {
      const data = await apiCall('/checkpoints');
      setCheckpoints(data.checkpoints || []);
    } catch (err) {
      console.error('Failed to fetch checkpoints:', err);
    }
  }, [apiCall]);

  // CSV files functionality removed - live trading only

  const fetchEnhancedModules = useCallback(async () => {
    try {
      const data = await apiCall('/modules');
      const modulesArr = toArray(data.modules);
      setModulesData(modulesArr);
      setModulesById(prev => {
        const next = { ...prev };
        for (const m of modulesArr) {
          const id = m.id ?? m.name;
          const prevM = prev[id];
          // replace only if changed
          const changed = !prevM || JSON.stringify(prevM) !== JSON.stringify({ ...prevM, ...m });
          next[id] = changed ? { ...prevM, ...m, id } : prevM;
        }
        return next;
      });
      setModuleCategories(data.categories || {});
      setModuleStats({
        total: data.total_modules || 0,
        enabled: data.enabled_modules || 0,
        withData: data.modules_with_data || 0,
        withErrors: data.modules_with_errors || 0
      });
      appDispatch({ type: 'MARK_DATA_LOADED', payload: 'modules' });
    } catch (err) {
      console.error('Failed to fetch enhanced modules:', err);
    }
  }, [apiCall, appDispatch]);

  // Tensorboard
  const startTensorBoard = async () => {
    try {
      const data = await apiCall('/tensorboard/start', { method: 'POST', noCache: true });
      if (data.success) setTensorboardUrl(data.url);
    } catch (err) {
      console.error('Failed to start TensorBoard:', err);
    }
  };

  // Uploads
  const uploadModel = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    try {
      const response = await fetch(`${API_BASE}/model/upload`, { method: 'POST', body: formData });
      if (response.ok) { setError(''); alert('Model uploaded successfully'); }
      else {
        const data = await response.json();
        setError(data.detail || 'Upload failed');
      }
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    }
  };

  // CSV upload functionality removed - live trading only

  // Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
  // WebSocket connection with throttled/batched updates (no flicker)
  // Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

  const processUpdateQueue = useCallback(async () => {
    if (isProcessingQueue.current || updateQueue.current.length === 0) return;
    isProcessingQueue.current = true;

    while (updateQueue.current.length > 0) {
      const update = updateQueue.current.shift();

      // Use startTransition to keep UI interactions responsive
      startTransition(() => {
        switch (update.type) {
          case 'system_state': {
            const stateData = update.data || {};
            setSystemState(prev => ({ ...prev, ...stateData }));
            setSystemStatus(prev => stateData.status ?? prev);
            setPerformance(prev => ({ ...prev, ...(stateData.performance || {}) }));
            setModuleStates(prev => ({ ...prev, ...(stateData.modules || {}) }));
            setAlerts(prev => {
              if (!Array.isArray(stateData.alerts)) return prev;
              const filtered = stateData.alerts.filter(a => !dismissedAlerts.has(alertId(a)));
              return filtered;
            });
            setSystemMetrics(prev => ({ ...prev, ...(stateData.system_metrics || {}) }));
            break;
          }
          case 'modules_update': {
            const arr = toArray(update.data?.modules);
            setModulesData(arr);
            setModulesById(prev => {
              const next = { ...prev };
              for (const m of arr) {
                const id = m.id ?? m.name;
                const prevM = prev[id];
                const changed = !prevM || JSON.stringify(prevM) !== JSON.stringify({ ...prevM, ...m });
                next[id] = changed ? { ...prevM, ...m, id } : prevM;
              }
              return next;
            });
            setModuleCategories(update.data?.categories || {});
            setModuleStats(update.data?.stats || {});
            appDispatch({ type: 'MARK_DATA_LOADED', payload: 'modules' });
            break;
          }
          case 'logs_update': {
            setLogs(prev => ({ ...prev, [update.category]: update.data }));
            appDispatch({ type: 'MARK_DATA_LOADED', payload: 'logs' });
            break;
          }
          case 'alerts_update': {
            const list = update.data || [];
            setAlerts(list.filter(a => !dismissedAlerts.has(alertId(a))));
            setReadAlerts(prev => {
              const alertIds = new Set(list.map(a => (a.timestamp ?? a.time) + (a.module ?? '')));
              return new Set([...prev].filter(id => alertIds.has(id)));
            });
            break;
          }
          case 'mt5_data_update': {
            appDispatch({
              type: 'SET_MT5_DATA',
              payload: {
                chartData: update.data?.chartData,
                recentTrades: update.data?.recentTrades,
                symbols: update.data?.symbols,
                positions: update.data?.positions,
                account: update.data?.account,
                positionCount: update.data?.positionCount,
              }
            });
            // Update performance if account data is available
            if (update.data?.account) {
              setPerformance(prev => ({
                ...prev,
                current_balance: update.data.account.balance,
                total_pnl: update.data.account.profit,
              }));
            }
            break;
          }
        }
      });

      await new Promise(r => setTimeout(r, 8)); // tiny breath between batches
    }

    isProcessingQueue.current = false;
  }, [appDispatch]);

  const connectWebSocket = useCallback(() => {
    // Avoid forcing a reconnect if already connected or in progress
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const backendHttp = import.meta.env.VITE_BACKEND_URL;
    let wsUrl;
    // Only use VITE_BACKEND_URL in development mode (when it's explicitly set)
    if (backendHttp && typeof backendHttp === 'string' && import.meta.env.DEV) {
      const http = backendHttp.endsWith('/') ? backendHttp.slice(0, -1) : backendHttp;
      const scheme = http.startsWith('https:') ? 'wss:' : 'ws:';
      wsUrl = `${scheme}//${http.replace(/^https?:\/\//, '')}/ws`;
    } else {
      // In production or when VITE_BACKEND_URL is not set, use the current host
      wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    }

    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      setWsConnected(true);
      reconnectAttempts.current = 0;
      // Cancel any pending reconnect timer now that we're connected
      if (reconnectTimer.current) {
        clearTimeout(reconnectTimer.current);
        reconnectTimer.current = null;
      }
      // Send initial ping
      if (ws.current?.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ type: 'ping' }));
      }
      // Start periodic ping to keep connection alive (every 15 seconds)
      if (pingInterval.current) clearInterval(pingInterval.current);
      pingInterval.current = setInterval(() => {
        if (ws.current?.readyState === WebSocket.OPEN) {
          ws.current.send(JSON.stringify({ type: 'ping' }));
        }
      }, 15000);
    };


    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const now = Date.now();

        const minIntervals = {
          'system_state': 2000,
          'modules_update': 5000,
          'mt5_data_update': 10000,
          'logs_update': 5000,
          'alerts_update': 3000
        };

        const minInterval = minIntervals[data.type] || 1000;
        const lastTime = lastMessageTime.current[data.type] || 0;

        if (now - lastTime < minInterval && data.type !== 'pong') return;

        lastMessageTime.current[data.type] = now;

        if (data.type !== 'pong') {
          updateQueue.current.push({
            type: data.type,
            data: data.data,
            category: data.category,
            timestamp: now
          });
          processUpdateQueue();
        }
      } catch (err) {
        console.error('Error parsing WS message:', err);
      }
    };

    ws.current.onclose = (evt) => {
      setWsConnected(false);
      // Clear ping interval
      if (pingInterval.current) {
        clearInterval(pingInterval.current);
        pingInterval.current = null;
      }
      try {
        console.warn('WebSocket closed', { code: evt?.code, reason: evt?.reason });
      } catch {}
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, Math.min(reconnectAttempts.current, 5)), 30000);
        reconnectAttempts.current++;
        if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
        reconnectTimer.current = setTimeout(connectWebSocket, delay);
      }
    };

    ws.current.onerror = (error) => console.error('WebSocket error:', error);
  }, [processUpdateQueue]);

  useEffect(() => {
    connectWebSocket();
    loadSystemConfiguration();
    return () => {
      if (ws.current) ws.current.close();
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      if (pingInterval.current) clearInterval(pingInterval.current);
    };
  }, []); // mount once

  // Initial loads after login (throttled)
  useEffect(() => {
    if (!isLoggedIn) return;
    const loadInitial = async () => {
      if (Date.now() - appState.lastUpdate.system < 10000) return;
      await Promise.all([
        fetchCheckpoints(),
        fetchLogs(selectedLogCategory),
        fetchEnhancedModules()
      ]);
    };
    loadInitial();
  }, [isLoggedIn, selectedLogCategory, fetchCheckpoints, fetchLogs, fetchEnhancedModules, appState.lastUpdate.system]);

  useEffect(() => {
    if (isLoggedIn && activeTab === 'modules' && !appState.dataLoaded.modules) {
      fetchEnhancedModules();
    }
  }, [isLoggedIn, activeTab, fetchEnhancedModules, appState.dataLoaded.modules]);

  useEffect(() => {
    if (isLoggedIn && activeTab === 'logs') fetchLogs(selectedLogCategory);
  }, [isLoggedIn, activeTab, selectedLogCategory, fetchLogs]);

  // derived selected module (stable across refreshes)
  const selectedModule = useMemo(() => {
    if (!selectedModuleId) return null;
    return modulesById[selectedModuleId] ?? null;
  }, [selectedModuleId, modulesById]);

  // Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
  // Login screen
  // Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center p-4">
        <div className="bg-gray-800/90 backdrop-blur-sm p-8 rounded-2xl shadow-2xl w-full max-w-md border border-gray-700">
          <div className="flex items-center justify-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center">
              <Brain className="w-10 h-10 text-white" />
            </div>
          </div>

          <div className="text-center mb-8">
            <h1 className="text-3xl font-bold text-white mb-2">AI Trading System</h1>
            <p className="text-gray-400">PPO-Lagrangian Trading Dashboard</p>
            <div className="flex items-center justify-center mt-4 space-x-2">
              <div className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-400' : 'bg-red-400'}`} />
              <span className="text-sm text-gray-500">{wsConnected ? 'Connected' : 'Disconnected'}</span>
            </div>
          </div>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">MT5 Login</label>
              <input
                type="text"
                value={loginForm.login}
                onChange={e => setLoginForm({ ...loginForm, login: e.target.value })}
                className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
                placeholder="12345678"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Password</label>
              <input
                type="password"
                value={loginForm.password}
                onChange={e => setLoginForm({ ...loginForm, password: e.target.value })}
                className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
                placeholder="Enter password"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Server</label>
              <input
                type="text"
                value={loginForm.server}
                onChange={e => setLoginForm({ ...loginForm, server: e.target.value })}
                className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
                placeholder="MetaQuotes-Demo"
              />
            </div>

            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-lg">
                <div className="flex items-center">
                  <AlertCircle className="w-4 h-4 mr-2" />
                  {error}
                </div>
              </div>
            )}

            <button
              onClick={handleLogin}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium py-3 rounded-lg hover:from-blue-600 hover:to-purple-700 transition-all duration-200 transform hover:scale-105"
            >
              Connect to MT5
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬
  // Main render
  // Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800/90 backdrop-blur-sm border-b border-gray-700 px-6 py-4 sticky top-0 z-40">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">AI Trading Dashboard</h1>
              <div className="flex items-center space-x-4 text-sm text-gray-400">
                <span className="flex items-center">
                  {wsConnected ? <Wifi className="w-4 h-4 mr-1 text-green-400" /> : <WifiOff className="w-4 h-4 mr-1 text-red-400" />}
                  {wsConnected ? 'Connected' : 'Disconnected'}
                </span>
                <StatusIndicator status={systemStatus} />
                <LiveUptimeDisplay fallbackUptime={systemState?.uptime} />
              </div>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            {systemStatus === 'IDLE' && (
              <button
                onClick={startTrading}
                className="group relative overflow-hidden bg-gradient-to-r from-emerald-500 via-green-600 to-teal-600 hover:from-emerald-600 hover:via-green-700 hover:to-teal-700 px-6 py-3 rounded-xl transition-all duration-300 transform hover:scale-105 hover:shadow-2xl text-white font-semibold"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-emerald-400 to-teal-500 opacity-0 group-hover:opacity-20 transition-opacity duration-300"></div>
                <div className="relative flex items-center space-x-2">
                  <Play className="w-5 h-5 group-hover:animate-bounce" />
                  <span className="tracking-wide">Start Trading</span>
                </div>
                <div className="absolute inset-0 rounded-xl ring-2 ring-emerald-400/30 group-hover:ring-emerald-300/50 transition-all duration-300"></div>
              </button>
            )}
            {systemStatus === 'TRADING' && (
              <>
                <button
                  onClick={stopTrading}
                  className="flex items-center space-x-2 bg-gradient-to-r from-yellow-600 to-yellow-700 hover:from-yellow-700 hover:to-yellow-800 px-4 py-2 rounded-lg transition-all duration-200 text-white font-medium"
                >
                  <Pause className="w-4 h-4" />
                  <span>Stop Trading</span>
                </button>
                <button
                  onClick={emergencyStop}
                  className="flex items-center space-x-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 px-4 py-2 rounded-lg transition-all duration-200 text-white font-medium"
                >
                  <AlertTriangle className="w-4 h-4" />
                  <span>Emergency Stop</span>
                </button>
              </>
            )}

            <button
              onClick={saveCheckpoint}
              className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors text-white font-medium"
            >
              <Save className="w-4 h-4" />
              <span>Save</span>
            </button>

            <button
              onClick={handleLogout}
              className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition-colors text-white font-medium"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </header>

      {/* Body */}
      <div className="flex">
        {/* Sidebar */}
        <nav className="w-64 bg-gray-800/50 backdrop-blur-sm border-r border-gray-700 min-h-screen sticky top-[73px]">
          <div className="p-4">
            <div className="space-y-2">
              <TabButton icon={BarChart3} label="Overview" active={activeTab === 'overview'} onClick={() => setActiveTab('overview')} />
              <TabButton
                icon={Cpu}
                label="Modules"
                active={activeTab === 'modules'}
                onClick={() => setActiveTab('modules')}
                badge={Object.values(moduleStates).filter(m => m?.errors?.length > 0).length || null}
              />
              <TabButton icon={TrendingUp} label="Trading" active={activeTab === 'trading'} onClick={() => setActiveTab('trading')} />
              <TabButton icon={BarChart2} label="Analytics" active={activeTab === 'analytics'} onClick={() => setActiveTab('analytics')} />
              <TabButton icon={Sparkles} label="Strategy" active={activeTab === 'strategy'} onClick={() => setActiveTab('strategy')} />
              <TabButton icon={HardDrive} label="Memory" active={activeTab === 'memory'} onClick={() => setActiveTab('memory')} />
              <TabButton icon={Shield} label="Risk" active={activeTab === 'risk'} onClick={() => setActiveTab('risk')} />
              <TabButton icon={Vote} label="Voting" active={activeTab === 'voting'} onClick={() => setActiveTab('voting')} />
              <TabButton icon={Database} label="Logs" active={activeTab === 'logs'} onClick={() => setActiveTab('logs')} />
            </div>

            <div className="mt-8 space-y-4">
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Current Balance</div>
                <div className="text-lg font-bold text-green-400">${performance.current_balance?.toLocaleString() || '0.00'}</div>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Daily P&L</div>
                <div className={`text-lg font-bold ${performance.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${performance.daily_pnl?.toLocaleString() || '0.00'}
                </div>
              </div>

              {systemStatus === 'TRADING' && (
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">Open Positions</div>
                  <div className="text-lg font-bold text-blue-400">{appState.positionCount || 0}</div>
                </div>
              )}
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1 p-6 overflow-y-auto">
          {error && (
            <div className="mb-6 bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-lg flex items-center justify-between animate-in slide-in-from-top duration-300">
              <div className="flex items-center">
                <AlertCircle className="w-5 h-5 mr-2" />
                <span>{error}</span>
              </div>
              <button onClick={() => setError('')} className="text-red-400 hover:text-red-300">
                <X className="w-5 h-5" />
              </button>
            </div>
          )}

          <div className="animate-in fade-in duration-500">
            {activeTab === 'overview' && (
              <OverviewTab
                performance={performance}
                systemStatus={systemStatus}
                systemState={systemState}
                moduleStates={moduleStates}
                alerts={alerts}
                stopTrading={stopTrading}
                emergencyStop={emergencyStop}
                readAlerts={readAlerts}
                setReadAlerts={setReadAlerts}
                accountInfo={accountInfo}
              />
            )}

            {activeTab === 'modules' && (
              <ModulesTab
                modules={modulesData}
                moduleCategories={moduleCategories}
                moduleStats={moduleStats}
                onToggle={toggleModule}
                onRefresh={(action) => {
                  if (action === 'enable-all') apiCall('/modules/enable-all', { method: 'POST', noCache: true }).then(fetchEnhancedModules);
                  else if (action === 'disable-all') apiCall('/modules/disable-all', { method: 'POST', noCache: true }).then(fetchEnhancedModules);
                  else fetchEnhancedModules();
                }}
                selectedModule={selectedModule}
                onSelectModule={setSelectedModuleId}
              />
            )}

            {activeTab === 'trading' && (
              <TradingTab
                systemStatus={systemStatus}
                startTrading={startTrading}
                stopTrading={stopTrading}
                emergencyStop={emergencyStop}
                positions={appState.mt5Positions}
                account={appState.mt5Account}
              />
            )}

            {activeTab === 'analytics' && <AnalyticsTab />}

            {activeTab === 'strategy' && <StrategyTab />}

            {activeTab === 'memory' && <MemoryTab />}

            {activeTab === 'risk' && <RiskTab />}
            {activeTab === 'voting' && <VotingTab />}

            {activeTab === 'logs' && <LogsTab logs={logs} fetchLogs={fetchLogs} />}
          </div>
        </main>
      </div>

      <AlertsModal
        showAlerts={showAlerts}
        alerts={alerts}
        alertFilter={alertFilter}
        onClose={() => appDispatch({ type: 'SET_SHOW_ALERTS', payload: false })}
        setFilter={(val) => appDispatch({ type: 'SET_ALERT_FILTER', payload: val })}
        onClearAll={() => {
          // Mark current alerts as dismissed so they don't return on next server push
          const ids = new Set(dismissedAlerts);
          for (const a of alerts) ids.add(alertId(a));
          setDismissedAlerts(ids);
          setAlerts([]);
        }}
        onDismissAlert={(a) => {
          const id = alertId(a);
          if (!dismissedAlerts.has(id)) {
            const next = new Set(dismissedAlerts);
            next.add(id);
            setDismissedAlerts(next);
          }
          setAlerts(prev => prev.filter(x => alertId(x) !== id));
        }}
      />
    </div>
  );
};


const App = () => (
  <ErrorBoundary>
      <EnhancedTradingDashboard />
  </ErrorBoundary>
);

export default App;
