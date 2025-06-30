import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Play, Pause, Square, AlertTriangle, TrendingUp, Activity, Database, Settings, LogOut, Brain,
  Shield, Target, BarChart3, Users, Zap, CheckCircle, XCircle, Clock, DollarSign,
  ArrowUp, ArrowDown, Wifi, WifiOff, Save, Upload, Download, RefreshCw, AlertCircle,
  Cpu, HardDrive, Network, Eye, BarChart2, PieChart, LineChart, Layers, Bell, X,
  TrendingDown, Percent, Timer, Gauge, Monitor, Server, CloudOff, Power, FileUp,
  Loader2, CheckSquare, Radio, CloudUpload, FolderOpen, FileText, ToggleLeft, ToggleRight
} from 'lucide-react';
import { 
  LineChart as RechartsLineChart, Line, AreaChart, Area, BarChart, Bar, 
  PieChart as RechartsPieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer, RadialBarChart, RadialBar,
  ComposedChart, Scatter
} from 'recharts';

// Use relative URLs when served from same origin
const API_BASE = '/api';

// Enhanced Trading Dashboard with Complete Features
const EnhancedTradingDashboard = () => {
  // ═══════════════════════════════════════════════════════════════════
  // Core State Management
  // ═══════════════════════════════════════════════════════════════════
  
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginForm, setLoginForm] = useState({ 
    login: '', 
    password: '', 
    server: 'MetaQuotes-Demo' 
  });
  
  // System state
  const [systemStatus, setSystemStatus] = useState('IDLE');
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedModule, setSelectedModule] = useState(null);
  
  // Comprehensive system data
  const [systemState, setSystemState] = useState(null);
  const [performance, setPerformance] = useState({});
  const [moduleStates, setModuleStates] = useState({});
  const [alerts, setAlerts] = useState([]);
  const [systemMetrics, setSystemMetrics] = useState({});
  
  // Training specific state
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [trainingMetrics, setTrainingMetrics] = useState(null);
  const [trainingHistory, setTrainingHistory] = useState([]);
  const [csvFiles, setCsvFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  // UI state
  const [logs, setLogs] = useState({});
  const [checkpoints, setCheckpoints] = useState([]);
  const [tensorboardUrl, setTensorboardUrl] = useState(null);
  const [selectedLogCategory, setSelectedLogCategory] = useState('system');
  const [showAlerts, setShowAlerts] = useState(false);
  const [alertFilter, setAlertFilter] = useState('all');
  
  // Enhanced training configuration with mode selection
  const [trainingConfig, setTrainingConfig] = useState({
    mode: 'offline', // 'offline' or 'online'
    timesteps: 100000,
    learning_rate: 3e-4,
    batch_size: 64,
    n_epochs: 10,
    gamma: 0.99,
    n_steps: 2048,
    clip_range: 0.2,
    ent_coef: 0.01,
    vf_coef: 0.5,
    max_grad_norm: 0.5,
    target_kl: 0.01,
    checkpoint_freq: 10000,
    eval_freq: 5000,
    num_envs: 1,
    data_dir: 'data/processed',
    initial_balance: 10000,
    pretrained_model: null,
    auto_pretrained: false,
    debug: false
  });
  
  const [tradingConfig, setTradingConfig] = useState({
    instruments: ["EURUSD", "XAUUSD"],
    timeframes: ["H1", "H4", "D1"],
    update_interval: 5,
    max_position_size: 0.1,
    max_total_exposure: 0.3,
    min_trade_interval: 60,
    use_trailing_stop: true,
    emergency_drawdown_limit: 0.25,
    debug: false
  });
  
  // WebSocket reference
  const ws = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  
  // ═══════════════════════════════════════════════════════════════════
  // WebSocket Connection Management
  // ═══════════════════════════════════════════════════════════════════
  
  const connectWebSocket = useCallback(() => {
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
    
    console.log('Connecting to WebSocket:', wsUrl);
    ws.current = new WebSocket(wsUrl);
    
    ws.current.onopen = () => {
      setWsConnected(true);
      reconnectAttempts.current = 0;
      console.log('WebSocket connected');
      
      // Send ping to verify connection
      ws.current.send(JSON.stringify({ type: 'ping' }));
    };
    
    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'system_state') {
          const stateData = data.data;
          setSystemState(stateData);
          setSystemStatus(stateData.status);
          setPerformance(stateData.performance || {});
          setModuleStates(stateData.modules || {});
          setAlerts(stateData.alerts || []);
          setSystemMetrics(stateData.system_metrics || {});
          setTrainingProgress(stateData.training_progress || null);
        } else if (data.type === 'training_metrics') {
          // Real-time training metrics update
          setTrainingMetrics(data.data);
          setTrainingHistory(prev => [...prev.slice(-99), data.data]);
        } else if (data.type === 'pong') {
          // Handle pong response
          console.log('WebSocket ping/pong successful');
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.current.onclose = () => {
      setWsConnected(false);
      console.log('WebSocket disconnected');
      
      if (reconnectAttempts.current < maxReconnectAttempts) {
        const delay = Math.min(1000 * Math.pow(2, reconnectAttempts.current), 30000);
        reconnectAttempts.current++;
        
        console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts.current})`);
        setTimeout(connectWebSocket, delay);
      } else {
        console.error('Max reconnection attempts reached');
        setError('Connection lost. Please refresh the page.');
      }
    };
    
    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }, []);
  
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, [connectWebSocket]);
  
  // ═══════════════════════════════════════════════════════════════════
  // API Functions
  // ═══════════════════════════════════════════════════════════════════
  
  const apiCall = async (endpoint, options = {}) => {
    try {
      const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
      });
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}`);
      }
      
      return await response.json();
    } catch (err) {
      console.error(`API call failed for ${endpoint}:`, err);
      throw err;
    }
  };
  
  const handleLogin = async () => {
    try {
      const data = await apiCall('/login', {
        method: 'POST',
        body: JSON.stringify({
          login: parseInt(loginForm.login),
          password: loginForm.password,
          server: loginForm.server
        })
      });
      
      if (data.success) {
        setIsLoggedIn(true);
        setError('');
      } else {
        setError(data.error || 'Login failed');
      }
    } catch (err) {
      setError(`Login failed: ${err.message}`);
    }
  };
  
  const handleLogout = async () => {
    try {
      await apiCall('/logout', { method: 'POST' });
      setIsLoggedIn(false);
      setSystemState(null);
      setModuleStates({});
      setPerformance({});
      setAlerts([]);
    } catch (err) {
      console.error('Logout error:', err);
    }
  };
  
  const startTraining = async () => {
    try {
      await apiCall('/training/start', {
        method: 'POST',
        body: JSON.stringify(trainingConfig)
      });
    } catch (err) {
      setError(`Failed to start training: ${err.message}`);
    }
  };
  
  const stopTraining = async () => {
    try {
      await apiCall('/training/stop', { method: 'POST' });
    } catch (err) {
      setError(`Failed to stop training: ${err.message}`);
    }
  };
  
  const startTrading = async () => {
    try {
      await apiCall('/trading/start', {
        method: 'POST',
        body: JSON.stringify(tradingConfig)
      });
    } catch (err) {
      setError(`Failed to start trading: ${err.message}`);
    }
  };
  
  const stopTrading = async () => {
    try {
      await apiCall('/trading/stop', { method: 'POST' });
    } catch (err) {
      setError(`Failed to stop trading: ${err.message}`);
    }
  };
  
  const emergencyStop = async () => {
    if (!confirm('⚠️ EMERGENCY STOP: This will close all positions immediately. Are you sure?')) {
      return;
    }
    
    try {
      await apiCall('/trading/emergency-stop', { method: 'POST' });
    } catch (err) {
      setError(`Emergency stop failed: ${err.message}`);
    }
  };
  
  const toggleModule = async (moduleName) => {
    try {
      await apiCall(`/modules/${moduleName}/toggle`, { method: 'POST' });
    } catch (err) {
      setError(`Failed to toggle module ${moduleName}: ${err.message}`);
    }
  };
  
  const saveCheckpoint = async () => {
    try {
      await apiCall('/checkpoints/save', { method: 'POST' });
      fetchCheckpoints();
    } catch (err) {
      setError(`Failed to save checkpoint: ${err.message}`);
    }
  };
  
  const fetchLogs = async (category) => {
    try {
      const data = await apiCall(`/logs/${category}`);
      setLogs(prev => ({ ...prev, [category]: data }));
    } catch (err) {
      console.error(`Failed to fetch ${category} logs:`, err);
    }
  };
  
  const fetchCheckpoints = async () => {
    try {
      const data = await apiCall('/checkpoints');
      setCheckpoints(data.checkpoints || []);
    } catch (err) {
      console.error('Failed to fetch checkpoints:', err);
    }
  };
  
  const fetchTrainingHistory = async () => {
    try {
      const data = await apiCall('/training/metrics/history?limit=100');
      setTrainingHistory(data.metrics || []);
    } catch (err) {
      console.error('Failed to fetch training history:', err);
    }
  };
  
  const fetchCsvFiles = async () => {
    try {
      const data = await apiCall('/data/list');
      setCsvFiles(data.files || []);
    } catch (err) {
      console.error('Failed to fetch CSV files:', err);
    }
  };
  
  const startTensorBoard = async () => {
    try {
      const data = await apiCall('/tensorboard/start', { method: 'POST' });
      if (data.success) {
        setTensorboardUrl(data.url);
      }
    } catch (err) {
      console.error('Failed to start TensorBoard:', err);
    }
  };
  
  const uploadModel = async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await fetch(`${API_BASE}/model/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        setError('');
        alert('Model uploaded successfully');
      } else {
        const data = await response.json();
        setError(data.detail || 'Upload failed');
      }
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    }
  };
  
  const uploadCsvFiles = async (files) => {
    const formData = new FormData();
    for (const file of files) {
      formData.append('files', file);
    }
    
    try {
      setUploadProgress(0);
      const response = await fetch(`${API_BASE}/data/upload`, {
        method: 'POST',
        body: formData
      });
      
      if (response.ok) {
        setError('');
        await fetchCsvFiles();
        alert('CSV files uploaded successfully');
      } else {
        const data = await response.json();
        setError(data.detail || 'Upload failed');
      }
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
    } finally {
      setUploadProgress(0);
    }
  };
  
  // ═══════════════════════════════════════════════════════════════════
  // Effects
  // ═══════════════════════════════════════════════════════════════════
  
  useEffect(() => {
    if (isLoggedIn) {
      fetchCheckpoints();
      fetchCsvFiles();
      fetchLogs(selectedLogCategory);
      if (systemStatus === 'TRAINING') {
        fetchTrainingHistory();
      }
    }
  }, [isLoggedIn, selectedLogCategory, systemStatus]);
  
  // Auto-refresh logs
  useEffect(() => {
    if (isLoggedIn && activeTab === 'logs') {
      const interval = setInterval(() => {
        fetchLogs(selectedLogCategory);
      }, 10000); // Refresh every 10 seconds
      
      return () => clearInterval(interval);
    }
  }, [isLoggedIn, activeTab, selectedLogCategory]);
  
  // Auto-refresh training history when training
  useEffect(() => {
    if (systemStatus === 'TRAINING') {
      const interval = setInterval(fetchTrainingHistory, 30000); // Every 30 seconds
      return () => clearInterval(interval);
    }
  }, [systemStatus]);
  
  // ═══════════════════════════════════════════════════════════════════
  // Helper Components
  // ═══════════════════════════════════════════════════════════════════
  
  const StatusIndicator = ({ status, className = "" }) => {
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
  };
  
  const TabButton = ({ icon: Icon, label, active, onClick, badge, disabled = false }) => (
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
      {badge && (
        <span className={`text-xs px-2 py-1 rounded-full font-bold ${
          typeof badge === 'number' && badge > 0 
            ? 'bg-red-500 text-white' 
            : 'bg-blue-500 text-white'
        }`}>
          {badge}
        </span>
      )}
    </button>
  );
  
  const MetricCard = ({ title, value, icon: Icon, trend, color = 'blue', subtitle, onClick }) => (
    <div 
      className={`bg-gray-800 rounded-xl p-6 border border-gray-700 transition-all duration-200 ${
        onClick ? 'cursor-pointer hover:bg-gray-750 hover:border-gray-600' : ''
      }`}
      onClick={onClick}
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
  
  const ModuleCard = ({ name, module, onToggle, onClick }) => {
    const statusColors = {
      active: 'text-green-400',
      monitoring: 'text-blue-400',
      idle: 'text-gray-400',
      error: 'text-red-400',
      disabled: 'text-gray-500'
    };
    
    const enabled = module?.enabled ?? false;
    const status = module?.status || 'unknown';
    const errorCount = module?.errors?.length || 0;
    
    return (
      <div className="bg-gray-800 rounded-xl p-4 border border-gray-700 hover:border-gray-600 transition-all duration-200">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center">
            <h3 className="text-lg font-semibold text-white capitalize">
              {name.replace(/_/g, ' ')}
            </h3>
            <div className={`ml-2 w-2 h-2 rounded-full ${enabled ? 'bg-green-400' : 'bg-gray-500'}`} />
          </div>
          <button
            onClick={() => onToggle(name)}
            className={`px-3 py-1 text-xs rounded-full font-medium transition-colors ${
              enabled 
                ? 'bg-green-600 text-white hover:bg-green-700' 
                : 'bg-gray-600 text-gray-300 hover:bg-gray-500'
            }`}
          >
            {enabled ? 'Enabled' : 'Disabled'}
          </button>
        </div>
        
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-400">Status:</span>
            <span className={`text-sm font-medium ${statusColors[status] || 'text-gray-400'}`}>
              {status.toUpperCase()}
            </span>
          </div>
          
          {module?.last_update && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Last Update:</span>
              <span className="text-sm text-gray-300">
                {new Date(module.last_update).toLocaleTimeString()}
              </span>
            </div>
          )}
          
          {errorCount > 0 && (
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-400">Errors:</span>
              <span className="text-sm text-red-400 font-medium">{errorCount}</span>
            </div>
          )}
        </div>
        
        {onClick && (
          <button
            onClick={() => onClick(name, module)}
            className="w-full mt-3 px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-700 transition-colors"
          >
            View Details
          </button>
        )}
      </div>
    );
  };
  
  const AlertBadge = ({ alerts, onClick }) => {
    const criticalCount = alerts.filter(a => a.severity === 'critical').length;
    
    if (alerts.length === 0) return null;
    
    return (
      <button
        onClick={onClick}
        className="relative flex items-center px-3 py-1 bg-red-600 text-white text-sm rounded-full hover:bg-red-700 transition-colors"
      >
        <Bell className="w-4 h-4 mr-1" />
        <span>{alerts.length}</span>
        {criticalCount > 0 && (
          <div className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full animate-pulse" />
        )}
      </button>
    );
  };
  
  const ProgressBar = ({ value, max, label, color = 'blue' }) => {
    const percentage = (value / max) * 100;
    
    return (
      <div className="w-full">
        <div className="flex justify-between text-sm mb-1">
          <span className="text-gray-400">{label}</span>
          <span className="text-white font-medium">{percentage.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-700 rounded-full h-2">
          <div 
            className={`bg-${color}-500 h-2 rounded-full transition-all duration-300`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
    );
  };
  
  const ModeSelector = ({ value, onChange }) => (
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
  
  // Helper functions
  const formatTime = (seconds) => {
    if (seconds < 60) return `${Math.round(seconds)}s`;
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
    return `${Math.round(seconds / 3600)}h`;
  };
  
  const getRewardTrend = () => {
    if (trainingHistory.length < 2) return 0;
    const recent = trainingHistory.slice(-10);
    const older = trainingHistory.slice(-20, -10);
    const recentAvg = recent.reduce((sum, m) => sum + (m.episode_reward_mean || 0), 0) / recent.length;
    const olderAvg = older.reduce((sum, m) => sum + (m.episode_reward_mean || 0), 0) / older.length;
    return ((recentAvg - olderAvg) / Math.abs(olderAvg)) * 100;
  };
  
  // ═══════════════════════════════════════════════════════════════════
  // Login Screen
  // ═══════════════════════════════════════════════════════════════════
  
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
              <span className="text-sm text-gray-500">
                {wsConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">MT5 Login</label>
              <input
                type="text"
                value={loginForm.login}
                onChange={e => setLoginForm({...loginForm, login: e.target.value})}
                className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
                placeholder="12345678"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Password</label>
              <input
                type="password"
                value={loginForm.password}
                onChange={e => setLoginForm({...loginForm, password: e.target.value})}
                className="w-full bg-gray-700/50 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500 focus:ring-2 focus:ring-blue-500/20 transition-all"
                placeholder="Enter password"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Server</label>
              <input
                type="text"
                value={loginForm.server}
                onChange={e => setLoginForm({...loginForm, server: e.target.value})}
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
  
  // ═══════════════════════════════════════════════════════════════════
  // Tab Content Components
  // ═══════════════════════════════════════════════════════════════════
  
  const OverviewTab = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white">System Overview</h2>
          <p className="text-gray-400 mt-1">
            Uptime: {systemState?.uptime || '0m'} • Session: {systemState?.session_id?.slice(-8) || 'Unknown'}
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <AlertBadge alerts={alerts} onClick={() => setShowAlerts(true)} />
          
          {systemStatus === 'IDLE' && (
            <>
              <button
                onClick={startTraining}
                className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 px-4 py-2 rounded-lg transition-all duration-200 text-white font-medium"
              >
                <Brain className="w-4 h-4" />
                <span>Start Training</span>
              </button>
              <button
                onClick={startTrading}
                className="flex items-center space-x-2 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 px-4 py-2 rounded-lg transition-all duration-200 text-white font-medium"
              >
                <Play className="w-4 h-4" />
                <span>Start Trading</span>
              </button>
            </>
          )}
          {systemStatus === 'TRAINING' && (
            <button
              onClick={stopTraining}
              className="flex items-center space-x-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 px-4 py-2 rounded-lg transition-all duration-200 text-white font-medium"
            >
              <Square className="w-4 h-4" />
              <span>Stop Training</span>
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
        </div>
      </div>
      
      {/* Performance Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Current Balance"
          value={`$${performance.current_balance?.toLocaleString() || '0.00'}`}
          icon={DollarSign}
          trend={performance.total_pnl ? (performance.total_pnl / performance.start_balance * 100) : 0}
          color="green"
        />
        <MetricCard
          title="Total P&L"
          value={`$${performance.total_pnl?.toLocaleString() || '0.00'}`}
          icon={TrendingUp}
          color={performance.total_pnl >= 0 ? 'green' : 'red'}
          subtitle="Today"
        />
        <MetricCard
          title="Win Rate"
          value={`${(performance.win_rate * 100 || 0).toFixed(1)}%`}
          icon={Target}
          subtitle={`${performance.winning_trades || 0}/${performance.total_trades || 0} trades`}
          color="blue"
        />
        <MetricCard
          title="Max Drawdown"
          value={`${(performance.max_drawdown * 100 || 0).toFixed(1)}%`}
          icon={TrendingDown}
          color="purple"
          subtitle="All time"
        />
      </div>
      
      {/* Real-time Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center text-white">
            <LineChart className="w-5 h-5 mr-2 text-blue-400" />
            Performance Chart
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsLineChart data={[
                { time: '00:00', balance: performance.start_balance || 10000 },
                { time: '06:00', balance: (performance.start_balance || 10000) * 1.02 },
                { time: '12:00', balance: (performance.start_balance || 10000) * 0.98 },
                { time: '18:00', balance: performance.current_balance || 10000 },
              ]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip 
                  contentStyle={{ 
                    backgroundColor: '#1F2937', 
                    border: '1px solid #374151',
                    borderRadius: '8px'
                  }} 
                />
                <Line 
                  type="monotone" 
                  dataKey="balance" 
                  stroke="#3B82F6" 
                  strokeWidth={2}
                  dot={{ fill: '#3B82F6', strokeWidth: 2, r: 4 }}
                />
              </RechartsLineChart>
            </ResponsiveContainer>
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center text-white">
            <PieChart className="w-5 h-5 mr-2 text-purple-400" />
            Risk Distribution
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart>
                <Pie
                  data={[
                    { name: 'Safe Zone', value: 70, fill: '#10B981' },
                    { name: 'Moderate Risk', value: 25, fill: '#F59E0B' },
                    { name: 'High Risk', value: 5, fill: '#EF4444' }
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                />
                <Tooltip />
                <Legend />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
  
  const ModulesTab = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Module Management</h2>
        <div className="text-sm text-gray-400">
          {Object.values(moduleStates).filter(m => m.enabled).length} of {Object.keys(moduleStates).length} modules active
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {Object.entries(moduleStates).map(([name, module]) => (
          <ModuleCard
            key={name}
            name={name}
            module={module}
            onToggle={toggleModule}
            onClick={setSelectedModule}
          />
        ))}
      </div>
      
      {/* Module Details Modal */}
      {selectedModule && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
          <div className="bg-gray-800 rounded-xl p-6 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-white capitalize">
                {selectedModule.replace(/_/g, ' ')} Details
              </h3>
              <button
                onClick={() => setSelectedModule(null)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
            
            <div className="space-y-4">
              <pre className="bg-gray-900 p-4 rounded-lg text-sm text-gray-300 overflow-x-auto">
                {JSON.stringify(moduleStates[selectedModule], null, 2)}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  );
  
  const TrainingTab = () => {
    const [selectedChart, setSelectedChart] = useState('rewards');
    
    // Prepare chart data
    const chartData = trainingHistory.map((m, idx) => ({
      step: m.timestep || idx,
      reward: m.episode_reward_mean || 0,
      std: m.episode_reward_std || 0,
      length: m.episode_length_mean || 0,
      loss: m.policy_loss || 0,
      value_loss: m.value_loss || 0,
      learning_rate: m.learning_rate || 0,
      explained_variance: m.explained_variance || 0
    }));
    
    return (
      <div className="space-y-6">
        {/* Training Control Panel */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold text-white">PPO Training Control</h2>
            {systemStatus === 'TRAINING' && (
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse" />
                <span className="text-green-400 font-medium">Training Active</span>
              </div>
            )}
          </div>
          
          {/* Mode Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-400 mb-2">Training Mode</label>
            <ModeSelector 
              value={trainingConfig.mode} 
              onChange={(mode) => setTrainingConfig({...trainingConfig, mode})}
            />
            {trainingConfig.mode === 'online' && !systemState?.mt5_connected && (
              <div className="mt-2 bg-yellow-900/20 border border-yellow-500/20 text-yellow-400 px-4 py-2 rounded-lg">
                <AlertCircle className="w-4 h-4 inline mr-2" />
                MT5 must be connected for online training
              </div>
            )}
          </div>
          
          {/* Action Buttons */}
          <div className="flex items-center space-x-4">
            {systemStatus === 'IDLE' ? (
              <button
                onClick={startTraining}
                disabled={trainingConfig.mode === 'online' && !systemState?.mt5_connected}
                className="flex items-center space-x-2 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 px-6 py-3 rounded-lg transition-all duration-200 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Play className="w-5 h-5" />
                <span>Start {trainingConfig.mode === 'online' ? 'Online' : 'Offline'} Training</span>
              </button>
            ) : systemStatus === 'TRAINING' ? (
              <button
                onClick={stopTraining}
                className="flex items-center space-x-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 px-6 py-3 rounded-lg transition-all duration-200 text-white font-medium"
              >
                <Square className="w-5 h-5" />
                <span>Stop Training</span>
              </button>
            ) : (
              <div className="flex items-center space-x-2 text-yellow-400">
                <Loader2 className="w-5 h-5 animate-spin" />
                <span>Processing...</span>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };
  
  const TradingTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold text-white">Live Trading Configuration</h2>
      
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4 text-white">Trading Parameters</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Instruments</label>
            <input
              type="text"
              value={tradingConfig.instruments.join(', ')}
              onChange={e => setTradingConfig({...tradingConfig, instruments: e.target.value.split(',').map(s => s.trim())})}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
              placeholder="EURUSD, XAUUSD"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Timeframes</label>
            <input
              type="text"
              value={tradingConfig.timeframes.join(', ')}
              onChange={e => setTradingConfig({...tradingConfig, timeframes: e.target.value.split(',').map(s => s.trim())})}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
              placeholder="H1, H4, D1"
            />
          </div>
        </div>
        
        <div className="mt-6 flex items-center space-x-4">
          {systemStatus === 'IDLE' ? (
            <button
              onClick={startTrading}
              className="flex items-center space-x-2 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 px-6 py-3 rounded-lg transition-all duration-200 text-white font-medium"
            >
              <Play className="w-5 h-5" />
              <span>Start Trading</span>
            </button>
          ) : systemStatus === 'TRADING' ? (
            <>
              <button
                onClick={stopTrading}
                className="flex items-center space-x-2 bg-gradient-to-r from-yellow-600 to-yellow-700 hover:from-yellow-700 hover:to-yellow-800 px-6 py-3 rounded-lg transition-all duration-200 text-white font-medium"
              >
                <Pause className="w-5 h-5" />
                <span>Stop Trading</span>
              </button>
              <button
                onClick={emergencyStop}
                className="flex items-center space-x-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 px-6 py-3 rounded-lg transition-all duration-200 text-white font-medium"
              >
                <AlertTriangle className="w-5 h-5" />
                <span>Emergency Stop</span>
              </button>
            </>
          ) : (
            <div className="flex items-center space-x-2 text-yellow-400">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Processing...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
  
  const LogsTab = () => {
    const logCategories = ['system', 'training', 'risk', 'strategy', 'position'];
    
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-white">System Logs</h2>
          <button
            onClick={() => fetchLogs(selectedLogCategory)}
            className="flex items-center space-x-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors text-white"
          >
            <RefreshCw className="w-4 h-4" />
            <span>Refresh</span>
          </button>
        </div>
        
        <div className="flex flex-wrap gap-2 mb-4">
          {logCategories.map(cat => (
            <button
              key={cat}
              onClick={() => {
                setSelectedLogCategory(cat);
                fetchLogs(cat);
              }}
              className={`px-4 py-2 rounded-lg transition-colors font-medium ${
                selectedLogCategory === cat 
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </button>
          ))}
        </div>
        
        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="p-4 border-b border-gray-700 flex justify-between items-center bg-gray-900">
            <h3 className="text-lg font-semibold text-white">
              {selectedLogCategory.charAt(0).toUpperCase() + selectedLogCategory.slice(1)} Logs
            </h3>
            <div className="text-sm text-gray-400">
              {logs[selectedLogCategory]?.content?.length || 0} lines
            </div>
          </div>
          <div className="bg-black max-h-96 overflow-y-auto">
            <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap p-4 leading-relaxed">
              {logs[selectedLogCategory]?.content ? 
                logs[selectedLogCategory].content.join('') : 
                'Loading logs...'}
            </pre>
          </div>
        </div>
      </div>
    );
  };
  
  // ═══════════════════════════════════════════════════════════════════
  // Alerts Modal
  // ═══════════════════════════════════════════════════════════════════
  
  const AlertsModal = () => {
    if (!showAlerts) return null;
    
    const filteredAlerts = alerts.filter(alert => 
      alertFilter === 'all' || alert.severity === alertFilter
    );
    
    return (
      <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
        <div className="bg-gray-800 rounded-xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
          <div className="p-6 border-b border-gray-700 flex items-center justify-between">
            <h3 className="text-xl font-bold text-white">System Alerts</h3>
            <div className="flex items-center space-x-4">
              <select
                value={alertFilter}
                onChange={e => setAlertFilter(e.target.value)}
                className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-1 text-white text-sm"
              >
                <option value="all">All Alerts</option>
                <option value="critical">Critical</option>
                <option value="warning">Warning</option>
                <option value="success">Success</option>
                <option value="info">Info</option>
              </select>
              <button
                onClick={() => setShowAlerts(false)}
                className="text-gray-400 hover:text-white"
              >
                <X className="w-6 h-6" />
              </button>
            </div>
          </div>
          
          <div className="p-6 max-h-96 overflow-y-auto">
            {filteredAlerts.length > 0 ? (
              <div className="space-y-3">
                {filteredAlerts.map((alert, idx) => (
                  <div
                    key={idx}
                    className={`p-4 rounded-lg border-l-4 ${
                      alert.severity === 'critical' ? 'bg-red-900/20 border-red-500' :
                      alert.severity === 'warning' ? 'bg-yellow-900/20 border-yellow-500' :
                      alert.severity === 'success' ? 'bg-green-900/20 border-green-500' :
                      'bg-blue-900/20 border-blue-500'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-2 mb-1">
                          <span className={`text-xs px-2 py-1 rounded font-medium ${
                            alert.severity === 'critical' ? 'bg-red-600 text-white' :
                            alert.severity === 'warning' ? 'bg-yellow-600 text-white' :
                            alert.severity === 'success' ? 'bg-green-600 text-white' :
                            'bg-blue-600 text-white'
                          }`}>
                            {alert.severity.toUpperCase()}
                          </span>
                          <span className="text-xs text-gray-400">{alert.module}</span>
                        </div>
                        <p className="text-white">{alert.alert}</p>
                      </div>
                      <div className="text-xs text-gray-400 ml-4">
                        {new Date(alert.timestamp).toLocaleTimeString()}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center text-gray-400 py-8">
                No alerts found for the selected filter.
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };
  
  // ═══════════════════════════════════════════════════════════════════
  // Main Render
  // ═══════════════════════════════════════════════════════════════════
  
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
                {systemState?.uptime && (
                  <span className="flex items-center">
                    <Clock className="w-4 h-4 mr-1" />
                    {systemState.uptime}
                  </span>
                )}
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <AlertBadge alerts={alerts} onClick={() => setShowAlerts(true)} />
            
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
      
      <div className="flex">
        {/* Sidebar */}
        <nav className="w-64 bg-gray-800/50 backdrop-blur-sm border-r border-gray-700 min-h-screen sticky top-[73px]">
          <div className="p-4">
            <div className="space-y-2">
              <TabButton
                icon={BarChart3}
                label="Overview"
                active={activeTab === 'overview'}
                onClick={() => setActiveTab('overview')}
              />
              <TabButton
                icon={Cpu}
                label="Modules"
                active={activeTab === 'modules'}
                onClick={() => setActiveTab('modules')}
                badge={Object.values(moduleStates).filter(m => m.errors?.length > 0).length || null}
              />
              <TabButton
                icon={Brain}
                label="Training"
                active={activeTab === 'training'}
                onClick={() => setActiveTab('training')}
                disabled={systemStatus === 'TRADING'}
              />
              <TabButton
                icon={TrendingUp}
                label="Trading"
                active={activeTab === 'trading'}
                onClick={() => setActiveTab('trading')}
                disabled={systemStatus === 'TRAINING'}
              />
              <TabButton
                icon={Database}
                label="Logs"
                active={activeTab === 'logs'}
                onClick={() => setActiveTab('logs')}
              />
            </div>
            
            {/* Quick Stats in Sidebar */}
            <div className="mt-8 space-y-4">
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Current Balance</div>
                <div className="text-lg font-bold text-green-400">
                  ${performance.current_balance?.toLocaleString() || '0.00'}
                </div>
              </div>
              
              <div className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-xs text-gray-400 mb-1">Daily P&L</div>
                <div className={`text-lg font-bold ${performance.daily_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${performance.daily_pnl?.toLocaleString() || '0.00'}
                </div>
              </div>
              
              {systemStatus === 'TRAINING' && trainingMetrics && (
                <>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">Training Progress</div>
                    <div className="text-lg font-bold text-blue-400">
                      {((trainingMetrics.progress_pct || 0)).toFixed(1)}%
                    </div>
                  </div>
                  
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xs text-gray-400 mb-1">Episode Reward</div>
                    <div className="text-lg font-bold text-purple-400">
                      {(trainingMetrics.episode_reward_mean || 0).toFixed(2)}
                    </div>
                  </div>
                </>
              )}
              
              {systemStatus === 'TRADING' && (
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-xs text-gray-400 mb-1">Open Positions</div>
                  <div className="text-lg font-bold text-blue-400">
                    {moduleStates.position_manager?.position_count || 0}
                  </div>
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
            {activeTab === 'overview' && <OverviewTab />}
            {activeTab === 'modules' && <ModulesTab />}
            {activeTab === 'training' && <TrainingTab />}
            {activeTab === 'trading' && <TradingTab />}
            {activeTab === 'logs' && <LogsTab />}
          </div>
        </main>
      </div>
      
      {/* Alerts Modal */}
      <AlertsModal />
    </div>
  );
};

export default EnhancedTradingDashboard;