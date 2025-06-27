import React, { useState, useEffect, useCallback, useRef } from 'react';
import { 
  Play, Pause, Square, AlertTriangle, TrendingUp, 
  Activity, Database, Settings, LogOut, Brain,
  Shield, Target, BarChart3, Users, Zap,
  CheckCircle, XCircle, Clock, DollarSign,
  ArrowUp, ArrowDown, Wifi, WifiOff, Save,
  Upload, Download, RefreshCw, AlertCircle,
  Cpu, HardDrive, Network, Eye,
  BarChart2, PieChart, LineChart, Layers
} from 'lucide-react';
import { LineChart as RechartsLineChart, Line, AreaChart, Area, BarChart, Bar, PieChart as RechartsPieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

// Use relative URLs when served from same origin
const API_BASE = '/api';

// Enhanced Trading Dashboard with Full Module Integration
const TradingDashboard = () => {
  // Core state
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginForm, setLoginForm] = useState({ login: '', password: '', server: 'MetaQuotes-Demo' });
  const [systemStatus, setSystemStatus] = useState('IDLE');
  const [wsConnected, setWsConnected] = useState(false);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');
  
  // System state from backend
  const [systemState, setSystemState] = useState(null);
  const [performance, setPerformance] = useState({});
  const [moduleStates, setModuleStates] = useState({});
  
  // UI state
  const [logs, setLogs] = useState({});
  const [checkpoints, setCheckpoints] = useState([]);
  const [tensorboardUrl, setTensorboardUrl] = useState(null);
  const [selectedLogCategory, setSelectedLogCategory] = useState('system');
  
  // Configuration
  const [trainingConfig, setTrainingConfig] = useState({
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
    debug: false
  });
  
  // WebSocket reference
  const ws = useRef(null);
  
  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      // Use relative WebSocket URL when served from same origin
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
      
      console.log('Connecting to WebSocket:', wsUrl);
      ws.current = new WebSocket(wsUrl);
      
      ws.current.onopen = () => {
        setWsConnected(true);
        console.log('WebSocket connected');
      };
      
      ws.current.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'system_state') {
          setSystemState(data.data);
          setSystemStatus(data.data.status);
          setPerformance(data.data.performance || {});
          setModuleStates(data.data.modules || {});
        }
      };
      
      ws.current.onclose = () => {
        setWsConnected(false);
        console.log('WebSocket disconnected, reconnecting...');
        setTimeout(connectWebSocket, 3000);
      };
      
      ws.current.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    };
    
    connectWebSocket();
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
    };
  }, []);
  
  // API calls
  const handleLogin = async () => {
    try {
      const response = await fetch(`${API_BASE}/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          login: parseInt(loginForm.login),
          password: loginForm.password,
          server: loginForm.server
        })
      });
      
      const data = await response.json();
      
      if (response.ok) {
        setIsLoggedIn(true);
        setError('');
      } else {
        setError(data.detail || 'Login failed');
      }
    } catch (err) {
      setError('Connection failed: ' + err.message);
    }
  };
  
  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/logout`, { method: 'POST' });
      setIsLoggedIn(false);
      setSystemState(null);
    } catch (err) {
      console.error('Logout error:', err);
    }
  };
  
  const startTraining = async () => {
    try {
      const response = await fetch(`${API_BASE}/training/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(trainingConfig)
      });
      
      if (!response.ok) {
        const data = await response.json();
        setError(data.detail || 'Failed to start training');
      }
    } catch (err) {
      setError('Failed to start training: ' + err.message);
    }
  };
  
  const stopTraining = async () => {
    try {
      await fetch(`${API_BASE}/training/stop`, { method: 'POST' });
    } catch (err) {
      setError('Failed to stop training: ' + err.message);
    }
  };
  
  const startTrading = async () => {
    try {
      const response = await fetch(`${API_BASE}/trading/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(tradingConfig)
      });
      
      if (!response.ok) {
        const data = await response.json();
        setError(data.detail || 'Failed to start trading');
      }
    } catch (err) {
      setError('Failed to start trading: ' + err.message);
    }
  };
  
  const stopTrading = async () => {
    try {
      await fetch(`${API_BASE}/trading/stop`, { method: 'POST' });
    } catch (err) {
      setError('Failed to stop trading: ' + err.message);
    }
  };
  
  const emergencyStop = async () => {
    if (!confirm('Are you sure? This will close all positions and stop trading immediately.')) {
      return;
    }
    
    try {
      await fetch(`${API_BASE}/trading/emergency-stop`, { method: 'POST' });
    } catch (err) {
      setError('Emergency stop failed: ' + err.message);
    }
  };
  
  const saveCheckpoint = async () => {
    try {
      const response = await fetch(`${API_BASE}/checkpoints/save`, {
        method: 'POST'
      });
      
      if (response.ok) {
        fetchCheckpoints();
      }
    } catch (err) {
      setError('Failed to save checkpoint: ' + err.message);
    }
  };
  
  const fetchLogs = async (category) => {
    try {
      const response = await fetch(`${API_BASE}/logs/${category}`);
      const data = await response.json();
      setLogs(prev => ({ ...prev, [category]: data }));
    } catch (err) {
      console.error(`Failed to fetch ${category} logs:`, err);
    }
  };
  
  const fetchCheckpoints = async () => {
    try {
      const response = await fetch(`${API_BASE}/checkpoints`);
      const data = await response.json();
      setCheckpoints(data.checkpoints || []);
    } catch (err) {
      console.error('Failed to fetch checkpoints:', err);
    }
  };
  
  const startTensorBoard = async () => {
    try {
      const response = await fetch(`${API_BASE}/tensorboard/start`, {
        method: 'POST'
      });
      const data = await response.json();
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
      setError('Upload failed: ' + err.message);
    }
  };
  
  // Effects
  useEffect(() => {
    if (isLoggedIn) {
      fetchCheckpoints();
      fetchLogs(selectedLogCategory);
    }
  }, [isLoggedIn, selectedLogCategory]);
  
  // Helper components
  const StatusIndicator = ({ status }) => {
    const colors = {
      IDLE: 'text-gray-400',
      TRAINING: 'text-blue-400',
      TRADING: 'text-green-400',
      STOPPING: 'text-yellow-400',
      ERROR: 'text-red-400',
      EMERGENCY_STOP: 'text-red-600'
    };
    
    const icons = {
      IDLE: Clock,
      TRAINING: Brain,
      TRADING: TrendingUp,
      STOPPING: Pause,
      ERROR: XCircle,
      EMERGENCY_STOP: AlertTriangle
    };
    
    const Icon = icons[status] || Clock;
    
    return (
      <span className={`flex items-center ${colors[status] || 'text-gray-400'}`}>
        <Icon className="w-4 h-4 mr-1" />
      </span>
    );
  };
  
  const TabButton = ({ icon: Icon, label, active, onClick, badge }) => (
    <button
      onClick={onClick}
      className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-colors ${
        active 
          ? 'bg-blue-600 text-white' 
          : 'text-gray-300 hover:bg-gray-700 hover:text-white'
      }`}
    >
      <div className="flex items-center">
        <Icon className="w-5 h-5 mr-3" />
        <span>{label}</span>
      </div>
      {badge && (
        <span className="bg-red-500 text-white text-xs px-2 py-1 rounded-full">
          {badge}
        </span>
      )}
    </button>
  );
  
  const MetricCard = ({ title, value, icon: Icon, trend, color = 'blue', subtitle }) => (
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
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
          <div className={`flex items-center text-sm ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend > 0 ? <ArrowUp className="w-4 h-4 mr-1" /> : <ArrowDown className="w-4 h-4 mr-1" />}
            {Math.abs(trend).toFixed(2)}%
          </div>
        )}
      </div>
    </div>
  );
  
  // Login screen
  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="bg-gray-800 p-8 rounded-2xl shadow-2xl w-full max-w-md">
          <div className="flex items-center justify-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center">
              <Brain className="w-10 h-10 text-white" />
            </div>
          </div>
          
          <h1 className="text-3xl font-bold text-center mb-2">AI Trading System</h1>
          <p className="text-gray-400 text-center mb-8">PPO-Lagrangian Trading Dashboard</p>
          
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">MT5 Login</label>
              <input
                type="text"
                value={loginForm.login}
                onChange={e => setLoginForm({...loginForm, login: e.target.value})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                placeholder="12345678"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Password</label>
              <input
                type="password"
                value={loginForm.password}
                onChange={e => setLoginForm({...loginForm, password: e.target.value})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                placeholder="Enter password"
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-400 mb-2">Server</label>
              <input
                type="text"
                value={loginForm.server}
                onChange={e => setLoginForm({...loginForm, server: e.target.value})}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:border-blue-500"
                placeholder="MetaQuotes-Demo"
              />
            </div>
            
            {error && (
              <div className="bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-lg">
                {error}
              </div>
            )}
            
            <button
              onClick={handleLogin}
              className="w-full bg-gradient-to-r from-blue-500 to-purple-600 text-white font-medium py-3 rounded-lg hover:from-blue-600 hover:to-purple-700 transition-colors"
            >
              Connect to MT5
            </button>
          </div>
        </div>
      </div>
    );
  }
  
  // Tab content components
  const OverviewTab = () => (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold">System Overview</h2>
        <div className="flex items-center space-x-4">
          {systemStatus === 'IDLE' && (
            <>
              <button
                onClick={startTraining}
                className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors"
              >
                <Brain className="w-4 h-4" />
                <span>Start Training</span>
              </button>
              <button
                onClick={startTrading}
                className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 px-4 py-2 rounded-lg transition-colors"
              >
                <Play className="w-4 h-4" />
                <span>Start Trading</span>
              </button>
            </>
          )}
          {systemStatus === 'TRAINING' && (
            <button
              onClick={stopTraining}
              className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition-colors"
            >
              <Square className="w-4 h-4" />
              <span>Stop Training</span>
            </button>
          )}
          {systemStatus === 'TRADING' && (
            <>
              <button
                onClick={stopTrading}
                className="flex items-center space-x-2 bg-yellow-600 hover:bg-yellow-700 px-4 py-2 rounded-lg transition-colors"
              >
                <Pause className="w-4 h-4" />
                <span>Stop Trading</span>
              </button>
              <button
                onClick={emergencyStop}
                className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition-colors"
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
          value={`$${performance.current_balance?.toFixed(2) || '0.00'}`}
          icon={DollarSign}
          trend={performance.total_pnl ? (performance.total_pnl / performance.start_balance * 100) : 0}
          color="green"
        />
        <MetricCard
          title="Total P&L"
          value={`$${performance.total_pnl?.toFixed(2) || '0.00'}`}
          icon={TrendingUp}
          color={performance.total_pnl >= 0 ? 'green' : 'red'}
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
          icon={Shield}
          color="purple"
        />
      </div>
      
      {/* Risk Status */}
      {moduleStates.risk_controller && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Shield className="w-5 h-5 mr-2 text-purple-400" />
            Risk Management Status
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-400">Risk Level</div>
              <div className={`text-lg font-medium ${
                moduleStates.risk_controller.risk_level === 'HIGH' ? 'text-red-400' :
                moduleStates.risk_controller.risk_level === 'MEDIUM' ? 'text-yellow-400' :
                'text-green-400'
              }`}>
                {moduleStates.risk_controller.risk_level || 'NORMAL'}
              </div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Risk Scale</div>
              <div className="text-lg font-medium">{moduleStates.risk_controller.risk_scale?.toFixed(2) || '1.00'}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">VaR 95%</div>
              <div className="text-lg font-medium">${moduleStates.risk_controller.var_95?.toFixed(2) || '0.00'}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Current Drawdown</div>
              <div className="text-lg font-medium">{(moduleStates.risk_controller.drawdown * 100 || 0).toFixed(1)}%</div>
            </div>
          </div>
        </div>
      )}
      
      {/* System Errors/Warnings */}
      {systemState?.errors && systemState.errors.length > 0 && (
        <div className="bg-red-900/20 border border-red-600/50 rounded-xl p-4">
          <h3 className="text-lg font-semibold mb-2 flex items-center text-red-400">
            <AlertCircle className="w-5 h-5 mr-2" />
            Recent Errors
          </h3>
          <div className="space-y-2">
            {systemState.errors.slice(-5).map((error, idx) => (
              <div key={idx} className="text-sm text-red-300">
                <span className="text-gray-400">{new Date(error.timestamp).toLocaleTimeString()}</span>
                {' - '}
                {error.error}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
  
  const ModulesTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Module Status</h2>
      
      {/* Position Manager */}
      {moduleStates.position_manager && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-blue-400" />
            Position Manager
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-400">Open Positions</div>
              <div className="text-lg font-medium">{moduleStates.position_manager.position_count || 0}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Total Exposure</div>
              <div className="text-lg font-medium">{(moduleStates.position_manager.total_exposure * 100 || 0).toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Avg Hold Time</div>
              <div className="text-lg font-medium">{moduleStates.position_manager.average_holding_time || 0} bars</div>
            </div>
          </div>
          
          {/* Open positions table */}
          {Object.keys(moduleStates.position_manager.open_positions || {}).length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-medium text-gray-400 mb-2">Active Positions</h4>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400">
                      <th className="text-left py-2">Symbol</th>
                      <th className="text-left py-2">Type</th>
                      <th className="text-left py-2">Size</th>
                      <th className="text-left py-2">Entry</th>
                      <th className="text-left py-2">P&L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(moduleStates.position_manager.open_positions).map(([id, pos]) => (
                      <tr key={id} className="border-t border-gray-700">
                        <td className="py-2">{pos.symbol}</td>
                        <td className="py-2">
                          <span className={`px-2 py-1 rounded text-xs ${
                            pos.type === 'BUY' ? 'bg-green-600/20 text-green-400' : 'bg-red-600/20 text-red-400'
                          }`}>
                            {pos.type}
                          </span>
                        </td>
                        <td className="py-2">{pos.size}</td>
                        <td className="py-2">{pos.entry_price}</td>
                        <td className={`py-2 font-medium ${pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          ${pos.pnl?.toFixed(2)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
      
      {/* Strategy Arbiter */}
      {moduleStates.strategy_arbiter && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Users className="w-5 h-5 mr-2 text-purple-400" />
            Strategy Committee Voting
          </h3>
          <div className="mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-400">Consensus Level</span>
              <span className="text-lg font-medium">{(moduleStates.strategy_arbiter.consensus * 100 || 0).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                style={{ width: `${moduleStates.strategy_arbiter.consensus * 100 || 0}%` }}
              />
            </div>
          </div>
          
          {/* Member votes */}
          {moduleStates.strategy_arbiter.member_votes && (
            <div className="space-y-2">
              {Object.entries(moduleStates.strategy_arbiter.member_votes).map(([member, vote]) => (
                <div key={member} className="flex items-center justify-between text-sm">
                  <span className="text-gray-400">{member}</span>
                  <div className="flex items-center">
                    <span className={`font-medium ${vote > 0 ? 'text-green-400' : vote < 0 ? 'text-red-400' : 'text-gray-400'}`}>
                      {vote > 0 ? 'LONG' : vote < 0 ? 'SHORT' : 'NEUTRAL'}
                    </span>
                    <span className="ml-2 text-gray-500">({Math.abs(vote).toFixed(2)})</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* Theme Detector */}
      {moduleStates.theme_detector && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Layers className="w-5 h-5 mr-2 text-green-400" />
            Market Themes
          </h3>
          <div className="mb-4">
            <span className="text-sm text-gray-400">Current Regime: </span>
            <span className="text-lg font-medium text-blue-400">{moduleStates.theme_detector.market_regime || 'NEUTRAL'}</span>
          </div>
          
          {moduleStates.theme_detector.active_themes && moduleStates.theme_detector.active_themes.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-400">Active Themes</h4>
              {moduleStates.theme_detector.active_themes.map((theme, idx) => (
                <div key={idx} className="flex items-center justify-between">
                  <span className="text-sm">{theme}</span>
                  <div className="w-32 bg-gray-700 rounded-full h-2">
                    <div 
                      className="bg-green-500 h-2 rounded-full"
                      style={{ width: `${(moduleStates.theme_detector.theme_strengths[theme] || 0) * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
      
      {/* Memory Systems */}
      {moduleStates.memory_systems && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Database className="w-5 h-5 mr-2 text-yellow-400" />
            Memory Systems
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-400">Mistakes Learned</div>
              <div className="text-lg font-medium">{moduleStates.memory_systems.mistake_count || 0}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Playbook Size</div>
              <div className="text-lg font-medium">{moduleStates.memory_systems.playbook_size || 0}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Memory Usage</div>
              <div className="text-lg font-medium">{(moduleStates.memory_systems.memory_usage || 0).toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Compression</div>
              <div className="text-lg font-medium">{moduleStates.memory_systems.compression_ratio?.toFixed(2) || '1.00'}x</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
  
  const TrainingTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">PPO Training Configuration</h2>
      
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Training Parameters</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Total Timesteps</label>
            <input
              type="number"
              value={trainingConfig.timesteps}
              onChange={e => setTrainingConfig({...trainingConfig, timesteps: parseInt(e.target.value)})}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Learning Rate</label>
            <input
              type="number"
              value={trainingConfig.learning_rate}
              onChange={e => setTrainingConfig({...trainingConfig, learning_rate: parseFloat(e.target.value)})}
              step="0.0001"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Batch Size</label>
            <input
              type="number"
              value={trainingConfig.batch_size}
              onChange={e => setTrainingConfig({...trainingConfig, batch_size: parseInt(e.target.value)})}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">N Epochs</label>
            <input
              type="number"
              value={trainingConfig.n_epochs}
              onChange={e => setTrainingConfig({...trainingConfig, n_epochs: parseInt(e.target.value)})}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Gamma (Discount Factor)</label>
            <input
              type="number"
              value={trainingConfig.gamma}
              onChange={e => setTrainingConfig({...trainingConfig, gamma: parseFloat(e.target.value)})}
              step="0.01"
              min="0"
              max="1"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Clip Range</label>
            <input
              type="number"
              value={trainingConfig.clip_range}
              onChange={e => setTrainingConfig({...trainingConfig, clip_range: parseFloat(e.target.value)})}
              step="0.01"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
        
        <div className="mt-6 flex items-center">
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={trainingConfig.debug}
              onChange={e => setTrainingConfig({...trainingConfig, debug: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm text-gray-400">Enable Debug Mode</span>
          </label>
        </div>
      </div>
      
      {/* Model Management */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Model Management</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Upload Model</label>
            <input
              type="file"
              accept=".zip"
              onChange={e => e.target.files[0] && uploadModel(e.target.files[0])}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white"
            />
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={saveCheckpoint}
              className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors"
            >
              <Save className="w-4 h-4" />
              <span>Save Checkpoint</span>
            </button>
            
            <button
              onClick={startTensorBoard}
              className="flex items-center space-x-2 bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg transition-colors"
            >
              <BarChart2 className="w-4 h-4" />
              <span>Open TensorBoard</span>
            </button>
          </div>
          
          {tensorboardUrl && (
            <div className="mt-2">
              <a 
                href={tensorboardUrl} 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-blue-400 hover:text-blue-300"
              >
                TensorBoard is running at {tensorboardUrl}
              </a>
            </div>
          )}
        </div>
      </div>
      
      {/* Checkpoints */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Model Checkpoints</h3>
        
        {checkpoints.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-400">
                  <th className="text-left py-2">Name</th>
                  <th className="text-left py-2">Size</th>
                  <th className="text-left py-2">Created</th>
                  <th className="text-left py-2">Actions</th>
                </tr>
              </thead>
              <tbody>
                {checkpoints.map((checkpoint, idx) => (
                  <tr key={idx} className="border-t border-gray-700">
                    <td className="py-2">{checkpoint.name}</td>
                    <td className="py-2">{(checkpoint.size / 1024 / 1024).toFixed(2)} MB</td>
                    <td className="py-2">{new Date(checkpoint.created).toLocaleString()}</td>
                    <td className="py-2">
                      <button className="text-blue-400 hover:text-blue-300">
                        <Download className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-gray-400">No checkpoints available</p>
        )}
      </div>
    </div>
  );
  
  const TradingTab = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Live Trading Configuration</h2>
      
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Trading Parameters</h3>
        
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
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Update Interval (seconds)</label>
            <input
              type="number"
              value={tradingConfig.update_interval}
              onChange={e => setTradingConfig({...tradingConfig, update_interval: parseInt(e.target.value)})}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Max Position Size</label>
            <input
              type="number"
              value={tradingConfig.max_position_size}
              onChange={e => setTradingConfig({...tradingConfig, max_position_size: parseFloat(e.target.value)})}
              step="0.01"
              min="0"
              max="1"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Max Total Exposure</label>
            <input
              type="number"
              value={tradingConfig.max_total_exposure}
              onChange={e => setTradingConfig({...tradingConfig, max_total_exposure: parseFloat(e.target.value)})}
              step="0.01"
              min="0"
              max="1"
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Min Trade Interval (seconds)</label>
            <input
              type="number"
              value={tradingConfig.min_trade_interval}
              onChange={e => setTradingConfig({...tradingConfig, min_trade_interval: parseInt(e.target.value)})}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-blue-500"
            />
          </div>
        </div>
        
        <div className="mt-6 flex items-center space-x-4">
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={tradingConfig.use_trailing_stop}
              onChange={e => setTradingConfig({...tradingConfig, use_trailing_stop: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm text-gray-400">Use Trailing Stop</span>
          </label>
          
          <label className="flex items-center cursor-pointer">
            <input
              type="checkbox"
              checked={tradingConfig.debug}
              onChange={e => setTradingConfig({...tradingConfig, debug: e.target.checked})}
              className="mr-2"
            />
            <span className="text-sm text-gray-400">Debug Mode</span>
          </label>
        </div>
      </div>
      
      {/* Execution Monitor */}
      {moduleStates.execution_monitor && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-green-400" />
            Execution Quality
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <div className="text-sm text-gray-400">Fill Rate</div>
              <div className="text-lg font-medium">{(moduleStates.execution_monitor.fill_rate * 100 || 100).toFixed(1)}%</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Avg Slippage</div>
              <div className="text-lg font-medium">{moduleStates.execution_monitor.slippage?.toFixed(4) || '0.0000'}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Avg Spread</div>
              <div className="text-lg font-medium">{moduleStates.execution_monitor.avg_spread?.toFixed(4) || '0.0000'}</div>
            </div>
            <div>
              <div className="text-sm text-gray-400">Quality Score</div>
              <div className="text-lg font-medium">{(moduleStates.execution_monitor.execution_quality * 100 || 100).toFixed(0)}%</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
  
  const LogsTab = () => {
    const logCategories = ['system', 'training', 'risk', 'strategy', 'position', 'simulation'];
    
    return (
      <div className="space-y-6">
        <h2 className="text-2xl font-bold">System Logs</h2>
        
        <div className="flex space-x-2 mb-4">
          {logCategories.map(cat => (
            <button
              key={cat}
              onClick={() => {
                setSelectedLogCategory(cat);
                fetchLogs(cat);
              }}
              className={`px-4 py-2 rounded-lg transition-colors ${
                selectedLogCategory === cat 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {cat.charAt(0).toUpperCase() + cat.slice(1)}
            </button>
          ))}
        </div>
        
        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="p-4 border-b border-gray-700 flex justify-between items-center">
            <h3 className="text-lg font-semibold">
              {selectedLogCategory.charAt(0).toUpperCase() + selectedLogCategory.slice(1)} Logs
            </h3>
            <button
              onClick={() => fetchLogs(selectedLogCategory)}
              className="flex items-center space-x-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Refresh</span>
            </button>
          </div>
          <div className="p-4 bg-black max-h-96 overflow-y-auto">
            <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
              {logs[selectedLogCategory]?.content ? 
                logs[selectedLogCategory].content.join('') : 
                'Loading logs...'}
            </pre>
          </div>
          {logs[selectedLogCategory]?.files && logs[selectedLogCategory].files.length > 0 && (
            <div className="p-4 border-t border-gray-700 bg-gray-900">
              <div className="text-sm text-gray-400">
                Available files: {logs[selectedLogCategory].files.map(f => f.name).join(', ')}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };
  
  // Main render
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
              <Brain className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold">AI Trading Dashboard</h1>
              <div className="flex items-center space-x-4 text-sm text-gray-400">
                <span className="flex items-center">
                  {wsConnected ? <Wifi className="w-4 h-4 mr-1 text-green-400" /> : <WifiOff className="w-4 h-4 mr-1 text-red-400" />}
                  {wsConnected ? 'Connected' : 'Disconnected'}
                </span>
                <span className="flex items-center">
                  <StatusIndicator status={systemStatus} />
                  {systemStatus}
                </span>
              </div>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={saveCheckpoint}
              className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg transition-colors"
            >
              <Save className="w-4 h-4" />
              <span>Save Checkpoint</span>
            </button>
            <button
              onClick={handleLogout}
              className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg transition-colors"
            >
              <LogOut className="w-4 h-4" />
              <span>Logout</span>
            </button>
          </div>
        </div>
      </header>
      
      <div className="flex">
        {/* Sidebar */}
        <nav className="w-64 bg-gray-800 border-r border-gray-700 min-h-screen">
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
                badge={systemState?.errors?.length || null}
              />
              <TabButton
                icon={Brain}
                label="Training"
                active={activeTab === 'training'}
                onClick={() => setActiveTab('training')}
              />
              <TabButton
                icon={TrendingUp}
                label="Trading"
                active={activeTab === 'trading'}
                onClick={() => setActiveTab('trading')}
              />
              <TabButton
                icon={Database}
                label="Logs"
                active={activeTab === 'logs'}
                onClick={() => setActiveTab('logs')}
              />
            </div>
          </div>
        </nav>
        
        {/* Main content */}
        <main className="flex-1 p-6">
          {error && (
            <div className="mb-6 bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-lg flex items-center justify-between">
              <span>{error}</span>
              <button onClick={() => setError('')} className="text-red-400 hover:text-red-300">
                <XCircle className="w-5 h-5" />
              </button>
            </div>
          )}
          
          {activeTab === 'overview' && <OverviewTab />}
          {activeTab === 'modules' && <ModulesTab />}
          {activeTab === 'training' && <TrainingTab />}
          {activeTab === 'trading' && <TradingTab />}
          {activeTab === 'logs' && <LogsTab />}
        </main>
      </div>
    </div>
  );
};

export default TradingDashboard;