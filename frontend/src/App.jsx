import React, { useState, useEffect, useCallback } from 'react';
import { 
  Play, Pause, Square, AlertTriangle, TrendingUp, 
  Activity, Database, Settings, LogOut, Brain,
  Shield, Target, BarChart3, Users, Zap,
  CheckCircle, XCircle, Clock, DollarSign,
  ArrowUp, ArrowDown, Wifi, WifiOff, Save
} from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

// Main Dashboard Component
const TradingDashboard = () => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [loginForm, setLoginForm] = useState({ login: '', password: '', server: 'MetaQuotes-Demo' });
  const [systemStatus, setSystemStatus] = useState('IDLE');
  const [metrics, setMetrics] = useState(null);
  const [logs, setLogs] = useState({});
  const [activeTab, setActiveTab] = useState('overview');
  const [wsConnected, setWsConnected] = useState(false);
  const [expertVotes, setExpertVotes] = useState([]);
  const [correlationRisk, setCorrelationRisk] = useState(null);
  const [tensorboardUrl, setTensorboardUrl] = useState(null);
  const [error, setError] = useState('');

  // WebSocket connection for real-time updates
  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      setWsConnected(true);
      console.log('WebSocket connected');
    };
    
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      if (data.type === 'metrics') {
        setMetrics(data.data);
      } else if (data.type === 'login_success') {
        setIsLoggedIn(true);
        setError('');
      } else if (data.type === 'logout') {
        setIsLoggedIn(false);
      }
    };
    
    ws.onclose = () => {
      setWsConnected(false);
      console.log('WebSocket disconnected');
    };
    
    return () => ws.close();
  }, []);

  // Fetch initial data
  useEffect(() => {
    if (isLoggedIn) {
      fetchStatus();
      fetchVotes();
      fetchCorrelationRisk();
      fetchTensorBoard();
    }
  }, [isLoggedIn]);

  const fetchStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/status`);
      const data = await response.json();
      setMetrics(data);
      setSystemStatus(data.system_status);
    } catch (err) {
      console.error('Failed to fetch status:', err);
    }
  };

  const fetchLogs = async (category) => {
    try {
      const response = await fetch(`${API_BASE}/logs/${category}`);
      const data = await response.json();
      setLogs(prev => ({ ...prev, [category]: data.content }));
    } catch (err) {
      console.error(`Failed to fetch ${category} logs:`, err);
    }
  };

  const fetchVotes = async () => {
    try {
      const response = await fetch(`${API_BASE}/votes`);
      const data = await response.json();
      setExpertVotes(data.votes || []);
    } catch (err) {
      console.error('Failed to fetch votes:', err);
    }
  };

  const fetchCorrelationRisk = async () => {
    try {
      const response = await fetch(`${API_BASE}/metrics/correlation`);
      const data = await response.json();
      setCorrelationRisk(data);
    } catch (err) {
      console.error('Failed to fetch correlation risk:', err);
    }
  };

  const fetchTensorBoard = async () => {
    try {
      const response = await fetch(`${API_BASE}/tensorboard`);
      const data = await response.json();
      setTensorboardUrl(data.url);
    } catch (err) {
      console.error('Failed to fetch TensorBoard URL:', err);
    }
  };

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
      
      if (response.ok) {
        const data = await response.json();
        setIsLoggedIn(true);
        setError('');
      } else {
        const error = await response.json();
        setError(error.detail || 'Login failed');
      }
    } catch (err) {
      setError('Connection failed');
    }
  };

  const handleLogout = async () => {
    try {
      await fetch(`${API_BASE}/logout`, { method: 'POST' });
      setIsLoggedIn(false);
    } catch (err) {
      console.error('Logout failed:', err);
    }
  };

  const controlAction = async (action, data = {}) => {
    try {
      const response = await fetch(`${API_BASE}/${action}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Action failed');
      }
      
      return await response.json();
    } catch (err) {
      setError(err.message);
      throw err;
    }
  };

  const saveCheckpoint = async () => {
    try {
      await controlAction('checkpoint/save', { name: 'manual' });
      setError('');
    } catch (err) {
      // Error already handled in controlAction
    }
  };

  // Login Screen
  if (!isLoggedIn) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 flex items-center justify-center p-4">
        <div className="bg-gray-800/90 backdrop-blur-sm rounded-2xl shadow-2xl p-8 w-full max-w-md border border-gray-700">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-white mb-2">AI Trading Dashboard</h1>
            <p className="text-gray-400">Connect to MetaTrader 5</p>
          </div>
          
          <div className="space-y-4">
            <div>
              <label className="block text-gray-300 text-sm font-medium mb-2">Login ID</label>
              <input
                type="number"
                value={loginForm.login}
                onChange={(e) => setLoginForm(prev => ({ ...prev, login: e.target.value }))}
                className="w-full bg-gray-700 text-white rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                placeholder="Enter MT5 login"
              />
            </div>
            
            <div>
              <label className="block text-gray-300 text-sm font-medium mb-2">Password</label>
              <input
                type="password"
                value={loginForm.password}
                onChange={(e) => setLoginForm(prev => ({ ...prev, password: e.target.value }))}
                className="w-full bg-gray-700 text-white rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                placeholder="Enter password"
              />
            </div>
            
            <div>
              <label className="block text-gray-300 text-sm font-medium mb-2">Server</label>
              <input
                type="text"
                value={loginForm.server}
                onChange={(e) => setLoginForm(prev => ({ ...prev, server: e.target.value }))}
                className="w-full bg-gray-700 text-white rounded-lg px-4 py-3 focus:ring-2 focus:ring-blue-500 focus:outline-none"
                placeholder="e.g. MetaQuotes-Demo"
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

  // Main Dashboard
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
                icon={Settings}
                label="Control Panel"
                active={activeTab === 'control'}
                onClick={() => setActiveTab('control')}
              />
              <TabButton
                icon={Activity}
                label="Live Monitoring"
                active={activeTab === 'monitoring'}
                onClick={() => setActiveTab('monitoring')}
              />
              <TabButton
                icon={Brain}
                label="Training Progress"
                active={activeTab === 'training'}
                onClick={() => setActiveTab('training')}
              />
              <TabButton
                icon={Users}
                label="Expert Voting"
                active={activeTab === 'voting'}
                onClick={() => setActiveTab('voting')}
              />
              <TabButton
                icon={Shield}
                label="Risk Monitor"
                active={activeTab === 'risk'}
                onClick={() => setActiveTab('risk')}
              />
              <TabButton
                icon={Target}
                label="Trade Overview"
                active={activeTab === 'trades'}
                onClick={() => setActiveTab('trades')}
              />
              <TabButton
                icon={Database}
                label="Logs & Audit"
                active={activeTab === 'logs'}
                onClick={() => setActiveTab('logs')}
              />
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {activeTab === 'overview' && <OverviewTab metrics={metrics} />}
          {activeTab === 'control' && <ControlTab systemStatus={systemStatus} controlAction={controlAction} />}
          {activeTab === 'monitoring' && <MonitoringTab metrics={metrics} />}
          {activeTab === 'training' && <TrainingTab tensorboardUrl={tensorboardUrl} />}
          {activeTab === 'voting' && <VotingTab votes={expertVotes} />}
          {activeTab === 'risk' && <RiskTab correlationRisk={correlationRisk} metrics={metrics} />}
          {activeTab === 'trades' && <TradesTab metrics={metrics} />}
          {activeTab === 'logs' && <LogsTab logs={logs} fetchLogs={fetchLogs} />}
        </main>
      </div>
    </div>
  );
};

// Helper Components
const StatusIndicator = ({ status }) => {
  const getColor = () => {
    switch (status) {
      case 'TRAINING': return 'text-blue-400';
      case 'TRADING': return 'text-green-400';
      case 'PAUSED': return 'text-yellow-400';
      case 'ERROR': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return <div className={`w-2 h-2 rounded-full ${getColor().replace('text-', 'bg-')} mr-2`} />;
};

const TabButton = ({ icon: Icon, label, active, onClick }) => (
  <button
    onClick={onClick}
    className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg transition-colors ${
      active 
        ? 'bg-blue-600 text-white' 
        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
    }`}
  >
    <Icon className="w-5 h-5" />
    <span>{label}</span>
  </button>
);

const MetricCard = ({ title, value, icon: Icon, trend, color = 'blue' }) => (
  <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
    <div className="flex items-center justify-between mb-4">
      <h3 className="text-gray-400 text-sm font-medium">{title}</h3>
      <Icon className={`w-5 h-5 text-${color}-400`} />
    </div>
    <div className="flex items-end justify-between">
      <div className="text-2xl font-bold text-white">{value}</div>
      {trend && (
        <div className={`flex items-center text-sm ${trend > 0 ? 'text-green-400' : 'text-red-400'}`}>
          {trend > 0 ? <ArrowUp className="w-4 h-4 mr-1" /> : <ArrowDown className="w-4 h-4 mr-1" />}
          {Math.abs(trend)}%
        </div>
      )}
    </div>
  </div>
);

// Tab Components
const OverviewTab = ({ metrics }) => (
  <div className="space-y-6">
    <h2 className="text-2xl font-bold">System Overview</h2>
    
    {/* Account Metrics */}
    {metrics?.account && (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard
          title="Account Balance"
          value={`$${metrics.account.balance?.toFixed(2) || '0.00'}`}
          icon={DollarSign}
          color="green"
        />
        <MetricCard
          title="Equity"
          value={`$${metrics.account.equity?.toFixed(2) || '0.00'}`}
          icon={TrendingUp}
          color="blue"
        />
        <MetricCard
          title="Free Margin"
          value={`$${metrics.account.free_margin?.toFixed(2) || '0.00'}`}
          icon={Shield}
          color="purple"
        />
        <MetricCard
          title="Current P&L"
          value={`$${metrics.account.profit?.toFixed(2) || '0.00'}`}
          icon={Activity}
          color={metrics.account.profit >= 0 ? 'green' : 'red'}
        />
      </div>
    )}

    {/* Process Status */}
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <h3 className="text-lg font-semibold mb-4">System Processes</h3>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <ProcessStatus 
          name="Training" 
          status={metrics?.processes?.training} 
          icon={Brain}
        />
        <ProcessStatus 
          name="Trading" 
          status={metrics?.processes?.trading} 
          icon={Activity}
        />
        <ProcessStatus 
          name="TensorBoard" 
          status={metrics?.processes?.tensorboard} 
          icon={BarChart3}
        />
      </div>
    </div>

    {/* Positions */}
    {metrics?.positions && metrics.positions.length > 0 && (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Open Positions</h3>
        <div className="space-y-2">
          {metrics.positions.slice(0, 5).map((pos, i) => (
            <div key={i} className="flex items-center justify-between py-2 border-b border-gray-700 last:border-b-0">
              <div className="flex items-center space-x-4">
                <span className="font-medium">{pos.symbol}</span>
                <span className={`px-2 py-1 rounded text-xs ${pos.type === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                  {pos.type}
                </span>
                <span className="text-gray-400">{pos.volume}</span>
              </div>
              <div className="text-right">
                <div className={`font-medium ${pos.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  ${pos.profit?.toFixed(2)}
                </div>
                <div className="text-sm text-gray-400">
                  {pos.price_open} → {pos.price_current}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )}
  </div>
);

const ProcessStatus = ({ name, status, icon: Icon }) => (
  <div className="flex items-center space-x-3 p-3 bg-gray-700 rounded-lg">
    <Icon className="w-5 h-5 text-gray-400" />
    <div className="flex-1">
      <div className="font-medium">{name}</div>
      <div className="text-sm text-gray-400">
        {status ? 'Running' : 'Stopped'}
      </div>
    </div>
    <div className={`w-3 h-3 rounded-full ${status ? 'bg-green-400' : 'bg-gray-500'}`} />
  </div>
);

const ControlTab = ({ systemStatus, controlAction }) => {
  const [trainConfig, setTrainConfig] = useState({
    model_type: 'ppo',
    timesteps: 100000,
    learning_rate: 0.0003,
    debug: true
  });

  const [tradeConfig, setTradeConfig] = useState({
    symbols: ['EURUSD', 'XAUUSD'],
    max_risk_per_trade: 0.02,
    max_drawdown: 0.20
  });

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Control Panel</h2>
      
      {/* System Control */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">System Control</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <ControlButton
            icon={Play}
            label="Start Training"
            onClick={() => controlAction('training/start', trainConfig)}
            disabled={systemStatus === 'TRAINING'}
            variant="blue"
          />
          <ControlButton
            icon={Pause}
            label="Stop Training"
            onClick={() => controlAction('training/stop')}
            disabled={systemStatus !== 'TRAINING'}
            variant="yellow"
          />
          <ControlButton
            icon={Activity}
            label="Start Trading"
            onClick={() => controlAction('trading/start', tradeConfig)}
            disabled={systemStatus === 'TRADING'}
            variant="green"
          />
          <ControlButton
            icon={Square}
            label="Emergency Stop"
            onClick={() => controlAction('trading/emergency_stop')}
            variant="red"
          />
        </div>
      </div>

      {/* Training Configuration */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Training Configuration</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-gray-300 text-sm font-medium mb-2">Model Type</label>
            <select
              value={trainConfig.model_type}
              onChange={(e) => setTrainConfig(prev => ({ ...prev, model_type: e.target.value }))}
              className="w-full bg-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            >
              <option value="ppo">PPO</option>
              <option value="sac">SAC</option>
              <option value="td3">TD3</option>
            </select>
          </div>
          <div>
            <label className="block text-gray-300 text-sm font-medium mb-2">Timesteps</label>
            <input
              type="number"
              value={trainConfig.timesteps}
              onChange={(e) => setTrainConfig(prev => ({ ...prev, timesteps: parseInt(e.target.value) }))}
              className="w-full bg-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-gray-300 text-sm font-medium mb-2">Learning Rate</label>
            <input
              type="number"
              step="0.0001"
              value={trainConfig.learning_rate}
              onChange={(e) => setTrainConfig(prev => ({ ...prev, learning_rate: parseFloat(e.target.value) }))}
              className="w-full bg-gray-700 text-white rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500 focus:outline-none"
            />
          </div>
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={trainConfig.debug}
              onChange={(e) => setTrainConfig(prev => ({ ...prev, debug: e.target.checked }))}
              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
            />
            <label className="text-gray-300 text-sm font-medium">Debug Mode</label>
          </div>
        </div>
      </div>
    </div>
  );
};

const ControlButton = ({ icon: Icon, label, onClick, disabled, variant = 'blue' }) => {
  const getColors = () => {
    const colors = {
      blue: disabled ? 'bg-gray-600 text-gray-400' : 'bg-blue-600 hover:bg-blue-700 text-white',
      green: disabled ? 'bg-gray-600 text-gray-400' : 'bg-green-600 hover:bg-green-700 text-white',
      yellow: disabled ? 'bg-gray-600 text-gray-400' : 'bg-yellow-600 hover:bg-yellow-700 text-white',
      red: disabled ? 'bg-gray-600 text-gray-400' : 'bg-red-600 hover:bg-red-700 text-white'
    };
    return colors[variant];
  };

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`flex items-center justify-center space-x-2 px-4 py-3 rounded-lg transition-colors ${getColors()}`}
    >
      <Icon className="w-5 h-5" />
      <span className="font-medium">{label}</span>
    </button>
  );
};

const MonitoringTab = ({ metrics }) => (
  <div className="space-y-6">
    <h2 className="text-2xl font-bold">Live Monitoring</h2>
    
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Real-time Metrics */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Real-time Metrics</h3>
        <div className="space-y-4">
          <MetricRow label="System Status" value={metrics?.system_status || 'Unknown'} />
          <MetricRow label="MT5 Connection" value={metrics?.mt5_connected ? 'Connected' : 'Disconnected'} />
          <MetricRow label="Last Update" value={new Date(metrics?.timestamp || Date.now()).toLocaleTimeString()} />
          {metrics?.account && (
            <>
              <MetricRow label="Margin Level" value={`${metrics.account.margin_level?.toFixed(1) || 0}%`} />
              <MetricRow label="Open Positions" value={metrics.positions?.length || 0} />
            </>
          )}
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Performance</h3>
        <div className="space-y-4">
          {metrics?.training && (
            <div className="text-sm text-gray-400">
              <div>Latest Training Log:</div>
              <div className="mt-2 p-2 bg-gray-700 rounded text-xs font-mono">
                {metrics.training.latest_log}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  </div>
);

const MetricRow = ({ label, value }) => (
  <div className="flex justify-between">
    <span className="text-gray-400">{label}</span>
    <span className="text-white font-medium">{value}</span>
  </div>
);

const TrainingTab = ({ tensorboardUrl }) => (
  <div className="space-y-6">
    <h2 className="text-2xl font-bold">Training Progress</h2>
    
    {tensorboardUrl ? (
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold">TensorBoard</h3>
          <p className="text-gray-400 text-sm">Real-time training metrics and visualizations</p>
        </div>
        <iframe
          src={tensorboardUrl}
          className="w-full h-96 bg-white"
          title="TensorBoard"
        />
      </div>
    ) : (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 text-center">
        <Brain className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">TensorBoard Not Available</h3>
        <p className="text-gray-400">Start training to see real-time metrics</p>
      </div>
    )}
  </div>
);

const VotingTab = ({ votes }) => (
  <div className="space-y-6">
    <h2 className="text-2xl font-bold">Expert Voting System</h2>
    
    <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
      <h3 className="text-lg font-semibold mb-4">Recent Votes</h3>
      {votes.length > 0 ? (
        <div className="space-y-2">
          {votes.map((vote, i) => (
            <div key={i} className="p-3 bg-gray-700 rounded-lg text-sm font-mono">
              {vote}
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center text-gray-400 py-8">
          <Users className="w-12 h-12 mx-auto mb-4" />
          <p>No recent voting activity</p>
        </div>
      )}
    </div>
  </div>
);

const RiskTab = ({ correlationRisk, metrics }) => (
  <div className="space-y-6">
    <h2 className="text-2xl font-bold">Risk Monitor</h2>
    
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Correlation Risk</h3>
        {correlationRisk ? (
          <div className="space-y-4">
            <MetricRow 
              label="Risk Score" 
              value={`${(correlationRisk.correlation_score * 100).toFixed(1)}%`}
            />
            <div>
              <div className="text-gray-400 text-sm mb-2">Warnings</div>
              {correlationRisk.risk_warnings?.length > 0 ? (
                <div className="space-y-1">
                  {correlationRisk.risk_warnings.map((warning, i) => (
                    <div key={i} className="text-yellow-400 text-sm">{warning}</div>
                  ))}
                </div>
              ) : (
                <div className="text-green-400 text-sm">No warnings</div>
              )}
            </div>
          </div>
        ) : (
          <div className="text-gray-400">No correlation data available</div>
        )}
      </div>

      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <h3 className="text-lg font-semibold mb-4">Risk Limits</h3>
        <div className="space-y-4">
          <MetricRow label="Max Drawdown" value="20%" />
          <MetricRow label="Position Limit" value="25%" />
          <MetricRow label="Correlation Limit" value="80%" />
          <MetricRow label="VaR Limit" value="10%" />
        </div>
      </div>
    </div>
  </div>
);

const TradesTab = ({ metrics }) => (
  <div className="space-y-6">
    <h2 className="text-2xl font-bold">Trade Overview</h2>
    
    {metrics?.positions && metrics.positions.length > 0 ? (
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold">Open Positions</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Symbol</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Type</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Volume</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Open Price</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Current</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">P&L</th>
                <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-700">
              {metrics.positions.map((pos, i) => (
                <tr key={i} className="hover:bg-gray-700">
                  <td className="px-4 py-3 font-medium">{pos.symbol}</td>
                  <td className="px-4 py-3">
                    <span className={`px-2 py-1 rounded text-xs ${pos.type === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>
                      {pos.type}
                    </span>
                  </td>
                  <td className="px-4 py-3">{pos.volume}</td>
                  <td className="px-4 py-3">{pos.price_open}</td>
                  <td className="px-4 py-3">{pos.price_current}</td>
                  <td className={`px-4 py-3 font-medium ${pos.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    ${pos.profit?.toFixed(2)}
                  </td>
                  <td className="px-4 py-3 text-gray-400 text-sm">
                    {new Date(pos.time).toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    ) : (
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700 text-center">
        <Target className="w-12 h-12 text-gray-400 mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Open Positions</h3>
        <p className="text-gray-400">Start trading to see active positions</p>
      </div>
    )}
  </div>
);

const LogsTab = ({ logs, fetchLogs }) => {
  const [selectedCategory, setSelectedCategory] = useState('training');
  
  const categories = [
    'training', 'risk', 'position', 'strategy', 
    'simulation', 'performance', 'trading'
  ];

  useEffect(() => {
    fetchLogs(selectedCategory);
  }, [selectedCategory, fetchLogs]);

  return (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">Logs & Audit Trail</h2>
      
      <div className="flex space-x-2 mb-4">
        {categories.map(cat => (
          <button
            key={cat}
            onClick={() => setSelectedCategory(cat)}
            className={`px-4 py-2 rounded-lg transition-colors ${
              selectedCategory === cat 
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
          <h3 className="text-lg font-semibold">{selectedCategory.charAt(0).toUpperCase() + selectedCategory.slice(1)} Logs</h3>
          <button
            onClick={() => fetchLogs(selectedCategory)}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors"
          >
            Refresh
          </button>
        </div>
        <div className="p-4 bg-black max-h-96 overflow-y-auto">
          <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap">
            {logs[selectedCategory] ? logs[selectedCategory].join('') : 'Loading logs...'}
          </pre>
        </div>
      </div>
    </div>
  );
};

export default TradingDashboard;