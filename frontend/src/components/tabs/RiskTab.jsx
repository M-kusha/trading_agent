import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  RefreshCw, Shield, AlertTriangle, CheckCircle, TrendingDown,
  Target, Activity, Bell, AlertCircle, PieChart as PieChartIcon
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart as RechartsLineChart,
  Line
} from 'recharts';

// Risk Overview Chart
const RiskOverviewChart = React.memo(function RiskOverviewChart({ data }) {
  const chartData = [
    { name: 'Current DD', value: (data.current_drawdown || 0) * 100, color: '#ef4444' },
    { name: 'Max DD', value: (data.max_drawdown || 0) * 100, color: '#dc2626' },
    { name: 'VAR 95%', value: (data.var_95 || 0) * 100, color: '#f97316' },
    { name: 'VAR 99%', value: (data.var_99 || 0) * 100, color: '#ea580c' }
  ];

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Risk Metrics Overview</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" stroke="#9CA3AF" />
            <YAxis stroke="#9CA3AF" />
            <Tooltip
              contentStyle={{
                backgroundColor: '#1F2937',
                border: '1px solid #374151',
                borderRadius: '0.5rem'
              }}
              formatter={(value) => [`${value.toFixed(2)}%`, 'Value']}
            />
            <Bar dataKey="value" fill="#ef4444" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
});

const RiskMetricsPanel = React.memo(function RiskMetricsPanel({ data }) {
  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Risk Metrics</h3>
      <div className="space-y-4">
        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Sharpe Ratio</span>
            <span className="text-blue-400 font-medium">
              {(data.sharpe_ratio || 0).toFixed(2)}
            </span>
          </div>
        </div>

        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Volatility Ratio</span>
            <span className="text-purple-400 font-medium">
              {(data.volatility_ratio || 1.0).toFixed(2)}x
            </span>
          </div>
        </div>

        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Risk Budget Used</span>
            <span className="text-yellow-400 font-medium">
              {((data.risk_budget_used || 0) * 100).toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">System Status</span>
            <span className="text-green-400 font-medium capitalize">
              {data.system_status || 'Unknown'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

const AnomaliesPanel = React.memo(function AnomaliesPanel({ data }) {
  const rawScore = data.anomaly_score;
  const anomalyScore = typeof rawScore === 'number' ? rawScore : (parseFloat(rawScore) || 0);
  const threshold = typeof data.anomaly_threshold === 'number' ? data.anomaly_threshold : 0.8;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-red-400" />
            Anomaly Detection
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Current Score</span>
              <div className="flex items-center space-x-2">
                <span className={`font-bold ${anomalyScore > threshold ? 'text-red-400' : 'text-green-400'}`}>
                  {anomalyScore.toFixed(3)}
                </span>
                <div className={`w-3 h-3 rounded-full ${anomalyScore > threshold ? 'bg-red-400' : 'bg-green-400'}`}></div>
              </div>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-300 ${
                  anomalyScore > threshold ? 'bg-red-400' : 'bg-green-400'
                }`}
                style={{ width: `${Math.min(anomalyScore * 100, 100)}%` }}
              />
            </div>
            <div className="flex justify-between text-sm text-gray-400">
              <span>Normal</span>
              <span>Threshold: {threshold}</span>
              <span>Anomaly</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">System Health</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Detection Mode</span>
              <span className="text-blue-400 font-medium">
                {data.detection_mode || 'NORMAL'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Active Alerts</span>
              <span className="text-red-400 font-medium">
                {(data.anomaly_alerts || []).length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">History Count</span>
              <span className="text-purple-400 font-medium">
                {(data.anomaly_history || []).length}
              </span>
            </div>
          </div>
        </div>
      </div>

      {(data.anomaly_alerts || []).length > 0 && (
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Anomaly Alerts</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {(data.anomaly_alerts || []).slice(0, 10).map((alert, index) => (
              <div key={index} className="bg-red-900/20 border border-red-500/30 rounded p-3">
                <div className="flex justify-between items-start">
                  <span className="text-red-300">{alert.message || alert.type || 'Anomaly detected'}</span>
                  <span className="text-xs text-gray-400">
                    {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : 'Recent'}
                  </span>
                </div>
                {alert.severity && (
                  <span className={`text-xs px-2 py-1 rounded mt-2 inline-block ${
                    alert.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                    alert.severity === 'warning' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-blue-500/20 text-blue-400'
                  }`}>
                    {alert.severity.toUpperCase()}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

const CompliancePanel = React.memo(function CompliancePanel({ data }) {
  const violations = data.compliance_violations || [];
  const limits = data.risk_limits || {};

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <CheckCircle className="w-5 h-5 mr-2 text-green-400" />
            Trade Compliance
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Status</span>
              <span className={`font-medium ${violations.length === 0 ? 'text-green-400' : 'text-red-400'}`}>
                {violations.length === 0 ? 'COMPLIANT' : 'VIOLATIONS'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Active Violations</span>
              <span className="text-red-400 font-medium">
                {violations.length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Leverage</span>
              <span className="text-blue-400 font-medium">
                {limits.max_leverage || 'N/A'}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Position Limits</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Max Position Risk</span>
              <span className="text-yellow-400 font-medium">
                {((data.position_compliance?.max_position_risk || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Current Exposure</span>
              <span className="text-purple-400 font-medium">
                {((data.position_compliance?.current_exposure || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Compliance Score</span>
              <span className="text-green-400 font-medium">
                {((data.position_compliance?.compliance_score || 1.0) * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Daily Limits</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Max Daily Trades</span>
              <span className="text-cyan-400 font-medium">
                {data.daily_limits?.max_daily_trades || 'N/A'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Today's Trades</span>
              <span className="text-orange-400 font-medium">
                {data.daily_limits?.current_trades || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Limit Utilization</span>
              <span className="text-blue-400 font-medium">
                {data.daily_limits?.max_daily_trades ?
                  `${((data.daily_limits.current_trades || 0) / data.daily_limits.max_daily_trades * 100).toFixed(1)}%` :
                  'N/A'
                }
              </span>
            </div>
          </div>
        </div>
      </div>

      {violations.length > 0 && (
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Compliance Violations</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {violations.slice(0, 10).map((violation, index) => (
              <div key={index} className="bg-red-900/20 border border-red-500/30 rounded p-3">
                <div className="flex justify-between items-start">
                  <span className="text-red-300">{violation.message || violation.type}</span>
                  <span className="text-xs text-gray-400">
                    {violation.timestamp ? new Date(violation.timestamp).toLocaleTimeString() : 'Recent'}
                  </span>
                </div>
                {violation.severity && (
                  <span className={`text-xs px-2 py-1 rounded mt-2 inline-block ${
                    violation.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                    violation.severity === 'warning' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-blue-500/20 text-blue-400'
                  }`}>
                    {violation.severity.toUpperCase()}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

const DrawdownPanel = React.memo(function DrawdownPanel({ data }) {
  const rescueActive = data.rescue_active || false;
  const currentDD = data.current_drawdown || 0;
  const maxDD = data.max_drawdown || 0;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className={`backdrop-blur-sm border rounded-lg p-6 ${
          rescueActive ? 'bg-red-500/20 border-red-500/50' : 'bg-gray-800/50 border-gray-700'
        }`}>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <TrendingDown className="w-5 h-5 mr-2 text-red-400" />
            Drawdown Status
            {rescueActive && (
              <span className="ml-2 px-2 py-1 bg-red-500/30 text-red-300 text-xs rounded">
                RESCUE ACTIVE
              </span>
            )}
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-gray-400">Current Drawdown</span>
              <span className="text-red-400 font-bold">
                {(currentDD * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Max Drawdown</span>
              <span className="text-red-500 font-bold">
                {(maxDD * 100).toFixed(2)}%
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div
                className="bg-red-400 h-3 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(currentDD * 100 * 4, 100)}%` }}
              />
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Recovery Analysis</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Recovery Progress</span>
              <span className="text-green-400 font-medium">
                {((data.recovery_progress?.recovery_ratio || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Velocity Analysis</span>
              <span className="text-blue-400 font-medium">
                {data.velocity_analysis?.velocity_trend || 'Neutral'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Rescue Triggers</span>
              <span className="text-purple-400 font-medium">
                {(data.rescue_triggers || []).length}
              </span>
            </div>
          </div>
        </div>
      </div>

      {(data.drawdown_history || []).length > 0 && (
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Drawdown History</h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsLineChart data={data.drawdown_history?.slice(-20) || []}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="timestamp" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '0.5rem'
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="drawdown"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                />
              </RechartsLineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}
    </div>
  );
});

const ExecutionPanel = React.memo(function ExecutionPanel({ data }) {
  const qualityScore = data.quality_score || 0;
  const executionVote = data.execution_vote || 'ABSTAIN';

  const getVoteColor = (vote) => {
    switch (vote) {
      case 'PROCEED': return 'text-green-400';
      case 'CAUTION': return 'text-yellow-400';
      case 'HALT': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getVoteBg = (vote) => {
    switch (vote) {
      case 'PROCEED': return 'bg-green-500/20';
      case 'CAUTION': return 'bg-yellow-500/20';
      case 'HALT': return 'bg-red-500/20';
      default: return 'bg-gray-500/20';
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className={`backdrop-blur-sm border rounded-lg p-6 ${getVoteBg(executionVote)} border-gray-700`}>
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2 text-blue-400" />
            Execution Vote
          </h3>
          <div className="text-center">
            <div className={`text-3xl font-bold ${getVoteColor(executionVote)}`}>
              {executionVote}
            </div>
            <div className="text-sm text-gray-400 mt-2">Current recommendation</div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Quality Score</h3>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400">
              {(qualityScore * 100).toFixed(1)}%
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 mt-4">
              <div
                className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${qualityScore * 100}%` }}
              />
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Execution Metrics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Fill Rate</span>
              <span className="text-green-400 font-medium">
                {((data.fill_rate_analysis?.current_fill_rate || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Slippage</span>
              <span className="text-yellow-400 font-medium">
                {((data.slippage_analysis?.average_slippage || 0) * 10000).toFixed(1)} pips
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Latency</span>
              <span className="text-purple-400 font-medium">
                {data.latency_metrics?.average_latency || 0}ms
              </span>
            </div>
          </div>
        </div>
      </div>

      {(data.execution_alerts || []).length > 0 && (
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Execution Alerts</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {(data.execution_alerts || []).slice(0, 5).map((alert, index) => (
              <div key={index} className="bg-yellow-900/20 border border-yellow-500/30 rounded p-3">
                <div className="flex justify-between items-start">
                  <span className="text-yellow-300">{alert.message || alert.type}</span>
                  <span className="text-xs text-gray-400">
                    {alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : 'Recent'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

const PortfolioRiskPanel = React.memo(function PortfolioRiskPanel({ data }) {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <PieChartIcon className="w-5 h-5 mr-2 text-cyan-400" />
            Portfolio Risk
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Exposure</span>
              <span className="text-cyan-400 font-medium">
                {((data.exposure_analysis?.total_exposure || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">VAR Analysis</span>
              <span className="text-red-400 font-medium">
                {((data.var_analysis?.current_var || 0) * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Correlation Risk</span>
              <span className="text-yellow-400 font-medium">
                {(data.correlation_risk?.risk_score || 0).toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Diversification</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Diversification Score</span>
              <span className="text-green-400 font-medium">
                {((data.diversification_metrics?.diversification_score || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Asset Classes</span>
              <span className="text-blue-400 font-medium">
                {data.diversification_metrics?.asset_class_count || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Position Count</span>
              <span className="text-purple-400 font-medium">
                {data.position_risk?.active_positions || 0}
              </span>
            </div>
          </div>
        </div>
      </div>

      {data.correlation_matrix && Object.keys(data.correlation_matrix).length > 0 && (
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Correlation Matrix</h3>
          <div className="text-sm text-gray-400 text-center">
            {Object.keys(data.correlation_matrix).length} pairs analyzed
          </div>
        </div>
      )}
    </div>
  );
});

const DynamicRiskPanel = React.memo(function DynamicRiskPanel({ data }) {
  const controlMode = data.control_mode || 'NORMAL';
  const riskScale = data.risk_scale || 1.0;

  const getModeColor = (mode) => {
    switch (mode) {
      case 'NORMAL': return 'text-green-400';
      case 'PROTECTIVE': return 'text-yellow-400';
      case 'AGGRESSIVE_REDUCTION': return 'text-red-400';
      case 'EMERGENCY': return 'text-red-500';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Activity className="w-5 h-5 mr-2 text-purple-400" />
            Control Mode
          </h3>
          <div className="text-center">
            <div className={`text-2xl font-bold ${getModeColor(controlMode)}`}>
              {controlMode}
            </div>
            <div className="text-sm text-gray-400 mt-2">Current operational mode</div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Scaling</h3>
          <div className="text-center">
            <div className="text-3xl font-bold text-blue-400">
              {riskScale.toFixed(2)}x
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2 mt-4">
              <div
                className="bg-blue-400 h-2 rounded-full transition-all duration-300"
                style={{ width: `${Math.min(riskScale * 50, 100)}%` }}
              />
            </div>
            <div className="text-sm text-gray-400 mt-2">Position size multiplier</div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">System State</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Risk Level</span>
              <span className={`font-medium ${getModeColor(data.risk_level)}`}>
                {data.risk_level || 'NORMAL'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Freeze Counter</span>
              <span className="text-red-400 font-medium">
                {data.freeze_counter || 0}
              </span>
            </div>
          </div>
        </div>
      </div>

      {(data.risk_adjustments || []).length > 0 && (
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Recent Risk Adjustments</h3>
          <div className="space-y-2 max-h-48 overflow-y-auto">
            {(data.risk_adjustments || []).slice(0, 5).map((adjustment, index) => (
              <div key={index} className="bg-blue-900/20 border border-blue-500/30 rounded p-3">
                <div className="flex justify-between items-start">
                  <span className="text-blue-300">
                    {adjustment.reason || `Risk scale adjusted to ${adjustment.new_scale}`}
                  </span>
                  <span className="text-xs text-gray-400">
                    {adjustment.timestamp ? new Date(adjustment.timestamp).toLocaleTimeString() : 'Recent'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

const RiskAlertsPanel = React.memo(function RiskAlertsPanel({ data }) {
  const allAlerts = [
    ...(data.anomaly_alerts || []).map(alert => ({ ...alert, source: 'Anomaly' })),
    ...(data.compliance_alerts || []).map(alert => ({ ...alert, source: 'Compliance' })),
    ...(data.drawdown_alerts || []).map(alert => ({ ...alert, source: 'Drawdown' })),
    ...(data.execution_alerts || []).map(alert => ({ ...alert, source: 'Execution' })),
    ...(data.portfolio_alerts || []).map(alert => ({ ...alert, source: 'Portfolio' })),
    ...(data.risk_alerts || []).map(alert => ({ ...alert, source: 'Risk' })),
    ...(data.system_alerts || []).map(alert => ({ ...alert, source: 'System' }))
  ].sort((a, b) => new Date(b.timestamp || 0).getTime() - new Date(a.timestamp || 0).getTime());

  const criticalCount = allAlerts.filter(alert => alert.severity === 'critical').length;
  const warningCount = allAlerts.filter(alert => alert.severity === 'warning').length;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-4">
          <div className="text-center">
            <Bell className="w-8 h-8 text-gray-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-white">{allAlerts.length}</div>
            <div className="text-sm text-gray-400">Total Alerts</div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-4">
          <div className="text-center">
            <AlertTriangle className="w-8 h-8 text-red-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-red-400">{criticalCount}</div>
            <div className="text-sm text-gray-400">Critical</div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-4">
          <div className="text-center">
            <AlertCircle className="w-8 h-8 text-yellow-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-yellow-400">{warningCount}</div>
            <div className="text-sm text-gray-400">Warning</div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-4">
          <div className="text-center">
            <CheckCircle className="w-8 h-8 text-blue-400 mx-auto mb-2" />
            <div className="text-2xl font-bold text-blue-400">{allAlerts.length - criticalCount - warningCount}</div>
            <div className="text-sm text-gray-400">Info</div>
          </div>
        </div>
      </div>

      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Risk Alerts</h3>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {allAlerts.slice(0, 20).map((alert, index) => (
            <div
              key={index}
              className={`border rounded p-4 ${
                alert.severity === 'critical' ? 'bg-red-900/20 border-red-500/30' :
                alert.severity === 'warning' ? 'bg-yellow-900/20 border-yellow-500/30' :
                'bg-blue-900/20 border-blue-500/30'
              }`}
            >
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center space-x-2">
                  <span className={`px-2 py-1 text-xs rounded ${
                    alert.severity === 'critical' ? 'bg-red-500/20 text-red-400' :
                    alert.severity === 'warning' ? 'bg-yellow-500/20 text-yellow-400' :
                    'bg-blue-500/20 text-blue-400'
                  }`}>
                    {alert.source}
                  </span>
                  <span className={`px-2 py-1 text-xs rounded ${
                    alert.severity === 'critical' ? 'bg-red-500/30 text-red-300' :
                    alert.severity === 'warning' ? 'bg-yellow-500/30 text-yellow-300' :
                    'bg-blue-500/30 text-blue-300'
                  }`}>
                    {(alert.severity || 'info').toUpperCase()}
                  </span>
                </div>
                <span className="text-xs text-gray-400">
                  {alert.timestamp ? new Date(alert.timestamp).toLocaleString() : 'Recent'}
                </span>
              </div>
              <div className={`${
                alert.severity === 'critical' ? 'text-red-300' :
                alert.severity === 'warning' ? 'text-yellow-300' :
                'text-blue-300'
              }`}>
                {alert.message || alert.type || 'Risk alert triggered'}
              </div>
            </div>
          ))}

          {allAlerts.length === 0 && (
            <div className="text-center text-gray-400 py-8">
              <CheckCircle className="w-12 h-12 mx-auto mb-4 text-green-400" />
              <div className="text-lg font-medium">No Active Risk Alerts</div>
              <div className="text-sm">All risk systems are operating normally</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

// Main RiskTab Component
const RiskTab = React.memo(function RiskTab() {
  const [riskData, setRiskData] = useState({
    overview: {},
    anomalies: {},
    compliance: {},
    drawdown: {},
    execution: {},
    portfolio: {},
    dynamic: {},
    alerts: {}
  });
  const [loading, setLoading] = useState(false);
  const [selectedComponent, setSelectedComponent] = useState('overview');
  const [lastUpdate, setLastUpdate] = useState(0);
  const [hasRiskData, setHasRiskData] = useState(false);

  // Use refs to track loading state without triggering re-renders
  const loadingRef = useRef(false);
  const lastUpdateRef = useRef(0);

  const fetchRiskData = useCallback(async () => {
    if (loadingRef.current || Date.now() - lastUpdateRef.current < 5000) return;

    loadingRef.current = true;
    setLoading(true);
    try {
      // Use AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);

      const [overviewRes, anomaliesRes, complianceRes, drawdownRes, executionRes, portfolioRes, dynamicRes, alertsRes] = await Promise.all([
        fetch('/api/risk/overview', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/risk/anomalies', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/risk/compliance', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/risk/drawdown', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/risk/execution', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/risk/portfolio', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/risk/dynamic', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/risk/alerts', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false }))
      ]);

      clearTimeout(timeoutId);

      const overview = overviewRes.success ? overviewRes : { error: overviewRes.error };
      
      // Check if we have meaningful risk data
      const hasMeaningfulData = (
        overview.risk_level && overview.risk_level !== 'UNKNOWN' ||
        (overview.current_drawdown || 0) > 0 ||
        (overview.max_drawdown || 0) > 0
      );
      setHasRiskData(hasMeaningfulData);

      setRiskData({
        overview,
        anomalies: anomaliesRes.success ? anomaliesRes.anomalies : {},
        compliance: complianceRes.success ? complianceRes.compliance : {},
        drawdown: drawdownRes.success ? drawdownRes.drawdown : {},
        execution: executionRes.success ? executionRes.execution : {},
        portfolio: portfolioRes.success ? portfolioRes.portfolio : {},
        dynamic: dynamicRes.success ? dynamicRes.dynamic : {},
        alerts: alertsRes.success ? alertsRes.alerts : {}
      });
      lastUpdateRef.current = Date.now();
      setLastUpdate(Date.now());
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Error fetching risk data:', error);
      }
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  }, []); // Empty deps - uses refs for mutable state

  useEffect(() => {
    fetchRiskData();
    const interval = setInterval(fetchRiskData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, [fetchRiskData]);

  const componentTabs = [
    { key: 'overview', label: 'Overview', icon: Shield },
    { key: 'anomalies', label: 'Anomalies', icon: AlertTriangle },
    { key: 'compliance', label: 'Compliance', icon: CheckCircle },
    { key: 'drawdown', label: 'Drawdown', icon: TrendingDown },
    { key: 'execution', label: 'Execution', icon: Target },
    { key: 'portfolio', label: 'Portfolio', icon: PieChartIcon },
    { key: 'dynamic', label: 'Dynamic', icon: Activity },
    { key: 'alerts', label: 'Alerts', icon: Bell }
  ];

  const getRiskLevelColor = (level) => {
    switch (level?.toUpperCase()) {
      case 'LOW':
      case 'NORMAL': return 'text-green-400';
      case 'ELEVATED':
      case 'WARNING': return 'text-yellow-400';
      case 'HIGH':
      case 'CRITICAL': return 'text-red-400';
      case 'EMERGENCY': return 'text-red-500';
      default: return 'text-gray-400';
    }
  };

  const getRiskLevelBg = (level) => {
    switch (level?.toUpperCase()) {
      case 'LOW':
      case 'NORMAL': return 'bg-green-500/20';
      case 'ELEVATED':
      case 'WARNING': return 'bg-yellow-500/20';
      case 'HIGH':
      case 'CRITICAL': return 'bg-red-500/20';
      case 'EMERGENCY': return 'bg-red-500/30';
      default: return 'bg-gray-500/20';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Risk Management</h1>
          <p className="text-gray-400">Monitor risk systems and alerts</p>
        </div>
        <button
          onClick={fetchRiskData}
          disabled={loading}
          className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 px-4 py-2 rounded-lg transition-colors text-white font-medium"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* No Data Banner */}
      {!hasRiskData && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <div>
              <h3 className="text-yellow-400 font-medium">No Risk Data Available</h3>
              <p className="text-gray-400 text-sm mt-1">
                Risk metrics will populate once live trading starts. Start trading to see real-time risk management data.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Risk Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className={`rounded-lg p-6 border backdrop-blur-sm ${getRiskLevelBg(riskData.overview.risk_level)} border-gray-700`}>
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-red-500/20 rounded-lg">
              <Shield className="w-5 h-5 text-red-400" />
            </div>
            <div>
              <h3 className="text-gray-400 text-sm font-medium">Risk Level</h3>
              <div className={`text-2xl font-bold ${getRiskLevelColor(riskData.overview.risk_level)}`}>
                {riskData.overview.risk_level || 'UNKNOWN'}
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500">System risk assessment</div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <TrendingDown className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h3 className="text-gray-400 text-sm font-medium">Current Drawdown</h3>
              <div className="text-2xl font-bold text-white">
                {((riskData.overview.current_drawdown || 0) * 100).toFixed(2)}%
              </div>
            </div>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div
              className="bg-red-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${Math.min((riskData.overview.current_drawdown || 0) * 100 * 4, 100)}%` }}
            />
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Activity className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-gray-400 text-sm font-medium">Risk Scale</h3>
              <div className="text-2xl font-bold text-white">
                {(riskData.overview.risk_scale || 1.0).toFixed(2)}x
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500">Position sizing multiplier</div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-green-500/20 rounded-lg">
              <CheckCircle className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h3 className="text-gray-400 text-sm font-medium">Win Rate</h3>
              <div className="text-2xl font-bold text-white">
                {((riskData.overview.win_rate || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div
              className="bg-green-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(riskData.overview.win_rate || 0) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Component Navigation */}
      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-4">
        <div className="flex flex-wrap gap-2">
          {componentTabs.map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => setSelectedComponent(key)}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
                selectedComponent === key
                  ? 'bg-gradient-to-r from-red-600 to-red-700 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Icon size={16} />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Dynamic Content Based on Selected Component */}
      <div className="animate-in fade-in duration-500">
        {selectedComponent === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <RiskOverviewChart data={riskData.overview} />
            <RiskMetricsPanel data={riskData.overview} />
          </div>
        )}

        {selectedComponent === 'anomalies' && (
          <AnomaliesPanel data={riskData.anomalies} />
        )}

        {selectedComponent === 'compliance' && (
          <CompliancePanel data={riskData.compliance} />
        )}

        {selectedComponent === 'drawdown' && (
          <DrawdownPanel data={riskData.drawdown} />
        )}

        {selectedComponent === 'execution' && (
          <ExecutionPanel data={riskData.execution} />
        )}

        {selectedComponent === 'portfolio' && (
          <PortfolioRiskPanel data={riskData.portfolio} />
        )}

        {selectedComponent === 'dynamic' && (
          <DynamicRiskPanel data={riskData.dynamic} />
        )}

        {selectedComponent === 'alerts' && (
          <RiskAlertsPanel data={riskData.alerts} />
        )}
      </div>
    </div>
  );
});

export default RiskTab;
