import React, { useEffect, useRef } from 'react';
import {
  TrendingUp, Target, Heart, Crown, Waves, Database, Monitor, Wifi, Bell, 
  AlertCircle, Loader2, Cpu
} from 'lucide-react';
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar,
  PieChart as RechartsPieChart, Pie, Cell, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ComposedChart
} from 'recharts';
import { useAppState, toArray } from '../shared';

const AnalyticsTab = React.memo(function AnalyticsTab() {
  const { state: appState, dispatch: appDispatch } = useAppState();
  const { analyticsData, analyticsLoading, selectedTimeframe, selectedView } = appState;
  const hasFetchedOnce = useRef(false);

  useEffect(() => {
    const fetchAnalyticsData = async () => {
      if (Date.now() - appState.lastUpdate.analytics < 10000) return;
      try {
        // Only show loading on first fetch
        if (!hasFetchedOnce.current) {
          appDispatch({ type: 'SET_ANALYTICS_LOADING', payload: true });
        }

        const [
          visualizationResponse,
          dashboardResponse,
          performanceResponse,
          modulesResponse
        ] = await Promise.all([
          fetch('/api/visualization-data'),
          fetch('/api/dashboard-data'),
          fetch('/api/performance-metrics'),
          fetch('/api/modules')
        ]);

        const visualizationData = visualizationResponse.ok ? await visualizationResponse.json() : {};
        const dashboardData = dashboardResponse.ok ? await dashboardResponse.json() : {};
        const performanceData = performanceResponse.ok ? await performanceResponse.json() : {};
        const modulesData = modulesResponse.ok ? await modulesResponse.json() : {};

        appDispatch({
          type: 'SET_ANALYTICS_DATA',
          payload: {
            visualization: visualizationData,
            dashboard: dashboardData,
            performance: performanceData,
            modules: modulesData
          }
        });
        hasFetchedOnce.current = true;
      } catch (error) {
        console.error('Failed to fetch analytics data:', error);
        appDispatch({ type: 'SET_ANALYTICS_LOADING', payload: false });
      }
    };

    if (!appState.dataLoaded.analytics || selectedTimeframe) {
      fetchAnalyticsData();
    }
  }, [selectedTimeframe, appState.dataLoaded.analytics, appState.lastUpdate.analytics, appDispatch]);

  const renderPerformanceMetrics = () => {
    const performance = analyticsData.performance || {};
    const visualization = analyticsData.visualization || {};

    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {/* Balance & PnL Trends */}
        <div className="lg:col-span-2 bg-gray-800 rounded-xl p-6 border border-gray-700">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-white flex items-center">
              <TrendingUp size={20} className="mr-2 text-green-400" />
              Balance & P&L Trends
            </h3>
            <div className="flex space-x-2">
              {['1h', '4h', '1d', '1w'].map(tf => (
                <button
                  key={tf}
                  onClick={() => appDispatch({ type: 'SET_TIMEFRAME', payload: tf })}
                  className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                    selectedTimeframe === tf
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }`}
                >
                  {tf}
                </button>
              ))}
            </div>
          </div>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart
                data={
                  analyticsData.visualization?.performance_metrics?.balance_history?.map((balance, i) => ({
                    time: i,
                    balance: balance,
                    pnl: analyticsData.visualization?.performance_metrics?.pnl_history?.[i] || 0,
                    drawdown: analyticsData.visualization?.performance_metrics?.drawdown_history?.[i] || 0
                  })) || []
                }
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="time" stroke="#9CA3AF" />
                <YAxis yAxisId="left" stroke="#9CA3AF" />
                <YAxis yAxisId="right" orientation="right" stroke="#9CA3AF" />
                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px', color: '#F9FAFB' }} />
                <Area yAxisId="left" type="monotone" dataKey="balance" stackId="1" stroke="#10B981" fill="#10B981" fillOpacity={0.3} isAnimationActive={false} />
                <Bar yAxisId="right" dataKey="pnl" fill="#3B82F6" isAnimationActive={false} />
                <Line yAxisId="right" type="monotone" dataKey="drawdown" stroke="#EF4444" strokeWidth={2} dot={false} isAnimationActive={false} />
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Trading Stats */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center">
            <Target size={18} className="mr-2 text-blue-400" />
            Trading Stats
          </h3>
          <div className="space-y-4">
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Win Rate</span>
              <span className="text-green-400 font-bold">
                {((analyticsData.visualization?.performance_metrics?.win_rate?.slice(-1)[0] || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Active Positions</span>
              <span className="text-blue-400 font-bold">
                {analyticsData.visualization?.performance_metrics?.position_count?.slice(-1)[0] || 0}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Total Trades</span>
              <span className="text-purple-400 font-bold">
                {analyticsData.visualization?.performance_metrics?.trades?.slice(-1)[0] || 0}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Risk Score</span>
              <span className={`font-bold ${
                (analyticsData.visualization?.performance_metrics?.risk_score?.slice(-1)[0] || 0) > 0.7 ? 'text-red-400' :
                (analyticsData.visualization?.performance_metrics?.risk_score?.slice(-1)[0] || 0) > 0.4 ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {((analyticsData.visualization?.performance_metrics?.risk_score?.slice(-1)[0] || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-400">Consensus</span>
              <div className="flex items-center">
                <div className="w-16 bg-gray-700 rounded-full h-2 mr-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${(analyticsData.visualization?.performance_metrics?.consensus?.slice(-1)[0] || 0) * 100}%` }}
                  />
                </div>
                <span className="text-cyan-400 font-bold text-sm">
                  {((analyticsData.visualization?.performance_metrics?.consensus?.slice(-1)[0] || 0) * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderModuleAnalytics = () => {
    const modules = toArray(analyticsData.modules?.modules);
    const modulesByCategory = modules.reduce((acc, module) => {
      if (!acc[module.category]) acc[module.category] = [];
      acc[module.category].push(module);
      return acc;
    }, {});

    return (
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Module Health */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Heart size={20} className="mr-2 text-red-400" />
            Module Health Overview
          </h3>

          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <RechartsPieChart>
                <Pie
                  data={Object.entries(modulesByCategory).map(([category, categoryModules]) => ({
                    name: category,
                    value: categoryModules.length,
                    health: categoryModules.reduce((acc, m) => acc + (m.health_score || 0), 0) / categoryModules.length
                  }))}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, value }) => `${name}: ${value}`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                  isAnimationActive={false}
                >
                  {Object.keys(modulesByCategory).map((category, index) => (
                    <Cell key={`cell-${index}`} fill={`hsl(${index * 45}, 70%, 60%)`} />
                  ))}
                </Pie>
                <Tooltip contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px', color: '#F9FAFB' }} />
              </RechartsPieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Top Modules */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Crown size={20} className="mr-2 text-yellow-400" />
            Top Performing Modules
          </h3>

          <div className="space-y-3">
            {modules
              .filter(m => m.health_score > 0)
              .sort((a, b) => (b.health_score || 0) - (a.health_score || 0))
              .slice(0, 8)
              .map((module, index) => (
                <div key={module.id ?? module.name} className="flex items-center justify-between p-3 bg-gray-700/50 rounded-lg">
                  <div className="flex items-center space-x-3">
                    <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold ${
                      index === 0 ? 'bg-yellow-500 text-black' :
                      index === 1 ? 'bg-gray-400 text-black' :
                      index === 2 ? 'bg-amber-600 text-black' : 'bg-gray-600 text-white'
                    }`}>
                      {index + 1}
                    </div>
                    <div>
                      <div className="text-white font-medium">{module.name}</div>
                      <div className="text-xs text-gray-400 capitalize">{module.category}</div>
                    </div>
                  </div>
                  <div className="flex items-center space-x-2">
                    <div className="w-16 bg-gray-600 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full transition-all duration-500 ${
                          module.health_score >= 90 ? 'bg-green-500' :
                          module.health_score >= 75 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${module.health_score}%` }}
                      />
                    </div>
                    <span className="text-sm font-bold text-white">{module.health_score}%</span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>
    );
  };

  const renderSystemFlow = () => {
    const visualization = analyticsData.visualization || {};

    return (
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-xl font-bold text-white mb-4 flex items-center">
            <Waves size={20} className="mr-2 text-cyan-400" />
            System Data Flow
          </h3>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[
              { label: 'Data Points', value: visualization.statistics?.data_points_collected || 0, icon: Database, color: 'text-blue-400' },
              { label: 'Dashboard Updates', value: visualization.statistics?.dashboard_updates || 0, icon: Monitor, color: 'text-green-400' },
              { label: 'Stream Updates', value: visualization.statistics?.stream_updates || 0, icon: Wifi, color: 'text-purple-400' },
              { label: 'Active Alerts', value: visualization.recent_alerts?.length || 0, icon: Bell, color: 'text-red-400' }
            ].map((stat, index) => (
              <div key={index} className="bg-gray-700/50 rounded-lg p-4 text-center">
                <stat.icon size={24} className={`mx-auto mb-2 ${stat.color}`} />
                <div className="text-2xl font-bold text-white">{stat.value.toLocaleString()}</div>
                <div className="text-xs text-gray-400">{stat.label}</div>
              </div>
            ))}
          </div>

          <div className="mt-6">
            <h4 className="text-lg font-semibold text-white mb-3">Market Regime Analytics</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {Object.entries(visualization.regime_analytics || {}).map(([regime, data]) => (
                <div key={regime} className="bg-gray-700/30 rounded-lg p-3">
                  <div className="text-sm font-medium text-white capitalize">{regime}</div>
                  <div className="text-xs text-gray-400 mt-1">Time: {data.time_spent || 0}min</div>
                  <div className="text-xs text-gray-400">Trades: {data.trade_count || 0}</div>
                  <div className={`text-sm font-bold mt-1 ${
                    (data.avg_pnl || 0) > 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    P&L: {(data.avg_pnl || 0) > 0 ? '+' : ''}${(data.avg_pnl || 0).toFixed(2)}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-bold text-white mb-4 flex items-center">
            <AlertCircle size={18} className="mr-2 text-orange-400" />
            Recent Activity
          </h3>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {(visualization.recent_alerts || []).slice(0, 20).map((alert, index) => (
              <div key={index} className="flex items-start space-x-3 p-2 bg-gray-700/30 rounded-lg">
                <div className={`w-2 h-2 rounded-full mt-2 flex-shrink-0 ${
                  alert.severity === 'critical' ? 'bg-red-500' :
                  alert.severity === 'warning' ? 'bg-yellow-500' : 'bg-blue-500'
                }`} />
                <div className="flex-1 min-w-0">
                  <div className="text-sm text-white">
                    {alert.alert?.message || JSON.stringify(alert.alert).slice(0, 50)}
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {alert.module} • {new Date(alert.time).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    );
  };

  const viewComponents = {
    performance: renderPerformanceMetrics,
    modules: renderModuleAnalytics,
    flow: renderSystemFlow
  };

  // Only show loading on first load, not on refreshes
  if (analyticsLoading && !hasFetchedOnce.current) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-4" />
          <div className="text-gray-400">Loading analytics data...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-3xl font-bold text-white">Analytics Dashboard</h2>
          <p className="text-gray-400">Comprehensive system performance and insights</p>
        </div>

        <div className="flex space-x-2">
          {[
            { key: 'performance', label: 'Performance', icon: TrendingUp },
            { key: 'modules', label: 'Modules', icon: Cpu },
            { key: 'flow', label: 'System Flow', icon: Waves }
          ].map(({ key, label, icon: Icon }) => (
            <button
              key={key}
              onClick={() => appDispatch({ type: 'SET_VIEW', payload: key })}
              className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium transition-all ${
                selectedView === key
                  ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Icon size={16} />
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="animate-in fade-in duration-500">
        {viewComponents[selectedView]()}
      </div>
    </div>
  );
});

export default AnalyticsTab;
