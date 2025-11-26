import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  RefreshCw, BarChart3, Brain, Target, AlertTriangle, Sparkles,
  Activity, AlertCircle, HardDrive, Cpu, Heart, Eye, Shield
} from 'lucide-react';
import {
  ResponsiveContainer,
  PieChart as RechartsPieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid
} from 'recharts';

// Memory Overview Chart
const MemoryOverviewChart = React.memo(function MemoryOverviewChart({ data }) {
  const chartData = [
    { name: 'Usage', value: (data.memory_utilization || 0) * 100, color: '#10b981' },
    { name: 'Free', value: 100 - ((data.memory_utilization || 0) * 100), color: '#374151' }
  ];

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Memory Usage Overview</h3>
      <div className="flex items-center justify-center h-64">
        <ResponsiveContainer width="100%" height="100%">
          <RechartsPieChart>
            <Pie
              data={chartData}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={100}
              fill="#8884d8"
              dataKey="value"
            >
              {chartData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip formatter={(value) => `${value.toFixed(1)}%`} />
          </RechartsPieChart>
        </ResponsiveContainer>
      </div>
      <div className="flex justify-center space-x-4 mt-4">
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
          <span className="text-sm text-gray-400">Used</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 bg-gray-600 rounded-full"></div>
          <span className="text-sm text-gray-400">Free</span>
        </div>
      </div>
    </div>
  );
});

const MemoryHealthPanel = React.memo(function MemoryHealthPanel({ data }) {
  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusBg = (status) => {
    switch (status) {
      case 'healthy': return 'bg-green-500/20';
      case 'warning': return 'bg-yellow-500/20';
      case 'critical': return 'bg-red-500/20';
      default: return 'bg-gray-500/20';
    }
  };

  return (
    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">System Health</h3>
      <div className="space-y-4">
        <div className={`p-4 rounded-lg ${getStatusBg(data.health_status)}`}>
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Overall Status</span>
            <span className={`font-bold capitalize ${getStatusColor(data.health_status)}`}>
              {data.health_status || 'Unknown'}
            </span>
          </div>
        </div>

        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Processing Status</span>
            <span className="text-blue-400 font-medium">
              {data.processing_status || 'Unknown'}
            </span>
          </div>
        </div>

        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">System Status</span>
            <span className="text-purple-400 font-medium">
              {data.status || 'Unknown'}
            </span>
          </div>
        </div>

        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <span className="text-gray-300">Components Enabled</span>
            <span className="text-green-400 font-bold">
              {data.components_enabled || 0}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

const NeuralMemoryPanel = React.memo(function NeuralMemoryPanel({ data }) {
  const neural = data.neural_memory || {};
  const attention = data.attention_retrieval || {};
  const embedding = data.memory_embedding || {};
  const scoring = data.importance_scoring || {};

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Brain className="w-5 h-5 mr-2 text-purple-400" />
          Neural Memory Status
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-400">Buffer Size</span>
            <span className="text-white font-medium">{neural.buffer_size || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Memory Utilization</span>
            <span className="text-blue-400 font-medium">
              {((neural.memory_utilization || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Total Embeddings</span>
            <span className="text-purple-400 font-medium">{embedding.total_embeddings || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Embedding Dimension</span>
            <span className="text-green-400 font-medium">{embedding.embedding_dim || 0}</span>
          </div>
        </div>
      </div>

      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Eye className="w-5 h-5 mr-2 text-blue-400" />
          Attention & Scoring
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-400">Retrieved Count</span>
            <span className="text-blue-400 font-medium">{attention.retrieved_count || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Avg Importance</span>
            <span className="text-yellow-400 font-medium">
              {(scoring.average_importance || 0).toFixed(3)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Total Scored</span>
            <span className="text-green-400 font-medium">{scoring.total_scored || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Similarity Scores</span>
            <span className="text-purple-400 font-medium">
              {(attention.similarity_scores || []).length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

const PlaybookMemoryPanel = React.memo(function PlaybookMemoryPanel({ data }) {
  const recall = data.playbook_recall || {};
  const patterns = data.pattern_memory || {};
  const quality = data.playbook_quality || {};
  const analytics = data.memory_analytics || {};

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2 text-cyan-400" />
          Playbook Status
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-400">Memory Entries</span>
            <span className="text-cyan-400 font-medium">{recall.memory_entries || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Patterns Identified</span>
            <span className="text-blue-400 font-medium">{recall.patterns_identified || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Total Patterns</span>
            <span className="text-purple-400 font-medium">{patterns.total_patterns || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Quality Score</span>
            <span className="text-green-400 font-medium">
              {(quality.quality_score || 0).toFixed(2)}
            </span>
          </div>
        </div>
      </div>

      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2 text-green-400" />
          Analytics
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-400">Total Recalls</span>
            <span className="text-green-400 font-medium">{analytics.total_recalls || 0}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Memory Health</span>
            <span className="text-yellow-400 font-medium capitalize">
              {analytics.memory_health || 'Unknown'}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Memory Utilization</span>
            <span className="text-blue-400 font-medium">
              {((quality.memory_utilization || 0) * 100).toFixed(1)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Pattern Effectiveness</span>
            <span className="text-purple-400 font-medium">
              {Object.keys(patterns.pattern_effectiveness || {}).length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

const MistakeMemoryPanel = React.memo(function MistakeMemoryPanel({ data }) {
  const mistakes = data.mistake_memory || {};
  const avoidance = data.mistake_avoidance || {};
  const dangers = data.danger_zones || {};
  const prevention = data.loss_prevention || {};
  const recognition = data.pattern_recognition || {};

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <AlertTriangle className="w-5 h-5 mr-2 text-red-400" />
            Mistake Tracking
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Recent Count</span>
              <span className="text-red-400 font-medium">
                {(mistakes.recent || []).length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Mistakes</span>
              <span className="text-orange-400 font-medium">
                {mistakes.stats?.count || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avoidance Signal</span>
              <span className="text-yellow-400 font-medium">
                {(avoidance.avoidance_signal || 0).toFixed(3)}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Shield className="w-5 h-5 mr-2 text-blue-400" />
            Loss Prevention
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Effectiveness</span>
              <span className="text-blue-400 font-medium">
                {((prevention.avoidance_effectiveness || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Learning Samples</span>
              <span className="text-green-400 font-medium">
                {prevention.learning_samples || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Consecutive Losses</span>
              <span className="text-red-400 font-medium">
                {avoidance.consecutive_losses || 0}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
            <Sparkles className="w-5 h-5 mr-2 text-purple-400" />
            Pattern Recognition
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Loss Patterns</span>
              <span className="text-red-400 font-medium">
                {Object.keys(recognition.loss_patterns || {}).length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Win Patterns</span>
              <span className="text-green-400 font-medium">
                {Object.keys(recognition.win_patterns || {}).length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Danger Zones</span>
              <span className="text-yellow-400 font-medium">
                {(dangers.zones || []).length}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

const PatternsPanel = React.memo(function PatternsPanel({ data }) {
  const neural = data.neural_patterns || {};
  const playbook = data.playbook_patterns || {};
  const compressed = data.compressed_patterns || {};
  const recognition = data.pattern_recognition || {};

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Brain className="w-5 h-5 mr-2 text-purple-400" />
          Neural Patterns
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-400">Attention Retrieval</span>
            <span className="text-purple-400 font-medium">
              {neural.attention_retrieval?.retrieved_count || 0}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Memory Embeddings</span>
            <span className="text-blue-400 font-medium">
              {neural.memory_embedding?.total_embeddings || 0}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Compressed Profit</span>
            <span className="text-green-400 font-medium">
              {(compressed.profit_direction || []).length}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Compressed Loss</span>
            <span className="text-red-400 font-medium">
              {(compressed.loss_direction || []).length}
            </span>
          </div>
        </div>
      </div>

      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2 text-cyan-400" />
          Playbook Patterns
        </h3>
        <div className="space-y-3">
          <div className="flex justify-between">
            <span className="text-gray-400">Pattern Memory</span>
            <span className="text-cyan-400 font-medium">
              {playbook.pattern_memory?.total_patterns || 0}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Pattern Analysis</span>
            <span className="text-blue-400 font-medium">
              {playbook.pattern_analysis?.total_patterns || 0}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Win Patterns</span>
            <span className="text-green-400 font-medium">
              {Object.keys(recognition.win_patterns || {}).length}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Loss Patterns</span>
            <span className="text-red-400 font-medium">
              {Object.keys(recognition.loss_patterns || {}).length}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
});

const MemoryPerformancePanel = React.memo(function MemoryPerformancePanel({ data }) {
  const overview = data.overview || {};
  const neural = data.neural_performance || {};
  const playbook = data.playbook_performance || {};
  const compression = data.compression_performance || {};
  const budget = data.budget_performance || {};

  const performanceData = [
    { name: 'Neural', score: (neural.importance_scoring?.average_importance || 0) * 100 },
    { name: 'Playbook', score: (playbook.playbook_quality?.quality_score || 0) * 100 },
    { name: 'Compression', score: (compression.memory_compression?.compression_efficiency || 0) * 100 },
    { name: 'Budget', score: (budget.budget_optimization?.optimality_score || 0) * 100 }
  ];

  return (
    <div className="space-y-6">
      <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Component Performance</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="name" stroke="#9CA3AF" />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1F2937',
                  border: '1px solid #374151',
                  borderRadius: '0.5rem'
                }}
              />
              <Bar dataKey="score" fill="#8B5CF6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Budget Optimization</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Optimality Score</span>
              <span className="text-green-400 font-medium">
                {(budget.budget_optimization?.optimality_score || 0).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Profit</span>
              <span className="text-blue-400 font-medium">
                ${(budget.budget_optimization?.total_profit || 0).toFixed(2)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Optimization Count</span>
              <span className="text-purple-400 font-medium">
                {budget.budget_optimization?.optimization_count || 0}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Memory Compression</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Memories</span>
              <span className="text-cyan-400 font-medium">
                {compression.memory_compression?.total_memories || 0}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Compression Efficiency</span>
              <span className="text-green-400 font-medium">
                {((compression.memory_compression?.compression_efficiency || 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Feature Components</span>
              <span className="text-yellow-400 font-medium">
                {(compression.feature_importance?.profit_components || []).length}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

// Main MemoryTab Component
const MemoryTab = React.memo(function MemoryTab() {
  const [memoryData, setMemoryData] = useState({
    overview: {},
    components: {},
    patterns: {},
    mistakes: {},
    performance: {}
  });
  const [loading, setLoading] = useState(false);
  const [selectedComponent, setSelectedComponent] = useState('overview');
  const [lastUpdate, setLastUpdate] = useState(0);
  const [hasTrainingData, setHasTrainingData] = useState(false);

  // Use refs to track loading state without triggering re-renders
  const loadingRef = useRef(false);
  const lastUpdateRef = useRef(0);

  const fetchMemoryData = useCallback(async () => {
    if (loadingRef.current || Date.now() - lastUpdateRef.current < 5000) return;

    loadingRef.current = true;
    setLoading(true);
    try {
      // Use AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);

      const [overviewRes, componentsRes, patternsRes, mistakesRes, performanceRes] = await Promise.all([
        fetch('/api/memory/overview', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/memory/components', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/memory/patterns', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/memory/mistakes', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/memory/performance', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false }))
      ]);

      clearTimeout(timeoutId);

      const overview = overviewRes.success ? overviewRes : { error: overviewRes.error };
      const components = componentsRes.success ? componentsRes.components : {};
      
      // Check if we have any meaningful data (training has run)
      const hasMeaningfulData = (
        (overview.total_memories || 0) > 0 ||
        (overview.components_active || 0) > 0 ||
        overview.health_status !== 'unknown'
      );
      setHasTrainingData(hasMeaningfulData);

      setMemoryData({
        overview,
        components,
        patterns: patternsRes.success ? patternsRes.patterns : {},
        mistakes: mistakesRes.success ? mistakesRes.mistakes : {},
        performance: performanceRes.success ? performanceRes.performance : {}
      });
      lastUpdateRef.current = Date.now();
      setLastUpdate(Date.now());
    } catch (error) {
      if (error.name !== 'AbortError') {
        console.error('Error fetching memory data:', error);
      }
    } finally {
      loadingRef.current = false;
      setLoading(false);
    }
  }, []); // Empty deps - uses refs for mutable state

  useEffect(() => {
    fetchMemoryData();
    const interval = setInterval(fetchMemoryData, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, [fetchMemoryData]);

  const componentTabs = [
    { key: 'overview', label: 'Overview', icon: BarChart3 },
    { key: 'neural', label: 'Neural', icon: Brain },
    { key: 'playbook', label: 'Playbook', icon: Target },
    { key: 'mistakes', label: 'Mistakes', icon: AlertTriangle },
    { key: 'patterns', label: 'Patterns', icon: Sparkles },
    { key: 'performance', label: 'Performance', icon: Activity }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-white mb-2">Memory System</h1>
          <p className="text-gray-400">Monitor unified memory components and performance</p>
        </div>
        <button
          onClick={fetchMemoryData}
          disabled={loading}
          className="flex items-center space-x-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 px-4 py-2 rounded-lg transition-colors text-white font-medium"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
      </div>

      {/* No Data Banner */}
      {!hasTrainingData && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <div>
              <h3 className="text-yellow-400 font-medium">No Memory Data Available</h3>
              <p className="text-gray-400 text-sm mt-1">
                Memory data will populate once live trading starts. Start trading to see real-time memory system metrics.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Memory Overview Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <HardDrive className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h3 className="text-gray-400 text-sm font-medium">Total Memories</h3>
              <div className="text-2xl font-bold text-white">
                {memoryData.overview.total_memories || 0}
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500">Stored experiences</div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-green-500/20 rounded-lg">
              <Activity className="w-5 h-5 text-green-400" />
            </div>
            <div>
              <h3 className="text-gray-400 text-sm font-medium">Memory Usage</h3>
              <div className="text-2xl font-bold text-white">
                {((memoryData.overview.memory_utilization || 0) * 100).toFixed(1)}%
              </div>
            </div>
          </div>
          <div className="w-full bg-gray-700 rounded-full h-2 mt-2">
            <div
              className="bg-green-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${(memoryData.overview.memory_utilization || 0) * 100}%` }}
            />
          </div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Cpu className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <h3 className="text-gray-400 text-sm font-medium">Active Components</h3>
              <div className="text-2xl font-bold text-white">
                {memoryData.overview.components_active || 0}
              </div>
            </div>
          </div>
          <div className="text-xs text-gray-500">Running modules</div>
        </div>

        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-lg p-6">
          <div className="flex items-center space-x-3 mb-4">
            <div className={`p-2 rounded-lg ${
              memoryData.overview.health_status === 'healthy' ? 'bg-green-500/20' :
              memoryData.overview.health_status === 'warning' ? 'bg-yellow-500/20' :
              'bg-red-500/20'
            }`}>
              <Heart className={`w-5 h-5 ${
                memoryData.overview.health_status === 'healthy' ? 'text-green-400' :
                memoryData.overview.health_status === 'warning' ? 'text-yellow-400' :
                'text-red-400'
              }`} />
            </div>
            <div>
              <h3 className="text-gray-400 text-sm font-medium">Health Status</h3>
              <div className={`text-2xl font-bold capitalize ${
                memoryData.overview.health_status === 'healthy' ? 'text-green-400' :
                memoryData.overview.health_status === 'warning' ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {memoryData.overview.health_status || 'Unknown'}
              </div>
            </div>
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
                  ? 'bg-gradient-to-r from-indigo-600 to-purple-600 text-white'
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
            <MemoryOverviewChart data={memoryData.overview} />
            <MemoryHealthPanel data={memoryData.overview} />
          </div>
        )}

        {selectedComponent === 'neural' && (
          <NeuralMemoryPanel data={memoryData.components.neural || {}} />
        )}

        {selectedComponent === 'playbook' && (
          <PlaybookMemoryPanel data={memoryData.components.playbook || {}} />
        )}

        {selectedComponent === 'mistakes' && (
          <MistakeMemoryPanel data={memoryData.mistakes} />
        )}

        {selectedComponent === 'patterns' && (
          <PatternsPanel data={memoryData.patterns} />
        )}

        {selectedComponent === 'performance' && (
          <MemoryPerformancePanel data={memoryData.performance} />
        )}
      </div>
    </div>
  );
});

export default MemoryTab;
