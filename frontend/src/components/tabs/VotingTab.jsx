import React, { useState, useCallback, useEffect, useRef } from 'react';
import { AlertCircle } from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════
// VOTING SYSTEM COMPONENTS v1.0
// Full voting dashboard with 8 sub-components:
// Overview, Committee, Consensus, Collusion, Alignment, Sampling, Strategy, Timeline
// ═══════════════════════════════════════════════════════════════════

const VotingTab = React.memo(function VotingTab() {
  const [votingData, setVotingData] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeComponent, setActiveComponent] = useState('overview');
  const [hasVotingData, setHasVotingData] = useState(false);
  const isFirstLoad = useRef(true);
  const hasFetchedOnce = useRef(false);

  const fetchVotingData = useCallback(async () => {
    // Only show loading spinner on first load
    if (isFirstLoad.current) {
      setIsLoading(true);
    }
    try {
      const endpoints = [
        '/api/voting/overview',
        '/api/voting/committee',
        '/api/voting/consensus',
        '/api/voting/collusion',
        '/api/voting/alignment',
        '/api/voting/sampling',
        '/api/voting/strategy',
        '/api/voting/timeline'
      ];

      // Use AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);

      const responses = await Promise.all(
        endpoints.map(endpoint => 
          fetch(endpoint, { signal: controller.signal })
            .then(res => res.json())
            .catch(() => ({ success: false }))
        )
      );

      clearTimeout(timeoutId);

      const data = {
        overview: responses[0].success ? responses[0] : {},
        committee: responses[1].success ? responses[1] : {},
        consensus: responses[2].success ? responses[2] : {},
        collusion: responses[3].success ? responses[3] : {},
        alignment: responses[4].success ? responses[4] : {},
        sampling: responses[5].success ? responses[5] : {},
        strategy: responses[6].success ? responses[6] : {},
        timeline: responses[7].success ? responses[7] : {}
      };

      // Check if we have meaningful voting data
      const overview = data.overview || {};
      const hasMeaningfulData = (
        overview.total_votes > 0 ||
        overview.decisions_made > 0 ||
        (overview.committee_members || []).length > 0
      );
      setHasVotingData(hasMeaningfulData);

      setVotingData(data);
      setError(null);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(`Failed to fetch voting data: ${err.message}`);
      }
    } finally {
      setIsLoading(false);
      isFirstLoad.current = false;
      hasFetchedOnce.current = true;
    }
  }, []);

  useEffect(() => {
    fetchVotingData();
    const interval = setInterval(fetchVotingData, 10000);
    return () => clearInterval(interval);
  }, [fetchVotingData]);

  const componentTabs = [
    { id: 'overview', name: 'Overview', icon: '📊' },
    { id: 'committee', name: 'Committee', icon: '👥' },
    { id: 'consensus', name: 'Consensus', icon: '🤝' },
    { id: 'collusion', name: 'Collusion', icon: '🕵️' },
    { id: 'alignment', name: 'Alignment', icon: '🕐' },
    { id: 'sampling', name: 'Sampling', icon: '🎯' },
    { id: 'strategy', name: 'Strategy', icon: '🏛️' },
    { id: 'timeline', name: 'Timeline', icon: '⏱️' }
  ];

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400 text-center">
          <div className="text-xl mb-2">⚠️</div>
          <div>{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-white flex items-center gap-2">
          <span className="w-6 h-6 text-blue-400">🗳️</span>
          Voting System
        </h2>
        <button
          onClick={() => fetchVotingData()}
          disabled={isLoading}
          className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white text-sm rounded transition-colors flex items-center gap-2"
        >
          <span className={isLoading ? 'animate-spin' : ''}>🔄</span>
          Refresh
        </button>
      </div>

      {/* No Data Banner */}
      {!hasVotingData && !isLoading && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <div>
              <h3 className="text-yellow-400 font-medium">No Voting Data Available</h3>
              <p className="text-gray-400 text-sm mt-1">
                Voting data will populate once live trading starts. Start trading to see real-time voting system metrics.
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="bg-gray-800/50 rounded-lg border border-gray-700">
        <div className="flex flex-wrap gap-1 p-4 border-b border-gray-700 bg-gray-800/30">
          {componentTabs.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveComponent(tab.id)}
              className={`px-3 py-2 text-sm font-medium rounded transition-colors flex items-center gap-2 ${
                activeComponent === tab.id
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
              }`}
            >
              <span className="text-xs">{tab.icon}</span>
              {tab.name}
            </button>
          ))}
        </div>

        <div className="p-6">
          {activeComponent === 'overview' && (
            <VotingOverviewComponent data={votingData.overview} isLoading={isLoading} />
          )}
          {activeComponent === 'committee' && (
            <VotingCommitteeComponent data={votingData.committee} isLoading={isLoading} />
          )}
          {activeComponent === 'consensus' && (
            <VotingConsensusComponent data={votingData.consensus} isLoading={isLoading} />
          )}
          {activeComponent === 'collusion' && (
            <VotingCollusionComponent data={votingData.collusion} isLoading={isLoading} />
          )}
          {activeComponent === 'alignment' && (
            <VotingAlignmentComponent data={votingData.alignment} isLoading={isLoading} />
          )}
          {activeComponent === 'sampling' && (
            <VotingSamplingComponent data={votingData.sampling} isLoading={isLoading} />
          )}
          {activeComponent === 'strategy' && (
            <VotingStrategyComponent data={votingData.strategy} isLoading={isLoading} />
          )}
          {activeComponent === 'timeline' && (
            <VotingTimelineComponent data={votingData.timeline} isLoading={isLoading} />
          )}
        </div>
      </div>
    </div>
  );
});

// Voting Overview Component
const VotingOverviewComponent = React.memo(function VotingOverviewComponent({ data, isLoading }) {
  if (isLoading) {
    return <div className="flex items-center justify-center h-32 text-blue-400">Loading voting overview...</div>;
  }

  // Ensure data is always an object
  const safeData = data || {};

  const getHealthColor = (status) => {
    switch (status) {
      case 'healthy': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getHealthIcon = (status) => {
    switch (status) {
      case 'healthy': return '✅';
      case 'warning': return '⚠️';
      case 'critical': return '🚨';
      default: return '❓';
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Total Decisions</div>
          <div className="text-2xl font-bold text-blue-400">{safeData.total_decisions || 0}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Success Rate</div>
          <div className="text-2xl font-bold text-green-400">{((safeData.success_rate || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Active Components</div>
          <div className="text-2xl font-bold text-purple-400">{safeData.components_active || 0}/6</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Health Status</div>
          <div className={`text-xl font-bold flex items-center gap-2 ${getHealthColor(safeData.health_status)}`}>
            <span>{getHealthIcon(safeData.health_status)}</span>
            {(safeData.health_status || 'unknown').charAt(0).toUpperCase() + (safeData.health_status || 'unknown').slice(1)}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-blue-400">🎯</span>
            Current Consensus
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Consensus Score</span>
              <span className="text-blue-400 font-bold">{(safeData.current_consensus || 0).toFixed(3)}</span>
            </div>
            <div className="w-full bg-gray-600 rounded-full h-2">
              <div
                className="bg-gradient-to-r from-blue-500 to-blue-400 h-2 rounded-full transition-all duration-500"
                style={{ width: `${(safeData.current_consensus || 0) * 100}%` }}
              />
            </div>
            <div className="text-xs text-gray-400">
              Higher scores indicate stronger committee agreement
            </div>
          </div>
        </div>

        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-green-400">⚡</span>
            Performance
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Avg Processing Time</span>
              <span className="text-green-400 font-bold">{(safeData.processing_time_ms || 0).toFixed(1)}ms</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Decision ID</span>
              <span className="text-gray-400 font-mono text-sm truncate max-w-32">
                {safeData.decision_id || 'none'}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Last Update</span>
              <span className="text-gray-400 text-sm">
                {safeData.last_update ? new Date(safeData.last_update).toLocaleTimeString() : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

// Voting Committee Component
const VotingCommitteeComponent = React.memo(function VotingCommitteeComponent({ data, isLoading }) {
  if (isLoading) {
    return <div className="flex items-center justify-center h-32 text-blue-400">Loading committee data...</div>;
  }

  const committee = data.committee || {};
  const summary = committee.summary || {};
  const analytics = committee.analytics || [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Total Members</div>
          <div className="text-2xl font-bold text-blue-400">{summary.total_members || 0}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Active Members</div>
          <div className="text-2xl font-bold text-green-400">{summary.active_members || 0}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Avg Confidence</div>
          <div className="text-2xl font-bold text-purple-400">{((summary.avg_confidence || 0) * 100).toFixed(1)}%</div>
        </div>
      </div>

      <div className="bg-gray-700/30 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <span className="text-blue-400">👥</span>
          Member Analytics
        </h3>
        {analytics.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {analytics.map(member => (
              <div key={member.member_id} className="bg-gray-600/30 rounded p-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-semibold text-white">{member.name}</span>
                  <span className="text-xs bg-blue-600 text-white px-2 py-1 rounded">
                    {member.specialization}
                  </span>
                </div>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-300">Performance</span>
                    <span className="text-green-400">{(member.performance_score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Reliability</span>
                    <span className="text-blue-400">{(member.reliability * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-300">Votes Cast</span>
                    <span className="text-gray-400">{member.votes_cast}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-gray-400 py-8">
            <div className="text-4xl mb-2">👥</div>
            <div className="text-lg">No Committee Members</div>
            <div className="text-sm">Committee data not available</div>
          </div>
        )}
      </div>
    </div>
  );
});

// Voting Consensus Component
const VotingConsensusComponent = React.memo(function VotingConsensusComponent({ data, isLoading }) {
  if (isLoading) {
    return <div className="flex items-center justify-center h-32 text-blue-400">Loading consensus data...</div>;
  }

  const consensus = data.consensus || {};
  const breakdown = consensus.breakdown || {};
  const analytics = consensus.analytics || {};

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Overall Score</div>
          <div className="text-2xl font-bold text-blue-400">{(consensus.score || 0).toFixed(3)}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Agreement Level</div>
          <div className="text-lg font-bold text-green-400 capitalize">{analytics.agreement_level || 'unknown'}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Trend</div>
          <div className="text-lg font-bold text-purple-400 capitalize">{analytics.trend || 'stable'}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-blue-400">📊</span>
            Component Breakdown
          </h3>
          <div className="space-y-3">
            {Object.entries(breakdown).map(([component, value]) => (
              <div key={component}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-gray-300 capitalize">{component}</span>
                  <span className="text-blue-400 font-bold">{(value || 0).toFixed(3)}</span>
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-blue-400 h-2 rounded-full"
                    style={{ width: `${(value || 0) * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-green-400">🔍</span>
            Analytics
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Quality Score</span>
              <span className="text-green-400 font-bold">{(analytics.quality || 0).toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Stability</span>
              <span className="text-blue-400 font-bold">{(analytics.stability || 0).toFixed(3)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Reliability</span>
              <span className="text-purple-400 font-bold">{(analytics.reliability || 0).toFixed(3)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

// Voting Collusion Component
const VotingCollusionComponent = React.memo(function VotingCollusionComponent({ data, isLoading }) {
  if (isLoading) {
    return <div className="flex items-center justify-center h-32 text-blue-400">Loading collusion data...</div>;
  }

  const collusion = data.collusion || {};
  const analysis = collusion.analysis || {};
  const integrity = collusion.member_integrity || [];
  const alerts = collusion.alerts || [];

  const getRiskColor = (level) => {
    switch (level) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Threat Score</div>
          <div className="text-2xl font-bold text-red-400">{(analysis.threat_score || 0).toFixed(3)}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Risk Level</div>
          <div className={`text-lg font-bold capitalize ${getRiskColor(analysis.risk_level)}`}>
            {analysis.risk_level || 'unknown'}
          </div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Suspicious Pairs</div>
          <div className="text-2xl font-bold text-orange-400">{analysis.suspicious_pairs_count || 0}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-blue-400">🔐</span>
            Member Integrity
          </h3>
          {integrity.length > 0 ? (
            <div className="space-y-3">
              {integrity.map(member => (
                <div key={member.member_id} className="bg-gray-600/30 rounded p-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-white">Expert {member.member_id + 1}</span>
                    <span className={`text-sm font-bold ${member.coordination_detected ? 'text-red-400' : 'text-green-400'}`}>
                      {member.coordination_detected ? '⚠️ FLAG' : '✅ CLEAR'}
                    </span>
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Integrity Score</span>
                      <span className="text-green-400">{(member.integrity_score * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Independence</span>
                      <span className="text-blue-400">{(member.independence_level * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-4">No integrity data available</div>
          )}
        </div>

        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-red-400">🚨</span>
            Security Alerts
          </h3>
          {alerts.length > 0 ? (
            <div className="space-y-3">
              {alerts.map((alert, index) => (
                <div key={index} className="bg-red-900/20 border border-red-700/50 rounded p-3">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-red-400 font-semibold uppercase text-xs">
                      {alert.severity}
                    </span>
                    <span className="text-gray-400 text-xs">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                  <div className="text-sm text-gray-300">{alert.message}</div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              <div className="text-4xl mb-2">🔐</div>
              <div className="text-lg">No Security Alerts</div>
              <div className="text-sm">All voting patterns appear normal</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
});

// Voting Alignment Component
const VotingAlignmentComponent = React.memo(function VotingAlignmentComponent({ data, isLoading }) {
  if (isLoading) {
    return <div className="flex items-center justify-center h-32 text-blue-400">Loading alignment data...</div>;
  }

  const alignment = data.alignment || {};
  const analysis = alignment.analysis || {};
  const metrics = alignment.metrics || {};
  const breakdown = alignment.horizon_breakdown || [];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Quality Score</div>
          <div className="text-2xl font-bold text-blue-400">{(analysis.alignment_quality || 0).toFixed(3)}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Temporal Coherence</div>
          <div className="text-2xl font-bold text-green-400">{(analysis.temporal_coherence || 0).toFixed(3)}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Alignment Strength</div>
          <div className="text-2xl font-bold text-purple-400">{(metrics.alignment_strength || 0).toFixed(3)}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Weight Variance</div>
          <div className="text-2xl font-bold text-orange-400">{(metrics.weight_variance || 0).toFixed(3)}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-blue-400">⏰</span>
            Horizon Distribution
          </h3>
          <div className="space-y-3">
            {Object.entries(analysis.horizon_distribution || {}).map(([horizon, weight]) => (
              <div key={horizon}>
                <div className="flex justify-between items-center mb-1">
                  <span className="text-gray-300 capitalize">{horizon.replace('_', ' ')}</span>
                  <span className="text-blue-400 font-bold">{(weight * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-600 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-blue-400 h-2 rounded-full"
                    style={{ width: `${weight * 100}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-green-400">📈</span>
            Horizon Breakdown
          </h3>
          {breakdown.length > 0 ? (
            <div className="space-y-3 max-h-64 overflow-y-auto">
              {breakdown.map(horizon => (
                <div key={horizon.horizon_minutes} className="bg-gray-600/30 rounded p-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-white">{horizon.horizon_minutes}min</span>
                    <span className="text-blue-400 font-bold">{(horizon.weight || 0).toFixed(3)}</span>
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Contribution</span>
                      <span className="text-green-400">{((horizon.contribution || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Stability</span>
                      <span className="text-purple-400">{((horizon.stability || 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-4">No horizon data available</div>
          )}
        </div>
      </div>
    </div>
  );
});

// Voting Sampling Component
const VotingSamplingComponent = React.memo(function VotingSamplingComponent({ data, isLoading }) {
  if (isLoading) {
    return <div className="flex items-center justify-center h-32 text-blue-400">Loading sampling data...</div>;
  }

  const sampling = data.sampling || {};
  const analysis = sampling.analysis || {};
  const metrics = sampling.metrics || {};
  const riskAssessment = sampling.risk_assessment || {};

  const getRiskColor = (level) => {
    switch (level) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getRecommendationColor = (rec) => {
    switch (rec) {
      case 'proceed': return 'text-green-400';
      case 'monitor': return 'text-yellow-400';
      case 'caution': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Uncertainty Level</div>
          <div className="text-2xl font-bold text-red-400">{((analysis.uncertainty_level || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Robustness</div>
          <div className="text-2xl font-bold text-green-400">{((analysis.robustness || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Sample Quality</div>
          <div className="text-2xl font-bold text-blue-400">{((metrics.sample_quality || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Effective Samples</div>
          <div className="text-2xl font-bold text-purple-400">{metrics.effective_samples || 0}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-blue-400">📊</span>
            Sampling Metrics
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Total Samples</span>
              <span className="text-blue-400 font-bold">{metrics.total_samples || 0}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Convergence Rate</span>
              <span className="text-green-400 font-bold">{((metrics.convergence_rate || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Exploration Breadth</span>
              <span className="text-purple-400 font-bold">{((metrics.exploration_breadth || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Sample Diversity</span>
              <span className="text-orange-400 font-bold">{((analysis.sample_diversity || 0) * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-red-400">⚠️</span>
            Risk Assessment
          </h3>
          <div className="space-y-4">
            <div className="bg-gray-600/30 rounded p-3">
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-300">Risk Level</span>
                <span className={`font-bold capitalize ${getRiskColor(riskAssessment.risk_level)}`}>
                  {riskAssessment.risk_level || 'unknown'}
                </span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-300">Recommendation</span>
                <span className={`font-bold capitalize ${getRecommendationColor(riskAssessment.recommendation)}`}>
                  {riskAssessment.recommendation || 'unknown'}
                </span>
              </div>
              <div className="flex justify-between items-center mb-2">
                <span className="text-gray-300">Confidence Score</span>
                <span className="text-blue-400 font-bold">
                  {((riskAssessment.confidence_score || 0) * 100).toFixed(1)}%
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Decision Quality</span>
                <span className="text-green-400 font-bold capitalize">
                  {riskAssessment.decision_quality || 'unknown'}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

// Voting Strategy Component
const VotingStrategyComponent = React.memo(function VotingStrategyComponent({ data, isLoading }) {
  if (isLoading) {
    return <div className="flex items-center justify-center h-32 text-blue-400">Loading strategy data...</div>;
  }

  const strategy = data.strategy || {};
  const analysis = strategy.analysis || {};
  const breakdown = strategy.breakdown || {};
  const metrics = strategy.metrics || {};
  const performance = strategy.performance || {};

  const getGatingColor = (status) => {
    switch (status) {
      case 'passed': return 'text-green-400';
      case 'blocked': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Final Decision</div>
          <div className="text-lg font-bold text-blue-400 capitalize">{analysis.final_decision || 'none'}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Confidence</div>
          <div className="text-2xl font-bold text-green-400">{((analysis.confidence || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Signal Strength</div>
          <div className="text-2xl font-bold text-purple-400">{((analysis.signal_strength || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Gating Status</div>
          <div className={`text-lg font-bold capitalize ${getGatingColor(analysis.gating_status)}`}>
            {analysis.gating_status || 'unknown'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-blue-400">📡</span>
            Signal Analysis
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Primary Signal</span>
              <span className="text-blue-400 font-bold capitalize">{breakdown.primary_signal || 'neutral'}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Signal Coherence</span>
              <span className="text-green-400 font-bold">{((breakdown.signal_coherence || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Cross Validation</span>
              <span className="text-purple-400 font-bold">{((breakdown.cross_validation || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Execution Readiness</span>
              <span className="text-orange-400 font-bold">{((breakdown.execution_readiness || 0) * 100).toFixed(1)}%</span>
            </div>
          </div>
        </div>

        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-green-400">📈</span>
            Performance Metrics
          </h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Success Rate</span>
              <span className="text-green-400 font-bold">{((metrics.arbitration_success_rate || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Signal Accuracy</span>
              <span className="text-blue-400 font-bold">{((metrics.signal_accuracy || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Gating Efficiency</span>
              <span className="text-purple-400 font-bold">{((metrics.gating_efficiency || 0) * 100).toFixed(1)}%</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Avg Latency</span>
              <span className="text-orange-400 font-bold">{metrics.decision_latency_ms || 0}ms</span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-700/30 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
          <span className="text-purple-400">📊</span>
          Arbitration History
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-400">{performance.total_arbitrations || 0}</div>
            <div className="text-sm text-gray-400">Total</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-400">{performance.successful_arbitrations || 0}</div>
            <div className="text-sm text-gray-400">Successful</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-red-400">{performance.blocked_decisions || 0}</div>
            <div className="text-sm text-gray-400">Blocked</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-purple-400">{((performance.avg_confidence || 0) * 100).toFixed(0)}%</div>
            <div className="text-sm text-gray-400">Avg Confidence</div>
          </div>
        </div>
      </div>
    </div>
  );
});

// Voting Timeline Component
const VotingTimelineComponent = React.memo(function VotingTimelineComponent({ data, isLoading }) {
  if (isLoading) {
    return <div className="flex items-center justify-center h-32 text-blue-400">Loading timeline data...</div>;
  }

  const timeline = data.timeline || {};
  const analysis = timeline.analysis || {};
  const breakdown = timeline.breakdown || [];
  const stages = timeline.stages || [];

  const getStatusColor = (status) => {
    switch (status) {
      case 'success': return 'text-green-400';
      case 'error': return 'text-red-400';
      case 'no_data': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'success': return '✅';
      case 'error': return '❌';
      case 'no_data': return '⚠️';
      default: return '❓';
    }
  };

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Total Stages</div>
          <div className="text-2xl font-bold text-blue-400">{analysis.total_stages || 0}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Successful</div>
          <div className="text-2xl font-bold text-green-400">{analysis.successful_stages || 0}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Failed</div>
          <div className="text-2xl font-bold text-red-400">{analysis.failed_stages || 0}</div>
        </div>
        <div className="bg-gray-700/30 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Avg Time</div>
          <div className="text-2xl font-bold text-purple-400">{(analysis.avg_stage_time || 0).toFixed(1)}ms</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-blue-400">⏱️</span>
            Current Pipeline
          </h3>
          <div className="space-y-3">
            <div className="text-sm text-gray-400 mb-3">
              Decision ID: <span className="font-mono text-white">{timeline.decision_id || 'none'}</span>
            </div>
            {stages.length > 0 ? (
              stages.map((stage, index) => (
                <div key={index} className="flex items-center justify-between bg-gray-600/30 rounded p-2">
                  <div className="flex items-center gap-3">
                    <span className="text-lg">{getStatusIcon(stage.status)}</span>
                    <span className="text-white font-medium capitalize">{stage.stage}</span>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-bold ${getStatusColor(stage.status)}`}>
                      {stage.status}
                    </div>
                    <div className="text-xs text-gray-400">
                      {(stage.duration_ms || 0).toFixed(1)}ms
                    </div>
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center text-gray-400 py-4">No pipeline data available</div>
            )}
          </div>
        </div>

        <div className="bg-gray-700/30 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <span className="text-green-400">📊</span>
            Stage Performance
          </h3>
          {breakdown.length > 0 ? (
            <div className="space-y-3">
              {breakdown.map(stage => (
                <div key={stage.stage} className="bg-gray-600/30 rounded p-3">
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-semibold text-white capitalize">{stage.stage}</span>
                    <span className={`text-sm font-bold ${getStatusColor(stage.status)}`}>
                      {getStatusIcon(stage.status)}
                    </span>
                  </div>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-300">Duration</span>
                      <span className="text-blue-400">{(stage.duration_ms || 0).toFixed(1)}ms</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-300">Success Rate</span>
                      <span className="text-green-400">{((stage.success_rate || 0) * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-4">No performance data available</div>
          )}
        </div>
      </div>

      {analysis.bottleneck_stage && analysis.bottleneck_stage !== 'none' && (
        <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-yellow-400 text-lg">⚠️</span>
            <span className="text-yellow-400 font-semibold">Performance Alert</span>
          </div>
          <div className="text-gray-300">
            Bottleneck detected in <span className="font-bold text-yellow-400 capitalize">{analysis.bottleneck_stage}</span> stage
          </div>
        </div>
      )}
    </div>
  );
});

export { VotingTab };
export default VotingTab;
