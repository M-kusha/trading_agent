import React, { useState, useCallback, useEffect, useRef } from 'react';
import {
  Sparkles, Target, Vote, BookOpen, Brain, Swords, Dna, Clock,
  RefreshCw, AlertCircle, TrendingUp, TrendingDown, Minus,
  Lightbulb, Settings
} from 'lucide-react';

// ═══════════════════════════════════════════════════════════════════
// STRATEGY TAB - AI Strategy Intelligence Dashboard
// Includes: Position Decisions, Voting Breakdown, Curriculum,
//           Bias Analysis, Opponent Simulation, Genome, Timeline
// ═══════════════════════════════════════════════════════════════════

const StrategyTab = React.memo(function StrategyTab() {
  const [strategyData, setStrategyData] = useState({});
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeSection, setActiveSection] = useState('positions');
  const [hasData, setHasData] = useState(false);
  const isInitialLoad = useRef(true);
  const hasFetchedOnce = useRef(false);

  const fetchStrategyData = useCallback(async () => {
    // Only show loading spinner on initial load, not refreshes
    if (isInitialLoad.current) {
      setIsLoading(true);
    }
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 8000);

      const [positionsRes, votingRes, curriculumRes, biasRes, opponentRes, genomeRes, timelineRes] = await Promise.all([
        fetch('/api/position/decisions', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/voting/breakdown', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/strategy/curriculum', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/strategy/bias', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/strategy/opponent', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/strategy/genome', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false })),
        fetch('/api/decisions/timeline', { signal: controller.signal }).then(r => r.json()).catch(() => ({ success: false }))
      ]);

      clearTimeout(timeoutId);

      const data = {
        positions: positionsRes.success ? positionsRes : {},
        voting: votingRes.success ? votingRes : {},
        curriculum: curriculumRes.success ? curriculumRes : {},
        bias: biasRes.success ? biasRes : {},
        opponent: opponentRes.success ? opponentRes : {},
        genome: genomeRes.success ? genomeRes : {},
        timeline: timelineRes.success ? timelineRes : {}
      };

      const hasMeaningfulData = (
        Object.keys(data.positions.decisions || {}).length > 0 ||
        data.curriculum.current_stage?.name ||
        Object.keys(data.bias.adjustments || {}).length > 0
      );
      setHasData(hasMeaningfulData);
      setStrategyData(data);
      setError(null);
    } catch (err) {
      if (err.name !== 'AbortError') {
        setError(`Failed to fetch strategy data: ${err.message}`);
      }
    } finally {
      setIsLoading(false);
      isInitialLoad.current = false;
      hasFetchedOnce.current = true;
    }
  }, []);

  useEffect(() => {
    fetchStrategyData();
    const interval = setInterval(fetchStrategyData, 10000);
    return () => clearInterval(interval);
  }, [fetchStrategyData]);

  const sectionTabs = [
    { id: 'positions', name: 'Position Decisions', icon: Target },
    { id: 'voting', name: 'Voting Breakdown', icon: Vote },
    { id: 'curriculum', name: 'Learning Progress', icon: BookOpen },
    { id: 'bias', name: 'Bias Analysis', icon: Brain },
    { id: 'opponent', name: 'Opponent Sim', icon: Swords },
    { id: 'genome', name: 'Strategy Genome', icon: Dna },
    { id: 'timeline', name: 'Decision Timeline', icon: Clock }
  ];

  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400 text-center">
          <AlertCircle className="w-12 h-12 mx-auto mb-2" />
          <div>{error}</div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Sparkles className="w-7 h-7 text-purple-400" />
            AI Strategy Intelligence
          </h1>
          <p className="text-gray-400 mt-1">Deep insights into decision-making, learning, and strategy evolution</p>
        </div>
        <button
          onClick={fetchStrategyData}
          disabled={isLoading}
          className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 text-white rounded-lg transition-colors flex items-center gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {!hasData && !isLoading && (
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <AlertCircle className="w-5 h-5 text-yellow-400" />
            <div>
              <h3 className="text-yellow-400 font-medium">Limited Strategy Data Available</h3>
              <p className="text-gray-400 text-sm mt-1">
                Full strategy insights will appear after trading activity begins. Some modules may need warm-up time.
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Section Navigation */}
      <div className="bg-gray-800/50 rounded-lg border border-gray-700 p-2">
        <div className="flex flex-wrap gap-2">
          {sectionTabs.map(({ id, name, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setActiveSection(id)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                activeSection === id
                  ? 'bg-gradient-to-r from-purple-600 to-blue-600 text-white shadow-lg'
                  : 'bg-gray-700/50 text-gray-300 hover:bg-gray-600/50'
              }`}
            >
              <Icon size={16} />
              {name}
            </button>
          ))}
        </div>
      </div>

      {/* Content Sections */}
      <div className="bg-gray-800/50 rounded-xl border border-gray-700 p-6">
        {activeSection === 'positions' && <PositionDecisionsSection data={strategyData.positions} isLoading={isLoading} />}
        {activeSection === 'voting' && <VotingBreakdownSection data={strategyData.voting} isLoading={isLoading} />}
        {activeSection === 'curriculum' && <CurriculumSection data={strategyData.curriculum} isLoading={isLoading} />}
        {activeSection === 'bias' && <BiasAnalysisSection data={strategyData.bias} isLoading={isLoading} />}
        {activeSection === 'opponent' && <OpponentSimSection data={strategyData.opponent} isLoading={isLoading} />}
        {activeSection === 'genome' && <GenomeSection data={strategyData.genome} isLoading={isLoading} />}
        {activeSection === 'timeline' && <DecisionTimelineSection data={strategyData.timeline} isLoading={isLoading} />}
      </div>
    </div>
  );
});

// Position Decisions Section
const PositionDecisionsSection = React.memo(function PositionDecisionsSection({ data, isLoading }) {
  if (isLoading) return <div className="flex items-center justify-center h-32 text-purple-400">Loading position decisions...</div>;

  const decisions = data?.decisions || {};
  const portfolioState = data?.portfolio_state || {};

  const getDecisionColor = (decision) => {
    if (decision?.includes('long') || decision === 'buy') return 'text-green-400 bg-green-500/20';
    if (decision?.includes('short') || decision === 'sell') return 'text-red-400 bg-red-500/20';
    if (decision === 'hold') return 'text-yellow-400 bg-yellow-500/20';
    return 'text-gray-400 bg-gray-500/20';
  };

  const getDecisionIcon = (decision) => {
    if (decision?.includes('long') || decision === 'buy') return <TrendingUp className="w-5 h-5" />;
    if (decision?.includes('short') || decision === 'sell') return <TrendingDown className="w-5 h-5" />;
    return <Minus className="w-5 h-5" />;
  };

  return (
    <div className="space-y-6">
      {/* Portfolio Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Health Score</div>
          <div className={`text-2xl font-bold ${portfolioState.health_score > 0.7 ? 'text-green-400' : portfolioState.health_score > 0.4 ? 'text-yellow-400' : 'text-red-400'}`}>
            {((portfolioState.health_score || 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Exposure</div>
          <div className="text-2xl font-bold text-blue-400">{((portfolioState.exposure_ratio || 0) * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Balance</div>
          <div className="text-2xl font-bold text-green-400">€{(portfolioState.balance || 0).toLocaleString()}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Drawdown</div>
          <div className={`text-2xl font-bold ${(portfolioState.drawdown || 0) < 0.05 ? 'text-green-400' : 'text-red-400'}`}>
            {((portfolioState.drawdown || 0) * 100).toFixed(2)}%
          </div>
        </div>
      </div>

      {/* Per-Instrument Decisions */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Target className="w-5 h-5 text-purple-400" />
          Per-Instrument Decisions
        </h3>
        
        {Object.keys(decisions).length === 0 ? (
          <div className="text-center text-gray-400 py-8">No position decisions available</div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {Object.entries(decisions).map(([instrument, dec]) => (
              <div key={instrument} className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-white font-semibold text-lg">{instrument}</h4>
                  <span className={`px-3 py-1 rounded-full text-sm font-medium flex items-center gap-2 ${getDecisionColor(dec.decision)}`}>
                    {getDecisionIcon(dec.decision)}
                    {(dec.decision || 'hold').toUpperCase()}
                  </span>
                </div>
                
                <div className="grid grid-cols-2 gap-3 mb-3">
                  <div>
                    <div className="text-gray-500 text-xs">Confidence</div>
                    <div className="text-white font-medium">{((dec.confidence || 0) * 100).toFixed(1)}%</div>
                    <div className="w-full bg-gray-700 rounded-full h-1.5 mt-1">
                      <div className="bg-blue-500 h-1.5 rounded-full" style={{ width: `${(dec.confidence || 0) * 100}%` }} />
                    </div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs">Intensity</div>
                    <div className="text-white font-medium">{(dec.intensity || 0).toFixed(3)}</div>
                  </div>
                  <div>
                    <div className="text-gray-500 text-xs">Size</div>
                    <div className="text-cyan-400 font-medium">€{(dec.size || 0).toLocaleString()}</div>
                  </div>
                  {dec.current_position?.units > 0 && (
                    <div>
                      <div className="text-gray-500 text-xs">Current Position</div>
                      <div className="text-green-400 font-medium">{dec.current_position.units} units</div>
                    </div>
                  )}
                </div>

                {dec.rationale && dec.rationale.length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-700">
                    <div className="text-gray-500 text-xs mb-1">Rationale</div>
                    <ul className="text-gray-300 text-sm space-y-1">
                      {dec.rationale.slice(0, 3).map((r, i) => (
                        <li key={i} className="flex items-start gap-2">
                          <span className="text-purple-400">•</span>
                          <span>{r}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
});

// Voting Breakdown Section
const VotingBreakdownSection = React.memo(function VotingBreakdownSection({ data, isLoading }) {
  if (isLoading) return <div className="flex items-center justify-center h-32 text-purple-400">Loading voting breakdown...</div>;

  const voters = data?.voters || {};
  const consensusScore = data?.consensus_score || 0;
  const agreementScore = data?.agreement_score || 0;
  const votingSummary = data?.voting_summary || {};

  const getVoterColor = (confidence) => {
    if (confidence >= 0.7) return 'border-green-500/50 bg-green-500/10';
    if (confidence >= 0.5) return 'border-yellow-500/50 bg-yellow-500/10';
    return 'border-red-500/50 bg-red-500/10';
  };

  const getActionColor = (action) => {
    if (action === 'buy' || action === 'long' || action === 'proceed') return 'text-green-400';
    if (action === 'sell' || action === 'short' || action === 'avoid') return 'text-red-400';
    return 'text-yellow-400';
  };

  return (
    <div className="space-y-6">
      {/* Consensus Overview */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Consensus Score</div>
          <div className={`text-2xl font-bold ${consensusScore > 0.7 ? 'text-green-400' : consensusScore > 0.5 ? 'text-yellow-400' : 'text-red-400'}`}>
            {(consensusScore * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Agreement Score</div>
          <div className="text-2xl font-bold text-blue-400">{(agreementScore * 100).toFixed(1)}%</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Active Voters</div>
          <div className="text-2xl font-bold text-purple-400">{Object.keys(voters).length}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Decision</div>
          <div className={`text-2xl font-bold capitalize ${getActionColor(votingSummary.action)}`}>
            {votingSummary.action || 'N/A'}
          </div>
        </div>
      </div>

      {/* Individual Voter Cards */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Vote className="w-5 h-5 text-blue-400" />
          Individual Voter Proposals
        </h3>
        
        {Object.keys(voters).length === 0 ? (
          <div className="text-center text-gray-400 py-8">No voter data available</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(voters).map(([name, voter]) => (
              <div key={name} className={`rounded-lg p-4 border ${getVoterColor(voter.confidence)}`}>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-white font-medium text-sm">{name.replace(/([A-Z])/g, ' $1').trim()}</h4>
                  {voter.active && <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-xs">Confidence</span>
                    <span className={`font-bold ${voter.confidence >= 0.6 ? 'text-green-400' : 'text-yellow-400'}`}>
                      {((voter.confidence || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-2">
                    <div 
                      className={`h-2 rounded-full ${voter.confidence >= 0.6 ? 'bg-green-500' : 'bg-yellow-500'}`} 
                      style={{ width: `${(voter.confidence || 0) * 100}%` }} 
                    />
                  </div>
                  
                  {voter.proposal?.action && (
                    <div className="flex justify-between items-center pt-1">
                      <span className="text-gray-400 text-xs">Vote</span>
                      <span className={`text-sm font-medium capitalize ${getActionColor(voter.proposal.action)}`}>
                        {voter.proposal.action}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
});

// Curriculum Section
const CurriculumSection = React.memo(function CurriculumSection({ data, isLoading }) {
  if (isLoading) return <div className="flex items-center justify-center h-32 text-purple-400">Loading curriculum...</div>;

  const currentStage = data?.current_stage || {};
  const competencyScores = data?.competency_scores || {};
  const recommendations = data?.learning_recommendations || [];
  const masteryAssessment = data?.mastery_assessment || {};

  return (
    <div className="space-y-6">
      {/* Current Stage */}
      <div className="bg-gradient-to-r from-purple-600/20 to-blue-600/20 rounded-xl p-6 border border-purple-500/30">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 bg-purple-500/30 rounded-xl flex items-center justify-center">
            <BookOpen className="w-8 h-8 text-purple-400" />
          </div>
          <div>
            <div className="text-gray-400 text-sm">Current Learning Stage</div>
            <div className="text-2xl font-bold text-white">{currentStage.name || 'Foundation'}</div>
            <div className="text-gray-400 text-sm mt-1">{currentStage.description || 'Basic trading mechanics'}</div>
          </div>
        </div>
        <div className="flex items-center gap-4">
          <div className="flex-1">
            <div className="flex justify-between text-sm mb-1">
              <span className="text-gray-400">Stage Progress</span>
              <span className="text-purple-400 font-medium">{((currentStage.progress || 0) * 100).toFixed(0)}%</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-3">
              <div className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all" style={{ width: `${(currentStage.progress || 0) * 100}%` }} />
            </div>
          </div>
          <div className="text-center px-4">
            <div className="text-3xl font-bold text-purple-400">{currentStage.index || 0}</div>
            <div className="text-gray-500 text-xs">Stage #</div>
          </div>
        </div>
      </div>

      {/* Competency Scores */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Target className="w-5 h-5 text-green-400" />
          Competency Scores
        </h3>
        
        {Object.keys(competencyScores).length === 0 ? (
          <div className="text-center text-gray-400 py-4">No competency data available</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(competencyScores).map(([skill, data]) => (
              <div key={skill} className="bg-gray-900/50 rounded-lg p-4">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-white font-medium capitalize">{skill.replace(/_/g, ' ')}</span>
                  <span className={`font-bold ${data.score >= data.target ? 'text-green-400' : 'text-yellow-400'}`}>
                    {((data.score || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="relative w-full bg-gray-700 rounded-full h-2">
                  <div className="absolute bg-blue-500 h-2 rounded-full" style={{ width: `${(data.score || 0) * 100}%` }} />
                  <div className="absolute w-0.5 h-4 bg-green-400 -top-1" style={{ left: `${(data.target || 0.8) * 100}%` }} />
                </div>
                <div className="flex justify-between text-xs text-gray-500 mt-1">
                  <span>Target: {((data.target || 0.8) * 100).toFixed(0)}%</span>
                  <span>Weight: {((data.weight || 0) * 100).toFixed(0)}%</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Lightbulb className="w-5 h-5 text-yellow-400" />
            Learning Recommendations
          </h3>
          <div className="space-y-2">
            {recommendations.slice(0, 5).map((rec, i) => (
              <div key={i} className="bg-gray-900/50 rounded-lg p-3 flex items-start gap-3">
                <span className="text-yellow-400">💡</span>
                <span className="text-gray-300 text-sm">{rec}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

// Bias Analysis Section
const BiasAnalysisSection = React.memo(function BiasAnalysisSection({ data, isLoading }) {
  if (isLoading) return <div className="flex items-center justify-center h-32 text-purple-400">Loading bias analysis...</div>;

  const biasScores = data?.bias_scores || {};
  const adjustments = data?.adjustments || {};
  const recommendations = data?.recommendations || [];
  const psychState = data?.psychological_state || {};

  const getBiasColor = (value) => {
    if (value >= 0.9) return 'text-green-400';
    if (value >= 0.7) return 'text-yellow-400';
    return 'text-red-400';
  };

  const biasTypes = ['revenge', 'fear', 'greed', 'fomo', 'anchoring'];

  return (
    <div className="space-y-6">
      {/* Bias Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Total Bias Score</div>
          <div className={`text-2xl font-bold ${biasScores.total_bias_score < 0.3 ? 'text-green-400' : 'text-yellow-400'}`}>
            {((biasScores.total_bias_score || 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Dominant Bias</div>
          <div className="text-2xl font-bold text-purple-400 capitalize">{biasScores.dominant_bias || 'None'}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Severity</div>
          <div className={`text-2xl font-bold capitalize ${
            biasScores.bias_severity === 'low' ? 'text-green-400' : 
            biasScores.bias_severity === 'medium' ? 'text-yellow-400' : 'text-red-400'
          }`}>
            {biasScores.bias_severity || 'Low'}
          </div>
        </div>
      </div>

      {/* Bias Adjustments */}
      <div className="space-y-4">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Brain className="w-5 h-5 text-pink-400" />
          Psychological Adjustments
        </h3>
        
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          {biasTypes.map(bias => {
            const value = adjustments[bias] ?? 1.0;
            return (
              <div key={bias} className="bg-gray-900/50 rounded-lg p-4 text-center">
                <div className="text-3xl mb-2">
                  {bias === 'revenge' && '😤'}
                  {bias === 'fear' && '😰'}
                  {bias === 'greed' && '🤑'}
                  {bias === 'fomo' && '😱'}
                  {bias === 'anchoring' && '⚓'}
                </div>
                <div className="text-white font-medium capitalize">{bias}</div>
                <div className={`text-lg font-bold ${getBiasColor(value)}`}>
                  {(value * 100).toFixed(0)}%
                </div>
                <div className="w-full bg-gray-700 rounded-full h-1.5 mt-2">
                  <div className={`h-1.5 rounded-full ${value >= 0.9 ? 'bg-green-500' : value >= 0.7 ? 'bg-yellow-500' : 'bg-red-500'}`} 
                       style={{ width: `${value * 100}%` }} />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Bias Mitigation Recommendations</h3>
          <div className="space-y-2">
            {recommendations.slice(0, 5).map((rec, i) => (
              <div key={i} className="bg-gray-900/50 rounded-lg p-3 flex items-start gap-3">
                <span className="text-pink-400">🧠</span>
                <span className="text-gray-300 text-sm">{typeof rec === 'string' ? rec : JSON.stringify(rec)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

// Opponent Simulation Section
const OpponentSimSection = React.memo(function OpponentSimSection({ data, isLoading }) {
  if (isLoading) return <div className="flex items-center justify-center h-32 text-purple-400">Loading opponent simulation...</div>;

  const simulation = data?.simulation || {};
  const effectiveness = data?.effectiveness || {};
  const scenarios = data?.adversarial_scenarios || [];

  const getModeColor = (mode) => {
    switch(mode) {
      case 'aggressive': return 'text-red-400 bg-red-500/20';
      case 'defensive': return 'text-blue-400 bg-blue-500/20';
      case 'adaptive': return 'text-purple-400 bg-purple-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  return (
    <div className="space-y-6">
      {/* Simulation Status */}
      <div className="bg-gradient-to-r from-red-600/20 to-orange-600/20 rounded-xl p-6 border border-red-500/30">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 bg-red-500/30 rounded-xl flex items-center justify-center">
            <Swords className="w-8 h-8 text-red-400" />
          </div>
          <div>
            <div className="text-gray-400 text-sm">Adversarial Training Mode</div>
            <div className="flex items-center gap-3">
              <span className={`text-2xl font-bold capitalize px-3 py-1 rounded-lg ${getModeColor(simulation.mode)}`}>
                {simulation.mode || 'Random'}
              </span>
              <span className="text-gray-400">Intensity: {((simulation.intensity || 1) * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Effectiveness Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Effectiveness Score</div>
          <div className={`text-2xl font-bold ${effectiveness.score > 0.5 ? 'text-green-400' : 'text-yellow-400'}`}>
            {((effectiveness.score || 0.5) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Impact Variance</div>
          <div className="text-2xl font-bold text-orange-400">{(effectiveness.impact_variance || 0).toFixed(4)}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Robustness Contribution</div>
          <div className="text-2xl font-bold text-blue-400">{((effectiveness.robustness_contribution || 0) * 100).toFixed(1)}%</div>
        </div>
      </div>

      {/* Context Adjustments */}
      {simulation.context_adjustments && Object.keys(simulation.context_adjustments).length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <Settings className="w-5 h-5 text-orange-400" />
            Context Adjustments
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(simulation.context_adjustments).map(([key, value]) => (
              <div key={key} className="bg-gray-900/50 rounded-lg p-3">
                <div className="text-gray-400 text-xs capitalize">{key.replace(/_/g, ' ')}</div>
                <div className="text-white font-medium">{typeof value === 'number' ? value.toFixed(3) : String(value)}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Adversarial Scenarios */}
      {scenarios.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Recent Adversarial Scenarios</h3>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {scenarios.slice(0, 5).map((scenario, i) => (
              <div key={i} className="bg-gray-900/50 rounded-lg p-3 text-sm text-gray-300">
                {typeof scenario === 'string' ? scenario : JSON.stringify(scenario)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

// Genome Section
const GenomeSection = React.memo(function GenomeSection({ data, isLoading }) {
  if (isLoading) return <div className="flex items-center justify-center h-32 text-purple-400">Loading genome data...</div>;

  const bestGenome = data?.best_genome || {};
  const activeWeights = data?.active_weights || {};
  const evolutionHistory = data?.evolution_history || [];
  const recommendations = data?.recommendations || [];

  return (
    <div className="space-y-6">
      {/* Best Genome */}
      <div className="bg-gradient-to-r from-green-600/20 to-teal-600/20 rounded-xl p-6 border border-green-500/30">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 bg-green-500/30 rounded-xl flex items-center justify-center">
            <Dna className="w-8 h-8 text-green-400" />
          </div>
          <div>
            <div className="text-gray-400 text-sm">Best Strategy Genome</div>
            <div className="flex items-center gap-4">
              <span className="text-2xl font-bold text-green-400">Gen #{bestGenome.generation || 0}</span>
              <span className="text-gray-400">Fitness: {(bestGenome.fitness || 0).toFixed(4)}</span>
            </div>
          </div>
        </div>
        
        {/* Genome Parameters */}
        {bestGenome.parameters && bestGenome.parameters.length > 0 && (
          <div className="grid grid-cols-4 gap-2 mt-4">
            {bestGenome.parameters.map((param, i) => (
              <div key={i} className="bg-gray-800/50 rounded-lg p-2 text-center">
                <div className="text-gray-500 text-xs">Param {i + 1}</div>
                <div className="text-white font-mono">{param.toFixed(3)}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Active Weights */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Active Genome Index</div>
          <div className="text-2xl font-bold text-green-400">#{activeWeights.genome_idx || 0}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Active Fitness</div>
          <div className="text-2xl font-bold text-blue-400">{(activeWeights.fitness || 0).toFixed(4)}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Population Size</div>
          <div className="text-2xl font-bold text-purple-400">{activeWeights.population_size || 0}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Evolution Status</div>
          <div className="text-2xl font-bold text-teal-400">Active</div>
        </div>
      </div>

      {/* Evolution History */}
      {evolutionHistory.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-green-400" />
            Evolution History
          </h3>
          <div className="bg-gray-900/50 rounded-lg p-4 max-h-48 overflow-y-auto">
            <div className="space-y-2">
              {evolutionHistory.slice(-10).reverse().map((entry, i) => (
                <div key={i} className="flex justify-between items-center text-sm">
                  <span className="text-gray-400">Gen {entry.generation || i}</span>
                  <span className="text-green-400 font-mono">{(entry.fitness || 0).toFixed(4)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Recommendations */}
      {recommendations.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Genome Recommendations</h3>
          <div className="space-y-2">
            {recommendations.map((rec, i) => (
              <div key={i} className="bg-gray-900/50 rounded-lg p-3 flex items-start gap-3">
                <span className="text-green-400">🧬</span>
                <span className="text-gray-300 text-sm">{typeof rec === 'string' ? rec : JSON.stringify(rec)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

// Decision Timeline Section
const DecisionTimelineSection = React.memo(function DecisionTimelineSection({ data, isLoading }) {
  if (isLoading) return <div className="flex items-center justify-center h-32 text-purple-400">Loading timeline...</div>;

  const rationales = data?.rationales || [];
  const explanationMetrics = data?.explanation_metrics || {};
  const thesis = data?.thesis || {};
  const narratives = data?.narratives || [];

  return (
    <div className="space-y-6">
      {/* Explanation Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Trades Audited</div>
          <div className="text-2xl font-bold text-blue-400">{explanationMetrics.total_trades_audited || 0}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">High Confidence</div>
          <div className="text-2xl font-bold text-green-400">{explanationMetrics.high_confidence_trades || 0}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Low Confidence</div>
          <div className="text-2xl font-bold text-yellow-400">{explanationMetrics.low_confidence_trades || 0}</div>
        </div>
        <div className="bg-gray-900/50 rounded-lg p-4">
          <div className="text-gray-400 text-sm">Missing Explanations</div>
          <div className="text-2xl font-bold text-red-400">{explanationMetrics.missing_explanations || 0}</div>
        </div>
      </div>

      {/* Current Thesis */}
      <div className="bg-gradient-to-r from-blue-600/20 to-indigo-600/20 rounded-xl p-6 border border-blue-500/30">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 bg-blue-500/30 rounded-xl flex items-center justify-center">
            <Sparkles className="w-6 h-6 text-blue-400" />
          </div>
          <div>
            <div className="text-gray-400 text-sm">Current Trading Thesis</div>
            <div className="text-xl font-bold text-white capitalize">{thesis.current || 'Neutral'}</div>
          </div>
        </div>
      </div>

      {/* Decision Rationales */}
      <div className="space-y-3">
        <h3 className="text-lg font-semibold text-white flex items-center gap-2">
          <Clock className="w-5 h-5 text-indigo-400" />
          Recent Decision Rationales
        </h3>
        
        {rationales.length === 0 ? (
          <div className="text-center text-gray-400 py-8">No decision rationales available</div>
        ) : (
          <div className="space-y-3 max-h-96 overflow-y-auto">
            {rationales.slice(0, 15).map((rationale, i) => (
              <div key={i} className="bg-gray-900/50 rounded-lg p-4 border-l-4 border-indigo-500">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 w-8 h-8 bg-indigo-500/20 rounded-full flex items-center justify-center text-indigo-400 text-sm font-bold">
                    {i + 1}
                  </div>
                  <div className="flex-1">
                    <p className="text-gray-300 text-sm">{typeof rationale === 'string' ? rationale : JSON.stringify(rationale)}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Contextual Narratives */}
      {narratives.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-lg font-semibold text-white">Contextual Narratives</h3>
          <div className="space-y-2">
            {narratives.map((narrative, i) => (
              <div key={i} className="bg-gray-900/50 rounded-lg p-3 text-gray-300 text-sm">
                {typeof narrative === 'string' ? narrative : JSON.stringify(narrative)}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

export { StrategyTab };
export default StrategyTab;
