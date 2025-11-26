// Custom hooks for data fetching with automatic caching and throttling
import { useCallback, useRef, useEffect } from 'react';
import { useDataDispatch, useDataState, DataActions } from './StoreProvider';

const API_BASE = '/api';

// Shared API cache
const apiCache = new Map();
const API_CACHE_TTL = 5000; // 5 seconds

/**
 * Generic API call with caching
 */
export async function apiCall(endpoint, options = {}) {
  const cacheKey = `${endpoint}_${JSON.stringify(options)}`;
  const cached = apiCache.get(cacheKey);
  
  if (cached && Date.now() - cached.timestamp < API_CACHE_TTL && !options.noCache) {
    return cached.data;
  }
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP ${response.status}`);
  }
  
  const data = await response.json();
  apiCache.set(cacheKey, { data, timestamp: Date.now() });
  return data;
}

/**
 * Hook for fetching analytics data
 */
export function useAnalyticsData(options = {}) {
  const dispatch = useDataDispatch();
  const { analyticsData, loading, lastUpdate } = useDataState();
  const { throttleMs = 10000, autoFetch = true } = options;
  
  const fetch = useCallback(async (force = false) => {
    if (!force && Date.now() - lastUpdate.analytics < throttleMs) {
      return analyticsData;
    }
    
    dispatch(DataActions.setAnalyticsLoading(true));
    
    try {
      const [visualization, dashboard, performance, modules] = await Promise.all([
        apiCall('/visualization-data'),
        apiCall('/dashboard-data'),
        apiCall('/performance-metrics'),
        apiCall('/modules'),
      ]);
      
      const data = { visualization, dashboard, performance, modules };
      dispatch(DataActions.setAnalyticsData(data));
      return data;
    } catch (error) {
      console.error('Failed to fetch analytics:', error);
      dispatch(DataActions.setAnalyticsLoading(false));
      throw error;
    }
  }, [dispatch, analyticsData, lastUpdate.analytics, throttleMs]);
  
  useEffect(() => {
    if (autoFetch && !loading.analytics && Date.now() - lastUpdate.analytics > throttleMs) {
      fetch();
    }
  }, [autoFetch, fetch, loading.analytics, lastUpdate.analytics, throttleMs]);
  
  return { data: analyticsData, loading: loading.analytics, refetch: fetch };
}

/**
 * Hook for fetching memory data
 */
export function useMemoryData(options = {}) {
  const dispatch = useDataDispatch();
  const { memoryData, loading, lastUpdate } = useDataState();
  const { throttleMs = 15000, autoFetch = true } = options;
  
  const fetch = useCallback(async (force = false) => {
    if (!force && Date.now() - lastUpdate.memory < throttleMs) {
      return memoryData;
    }
    
    dispatch(DataActions.setMemoryLoading(true));
    
    try {
      const [overview, patterns, signals, gate, dangers, stats] = await Promise.all([
        apiCall('/memory/overview').catch(() => ({})),
        apiCall('/memory/patterns').catch(() => ({ patterns: [] })),
        apiCall('/memory/signals').catch(() => ({ signals: {} })),
        apiCall('/memory/gate').catch(() => ({})),
        apiCall('/memory/dangers').catch(() => ({ danger_zones: [] })),
        apiCall('/memory/stats').catch(() => ({})),
      ]);
      
      const data = { overview, patterns, signals, gate, dangers, stats };
      dispatch(DataActions.setMemoryData(data));
      return data;
    } catch (error) {
      console.error('Failed to fetch memory data:', error);
      dispatch(DataActions.setMemoryLoading(false));
      throw error;
    }
  }, [dispatch, memoryData, lastUpdate.memory, throttleMs]);
  
  useEffect(() => {
    if (autoFetch && !loading.memory && Date.now() - lastUpdate.memory > throttleMs) {
      fetch();
    }
  }, [autoFetch, fetch, loading.memory, lastUpdate.memory, throttleMs]);
  
  return { data: memoryData, loading: loading.memory, refetch: fetch };
}

/**
 * Hook for fetching risk data
 */
export function useRiskData(options = {}) {
  const dispatch = useDataDispatch();
  const { riskData, loading, lastUpdate } = useDataState();
  const { throttleMs = 10000, autoFetch = true } = options;
  
  const fetch = useCallback(async (force = false) => {
    if (!force && Date.now() - lastUpdate.risk < throttleMs) {
      return riskData;
    }
    
    dispatch(DataActions.setRiskLoading(true));
    
    try {
      const [overview, metrics, policy, exposure, drawdown, limits] = await Promise.all([
        apiCall('/risk/overview').catch(() => ({})),
        apiCall('/risk/metrics').catch(() => ({})),
        apiCall('/risk/policy').catch(() => ({})),
        apiCall('/risk/exposure').catch(() => ({})),
        apiCall('/risk/drawdown').catch(() => ({})),
        apiCall('/risk/limits').catch(() => ({})),
      ]);
      
      const data = { overview, metrics, policy, exposure, drawdown, limits };
      dispatch(DataActions.setRiskData(data));
      return data;
    } catch (error) {
      console.error('Failed to fetch risk data:', error);
      dispatch(DataActions.setRiskLoading(false));
      throw error;
    }
  }, [dispatch, riskData, lastUpdate.risk, throttleMs]);
  
  useEffect(() => {
    if (autoFetch && !loading.risk && Date.now() - lastUpdate.risk > throttleMs) {
      fetch();
    }
  }, [autoFetch, fetch, loading.risk, lastUpdate.risk, throttleMs]);
  
  return { data: riskData, loading: loading.risk, refetch: fetch };
}

/**
 * Hook for fetching strategy data
 */
export function useStrategyData(options = {}) {
  const dispatch = useDataDispatch();
  const { strategyData, loading, lastUpdate } = useDataState();
  const { throttleMs = 15000, autoFetch = true } = options;
  
  const fetch = useCallback(async (force = false) => {
    if (!force && Date.now() - lastUpdate.strategy < throttleMs) {
      return strategyData;
    }
    
    dispatch(DataActions.setStrategyLoading(true));
    
    try {
      const [active, performance, signals, arbiter, correlation, theses] = await Promise.all([
        apiCall('/strategy/active').catch(() => ({})),
        apiCall('/strategy/performance').catch(() => ({})),
        apiCall('/strategy/signals').catch(() => ({ signals: [] })),
        apiCall('/strategy/arbiter').catch(() => ({})),
        apiCall('/strategy/correlation').catch(() => ({})),
        apiCall('/strategy/theses').catch(() => ({ theses: [] })),
      ]);
      
      const data = { active, performance, signals, arbiter, correlation, theses };
      dispatch(DataActions.setStrategyData(data));
      return data;
    } catch (error) {
      console.error('Failed to fetch strategy data:', error);
      dispatch(DataActions.setStrategyLoading(false));
      throw error;
    }
  }, [dispatch, strategyData, lastUpdate.strategy, throttleMs]);
  
  useEffect(() => {
    if (autoFetch && !loading.strategy && Date.now() - lastUpdate.strategy > throttleMs) {
      fetch();
    }
  }, [autoFetch, fetch, loading.strategy, lastUpdate.strategy, throttleMs]);
  
  return { data: strategyData, loading: loading.strategy, refetch: fetch };
}

/**
 * Hook for fetching voting data
 */
export function useVotingData(options = {}) {
  const dispatch = useDataDispatch();
  const { votingData, loading, lastUpdate } = useDataState();
  const { throttleMs = 10000, autoFetch = true } = options;
  
  const fetch = useCallback(async (force = false) => {
    if (!force && Date.now() - lastUpdate.voting < throttleMs) {
      return votingData;
    }
    
    dispatch(DataActions.setVotingLoading(true));
    
    try {
      const [consensus, proposals, analysis, metrics, history] = await Promise.all([
        apiCall('/voting/consensus').catch(() => ({})),
        apiCall('/voting/proposals').catch(() => ({ proposals: [] })),
        apiCall('/voting/analysis').catch(() => ({})),
        apiCall('/voting/metrics').catch(() => ({})),
        apiCall('/voting/history').catch(() => ({ history: [] })),
      ]);
      
      const data = { consensus, proposals, analysis, metrics, history };
      dispatch(DataActions.setVotingData(data));
      return data;
    } catch (error) {
      console.error('Failed to fetch voting data:', error);
      dispatch(DataActions.setVotingLoading(false));
      throw error;
    }
  }, [dispatch, votingData, lastUpdate.voting, throttleMs]);
  
  useEffect(() => {
    if (autoFetch && !loading.voting && Date.now() - lastUpdate.voting > throttleMs) {
      fetch();
    }
  }, [autoFetch, fetch, loading.voting, lastUpdate.voting, throttleMs]);
  
  return { data: votingData, loading: loading.voting, refetch: fetch };
}

/**
 * Hook for fetching logs
 */
export function useLogs(category) {
  const dispatch = useDataDispatch();
  const { logs, loading } = useDataState();
  
  const fetch = useCallback(async () => {
    dispatch(DataActions.setLogsLoading(true));
    
    try {
      const data = await apiCall(`/logs/${category}`);
      dispatch(DataActions.setLogs(category, data));
      return data;
    } catch (error) {
      console.error(`Failed to fetch ${category} logs:`, error);
      dispatch(DataActions.setLogsLoading(false));
      throw error;
    }
  }, [dispatch, category]);
  
  return { data: logs[category] || {}, loading: loading.logs, refetch: fetch };
}

/**
 * Clear API cache (useful after mutations)
 */
export function clearApiCache(pattern) {
  if (pattern) {
    for (const key of apiCache.keys()) {
      if (key.includes(pattern)) {
        apiCache.delete(key);
      }
    }
  } else {
    apiCache.clear();
  }
}
