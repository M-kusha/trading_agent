import React, {
  createContext,
  useContext,
  useReducer,
  useMemo,
  useCallback,
  useRef,
  useSyncExternalStore,
} from 'react';

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY: Shallow equality check for selectors
// ═══════════════════════════════════════════════════════════════════════════

function shallowEqual(objA, objB) {
  if (objA === objB) return true;
  if (!objA || !objB) return false;
  
  const keysA = Object.keys(objA);
  const keysB = Object.keys(objB);
  
  if (keysA.length !== keysB.length) return false;
  
  for (const key of keysA) {
    if (objA[key] !== objB[key]) return false;
  }
  return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// UI STATE - Tab selections, filters, view modes
// ═══════════════════════════════════════════════════════════════════════════

const UIStateContext = createContext(null);
const UIDispatchContext = createContext(null);

const initialUIState = {
  activeTab: 'overview',
  selectedTimeframe: '1h',
  selectedView: 'performance',
  selectedSymbol: 'EURUSD',
  moduleSearch: '',
  moduleFilter: 'all',
  moduleViewMode: 'grid',
  selectedLogCategory: 'system',
  alertFilter: 'all',
  showAlerts: false,
};

function uiReducer(state, action) {
  switch (action.type) {
    case 'SET_ACTIVE_TAB':
      return state.activeTab === action.payload ? state : { ...state, activeTab: action.payload };
    case 'SET_TIMEFRAME':
      return state.selectedTimeframe === action.payload ? state : { ...state, selectedTimeframe: action.payload };
    case 'SET_VIEW':
      return state.selectedView === action.payload ? state : { ...state, selectedView: action.payload };
    case 'SET_SYMBOL':
      return state.selectedSymbol === action.payload ? state : { ...state, selectedSymbol: action.payload };
    case 'SET_MODULE_SEARCH':
      return state.moduleSearch === action.payload ? state : { ...state, moduleSearch: action.payload };
    case 'SET_MODULE_FILTER':
      return state.moduleFilter === action.payload ? state : { ...state, moduleFilter: action.payload };
    case 'SET_MODULE_VIEW_MODE':
      return state.moduleViewMode === action.payload ? state : { ...state, moduleViewMode: action.payload };
    case 'SET_LOG_CATEGORY':
      return state.selectedLogCategory === action.payload ? state : { ...state, selectedLogCategory: action.payload };
    case 'SET_ALERT_FILTER':
      return state.alertFilter === action.payload ? state : { ...state, alertFilter: action.payload };
    case 'SET_SHOW_ALERTS':
      return state.showAlerts === action.payload ? state : { ...state, showAlerts: action.payload };
    default:
      return state;
  }
}

export const UIActions = {
  setActiveTab: (tab) => ({ type: 'SET_ACTIVE_TAB', payload: tab }),
  setTimeframe: (tf) => ({ type: 'SET_TIMEFRAME', payload: tf }),
  setView: (view) => ({ type: 'SET_VIEW', payload: view }),
  setSymbol: (symbol) => ({ type: 'SET_SYMBOL', payload: symbol }),
  setModuleSearch: (search) => ({ type: 'SET_MODULE_SEARCH', payload: search }),
  setModuleFilter: (filter) => ({ type: 'SET_MODULE_FILTER', payload: filter }),
  setModuleViewMode: (mode) => ({ type: 'SET_MODULE_VIEW_MODE', payload: mode }),
  setLogCategory: (category) => ({ type: 'SET_LOG_CATEGORY', payload: category }),
  setAlertFilter: (filter) => ({ type: 'SET_ALERT_FILTER', payload: filter }),
  setShowAlerts: (show) => ({ type: 'SET_SHOW_ALERTS', payload: show }),
};

// ═══════════════════════════════════════════════════════════════════════════
// TRADING STATE - System status, positions, account
// ═══════════════════════════════════════════════════════════════════════════

const TradingStateContext = createContext(null);
const TradingDispatchContext = createContext(null);

const initialTradingState = {
  isLoggedIn: false,
  systemStatus: 'IDLE',
  wsConnected: false,
  error: '',
  
  // Account & Positions
  accountInfo: null,
  positions: [],
  positionCount: 0,
  recentTrades: [],
  
  // MT5 data
  mt5ChartData: [],
  mt5Symbols: [],
  
  // Performance
  performance: {},
  systemMetrics: {},
  
  // Alerts
  alerts: [],
  dismissedAlerts: new Set(),
  readAlerts: new Set(),
  
  // Config
  tradingConfig: {
    instruments: ["EURUSD", "XAUUSD"],
    timeframes: ["M15", "H1", "H4", "D1"],
    update_interval: 5,
    max_position_size: 0.1,
    max_total_exposure: 0.3,
    min_trade_interval: 60,
    use_trailing_stop: true,
    emergency_drawdown_limit: 0.25,
    debug: false
  },
};

function tradingReducer(state, action) {
  switch (action.type) {
    case 'SET_LOGGED_IN':
      return { ...state, isLoggedIn: action.payload };
    case 'SET_SYSTEM_STATUS':
      return state.systemStatus === action.payload ? state : { ...state, systemStatus: action.payload };
    case 'SET_WS_CONNECTED':
      return state.wsConnected === action.payload ? state : { ...state, wsConnected: action.payload };
    case 'SET_ERROR':
      return { ...state, error: action.payload };
    case 'CLEAR_ERROR':
      return { ...state, error: '' };
      
    case 'SET_ACCOUNT_INFO':
      return { ...state, accountInfo: action.payload };
    case 'SET_POSITIONS': {
      // Only update if actually changed
      if (JSON.stringify(state.positions) === JSON.stringify(action.payload)) return state;
      return { ...state, positions: action.payload, positionCount: action.payload?.length || 0 };
    }
    case 'SET_RECENT_TRADES': {
      if (JSON.stringify(state.recentTrades) === JSON.stringify(action.payload)) return state;
      return { ...state, recentTrades: action.payload };
    }
    
    case 'SET_MT5_CHART_DATA': {
      if (JSON.stringify(state.mt5ChartData) === JSON.stringify(action.payload)) return state;
      return { ...state, mt5ChartData: action.payload };
    }
    case 'SET_MT5_SYMBOLS': {
      if (JSON.stringify(state.mt5Symbols) === JSON.stringify(action.payload)) return state;
      return { ...state, mt5Symbols: action.payload };
    }
    
    case 'SET_PERFORMANCE': {
      // Merge performance data
      const merged = { ...state.performance, ...action.payload };
      if (JSON.stringify(state.performance) === JSON.stringify(merged)) return state;
      return { ...state, performance: merged };
    }
    case 'SET_SYSTEM_METRICS': {
      const merged = { ...state.systemMetrics, ...action.payload };
      if (JSON.stringify(state.systemMetrics) === JSON.stringify(merged)) return state;
      return { ...state, systemMetrics: merged };
    }
    
    case 'SET_ALERTS':
      return { ...state, alerts: action.payload };
    case 'DISMISS_ALERT': {
      const newDismissed = new Set(state.dismissedAlerts);
      newDismissed.add(action.payload);
      return { ...state, dismissedAlerts: newDismissed };
    }
    case 'MARK_ALERTS_READ': {
      const newRead = new Set([...state.readAlerts, ...action.payload]);
      return { ...state, readAlerts: newRead };
    }
    
    case 'SET_TRADING_CONFIG':
      return { ...state, tradingConfig: { ...state.tradingConfig, ...action.payload } };
    
    case 'BATCH_UPDATE': {
      // For WebSocket bulk updates - only change what's different
      const updates = {};
      if (action.payload.status !== undefined && action.payload.status !== state.systemStatus) {
        updates.systemStatus = action.payload.status;
      }
      if (action.payload.performance) {
        const merged = { ...state.performance, ...action.payload.performance };
        if (JSON.stringify(state.performance) !== JSON.stringify(merged)) {
          updates.performance = merged;
        }
      }
      if (action.payload.alerts && JSON.stringify(state.alerts) !== JSON.stringify(action.payload.alerts)) {
        updates.alerts = action.payload.alerts;
      }
      if (action.payload.systemMetrics) {
        const merged = { ...state.systemMetrics, ...action.payload.systemMetrics };
        if (JSON.stringify(state.systemMetrics) !== JSON.stringify(merged)) {
          updates.systemMetrics = merged;
        }
      }
      return Object.keys(updates).length > 0 ? { ...state, ...updates } : state;
    }
    
    case 'LOGOUT':
      return {
        ...initialTradingState,
        wsConnected: state.wsConnected, // preserve connection state
        tradingConfig: state.tradingConfig, // preserve config
      };
    
    default:
      return state;
  }
}

export const TradingActions = {
  setLoggedIn: (val) => ({ type: 'SET_LOGGED_IN', payload: val }),
  setSystemStatus: (status) => ({ type: 'SET_SYSTEM_STATUS', payload: status }),
  setWsConnected: (val) => ({ type: 'SET_WS_CONNECTED', payload: val }),
  setError: (error) => ({ type: 'SET_ERROR', payload: error }),
  clearError: () => ({ type: 'CLEAR_ERROR' }),
  setAccountInfo: (info) => ({ type: 'SET_ACCOUNT_INFO', payload: info }),
  setPositions: (positions) => ({ type: 'SET_POSITIONS', payload: positions }),
  setRecentTrades: (trades) => ({ type: 'SET_RECENT_TRADES', payload: trades }),
  setMT5ChartData: (data) => ({ type: 'SET_MT5_CHART_DATA', payload: data }),
  setMT5Symbols: (symbols) => ({ type: 'SET_MT5_SYMBOLS', payload: symbols }),
  setPerformance: (perf) => ({ type: 'SET_PERFORMANCE', payload: perf }),
  setSystemMetrics: (metrics) => ({ type: 'SET_SYSTEM_METRICS', payload: metrics }),
  setAlerts: (alerts) => ({ type: 'SET_ALERTS', payload: alerts }),
  dismissAlert: (alertId) => ({ type: 'DISMISS_ALERT', payload: alertId }),
  markAlertsRead: (alertIds) => ({ type: 'MARK_ALERTS_READ', payload: alertIds }),
  setTradingConfig: (config) => ({ type: 'SET_TRADING_CONFIG', payload: config }),
  batchUpdate: (data) => ({ type: 'BATCH_UPDATE', payload: data }),
  logout: () => ({ type: 'LOGOUT' }),
};

// ═══════════════════════════════════════════════════════════════════════════
// MODULES STATE - Module list, categories, stats
// ═══════════════════════════════════════════════════════════════════════════

const ModulesStateContext = createContext(null);
const ModulesDispatchContext = createContext(null);

const initialModulesState = {
  modules: [],
  modulesById: {},
  categories: {},
  stats: { total: 0, enabled: 0, withData: 0, withErrors: 0 },
  moduleStates: {},
  selectedModuleId: null,
  lastUpdate: 0,
  isLoaded: false,
};

function modulesReducer(state, action) {
  switch (action.type) {
    case 'SET_MODULES': {
      const modules = action.payload || [];
      const modulesById = {};
      for (const m of modules) {
        const id = m.id ?? m.name;
        const prev = state.modulesById[id];
        // Only create new reference if data changed
        if (!prev || JSON.stringify(prev) !== JSON.stringify({ ...prev, ...m })) {
          modulesById[id] = { ...prev, ...m, id };
        } else {
          modulesById[id] = prev;
        }
      }
      return {
        ...state,
        modules,
        modulesById,
        lastUpdate: Date.now(),
        isLoaded: true,
      };
    }
    case 'SET_CATEGORIES':
      return { ...state, categories: action.payload };
    case 'SET_MODULE_STATS':
      return { ...state, stats: action.payload };
    case 'SET_MODULE_STATES': {
      if (JSON.stringify(state.moduleStates) === JSON.stringify(action.payload)) return state;
      return { ...state, moduleStates: action.payload };
    }
    case 'UPDATE_MODULE_STATE': {
      const { moduleId, stateData } = action.payload;
      const prev = state.moduleStates[moduleId];
      if (prev && JSON.stringify(prev) === JSON.stringify(stateData)) return state;
      return {
        ...state,
        moduleStates: { ...state.moduleStates, [moduleId]: stateData }
      };
    }
    case 'SET_SELECTED_MODULE':
      return state.selectedModuleId === action.payload ? state : { ...state, selectedModuleId: action.payload };
    default:
      return state;
  }
}

export const ModulesActions = {
  setModules: (modules) => ({ type: 'SET_MODULES', payload: modules }),
  setCategories: (categories) => ({ type: 'SET_CATEGORIES', payload: categories }),
  setModuleStats: (stats) => ({ type: 'SET_MODULE_STATS', payload: stats }),
  setModuleStates: (states) => ({ type: 'SET_MODULE_STATES', payload: states }),
  updateModuleState: (moduleId, stateData) => ({ type: 'UPDATE_MODULE_STATE', payload: { moduleId, stateData } }),
  setSelectedModule: (id) => ({ type: 'SET_SELECTED_MODULE', payload: id }),
};

// ═══════════════════════════════════════════════════════════════════════════
// DATA STATE - Analytics, Memory, Risk, Voting, Logs
// ═══════════════════════════════════════════════════════════════════════════

const DataStateContext = createContext(null);
const DataDispatchContext = createContext(null);

const initialDataState = {
  // Tab-specific data
  analyticsData: {},
  memoryData: {},
  riskData: {},
  strategyData: {},
  votingData: {},
  
  // Logs
  logs: {},
  checkpoints: [],
  
  // System state from WS
  systemState: null,
  
  // Loading states
  loading: {
    analytics: false,
    memory: false,
    risk: false,
    strategy: false,
    voting: false,
    logs: false,
  },
  
  // Loaded flags
  loaded: {
    analytics: false,
    memory: false,
    risk: false,
    strategy: false,
    voting: false,
    logs: false,
    modules: false,
    mt5Data: false,
  },
  
  // Last update timestamps for throttling
  lastUpdate: {
    analytics: 0,
    memory: 0,
    risk: 0,
    strategy: 0,
    voting: 0,
    system: 0,
    modules: 0,
    mt5: 0,
  },
};

function dataReducer(state, action) {
  switch (action.type) {
    // Analytics
    case 'SET_ANALYTICS_DATA':
      return {
        ...state,
        analyticsData: action.payload,
        loading: { ...state.loading, analytics: false },
        loaded: { ...state.loaded, analytics: true },
        lastUpdate: { ...state.lastUpdate, analytics: Date.now() },
      };
    case 'SET_ANALYTICS_LOADING':
      return { ...state, loading: { ...state.loading, analytics: action.payload } };
    
    // Memory
    case 'SET_MEMORY_DATA':
      return {
        ...state,
        memoryData: action.payload,
        loading: { ...state.loading, memory: false },
        loaded: { ...state.loaded, memory: true },
        lastUpdate: { ...state.lastUpdate, memory: Date.now() },
      };
    case 'SET_MEMORY_LOADING':
      return { ...state, loading: { ...state.loading, memory: action.payload } };
    
    // Risk
    case 'SET_RISK_DATA':
      return {
        ...state,
        riskData: action.payload,
        loading: { ...state.loading, risk: false },
        loaded: { ...state.loaded, risk: true },
        lastUpdate: { ...state.lastUpdate, risk: Date.now() },
      };
    case 'SET_RISK_LOADING':
      return { ...state, loading: { ...state.loading, risk: action.payload } };
    
    // Strategy
    case 'SET_STRATEGY_DATA':
      return {
        ...state,
        strategyData: action.payload,
        loading: { ...state.loading, strategy: false },
        loaded: { ...state.loaded, strategy: true },
        lastUpdate: { ...state.lastUpdate, strategy: Date.now() },
      };
    case 'SET_STRATEGY_LOADING':
      return { ...state, loading: { ...state.loading, strategy: action.payload } };
    
    // Voting
    case 'SET_VOTING_DATA':
      return {
        ...state,
        votingData: action.payload,
        loading: { ...state.loading, voting: false },
        loaded: { ...state.loaded, voting: true },
        lastUpdate: { ...state.lastUpdate, voting: Date.now() },
      };
    case 'SET_VOTING_LOADING':
      return { ...state, loading: { ...state.loading, voting: action.payload } };
    
    // Logs
    case 'SET_LOGS':
      return {
        ...state,
        logs: { ...state.logs, [action.payload.category]: action.payload.data },
        loading: { ...state.loading, logs: false },
        loaded: { ...state.loaded, logs: true },
      };
    case 'SET_LOGS_LOADING':
      return { ...state, loading: { ...state.loading, logs: action.payload } };
    
    // Checkpoints
    case 'SET_CHECKPOINTS':
      return { ...state, checkpoints: action.payload };
    
    // System state (from WS)
    case 'SET_SYSTEM_STATE': {
      if (state.systemState && JSON.stringify(state.systemState) === JSON.stringify(action.payload)) {
        return state;
      }
      return {
        ...state,
        systemState: action.payload,
        lastUpdate: { ...state.lastUpdate, system: Date.now() },
      };
    }
    
    // Mark loaded
    case 'MARK_LOADED':
      return { ...state, loaded: { ...state.loaded, [action.payload]: true } };
    
    // Update timestamp
    case 'UPDATE_TIMESTAMP':
      return { ...state, lastUpdate: { ...state.lastUpdate, [action.payload.key]: action.payload.time } };
    
    default:
      return state;
  }
}

export const DataActions = {
  setAnalyticsData: (data) => ({ type: 'SET_ANALYTICS_DATA', payload: data }),
  setAnalyticsLoading: (val) => ({ type: 'SET_ANALYTICS_LOADING', payload: val }),
  setMemoryData: (data) => ({ type: 'SET_MEMORY_DATA', payload: data }),
  setMemoryLoading: (val) => ({ type: 'SET_MEMORY_LOADING', payload: val }),
  setRiskData: (data) => ({ type: 'SET_RISK_DATA', payload: data }),
  setRiskLoading: (val) => ({ type: 'SET_RISK_LOADING', payload: val }),
  setStrategyData: (data) => ({ type: 'SET_STRATEGY_DATA', payload: data }),
  setStrategyLoading: (val) => ({ type: 'SET_STRATEGY_LOADING', payload: val }),
  setVotingData: (data) => ({ type: 'SET_VOTING_DATA', payload: data }),
  setVotingLoading: (val) => ({ type: 'SET_VOTING_LOADING', payload: val }),
  setLogs: (category, data) => ({ type: 'SET_LOGS', payload: { category, data } }),
  setLogsLoading: (val) => ({ type: 'SET_LOGS_LOADING', payload: val }),
  setCheckpoints: (data) => ({ type: 'SET_CHECKPOINTS', payload: data }),
  setSystemState: (data) => ({ type: 'SET_SYSTEM_STATE', payload: data }),
  markLoaded: (key) => ({ type: 'MARK_LOADED', payload: key }),
  updateTimestamp: (key, time = Date.now()) => ({ type: 'UPDATE_TIMESTAMP', payload: { key, time } }),
};

// ═══════════════════════════════════════════════════════════════════════════
// HOOKS - Domain-specific state and dispatch accessors
// ═══════════════════════════════════════════════════════════════════════════

export function useUIState() {
  const ctx = useContext(UIStateContext);
  if (!ctx) throw new Error('useUIState must be used within StoreProvider');
  return ctx;
}

export function useUIDispatch() {
  const ctx = useContext(UIDispatchContext);
  if (!ctx) throw new Error('useUIDispatch must be used within StoreProvider');
  return ctx;
}

export function useTradingState() {
  const ctx = useContext(TradingStateContext);
  if (!ctx) throw new Error('useTradingState must be used within StoreProvider');
  return ctx;
}

export function useTradingDispatch() {
  const ctx = useContext(TradingDispatchContext);
  if (!ctx) throw new Error('useTradingDispatch must be used within StoreProvider');
  return ctx;
}

export function useModulesState() {
  const ctx = useContext(ModulesStateContext);
  if (!ctx) throw new Error('useModulesState must be used within StoreProvider');
  return ctx;
}

export function useModulesDispatch() {
  const ctx = useContext(ModulesDispatchContext);
  if (!ctx) throw new Error('useModulesDispatch must be used within StoreProvider');
  return ctx;
}

export function useDataState() {
  const ctx = useContext(DataStateContext);
  if (!ctx) throw new Error('useDataState must be used within StoreProvider');
  return ctx;
}

export function useDataDispatch() {
  const ctx = useContext(DataDispatchContext);
  if (!ctx) throw new Error('useDataDispatch must be used within StoreProvider');
  return ctx;
}

// ═══════════════════════════════════════════════════════════════════════════
// SELECTOR HOOKS - Fine-grained subscriptions
// ═══════════════════════════════════════════════════════════════════════════

// Store ref for external subscription (used by useSelector)
const storeRef = { current: null };

export function useSelector(selector) {
  const store = storeRef.current;
  if (!store) throw new Error('useSelector must be used within StoreProvider');
  
  const getSnapshot = useCallback(() => {
    const { ui, trading, modules, data } = store.getState();
    return selector({ ui, trading, modules, data });
  }, [selector, store]);
  
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot);
}

export function useShallowSelector(selector) {
  const store = storeRef.current;
  if (!store) throw new Error('useShallowSelector must be used within StoreProvider');
  
  const prevRef = useRef();
  
  const getSnapshot = useCallback(() => {
    const { ui, trading, modules, data } = store.getState();
    const next = selector({ ui, trading, modules, data });
    
    if (shallowEqual(prevRef.current, next)) {
      return prevRef.current;
    }
    prevRef.current = next;
    return next;
  }, [selector, store]);
  
  return useSyncExternalStore(store.subscribe, getSnapshot, getSnapshot);
}

// ═══════════════════════════════════════════════════════════════════════════
// PROVIDER - Combines all domain contexts
// ═══════════════════════════════════════════════════════════════════════════

export function StoreProvider({ children }) {
  const [uiState, uiDispatch] = useReducer(uiReducer, initialUIState);
  const [tradingState, tradingDispatch] = useReducer(tradingReducer, initialTradingState);
  const [modulesState, modulesDispatch] = useReducer(modulesReducer, initialModulesState);
  const [dataState, dataDispatch] = useReducer(dataReducer, initialDataState);
  
  // Subscribers for useSelector
  const subscribersRef = useRef(new Set());
  
  // Memoize state objects to prevent unnecessary re-renders
  const uiValue = useMemo(() => uiState, [uiState]);
  const tradingValue = useMemo(() => tradingState, [tradingState]);
  const modulesValue = useMemo(() => modulesState, [modulesState]);
  const dataValue = useMemo(() => dataState, [dataState]);
  
  // Store ref for selectors
  const storeValue = useMemo(() => ({
    getState: () => ({
      ui: uiState,
      trading: tradingState,
      modules: modulesState,
      data: dataState,
    }),
    subscribe: (callback) => {
      subscribersRef.current.add(callback);
      return () => subscribersRef.current.delete(callback);
    },
  }), [uiState, tradingState, modulesState, dataState]);
  
  storeRef.current = storeValue;
  
  // Notify subscribers on state change
  const prevStateRef = useRef();
  useMemo(() => {
    const currentState = { ui: uiState, trading: tradingState, modules: modulesState, data: dataState };
    if (prevStateRef.current && prevStateRef.current !== currentState) {
      subscribersRef.current.forEach(cb => cb());
    }
    prevStateRef.current = currentState;
  }, [uiState, tradingState, modulesState, dataState]);
  
  return (
    <UIStateContext.Provider value={uiValue}>
      <UIDispatchContext.Provider value={uiDispatch}>
        <TradingStateContext.Provider value={tradingValue}>
          <TradingDispatchContext.Provider value={tradingDispatch}>
            <ModulesStateContext.Provider value={modulesValue}>
              <ModulesDispatchContext.Provider value={modulesDispatch}>
                <DataStateContext.Provider value={dataValue}>
                  <DataDispatchContext.Provider value={dataDispatch}>
                    {children}
                  </DataDispatchContext.Provider>
                </DataStateContext.Provider>
              </ModulesDispatchContext.Provider>
            </ModulesStateContext.Provider>
          </TradingDispatchContext.Provider>
        </TradingStateContext.Provider>
      </UIDispatchContext.Provider>
    </UIStateContext.Provider>
  );
}

export default StoreProvider;
