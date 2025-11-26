// Backward compatibility layer for useAppState
// This provides the same API as the old AppStateContext for gradual migration

import { useContext, useMemo, useCallback } from 'react';
import {
  useUIState,
  useUIDispatch,
  useTradingState,
  useTradingDispatch,
  useModulesState,
  useModulesDispatch,
  useDataState,
  useDataDispatch,
} from './StoreProvider';

/**
 * Backward-compatible hook that provides the same interface as the old useAppState.
 * This combines all domain states into a single object for components that haven't
 * been migrated yet.
 * 
 * NOTE: For new components, prefer using domain-specific hooks:
 * - useUIState() / useUIDispatch() for UI state
 * - useTradingState() / useTradingDispatch() for trading state
 * - useModulesState() / useModulesDispatch() for modules state
 * - useDataState() / useDataDispatch() for data/analytics state
 */
export function useAppState() {
  const ui = useUIState();
  const trading = useTradingState();
  const modules = useModulesState();
  const data = useDataState();
  
  const uiDispatch = useUIDispatch();
  const tradingDispatch = useTradingDispatch();
  const modulesDispatch = useModulesDispatch();
  const dataDispatch = useDataDispatch();
  
  // Combine all states into the legacy format
  const state = useMemo(() => ({
    // UI State
    activeTab: ui.activeTab,
    selectedTimeframe: ui.selectedTimeframe,
    selectedView: ui.selectedView,
    selectedSymbol: ui.selectedSymbol,
    moduleSearch: ui.moduleSearch,
    moduleFilter: ui.moduleFilter,
    moduleViewMode: ui.moduleViewMode,
    selectedLogCategory: ui.selectedLogCategory,
    alertFilter: ui.alertFilter,
    showAlerts: ui.showAlerts,
    
    // Trading State
    isLoggedIn: trading.isLoggedIn,
    systemStatus: trading.systemStatus,
    wsConnected: trading.wsConnected,
    error: trading.error,
    accountInfo: trading.accountInfo,
    positions: trading.positions,
    positionCount: trading.positionCount,
    recentTrades: trading.recentTrades,
    mt5ChartData: trading.mt5ChartData,
    mt5Symbols: trading.mt5Symbols,
    performance: trading.performance,
    systemMetrics: trading.systemMetrics,
    alerts: trading.alerts,
    tradingConfig: trading.tradingConfig,
    
    // Modules State
    modules: modules.modules,
    modulesById: modules.modulesById,
    moduleCategories: modules.categories,
    moduleStats: modules.stats,
    moduleStates: modules.moduleStates,
    selectedModuleId: modules.selectedModuleId,
    
    // Data State
    analyticsData: data.analyticsData,
    memoryData: data.memoryData,
    riskData: data.riskData,
    strategyData: data.strategyData,
    votingData: data.votingData,
    logs: data.logs,
    checkpoints: data.checkpoints,
    systemState: data.systemState,
    
    // Loading states
    analyticsLoading: data.loading.analytics,
    memoryLoading: data.loading.memory,
    riskLoading: data.loading.risk,
    strategyLoading: data.loading.strategy,
    votingLoading: data.loading.voting,
    
    // Loaded flags (legacy format)
    dataLoaded: {
      analytics: data.loaded.analytics,
      memory: data.loaded.memory,
      risk: data.loaded.risk,
      strategy: data.loaded.strategy,
      voting: data.loaded.voting,
      logs: data.loaded.logs,
      modules: data.loaded.modules,
      mt5Data: data.loaded.mt5Data,
    },
    
    // Last update timestamps
    lastUpdate: data.lastUpdate,
    
    // MT5 data aliases
    mt5Positions: trading.positions,
    mt5Account: trading.accountInfo,
  }), [ui, trading, modules, data]);
  
  // Unified dispatch that routes to appropriate domain dispatcher
  const dispatch = useCallback((action) => {
    const uiActions = [
      'SET_ACTIVE_TAB', 'SET_TIMEFRAME', 'SET_VIEW', 'SET_SYMBOL',
      'SET_MODULE_SEARCH', 'SET_MODULE_FILTER', 'SET_MODULE_VIEW_MODE',
      'SET_LOG_CATEGORY', 'SET_ALERT_FILTER', 'SET_SHOW_ALERTS'
    ];
    
    const tradingActions = [
      'SET_LOGGED_IN', 'SET_SYSTEM_STATUS', 'SET_WS_CONNECTED', 'SET_ERROR',
      'SET_ACCOUNT_INFO', 'SET_POSITIONS', 'SET_RECENT_TRADES',
      'SET_MT5_CHART_DATA', 'SET_MT5_SYMBOLS', 'SET_PERFORMANCE',
      'SET_SYSTEM_METRICS', 'SET_ALERTS', 'DISMISS_ALERT', 'SET_TRADING_CONFIG',
      'BATCH_UPDATE', 'LOGOUT'
    ];
    
    const modulesActions = [
      'SET_MODULES', 'SET_CATEGORIES', 'SET_MODULE_STATS',
      'SET_MODULE_STATES', 'UPDATE_MODULE_STATE', 'SET_SELECTED_MODULE'
    ];
    
    // Legacy action name mapping
    const legacyMapping = {
      'SET_MT5_DATA': (payload) => {
        if (payload.chartData) tradingDispatch({ type: 'SET_MT5_CHART_DATA', payload: payload.chartData });
        if (payload.recentTrades) tradingDispatch({ type: 'SET_RECENT_TRADES', payload: payload.recentTrades });
        if (payload.symbols) tradingDispatch({ type: 'SET_MT5_SYMBOLS', payload: payload.symbols });
        if (payload.positions) tradingDispatch({ type: 'SET_POSITIONS', payload: payload.positions });
        if (payload.account) tradingDispatch({ type: 'SET_ACCOUNT_INFO', payload: payload.account });
        dataDispatch({ type: 'MARK_LOADED', payload: 'mt5Data' });
      },
      'MARK_DATA_LOADED': (payload) => {
        dataDispatch({ type: 'MARK_LOADED', payload });
      },
    };
    
    // Handle legacy actions
    if (legacyMapping[action.type]) {
      legacyMapping[action.type](action.payload);
      return;
    }
    
    // Route to appropriate dispatcher
    if (uiActions.includes(action.type)) {
      uiDispatch(action);
    } else if (tradingActions.includes(action.type)) {
      tradingDispatch(action);
    } else if (modulesActions.includes(action.type)) {
      modulesDispatch(action);
    } else {
      // Default to data dispatch for unknown actions
      dataDispatch(action);
    }
  }, [uiDispatch, tradingDispatch, modulesDispatch, dataDispatch]);
  
  return { state, dispatch };
}

export default useAppState;
