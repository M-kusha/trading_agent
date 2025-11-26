// Optimized State Management System
// Split contexts to minimize re-renders - components only subscribe to what they need

export { 
  // Main providers
  StoreProvider,
  
  // Domain-specific hooks (use these for granular updates)
  useUIState,
  useUIDispatch,
  useTradingState,
  useTradingDispatch,
  useModulesState,
  useModulesDispatch,
  useDataState,
  useDataDispatch,
  
  // Selector hooks for fine-grained subscriptions
  useSelector,
  useShallowSelector,
  
  // Actions
  UIActions,
  TradingActions,
  ModulesActions,
  DataActions,
} from './StoreProvider';

// Data fetching hooks with caching
export {
  useAnalyticsData,
  useMemoryData,
  useRiskData,
  useStrategyData,
  useVotingData,
  useLogs,
  apiCall,
  clearApiCache,
} from './hooks';

// Re-export types for backward compatibility
export { useAppState } from './compat';

