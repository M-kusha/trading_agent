// Re-export all shared components
export {
  AppStateContext,
  useAppState,
  initialAppState,
  appReducer,
  AppStateProvider
} from './AppContext';

export {
  StatusIndicator,
  TabButton,
  MetricCard,
  AlertBadge,
  ProgressBar,
  ModeSelector,
  EnhancedModuleCard,
  getCategoryIcon,
  getCategoryColor,
  getStatusColor,
  getHealthColor,
  getHealthBarColor,
  toArray
} from './UIComponents';
