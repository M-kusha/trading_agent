// Tab Components Index
// All tab components extracted from App.jsx for maintainability

// Extracted Tabs
export { default as AnalyticsTab } from './AnalyticsTab';
export { default as MemoryTab } from './MemoryTab';
export { default as RiskTab } from './RiskTab';
export { default as StrategyTab } from './StrategyTab';
export { default as VotingTab } from './VotingTab';
export { TradingTab, LogsTab, AlertsModal } from './TradingTab';

// Note: OverviewTab and ModulesTab remain in App.jsx due to complex dependencies
// They rely on many props from the main component state and would require
// significant refactoring to extract cleanly
