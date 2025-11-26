// App State Context - re-exported from centralized store for backward compatibility
// New components should import directly from '../../store'

export { useAppState } from '../../store/compat';

// Legacy exports for backward compatibility with existing components
// These are no longer used but exported to prevent import errors during migration
export const AppStateContext = null;
export const initialAppState = {};
export const appReducer = (state) => state;
export const AppStateProvider = ({ children }) => children;

