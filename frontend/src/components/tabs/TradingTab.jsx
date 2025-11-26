import React, { useState } from 'react';
import { createPortal } from 'react-dom';
import {
  Play, Pause, AlertTriangle, TrendingUp, Shield,
  Loader2, X, RefreshCw
} from 'lucide-react';
import { useAppState } from '../shared';

const API_BASE = '/api';

// ═══════════════════════════════════════════════════════════════════
// TRADING TAB - Live Trading Control Panel
// Includes position management with SL/TP risk warnings
// ═══════════════════════════════════════════════════════════════════

const TradingTab = React.memo(function TradingTab({
  systemStatus,
  startTrading,
  stopTrading,
  emergencyStop,
  positions = [],
  account = null
}) {
  const [fixingSlTp, setFixingSlTp] = useState(false);
  
  // Count positions without SL/TP
  const missingSL = positions.filter(p => !p.sl || p.sl <= 0).length;
  const missingTP = positions.filter(p => !p.tp || p.tp <= 0).length;
  const hasRiskWarning = missingSL > 0 || missingTP > 0;

  const handleFixSlTp = async () => {
    setFixingSlTp(true);
    try {
      const response = await fetch(`${API_BASE}/mt5/positions/fix-sl-tp`, { method: 'POST' });
      const data = await response.json();
      if (data.success) {
        alert(`Fixed ${data.fixed} of ${data.total} positions`);
      } else {
        alert(`Error: ${data.error}`);
      }
    } catch (error) {
      alert(`Failed to fix SL/TP: ${error.message}`);
    } finally {
      setFixingSlTp(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Risk Warning Banner */}
      {hasRiskWarning && positions.length > 0 && (
        <div className="bg-gradient-to-r from-red-900/50 to-orange-900/50 rounded-xl p-4 border border-red-500/50 animate-pulse">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-red-500/20 rounded-lg">
                <AlertTriangle className="w-6 h-6 text-red-400" />
              </div>
              <div>
                <h3 className="text-red-400 font-bold text-lg">⚠️ Risk Protection Warning</h3>
                <p className="text-red-300 text-sm">
                  {missingSL > 0 && `${missingSL} position(s) without Stop Loss`}
                  {missingSL > 0 && missingTP > 0 && ' • '}
                  {missingTP > 0 && `${missingTP} position(s) without Take Profit`}
                </p>
                <p className="text-gray-400 text-xs mt-1">
                  Positions without SL/TP are unprotected if network disconnects!
                </p>
              </div>
            </div>
            <button
              onClick={handleFixSlTp}
              disabled={fixingSlTp}
              className="flex items-center space-x-2 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded-lg transition-colors text-white font-medium"
            >
              {fixingSlTp ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Fixing...</span>
                </>
              ) : (
                <>
                  <Shield className="w-4 h-4" />
                  <span>Auto-Fix SL/TP</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white">Live Trading Control</h2>
          <div className={`px-3 py-1 rounded-full text-sm font-medium ${
            systemStatus === 'TRADING' ? 'bg-green-500/20 text-green-400' :
            systemStatus === 'STOPPING' ? 'bg-yellow-500/20 text-yellow-400' :
            systemStatus === 'EMERGENCY_STOPPED' ? 'bg-red-500/20 text-red-400' :
            'bg-gray-500/20 text-gray-400'
          }`}>
            {systemStatus === 'TRADING' ? 'Active' : 
             systemStatus === 'STOPPING' ? 'Stopping...' : 
             systemStatus === 'EMERGENCY_STOPPED' ? 'Emergency Stopped' : 
             'Idle'}
          </div>
        </div>

        <div className="flex items-center space-x-4">
          {(systemStatus === 'IDLE' || systemStatus === 'EMERGENCY_STOPPED') ? (
            <button
              onClick={startTrading}
              className="flex items-center space-x-2 bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 px-6 py-3 rounded-lg transition-all duration-200 text-white font-medium"
            >
              <Play className="w-5 h-5" />
              <span>Start Trading</span>
            </button>
          ) : systemStatus === 'TRADING' ? (
            <>
              <button
                onClick={stopTrading}
                className="flex items-center space-x-2 bg-gradient-to-r from-yellow-600 to-yellow-700 hover:from-yellow-700 hover:to-yellow-800 px-6 py-3 rounded-lg transition-all duration-200 text-white font-medium"
              >
                <Pause className="w-5 h-5" />
                <span>Stop Trading</span>
              </button>
              <button
                onClick={emergencyStop}
                className="flex items-center space-x-2 bg-gradient-to-r from-red-600 to-red-700 hover:from-red-700 hover:to-red-800 px-6 py-3 rounded-lg transition-all duration-200 text-white font-medium"
              >
                <AlertTriangle className="w-5 h-5" />
                <span>Emergency Stop</span>
              </button>
            </>
          ) : (
            <div className="flex items-center space-x-2 text-yellow-400">
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Processing...</span>
            </div>
          )}
        </div>
      </div>

      {/* Account Summary */}
      {account && (
        <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
          <h3 className="text-lg font-semibold text-white mb-4">Account Summary</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="bg-gray-900/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Balance</div>
              <div className="text-lg font-bold text-white">${account.balance?.toLocaleString()}</div>
            </div>
            <div className="bg-gray-900/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Equity</div>
              <div className="text-lg font-bold text-white">${account.equity?.toLocaleString()}</div>
            </div>
            <div className="bg-gray-900/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Floating P/L</div>
              <div className={`text-lg font-bold ${account.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                ${account.profit?.toFixed(2)}
              </div>
            </div>
            <div className="bg-gray-900/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Margin Used</div>
              <div className="text-lg font-bold text-yellow-400">${account.margin?.toLocaleString()}</div>
            </div>
            <div className="bg-gray-900/50 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Free Margin</div>
              <div className="text-lg font-bold text-blue-400">${account.margin_free?.toLocaleString()}</div>
            </div>
          </div>
        </div>
      )}

      {/* Open Positions */}
      <div className="bg-gray-800 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center space-x-3">
            <h3 className="text-lg font-semibold text-white">Open Positions ({positions.length})</h3>
            {hasRiskWarning && (
              <span className="px-2 py-1 bg-red-500/20 text-red-400 text-xs rounded-full animate-pulse">
                ⚠️ Missing SL/TP
              </span>
            )}
          </div>
          <div className="flex items-center space-x-3">
            {positions.length > 0 && (
              <button
                onClick={handleFixSlTp}
                disabled={fixingSlTp || !hasRiskWarning}
                className={`flex items-center space-x-1 px-3 py-1 rounded text-xs transition-colors ${
                  hasRiskWarning 
                    ? 'bg-orange-600 hover:bg-orange-700 text-white' 
                    : 'bg-gray-700 text-gray-400 cursor-not-allowed'
                }`}
              >
                <Shield size={12} />
                <span>Fix SL/TP</span>
              </button>
            )}
            <div className="text-sm text-gray-400">Live MT5 Data</div>
          </div>
        </div>

        {positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="text-gray-400 text-xs uppercase border-b border-gray-700">
                  <th className="px-4 py-3 text-left">Ticket</th>
                  <th className="px-4 py-3 text-left">Symbol</th>
                  <th className="px-4 py-3 text-left">Type</th>
                  <th className="px-4 py-3 text-right">Volume</th>
                  <th className="px-4 py-3 text-right">Open Price</th>
                  <th className="px-4 py-3 text-right">Current</th>
                  <th className="px-4 py-3 text-right">P/L</th>
                  <th className="px-4 py-3 text-right">SL</th>
                  <th className="px-4 py-3 text-right">TP</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((pos) => {
                  const hasSL = pos.sl && pos.sl > 0;
                  const hasTP = pos.tp && pos.tp > 0;
                  return (
                    <tr key={pos.ticket} className={`border-b border-gray-700/50 hover:bg-gray-700/30 ${
                      (!hasSL || !hasTP) ? 'bg-red-900/10' : ''
                    }`}>
                      <td className="px-4 py-3 text-gray-300 font-mono text-sm">{pos.ticket}</td>
                      <td className="px-4 py-3 text-white font-medium">{pos.symbol}</td>
                      <td className="px-4 py-3">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          pos.type === 'BUY' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                        }`}>
                          {pos.type}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-gray-300">{pos.volume}</td>
                      <td className="px-4 py-3 text-right text-gray-300 font-mono">{pos.price_open?.toFixed(5)}</td>
                      <td className="px-4 py-3 text-right text-white font-mono">{pos.price_current?.toFixed(5)}</td>
                      <td className={`px-4 py-3 text-right font-bold ${pos.profit >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                        ${pos.profit?.toFixed(2)}
                      </td>
                      <td className={`px-4 py-3 text-right font-mono text-sm ${
                        hasSL ? 'text-gray-400' : 'text-red-400 font-bold'
                      }`}>
                        {hasSL ? pos.sl.toFixed(5) : '⚠️ NONE'}
                      </td>
                      <td className={`px-4 py-3 text-right font-mono text-sm ${
                        hasTP ? 'text-gray-400' : 'text-orange-400 font-bold'
                      }`}>
                        {hasTP ? pos.tp.toFixed(5) : '⚠️ NONE'}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-gray-400">
            <TrendingUp className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No open positions</p>
            <p className="text-sm mt-1">Positions will appear here when trades are opened</p>
          </div>
        )}
      </div>
    </div>
  );
});

// ═══════════════════════════════════════════════════════════════════
// LOGS TAB - System Logs Viewer
// ═══════════════════════════════════════════════════════════════════

const LogsTab = React.memo(function LogsTab({ logs, fetchLogs }) {
  const { state: appState, dispatch: appDispatch } = useAppState();
  const selectedLogCategory = appState.selectedLogCategory;

  const logCategories = ['system', 'risk', 'strategy', 'position', 'trading'];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">System Logs</h2>
        <button
          onClick={() => fetchLogs(selectedLogCategory)}
          className="flex items-center space-x-2 px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-sm transition-colors text-white"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      <div className="flex flex-wrap gap-2 mb-4">
        {logCategories.map(cat => (
          <button
            key={cat}
            onClick={() => {
              appDispatch({ type: 'SET_LOG_CATEGORY', payload: cat });
              fetchLogs(cat);
            }}
            className={`px-4 py-2 rounded-lg transition-colors font-medium ${
              selectedLogCategory === cat
                ? 'bg-gradient-to-r from-blue-600 to-purple-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>

      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <div className="p-4 border-b border-gray-700 flex justify-between items-center bg-gray-900">
          <h3 className="text-lg font-semibold text-white">
            {selectedLogCategory.charAt(0).toUpperCase() + selectedLogCategory.slice(1)} Logs
          </h3>
          <div className="text-sm text-gray-400">
            {logs[selectedLogCategory]?.content?.length || 0} lines
          </div>
        </div>
        <div className="bg-black max-h-96 overflow-y-auto">
          <pre className="text-green-400 text-sm font-mono whitespace-pre-wrap p-4 leading-relaxed">
            {logs[selectedLogCategory]?.content
              ? logs[selectedLogCategory].content.join('')
              : 'Loading logs...'}
          </pre>
        </div>
      </div>
    </div>
  );
});

// ═══════════════════════════════════════════════════════════════════
// ALERTS MODAL - System Alerts Popup
// ═══════════════════════════════════════════════════════════════════

const AlertsModal = React.memo(function AlertsModal({
  showAlerts,
  alerts,
  alertFilter,
  onClose,
  setFilter,
  onClearAll,
  onDismissAlert
}) {
  if (!showAlerts) return null;

  const filteredAlerts = alerts.filter(alert =>
    alertFilter === 'all' || alert.severity === alertFilter
  );

  return createPortal(
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center p-4 z-50">
      <div className="bg-gray-800 rounded-xl max-w-4xl w-full max-h-[80vh] overflow-hidden">
        <div className="p-6 border-b border-gray-700 flex items-center justify-between">
          <h3 className="text-xl font-bold text-white">System Alerts ({alerts.length})</h3>
          <div className="flex items-center space-x-4">
            <button
              onClick={onClearAll}
              className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 px-3 py-1 rounded-lg text-white text-sm transition-colors"
              disabled={alerts.length === 0}
            >
              <X className="w-4 h-4" />
              <span>Clear All</span>
            </button>
            <select
              value={alertFilter}
              onChange={(e) => setFilter(e.target.value)}
              className="bg-gray-700 border border-gray-600 rounded-lg px-3 py-1 text-white text-sm"
            >
              <option value="all">All Alerts</option>
              <option value="critical">Critical</option>
              <option value="warning">Warning</option>
              <option value="success">Success</option>
              <option value="info">Info</option>
            </select>
            <button onClick={onClose} className="text-gray-400 hover:text-white">
              <X className="w-6 h-6" />
            </button>
          </div>
        </div>

        <div className="p-6 max-h-96 overflow-y-auto">
          {filteredAlerts.length > 0 ? (
            <div className="space-y-3">
              {filteredAlerts.map((alert, idx) => (
                <div
                  key={(alert.timestamp ?? alert.time) + idx}
                  className={`p-4 rounded-lg border-l-4 ${
                    alert.severity === 'critical' ? 'bg-red-900/20 border-red-500' :
                    alert.severity === 'warning' ? 'bg-yellow-900/20 border-yellow-500' :
                    alert.severity === 'success' ? 'bg-green-900/20 border-green-500' :
                    'bg-blue-900/20 border-blue-500'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className={`text-xs px-2 py-1 rounded font-medium ${
                          alert.severity === 'critical' ? 'bg-red-600 text-white' :
                          alert.severity === 'warning' ? 'bg-yellow-600 text-white' :
                          alert.severity === 'success' ? 'bg-green-600 text-white' :
                          'bg-blue-600 text-white'
                        }`}>
                          {alert.severity?.toUpperCase?.() || 'INFO'}
                        </span>
                        <span className="text-xs text-gray-400">{alert.module}</span>
                      </div>
                      <p className="text-white">{alert.alert?.message || alert.alert || '—'}</p>
                    </div>
                    <div className="flex items-center space-x-3 ml-4">
                      <div className="text-xs text-gray-400">
                        {new Date(alert.timestamp ?? alert.time ?? Date.now()).toLocaleTimeString()}
                      </div>
                      <button
                        onClick={() => onDismissAlert(alert)}
                        className="text-gray-400 hover:text-red-400 transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-gray-400 py-8">
              No alerts found for the selected filter.
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body
  );
});

export { TradingTab, LogsTab, AlertsModal };
export default TradingTab;
