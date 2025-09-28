# Risk System Integration

This document describes the newly integrated Risk System tab in the AI Trading Dashboard.

## Overview

The Risk System provides comprehensive monitoring and visualization of all risk management components used by the AI trading system. This includes anomaly detection, compliance monitoring, drawdown rescue, execution quality, portfolio risk, and dynamic risk control.

## Features

### 1. Risk Overview
- **Risk Level**: Current system risk assessment (NORMAL/ELEVATED/CRITICAL/EMERGENCY)
- **Current Drawdown**: Real-time drawdown percentage with visual indicator
- **Risk Scale**: Position sizing multiplier showing current risk scaling
- **Win Rate**: Trading success rate with progress visualization

### 2. Component Tabs

#### Anomaly Detection
- Real-time anomaly score with threshold monitoring
- Visual anomaly detection gauge (green/red status)
- Detection mode status and system health metrics
- Recent anomaly alerts with severity indicators
- Historical anomaly pattern tracking

#### Compliance Monitoring
- Trade compliance status and violation tracking
- Position limits and current exposure monitoring
- Daily trading limits with utilization percentages
- Leverage compliance and risk limit enforcement
- Real-time compliance violation alerts with details

#### Drawdown Management
- Current vs maximum drawdown comparison
- Drawdown rescue system status (active/inactive)
- Recovery progress analysis and velocity tracking
- Rescue trigger history and intervention points
- Visual drawdown history charts

#### Execution Quality
- Execution vote recommendation (PROCEED/CAUTION/HALT)
- Quality score percentage with visual gauge
- Fill rate analysis and slippage monitoring
- Latency metrics and execution alerts
- Real-time execution performance tracking

#### Portfolio Risk
- Total portfolio exposure and VAR analysis
- Correlation risk scoring and matrix analysis
- Diversification metrics and asset class distribution
- Position risk monitoring and concentration alerts
- Risk attribution and exposure analysis

#### Dynamic Risk Control
- Current control mode (NORMAL/PROTECTIVE/EMERGENCY)
- Risk scaling multiplier with adjustment history
- Volatility analysis and system state monitoring
- Recent risk adjustments with reasoning
- Freeze counter and system protection status

#### Risk Alerts
- Consolidated alerts from all risk modules
- Alert categorization by source and severity
- Critical/Warning/Info alert counts
- Real-time alert feed with timestamps
- Alert filtering and management

## API Endpoints

The Risk System exposes the following REST API endpoints:

### GET /api/risk/overview
Returns comprehensive risk system overview:
```json
{
  "success": true,
  "risk_level": "ELEVATED",
  "current_drawdown": 0.12,
  "max_drawdown": 0.18,
  "risk_scale": 0.85,
  "win_rate": 0.68,
  "sharpe_ratio": 1.45,
  "var_95": 0.05,
  "var_99": 0.08,
  "system_status": "monitoring",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### GET /api/risk/anomalies
Returns anomaly detection data and alerts:
```json
{
  "success": true,
  "anomalies": {
    "anomaly_score": 0.75,
    "anomaly_threshold": 0.8,
    "detection_mode": "ACTIVE",
    "anomaly_alerts": [...],
    "system_health": {...}
  }
}
```

### GET /api/risk/compliance
Returns compliance monitoring data and violations

### GET /api/risk/drawdown
Returns drawdown monitoring and rescue system data

### GET /api/risk/execution
Returns execution quality monitoring data

### GET /api/risk/portfolio
Returns portfolio risk system data and correlation analysis

### GET /api/risk/dynamic
Returns dynamic risk controller data and scaling metrics

### GET /api/risk/alerts
Returns consolidated alerts from all risk modules

## Data Sources

The Risk System integrates with multiple InfoBus modules to retrieve real-time data:

- **AnomalyDetector**: Anomaly detection and system health
- **Compliance**: Trade compliance and violation monitoring
- **DrawdownRescue**: Drawdown monitoring and rescue triggers
- **ExecutionQualityMonitor**: Execution quality and performance
- **PortfolioRiskSystem**: Portfolio risk and correlation analysis
- **DynamicRiskController**: Dynamic risk scaling and control
- **BackendAPI**: General risk metrics and system state

## Usage

1. **Navigate to Risk Tab**: Click on the "Risk" tab in the main navigation sidebar
2. **View Overview**: The default view shows key risk metrics and system status
3. **Explore Components**: Use the component tabs to dive into specific risk subsystems
4. **Monitor Alerts**: Check the Alerts tab for consolidated risk notifications
5. **Refresh Data**: Click the Refresh button to update all data immediately

## Auto-Refresh

The Risk tab automatically refreshes data every 10 seconds to provide real-time monitoring. The refresh interval is throttled to prevent excessive API calls.

## Visual Elements

- **Color-Coded Risk Levels**: Green (NORMAL), Yellow (ELEVATED), Red (CRITICAL/EMERGENCY)
- **Progress Bars**: Show utilization percentages for various risk metrics
- **Status Indicators**: Real-time status badges for different risk components
- **Charts**: Bar charts for risk metrics, line charts for historical data
- **Alert Cards**: Color-coded alerts based on severity levels
- **Interactive Tabs**: Easy navigation between component views

## Risk Level Color Coding

The system uses consistent color coding throughout:

- **Green**: Normal/Low risk, healthy status
- **Yellow**: Elevated/Warning risk, caution advised
- **Red**: High/Critical risk, immediate attention required
- **Dark Red**: Emergency risk, system protection active

## Integration Notes

The Risk System is fully integrated with the existing dashboard architecture:

- Uses consistent styling and component patterns
- Follows established error handling and loading states
- Integrates with InfoBus for real-time data access
- Maintains consistency with overall user experience
- Responsive design for all screen sizes

## Risk Module Components

### Core Risk Modules
1. **Anomaly Detector**: Pattern recognition and anomaly detection
2. **Compliance Module**: Regulatory and risk limit enforcement
3. **Drawdown Rescue**: Automatic drawdown protection and recovery
4. **Execution Quality Monitor**: Trade execution performance tracking
5. **Portfolio Risk System**: Portfolio-level risk management
6. **Dynamic Risk Controller**: Adaptive risk scaling and control
7. **Correlated Risk Controller**: Cross-asset correlation monitoring
8. **Active Trade Monitor**: Real-time position monitoring

### Key Features per Module
- **Real-time monitoring** with configurable thresholds
- **Automatic alerts** and notifications
- **Historical tracking** and trend analysis
- **Intelligent adaptation** to market conditions
- **Multi-timeframe analysis** and decision making

## Testing

To test the Risk System integration:

1. Run the backend server: `python backend/main.py`
2. Ensure risk modules are active and publishing data to InfoBus
3. Use the test script: `python test_risk_integration.py`
4. Navigate to the Risk tab in the frontend
5. Verify all components load and display data correctly

## Troubleshooting

**No Data Displayed**:
- Ensure risk modules are enabled and running
- Check that InfoBus is publishing risk data
- Verify API endpoints return success responses

**Connection Errors**:
- Confirm backend server is running on port 8000
- Check for firewall or network issues
- Verify API endpoint URLs are correct

**Performance Issues**:
- Monitor API response times
- Check for memory leaks in the frontend
- Verify auto-refresh intervals are appropriate

**Alert Issues**:
- Check InfoBus alert publishing from risk modules
- Verify alert formatting and structure
- Ensure proper alert categorization and filtering

## Advanced Features

### Alert Management
- **Severity Filtering**: Filter alerts by critical, warning, info levels
- **Source Filtering**: Filter by specific risk module
- **Time-based Filtering**: View recent vs historical alerts
- **Alert Dismissal**: Mark alerts as read/acknowledged

### Data Export
- Export risk metrics for external analysis
- Historical data download capabilities
- Alert logs and audit trails
- Performance report generation

### Customization
- Configurable alert thresholds
- Custom risk level definitions
- Personalized dashboard layouts
- Module-specific settings and preferences

This Risk System integration provides comprehensive, real-time monitoring of all trading risks with professional-grade visualization and alerting capabilities.