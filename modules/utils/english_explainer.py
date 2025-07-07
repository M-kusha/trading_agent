# ─────────────────────────────────────────────────────────────
# File: modules/utils/english_explainer.py
# 🚀 Plain English explanation generator for SmartInfoBus
# ─────────────────────────────────────────────────────────────

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import json
import re
from dataclasses import dataclass

@dataclass
class ExplanationTemplate:
    """Template for generating explanations"""
    template: str
    required_fields: List[str]
    category: str

class EnglishExplainer:
    """
    Generates plain English explanations for all SmartInfoBus components.
    Ensures consistency in human-readable output across the system.
    """
    
    def __init__(self):
        self.templates = self._load_templates()
        
    def _load_templates(self) -> Dict[str, ExplanationTemplate]:
        """Load explanation templates"""
        return {
            'module_decision': ExplanationTemplate(
                template="""
{module_name} Decision Explanation
==================================
Decision: {decision}
Confidence: {confidence:.1%}

REASONING:
The module made this decision based on the following factors:
{reasoning_points}

{additional_context}

This decision was made because {primary_reason}.
""",
                required_fields=['module_name', 'decision', 'confidence', 'reasoning_points', 'primary_reason'],
                category='decision'
            ),
            
            'error_diagnosis': ExplanationTemplate(
                template="""
Error Diagnosis for {module_name}
=================================
What went wrong: {error_type} occurred in {method_name}
When: {timestamp}
Where: {file_path}:{line_number}

SIMPLE EXPLANATION:
{plain_english_explanation}

LIKELY CAUSE:
{likely_cause}

SUGGESTED FIX:
{suggested_fix}

TECHNICAL DETAILS:
{technical_details}
""",
                required_fields=['module_name', 'error_type', 'method_name', 'plain_english_explanation'],
                category='error'
            ),
            
            'performance_report': ExplanationTemplate(
                template="""
Performance Report: {module_name}
================================
Overall Status: {status_emoji} {status_text}
Report Period: {period}

SUMMARY:
{summary_text}

KEY METRICS:
{metrics_text}

{recommendations_section}

TRENDS:
{trends_text}
""",
                required_fields=['module_name', 'status_text', 'summary_text', 'metrics_text'],
                category='performance'
            ),
            
            'data_flow': ExplanationTemplate(
                template="""
Data Flow Analysis: {data_key}
==============================
Current Status: {status}

PROVIDERS:
{providers_text}

CONSUMERS:
{consumers_text}

DATA CHARACTERISTICS:
- Age: {age_text}
- Version: {version}
- Confidence: {confidence:.1%}
- Last Updated By: {source_module}

{explanation_text}
""",
                required_fields=['data_key', 'status', 'providers_text', 'consumers_text'],
                category='data_flow'
            ),
            
            'health_status': ExplanationTemplate(
                template="""
System Health Report
===================
Overall Health: {overall_status_emoji} {overall_status}
Generated: {timestamp}

SYSTEM RESOURCES:
{resource_status}

MODULE HEALTH:
{module_status}

{alert_section}

RECOMMENDATIONS:
{recommendations}
""",
                required_fields=['overall_status', 'resource_status', 'module_status'],
                category='health'
            ),
            
            'dependency_analysis': ExplanationTemplate(
                template="""
Dependency Analysis Report
=========================
Total Modules: {total_modules}
Total Dependencies: {total_dependencies}

{issues_section}

DEPENDENCY GRAPH:
{graph_description}

OPTIMIZATION OPPORTUNITIES:
{optimization_suggestions}

ACTION ITEMS:
{action_items}
""",
                required_fields=['total_modules', 'total_dependencies', 'graph_description'],
                category='dependency'
            ),
            
            'trade_decision': ExplanationTemplate(
                template="""
TRADE DECISION: {action} {size} {instrument}
============================================
Confidence: {confidence:.1%}
Risk Score: {risk_score:.2f}

REASONING:
{reasoning}

MARKET CONTEXT:
{market_context}

RISK ASSESSMENT:
{risk_assessment}

Expected Outcome: {expected_outcome}
""",
                required_fields=['action', 'size', 'instrument', 'confidence', 'reasoning'],
                category='trade'
            )
        }
    
    def explain_module_decision(self, module_name: str, decision: Any, 
                              context: Dict[str, Any], confidence: float) -> str:
        """Generate explanation for a module decision"""
        # Extract reasoning points
        reasoning_points = self._extract_reasoning_points(context)
        
        # Determine primary reason
        primary_reason = self._determine_primary_reason(decision, context)
        
        # Additional context
        additional_context = ""
        if 'market_regime' in context:
            additional_context = f"Market Regime: {context['market_regime']}"
        if 'risk_score' in context:
            additional_context += f"\nRisk Level: {self._describe_risk_level(context['risk_score'])}"
        
        return self._fill_template('module_decision', {
            'module_name': module_name,
            'decision': self._format_decision(decision),
            'confidence': confidence,
            'reasoning_points': reasoning_points,
            'primary_reason': primary_reason,
            'additional_context': additional_context
        })
    
    def explain_error(self, error_type: str, error_message: str, 
                     module_name: str, method_name: str, 
                     file_path: str, line_number: int,
                     local_variables: Dict[str, Any]) -> str:
        """Generate plain English error explanation"""
        plain_explanation = self._translate_error_to_plain_english(error_type, error_message)
        likely_cause = self._diagnose_likely_cause(error_type, error_message, local_variables)
        suggested_fix = self._suggest_fix(error_type, error_message)
        
        return self._fill_template('error_diagnosis', {
            'module_name': module_name,
            'error_type': error_type,
            'method_name': method_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'file_path': file_path,
            'line_number': line_number,
            'plain_english_explanation': plain_explanation,
            'likely_cause': likely_cause,
            'suggested_fix': suggested_fix,
            'technical_details': error_message[:200]
        })
    
    def explain_performance(self, module_name: str, metrics: Dict[str, float],
                          period: str = "Last 24 hours") -> str:
        """Generate performance explanation"""
        status = self._assess_performance_status(metrics)
        status_emoji, status_text = status
        
        summary_text = self._generate_performance_summary(metrics)
        metrics_text = self._format_metrics(metrics)
        recommendations = self._generate_performance_recommendations(metrics)
        trends_text = self._describe_trends(metrics)
        
        recommendations_section = ""
        if recommendations:
            recommendations_section = "RECOMMENDATIONS:\n" + "\n".join(f"• {r}" for r in recommendations)
        
        return self._fill_template('performance_report', {
            'module_name': module_name,
            'status_emoji': status_emoji,
            'status_text': status_text,
            'period': period,
            'summary_text': summary_text,
            'metrics_text': metrics_text,
            'recommendations_section': recommendations_section,
            'trends_text': trends_text
        })
    
    def explain_data_flow(self, data_key: str, providers: List[str], 
                         consumers: List[str], current_data: Optional[Dict[str, Any]]) -> str:
        """Explain data flow in plain English"""
        if not providers and not consumers:
            status = "Unused"
            explanation_text = f"The data key '{data_key}' is not currently used by any modules."
        elif not providers:
            status = "Missing Provider"
            explanation_text = f"Warning: Modules are trying to use '{data_key}' but no module provides it."
        elif not consumers:
            status = "No Consumers"
            explanation_text = f"The data '{data_key}' is being produced but not used by any modules."
        else:
            status = "Active"
            explanation_text = f"This data flows from {self._list_modules(providers)} to {self._list_modules(consumers)}."
        
        providers_text = self._format_module_list(providers, "No modules provide this data")
        consumers_text = self._format_module_list(consumers, "No modules consume this data")
        
        # Current data info
        if current_data:
            age_text = self._format_age(current_data.get('age_seconds', 0))
            version = current_data.get('version', 'Unknown')
            confidence = current_data.get('confidence', 0)
            source_module = current_data.get('source_module', 'Unknown')
        else:
            age_text = "No data available"
            version = "N/A"
            confidence = 0
            source_module = "None"
        
        return self._fill_template('data_flow', {
            'data_key': data_key,
            'status': status,
            'providers_text': providers_text,
            'consumers_text': consumers_text,
            'age_text': age_text,
            'version': version,
            'confidence': confidence,
            'source_module': source_module,
            'explanation_text': explanation_text
        })
    
    def explain_health_status(self, overall_status: str, system_metrics: Dict[str, float],
                            module_health: Dict[str, str], alerts: List[Dict[str, Any]],
                            recommendations: List[str]) -> str:
        """Generate health status explanation"""
        status_emoji = self._get_health_emoji(overall_status)
        
        # Format system resources
        resource_status = self._format_system_resources(system_metrics)
        
        # Format module status
        module_status = self._format_module_health(module_health)
        
        # Format alerts
        alert_section = ""
        if alerts:
            alert_section = "\nACTIVE ALERTS:\n" + self._format_alerts(alerts)
        
        # Format recommendations
        recommendations_text = "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations))
        if not recommendations_text:
            recommendations_text = "System is healthy - no immediate actions required."
        
        return self._fill_template('health_status', {
            'overall_status_emoji': status_emoji,
            'overall_status': overall_status.title(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'resource_status': resource_status,
            'module_status': module_status,
            'alert_section': alert_section,
            'recommendations': recommendations_text
        })
    
    def explain_dependencies(self, total_modules: int, total_dependencies: int,
                           issues: List[str], optimization_suggestions: List[str]) -> str:
        """Explain module dependencies"""
        # Format issues section
        issues_section = ""
        if issues:
            issues_section = "ISSUES FOUND:\n" + "\n".join(f"⚠️ {issue}" for issue in issues)
        else:
            issues_section = "✅ No dependency issues found"
        
        # Create graph description
        avg_deps = total_dependencies / max(total_modules, 1)
        graph_description = f"""
The system has {total_modules} modules with {total_dependencies} dependencies.
Average dependencies per module: {avg_deps:.1f}
"""
        
        # Format suggestions
        suggestions_text = "\n".join(f"• {sug}" for sug in optimization_suggestions)
        if not suggestions_text:
            suggestions_text = "• No optimization needed - dependencies are well structured"
        
        # Action items
        action_items = self._generate_dependency_action_items(issues, optimization_suggestions)
        
        return self._fill_template('dependency_analysis', {
            'total_modules': total_modules,
            'total_dependencies': total_dependencies,
            'issues_section': issues_section,
            'graph_description': graph_description.strip(),
            'optimization_suggestions': suggestions_text,
            'action_items': action_items
        })
    
    # ─────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────
    
    def _fill_template(self, template_name: str, values: Dict[str, Any]) -> str:
        """Fill a template with values"""
        if template_name not in self.templates:
            return f"Unknown template: {template_name}"
        
        template = self.templates[template_name]
        
        # Add default values for optional fields
        filled_values = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        filled_values.update(values)
        
        # Check required fields
        missing = [f for f in template.required_fields if f not in filled_values]
        if missing:
            filled_values.update({f: f"<missing: {f}>" for f in missing})
        
        try:
            return template.template.format(**filled_values).strip()
        except Exception as e:
            return f"Error formatting template: {e}"
    
    def _extract_reasoning_points(self, context: Dict[str, Any]) -> str:
        """Extract reasoning points from context"""
        points = []
        
        if 'market_data' in context:
            points.append("• Current market conditions analyzed")
        if 'risk_score' in context:
            risk_level = self._describe_risk_level(context['risk_score'])
            points.append(f"• Risk assessment: {risk_level}")
        if 'confidence_factors' in context:
            points.append(f"• {len(context['confidence_factors'])} confidence factors evaluated")
        if 'historical_performance' in context:
            points.append("• Historical performance patterns considered")
        
        return "\n".join(points) if points else "• Multiple factors analyzed"
    
    def _determine_primary_reason(self, decision: Any, context: Dict[str, Any]) -> str:
        """Determine the primary reason for a decision"""
        if isinstance(decision, dict):
            if 'reason' in decision:
                return decision['reason']
            if 'action' in decision:
                return f"the best action is to {decision['action']}"
        
        if 'risk_score' in context and context['risk_score'] > 0.7:
            return "risk levels are too high for aggressive action"
        
        return "the current conditions favor this approach"
    
    def _format_decision(self, decision: Any) -> str:
        """Format decision for display"""
        if isinstance(decision, dict):
            return json.dumps(decision, indent=2)
        return str(decision)
    
    def _describe_risk_level(self, risk_score: float) -> str:
        """Convert risk score to plain English"""
        if risk_score < 0.3:
            return "Low risk - conditions are favorable"
        elif risk_score < 0.5:
            return "Moderate risk - proceed with caution"
        elif risk_score < 0.7:
            return "Elevated risk - careful monitoring required"
        else:
            return "High risk - defensive positioning recommended"
    
    def _translate_error_to_plain_english(self, error_type: str, error_message: str) -> str:
        """Translate technical errors to plain English"""
        translations = {
            'KeyError': "The system tried to access data that doesn't exist",
            'TypeError': "The system received data in an unexpected format",
            'ValueError': "The system received invalid data values",
            'AttributeError': "The system tried to use a feature that doesn't exist",
            'TimeoutError': "The operation took too long to complete",
            'ConnectionError': "The system couldn't connect to a required service"
        }
        
        base = translations.get(error_type, "An unexpected error occurred")
        
        # Add specific details
        if 'NoneType' in error_message:
            return f"{base}. Specifically, expected data was missing (None/null)."
        
        return f"{base}. {error_message[:100]}"
    
    def _diagnose_likely_cause(self, error_type: str, error_message: str, 
                              local_variables: Dict[str, Any]) -> str:
        """Diagnose the likely cause of an error"""
        error_lower = error_message.lower()
        
        if "none" in error_lower or "nonetype" in error_lower:
            return "A required value was not initialized or was cleared unexpectedly"
        elif "key" in error_lower:
            return "The module expected data that wasn't provided by upstream modules"
        elif "timeout" in error_lower:
            return "The operation is taking longer than expected, possibly due to heavy computation or external delays"
        elif "connection" in error_lower:
            return "Network issues or the external service may be down"
        elif "permission" in error_lower:
            return "The system lacks necessary permissions to perform this operation"
        elif "memory" in error_lower:
            return "The system is running low on memory, possibly processing too much data"
        
        return "The error appears to be related to unexpected input or system state"
    
    def _suggest_fix(self, error_type: str, error_message: str) -> str:
        """Suggest fixes for common errors"""
        suggestions = {
            'KeyError': "1. Verify all required data is being provided\n2. Add checks for data existence before access\n3. Use .get() with default values",
            'TypeError': "1. Validate input data types\n2. Add type conversion where needed\n3. Check for None values",
            'ValueError': "1. Add input validation\n2. Check value ranges\n3. Handle edge cases",
            'AttributeError': "1. Check object initialization\n2. Verify method/attribute names\n3. Add hasattr() checks",
            'TimeoutError': "1. Optimize the slow operation\n2. Increase timeout limits\n3. Break into smaller operations",
            'ConnectionError': "1. Check network connectivity\n2. Verify service URLs\n3. Add retry logic"
        }
        
        return suggestions.get(error_type, "Review the error details and check module implementation")
    
    def _assess_performance_status(self, metrics: Dict[str, float]) -> tuple:
        """Assess overall performance status"""
        avg_time = metrics.get('avg_time_ms', 0)
        error_rate = metrics.get('error_rate', 0)
        
        if error_rate > 0.1:
            return "🔴", "Critical - High Error Rate"
        elif avg_time > 200:
            return "🔴", "Critical - Very Slow"
        elif error_rate > 0.05 or avg_time > 100:
            return "🟡", "Warning - Degraded Performance"
        else:
            return "🟢", "Healthy"
    
    def _generate_performance_summary(self, metrics: Dict[str, float]) -> str:
        """Generate performance summary"""
        avg_time = metrics.get('avg_time_ms', 0)
        error_rate = metrics.get('error_rate', 0)
        throughput = metrics.get('throughput_per_min', 0)
        
        parts = []
        
        if avg_time > 0:
            speed = "very slow" if avg_time > 200 else "slow" if avg_time > 100 else "fast"
            parts.append(f"The module is running {speed} ({avg_time:.0f}ms average)")
        
        if error_rate > 0:
            error_desc = "high" if error_rate > 0.1 else "some" if error_rate > 0.05 else "occasional"
            parts.append(f"experiencing {error_desc} errors ({error_rate:.1%} failure rate)")
        
        if throughput > 0:
            parts.append(f"processing {throughput:.0f} operations per minute")
        
        return ". ".join(parts) if parts else "No performance data available"
    
    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        """Format metrics for display"""
        lines = []
        
        metric_names = {
            'avg_time_ms': ('Average Response Time', 'ms'),
            'max_time_ms': ('Maximum Response Time', 'ms'),
            'p95_time_ms': ('95th Percentile Time', 'ms'),
            'error_rate': ('Error Rate', '%'),
            'throughput_per_min': ('Throughput', 'ops/min'),
            'cache_hit_rate': ('Cache Hit Rate', '%')
        }
        
        for key, (name, unit) in metric_names.items():
            if key in metrics:
                value = metrics[key]
                if unit == '%':
                    value = value * 100
                lines.append(f"• {name}: {value:.1f}{unit}")
        
        return "\n".join(lines) if lines else "• No metrics available"
    
    def _generate_performance_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []
        
        avg_time = metrics.get('avg_time_ms', 0)
        error_rate = metrics.get('error_rate', 0)
        cache_hit_rate = metrics.get('cache_hit_rate', 1)
        
        if avg_time > 150:
            recommendations.append("Consider optimizing computation-heavy operations")
        if error_rate > 0.05:
            recommendations.append("Investigate and fix the source of errors")
        if cache_hit_rate < 0.7:
            recommendations.append("Improve caching strategy to reduce redundant computations")
        
        return recommendations
    
    def _describe_trends(self, metrics: Dict[str, float]) -> str:
        """Describe performance trends"""
        if 'trend' in metrics:
            trend = metrics['trend']
            if trend == 'improving':
                return "Performance has been improving over time ↗"
            elif trend == 'degrading':
                return "Performance has been degrading - attention needed ↘"
            else:
                return "Performance is stable →"
        
        return "Insufficient data to determine trends"
    
    def _list_modules(self, modules: List[str]) -> str:
        """Format module list for sentence"""
        if not modules:
            return "no modules"
        elif len(modules) == 1:
            return modules[0]
        elif len(modules) == 2:
            return f"{modules[0]} and {modules[1]}"
        else:
            return f"{', '.join(modules[:-1])}, and {modules[-1]}"
    
    def _format_module_list(self, modules: List[str], empty_message: str) -> str:
        """Format module list for display"""
        if not modules:
            return empty_message
        
        return "\n".join(f"• {module}" for module in sorted(modules))
    
    def _format_age(self, age_seconds: float) -> str:
        """Format age in human-readable form"""
        if age_seconds < 1:
            return "just now"
        elif age_seconds < 60:
            return f"{age_seconds:.0f} seconds ago"
        elif age_seconds < 3600:
            return f"{age_seconds/60:.0f} minutes ago"
        elif age_seconds < 86400:
            return f"{age_seconds/3600:.1f} hours ago"
        else:
            return f"{age_seconds/86400:.1f} days ago"
    
    def _get_health_emoji(self, status: str) -> str:
        """Get emoji for health status"""
        status_lower = status.lower()
        if 'critical' in status_lower:
            return "🔴"
        elif 'warning' in status_lower or 'degraded' in status_lower:
            return "🟡"
        else:
            return "🟢"
    
    def _format_system_resources(self, metrics: Dict[str, float]) -> str:
        """Format system resource metrics"""
        lines = []
        
        if 'cpu_percent' in metrics:
            cpu = metrics['cpu_percent']
            cpu_status = "🔴 High" if cpu > 80 else "🟡 Moderate" if cpu > 50 else "🟢 Normal"
            lines.append(f"• CPU Usage: {cpu:.1f}% ({cpu_status})")
        
        if 'memory_percent' in metrics:
            mem = metrics['memory_percent']
            mem_status = "🔴 High" if mem > 80 else "🟡 Moderate" if mem > 60 else "🟢 Normal"
            lines.append(f"• Memory Usage: {mem:.1f}% ({mem_status})")
        
        if 'disk_percent' in metrics:
            disk = metrics['disk_percent']
            disk_status = "🔴 Critical" if disk > 90 else "🟡 Warning" if disk > 80 else "🟢 OK"
            lines.append(f"• Disk Usage: {disk:.1f}% ({disk_status})")
        
        return "\n".join(lines) if lines else "No resource data available"
    
    def _format_module_health(self, module_health: Dict[str, str]) -> str:
        """Format module health status"""
        if not module_health:
            return "No modules registered"
        
        # Group by status
        by_status = {'healthy': [], 'warning': [], 'critical': [], 'disabled': []}
        
        for module, status in module_health.items():
            status_key = status.lower()
            if status_key in by_status:
                by_status[status_key].append(module)
            else:
                by_status['warning'].append(module)
        
        lines = []
        
        if by_status['healthy']:
            lines.append(f"✅ Healthy ({len(by_status['healthy'])}): {', '.join(by_status['healthy'][:3])}")
            if len(by_status['healthy']) > 3:
                lines.append(f"   and {len(by_status['healthy']) - 3} more...")
        
        if by_status['warning']:
            lines.append(f"⚠️ Warning ({len(by_status['warning'])}): {', '.join(by_status['warning'])}")
        
        if by_status['critical']:
            lines.append(f"🔴 Critical ({len(by_status['critical'])}): {', '.join(by_status['critical'])}")
        
        if by_status['disabled']:
            lines.append(f"🚫 Disabled ({len(by_status['disabled'])}): {', '.join(by_status['disabled'])}")
        
        return "\n".join(lines)
    
    def _format_alerts(self, alerts: List[Dict[str, Any]]) -> str:
        """Format alerts for display"""
        lines = []
        
        for alert in alerts[:5]:  # Show max 5 alerts
            alert_type = alert.get('type', 'unknown')
            if alert_type == 'system_resource':
                metric = alert.get('metric', 'unknown')
                value = alert.get('value', 0)
                lines.append(f"• {metric}: {value:.1f} (threshold exceeded)")
            elif alert_type == 'module_health':
                module = alert.get('module', 'unknown')
                status = alert.get('status', 'unknown')
                lines.append(f"• Module {module} is {status}")
            else:
                lines.append(f"• {alert}")
        
        if len(alerts) > 5:
            lines.append(f"• ... and {len(alerts) - 5} more alerts")
        
        return "\n".join(lines)
    
    def _generate_dependency_action_items(self, issues: List[str], 
                                        suggestions: List[str]) -> str:
        """Generate action items for dependency issues"""
        items = []
        
        if any('circular' in issue.lower() for issue in issues):
            items.append("1. Review and break circular dependencies")
        
        if any('isolated' in issue.lower() for issue in issues):
            items.append("2. Remove or integrate isolated modules")
        
        if len(suggestions) > 3:
            items.append("3. Implement optimization suggestions to improve performance")
        
        if not items:
            items.append("1. Continue monitoring dependency health")
            items.append("2. Document any new module dependencies")
        
        return "\n".join(items)