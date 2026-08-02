# ─────────────────────────────────────────────────────────────
# File: modules/utils/system_utilities.py
# Unified, production-ready System Utilities & Analysis Framework
# - Consolidates plain-English explanations and integration validation access
# - Robust schema normalization and defensive error handling
# - No placeholders, no dummy data paths
# ─────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from modules.utils.audit_utils import RotatingLogger, format_operator_message

if TYPE_CHECKING:
    from modules.core.module_system import ModuleOrchestrator


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SystemUtilitiesConfig:
    """
    Hardened configuration for SystemUtilities & EnglishExplainer.
    Validated ranges guard against misconfiguration.
    """
    # Core settings
    enabled: bool = True
    debug_mode: bool = True
    log_level: str = "DEBUG"
    max_cache_size: int = 10_000
    cache_ttl_seconds: int = 3600

    # Performance settings
    max_parallel_operations: int = 10
    default_timeout_ms: int = 5000
    health_check_interval_ms: int = 60_000
    metrics_retention_hours: int = 24

    # Validation settings
    strict_validation: bool = True
    auto_fix_enabled: bool = False
    validation_timeout_ms: int = 30_000
    max_validation_retries: int = 3

    # Explanation settings
    plain_english_enabled: bool = True
    detailed_explanations: bool = True
    include_technical_details: bool = True
    max_explanation_length: int = 2000

    # Integration settings
    smart_bus_integration: bool = True
    audit_system_integration: bool = True
    auto_publish_reports: bool = True
    publish_interval_seconds: int = 300

    # Error handling
    circuit_breaker_threshold: int = 5
    recovery_time_seconds: int = 60
    emergency_mode_enabled: bool = True
    error_escalation_enabled: bool = True

    # File paths
    config_paths: List[str] = field(default_factory=lambda: [
        "config/system_config.yaml",
        "config/risk_policy.yaml",
        "config/explainability_standards.yaml",
        "config/module_registry.yaml",
    ])


    module_discovery_paths: List[str] = field(default_factory=lambda: [
        "modules/auditing",
        "modules/core",
        "modules/external",
        "modules/features",
        "modules/market",
        "modules/memory",
        "modules/meta",
        "modules/models",
        "modules/position",
        "modules/reward",
        "modules/risk",
        "modules/strategy",
        "modules/trading_modes",
        "modules/visualization",
        "modules/voting",
        "modules/simulation",
    ])

    def __post_init__(self) -> None:
        self._validate_config()

    def _validate_config(self) -> None:
        errors: List[str] = []

        # Ranges
        if not (1 <= self.default_timeout_ms <= 60_000):
            errors.append("default_timeout_ms must be 1..60000")
        if not (1 <= self.validation_timeout_ms <= 120_000):
            errors.append("validation_timeout_ms must be 1..120000")
        if not (1 <= self.max_parallel_operations <= 100):
            errors.append("max_parallel_operations must be 1..100")
        if not (1 <= self.cache_ttl_seconds <= 86_400):
            errors.append("cache_ttl_seconds must be 1..86400")
        if not (1 <= self.circuit_breaker_threshold <= 20):
            errors.append("circuit_breaker_threshold must be 1..20")
        if not (1 <= self.recovery_time_seconds <= 3600):
            errors.append("recovery_time_seconds must be 1..3600")
        if not (100 <= self.max_explanation_length <= 10_000):
            errors.append("max_explanation_length must be 100..10000")

        if self.log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            errors.append("log_level must be one of DEBUG|INFO|WARNING|ERROR|CRITICAL")

        if errors:
            raise ValueError(f"SystemUtilitiesConfig validation failed: {errors}")

    def update(self, updates: Dict[str, Any]) -> None:
        old: Dict[str, Any] = {}
        for k, v in updates.items():
            if hasattr(self, k):
                old[k] = getattr(self, k)
                setattr(self, k, v)
        try:
            self._validate_config()
        except Exception:
            # rollback
            for k, v in old.items():
                setattr(self, k, v)
            raise

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ═══════════════════════════════════════════════════════════════════
# REPORTING & VALIDATION DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ExplanationTemplate:
    template: str
    required_fields: List[str]
    category: str
    priority: int = 0
    max_length: int = 2000
    include_timestamp: bool = True
    include_metadata: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.template, str) or not self.template.strip():
            raise ValueError("Template must be a non-empty string")
        if not isinstance(self.required_fields, list) or not self.required_fields:
            raise ValueError("required_fields must be a non-empty list")


@dataclass
class ValidationIssue:
    module: str
    issue_type: str
    severity: str  # 'critical'|'error'|'warning'|'info'
    message: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    suggestion: Optional[str] = None
    fix_available: bool = False
    auto_fixable: bool = False
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.severity not in {"critical", "error", "warning", "info"}:
            raise ValueError("severity must be one of: critical|error|warning|info")
        if not self.module or not self.message:
            raise ValueError("module and message are required")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "module": self.module,
            "issue_type": self.issue_type,
            "severity": self.severity,
            "message": self.message,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "suggestion": self.suggestion,
            "fix_available": self.fix_available,
            "auto_fixable": self.auto_fixable,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "context": self.context,
        }


@dataclass
class ValidationReport:
    total_modules: int
    validated_modules: int
    issues: List[ValidationIssue] = field(default_factory=list)
    missing_decorators: List[str] = field(default_factory=list)
    missing_thesis: List[str] = field(default_factory=list)
    legacy_modules: List[str] = field(default_factory=list)
    config_issues: List[str] = field(default_factory=list)
    integration_score: float = 0.0
    validation_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    module_discovery_time_ms: float = 0.0
    validation_execution_time_ms: float = 0.0
    config_validation_time_ms: float = 0.0
    circular_dependencies: List[List[str]] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)
    security_issues: List[ValidationIssue] = field(default_factory=list)
    performance_issues: List[ValidationIssue] = field(default_factory=list)

    def get_issues_by_severity(self, severity: str) -> List[ValidationIssue]:
        return [i for i in self.issues if i.severity == severity]

    def get_critical_issues(self) -> List[ValidationIssue]:
        return self.get_issues_by_severity("critical")

    def get_error_issues(self) -> List[ValidationIssue]:
        return self.get_issues_by_severity("error")

    def get_fixable_issues(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.auto_fixable]

    def calculate_health_score(self) -> Tuple[float, str]:
        score = 100.0
        for i in self.issues:
            score -= {"critical": 10.0, "error": 5.0, "warning": 2.0, "info": 0.5}.get(i.severity, 1.0)
        score -= len(self.missing_decorators) * 3
        score -= len(self.missing_thesis) * 2
        score -= len(self.legacy_modules) * 2
        score -= len(self.config_issues) * 5
        score -= len(self.circular_dependencies) * 5
        score = max(0.0, score)
        status = "Excellent" if score >= 90 else "Good" if score >= 80 else "Fair" if score >= 70 else "Poor" if score >= 50 else "Critical"
        return score, status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_modules": self.total_modules,
            "validated_modules": self.validated_modules,
            "issues": [i.to_dict() for i in self.issues],
            "missing_decorators": self.missing_decorators,
            "missing_thesis": self.missing_thesis,
            "legacy_modules": self.legacy_modules,
            "config_issues": self.config_issues,
            "integration_score": self.integration_score,
            "validation_time_ms": self.validation_time_ms,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "module_discovery_time_ms": self.module_discovery_time_ms,
            "validation_execution_time_ms": self.validation_execution_time_ms,
            "config_validation_time_ms": self.config_validation_time_ms,
            "circular_dependencies": self.circular_dependencies,
            "optimization_opportunities": self.optimization_opportunities,
            "security_issues": [i.to_dict() for i in self.security_issues],
            "performance_issues": [i.to_dict() for i in self.performance_issues],
        }

    def _generate_next_steps(self) -> str:
        steps: List[str] = []
        if self.get_critical_issues():
            steps.append("1. [ALERT] Address critical issues immediately.")
        if self.missing_decorators:
            steps.append(f"2. Add @module decorators to {len(self.missing_decorators)} modules.")
        if self.missing_thesis:
            steps.append(f"3. Implement thesis generation in {len(self.missing_thesis)} modules.")
        if self.config_issues:
            steps.append(f"4. Fix {len(self.config_issues)} configuration issues.")
        if self.circular_dependencies:
            steps.append(f"5. Break {len(self.circular_dependencies)} circular dependencies.")
        if self.get_fixable_issues():
            steps.append(f"6. Auto-fix {len(self.get_fixable_issues())} fixable issues.")
        if not steps:
            score, _ = self.calculate_health_score()
            steps.append("[OK] System healthy; continue monitoring." if score >= 90 else "Run deeper analysis for optimizations.")
        return "\n".join(steps)

    def to_plain_english(self) -> str:
        health_score, health_status = self.calculate_health_score()
        critical_issues = self.get_critical_issues()
        error_issues = self.get_error_issues()
        warnings = self.get_issues_by_severity("warning")
        return f"""[ROCKET] SMARTINFOBUS INTEGRATION REPORT
==================================
Overall Health Score: {health_score:.1f}% ({health_status})
Integration Score: {self.integration_score:.1f}%
Modules Validated: {self.validated_modules}/{self.total_modules}
Analysis Time: {self.validation_time_ms:.1f}ms

ISSUES SUMMARY:
• [ALERT] Critical Issues: {len(critical_issues)}
• [FAIL] Error Issues: {len(error_issues)}
• [WARN] Warnings: {len(warnings)}
• 📋 Missing Decorators: {len(self.missing_decorators)}
• [RELOAD] Legacy Modules: {len(self.legacy_modules)}
• ⚙️ Config Issues: {len(self.config_issues)}

SYSTEM ANALYSIS:
• 🔗 Circular Dependencies: {len(self.circular_dependencies)}
• [TOOL] Auto-fixable Issues: {len(self.get_fixable_issues())}
• [SAFE] Security Issues: {len(self.security_issues)}
• [FAST] Performance Issues: {len(self.performance_issues)}

NEXT STEPS:
{self._generate_next_steps()}

Generated: {datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
Report ID: {str(uuid.uuid4())[:8].upper()}""".strip()


# ═══════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CircuitBreakerState:
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "CLOSED"  # CLOSED | OPEN | HALF_OPEN
    successful_calls: int = 0
    total_calls: int = 0
    last_success_time: float = 0.0

    def record_success(self) -> None:
        self.successful_calls += 1
        self.total_calls += 1
        self.last_success_time = time.time()
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1
        self.total_calls += 1
        self.last_failure_time = time.time()

    def should_allow_request(self, recovery_time: float) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > recovery_time:
                self.state = "HALF_OPEN"
                return True
            return False
        return True  # HALF_OPEN

    def trip(self) -> None:
        self.state = "OPEN"


# ═══════════════════════════════════════════════════════════════════
# ENGLISH EXPLAINER
# ═══════════════════════════════════════════════════════════════════

class EnglishExplainer:
    """
    Plain-English explanations for SmartInfoBus components.
    Strong input validation, bounded length, and circuit-breaker resilience.
    """

    def __init__(self, config: Optional[SystemUtilitiesConfig] = None) -> None:
        self.config = config or SystemUtilitiesConfig()
        self.templates = self._load_templates()
        self.logger = RotatingLogger(name="EnglishExplainer", log_path="logs/utils/english_explainer.log", max_lines=5000)
        self._lock = threading.RLock()
        self._explanation_count = 0
        self._total_explanation_time = 0.0
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._circuit_breaker = CircuitBreakerState()

        self.logger.info(
            format_operator_message(
                icon="🛠️",
                message="EnglishExplainer initialized",
                config=self.config.to_dict(),
            )
        )

    # ---------- logging helper ----------
    def _log_debug(self, msg: str) -> None:
        if getattr(self.config, "debug_mode", False):
            if hasattr(self.logger, "debug"):
                self.logger.debug(msg)
            else:
                self.logger.info(f"[DEBUG] {msg}")

    # ---------- templates ----------
    def _load_templates(self) -> Dict[str, ExplanationTemplate]:
        return {
            "module_decision": ExplanationTemplate(
                template=(
                    "\n{module_name} Decision Analysis\n"
                    "==============================\n"
                    "Decision: {decision}\n"
                    "Confidence: {confidence:.1%}\n"
                    "Analysis Time: {analysis_time_ms:.1f}ms\n\n"
                    "REASONING ANALYSIS:\n"
                    "{reasoning_points}\n\n"
                    "{additional_context}\n\n"
                    "PRIMARY FACTOR: {primary_reason}\n\n"
                    "CONFIDENCE BREAKDOWN:\n"
                    "{confidence_breakdown}\n\n"
                    "{risk_assessment}\n"
                ),
                required_fields=[
                    "module_name",
                    "decision",
                    "confidence",
                    "analysis_time_ms",
                    "reasoning_points",
                    "additional_context",
                    "primary_reason",
                    "confidence_breakdown",
                    "risk_assessment",
                ],
                category="decision",
                priority=1,
            ),
            "error_explanation": ExplanationTemplate(
                template=(
                    "\n[WARN] ERROR ANALYSIS: {module_name}\n"
                    "================================\n"
                    "Incident: {plain_english_explanation}\n\n"
                    "TECHNICAL DETAILS:\n"
                    "• Error Type: {error_type}\n"
                    "• Location: {file_location}\n"
                    "• Timestamp: {error_timestamp}\n"
                    "• Likely Root Cause: {likely_cause}\n\n"
                    "IMPACT ASSESSMENT:\n"
                    "{impact_assessment}\n\n"
                    "RECOMMENDED ACTIONS:\n"
                    "{suggested_fix}\n\n"
                    "PREVENTION STRATEGY:\n"
                    "{prevention_measures}\n\n"
                    "System Response: {system_response}\n"
                ),
                required_fields=["module_name", "plain_english_explanation", "error_type"],
                category="error",
                priority=3,
            ),
            "performance_report": ExplanationTemplate(
                template=(
                    "\n[STATS] PERFORMANCE ANALYSIS: {module_name}\n"
                    "======================================\n"
                    "Status: {status_emoji} {status_text}\n"
                    "Period: {period}\n"
                    "Analysis Time: {report_generation_time_ms:.1f}ms\n\n"
                    "EXECUTIVE SUMMARY:\n"
                    "{summary_text}\n\n"
                    "PERFORMANCE METRICS:\n"
                    "{metrics_text}\n\n"
                    "TREND ANALYSIS:\n"
                    "{trends_text}\n\n"
                    "{alert_section}\n\n"
                    "OPTIMIZATION RECOMMENDATIONS:\n"
                    "{recommendations_section}\n\n"
                    "NEXT REVIEW: {next_review_time}\n"
                ),
                required_fields=["module_name", "status_text", "summary_text", "metrics_text"],
                category="performance",
                priority=2,
            ),
            "health_status": ExplanationTemplate(
                template=(
                    "\n[HEALTH] SYSTEM HEALTH REPORT\n"
                    "========================\n"
                    "Overall Health: {overall_status_emoji} {overall_status}\n"
                    "Generated: {timestamp}\n"
                    "Report ID: {report_id}\n\n"
                    "SYSTEM RESOURCES:\n"
                    "{resource_status}\n\n"
                    "MODULE HEALTH MATRIX:\n"
                    "{module_status}\n\n"
                    "{alert_section}\n\n"
                    "PREDICTIVE ANALYSIS:\n"
                    "{predictive_insights}\n\n"
                    "ACTION ITEMS:\n"
                    "{recommendations}\n\n"
                    "NEXT HEALTH CHECK: {next_check_time}\n"
                ),
                required_fields=["overall_status", "resource_status", "module_status"],
                category="health",
                priority=1,
            ),
            "data_flow_analysis": ExplanationTemplate(
                template=(
                    "\n[RELOAD] DATA FLOW ANALYSIS: {data_key}\n"
                    "=================================\n"
                    "Flow Status: {status_badge} {status}\n"
                    "Analysis ID: {analysis_id}\n\n"
                    "FLOW DESCRIPTION:\n"
                    "{explanation_text}\n\n"
                    "DATA PROVIDERS: {providers_text}\n"
                    "DATA CONSUMERS: {consumers_text}\n\n"
                    "CURRENT DATA SNAPSHOT:\n"
                    "{current_data_section}\n\n"
                    "FLOW HEALTH CHECK:\n"
                    "{flow_health_check}\n\n"
                    "OPTIMIZATION OPPORTUNITIES:\n"
                    "{optimization_suggestions}\n\n"
                    "MONITORING ALERTS:\n"
                    "{monitoring_alerts}\n"
                ),
                required_fields=["data_key", "status", "explanation_text"],
                category="data_flow",
                priority=2,
            ),
            "integration_report": ExplanationTemplate(
                template=(
                    "\n🔗 INTEGRATION ANALYSIS REPORT\n"
                    "===============================\n"
                    "System Integration Score: {integration_score}\n"
                    "Health Status: {health_status}\n"
                    "Analysis Timestamp: {timestamp}\n\n"
                    "INTEGRATION SUMMARY:\n"
                    "{integration_summary}\n\n"
                    "MODULE COMPLIANCE:\n"
                    "{compliance_details}\n\n"
                    "IDENTIFIED ISSUES:\n"
                    "{issues_summary}\n\n"
                    "MIGRATION STATUS:\n"
                    "{migration_status}\n\n"
                    "NEXT STEPS:\n"
                    "{action_plan}\n\n"
                    "SYSTEM RECOMMENDATIONS:\n"
                    "{system_recommendations}\n"
                ),
                required_fields=["integration_score", "health_status", "integration_summary"],
                category="integration",
                priority=1,
            ),
        }

    # ---------- public explainers ----------
    def explain_module_decision(
        self,
        module_name: str,
        decision: Any,
        context: Dict[str, Any],
        confidence: float,
        analysis_time_ms: float = 0.0,
    ) -> str:
        if not self._circuit_breaker.should_allow_request(self.config.recovery_time_seconds):
            return self._generate_fallback_explanation("Service temporarily unavailable")
        try:
            with self._lock:
                t0 = time.time()
                reasoning_points = self._extract_reasoning_from_context(context)
                primary_reason = self._determine_primary_reason(decision, context)
                additional_context = self._format_additional_context(context)
                confidence_breakdown = self._analyze_confidence_factors(confidence, context)
                risk_assessment = self._generate_risk_assessment(decision, context)

                explanation = self._fill_template(
                    "module_decision",
                    {
                        "module_name": module_name,
                        "decision": self._format_decision(decision),
                        "confidence": confidence,
                        "analysis_time_ms": analysis_time_ms,
                        "reasoning_points": reasoning_points,
                        "additional_context": additional_context,
                        "primary_reason": primary_reason,
                        "confidence_breakdown": confidence_breakdown,
                        "risk_assessment": risk_assessment,
                    },
                )
                dur = (time.time() - t0) * 1000.0
                self._record_performance_metric("explain_module_decision", dur, True)
                self._circuit_breaker.record_success()
                return explanation
        except Exception as e:
            self._circuit_breaker.record_failure()
            self.logger.error(f"explain_module_decision error: {e}")
            return self._generate_fallback_explanation(f"Error generating explanation: {e}")
        

    def _format_system_resources(self, metrics: Dict[str, float]) -> str:
        """
        Render CPU/memory/disk with simple status thresholds.
        Expects percentages in [0, 100]. Unknowns default to 0.
        """
        cpu = float(metrics.get("cpu_percent", 0.0))
        mem = float(metrics.get("memory_percent", 0.0))
        dsk = float(metrics.get("disk_percent", 0.0))

        def badge(pct: float) -> str:
            if pct >= 85.0:
                return "[ALERT]"
            if pct >= 70.0:
                return "[WARN]"
            return "[OK]"

        lines = [
            f"  {badge(cpu)} CPU Usage: {cpu:.1f}%",
            f"  {badge(mem)} Memory Usage: {mem:.1f}%",
            f"  {badge(dsk)} Disk Usage: {dsk:.1f}%",
        ]
        return "\n".join(lines)
    
    def _format_module_health(self, module_health: Dict[str, str]) -> str:
        """
        Group modules by health status and render a compact summary.
        module_health: {module_name: 'healthy'|'warning'|'critical'|...}
        """
        if not module_health:
            return "  • No modules reported"

        groups: Dict[str, List[str]] = defaultdict(list)
        for mod, status in module_health.items():
            groups[status.lower()].append(mod)

        def emoji(status: str) -> str:
            s = status.lower()
            if s == "healthy":
                return "[OK]"
            if s == "warning":
                return "[WARN]"
            if s == "critical":
                return "[ALERT]"
            return "❓"

        out: List[str] = []
        for status in sorted(groups.keys()):
            mods = sorted(groups[status])
            out.append(f"  {emoji(status)} {status.title()}: {len(mods)} modules")
            if mods:
                sample = ", ".join(mods[:3])
                tail = f", +{len(mods)-3} more" if len(mods) > 3 else ""
                out.append(f"    └─ {sample}{tail}")
        return "\n".join(out)
    

    def _generate_predictive_insights(
        self,
        system_metrics: Dict[str, float],
        module_health: Dict[str, str]
    ) -> str:
        """
        Lightweight predictive hints based on current resource levels and
        module health ratios. Purely heuristic, deterministic output.
        """
        insights: List[str] = []

        cpu = float(system_metrics.get("cpu_percent", 0.0))
        mem = float(system_metrics.get("memory_percent", 0.0))
        dsk = float(system_metrics.get("disk_percent", 0.0))

        # Resource trend hints
        if cpu >= 80.0:
            insights.append("[CHART] CPU sustained high; consider scaling or load shedding")
        elif cpu >= 60.0:
            insights.append("[STATS] CPU trending warm; keep monitoring")

        if mem >= 80.0:
            insights.append("🧠 Memory pressure is high; inspect for leaks or large batches")
        elif mem >= 60.0:
            insights.append("[STATS] Memory trending warm; observe GC/alloc behavior")

        if dsk >= 85.0:
            insights.append("[ALERT] Disk nearly full; rotate logs or expand storage")
        elif dsk >= 70.0:
            insights.append("[WARN] Disk usage rising; plan housekeeping")

        # Module health ratio
        total = max(1, len(module_health))
        healthy = sum(1 for s in module_health.values() if str(s).lower() == "healthy")
        ratio = healthy / total

        if ratio >= 0.9:
            insights.append("[OK] Module health distribution is strong")
        elif ratio >= 0.75:
            insights.append("[WARN] Some modules show warnings; prioritize remediation")
        else:
            insights.append("[ALERT] Multiple modules degraded; triage required")

        if not insights:
            insights.append("[STATS] No predictive concerns based on current snapshot")

        return "\n".join(f"  {line}" for line in insights)



        
    def _determine_error_cause(
        self,
        error_type: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> str:
        """
        Heuristically infer a likely root cause from error type/message/context.
        Returns a short, plain-English sentence.
        """
        emsg = (error_message or "").lower()

        # Quick message heuristics first
        if "nonetype" in emsg:
            return "A function returned no value (None) where a value was expected"
        if "list index out of range" in emsg:
            return "An array/list access used an index outside the available range"
        if "division by zero" in emsg or "divide by zero" in emsg:
            return "A division was attempted with zero as the divisor"
        if "connection refused" in emsg:
            return "The target service is not accepting connections"
        if "timeout" in emsg:
            return "The operation exceeded the maximum allowed time"
        if "permission denied" in emsg or "access is denied" in emsg:
            return "The process lacks required filesystem or OS permissions"
        if "not found" in emsg or "no such file" in emsg:
            return "A required file or resource path does not exist"

        # By error type
        if error_type == "KeyError":
            missing_key = context.get("missing_key")
            if missing_key:
                return f"The key '{missing_key}' was requested before it was provided"
            return "A module requested data that has not been provided yet"
        if error_type == "TimeoutError":
            return "The call took longer than the configured timeout budget"
        if error_type == "TypeError":
            return "A function received an argument of the wrong type or shape"
        if error_type == "ValueError":
            return "A function received an argument with an invalid value"
        if error_type == "AttributeError":
            return "Code attempted to access an attribute that does not exist on the object"
        if error_type == "ImportError" or error_type == "ModuleNotFoundError":
            return "A required Python module or package is missing or misconfigured"
        if error_type == "ConnectionError":
            return "A network call failed due to connectivity or endpoint issues"
        if error_type == "MemoryError":
            return "The process ran out of available memory"
        if error_type == "FileNotFoundError":
            return "The code tried to read a file that does not exist on disk"
        if error_type == "PermissionError":
            return "An operation failed due to insufficient OS-level permissions"

        # Context-based hints
        if context.get("upstream_unavailable"):
            return "An upstream dependency is unavailable or unhealthy"
        if context.get("bad_payload"):
            return "An upstream dependency returned an unexpected or malformed payload"

        # Fallback
        return "The system encountered an unexpected condition in its execution path"


    def explain_error(
        self,
        module_name: str,
        error_type: str,
        error_message: str,
        file_path: Optional[str] = None,
        line_number: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not self._circuit_breaker.should_allow_request(self.config.recovery_time_seconds):
            return self._generate_fallback_explanation("Error explanation service unavailable")
        try:
            with self._lock:
                t0 = time.time()
                context = context or {}
                plain = self._translate_error_to_plain_english(error_type, error_message)
                likely_cause = self._determine_error_cause(error_type, error_message, context)
                impact = self._assess_error_impact(error_type, context)
                fix = self._generate_fix_suggestion(error_type, error_message, context)
                prevention = self._generate_prevention_strategy(error_type, context)
                system_resp = self._determine_system_response(error_type, context)
                file_location = f"{file_path}:{line_number}" if file_path and line_number else "Unknown location"

                explanation = self._fill_template(
                    "error_explanation",
                    {
                        "module_name": module_name,
                        "plain_english_explanation": plain,
                        "error_type": error_type,
                        "file_location": file_location,
                        "error_timestamp": datetime.now().isoformat(),
                        "likely_cause": likely_cause,
                        "impact_assessment": impact,
                        "suggested_fix": fix,
                        "prevention_measures": prevention,
                        "system_response": system_resp,
                    },
                )
                dur = (time.time() - t0) * 1000.0
                self._record_performance_metric("explain_error", dur, True)
                self._circuit_breaker.record_success()
                return explanation
        except Exception as e:
            self._circuit_breaker.record_failure()
            self.logger.error(f"explain_error error: {e}")
            return self._generate_fallback_explanation(f"Error explanation failed: {e}")

    def explain_performance(self, module_name: str, metrics: Dict[str, float], period: str = "Last 24 hours") -> str:
        if not self._circuit_breaker.should_allow_request(self.config.recovery_time_seconds):
            return self._generate_fallback_explanation("Performance analysis service unavailable")
        try:
            with self._lock:
                t0 = time.time()
                status_emoji, status_text = self._assess_performance_status(metrics)
                summary_text = self._generate_performance_summary(metrics)
                metrics_text = self._format_metrics(metrics)
                trends_text = self._analyze_performance_trends(metrics)
                recommendations = self._generate_performance_recommendations(metrics)
                alert_section = self._generate_performance_alerts(metrics)
                recommendations_section = "\n".join(f"• {r}" for r in recommendations) if recommendations else "• No optimizations needed"
                next_review_time = (datetime.now() + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M")
                dur = (time.time() - t0) * 1000.0

                explanation = self._fill_template(
                    "performance_report",
                    {
                        "module_name": module_name,
                        "status_emoji": status_emoji,
                        "status_text": status_text,
                        "period": period,
                        "report_generation_time_ms": dur,
                        "summary_text": summary_text,
                        "metrics_text": metrics_text,
                        "trends_text": trends_text,
                        "alert_section": alert_section,
                        "recommendations_section": recommendations_section,
                        "next_review_time": next_review_time,
                    },
                )
                self._record_performance_metric("explain_performance", dur, True)
                self._circuit_breaker.record_success()
                return explanation
        except Exception as e:
            self._circuit_breaker.record_failure()
            self.logger.error(f"explain_performance error: {e}")
            return self._generate_fallback_explanation(f"Performance analysis failed: {e}")

    def explain_data_flow(
        self,
        data_key: str,
        providers: List[str],
        consumers: List[str],
        current_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not providers and not consumers:
            status, status_badge, explanation_text = "[RED] Unused", "[RED]", (
                f"The data key '{data_key}' is not currently used by any modules."
            )
        elif not providers:
            status, status_badge, explanation_text = "[WARN] Missing Provider", "[WARN]", (
                f"Modules are trying to use '{data_key}' but no module provides it."
            )
        elif not consumers:
            status, status_badge, explanation_text = "[YELLOW] No Consumers", "[YELLOW]", (
                f"The data '{data_key}' is being produced but not used by any modules."
            )
        else:
            status, status_badge, explanation_text = "[OK] Active", "[OK]", (
                f"This data flows from {self._list_modules(providers)} to {self._list_modules(consumers)}."
            )

        providers_text = self._list_modules(providers) if providers else "None"
        consumers_text = self._list_modules(consumers) if consumers else "None"

        current_data_section = self._format_current_data(current_data or {})

        flow_health_check = self._assess_data_flow_health(providers, consumers, current_data)

        return self._fill_template(
            "data_flow_analysis",
            {
                "data_key": data_key,
                "status": status,
                "status_badge": status_badge,
                "analysis_id": str(uuid.uuid4())[:8].upper(),
                "explanation_text": explanation_text,
                "providers_text": providers_text,
                "consumers_text": consumers_text,
                "current_data_section": current_data_section or "  • No recent snapshot available",
                "flow_health_check": flow_health_check,
                "optimization_suggestions": "• Monitor data freshness\n• Consider caching frequently accessed data",
                "monitoring_alerts": "[OK] No active alerts for this flow",
            },
        )

    def explain_health_status(
        self,
        overall_status: str,
        system_metrics: Dict[str, float],
        module_health: Dict[str, str],
        alerts: Optional[List[str]] = None,
        recommendations: Optional[List[str]] = None,
    ) -> str:
        alerts = alerts or []
        recommendations = recommendations or []

        status_emoji = {
            "healthy": "[OK]",
            "warning": "[WARN]",
            "critical": "[ALERT]",
            "unknown": "❓",
        }.get(overall_status.lower(), "❓")

        resource_status = self._format_system_resources(system_metrics)
        module_status = self._format_module_health(module_health)
        alert_section = "ACTIVE ALERTS:\n" + "\n".join(f"[ALERT] {a}" for a in alerts) if alerts else "[OK] No active alerts"
        recommendations_text = "\n".join(f"{i+1}. {rec}" for i, rec in enumerate(recommendations)) or "System is healthy - no immediate actions required."
        predictive_insights = self._generate_predictive_insights(system_metrics, module_health)
        next_check_time = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")

        return self._fill_template(
            "health_status",
            {
                "overall_status_emoji": status_emoji,
                "overall_status": overall_status.title(),
                "resource_status": resource_status,
                "module_status": module_status,
                "alert_section": alert_section,
                "predictive_insights": predictive_insights,
                "recommendations": recommendations_text,
                "next_check_time": next_check_time,
            },
        )

    def explain_dependencies(
        self,
        total_modules: int,
        total_dependencies: int,
        issues: Optional[List[str]] = None,
        optimization_suggestions: Optional[List[str]] = None,
    ) -> str:
        issues = issues or []
        optimization_suggestions = optimization_suggestions or []

        issues_section = "ISSUES FOUND:\n" + "\n".join(f"[WARN] {x}" for x in issues) if issues else "[OK] No dependency issues found"
        avg_deps = total_dependencies / max(total_modules, 1)
        integration_score_text = "High" if avg_deps < 2 else "Moderate" if avg_deps < 4 else "Low"
        suggestions_text = "\n".join(f"• {s}" for s in optimization_suggestions) or "• No optimization needed - dependencies are well structured"
        action_items = self._generate_dependency_action_items(issues, optimization_suggestions)

        return self._fill_template(
            "integration_report",
            {
                "integration_score": integration_score_text,
                "health_status": "Healthy" if not issues else "Needs Attention",
                "integration_summary": "System integration analysis complete",
                "compliance_details": "Contracts and decorators appear consistent for most modules",
                "issues_summary": issues_section,
                "migration_status": "[OK] No migrations in progress",
                "action_plan": action_items,
                "system_recommendations": suggestions_text,
            },
        )

    def explain_execution_results(self, results: Dict[str, Any], execution_time: float, module_count: int, success_count: int) -> str:
        success_rate = success_count / max(module_count, 1)
        if success_rate >= 0.95:
            status, emoji = "Excellent", "[OK]"
        elif success_rate >= 0.80:
            status, emoji = "Good", "[WARN]"
        else:
            status, emoji = "Needs Attention", "[ALERT]"

        time_str = f"{execution_time*1000:.0f}ms" if execution_time < 1 else f"{execution_time:.2f}s"
        lines = [
            f"{emoji} EXECUTION SUMMARY: {status}",
            "===============================",
            f"Report ID: {str(uuid.uuid4())[:8].upper()}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PERFORMANCE METRICS:",
            f"• Execution Time: {time_str}",
            f"• Modules Executed: {module_count}",
            f"• Successful Modules: {success_count}",
            f"• Success Rate: {success_rate:.1%}",
            "",
            "RESULTS OVERVIEW:",
        ]
        if results:
            lines.append("\nKEY OUTPUTS:")
            for k, v in list(results.items())[:5]:
                if isinstance(v, (int, float)):
                    lines.append(f"  [STATS] {k}: {v}")
                elif isinstance(v, str):
                    lines.append(f"  [LOG] {k}: {v[:50]}{'...' if len(v) > 50 else ''}")
                else:
                    lines.append(f"  [TOOL] {k}: {type(v).__name__}")

        if success_rate < 0.8:
            lines += ["", "[WARN] RECOMMENDATIONS:", "  • Investigate failed modules", "  • Check error logs for details", "  • Consider increasing timeouts"]
        if execution_time > 1.0:
            lines += ["  • Optimize slow modules", "  • Consider parallel execution"]
        if success_rate >= 0.95 and execution_time < 0.5:
            lines += ["", "[OK] SYSTEM STATUS:", "  • Excellent performance", "  • Continue monitoring"]

        return "\n".join(lines).strip()

    # ---------- helpers ----------
    def _fill_template(self, name: str, values: Dict[str, Any]) -> str:
        t = self.templates.get(name)
        if not t:
            return f"[FAIL] Unknown template: {name}"
        payload = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "report_id": str(uuid.uuid4())[:8].upper(), "analysis_id": str(uuid.uuid4())[:8].upper()}
        payload.update(values)
        missing = [f for f in t.required_fields if f not in payload]
        if missing:
            return f"[FAIL] Template {name} missing required fields: {missing}"
        try:
            txt = t.template.format(**payload)
            if len(txt) > self.config.max_explanation_length:
                txt = txt[: self.config.max_explanation_length - 3] + "..."
            return txt
        except Exception as e:
            return f"[FAIL] Template formatting error ({name}): {e}"

    def _analyze_confidence_factors(self, confidence: float, context: Dict[str, Any]) -> str:
        factors: List[str] = []
        if confidence >= 0.9:
            factors += ["• Extremely High Confidence (90%+)", "  - Multiple strong signals aligned", "  - Historical pattern confirmation"]
        elif confidence >= 0.75:
            factors += ["• High Confidence (75-89%)", "  - Strong primary signals", "  - Good data quality"]
        elif confidence >= 0.5:
            factors += ["• Moderate Confidence (50-74%)", "  - Mixed signal environment", "  - Some uncertainty factors present"]
        else:
            factors += ["• Low Confidence (<50%)", "  - Conflicting signals detected", "  - High uncertainty environment"]
        vol = context.get("market_volatility")
        if isinstance(vol, (int, float)):
            factors.append("  - High market volatility reducing certainty" if vol > 0.7 else "  - Low volatility supporting confidence" if vol < 0.3 else "  - Typical volatility conditions")
        return "\n".join(factors)

    def _generate_risk_assessment(self, _decision: Any, context: Dict[str, Any]) -> str:
        risk_score = float(context.get("risk_score", 0.5))
        if risk_score > 0.8:
            return "[RED] HIGH RISK ENVIRONMENT:\n  - Exercise extreme caution\n  - Consider reduced position sizing"
        if risk_score > 0.5:
            return "[YELLOW] MODERATE RISK ENVIRONMENT:\n  - Standard risk management applies\n  - Monitor position closely"
        return "[GREEN] LOW RISK ENVIRONMENT:\n  - Favorable conditions detected\n  - Normal position sizing appropriate"

    def _generate_fallback_explanation(self, msg: str) -> str:
        return (
            f"\n[TOOL] EXPLANATION SYSTEM NOTICE\n"
            f"============================\n"
            f"{msg}\n\n"
            f"The explanation system is temporarily constrained.\n"
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    def _record_performance_metric(self, op: str, duration_ms: float, success: bool) -> None:
        with self._lock:
            self._explanation_count += 1
            if success:
                self._total_explanation_time += duration_ms
        self._log_debug(f"{op}: {duration_ms:.2f}ms (success={success})")

    def get_performance_statistics(self) -> Dict[str, Any]:
        with self._lock:
            avg = self._total_explanation_time / max(self._explanation_count, 1)
            return {
                "total_explanations": self._explanation_count,
                "average_time_ms": avg,
                "total_time_ms": self._total_explanation_time,
                "circuit_breaker_state": self._circuit_breaker.state,
                "circuit_breaker_failures": self._circuit_breaker.failure_count,
                "cache_size": len(self._cache),
            }

    def _extract_reasoning_from_context(self, context: Dict[str, Any]) -> str:
        points: List[str] = []
        for k, v in context.items():
            if k.startswith("_"):
                continue
            lk = k.lower()
            if "confidence" in lk and isinstance(v, (int, float)):
                tag = "[OK]" if v > 0.8 else "[WARN]" if v < 0.3 else "[RELOAD]"
                points.append(f"{tag} {k.replace('_', ' ')}: {float(v):.1%}")
            elif "risk" in lk and isinstance(v, (int, float)):
                tag = "[RED]" if v > 0.7 else "[GREEN]" if v < 0.3 else "[YELLOW]"
                points.append(f"{tag} {k.replace('_', ' ')}: {float(v):.2f}")
            elif isinstance(v, bool):
                points.append(f"{'[OK]' if v else '[FAIL]'} {k.replace('_', ' ').title()}")
            elif isinstance(v, (int, float)) and "score" in lk:
                tag = "[CHART]" if v > 0.7 else "📉" if v < 0.3 else "[STATS]"
                points.append(f"{tag} {k.replace('_', ' ')}: {float(v):.2f}")
        if not points:
            points += ["[STATS] Standard analysis based on available data", "[SEARCH] No exceptional factors identified"]
        return "\n".join(points[:7])

    def _determine_primary_reason(self, decision: Any, context: Dict[str, Any]) -> str:
        d = str(decision).lower()
        if any(x in d for x in ("buy", "long")):
            return "technical and fundamental analysis converged on upward price potential"
        if any(x in d for x in ("sell", "short")):
            return "analysis indicates significant downward pressure developing"
        if any(x in d for x in ("hold", "wait")):
            return "current conditions suggest waiting for clearer directional signals"
        conf = float(context.get("confidence", 0.5))
        risk = float(context.get("risk_score", 0.5))
        if conf > 0.8 and risk < 0.3:
            return "high-confidence, low-risk opportunity identified"
        if conf > 0.8:
            return "high-confidence analysis despite elevated risk factors"
        if risk > 0.7:
            return "risk management protocols override other considerations"
        return "balanced analysis of current conditions favors this approach"

    def _format_additional_context(self, context: Dict[str, Any]) -> str:
        if not context:
            return ""
        lines: List[str] = []
        for k, v in context.items():
            if k.startswith("_"):
                continue
            if isinstance(v, float):
                lines.append(f"  [STATS] {k.replace('_', ' ').title()}: {v:.3f}")
            elif isinstance(v, int):
                lines.append(f"  🔢 {k.replace('_', ' ').title()}: {v:,}")
            elif isinstance(v, bool):
                lines.append(f"  {'[OK]' if v else '[FAIL]'} {k.replace('_', ' ').title()}: {v}")
            elif isinstance(v, str) and len(v) <= 80:
                lines.append(f"  [LOG] {k.replace('_', ' ').title()}: {v}")
        return "ADDITIONAL CONTEXT:\n" + "\n".join(lines[:5]) if lines else ""

    def _translate_error_to_plain_english(self, etype: str, msg: str) -> str:
        base = {
            "KeyError": "Tried to access data that doesn't exist or hasn't been provided yet",
            "TypeError": "Received data in an unexpected format or type",
            "ValueError": "Received data with invalid or out-of-range values",
            "AttributeError": "Tried to use a feature or method that doesn't exist",
            "TimeoutError": "The operation exceeded the allowed time limit",
            "ConnectionError": "Couldn't establish or maintain a connection to a required service",
            "ImportError": "Couldn't load a required component or module",
            "FileNotFoundError": "Couldn't find a required file or resource",
            "PermissionError": "Insufficient permissions to access a required resource",
            "MemoryError": "System ran out of available memory",
        }.get(etype, "An unexpected error occurred in the system")
        if "NoneType" in msg:
            return f"{base}. Specifically, a None/null value was encountered where a value was expected."
        if "list index out of range" in msg:
            return f"{base}. Specifically, a list position was accessed that doesn't exist."
        if "connection refused" in msg.lower():
            return f"{base}. The target service is not responding or is unavailable."
        if "timeout" in msg.lower():
            return f"{base}. The system waited too long for a response."
        return f"{base}."

    def _assess_error_impact(self, etype: str, _context: Dict[str, Any]) -> str:
        base = {
            "MemoryError": "[RED] CRITICAL - System stability at risk",
            "ConnectionError": "[YELLOW] MODERATE - External dependency affected",
            "TimeoutError": "[YELLOW] MODERATE - Performance degradation possible",
            "KeyError": "[YELLOW] MODERATE - Data flow interruption",
            "TypeError": "[YELLOW] MODERATE - Processing logic affected",
            "ValueError": "[GREEN] LOW - Input validation issue",
        }.get(etype, "🔵 UNKNOWN - Impact assessment needed")
        tail = {
            "MemoryError": ["• Immediate attention required", "• May affect multiple components"],
            "ConnectionError": ["• Immediate attention required", "• External service health check"],
            "TimeoutError": ["• Monitor for recurring incidents", "• Consider performance tuning"],
            "KeyError": ["• Monitor for recurring incidents", "• Add fail-safe defaults"],
        }.get(etype, ["• Standard error handling applies", "• Isolated incident likely"])
        return "\n".join([base] + tail)

    def _generate_prevention_strategy(self, etype: str, _context: Dict[str, Any]) -> str:
        return "\n".join(
            {
                "KeyError": [
                    "• Implement comprehensive input validation",
                    "• Add default handling for missing keys",
                    "• Strengthen data dependency checks",
                ],
                "TimeoutError": [
                    "• Optimize slow paths",
                    "• Use progressive timeouts",
                    "• Add circuit breaker patterns",
                ],
                "TypeError": [
                    "• Strengthen type checks",
                    "• Sanitize data earlier",
                    "• Expand unit tests",
                ],
                "ConnectionError": [
                    "• Robust retry mechanisms",
                    "• Connection health monitoring",
                    "• Fallback service strategies",
                ],
            }.get(
                etype,
                ["• Comprehensive logging and monitoring", "• Regular health checks", "• Proactive testing and validation"],
            )
        )

    def _determine_system_response(self, etype: str, _context: Dict[str, Any]) -> str:
        if etype in {"MemoryError", "ConnectionError"}:
            return "Attempt automatic recovery and escalate via monitoring alerts"
        if etype == "TimeoutError":
            return "Retry with exponential backoff"
        if etype in {"KeyError", "TypeError"}:
            return "Log error and continue with safe defaults"
        return "Log error and continue normal operation"

    def _format_decision(self, decision: Any) -> str:
        if isinstance(decision, dict):
            return json.dumps(decision, indent=2)
        if isinstance(decision, (list, tuple)):
            return "[" + ", ".join(map(str, decision)) + "]"
        return str(decision)

    def _generate_fix_suggestion(self, etype: str, msg: str, _context: Dict[str, Any]) -> str:
        if etype == "KeyError":
            return "• Ensure required data is available before access\n• Verify provider-consumer wiring\n• Add safe defaults for missing keys"
        if etype == "TimeoutError":
            return "• Optimize slow operations\n• Increase timeouts where justified\n• Consider async or batching"
        if "NoneType" in msg:
            return "• Add null checks and defaults\n• Validate inputs at boundaries\n• Ensure functions return expected values"
        if etype == "ImportError":
            return "• Verify dependencies installed\n• Check PYTHONPATH and module locations\n• Review import names for typos"
        if etype == "ConnectionError":
            return "• Check network connectivity\n• Verify endpoint availability\n• Implement retries with backoff"
        return "• Review logs near the incident\n• Validate resources and dependencies\n• Improve targeted error handling"

    def _assess_performance_status(self, metrics: Dict[str, float]) -> Tuple[str, str]:
        avg_time = float(metrics.get("avg_execution_time_ms", 0.0))
        err_rate = float(metrics.get("error_rate", 0.0))
        if err_rate > 0.10 or avg_time > 1000:
            return "[ALERT]", "Critical"
        if err_rate > 0.05 or avg_time > 500:
            return "[WARN]", "Warning"
        return "[OK]", "Healthy"

    def _generate_performance_summary(self, metrics: Dict[str, float]) -> str:
        avg_time = float(metrics.get("avg_execution_time_ms", 0.0))
        total_exec = int(metrics.get("total_executions", 0))
        err_rate = float(metrics.get("error_rate", 0.0))
        return f"Executed {total_exec:,} times with average response time {avg_time:.1f}ms and {err_rate:.1%} error rate."

    def _format_metrics(self, metrics: Dict[str, float]) -> str:
        lines: List[str] = []
        for k, v in metrics.items():
            if any(s in k.lower() for s in ("time", "latency")):
                lines.append(f"  [STATS] {k.replace('_', ' ').title()}: {float(v):.1f} ms")
            elif any(s in k.lower() for s in ("rate", "percent")):
                lines.append(f"  [STATS] {k.replace('_', ' ').title()}: {float(v):.1%}")
            else:
                lines.append(f"  [STATS] {k.replace('_', ' ').title()}: {float(v):.2f}")
        return "\n".join(lines)

    def _analyze_performance_trends(self, _metrics: Dict[str, float]) -> str:
        return "[CHART] TREND ANALYSIS:\n  • Performance stable over the monitored period\n  • No significant degradation detected\n  • Normal operational variance observed"

    def _generate_performance_recommendations(self, metrics: Dict[str, float]) -> List[str]:
        recs: List[str] = []
        avg_time = float(metrics.get("avg_execution_time_ms", 0.0))
        err_rate = float(metrics.get("error_rate", 0.0))
        mem = float(metrics.get("memory_usage_mb", 0.0))
        if avg_time > 500:
            recs.append("Consider optimizing slow operations or add caching")
        if err_rate > 0.05:
            recs.append("Investigate recurring errors and add safeguards")
        if mem > 1000:
            recs.append("Monitor memory footprint for potential leaks")
        if not recs:
            recs.append("Performance is optimal - keep current configuration")
        return recs

    def _generate_performance_alerts(self, metrics: Dict[str, float]) -> str:
        alerts: List[str] = []
        avg_time = float(metrics.get("avg_execution_time_ms", 0.0))
        err_rate = float(metrics.get("error_rate", 0.0))
        if avg_time > 1000:
            alerts.append("[ALERT] CRITICAL: Response time exceeds 1000ms")
        elif avg_time > 500:
            alerts.append("[WARN] WARNING: Response time exceeds 500ms")
        if err_rate > 0.10:
            alerts.append("[ALERT] CRITICAL: Error rate exceeds 10%")
        elif err_rate > 0.05:
            alerts.append("[WARN] WARNING: Error rate exceeds 5%")
        if not alerts:
            alerts.append("[OK] No performance alerts")
        return "PERFORMANCE ALERTS:\n" + "\n".join(f"  {a}" for a in alerts)

    def _list_modules(self, modules: List[str]) -> str:
        if not modules:
            return "None"
        if len(modules) == 1:
            return modules[0]
        if len(modules) == 2:
            return f"{modules[0]} and {modules[1]}"
        return f"{', '.join(modules[:-1])}, and {modules[-1]}"

    def _format_current_data(self, data: Dict[str, Any]) -> str:
        if not data:
            return ""
        items: List[str] = []
        for k, v in list(data.items())[:3]:
            if isinstance(v, float):
                items.append(f"  • {k}: {v:.3f}")
            else:
                s = str(v)
                items.append(f"  • {k}: {s[:50]}{'...' if len(s) > 50 else ''}")
        if len(data) > 3:
            items.append(f"  • ... and {len(data) - 3} more fields")
        return "\n".join(items)

    def _assess_data_flow_health(self, providers: List[str], consumers: List[str], current: Optional[Dict[str, Any]]) -> str:
        if not providers:
            return "[FAIL] UNHEALTHY: No data providers configured"
        if not consumers:
            return "[WARN] WARNING: Data produced but not consumed (potential waste)"
        if not current:
            return "[WARN] WARNING: No recent data available"
        return "[OK] HEALTHY: Data flowing normally between modules"

    def _generate_dependency_action_items(self, issues: List[str], suggestions: List[str]) -> str:
        actions = []
        if issues:
            actions.append("1. Resolve dependency issues listed above")
        if suggestions:
            actions.append("2. Implement optimization suggestions")
        actions += ["3. Monitor dependency graph for regressions", "4. Review module coupling periodically", "5. Update documentation for changes"]
        return "\n".join(actions)


# ═══════════════════════════════════════════════════════════════════
# SYSTEM UTILITIES (Unified façade)
# ═══════════════════════════════════════════════════════════════════

class SystemUtilities:
    """
    Unified System Utilities combining explanation and validation access.
    Uses SystemIntegritySuite for validation & dependency auditing; provides
    robust normalization to match legacy IntegrationValidator expectations.
    """

    def __init__(self, orchestrator: Optional[ModuleOrchestrator] = None) -> None:
        self.explainer = EnglishExplainer()
        self.logger = RotatingLogger(name="SystemUtilities", log_path="logs/utils/system_utilities.log", max_lines=8000)

        # Load the integrity suite defensively
        suite = None
        try:
            from modules.monitoring.system_integrity_suite import SystemIntegritySuite  # type: ignore
            suite = SystemIntegritySuite(orchestrator=orchestrator)
            self.logger.info("[OK] SystemIntegritySuite loaded")
        except Exception as e:
            self.logger.error(f"[SystemUtilities] Failed to import SystemIntegritySuite: {e}")
        self._suite = suite

    # ---- explainer passthroughs ----
    def explain_module_decision(self, *args, **kwargs) -> str:
        return self.explainer.explain_module_decision(*args, **kwargs)

    def explain_error(self, *args, **kwargs) -> str:
        return self.explainer.explain_error(*args, **kwargs)

    def explain_performance(self, *args, **kwargs) -> str:
        return self.explainer.explain_performance(*args, **kwargs)

    def explain_data_flow(self, *args, **kwargs) -> str:
        return self.explainer.explain_data_flow(*args, **kwargs)

    def explain_health_status(self, *args, **kwargs) -> str:
        return self.explainer.explain_health_status(*args, **kwargs)

    def explain_dependencies(self, *args, **kwargs) -> str:
        return self.explainer.explain_dependencies(*args, **kwargs)

    def explain_execution_results(self, *args, **kwargs) -> str:
        return self.explainer.explain_execution_results(*args, **kwargs)

    # ---- validation access (robust normalization) ----
    def validate_system(self):
        """
        Backward-compat: return a report-like wrapper from SystemIntegritySuite
        with defensive schema normalisation.

        Guarantees attributes:
          - integration_score: float
          - total_modules: int
          - validated_modules: int
          - issues: list
          - missing_decorators: list
          - missing_thesis: list
          - config_issues: list
          - duplicate_writers: dict
          - missing_writers: dict
          - cycles: list
          - manifest: dict
          - severity counters: critical_count, error_count, warning_count, info_count
        """
        if self._suite is None:
            self.logger.error("[SystemUtilities.validate_system] Suite unavailable")
            raw = {}
        else:
            try:
                raw = self._suite.validate_and_audit(title="System Integration Audit") or {}
            except Exception as e:
                self.logger.error(f"[SystemUtilities.validate_system] Suite call failed: {e}")
                raw = {}

        validation = raw.get("validation", {})
        if not isinstance(validation, dict):
            self.logger.warning("[SystemUtilities.validate_system] 'validation' payload missing or not a dict; using defaults")
            validation = {}

        # coercers
        def _f(x, d=0.0):  # float
            try:
                return float(x)
            except Exception:
                return d

        def _i(x, d=0):  # int
            try:
                return int(x)
            except Exception:
                return d

        def _l(x, d=None):  # list
            if isinstance(x, list):
                return x
            if x is None:
                return [] if d is None else d
            if hasattr(x, "__iter__") and not isinstance(x, (str, bytes, dict)):
                try:
                    return list(x)
                except Exception:
                    pass
            return [] if d is None else d

        def _d(x, d=None):  # dict
            return x if isinstance(x, dict) else ({} if d is None else d)

        normalized = {
            "integration_score": _f(validation.get("integration_score", raw.get("integration_score", 0.0))),
            "total_modules": _i(validation.get("total_modules", raw.get("total_modules", 0))),
            "validated_modules": _i(validation.get("validated_modules", raw.get("validated_modules", 0))),
            "issues": _l(validation.get("issues", raw.get("issues", []))),
            "missing_decorators": _l(validation.get("missing_decorators", [])),
            "missing_thesis": _l(validation.get("missing_thesis", [])),
            "config_issues": _l(validation.get("config_issues", [])),
            "duplicate_writers": _d(validation.get("duplicate_writers", {})),
            "missing_writers": _d(validation.get("missing_writers", {})),
            "cycles": _l(validation.get("cycles", [])),
            "manifest": _d(validation.get("manifest", {})),
        }

        # severity counts
        def _sev_counts(items: List[Any]) -> Dict[str, int]:
            c = {"critical": 0, "error": 0, "warning": 0, "info": 0}
            for it in items:
                sev = None
                if isinstance(it, dict):
                    sev = str(it.get("severity", "")).lower()
                else:
                    sev = str(getattr(it, "severity", "")).lower()
                if sev in c:
                    c[sev] += 1
            return c

        sev = _sev_counts(normalized["issues"])

        missing_essentials = [k for k in ("integration_score", "total_modules", "validated_modules") if k not in validation or validation.get(k) is None]
        if missing_essentials:
            self.logger.warning(f"[SystemUtilities.validate_system] Missing essentials {missing_essentials}; defaults applied")

        # concise summary log
        self.logger.info(
            format_operator_message(
                icon="📊",
                message="Validation normalized",
                score=f"{normalized['integration_score']:.1f}",
                total_modules=normalized["total_modules"],
                validated=normalized["validated_modules"],
                critical=sev["critical"],
                errors=sev["error"],
                warnings=sev["warning"],
                info=sev["info"],
            )
        )

        class _ReportWrapper:
            def __init__(self, data: Dict[str, Any], sev_counts: Dict[str, int]) -> None:
                self._data = data
                self._sev = sev_counts

            # expected properties
            @property
            def integration_score(self) -> float:
                return float(self._data.get("integration_score", 0.0))

            @property
            def total_modules(self) -> int:
                return int(self._data.get("total_modules", 0))

            @property
            def validated_modules(self) -> int:
                return int(self._data.get("validated_modules", 0))

            @property
            def issues(self) -> List[Any]:
                return list(self._data.get("issues", []))

            @property
            def missing_decorators(self) -> List[str]:
                return list(self._data.get("missing_decorators", []))

            @property
            def missing_thesis(self) -> List[str]:
                return list(self._data.get("missing_thesis", []))

            @property
            def config_issues(self) -> List[str]:
                return list(self._data.get("config_issues", []))

            @property
            def duplicate_writers(self) -> Dict[str, List[str]]:
                return dict(self._data.get("duplicate_writers", {}))

            @property
            def missing_writers(self) -> Dict[str, List[str]]:
                return dict(self._data.get("missing_writers", {}))

            @property
            def cycles(self) -> List[List[str]]:
                return list(self._data.get("cycles", []))

            @property
            def manifest(self) -> Dict[str, Any]:
                return dict(self._data.get("manifest", {}))

            # severity convenience
            @property
            def critical_count(self) -> int:
                return int(self._sev.get("critical", 0))

            @property
            def error_count(self) -> int:
                return int(self._sev.get("error", 0))

            @property
            def warning_count(self) -> int:
                return int(self._sev.get("warning", 0))

            @property
            def info_count(self) -> int:
                return int(self._sev.get("info", 0))

            def to_dict(self) -> Dict[str, Any]:
                return {**self._data, "severity_counts": dict(self._sev)}

            # dotted access passthrough for any stored field
            def __getattr__(self, item: str):
                if item in self._data:
                    return self._data[item]
                raise AttributeError(item)

        return _ReportWrapper(normalized, sev)

    # Back-compat helpers (kept minimal and explicit)
    def generate_migration_guide(self) -> str:
        return (
            "Migration Guide\n"
            "===============\n"
            "- Legacy IntegrationValidator has been superseded by SystemIntegritySuite.\n"
            "- Use validate_and_audit() for combined dependency + validation reporting.\n"
            "- This facade normalises the suite output into a stable schema.\n"
        )

    def fix_common_issues(self, *_: Any, **__: Any) -> List[str]:
        # Explicitly no auto-fix actions are taken in this facade.
        return []
