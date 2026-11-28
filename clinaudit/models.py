"""Core data structures for clinaudit forensic audit alerts and reports."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal

Severity = Literal["CRITICAL", "HIGH", "MEDIUM"]
Priority = Literal["P0", "P1", "P2"]

SEVERITY_RANK: dict[str, int] = {
    "CRITICAL": 3,
    "HIGH": 2,
    "MEDIUM": 1,
}

PRIORITY_RANK: dict[str, int] = {
    "P0": 3,
    "P1": 2,
    "P2": 1,
}


@dataclass(slots=True)
class AuditAlert:
    """Single structured forensic finding emitted by an audit module."""

    module: str
    rule_id: str
    severity: Severity
    title: str
    message: str
    file: str | None = None
    location: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    priority: Priority = "P1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "rule_id": self.rule_id,
            "severity": self.severity,
            "priority": self.priority,
            "title": self.title,
            "message": self.message,
            "file": self.file,
            "location": self.location,
            "evidence": _json_safe(self.evidence),
        }


@dataclass(slots=True)
class AuditReport:
    """Aggregated forensic audit report across all enabled modules."""

    tool: str
    version: str
    target: str
    files_scanned: list[str]
    alerts: list[AuditAlert]
    module_runtimes_ms: dict[str, float]
    total_runtime_ms: float
    files_skipped: list[str] = field(default_factory=list)
    parse_errors: list[dict[str, str]] = field(default_factory=list)

    @property
    def summary(self) -> dict[str, Any]:
        by_severity: dict[str, int] = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0}
        by_priority: dict[str, int] = {"P0": 0, "P1": 0, "P2": 0}
        by_module: dict[str, int] = {
            "leakage": 0,
            "evalcheck": 0,
            "statcheck": 0,
            "deploycheck": 0,
        }
        for alert in self.alerts:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
            by_priority[alert.priority] = by_priority.get(alert.priority, 0) + 1
            by_module[alert.module] = by_module.get(alert.module, 0) + 1
        hc_alerts = by_priority["P0"] + by_priority["P1"]
        return {
            "total_alerts": len(self.alerts),
            "high_confidence_alerts": hc_alerts,
            "files_scanned": len(self.files_scanned),
            "files_skipped": len(self.files_skipped),
            "parse_errors": len(self.parse_errors),
            "flagged": len(self.alerts) > 0,
            "flagged_high_confidence": hc_alerts > 0,
            "max_severity": (
                max(self.alerts, key=lambda a: SEVERITY_RANK.get(a.severity, 0)).severity
                if self.alerts
                else None
            ),
            "by_severity": by_severity,
            "by_priority": by_priority,
            "by_module": by_module,
        }

    def to_dict(self) -> dict[str, Any]:
        sorted_alerts = sorted(
            self.alerts,
            key=lambda a: (
                -SEVERITY_RANK.get(a.severity, 0),
                a.module,
                a.file or "",
                a.location or "",
                a.rule_id,
            ),
        )
        return {
            "tool": self.tool,
            "version": self.version,
            "target": self.target,
            "files_scanned": list(self.files_scanned),
            "files_skipped": list(self.files_skipped),
            "parse_errors": list(self.parse_errors),
            "summary": self.summary,
            "module_runtimes_ms": {
                k: round(float(v), 4) for k, v in self.module_runtimes_ms.items()
            },
            "total_runtime_ms": round(float(self.total_runtime_ms), 4),
            "alerts": [a.to_dict() for a in sorted_alerts],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _json_safe(obj: Any) -> Any:
    """Convert NumPy scalars/arrays and non-standard objects to JSON-safe values."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"
        return obj
    # Avoid hard import at module load if possible, but numpy is our sole dep.
    type_name = type(obj).__module__
    if type_name.startswith("numpy"):
        if hasattr(obj, "item") and getattr(obj, "ndim", 0) == 0:
            return _json_safe(obj.item())
        if hasattr(obj, "tolist"):
            return _json_safe(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)
