"""Command-line interface for clinaudit."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from .scanner import VERSION, audit_directory


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clinaudit",
        description=(
            "clinaudit: Lightweight zero-heavy-dependency forensic auditor for clinical AI "
            "splits, evaluation pipelines, Source Data statistics, and deployed risk calculators."
        ),
    )
    parser.add_argument("--version", action="version", version=f"clinaudit {VERSION}")

    sub = parser.add_subparsers(dest="command", required=True)
    scan_p = sub.add_parser(
        "scan",
        help="Audit a directory or single file across all 4 clinaudit modules.",
    )
    scan_p.add_argument("target", help="Directory or file path to audit.")
    scan_p.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Optional output path to write the structured JSON audit report.",
    )
    scan_p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Scan only the top-level directory without recursing into subdirectories.",
    )
    scan_p.add_argument(
        "--tol",
        type=float,
        default=1e-10,
        help="L_infinity tolerance for near-exact feature-vector leakage matching (default: 1e-10).",
    )
    scan_p.add_argument(
        "--fail-on-critical",
        action="store_true",
        help="Return non-zero exit code (2) if any CRITICAL alert is detected.",
    )
    scan_p.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress human-readable stdout summary.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "scan":
        report = audit_directory(
            args.target,
            recursive=not args.no_recursive,
            tol=args.tol,
        )
        if args.out:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(report.to_json(indent=2), encoding="utf-8")

        summary = report.summary
        if not args.quiet:
            bp = summary["by_priority"]
            print(f"clinaudit v{report.version} | target: {report.target}")
            print(
                f"Files scanned: {summary['files_scanned']} | "
                f"Alerts: {summary['total_alerts']} "
                f"(P0={bp['P0']}, P1={bp['P1']}, P2={bp['P2']}; "
                f"CRITICAL={summary['by_severity']['CRITICAL']}, "
                f"HIGH={summary['by_severity']['HIGH']}, "
                f"MEDIUM={summary['by_severity']['MEDIUM']}) | "
                f"Runtime: {report.total_runtime_ms:.2f} ms"
            )
            for alert in report.alerts:
                loc = f" ({alert.location})" if alert.location else ""
                finfo = f" [{alert.file}]" if alert.file else ""
                print(
                    f"  [{alert.priority}|{alert.severity}] {alert.rule_id} "
                    f"({alert.module}){finfo}{loc}: {alert.title}"
                )
                print(f"      {alert.message}")
            if args.out:
                print(f"JSON report written to: {args.out}")

        if args.fail_on_critical and summary["by_severity"]["CRITICAL"] > 0:
            return 2
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
