"""Unified multi-module forensic audit runner for directories and file lists."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Sequence

from .deploycheck import scan_deploy_code_document
from .evalcheck import audit_dca_table, scan_code_document
from .io_utils import TableData, read_code_file, read_tables
from .leakage import audit_tabular_split_tables, audit_xlsx_splits
from .models import AuditAlert, AuditReport
from .statcheck import audit_cross_table_ci_consistency, audit_table_stats

VERSION = "0.2.0"

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    "env",
    "node_modules",
    ".ipynb_checkpoints",
}

_CODE_EXTS = {".py", ".ipynb", ".r", ".js"}
_TABLE_EXTS = {".xlsx", ".csv", ".tsv"}

_TRAIN_FILE_RE = re.compile(
    r"(?:^|[_\-\s])(train|training|dev|development|discovery)(?:$|[_\-\s])",
    re.IGNORECASE,
)
_TEST_FILE_RE = re.compile(
    r"(?:^|[_\-\s])(test|testing|holdout|held_?out|external|val|validation)(?:$|[_\-\s])",
    re.IGNORECASE,
)


def _pair_train_test_files(table_files: list[Path]) -> list[tuple[Path, Path]]:
    """Pair train and test CSV/TSV files residing in the same directory."""
    by_dir: dict[Path, list[Path]] = {}
    for p in table_files:
        if p.suffix.lower() in (".csv", ".tsv"):
            by_dir.setdefault(p.parent, []).append(p)

    pairs: list[tuple[Path, Path]] = []
    for _, files in sorted(by_dir.items()):
        trains = [p for p in files if _TRAIN_FILE_RE.search(p.stem)]
        tests = [p for p in files if _TEST_FILE_RE.search(p.stem) and p not in trains]
        for tr in trains:
            tr_stem_norm = _TRAIN_FILE_RE.sub("_", tr.stem).strip("_").lower()
            matched_te = None
            for te in tests:
                te_stem_norm = _TEST_FILE_RE.sub("_", te.stem).strip("_").lower()
                if tr_stem_norm == te_stem_norm:
                    matched_te = te
                    break
            if matched_te is not None:
                pairs.append((tr, matched_te))
            elif len(trains) == 1 and len(tests) == 1:
                pairs.append((tr, tests[0]))
    return pairs


def discover_auditable_files(path: str | Path, recursive: bool = True) -> list[Path]:
    """Discover all supported code and tabular files under a directory or single file."""
    root = Path(path)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Audit target does not exist: {root}")

    supported = _CODE_EXTS | _TABLE_EXTS
    found: list[Path] = []
    iterator = root.rglob("*") if recursive else root.glob("*")
    for p in iterator:
        if not p.is_file():
            continue
        if any(part in _SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() in supported:
            found.append(p)
    found.sort()
    return found


def audit_files(
    files: Sequence[str | Path],
    target_label: str | None = None,
    tol: float = 1e-10,
    skipped_files: Sequence[str | Path] | None = None,
) -> AuditReport:
    """Run all 4 clinaudit modules across an explicit sequence of files."""
    t_start = time.perf_counter()
    paths = [Path(f) for f in files if Path(f).is_file()]
    supported = _CODE_EXTS | _TABLE_EXTS
    scanned_paths = [p for p in paths if p.suffix.lower() in supported]
    unsupported_paths = [p for p in paths if p.suffix.lower() not in supported]
    skipped_strs = [str(Path(f)) for f in (skipped_files or [])] + [str(p) for p in unsupported_paths]
    file_strs = [str(p) for p in scanned_paths]
    parse_errors: list[dict[str, str]] = []

    alerts: list[AuditAlert] = []
    runtimes: dict[str, float] = {
        "leakage": 0.0,
        "evalcheck": 0.0,
        "statcheck": 0.0,
        "deploycheck": 0.0,
    }

    code_files = [p for p in scanned_paths if p.suffix.lower() in _CODE_EXTS]
    table_files = [p for p in scanned_paths if p.suffix.lower() in _TABLE_EXTS]

    # Parse tabular files once and reuse across modules 1, 2, and 3
    loaded_tables_by_file: dict[Path, list[TableData]] = {}
    all_tables: list[TableData] = []
    for tf in table_files:
        try:
            tbls = read_tables(tf)
            loaded_tables_by_file[tf] = tbls
            all_tables.extend(tbls)
        except Exception as exc:
            parse_errors.append({"file": str(tf), "stage": "read_tables", "error": f"{type(exc).__name__}: {exc}"})
            continue

    # --- Module 1: Split Integrity & Mirror-Clone Detector ---
    t0 = time.perf_counter()
    for tf in table_files:
        if tf.suffix.lower() == ".xlsx":
            try:
                alerts.extend(audit_xlsx_splits(tf, tol=tol))
            except Exception as exc:
                parse_errors.append({"file": str(tf), "stage": "audit_xlsx_splits", "error": f"{type(exc).__name__}: {exc}"})
    for tr_path, te_path in _pair_train_test_files(table_files):
        tr_tbls = loaded_tables_by_file.get(tr_path)
        te_tbls = loaded_tables_by_file.get(te_path)
        if tr_tbls and te_tbls:
            pair_alerts, _ = audit_tabular_split_tables(tr_tbls[0], te_tbls[0], tol=tol)
            alerts.extend(pair_alerts)
    runtimes["leakage"] = (time.perf_counter() - t0) * 1000.0

    # --- Parse code documents once for Modules 2 & 4 ---
    code_docs = []
    for cf in code_files:
        try:
            code_docs.append(read_code_file(cf))
        except Exception as exc:
            parse_errors.append({"file": str(cf), "stage": "read_code_file", "error": f"{type(exc).__name__}: {exc}"})
            continue

    # --- Module 2: Clinical Evaluation & Baseline Integrity Scanner ---
    t0 = time.perf_counter()
    for doc in code_docs:
        alerts.extend(scan_code_document(doc))
    for tbl in all_tables:
        alerts.extend(audit_dca_table(tbl))
    runtimes["evalcheck"] = (time.perf_counter() - t0) * 1000.0

    # --- Module 3: Statistical Inference & CI Sanity Auditor ---
    t0 = time.perf_counter()
    for tbl in all_tables:
        t_alerts, _ = audit_table_stats(tbl)
        alerts.extend(t_alerts)
    # Cross-sheet consistency within each workbook AND cross-table across directory
    if len(all_tables) >= 2:
        cross_alerts, _ = audit_cross_table_ci_consistency(all_tables)
        alerts.extend(cross_alerts)
    runtimes["statcheck"] = (time.perf_counter() - t0) * 1000.0

    # --- Module 4: Deployed Calculator Static Verification ---
    t0 = time.perf_counter()
    for doc in code_docs:
        alerts.extend(scan_deploy_code_document(doc))
    runtimes["deploycheck"] = (time.perf_counter() - t0) * 1000.0

    total_ms = (time.perf_counter() - t_start) * 1000.0
    return AuditReport(
        tool="clinaudit",
        version=VERSION,
        target=target_label or (file_strs[0] if len(file_strs) == 1 else f"{len(file_strs)} files"),
        files_scanned=file_strs,
        alerts=alerts,
        module_runtimes_ms=runtimes,
        total_runtime_ms=total_ms,
        files_skipped=sorted(set(skipped_strs)),
        parse_errors=parse_errors,
    )


def audit_directory(
    path: str | Path,
    recursive: bool = True,
    tol: float = 1e-10,
) -> AuditReport:
    """Discover and audit all supported clinical AI artifacts under a directory or file."""
    root = Path(path)
    files = discover_auditable_files(root, recursive=recursive)
    skipped: list[Path] = []
    if root.is_dir():
        supported = _CODE_EXTS | _TABLE_EXTS
        iterator = root.rglob("*") if recursive else root.glob("*")
        for p in iterator:
            if p.is_file() and not any(part in _SKIP_DIRS for part in p.parts) and p.suffix.lower() not in supported:
                skipped.append(p)
        skipped.sort()
    return audit_files(files, target_label=str(path), tol=tol, skipped_files=skipped)
