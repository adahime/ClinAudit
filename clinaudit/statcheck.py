"""Module 3: Statistical Inference, Confidence Interval & P-Value Sanity Auditor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from .io_utils import TableData, read_tables
from .models import AuditAlert

# Regex patterns for inline CI cells
_INLINE_CI_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*"
    r"[\(\[]\s*(?:95%?\s*CI\s*[:=]?\s*)?"
    r"([+-]?\d+(?:\.\d+)?)\s*(?:[-–—,~]|to)\s*([+-]?\d+(?:\.\d+)?)\s*[\)\]]\s*$",
    re.IGNORECASE,
)

_INLINE_PM_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*(?:±|\+/-|\+-)\s*(\d+(?:\.\d+)?)\s*$"
)

_INTERVAL_ONLY_RE = re.compile(
    r"^\s*[\(\[]?\s*([+-]?\d+(?:\.\d+)?)\s*(?:[-–—,~]|to)\s*([+-]?\d+(?:\.\d+)?)\s*[\)\]]?\s*$",
    re.IGNORECASE,
)

_PVAL_SCI_RE = re.compile(
    r"^\s*([<>≤≥]?)\s*(\d+(?:\.\d+)?)\s*(?:[x×\*]\s*10\s*\^?\s*([+-]?\d+)|[eE]([+-]?\d+))?\s*$"
)

_ERR_OFFSET_COL_RE = re.compile(r"(?:^|[_\s\-])(?:yerr|xerr|err_?lo|err_?hi|error_?bar|sem|std_?err)(?:$|[_\s\-])", re.IGNORECASE)
_ASSOC_REG_COL_RE = re.compile(
    r"^(?:beta|exp_?coef|or|hr|odds_?ratio|hazard_?ratio|estimate|std_?err(?:or)?|z_?value|t_?stat|metabolite.*|assay_?value|protein.*|feature|gene)$",
    re.IGNORECASE,
)
_LO_COL_RE = re.compile(
    r"(?:^|[_\s\-])(?:ci_?lo(?:w|wer)?|lo(?:w|wer)?_?(?:95)?_?ci|lci|lower_?bound|ci_?min|low_?95|lower)(?:$|[_\s\-])",
    re.IGNORECASE,
)
_HI_COL_RE = re.compile(
    r"(?:^|[_\s\-])(?:ci_?hi(?:gh|gher)?|hi(?:gh|gher)?_?(?:95)?_?ci|up(?:per)?_?(?:95)?_?ci|uci|upper_?bound|ci_?max|high_?95|upper)(?:$|[_\s\-])",
    re.IGNORECASE,
)
_PM_COL_RE = re.compile(
    r"(?:^|[_\s\-])(?:ci_?half|half_?width|margin|ci_?margin|pm|plus_?minus|ci_?delta)(?:$|[_\s\-])",
    re.IGNORECASE,
)
_CI_STR_COL_RE = re.compile(r"(?:95%?\s*ci|^ci$|conf.*int)", re.IGNORECASE)
_PVAL_COL_RE = re.compile(
    r"^(?:p|p_?val(?:ue)?|pval|p-val(?:ue)?|p_?values?|adj_?p|fdr|q_?val(?:ue)?|"
    r"wilcoxon[_\s]?p(?:[_\s-]?val(?:ue)?)?|delong[_\s]?p(?:[_\s-]?val(?:ue)?)?|"
    r"mcnemar[_\s]?p(?:[_\s-]?val(?:ue)?)?|paired_?p|p_?diff|p_?comp(?:arison)?)$",
    re.IGNORECASE,
)
_N_COL_RE = re.compile(
    r"^(?:n|n_?total|n_?test|n_?samples?|n_?patients?|sample_?size|cohort_?size|total_?n)$",
    re.IGNORECASE,
)
_DELTA_COL_RE = re.compile(
    r"^(?:delta|diff|difference|delta_?auc|auc_?diff|delta_?metric|effect_?size|improvement|abs_?diff)$",
    re.IGNORECASE,
)
_UNIT_BOUNDED_METRIC_RE = re.compile(
    r"(?:auc|auroc|auprc|c_?index|concordance|acc(?:uracy)?|sens(?:itivity)?|"
    r"spec(?:ificity)?|ppv|npv|f1|recall|precision|brier|r2|r_?squared)",
    re.IGNORECASE,
)
_BOOTSTRAP_CANDIDATES = (50, 100, 200, 250, 500, 1000, 2000, 5000, 10000)
_Z_95 = 1.959963984540054


@dataclass(slots=True)
class CIRecord:
    """Single parsed confidence interval observation from a table row."""

    table_name: str
    source_file: str | None
    row_index: int
    row_key: str
    metric_name: str
    estimate: float | None
    lo: float
    hi: float
    decimals: int
    n_sample: float | None = None
    is_unit_bounded: bool = False
    is_pm_format: bool = False

    @property
    def width(self) -> float:
        return self.hi - self.lo

    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)


def _count_decimals(token: str) -> int:
    s = token.strip()
    if "e" in s.lower():
        return 4
    if "." in s:
        return len(s.split(".", 1)[1])
    return 0


def _normal_two_sided_p_log10(z: float) -> tuple[float, float]:
    """Exact two-sided standard normal P-value and log10(P) without float64 underflow."""
    az = abs(float(z))
    if not math.isfinite(az):
        return 0.0, -math.inf
    if az == 0.0:
        return 1.0, 0.0
    if az <= 35.0:
        p = math.erfc(az / math.sqrt(2.0))
        if p > 0.0:
            return min(1.0, p), math.log10(p)
    # Asymptotic Mill's ratio expansion: 2 * Phi(-z) ~ sqrt(2/pi) * exp(-z^2/2) / z * (1 - 1/z^2 + 3/z^4)
    inv_z2 = 1.0 / (az * az)
    series = max(1e-12, 1.0 - inv_z2 + 3.0 * inv_z2 * inv_z2)
    log_p = 0.5 * math.log(2.0 / math.pi) - 0.5 * az * az - math.log(az) + math.log(series)
    log10_p = log_p / math.log(10.0)
    p = math.exp(log_p) if log_p > -745.0 else 0.0
    return min(1.0, p), log10_p


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -math.inf
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def exact_binomial_test(
    k: int,
    n: int,
    p0: float = 0.5,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Exact binomial test computed in log-space using stdlib math.lgamma."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"Invalid binomial parameters k={k}, n={n}")
    if not (0.0 <= p0 <= 1.0):
        raise ValueError(f"p0 must be in [0, 1], got {p0}")
    if n == 0:
        return {"k": k, "n": n, "p0": p0, "p_value": 1.0, "log10_p_value": 0.0}

    def _log_pmf(j: int) -> float:
        if p0 == 0.0:
            return 0.0 if j == 0 else -math.inf
        if p0 == 1.0:
            return 0.0 if j == n else -math.inf
        return _log_comb(n, j) + j * math.log(p0) + (n - j) * math.log1p(-p0)

    def _logsumexp(vals: list[float]) -> float:
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            return -math.inf
        m = max(finite)
        return m + math.log(sum(math.exp(v - m) for v in finite))

    if alternative == "greater":
        log_p = _logsumexp([_log_pmf(j) for j in range(k, n + 1)])
    elif alternative == "less":
        log_p = _logsumexp([_log_pmf(j) for j in range(0, k + 1)])
    elif alternative == "two-sided":
        if abs(p0 - 0.5) < 1e-12:
            tail_k = max(k, n - k)
            log_tail = _logsumexp([_log_pmf(j) for j in range(tail_k, n + 1)])
            log_p = min(0.0, log_tail + math.log(2.0))
        else:
            lp_k = _log_pmf(k)
            log_p = _logsumexp([_log_pmf(j) for j in range(0, n + 1) if _log_pmf(j) <= lp_k + 1e-12])
    else:
        raise ValueError(f"Unknown alternative: {alternative}")

    log_p = min(0.0, log_p)
    p_val = math.exp(log_p) if log_p > -745.0 else 0.0
    return {
        "k": k,
        "n": n,
        "p0": p0,
        "alternative": alternative,
        "p_value": min(1.0, p_val),
        "log10_p_value": log_p / math.log(10.0) if math.isfinite(log_p) else -math.inf,
    }


def exact_mcnemar_test(b: int, c: int) -> dict[str, Any]:
    """Exact two-sided McNemar binomial test plus Edwards continuity-corrected chi2."""
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative: b={b}, c={c}")
    d = b + c
    if d == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "p_value": 1.0,
            "log10_p_value": 0.0,
            "chi2_edwards": 0.0,
        }
    res = exact_binomial_test(max(b, c), d, p0=0.5, alternative="two-sided")
    chi2_cc = (max(0.0, abs(b - c) - 1.0) ** 2) / float(d)
    return {
        "b": b,
        "c": c,
        "n_discordant": d,
        "p_value": res["p_value"],
        "log10_p_value": res["log10_p_value"],
        "chi2_edwards": chi2_cc,
    }


def min_mcnemar_p_for_net_diff(net_diff_count: int) -> tuple[float, float]:
    """Smallest mathematically possible two-sided McNemar P-value for net difference |b - c|."""
    d = max(0, int(net_diff_count))
    if d == 0:
        return 1.0, 0.0
    # Achieved when all discordant pairs go in one direction: b = d, c = 0 -> P = min(1, 2 * 2^-d)
    log10_p = min(0.0, (1 - d) * math.log10(2.0))
    p = min(1.0, 10.0**log10_p) if log10_p > -323.0 else 0.0
    return p, log10_p


def paired_z_sensitivity_check(
    delta: float,
    se1: float | None = None,
    se2: float | None = None,
    se_diff: float | None = None,
    p_reported: float | None = None,
    rhos: tuple[float, ...] = (0.0, 0.5, 0.8, 0.9, 0.95),
) -> dict[str, Any]:
    """Normal-approximation paired-Z sensitivity grid across correlations rho."""
    abs_d = abs(float(delta))
    grid: dict[str, dict[str, float]] = {}

    if se_diff is not None and se_diff > 0.0:
        z = abs_d / float(se_diff)
        p, log10_p = _normal_two_sided_p_log10(z)
        grid["direct"] = {
            "se_diff": float(se_diff),
            "z": z,
            "p_value": p,
            "log10_p": log10_p,
        }
    elif se1 is not None and se2 is not None and se1 > 0.0 and se2 > 0.0:
        for rho in rhos:
            var_d = max(1e-24, se1 * se1 + se2 * se2 - 2.0 * rho * se1 * se2)
            sed = math.sqrt(var_d)
            z = abs_d / sed
            p, log10_p = _normal_two_sided_p_log10(z)
            grid[f"rho={rho:.2f}"] = {
                "rho": rho,
                "se_diff": sed,
                "z": z,
                "p_value": p,
                "log10_p": log10_p,
            }

    inconsistent = False
    max_rho_key = f"rho={max(rhos):.2f}" if rhos else "direct"
    ref_entry = grid.get("direct") or grid.get(max_rho_key)
    if p_reported is not None and ref_entry is not None and p_reported > 0.0:
        log10_rep = math.log10(p_reported)
        ref_p = ref_entry["p_value"]
        ref_log10_p = ref_entry["log10_p"]
        if p_reported < 1e-4 and (
            (ref_p > 0.05) or (log10_rep < ref_log10_p - 5.0)
        ):
            inconsistent = True
    elif p_reported == 0.0 and ref_entry is not None and ref_entry["p_value"] > 1e-4:
        inconsistent = True

    return {
        "delta": abs_d,
        "p_reported": p_reported,
        "sensitivity_grid": grid,
        "contradicts_reported_p": inconsistent,
    }


def _parse_p_value_cell(val: Any, raw_str: str = "") -> tuple[float | None, bool]:
    """Parse numeric or string P-value cell into (p_float, is_inequality_less_than)."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        f = float(val)
        if 0.0 <= f <= 1.0:
            return f, False
        return None, False
    s = (raw_str or (str(val) if val is not None else "")).strip()
    if not s:
        return None, False
    m = _PVAL_SCI_RE.match(s)
    if not m:
        return None, False
    op, mant_s, exp1, exp2 = m.group(1), m.group(2), m.group(3), m.group(4)
    try:
        mant = float(mant_s)
        exp = int(exp1) if exp1 is not None else (int(exp2) if exp2 is not None else 0)
        p = mant * (10.0**exp)
    except (ValueError, OverflowError):
        return None, False
    if not (0.0 <= p <= 1.0):
        return None, False
    is_lt = op in ("<", "≤")
    return p, is_lt


def _raw_cell(table: TableData, r_idx: int, c_idx: int) -> str:
    if r_idx < len(table.raw_rows) and c_idx < len(table.raw_rows[r_idx]):
        return str(table.raw_rows[r_idx][c_idx])
    if r_idx < len(table.rows) and c_idx < len(table.rows[r_idx]) and table.rows[r_idx][c_idx] is not None:
        return str(table.rows[r_idx][c_idx])
    return ""


def _to_float_or_none(val: Any) -> float | None:
    if val is None or isinstance(val, bool):
        return None
    if isinstance(val, (int, float)):
        f = float(val)
        return f if math.isfinite(f) else None
    try:
        f = float(str(val).strip())
        return f if math.isfinite(f) else None
    except ValueError:
        return None


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def extract_ci_records(table: TableData) -> list[CIRecord]:
    """Extract structured CI records from inline formatted cells or paired low/high columns."""
    if not table.headers or not table.rows:
        return []

    n_cols = len(table.headers)
    headers = table.headers

    # Identify sample-size column and key columns
    n_col_idx: int | None = None
    for j, h in enumerate(headers):
        if _N_COL_RE.match(h.strip()):
            n_col_idx = j
            break

    key_col_indices: list[int] = []
    for j, h in enumerate(headers):
        hl = h.strip().lower()
        # Never treat metric, CI, P-value, or sample-size columns as row-label key columns
        if (
            j == n_col_idx
            or _UNIT_BOUNDED_METRIC_RE.search(h)
            or _LO_COL_RE.search(h)
            or _HI_COL_RE.search(h)
            or _PM_COL_RE.search(h)
            or _CI_STR_COL_RE.search(h)
            or _PVAL_COL_RE.match(h.strip())
            or _DELTA_COL_RE.match(h.strip())
        ):
            continue
        # Also skip columns whose sample cells parse as inline CIs
        sample_cells = [_raw_cell(table, r, j).strip() for r in range(min(5, len(table.rows)))]
        if any(_INLINE_CI_RE.match(s) or _INLINE_PM_RE.match(s) for s in sample_cells if s):
            continue
        if any(
            k in hl
            for k in ("model", "method", "task", "cohort", "dataset", "endpoint", "subgroup", "name", "metric", "outcome", "algorithm", "comparison")
        ):
            key_col_indices.append(j)
    if not key_col_indices:
        # Use first predominantly-string non-CI column as row label
        for j in range(min(2, n_cols)):
            sample_cells = [_raw_cell(table, r, j).strip() for r in range(min(5, len(table.rows)))]
            if any(_INLINE_CI_RE.match(s) or _INLINE_PM_RE.match(s) for s in sample_cells if s):
                continue
            str_hits = sum(
                1 for row in table.rows[:20] if j < len(row) and isinstance(row[j], str)
            )
            if str_hits >= max(1, min(len(table.rows), 20) // 2):
                key_col_indices.append(j)
                break

    def _row_key(r_idx: int) -> str:
        parts = []
        row = table.rows[r_idx]
        for j in key_col_indices:
            if j < len(row) and row[j] is not None and str(row[j]).strip():
                parts.append(_normalize_key(str(row[j])))
        return "|".join(parts) if parts else f"row_{r_idx}"

    def _row_n(r_idx: int) -> float | None:
        if n_col_idx is None or n_col_idx >= len(table.rows[r_idx]):
            return None
        val = _to_float_or_none(table.rows[r_idx][n_col_idx])
        return val if (val is not None and val > 0) else None

    records: list[CIRecord] = []
    used_cols: set[int] = set()

    # Strategy 1: Paired low/high columns (e.g., AUC, CI_low, CI_high), excluding matplotlib error-bar offset columns (yerr_lower/upper)
    lo_cols = [j for j, h in enumerate(headers) if _LO_COL_RE.search(h) and not _ERR_OFFSET_COL_RE.search(h)]
    hi_cols = [j for j, h in enumerate(headers) if _HI_COL_RE.search(h) and not _ERR_OFFSET_COL_RE.search(h)]
    matched_pairs: list[tuple[int | None, int, int, str]] = []

    used_hi: set[int] = set()
    for lo_j in lo_cols:
        # Find nearest unmatched hi column (prefer matching prefix or immediately following)
        lo_prefix = _LO_COL_RE.sub("", headers[lo_j]).strip("_ -").lower()
        best_hi = None
        for hi_j in hi_cols:
            if hi_j in used_hi:
                continue
            hi_prefix = _HI_COL_RE.sub("", headers[hi_j]).strip("_ -").lower()
            if lo_prefix and lo_prefix == hi_prefix:
                best_hi = hi_j
                break
        if best_hi is None:
            cands = [h_j for h_j in hi_cols if h_j not in used_hi and h_j > lo_j]
            if cands:
                best_hi = min(cands)
        if best_hi is not None:
            used_hi.add(best_hi)
            # Find associated point estimate column (same prefix or immediately preceding numeric column)
            est_j: int | None = None
            metric_name = lo_prefix or headers[lo_j]
            if lo_prefix:
                for k_j, kh in enumerate(headers):
                    if k_j not in (lo_j, best_hi) and kh.strip().lower() == lo_prefix:
                        est_j = k_j
                        metric_name = kh
                        break
            if est_j is None and lo_j > 0 and (lo_j - 1) not in used_cols and (lo_j - 1) not in key_col_indices:
                cand_j = lo_j - 1
                num_hits = sum(1 for row in table.rows[:20] if cand_j < len(row) and _to_float_or_none(row[cand_j]) is not None)
                if num_hits >= max(1, min(len(table.rows), 20) // 2) and cand_j != n_col_idx:
                    est_j = cand_j
                    metric_name = headers[cand_j]
            matched_pairs.append((est_j, lo_j, best_hi, metric_name))
            used_cols.update({lo_j, best_hi})
            if est_j is not None:
                used_cols.add(est_j)

    for est_j, lo_j, hi_j, metric_name in matched_pairs:
        is_unit = bool(_UNIT_BOUNDED_METRIC_RE.search(metric_name))
        for r_idx, row in enumerate(table.rows):
            lo_v = _to_float_or_none(row[lo_j] if lo_j < len(row) else None)
            hi_v = _to_float_or_none(row[hi_j] if hi_j < len(row) else None)
            if lo_v is None or hi_v is None:
                continue
            est_v = _to_float_or_none(row[est_j]) if (est_j is not None and est_j < len(row)) else None
            dec = max(
                _count_decimals(_raw_cell(table, r_idx, lo_j)),
                _count_decimals(_raw_cell(table, r_idx, hi_j)),
            )
            records.append(
                CIRecord(
                    table_name=table.name,
                    source_file=table.source_file,
                    row_index=r_idx,
                    row_key=_row_key(r_idx),
                    metric_name=metric_name,
                    estimate=est_v,
                    lo=lo_v,
                    hi=hi_v,
                    decimals=dec,
                    n_sample=_row_n(r_idx),
                    is_unit_bounded=is_unit,
                )
            )

    # Strategy 2: Estimate + margin/half-width columns
    pm_cols = [j for j, h in enumerate(headers) if j not in used_cols and _PM_COL_RE.search(h)]
    for pm_j in pm_cols:
        est_j = pm_j - 1 if pm_j > 0 and (pm_j - 1) not in used_cols and (pm_j - 1) not in key_col_indices else None
        if est_j is None:
            continue
        metric_name = headers[est_j]
        is_unit = bool(_UNIT_BOUNDED_METRIC_RE.search(metric_name))
        used_cols.update({est_j, pm_j})
        for r_idx, row in enumerate(table.rows):
            est_v = _to_float_or_none(row[est_j] if est_j < len(row) else None)
            hw_v = _to_float_or_none(row[pm_j] if pm_j < len(row) else None)
            if est_v is None or hw_v is None or hw_v < 0:
                continue
            dec = _count_decimals(_raw_cell(table, r_idx, pm_j))
            records.append(
                CIRecord(
                    table_name=table.name,
                    source_file=table.source_file,
                    row_index=r_idx,
                    row_key=_row_key(r_idx),
                    metric_name=metric_name,
                    estimate=est_v,
                    lo=est_v - hw_v,
                    hi=est_v + hw_v,
                    decimals=dec,
                    n_sample=_row_n(r_idx),
                    is_unit_bounded=is_unit,
                    is_pm_format=True,
                )
            )

    # Strategy 3: Inline formatted string columns: "0.842 (0.810-0.874)", "0.842 ± 0.015", or "(0.810, 0.874)"
    for j, h in enumerate(headers):
        if j in used_cols or j in key_col_indices or j == n_col_idx:
            continue
        is_unit = bool(_UNIT_BOUNDED_METRIC_RE.search(h))
        for r_idx, row in enumerate(table.rows):
            if j >= len(row):
                continue
            s = _raw_cell(table, r_idx, j).strip()
            if not s:
                continue
            m_ci = _INLINE_CI_RE.match(s)
            if m_ci:
                est_v = float(m_ci.group(1))
                lo_v = float(m_ci.group(2))
                hi_v = float(m_ci.group(3))
                dec = max(_count_decimals(m_ci.group(2)), _count_decimals(m_ci.group(3)))
                records.append(
                    CIRecord(
                        table_name=table.name,
                        source_file=table.source_file,
                        row_index=r_idx,
                        row_key=_row_key(r_idx),
                        metric_name=h,
                        estimate=est_v,
                        lo=lo_v,
                        hi=hi_v,
                        decimals=dec,
                        n_sample=_row_n(r_idx),
                        is_unit_bounded=is_unit,
                    )
                )
                continue
            m_pm = _INLINE_PM_RE.match(s)
            if m_pm:
                est_v = float(m_pm.group(1))
                hw_v = float(m_pm.group(2))
                dec = _count_decimals(m_pm.group(2))
                records.append(
                    CIRecord(
                        table_name=table.name,
                        source_file=table.source_file,
                        row_index=r_idx,
                        row_key=_row_key(r_idx),
                        metric_name=h,
                        estimate=est_v,
                        lo=est_v - hw_v,
                        hi=est_v + hw_v,
                        decimals=dec,
                        n_sample=_row_n(r_idx),
                        is_unit_bounded=is_unit,
                        is_pm_format=True,
                    )
                )
                continue
            if _CI_STR_COL_RE.search(h):
                m_int = _INTERVAL_ONLY_RE.match(s)
                if m_int:
                    lo_v = float(m_int.group(1))
                    hi_v = float(m_int.group(2))
                    est_v = (
                        _to_float_or_none(row[j - 1])
                        if (j > 0 and (j - 1) not in key_col_indices and (j - 1) != n_col_idx)
                        else None
                    )
                    m_name = headers[j - 1] if est_v is not None else h
                    dec = max(_count_decimals(m_int.group(1)), _count_decimals(m_int.group(2)))
                    records.append(
                        CIRecord(
                            table_name=table.name,
                            source_file=table.source_file,
                            row_index=r_idx,
                            row_key=_row_key(r_idx),
                            metric_name=m_name,
                            estimate=est_v,
                            lo=lo_v,
                            hi=hi_v,
                            decimals=dec,
                            n_sample=_row_n(r_idx),
                            is_unit_bounded=bool(_UNIT_BOUNDED_METRIC_RE.search(m_name)),
                        )
                    )

    return records


def audit_table_stats(
    table: TableData,
    methods_context: dict[str, Any] | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Run single-table Module 3 checks (fixed-width CI, zero-width CI, impossible P-values, paired-Z)."""
    alerts: list[AuditAlert] = []
    ctx = methods_context or {}
    ci_records = extract_ci_records(table)

    # Group CI records by metric column
    by_metric: dict[str, list[CIRecord]] = {}
    for rec in ci_records:
        by_metric.setdefault(rec.metric_name, []).append(rec)

    zero_width_interior: list[CIRecord] = []
    zero_width_boundary: list[CIRecord] = []
    invalid_order_rows: list[int] = []

    for rec in ci_records:
        if rec.hi < rec.lo - 1e-9:
            invalid_order_rows.append(rec.row_index)
        if abs(rec.width) <= 1e-12:
            if rec.estimate is not None and (
                abs(rec.estimate) <= 1e-6
                or abs(rec.estimate - 1.0) <= 1e-6
                or abs(rec.estimate - 100.0) <= 1e-6
            ):
                zero_width_boundary.append(rec)
            else:
                zero_width_interior.append(rec)

    # Rule STAT-002: Zero-width CIs (lo == hi) on finite-sample metrics
    if len(zero_width_interior) >= 1 or (len(zero_width_interior) + len(zero_width_boundary)) >= 2:
        n_tot = len(zero_width_interior) + len(zero_width_boundary)
        sev = "CRITICAL" if len(zero_width_interior) >= 1 else "HIGH"
        sample_rec = zero_width_interior[0] if zero_width_interior else zero_width_boundary[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-002",
                severity=sev,
                title="Zero-width confidence interval (lo == hi) on finite-sample metric",
                message=(
                    f"Detected {n_tot} zero-width CI(s) ({len(zero_width_interior)} interior) in "
                    f"'{table.name}' (e.g., metric '{sample_rec.metric_name}' = {sample_rec.estimate}, "
                    f"CI [{sample_rec.lo}, {sample_rec.hi}])."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={
                    "n_zero_width_interior": len(zero_width_interior),
                    "n_zero_width_boundary": len(zero_width_boundary),
                    "example_rows": [r.row_index for r in (zero_width_interior + zero_width_boundary)[:10]],
                },
            )
        )

    if invalid_order_rows:
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-001",
                severity="CRITICAL",
                title="Inverted confidence interval bounds (hi < lo)",
                message=f"Found {len(invalid_order_rows)} row(s) with upper CI bound strictly less than lower bound.",
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={"rows": invalid_order_rows[:10]},
            )
        )

    # Rule STAT-001: Fixed-width / zero-variance placeholder CIs
    for metric_name, recs in by_metric.items():
        pos_recs = [r for r in recs if r.width > 1e-12]
        if len(pos_recs) < 5:
            continue
        med_dec = int(np.median([r.decimals for r in pos_recs]))
        # Round half-width to reported precision to detect exact ties
        round_dp = max(med_dec + 1, 4) if med_dec >= 3 else 3
        hw_vals = [round(r.half_width, round_dp) for r in pos_recs]
        counts: dict[float, int] = {}
        for hw in hw_vals:
            counts[hw] = counts.get(hw, 0) + 1
        mode_hw, mode_cnt = max(counts.items(), key=lambda kv: kv[1])
        mode_frac = mode_cnt / float(len(pos_recs))

        ests = [r.estimate for r in pos_recs if r.estimate is not None]
        est_span = (max(ests) - min(ests)) if len(ests) >= 2 else 0.0
        n_uniq_est = len({round(e, 4) for e in ests})

        ns = [r.n_sample for r in pos_recs if r.n_sample is not None and r.n_sample > 0]
        n_ratio = (max(ns) / min(ns)) if len(ns) >= 2 else 1.0

        # Check symmetry on modal rows
        sym_flags = [
            abs((r.hi - r.estimate) - (r.estimate - r.lo)) <= 1e-6
            for r in pos_recs
            if r.estimate is not None and round(r.half_width, round_dp) == mode_hw
        ]
        all_symmetric = bool(sym_flags) and all(sym_flags)
        any_pm = any(r.is_pm_format for r in pos_recs)

        flagged_fixed = False
        if med_dec >= 3 or any_pm:
            if mode_frac >= 0.80 and len(pos_recs) >= 5 and (
                (n_uniq_est >= 4 and est_span >= 0.03)
                or n_ratio >= 1.5
                or len(pos_recs) >= 8
            ):
                flagged_fixed = True
        else:
            # 2-decimal precision: strict safeguard against legitimate rounding ties
            if mode_frac >= 0.85 and len(pos_recs) >= 8 and (
                n_ratio >= 2.0
                or (est_span >= 0.12 and n_uniq_est >= 5 and all_symmetric)
            ):
                flagged_fixed = True

        if flagged_fixed:
            sev = "CRITICAL" if (mode_frac >= 0.95 and len(pos_recs) >= 6) or n_ratio >= 2.0 else "HIGH"
            alerts.append(
                AuditAlert(
                    module="statcheck",
                    rule_id="STAT-001",
                    severity=sev,
                    title="Fixed-width / zero-variance placeholder confidence intervals",
                    message=(
                        f"{mode_cnt}/{len(pos_recs)} ({mode_frac:.0%}) rows for '{metric_name}' in "
                        f"'{table.name}' share the exact same CI half-width (±{mode_hw:g}) despite "
                        f"estimate span={est_span:.3f} and cohort N ratio={n_ratio:.2f}x."
                    ),
                    file=table.source_file,
                    location=f"sheet '{table.name}' col '{metric_name}'",
                    evidence={
                        "metric": metric_name,
                        "modal_half_width": mode_hw,
                        "modal_count": mode_cnt,
                        "total_rows": len(pos_recs),
                        "modal_fraction": round(mode_frac, 4),
                        "median_decimals": med_dec,
                        "estimate_span": round(est_span, 5),
                        "n_ratio": round(n_ratio, 3),
                    },
                )
            )

    # Rule STAT-004 & STAT-005: P-value impossibility, bootstrap pseudoreplication & paired-Z contradiction
    pval_cols = [j for j, h in enumerate(table.headers) if _PVAL_COL_RE.match(h.strip())]
    n_col_idx = next((j for j, h in enumerate(table.headers) if _N_COL_RE.match(h.strip())), None)
    delta_col_idx = next((j for j, h in enumerate(table.headers) if _DELTA_COL_RE.match(h.strip())), None)
    method_col_idx = next(
        (j for j, h in enumerate(table.headers) if re.search(r"^(?:test|stat_?test|method|p_?method)$", h.strip(), re.I)),
        None,
    )
    folds_col_idx = next(
        (j for j, h in enumerate(table.headers) if re.search(r"^(?:k_?folds?|n_?folds?|folds)$", h.strip(), re.I)),
        None,
    )

    # Distinguish ML model-comparison tables from biomarker/GWAS/regression feature-association tables
    has_assoc_reg_headers = any(_ASSOC_REG_COL_RE.match(h.strip()) for h in table.headers)
    has_ml_discrim_headers = any(_UNIT_BOUNDED_METRIC_RE.search(h) for h in table.headers) or any(
        re.search(r"(?:wilcoxon|delong|mcnemar|model|metric)", h, re.I) for h in table.headers
    )
    has_ml_perf_rows = any(
        any(isinstance(c, str) and _UNIT_BOUNDED_METRIC_RE.search(c) for c in row[:4])
        for row in table.rows[:25]
    )
    is_biomarker_assoc_table = has_assoc_reg_headers and not (has_ml_discrim_headers or has_ml_perf_rows)
    has_perf_context = (not is_biomarker_assoc_table) and (
        has_ml_discrim_headers
        or has_ml_perf_rows
        or (delta_col_idx is not None)
        or bool(ctx.get("test") or ctx.get("n_folds"))
        or (bool(ci_records) and all(r.estimate is None or (0.0 <= r.estimate <= 1.0) for r in ci_records[:20]))
    )

    # Map row_index -> list of CIRecords in that row (for paired-Z check if 2 models are compared in a row)
    ci_by_row: dict[int, list[CIRecord]] = {}
    for rec in ci_records:
        ci_by_row.setdefault(rec.row_index, []).append(rec)

    wilcoxon_violations: list[dict[str, Any]] = []
    pseudorep_violations: list[dict[str, Any]] = []
    paired_z_violations: list[dict[str, Any]] = []

    for r_idx, row in enumerate(table.rows):
        row_n = _to_float_or_none(row[n_col_idx]) if (n_col_idx is not None and n_col_idx < len(row)) else ctx.get("n")
        row_delta = (
            abs(_to_float_or_none(row[delta_col_idx]))  # type: ignore[arg-type]
            if (delta_col_idx is not None and delta_col_idx < len(row) and _to_float_or_none(row[delta_col_idx]) is not None)
            else None
        )
        row_method = (
            str(row[method_col_idx]).lower()
            if (method_col_idx is not None and method_col_idx < len(row) and row[method_col_idx] is not None)
            else str(ctx.get("test", "")).lower()
        )
        row_folds_val = (
            _to_float_or_none(row[folds_col_idx])
            if (folds_col_idx is not None and folds_col_idx < len(row))
            else ctx.get("n_folds")
        )
        row_folds = int(row_folds_val) if row_folds_val is not None and row_folds_val > 0 else None

        # Detect "5-fold wilcoxon" in method text or table name
        if row_folds is None:
            m_kf = re.search(r"(\d+)\s*[- ]?fold", f"{row_method} {table.name}", re.I)
            if m_kf:
                row_folds = int(m_kf.group(1))
        is_wilcoxon = "wilcoxon" in f"{row_method} {table.name} {' '.join(table.headers)}".lower()

        for pj in pval_cols:
            if pj >= len(row):
                continue
            p_val, is_lt = _parse_p_value_cell(row[pj], _raw_cell(table, r_idx, pj))
            if p_val is None:
                continue
            # Effective P-value for "<1e-10" inequalities
            eff_p = p_val * 0.5 if is_lt else p_val

            # 4A: Exact K-fold paired Wilcoxon minimum P-value violation
            if is_wilcoxon and row_folds is not None and 2 <= row_folds <= 12:
                min_one_sided = 2.0 ** (-row_folds)
                if eff_p < min_one_sided - 1e-9:
                    wilcoxon_violations.append(
                        {
                            "row": r_idx,
                            "col": table.headers[pj],
                            "p_reported": p_val,
                            "n_folds": row_folds,
                            "min_possible_one_sided": min_one_sided,
                            "min_possible_two_sided": 2.0 * min_one_sided,
                        }
                    )
                    continue

            if is_biomarker_assoc_table or not has_perf_context:
                continue

            # 4B: Bootstrap pseudoreplication / impossible microscopic P-value
            if eff_p < 1e-10:
                if row_n is not None and row_delta is not None and 0.0 < row_delta < 1.0:
                    net_disc = max(1, int(round(row_delta * row_n)))
                    p_min_mc, log10_p_mc = min_mcnemar_p_for_net_diff(net_disc)
                    if net_disc < 35 or (eff_p > 0 and math.log10(eff_p) < log10_p_mc - 2.0) or eff_p == 0.0:
                        pseudorep_violations.append(
                            {
                                "row": r_idx,
                                "p_reported": p_val,
                                "n": row_n,
                                "delta": row_delta,
                                "net_discordant_approx": net_disc,
                                "min_possible_mcnemar_p": p_min_mc,
                            }
                        )
                elif row_n is not None and row_n <= 2000 and (row_delta is None or row_delta < 0.05):
                    pseudorep_violations.append(
                        {
                            "row": r_idx,
                            "p_reported": p_val,
                            "n": row_n,
                            "delta": row_delta,
                            "reason": "microscopic_p_on_modest_N",
                        }
                    )
                elif row_n is None and row_delta is not None and row_delta < 0.02:
                    pseudorep_violations.append(
                        {
                            "row": r_idx,
                            "p_reported": p_val,
                            "delta": row_delta,
                            "reason": "microscopic_p_on_tiny_delta",
                        }
                    )
                elif row_n is None and row_delta is None and has_perf_context and eff_p < 1e-15:
                    pseudorep_violations.append(
                        {
                            "row": r_idx,
                            "p_reported": p_val,
                            "reason": "extreme_microscopic_p_in_metric_table",
                        }
                    )

            # 4C / STAT-005: Paired-Z sensitivity check against reported CIs in same row
            row_cis = [r for r in ci_by_row.get(r_idx, []) if r.width > 1e-9 and r.estimate is not None]
            if len(row_cis) >= 2:
                c1, c2 = row_cis[0], row_cis[1]
                se1 = c1.half_width / _Z_95
                se2 = c2.half_width / _Z_95
                d_val = row_delta if row_delta is not None else abs(c1.estimate - c2.estimate)  # type: ignore[operator]
                pz = paired_z_sensitivity_check(d_val, se1=se1, se2=se2, p_reported=eff_p)
                if pz["contradicts_reported_p"]:
                    paired_z_violations.append(
                        {
                            "row": r_idx,
                            "p_reported": p_val,
                            "delta": round(d_val, 6),
                            "se1": round(se1, 6),
                            "se2": round(se2, 6),
                            "p_rho_095": pz["sensitivity_grid"]["rho=0.95"]["p_value"],
                        }
                    )

    if wilcoxon_violations:
        ex = wilcoxon_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-004",
                severity="CRITICAL",
                title="Mathematically impossible K-fold paired Wilcoxon P-value",
                message=(
                    f"{len(wilcoxon_violations)} row(s) report P-values below the exact theoretical "
                    f"minimum for {ex['n_folds']}-fold paired Wilcoxon (reported P={ex['p_reported']:g} < "
                    f"min one-sided {ex['min_possible_one_sided']:g}, two-sided {ex['min_possible_two_sided']:g})."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={"violations": wilcoxon_violations[:10]},
            )
        )

    if pseudorep_violations:
        ex = pseudorep_violations[0]
        sev = "CRITICAL" if ("min_possible_mcnemar_p" in ex or ex["p_reported"] < 1e-15) else "HIGH"
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-004",
                severity=sev,
                title="Bootstrap-pseudoreplication / implausibly microscopic P-value alert",
                message=(
                    f"{len(pseudorep_violations)} row(s) in '{table.name}' report extreme P-values "
                    f"(e.g., P={ex['p_reported']:g}) incompatible with sample size/effect size "
                    "(classic signature of running paired tests across bootstrap resamples)."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={"violations": pseudorep_violations[:10]},
            )
        )

    if paired_z_violations:
        ex = paired_z_violations[0]
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-005",
                severity="CRITICAL",
                title="Reported P-value severely contradicts reported confidence intervals (paired-Z check)",
                message=(
                    f"{len(paired_z_violations)} row(s) in '{table.name}' report P={ex['p_reported']:g} "
                    f"for |delta|={ex['delta']:.4f}, whereas even assuming inter-model correlation rho=0.95 "
                    f"the paired-Z P-value is {ex['p_rho_095']:.4g}."
                ),
                file=table.source_file,
                location=f"sheet '{table.name}'",
                evidence={"violations": paired_z_violations[:10]},
            )
        )

    diagnostics = {
        "table_name": table.name,
        "n_ci_records": len(ci_records),
        "n_zero_width_interior": len(zero_width_interior),
        "n_wilcoxon_violations": len(wilcoxon_violations),
        "n_pseudorep_violations": len(pseudorep_violations),
        "n_paired_z_violations": len(paired_z_violations),
    }
    return alerts, diagnostics


def audit_cross_table_ci_consistency(
    tables: list[TableData],
    ratio_threshold: float = 5.0,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Detect cross-sheet / cross-table CI width contradictions (>5x or sqrt(B) SE-of-bootstrap bug)."""
    all_recs: list[CIRecord] = []
    for tbl in tables:
        all_recs.extend(extract_ci_records(tbl))

    # Group positive-width records by normalized (row_key, metric_name) across distinct tables
    buckets: dict[tuple[str, str], list[CIRecord]] = {}
    for rec in all_recs:
        if rec.width <= 1e-12:
            continue
        k = (rec.row_key, _normalize_key(rec.metric_name))
        buckets.setdefault(k, []).append(rec)

    contradictions: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str, str]] = set()

    for (rkey, mkey), group in buckets.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a.table_name == b.table_name and a.source_file == b.source_file:
                    continue
                # Require matching point estimate if both present (confirms identical task/cohort)
                if a.estimate is not None and b.estimate is not None:
                    if abs(a.estimate - b.estimate) > 0.015:
                        continue
                w_lo = min(a.width, b.width)
                w_hi = max(a.width, b.width)
                ratio = w_hi / max(w_lo, 1e-12)
                if ratio <= ratio_threshold:
                    continue
                pair_id = tuple(sorted([f"{a.table_name}:{rkey}:{mkey}", f"{b.table_name}:{rkey}:{mkey}"]))
                sig = (pair_id[0], pair_id[1], rkey, mkey)
                if sig in seen_pairs:
                    continue
                seen_pairs.add(sig)

                # Check match against sqrt(B) bootstrap SE division error
                matched_b: int | None = None
                best_rel_err = 1.0
                for b_cand in _BOOTSTRAP_CANDIDATES:
                    sq_b = math.sqrt(b_cand)
                    rel_err = abs(ratio - sq_b) / sq_b
                    if rel_err <= 0.18 and rel_err < best_rel_err:
                        best_rel_err = rel_err
                        matched_b = b_cand

                contradictions.append(
                    {
                        "row_key": rkey,
                        "metric": mkey,
                        "table_a": a.table_name,
                        "table_b": b.table_name,
                        "width_a": round(a.width, 6),
                        "width_b": round(b.width, 6),
                        "width_ratio": round(ratio, 3),
                        "estimate_a": a.estimate,
                        "estimate_b": b.estimate,
                        "matched_sqrt_B": matched_b,
                    }
                )

    alerts: list[AuditAlert] = []
    if contradictions:
        contradictions.sort(key=lambda d: -d["width_ratio"])
        top = contradictions[0]
        any_sqrt_b = any(c["matched_sqrt_B"] is not None for c in contradictions)
        sev = "CRITICAL" if (top["width_ratio"] >= 10.0 or any_sqrt_b) else "HIGH"
        sqrt_note = (
            f" (ratio {top['width_ratio']:.1f}x matches sqrt({top['matched_sqrt_B']})={math.sqrt(top['matched_sqrt_B']):.1f}x, "
            "diagnostic of erroneous SE-of-bootstrap-mean division by sqrt(B))"
            if top["matched_sqrt_B"] is not None
            else ""
        )
        src_file = tables[0].source_file if tables else None
        alerts.append(
            AuditAlert(
                module="statcheck",
                rule_id="STAT-003",
                severity=sev,
                title="Cross-sheet/cross-table confidence interval width contradiction",
                message=(
                    f"Matched task '{top['row_key']}' ({top['metric']}) exhibits a {top['width_ratio']:.1f}x "
                    f"CI width contradiction between '{top['table_a']}' (width={top['width_a']:.5f}) and "
                    f"'{top['table_b']}' (width={top['width_b']:.5f}){sqrt_note}."
                ),
                file=src_file,
                location=f"'{top['table_a']}' vs '{top['table_b']}'",
                evidence={
                    "n_contradictions": len(contradictions),
                    "max_ratio": top["width_ratio"],
                    "contradictions": contradictions[:10],
                },
            )
        )

    return alerts, {"n_contradictions": len(contradictions), "contradictions": contradictions}


def audit_tables_suite(
    tables: list[TableData],
    methods_context: dict[str, Any] | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Run all Module 3 single-table and cross-table statistical checks on a list of tables."""
    alerts: list[AuditAlert] = []
    per_table_diag: list[dict[str, Any]] = []
    for tbl in tables:
        t_alerts, t_diag = audit_table_stats(tbl, methods_context=methods_context)
        alerts.extend(t_alerts)
        per_table_diag.append(t_diag)

    cross_diag: dict[str, Any] = {"n_contradictions": 0}
    if len(tables) >= 2:
        c_alerts, cross_diag = audit_cross_table_ci_consistency(tables)
        alerts.extend(c_alerts)

    return alerts, {"per_table": per_table_diag, "cross_table": cross_diag}


def audit_stat_file(
    path: str | Path,
    methods_context: dict[str, Any] | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Load an .xlsx/.csv/.tsv file and run all Module 3 statistical sanity checks."""
    tables = read_tables(path)
    return audit_tables_suite(tables, methods_context=methods_context)
