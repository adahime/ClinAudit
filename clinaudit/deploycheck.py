"""Module 4: Clinical Calculator Probability Axiom, Encoding & Monotonicity Auditor."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np

from .io_utils import CodeDocument, read_code_file
from .models import AuditAlert

# Strictly established monotone clinical risk markers (+1: higher -> higher risk; -1: higher -> lower risk).
# Intentionally excludes U-shaped physiological variables (e.g., BMI, sodium, potassium, heart rate).
DEFAULT_CLINICAL_DIRECTIONS: dict[str, int] = {
    # Positive risk factors (+1)
    "age": 1,
    "sbp": 1,
    "systolic_bp": 1,
    "creatinine": 1,
    "scr": 1,
    "serum_creatinine": 1,
    "bun": 1,
    "glucose": 1,
    "hba1c": 1,
    "uacr": 1,
    "acr": 1,
    "albuminuria": 1,
    "proteinuria": 1,
    "bilirubin": 1,
    "total_bilirubin": 1,
    "lactate": 1,
    "troponin": 1,
    "bnp": 1,
    "nt_probnp": 1,
    "crp": 1,
    "sofa": 1,
    "sofa_score": 1,
    "charlson": 1,
    "cci": 1,
    "smoking": 1,
    "current_smoker": 1,
    "diabetes": 1,
    "hypertension": 1,
    "heart_failure": 1,
    "ckd_stage": 1,
    "tumor_size": 1,
    "metastasis": 1,
    # Protective / inverse risk markers (-1: declining value -> higher risk)
    "egfr": -1,
    "gfr": -1,
    "albumin": -1,
    "serum_albumin": -1,
    "alb": -1,
    "hemoglobin": -1,
    "hgb": -1,
    "hb": -1,
    "lvef": -1,
    "ejection_fraction": -1,
    "hdl": -1,
    "spo2": -1,
    "oxygen_saturation": -1,
    "gcs": -1,
    "glasgow_coma_scale": -1,
}

_BINARY_DEMO_NAMES = re.compile(r"^(?:sex|gender|is_?male|male|female)$", re.IGNORECASE)

_COEF_IDENT_RE = re.compile(
    r"(?:coef(?:ficient)?s?|beta(?:s|_vec|_hat)?|weights?|w_?vec|log_?hr|cox_?coef|params|betas)",
    re.IGNORECASE,
)
_IMPORTANCE_SAFE_RE = re.compile(
    r"(?:argsort|sort(?:ed)?|order|rank|nlargest|top_?k|top_?feat|importance|feat_?imp|"
    r"barh|plot|plt\.|sns\.|ggplot|geom_|aes\s*\(|labs\s*\(|filter\s*\(|ifelse\s*\(|"
    r"print|l1_?pen|lasso|reg_?loss|penalty|abs_?diff)",
    re.IGNORECASE,
)
_RISK_USAGE_RE = re.compile(
    r"(?:@|dot|matmul|%\*%|\bsum\b|score|risk|prob|lp\b|linear_?pred|logit|hazard|points|predict|sigmoid|exp\s*\(|\*\s*[xX]\b|[xX]\s*\*)",
    re.IGNORECASE,
)


def _normalize_feat(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _lookup_direction(feat_name: str, directions: dict[str, int]) -> int | None:
    norm = _normalize_feat(feat_name)
    if norm in directions:
        return directions[norm]
    # Check exact token match (e.g. "baseline_egfr" -> "egfr", "serum_albumin_g_dl" -> "albumin")
    tokens = norm.split("_")
    for tok in tokens:
        if tok in directions:
            return directions[tok]
    for key, d in directions.items():
        if key in norm and len(key) >= 4:
            return d
    return None


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pure-NumPy Spearman rank correlation (handles ties via average ranks)."""
    if x.size < 2 or float(np.ptp(x)) <= 1e-15 or float(np.ptp(y)) <= 1e-15:
        return 0.0

    def _rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty_like(a, dtype=np.float64)
        s = a[order]
        n = a.size
        i = 0
        while i < n:
            j = i + 1
            while j < n and s[j] == s[i]:
                j += 1
            avg_r = 0.5 * (i + j - 1)
            ranks[order[i:j]] = avg_r
            i = j
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    rx -= np.mean(rx)
    ry -= np.mean(ry)
    denom = float(np.sqrt(np.sum(rx * rx) * np.sum(ry * ry)))
    return float(np.sum(rx * ry) / denom) if denom > 1e-15 else 0.0


def _eval_model(fn: Callable[[np.ndarray], Any], X: np.ndarray) -> np.ndarray:
    out = np.asarray(fn(X), dtype=np.float64)
    return out.ravel()


def audit_linear_coefficients(
    coefs: dict[str, float],
    expected_directions: dict[str, int] | None = None,
    file_label: str | None = None,
    location_label: str | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Audit named regression/Cox coefficients against clinical directionality."""
    dir_map = dict(DEFAULT_CLINICAL_DIRECTIONS)
    if expected_directions:
        dir_map.update({_normalize_feat(k): int(v) for k, v in expected_directions.items()})

    violations: list[dict[str, Any]] = []
    checked: list[str] = []
    protective_all_pos = []

    for fname, val in coefs.items():
        d = _lookup_direction(fname, dir_map)
        if d is None:
            continue
        checked.append(fname)
        fval = float(val)
        if d == -1:
            protective_all_pos.append(fval > 1e-6)
        if fval * d < -1e-6:
            violations.append(
                {
                    "feature": fname,
                    "coefficient": fval,
                    "expected_sign": "+" if d > 0 else "-",
                }
            )

    alerts: list[AuditAlert] = []
    if violations:
        all_nonneg = all(float(v) >= -1e-9 for v in coefs.values()) and any(
            float(v) > 1e-6 for v in coefs.values()
        )
        sev = "CRITICAL" if (len(violations) >= 2 or all_nonneg) else "HIGH"
        sign_note = (
            " All model coefficients are non-negative (classic abs(coef) sign-destruction signature)."
            if all_nonneg and any(v["expected_sign"] == "-" for v in violations)
            else ""
        )
        summary_items = ", ".join(
            f"{v['feature']}={v['coefficient']:+.4f}" for v in violations[:5]
        )
        alerts.append(
            AuditAlert(
                module="deploycheck",
                rule_id="DEPLOY-003",
                severity=sev,
                title="Clinical coefficient directionality inversion detected",
                message=(
                    f"{len(violations)} clinical risk marker(s) have inverted coefficient signs "
                    f"({summary_items}){sign_note}"
                ),
                file=file_label,
                location=location_label,
                evidence={
                    "violations": violations,
                    "n_checked": len(checked),
                    "all_coefficients_non_negative": all_nonneg,
                },
            )
        )
    return alerts, {"n_checked": len(checked), "violations": violations}


def audit_risk_calculator(
    models: Callable[[np.ndarray], np.ndarray] | dict[str, Callable[[np.ndarray], np.ndarray]],
    feature_names: list[str],
    X_grid: np.ndarray,
    hierarchy_pairs: list[tuple[str, str]] | None = None,
    categorical_probes: dict[str, dict[str, Any]] | None = None,
    expected_directions: dict[str, int] | None = None,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    bounds_tol: float = 1e-6,
    hierarchy_tol: float = 1e-4,
    file_label: str | None = None,
    location_label: str | None = None,
) -> tuple[list[AuditAlert], dict[str, Any]]:
    """Black-box verifier for deployed clinical risk functions.

    Checks:
      (a) Probability bounds [0, 1] and strict hierarchy P_sub(x) <= P_parent(x) (DEPLOY-001)
      (b) Categorical encoding consistency across endpoints & {0,1} vs {1,2} probes (DEPLOY-002)
      (c) Clinical directionality / monotonicity across physiological marker sweeps (DEPLOY-003)
    """
    if callable(models):
        model_map: dict[str, Callable[[np.ndarray], np.ndarray]] = {"risk": models}
    else:
        model_map = dict(models)

    X = np.asarray(X_grid, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    n_samples, n_features = X.shape
    if n_samples == 0 or n_features != len(feature_names):
        return [], {"skipped": True, "reason": "Empty X_grid or feature_names length mismatch"}

    alerts: list[AuditAlert] = []
    preds: dict[str, np.ndarray] = {}

    # --- (a1) Probability bounds [0, 1] ---
    bounds_diag: dict[str, Any] = {}
    for mname, fn in model_map.items():
        p = _eval_model(fn, X)
        preds[mname] = p
        n_nonfinite = int(np.sum(~np.isfinite(p)))
        finite_p = p[np.isfinite(p)]
        n_below = int(np.sum(finite_p < -bounds_tol))
        n_above = int(np.sum(finite_p > 1.0 + bounds_tol))
        p_min = float(np.min(finite_p)) if finite_p.size else float("nan")
        p_max = float(np.max(finite_p)) if finite_p.size else float("nan")
        bounds_diag[mname] = {
            "min": p_min,
            "max": p_max,
            "n_nonfinite": n_nonfinite,
            "n_below_zero": n_below,
            "n_above_one": n_above,
        }
        if n_nonfinite > 0 or n_below > 0 or n_above > 0:
            alerts.append(
                AuditAlert(
                    module="deploycheck",
                    rule_id="DEPLOY-001",
                    severity="CRITICAL",
                    title="Probability axiom bounds violation ([0, 1] or non-finite output)",
                    message=(
                        f"Endpoint '{mname}' produced invalid probabilities on {n_nonfinite + n_below + n_above}/{n_samples} "
                        f"profiles (min={p_min:.5g}, max={p_max:.5g}, non-finite={n_nonfinite})."
                    ),
                    file=file_label,
                    location=location_label or f"endpoint '{mname}'",
                    evidence={"endpoint": mname, **bounds_diag[mname]},
                )
            )

    # --- (a2) Strict probability hierarchy: P_cause(x) <= P_all_cause(x) ---
    hierarchy_diag: list[dict[str, Any]] = []
    for sub_name, parent_name in hierarchy_pairs or []:
        if sub_name not in preds or parent_name not in preds:
            continue
        p_sub = preds[sub_name]
        p_par = preds[parent_name]
        valid = np.isfinite(p_sub) & np.isfinite(p_par)
        if not np.any(valid):
            continue
        excess = np.where(valid, p_sub - p_par, -np.inf)
        viol_mask = valid & (excess > hierarchy_tol)
        n_viol = int(np.sum(viol_mask))
        max_excess = float(np.max(excess)) if n_viol > 0 else 0.0
        entry: dict[str, Any] = {
            "sub_endpoint": sub_name,
            "parent_endpoint": parent_name,
            "n_violations": n_viol,
            "n_total": int(np.sum(valid)),
            "violation_fraction": round(n_viol / max(int(np.sum(valid)), 1), 6),
            "max_excess": round(max_excess, 6),
        }
        if n_viol > 0:
            worst_idx = int(np.argmax(excess))
            entry["worst_sub_prob"] = round(float(p_sub[worst_idx]), 6)
            entry["worst_parent_prob"] = round(float(p_par[worst_idx]), 6)
            entry["worst_profile"] = {
                feature_names[j]: round(float(X[worst_idx, j]), 4)
                for j in range(n_features)
            }
            alerts.append(
                AuditAlert(
                    module="deploycheck",
                    rule_id="DEPLOY-001",
                    severity="CRITICAL",
                    title="Probability hierarchy axiom violation (P_cause > P_all_cause)",
                    message=(
                        f"Sub-endpoint '{sub_name}' exceeds parent '{parent_name}' on "
                        f"{n_viol}/{int(np.sum(valid))} ({entry['violation_fraction']:.1%}) clinical profiles "
                        f"(worst: {entry['worst_sub_prob']:.4f} > {entry['worst_parent_prob']:.4f}, "
                        f"excess=+{max_excess:.4f})."
                    ),
                    file=file_label,
                    location=location_label or f"'{sub_name}' vs '{parent_name}'",
                    evidence=entry,
                )
            )
        hierarchy_diag.append(entry)

    # Select deterministic representative reference subset for probing & monotonicity sweeps
    n_ref = min(20, n_samples)
    ref_indices = np.linspace(0, n_samples - 1, num=n_ref, dtype=int)
    X_ref = X[ref_indices]

    # --- (b) Categorical encoding consistency check ({0,1} vs {1,2} and cross-endpoint polarity) ---
    probe_specs: dict[str, dict[str, Any]] = {}
    if categorical_probes:
        probe_specs.update(categorical_probes)
    else:
        for fname in feature_names:
            if _BINARY_DEMO_NAMES.match(fname.strip()):
                probe_specs[fname] = {"valid_values": [0, 1], "alt_encoding": [1, 2]}

    encoding_diag: list[dict[str, Any]] = []
    name_to_idx = {fname: j for j, fname in enumerate(feature_names)}
    name_to_idx_lower = {fname.lower(): j for j, fname in enumerate(feature_names)}

    for feat_name, spec in probe_specs.items():
        j = name_to_idx.get(feat_name)
        if j is None:
            j = name_to_idx_lower.get(feat_name.lower())
        if j is None:
            continue
        v0, v1 = spec.get("valid_values", [0, 1])[:2]
        alt_vals = spec.get("alt_encoding", [1, 2])
        v2 = alt_vals[1] if len(alt_vals) >= 2 else (v1 + 1)

        X0 = X_ref.copy()
        X1 = X_ref.copy()
        X2 = X_ref.copy()
        X0[:, j] = float(v0)
        X1[:, j] = float(v1)
        X2[:, j] = float(v2)

        obs_vals = set(np.unique(np.round(X[:, j], 6)).tolist())
        per_model_effects: dict[str, dict[str, float]] = {}

        for mname, fn in model_map.items():
            try:
                p0 = _eval_model(fn, X0)
                p1 = _eval_model(fn, X1)
                p2 = _eval_model(fn, X2)
            except Exception:
                continue
            d01 = p1 - p0
            d12 = p2 - p1
            mean_d01 = float(np.nanmean(d01))
            mean_d12 = float(np.nanmean(d12))
            max_abs_d01 = float(np.nanmax(np.abs(d01)))
            max_abs_d12 = float(np.nanmax(np.abs(d12)))
            per_model_effects[mname] = {
                "mean_d01": mean_d01,
                "mean_d12": mean_d12,
                "max_abs_d01": max_abs_d01,
                "max_abs_d12": max_abs_d12,
                "frac_pos_d01": float(np.mean(d01 > 1e-6)),
                "frac_neg_d01": float(np.mean(d01 < -1e-6)),
            }

            # Domain-shift mismatch: cohort uses {v0, v1}, yet model is completely flat on {v0, v1}
            # and jumps ONLY at alt value v2 (e.g., expects {1, 2} instead of {0, 1}!)
            if obs_vals <= {float(v0), float(v1)} and max_abs_d01 < 1e-6 and max_abs_d12 > 0.005:
                alerts.append(
                    AuditAlert(
                        module="deploycheck",
                        rule_id="DEPLOY-002",
                        severity="CRITICAL",
                        title="Categorical encoding domain mismatch ({0,1} vs {1,2})",
                        message=(
                            f"Endpoint '{mname}' shows zero response to '{feat_name}' across cohort domain "
                            f"{{{v0}, {v1}}} (max |delta|={max_abs_d01:.2e}) but shifts by {mean_d12:+.4f} "
                            f"at {{{v1}, {v2}}}, indicating a shifted categorical encoding."
                        ),
                        file=file_label,
                        location=location_label or f"endpoint '{mname}' feature '{feat_name}'",
                        evidence={"endpoint": mname, "feature": feat_name, **per_model_effects[mname]},
                    )
                )

        # Cross-endpoint encoding contradictions on hierarchy pairs (or across all multi-endpoints)
        pair_list = list(hierarchy_pairs or [])
        if not pair_list and len(per_model_effects) >= 2:
            mnames = list(per_model_effects.keys())
            pair_list = [(mnames[a], mnames[b]) for a in range(len(mnames)) for b in range(a + 1, len(mnames))]

        for ma, mb in pair_list:
            if ma not in per_model_effects or mb not in per_model_effects:
                continue
            ea, eb = per_model_effects[ma], per_model_effects[mb]
            # (1) One model responds on {0,1} & flat on {1,2}, other responds on {1,2} & flat on {0,1}
            a_01_only = ea["max_abs_d01"] > 0.005 and ea["max_abs_d12"] < 1e-6
            b_12_only = eb["max_abs_d01"] < 1e-6 and eb["max_abs_d12"] > 0.005
            b_01_only = eb["max_abs_d01"] > 0.005 and eb["max_abs_d12"] < 1e-6
            a_12_only = ea["max_abs_d01"] < 1e-6 and ea["max_abs_d12"] > 0.005
            if (a_01_only and b_12_only) or (b_01_only and a_12_only):
                alerts.append(
                    AuditAlert(
                        module="deploycheck",
                        rule_id="DEPLOY-002",
                        severity="CRITICAL",
                        title="Cross-endpoint categorical encoding mismatch ({0,1} vs {1,2})",
                        message=(
                            f"Endpoints '{ma}' and '{mb}' respond on disjoint categorical codes for "
                            f"'{feat_name}' ({ma}: d01={ea['mean_d01']:+.4f}, d12={ea['mean_d12']:+.4f}; "
                            f"{mb}: d01={eb['mean_d01']:+.4f}, d12={eb['mean_d12']:+.4f})."
                        ),
                        file=file_label,
                        location=location_label or f"'{ma}' vs '{mb}' feature '{feat_name}'",
                        evidence={"feature": feat_name, ma: ea, mb: eb},
                    )
                )
            # (2) Opposite polarity on a binary sex/demographic variable across correlated/hierarchical endpoints
            elif (
                abs(ea["mean_d01"]) > 0.008
                and abs(eb["mean_d01"]) > 0.008
                and (ea["mean_d01"] * eb["mean_d01"] < 0.0)
                and max(ea["frac_pos_d01"], ea["frac_neg_d01"]) >= 0.8
                and max(eb["frac_pos_d01"], eb["frac_neg_d01"]) >= 0.8
            ):
                alerts.append(
                    AuditAlert(
                        module="deploycheck",
                        rule_id="DEPLOY-002",
                        severity="HIGH",
                        title="Cross-endpoint categorical polarity reversal",
                        message=(
                            f"Feature '{feat_name}' has opposite directional effects across endpoints "
                            f"'{ma}' ({ea['mean_d01']:+.4f}) and '{mb}' ({eb['mean_d01']:+.4f}), "
                            "consistent with reversed categorical coding (e.g., 0=M/1=F vs 0=F/1=M)."
                        ),
                        file=file_label,
                        location=location_label or f"'{ma}' vs '{mb}' feature '{feat_name}'",
                        evidence={"feature": feat_name, ma: ea, mb: eb},
                    )
                )

        encoding_diag.append({"feature": feat_name, "effects": per_model_effects})

    # --- (c) Clinical directionality / monotonicity sweep ---
    dir_map = dict(DEFAULT_CLINICAL_DIRECTIONS)
    if expected_directions:
        dir_map.update({_normalize_feat(k): int(v) for k, v in expected_directions.items()})

    mono_diag: list[dict[str, Any]] = []
    for mname, fn in model_map.items():
        model_violations: list[dict[str, Any]] = []
        for j, fname in enumerate(feature_names):
            exp_d = _lookup_direction(fname, dir_map)
            if exp_d is None:
                continue
            col = X[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                continue
            uniq = np.unique(finite)
            if uniq.size < 2:
                continue

            if feature_ranges and fname in feature_ranges:
                lo_v, hi_v = feature_ranges[fname]
            elif uniq.size == 2:
                lo_v, hi_v = float(uniq[0]), float(uniq[1])
            else:
                lo_v = float(np.percentile(finite, 5))
                hi_v = float(np.percentile(finite, 95))
                if hi_v <= lo_v:
                    lo_v, hi_v = float(np.min(finite)), float(np.max(finite))
            if hi_v <= lo_v:
                continue

            sweep_vals = np.array([lo_v, hi_v], dtype=np.float64) if uniq.size == 2 else np.linspace(lo_v, hi_v, 10)
            # Vectorized evaluation across all reference profiles x sweep grid points
            n_g = sweep_vals.size
            X_sweep = np.repeat(X_ref, n_g, axis=0)
            X_sweep[:, j] = np.tile(sweep_vals, n_ref)
            p_sweep = _eval_model(fn, X_sweep).reshape(n_ref, n_g)
            mean_curve = np.nanmean(p_sweep, axis=0)
            if not np.all(np.isfinite(mean_curve)):
                continue

            overall_delta = float(mean_curve[-1] - mean_curve[0])
            signed_delta = exp_d * overall_delta
            mean_level = max(float(np.mean(mean_curve)), 1e-4)
            rel_delta = abs(overall_delta) / mean_level

            if n_g >= 4:
                rho = _spearman_corr(sweep_vals, mean_curve)
                diffs = np.diff(mean_curve)
                nontrivial = np.abs(diffs) > 1e-4
                wrong_frac = (
                    float(np.mean((exp_d * diffs[nontrivial]) < 0.0))
                    if np.any(nontrivial)
                    else 0.0
                )
            else:
                rho = 1.0 if signed_delta > 0 else (-1.0 if signed_delta < 0 else 0.0)
                wrong_frac = 1.0 if signed_delta < -1e-4 else 0.0

            is_inverted = (
                (signed_delta < -0.01 and (n_g == 2 or (rho * exp_d < -0.5 and wrong_frac >= 0.6)))
                or (
                    signed_delta < -0.005
                    and rel_delta > 0.15
                    and (n_g == 2 or (rho * exp_d < -0.7 and wrong_frac >= 0.75))
                )
            )
            rec = {
                "endpoint": mname,
                "feature": fname,
                "expected_direction": "+" if exp_d > 0 else "-",
                "sweep_range": [round(lo_v, 4), round(hi_v, 4)],
                "p_at_low": round(float(mean_curve[0]), 6),
                "p_at_high": round(float(mean_curve[-1]), 6),
                "overall_delta": round(overall_delta, 6),
                "spearman_rho": round(rho, 4),
                "wrong_step_fraction": round(wrong_frac, 4),
                "inverted": is_inverted,
            }
            mono_diag.append(rec)
            if is_inverted:
                model_violations.append(rec)

        if model_violations:
            sev = "CRITICAL" if len(model_violations) >= 2 or any(abs(v["overall_delta"]) >= 0.05 for v in model_violations) else "HIGH"
            summary_items = ", ".join(
                f"{v['feature']} (expected {v['expected_direction']}, delta={v['overall_delta']:+.4f})"
                for v in model_violations[:5]
            )
            alerts.append(
                AuditAlert(
                    module="deploycheck",
                    rule_id="DEPLOY-003",
                    severity=sev,
                    title="Clinical directionality / monotonicity violation in risk calculator",
                    message=(
                        f"Endpoint '{mname}' moves risk in the clinically opposite direction for "
                        f"{len(model_violations)} marker(s): {summary_items}."
                    ),
                    file=file_label,
                    location=location_label or f"endpoint '{mname}'",
                    evidence={"endpoint": mname, "violations": model_violations},
                )
            )

    diagnostics = {
        "n_profiles": n_samples,
        "bounds": bounds_diag,
        "hierarchy": hierarchy_diag,
        "encoding": encoding_diag,
        "monotonicity": mono_diag,
    }
    return alerts, diagnostics


def _scan_python_deploy_code(doc: CodeDocument) -> list[AuditAlert]:
    """Static Python AST/line scanner for abs(coef) sign destruction & encoding contradictions."""
    alerts: list[AuditAlert] = []
    seen: set[tuple[str, int]] = set()
    lines = doc.source.splitlines()

    def _emit(rule_id: str, severity: str, title: str, message: str, lineno: int, evidence: dict[str, Any]) -> None:
        key = (rule_id, lineno)
        if key in seen:
            return
        seen.add(key)
        alerts.append(
            AuditAlert(
                module="deploycheck",
                rule_id=rule_id,
                severity=severity,  # type: ignore[arg-type]
                title=title,
                message=message,
                file=doc.path,
                location=doc.format_location(lineno),
                evidence=evidence,
            )
        )

    try:
        tree = ast.parse(doc.source)
    except SyntaxError:
        tree = None

    if tree is not None:
        # Track variables assigned from abs(coef) and check whether they are used for risk scoring
        abs_coef_vars: dict[str, tuple[int, str]] = {}
        # Track element loop variables bound from iterating over coefficient vectors (e.g., zip(coefficients, x))
        coef_iter_vars: set[str] = set()
        for node in ast.walk(tree):
            gens = []
            if isinstance(node, (ast.GeneratorExp, ast.ListComp, ast.SetComp)):
                gens = list(node.generators)
            elif isinstance(node, ast.For):
                gens = [ast.comprehension(target=node.target, iter=node.iter, ifs=[], is_async=0)]
            for gen in gens:
                iter_node = gen.iter
                if isinstance(iter_node, ast.Call):
                    icname = iter_node.func.id if isinstance(iter_node.func, ast.Name) else ""
                    if icname == "zip" and isinstance(gen.target, (ast.Tuple, ast.List)):
                        for tgt_elt, arg_elt in zip(gen.target.elts, iter_node.args):
                            arg_txt = ast.unparse(arg_elt) if hasattr(ast, "unparse") else ""
                            if isinstance(tgt_elt, ast.Name) and _COEF_IDENT_RE.search(arg_txt):
                                coef_iter_vars.add(tgt_elt.id)
                else:
                    iter_txt = ast.unparse(iter_node) if hasattr(ast, "unparse") else ""
                    if isinstance(gen.target, ast.Name) and _COEF_IDENT_RE.search(iter_txt):
                        coef_iter_vars.add(gen.target.id)

        def _is_abs_coef_call(node: ast.AST) -> str | None:
            if not isinstance(node, ast.Call):
                return None
            cname = node.func.id if isinstance(node.func, ast.Name) else (node.func.attr if isinstance(node.func, ast.Attribute) else "")
            if cname not in ("abs", "fabs"):
                return None
            if not node.args:
                return None
            arg_src = ast.unparse(node.args[0]) if hasattr(ast, "unparse") else ""
            if _COEF_IDENT_RE.search(arg_src) or (isinstance(node.args[0], ast.Name) and node.args[0].id in coef_iter_vars):
                return arg_src
            return None

        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                lineno = getattr(node, "lineno", 1)
                line_text = lines[lineno - 1] if 1 <= lineno <= len(lines) else ""
                if _IMPORTANCE_SAFE_RE.search(line_text):
                    continue
                for sub in ast.walk(node.value):
                    arg_src = _is_abs_coef_call(sub)
                    if arg_src is not None:
                        # Check if immediately used in risk scoring on same line
                        if _RISK_USAGE_RE.search(line_text):
                            _emit(
                                "DEPLOY-004",
                                "CRITICAL",
                                "Coefficient sign destruction via abs() in risk calculation",
                                f"`abs({arg_src})` destroys negative protective regression signs inside risk calculation.",
                                lineno,
                                {"expression": f"abs({arg_src})", "line_text": line_text.strip()},
                            )
                        else:
                            for tgt in node.targets:
                                if isinstance(tgt, ast.Name):
                                    if not _IMPORTANCE_SAFE_RE.search(tgt.id):
                                        abs_coef_vars[tgt.id] = (lineno, arg_src)
                        break
            elif isinstance(node, ast.Return) and node.value is not None:
                lineno = getattr(node, "lineno", 1)
                line_text = lines[lineno - 1] if 1 <= lineno <= len(lines) else ""
                if _IMPORTANCE_SAFE_RE.search(line_text):
                    continue
                for sub in ast.walk(node.value):
                    arg_src = _is_abs_coef_call(sub)
                    if arg_src is not None and _RISK_USAGE_RE.search(line_text):
                        _emit(
                            "DEPLOY-004",
                            "CRITICAL",
                            "Coefficient sign destruction via abs() in returned risk score",
                            f"`abs({arg_src})` is applied to model coefficients in risk return expression.",
                            lineno,
                            {"expression": f"abs({arg_src})", "line_text": line_text.strip()},
                        )
                        break

        # Check downstream usage of abs_coef_vars in risk calculations
        if abs_coef_vars:
            for lineno, line_text in enumerate(lines, start=1):
                code_part = line_text.split("#", 1)[0]
                if not code_part.strip() or _IMPORTANCE_SAFE_RE.search(code_part):
                    continue
                for vname, (def_line, arg_src) in abs_coef_vars.items():
                    if lineno <= def_line:
                        continue
                    if re.search(rf"\b{re.escape(vname)}\b", code_part) and _RISK_USAGE_RE.search(code_part):
                        _emit(
                            "DEPLOY-004",
                            "CRITICAL",
                            "Sign-destroyed coefficients used in clinical risk scoring",
                            f"Variable `{vname}` (created via `abs({arg_src})` at {doc.format_location(def_line)}) "
                            "is used in linear predictor / risk score calculation, inverting protective markers.",
                            lineno,
                            {"variable": vname, "def_line": def_line, "source_coef": arg_src},
                        )

    # Static categorical encoding contradiction check (e.g., sex==0/1 mixed with sex==2 in same script)
    _check_static_encoding_contradictions(doc, _emit)
    return alerts


def _scan_r_or_js_deploy_code(doc: CodeDocument) -> list[AuditAlert]:
    """Static R / JavaScript scanner for abs(coef) sign destruction & encoding contradictions."""
    alerts: list[AuditAlert] = []
    seen: set[tuple[str, int]] = set()
    lines = doc.source.splitlines()

    def _emit(rule_id: str, severity: str, title: str, message: str, lineno: int, evidence: dict[str, Any]) -> None:
        key = (rule_id, lineno)
        if key in seen:
            return
        seen.add(key)
        alerts.append(
            AuditAlert(
                module="deploycheck",
                rule_id=rule_id,
                severity=severity,  # type: ignore[arg-type]
                title=title,
                message=message,
                file=doc.path,
                location=doc.format_location(lineno),
                evidence=evidence,
            )
        )

    abs_pat = re.compile(
        r"(?:\bMath\.abs|\babs)\s*\(\s*([^)]*(?:coef(?:ficient)?s?|beta(?:s)?|weights?|log_?hr|params|fit\$coefficients|coef\s*\([^)]*\))[^)]*)\)",
        re.IGNORECASE,
    )
    abs_vars: dict[str, tuple[int, str]] = {}

    for lineno, raw_line in enumerate(lines, start=1):
        code_part = re.split(r"(?:#|//)", raw_line, maxsplit=1)[0]
        if not code_part.strip() or _IMPORTANCE_SAFE_RE.search(code_part):
            continue
        m = abs_pat.search(code_part)
        if m:
            inner = m.group(1).strip()
            if _RISK_USAGE_RE.search(code_part):
                _emit(
                    "DEPLOY-004",
                    "CRITICAL",
                    "Coefficient sign destruction via abs() in risk scoring",
                    f"`abs({inner})` strips negative signs from regression coefficients in risk score calculation.",
                    lineno,
                    {"expression": m.group(0), "line_text": code_part.strip()},
                )
            else:
                m_assign = re.match(r"^\s*(?:const|let|var)?\s*([A-Za-z_.][A-Za-z0-9_.]*)\s*(?:<-|=)", code_part)
                if m_assign and not _IMPORTANCE_SAFE_RE.search(m_assign.group(1)):
                    abs_vars[m_assign.group(1)] = (lineno, inner)
        elif abs_vars:
            m_assign = re.match(r"^\s*(?:const|let|var)?\s*([A-Za-z_.][A-Za-z0-9_.]*)\s*(?:<-|=)\s*(.+)$", code_part)
            for vname, (def_line, inner) in list(abs_vars.items()):
                if lineno > def_line and re.search(rf"\b{re.escape(vname)}\b", code_part):
                    if _RISK_USAGE_RE.search(code_part):
                        _emit(
                            "DEPLOY-004",
                            "CRITICAL",
                            "Sign-destroyed coefficients used in clinical risk scoring",
                            f"Variable `{vname}` (created via `abs({inner})` at line {def_line}) is used in risk score computation.",
                            lineno,
                            {"variable": vname, "def_line": def_line, "source_coef": inner},
                        )
                    elif m_assign and m_assign.group(1) not in abs_vars:
                        # Propagate taint through normalization steps (e.g. weights_norm <- weights / max(weights))
                        rhs = m_assign.group(2)
                        if re.search(rf"\b{re.escape(vname)}\b", rhs) and not _IMPORTANCE_SAFE_RE.search(m_assign.group(1)):
                            abs_vars[m_assign.group(1)] = (def_line, inner)

    _check_static_encoding_contradictions(doc, _emit)
    return alerts


def _check_static_encoding_contradictions(doc: CodeDocument, emit_fn: Any) -> None:
    """Detect conflicting {0,1} vs {1,2} sex/gender encodings within the same calculator file."""
    lines = doc.source.splitlines()
    zero_lines: list[int] = []
    two_lines: list[int] = []

    for lineno, raw_line in enumerate(lines, start=1):
        code_part = re.split(r"(?:#|//)", raw_line, maxsplit=1)[0]
        if not code_part.strip():
            continue
        # Detect sex == 0 / female: 0 / male: 0 alongside sex == 2 / female: 2 / male: 2
        if re.search(r"\b(?:sex|gender)\s*===?\s*0\b|['\"](?:female|male)['\"]\s*[:=]\s*0\b", code_part, re.I):
            zero_lines.append(lineno)
        if re.search(r"\b(?:sex|gender)\s*===?\s*2\b|['\"](?:female|male)['\"]\s*[:=]\s*2\b", code_part, re.I):
            two_lines.append(lineno)

    if zero_lines and two_lines:
        emit_fn(
            "DEPLOY-002",
            "HIGH",
            "Contradictory sex/gender categorical encodings ({0,1} vs {1,2}) in calculator code",
            f"Code references both 0-based coding (line {zero_lines[0]}) and 2-valued coding (line {two_lines[0]}) "
            "for sex/gender within the same file.",
            two_lines[0],
            {"zero_coded_lines": zero_lines[:5], "two_coded_lines": two_lines[:5]},
        )


def scan_deploy_code_document(doc: CodeDocument) -> list[AuditAlert]:
    """Static Module 4 audit of Python, R, notebook, or JS risk calculator source code."""
    if doc.language == "python":
        return _scan_python_deploy_code(doc)
    if doc.language in ("r", "javascript"):
        return _scan_r_or_js_deploy_code(doc)
    return []


def scan_deploy_code_file(path: str | Path) -> list[AuditAlert]:
    """Read and statically audit a calculator source file with Module 4."""
    doc = read_code_file(path)
    return scan_deploy_code_document(doc)
