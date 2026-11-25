"""400-Case Deterministic Synthetic Fault-Injection Benchmark for clinaudit.

Generates 400 realistic clinical AI test cases (100 per module: 50 clean negative
controls including challenging edge cases + 50 fault-injected positives spanning
mild, moderate, and severe intensities) with fixed seed=2026.

Outputs:
  - benchmarks/synthetic_benchmark_results.json
  - benchmarks/synthetic_benchmark_summary.csv
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

# Ensure repository root is importable when invoked as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from clinaudit import (  # noqa: E402
    CodeDocument,
    TableData,
    __version__,
    audit_risk_calculator,
    audit_split_leakage,
    audit_tables_suite,
    check_dca_curve,
    scan_code_document,
    scan_deploy_code_document,
)

SEED = 2026


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-zc))


def _make_clinical_cohort(rng: np.random.Generator, n: int, n_cont: int = 8, n_bin: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic mixed continuous/binary clinical cohort (X, y)."""
    cols = []
    # Realistic clinical scales: age, sbp, egfr, albumin, creatinine, hba1c, crp, hemoglobin, ...
    scales = [
        (62.0, 12.0),
        (132.0, 18.0),
        (72.0, 22.0),
        (4.0, 0.5),
        (1.1, 0.45),
        (6.4, 1.3),
        (5.5, 4.0),
        (13.2, 1.8),
    ]
    for j in range(n_cont):
        mu, sd = scales[j % len(scales)]
        cols.append(rng.normal(mu, sd, size=n))
    for _ in range(n_bin):
        p = rng.uniform(0.2, 0.65)
        cols.append(rng.binomial(1, p, size=n).astype(np.float64))
    X = np.column_stack(cols)
    # Generate calibrated binary outcome
    w = rng.normal(0.0, 0.35, size=X.shape[1])
    Xz = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
    logits = Xz @ w - rng.uniform(0.3, 1.2)
    y = rng.binomial(1, _sigmoid(logits)).astype(np.float64)
    return X, y


# =============================================================================
# MODULE 1 BENCHMARK: Split Integrity & Mirror-Clone Detector (50 clean + 50 faulty)
# =============================================================================
def run_module1_benchmark(rng: np.random.Generator) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def _eval_case(
        case_id: str,
        is_faulty: bool,
        intensity: str,
        subtype: str,
        X_tr: np.ndarray,
        X_te: np.ndarray,
        y_tr: np.ndarray | None,
        y_te: np.ndarray | None,
    ) -> None:
        t0 = time.perf_counter()
        alerts, _ = audit_split_leakage(X_tr, X_te, y_train=y_tr, y_test=y_te, tol=1e-10)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "module": "leakage",
                "case_id": case_id,
                "ground_truth_faulty": is_faulty,
                "intensity": intensity,
                "subtype": subtype,
                "predicted_faulty": len(alerts) > 0,
                "n_alerts": len(alerts),
                "rule_ids": sorted({a.rule_id for a in alerts}),
                "runtime_ms": round(rt_ms, 4),
            }
        )

    # --- 50 Clean Negative Controls ---
    # 15 standard mixed clinical cohorts
    for i in range(15):
        n = int(rng.integers(180, 650))
        X, y = _make_clinical_cohort(rng, n, n_cont=int(rng.integers(5, 11)), n_bin=int(rng.integers(2, 5)))
        perm = rng.permutation(n)
        cut = int(n * rng.uniform(0.65, 0.82))
        _eval_case(f"M1_clean_std_{i:02d}", False, "clean", "standard_mixed_split", X[perm[:cut]], X[perm[cut:]], y[perm[:cut]], y[perm[cut:]])

    # 15 low-cardinality binary comorbidity cohorts (genuine heavy overlap, honest random split)
    for i in range(15):
        n = int(rng.integers(220, 600))
        d_bin = int(rng.integers(5, 7))
        probs = rng.uniform(0.25, 0.70, size=d_bin)
        X = np.column_stack([rng.binomial(1, p, size=n).astype(np.float64) for p in probs])
        y = rng.binomial(1, _sigmoid(X @ rng.uniform(0.2, 0.6, size=d_bin) - 1.5)).astype(np.float64)
        perm = rng.permutation(n)
        cut = int(n * rng.uniform(0.65, 0.80))
        _eval_case(f"M1_clean_lowcard_{i:02d}", False, "clean", "low_cardinality_binary", X[perm[:cut]], X[perm[cut:]], y[perm[:cut]], y[perm[cut:]])

    # 10 rounded integer lab cohorts with 1-2 genuine natural duplicate rows before honest random split
    for i in range(10):
        n = int(rng.integers(260, 600))
        X, y = _make_clinical_cohort(rng, n, n_cont=7, n_bin=2)
        X[:, :7] = np.round(X[:, :7], 1)
        # Insert 1 or 2 natural duplicates into raw cohort prior to splitting
        n_nat_dup = int(rng.integers(1, 3))
        dup_src = rng.choice(n, size=n_nat_dup, replace=False)
        X = np.vstack([X, X[dup_src]])
        y = np.concatenate([y, y[dup_src]])
        perm = rng.permutation(X.shape[0])
        cut = int(X.shape[0] * 0.75)
        _eval_case(f"M1_clean_natdup_{i:02d}", False, "clean", "rounded_natural_1_2_dups", X[perm[:cut]], X[perm[cut:]], y[perm[:cut]], y[perm[cut:]])

    # 10 imbalanced high-dimensional clinical cohorts
    for i in range(10):
        n = int(rng.integers(250, 700))
        X, _ = _make_clinical_cohort(rng, n, n_cont=12, n_bin=4)
        y = rng.binomial(1, rng.uniform(0.06, 0.14), size=n).astype(np.float64)
        perm = rng.permutation(n)
        cut = int(n * 0.70)
        _eval_case(f"M1_clean_imbal_{i:02d}", False, "clean", "imbalanced_high_dim", X[perm[:cut]], X[perm[cut:]], y[perm[:cut]], y[perm[cut:]])

    # --- 50 Fault-Injected Positives (17 mild, 17 moderate, 16 severe) ---
    # 18 exact train-test row leakage (6 mild, 6 moderate, 6 severe)
    exact_specs = (
        [("mild", (0.04, 0.08))] * 6
        + [("moderate", (0.15, 0.35))] * 6
        + [("severe", (0.50, 0.95))] * 6
    )
    for i, (inten, (f_lo, f_hi)) in enumerate(exact_specs):
        n = int(rng.integers(240, 560))
        X, y = _make_clinical_cohort(rng, n, n_cont=8, n_bin=3)
        perm = rng.permutation(n)
        cut = int(n * 0.72)
        X_tr, y_tr = X[perm[:cut]].copy(), y[perm[:cut]].copy()
        X_te, y_te = X[perm[cut:]].copy(), y[perm[cut:]].copy()
        k_leak = max(4, int(round(X_te.shape[0] * rng.uniform(f_lo, f_hi))))
        k_leak = min(k_leak, X_te.shape[0], X_tr.shape[0])
        src = rng.choice(X_tr.shape[0], size=k_leak, replace=False)
        dst = rng.choice(X_te.shape[0], size=k_leak, replace=False)
        X_te[dst] = X_tr[src]
        y_te[dst] = y_tr[src]
        _eval_case(f"M1_fault_exact_{i:02d}", True, inten, "exact_row_leakage", X_tr, X_te, y_tr, y_te)

    # 12 near-exact float-perturbed leakage within tol=1e-10 (4 mild, 4 moderate, 4 severe)
    near_specs = (
        [("mild", (0.04, 0.09))] * 4
        + [("moderate", (0.18, 0.35))] * 4
        + [("severe", (0.55, 0.90))] * 4
    )
    for i, (inten, (f_lo, f_hi)) in enumerate(near_specs):
        n = int(rng.integers(220, 520))
        X, y = _make_clinical_cohort(rng, n, n_cont=8, n_bin=2)
        perm = rng.permutation(n)
        cut = int(n * 0.75)
        X_tr, y_tr = X[perm[:cut]].copy(), y[perm[:cut]].copy()
        X_te, y_te = X[perm[cut:]].copy(), y[perm[cut:]].copy()
        k_leak = max(4, int(round(X_te.shape[0] * rng.uniform(f_lo, f_hi))))
        k_leak = min(k_leak, X_te.shape[0], X_tr.shape[0])
        src = rng.choice(X_tr.shape[0], size=k_leak, replace=False)
        dst = rng.choice(X_te.shape[0], size=k_leak, replace=False)
        noise = rng.uniform(-5e-11, 5e-11, size=(k_leak, X_tr.shape[1]))
        X_te[dst] = X_tr[src] + noise
        y_te[dst] = y_tr[src]
        _eval_case(f"M1_fault_near_{i:02d}", True, inten, "near_exact_float_leakage", X_tr, X_te, y_tr, y_te)

    # 10 mirror-clone (1, 1) multiplicity parity injection (3 mild, 4 moderate, 3 severe)
    mirror_specs = (
        [("mild", (16, 22))] * 3
        + [("moderate", (28, 40))] * 4
        + [("severe", (50, 80))] * 3
    )
    for i, (inten, (m_lo, m_hi)) in enumerate(mirror_specs):
        # Mix discrete + continuous mirror-clone scenarios
        n_mirror = int(rng.integers(m_lo, m_hi))
        n_base_tr = int(rng.integers(180, 320))
        n_base_te = int(rng.integers(50, 90))
        if i % 2 == 0:
            # Discrete high-ish cardinality table where mirror parity is the true signature
            d = 10
            X_tr_base = rng.integers(0, 3, size=(n_base_tr, d)).astype(np.float64)
            X_te_base = rng.integers(0, 3, size=(n_base_te, d)).astype(np.float64)
            X_mir = rng.integers(0, 3, size=(n_mirror, d)).astype(np.float64)
            # Ensure mirrored rows are distinct unique profiles
            X_mir = np.unique(X_mir, axis=0)
            while X_mir.shape[0] < n_mirror:
                extra = rng.integers(0, 3, size=(n_mirror, d)).astype(np.float64)
                X_mir = np.unique(np.vstack([X_mir, extra]), axis=0)
            X_mir = X_mir[:n_mirror]
            X_tr = np.vstack([X_tr_base, X_mir])
            X_te = np.vstack([X_te_base, X_mir])
            y_tr = rng.binomial(1, 0.35, size=X_tr.shape[0]).astype(np.float64)
            y_te = rng.binomial(1, 0.35, size=X_te.shape[0]).astype(np.float64)
            y_te[-n_mirror:] = y_tr[-n_mirror:]
        else:
            X_all, y_all = _make_clinical_cohort(rng, n_base_tr + n_base_te + n_mirror, n_cont=7, n_bin=2)
            X_mir, y_mir = X_all[:n_mirror], y_all[:n_mirror]
            X_tr = np.vstack([X_all[n_mirror : n_mirror + n_base_tr], X_mir])
            y_tr = np.concatenate([y_all[n_mirror : n_mirror + n_base_tr], y_mir])
            X_te = np.vstack([X_all[n_mirror + n_base_tr :], X_mir])
            y_te = np.concatenate([y_all[n_mirror + n_base_tr :], y_mir])
        _eval_case(f"M1_fault_mirror_{i:02d}", True, inten, "mirror_clone_parity", X_tr, X_te, y_tr, y_te)

    # 10 conflicting-label identical continuous profiles (4 mild, 3 moderate, 3 severe)
    conflict_specs = [("mild", 1)] * 4 + [("moderate", 4)] * 3 + [("severe", 12)] * 3
    for i, (inten, n_conf) in enumerate(conflict_specs):
        n = int(rng.integers(240, 480))
        X, y = _make_clinical_cohort(rng, n, n_cont=8, n_bin=2)
        perm = rng.permutation(n)
        cut = int(n * 0.75)
        X_tr, y_tr = X[perm[:cut]].copy(), y[perm[:cut]].copy()
        X_te, y_te = X[perm[cut:]].copy(), y[perm[cut:]].copy()
        # Inject identical continuous profiles carrying opposite labels across splits (or within train)
        src = rng.choice(X_tr.shape[0], size=n_conf, replace=False)
        dst = rng.choice(X_te.shape[0], size=n_conf, replace=False)
        if i % 3 == 0:
            # Within-train label conflict only (tests LEAK-003 without train-test overlap!)
            X_tr = np.vstack([X_tr, X_tr[src]])
            y_tr = np.concatenate([y_tr, 1.0 - y_tr[src]])
        else:
            X_te[dst] = X_tr[src]
            y_te[dst] = 1.0 - y_tr[src]
        _eval_case(f"M1_fault_conflict_{i:02d}", True, inten, "conflicting_labels", X_tr, X_te, y_tr, y_te)

    return records


# =============================================================================
# MODULE 2 BENCHMARK: Clinical Evaluation & Baseline Integrity Scanner (50 + 50)
# =============================================================================
def _simulate_clean_dca(rng: np.random.Generator, miscalibrated: bool = False) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    n = int(rng.integers(400, 1500))
    prev = float(rng.uniform(0.18, 0.45))
    y = rng.binomial(1, prev, size=n)
    emp_prev = float(np.mean(y))
    # Continuous predicted probabilities from separation model
    logits = np.where(y == 1, rng.normal(1.2, 1.0, size=n), rng.normal(-1.0, 1.0, size=n))
    if miscalibrated:
        logits = logits * 1.4 + 0.35
    probs = _sigmoid(logits)
    pts = np.linspace(0.05, 0.85, 17)
    nbs = []
    for pt in pts:
        pred = probs >= pt
        tp = float(np.sum(pred & (y == 1)))
        fp = float(np.sum(pred & (y == 0)))
        nbs.append(tp / n - (fp / n) * (pt / (1.0 - pt)))
    treat_all = emp_prev - (1.0 - emp_prev) * (pts / (1.0 - pts))
    return pts, np.asarray(nbs), emp_prev, treat_all


def run_module2_benchmark(rng: np.random.Generator) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def _eval_code(case_id: str, is_faulty: bool, intensity: str, subtype: str, lang: str, src: str) -> None:
        doc = CodeDocument(path=f"{case_id}.{'R' if lang == 'r' else 'py'}", language=lang, source=src)
        t0 = time.perf_counter()
        alerts = scan_code_document(doc)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "module": "evalcheck",
                "case_id": case_id,
                "ground_truth_faulty": is_faulty,
                "intensity": intensity,
                "subtype": subtype,
                "predicted_faulty": len(alerts) > 0,
                "n_alerts": len(alerts),
                "rule_ids": sorted({a.rule_id for a in alerts}),
                "runtime_ms": round(rt_ms, 4),
            }
        )

    def _eval_dca(case_id: str, is_faulty: bool, intensity: str, subtype: str, pts: np.ndarray, nbs: np.ndarray, prev: float | None, ta: np.ndarray | None) -> None:
        t0 = time.perf_counter()
        alerts, _ = check_dca_curve(pts, nbs, prevalence=prev, treat_all=ta)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "module": "evalcheck",
                "case_id": case_id,
                "ground_truth_faulty": is_faulty,
                "intensity": intensity,
                "subtype": subtype,
                "predicted_faulty": len(alerts) > 0,
                "n_alerts": len(alerts),
                "rule_ids": sorted({a.rule_id for a in alerts}),
                "runtime_ms": round(rt_ms, 4),
            }
        )

    # --- 50 Clean Negative Controls ---
    # 10 clean sklearn pipelines (post-split preprocessing, Pipeline SMOTE, predict_proba Brier)
    clean_preprocessors = ["StandardScaler", "MinMaxScaler", "RobustScaler", "SimpleImputer", "SMOTE"]
    for i in range(10):
        prep = clean_preprocessors[i % len(clean_preprocessors)]
        if i % 2 == 0:
            src = f"""
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

pipe = Pipeline([('step', {prep}()), ('clf', RandomForestClassifier(random_state={i}))])
scores = cross_val_score(pipe, X, y, cv=5, scoring='roc_auc')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y)
pipe.fit(X_train, y_train)
y_prob = pipe.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, y_prob)
brier = brier_score_loss(y_test, y_prob)
"""
        else:
            src = f"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, calibration_curve

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
clf = LogisticRegression()
clf.fit(X_train_s, y_train)
probs = clf.predict_proba(X_test_s)[:, 1]
auc = roc_auc_score(y_test, probs)
brier = brier_score_loss(y_test, probs)
frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10)
"""
        _eval_code(f"M2_clean_pipe_{i:02d}", False, "clean", "clean_pipeline", "python", src)

    # 10 tricky clean clinical score evaluations (continuous to ROC, binarized ONLY to confusion_matrix/F1)
    scores_list = ["cha2ds2_vasc", "meld_score", "sofa_score", "framingham_score", "grace_score", "curb65", "has_bled", "fib4_score", "albi_score", "apache_ii"]
    for i, sc in enumerate(scores_list):
        if i < 7:
            src = f"""
from sklearn.metrics import roc_auc_score, confusion_matrix, f1_score, classification_report
auc_{sc} = roc_auc_score(y_test, df['{sc}'])
high_risk = (df['{sc}'] >= 2).astype(int)
cm = confusion_matrix(y_test, high_risk)
f1 = f1_score(y_test, high_risk)
rep = classification_report(y_test, high_risk)
"""
            _eval_code(f"M2_clean_score_{i:02d}", False, "clean", "clean_clinical_score", "python", src)
        else:
            src = f"""
library(pROC)
roc_obj <- pROC::roc(df$outcome, df${sc})
high_risk <- ifelse(df${sc} >= 2, 1, 0)
tab <- table(df$outcome, high_risk)
"""
            _eval_code(f"M2_clean_score_{i:02d}", False, "clean", "clean_clinical_score_r", "r", src)

    # 10 clean DCA (4 clean code + 6 clean simulated tabular curves including miscalibrated & zero-tail)
    for i in range(4):
        src = f"""
import numpy as np
thresholds = np.linspace(0.05, 0.85, 17)
net_benefits = []
for pt in thresholds:
    y_pred = (y_prob >= pt).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    nb = tp / len(y_true) - (fp / len(y_true)) * (pt / (1.0 - pt))
    net_benefits.append(nb)
"""
        _eval_code(f"M2_clean_dca_code_{i:02d}", False, "clean", "clean_dca_code", "python", src)
    for i in range(6):
        pts, nbs, prev, ta = _simulate_clean_dca(rng, miscalibrated=(i >= 3))
        _eval_dca(f"M2_clean_dca_tab_{i:02d}", False, "clean", "clean_dca_tabular", pts, nbs, prev, ta)

    # 10 clean survival pipelines (baseline covariates + time-varying/landmarking safeguards)
    clean_surv_cases = [
        ("python", "from lifelines import KaplanMeierFitter, CoxPHFitter\nkmf = KaplanMeierFitter()\nfor arm in [0, 1]:\n    sub = df[df['randomized_arm'] == arm]\n    kmf.fit(sub['time'], sub['event'])\ncph = CoxPHFitter()\ncph.fit(df[['time', 'event', 'age', 'stage', 'baseline_egfr']], 'time', 'event')\n"),
        ("python", "from lifelines import CoxTimeVaryingFitter\nctv = CoxTimeVaryingFitter()\nctv.fit(df_long, id_col='pid', start_col='tstart', stop_col='tstop', event_col='event')\n# Uses received_adjuvant_chemo as time-varying covariate\n"),
        ("python", "from lifelines import KaplanMeierFitter\nlandmark_time = 90\ndf_lm = df[df['time'] >= landmark_time]\nkmf = KaplanMeierFitter()\nfor grp in [0, 1]:\n    sub = df_lm[df_lm['received_chemo'] == grp]\n    kmf.fit(sub['time'] - landmark_time, sub['event'])\n"),
        ("r", "library(survival)\nfit <- survfit(Surv(time, status) ~ baseline_stage + age, data = cohort)\ncox <- coxph(Surv(time, status) ~ age + sex + baseline_egfr, data = cohort)\n"),
        ("r", "library(survival)\n# Counting-process time-dependent Cox avoids immortal-time bias\ncox_tv <- coxph(Surv(tstart, tstop, event) ~ post_discharge_statin + age, data = td_data)\n"),
    ]
    for i in range(10):
        lang, src = clean_surv_cases[i % len(clean_surv_cases)]
        _eval_code(f"M2_clean_surv_{i:02d}", False, "clean", "clean_survival", lang, src)

    # 10 clean validation early-stopping & statsmodels/Keras probabilistic calibration
    for i in range(10):
        if i % 2 == 0:
            src = f"""
import xgboost as xgb
import torch
model = xgb.XGBClassifier(n_estimators=200)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
best_val_auc = 0.0
for epoch in range(20):
    val_auc = evaluate(net, val_loader)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(net.state_dict(), 'best_val.pt')
final_test_auc = evaluate(net, test_loader)
"""
        else:
            src = f"""
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, roc_auc_score
logit_res = sm.Logit(y_train, X_train).fit(disp=False)
probs = logit_res.predict(X_test)
auc = roc_auc_score(y_test, probs)
brier = brier_score_loss(y_test, probs)
"""
        _eval_code(f"M2_clean_val_{i:02d}", False, "clean", "clean_val_and_prob", "python", src)

    # --- 50 Fault-Injected Positives (17 mild, 17 moderate, 16 severe) ---
    # 9 (a) Binarized clinical risk scores passed to ROC/Brier/DeLong (3 mild, 3 moderate, 3 severe)
    bin_score_cases = [
        ("mild", "python", "from sklearn.metrics import roc_auc_score\nchads_bin = (df['cha2ds2_vasc'] >= 2).astype(int)\nauc = roc_auc_score(y_test, chads_bin)\n"),
        ("mild", "python", "from sklearn.metrics import roc_curve\nmeld_high = np.where(meld_score > 15, 1, 0)\nfpr, tpr, _ = roc_curve(y_test, meld_high)\n"),
        ("mild", "python", "from sklearn.metrics import brier_score_loss\nsofa_flag = (sofa_score >= 2).astype(int)\nbrier = brier_score_loss(y_test, sofa_flag)\n"),
        ("moderate", "python", "from sklearn.metrics import roc_auc_score\nauc = roc_auc_score(y_test, (framingham_score >= 10).astype(int))\n"),
        ("moderate", "python", "from sklearn.metrics import roc_auc_score\ngrace_cat = pd.cut(grace_score, bins=[0, 140, 300], labels=[0, 1])\nauc = roc_auc_score(y_test, grace_cat)\n"),
        ("moderate", "r", "library(pROC)\ncurb_bin <- ifelse(df$curb65 >= 2, 1, 0)\nroc_curb <- pROC::roc(df$death, curb_bin)\n"),
        ("severe", "python", "from sklearn.metrics import roc_auc_score\nauc = roc_auc_score(y_test, has_bled >= 3)\n"),
        ("severe", "python", "from sklearn.metrics import roc_auc_score\np = delong_roc_test(y_test, ml_prob, (fib4_score > 2.67).astype(int))\n"),
        ("severe", "r", "library(pROC)\nroc_meld <- roc(df$mortality, as.numeric(df$meld_na > 20))\n"),
    ]
    for i, (inten, lang, src) in enumerate(bin_score_cases):
        _eval_code(f"M2_fault_binscore_{i:02d}", True, inten, "binarized_clinical_score", lang, src)

    # 10 (b) DCA defects: 4 code + 6 tabular curves (3 mild, 4 moderate, 3 severe)
    dca_code_bugs = [
        ("moderate", "python", "import numpy as np\ny_pred = (y_prob >= 0.5).astype(int)\nnet_benefits = []\nfor pt in thresholds:\n    tp = np.sum((y_pred == 1) & (y_true == 1))\n    fp = np.sum((y_pred == 1) & (y_true == 0))\n    net_benefits.append(tp/n - fp/n * (pt/(1-pt)))\n"),
        ("moderate", "python", "import numpy as np\nclf = RandomForestClassifier()\ny_pred = clf.predict(X_test)\nnb_list = []\nfor t in thresh_grid:\n    tp = ((y_pred == 1) & (y_test == 1)).sum()\n    fp = ((y_pred == 1) & (y_test == 0)).sum()\n    nb_list.append(tp/N - fp/N * (t/(1-t)))\n"),
        ("severe", "python", "import numpy as np\nnb_single = 0.24\nnet_benefit = np.full_like(thresholds, nb_single)\n"),
        ("severe", "r", "nb_val <- 0.19\nnet_benefit <- rep(nb_val, length(thresholds))\n"),
    ]
    for i, (inten, lang, src) in enumerate(dca_code_bugs):
        _eval_code(f"M2_fault_dcacode_{i:02d}", True, inten, "hardcoded_dca_code", lang, src)

    # 6 tabular DCA faults (3 mild/mod single-threshold & constant, 3 severe impossible)
    pts_grid = np.linspace(0.05, 0.85, 17)
    odds_grid = pts_grid / (1.0 - pts_grid)
    # 2 constant flat NB curves
    for i in range(2):
        prev = float(rng.uniform(0.25, 0.40))
        ta = prev - (1.0 - prev) * odds_grid
        nb_flat = np.full_like(pts_grid, prev * 0.65)
        _eval_dca(f"M2_fault_dcaflat_{i:02d}", True, "severe" if i == 1 else "moderate", "flat_constant_dca", pts_grid, nb_flat, prev, ta)
    # 2 exact single-threshold fixed-binary DCA lines
    for i in range(2):
        prev = float(rng.uniform(0.25, 0.40))
        ta = prev - (1.0 - prev) * odds_grid
        tp_rate = prev * 0.80
        fp_rate = (1.0 - prev) * 0.18
        nb_fixed = tp_rate - fp_rate * odds_grid
        _eval_dca(f"M2_fault_dcafixed_{i:02d}", True, "mild", "fixed_binary_dca_line", pts_grid, nb_fixed, prev, ta)
    # 2 mathematically impossible DCA curves (exceeds prevalence / rising treat-all)
    prev = 0.22
    ta_ok = prev - (1.0 - prev) * odds_grid
    nb_over = np.linspace(0.35, 0.28, 17)  # exceeds prevalence=0.22!
    _eval_dca("M2_fault_dcabound_00", True, "severe", "dca_exceeds_prevalence", pts_grid, nb_over, prev, ta_ok)
    ta_bad = np.linspace(-0.20, 0.30, 17)  # increasing treat_all!
    nb_climb = np.linspace(0.02, 0.25, 17)  # monotonically rising NB into high thresholds!
    _eval_dca("M2_fault_dcabound_01", True, "mild", "dca_impossible_monotonicity", pts_grid, nb_climb, None, ta_bad)

    # 8 (c) Binary {0,1} to Brier/calibration + constant NRI (3 mild, 3 moderate, 2 severe)
    bin_cal_cases = [
        ("mild", "python", "from sklearn.ensemble import RandomForestClassifier\nfrom sklearn.metrics import brier_score_loss\nclf = RandomForestClassifier().fit(X_train, y_train)\ny_pred = clf.predict(X_test)\nbrier = brier_score_loss(y_test, y_pred)\n"),
        ("mild", "python", "from sklearn.metrics import calibration_curve\ny_hat = (y_prob >= 0.5).astype(int)\nfrac_pos, mean_pred = calibration_curve(y_test, y_hat, n_bins=10)\n"),
        ("mild", "python", "nri_list = []\npred_bin = (probs >= 0.5).astype(int)\nfor cutoff in cutoffs:\n    nri_val = compute_nri(y_true, base_bin, pred_bin)\n    nri_list.append(nri_val)\n"),
        ("moderate", "python", "from xgboost import XGBClassifier\nfrom sklearn.metrics import log_loss\nxgb_clf = XGBClassifier().fit(X_train, y_train)\nloss = log_loss(y_test, xgb_clf.predict(X_test))\n"),
        ("moderate", "python", "from sklearn.metrics import brier_score_loss\nbrier = brier_score_loss(y_test, np.round(y_prob))\n"),
        ("moderate", "r", "pred_cls <- ifelse(probs >= 0.5, 1, 0)\nval.prob(pred_cls, df$outcome)\n"),
        ("severe", "python", "from sklearn.linear_model import LogisticRegression\nfrom sklearn.metrics import brier_score_loss, calibration_curve\nlr = LogisticRegression().fit(X_train, y_train)\nb = brier_score_loss(y_test, lr.predict(X_test))\nf, m = calibration_curve(y_test, lr.predict(X_test))\n"),
        ("severe", "python", "nri_scores = []\nfor thresh in thresholds:\n    val = calc_nri(y_test, old_class, new_class)\n    nri_scores.append(val)\n"),
    ]
    for i, (inten, lang, src) in enumerate(bin_cal_cases):
        _eval_code(f"M2_fault_bincal_{i:02d}", True, inten, "binary_calibration_or_nri", lang, src)

    # 9 (d) Pre-split oversampling / scaling / feature selection (3 mild, 3 moderate, 3 severe)
    presplit_cases = [
        ("mild", "python", "from sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import train_test_split\nscaler = StandardScaler()\nX_scaled = scaler.fit_transform(X)\nX_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)\n"),
        ("mild", "python", "from sklearn.impute import KNNImputer\nfrom sklearn.model_selection import cross_val_score\nX_imp = KNNImputer().fit_transform(X)\nscores = cross_val_score(clf, X_imp, y, cv=5)\n"),
        ("mild", "python", "from sklearn.preprocessing import StandardScaler\nscaler = StandardScaler()\nX_train_s = scaler.fit_transform(X_train)\nX_test_s = scaler.fit_transform(X_test)\n"),
        ("moderate", "python", "from sklearn.feature_selection import SelectKBest\nfrom sklearn.model_selection import train_test_split\nX_sel = SelectKBest(k=10).fit_transform(X, y)\nX_tr, X_te, y_tr, y_te = train_test_split(X_sel, y)\n"),
        ("moderate", "python", "from imblearn.over_sampling import ADASYN\nfrom sklearn.model_selection import StratifiedKFold\nX_res, y_res = ADASYN().fit_resample(X, y)\nskf = StratifiedKFold(n_splits=5)\nfor tr, te in skf.split(X_res, y_res): pass\n"),
        ("moderate", "r", "library(caret)\nX_scaled <- scale(df_all)\nidx <- createDataPartition(X_scaled$y, p = 0.8)\n"),
        ("severe", "python", "from imblearn.over_sampling import SMOTE\nfrom sklearn.model_selection import train_test_split\nX_res, y_res = SMOTE(random_state=42).fit_resample(X, y)\nX_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25)\n"),
        ("severe", "python", "from imblearn.combine import SMOTETomek\nfrom sklearn.model_selection import cross_val_score\nX_bal, y_bal = SMOTETomek().fit_resample(X, y)\nscores = cross_val_score(rf, X_bal, y_bal, cv=10)\n"),
        ("severe", "r", "library(ROSE)\ndf_bal <- ovun.sample(outcome ~ ., data = full_df)$data\nsplit_idx <- createDataPartition(df_bal$outcome, p = 0.7)\n"),
    ]
    for i, (inten, lang, src) in enumerate(presplit_cases):
        _eval_code(f"M2_fault_presplit_{i:02d}", True, inten, "presplit_leakage", lang, src)

    # 7 (e) Immortal-time bias (3 mild, 2 moderate, 2 severe)
    immortal_cases = [
        ("mild", "python", "from lifelines import KaplanMeierFitter\nkmf = KaplanMeierFitter()\nfor grp in [0, 1]:\n    sub = df[df['post_discharge_statin'] == grp]\n    kmf.fit(sub['time'], sub['death'])\n"),
        ("mild", "python", "from lifelines import CoxPHFitter\ncph = CoxPHFitter()\ncph.fit(df[['time', 'event', 'age', 'length_of_stay']], duration_col='time', event_col='event')\n"),
        ("mild", "r", "library(survival)\nfit <- survfit(Surv(os_time, os_event) ~ responder_status, data = cohort)\n"),
        ("moderate", "python", "from lifelines.statistics import logrank_test\ng1 = df[df['received_adjuvant_chemo'] == 1]\ng0 = df[df['received_adjuvant_chemo'] == 0]\nres = logrank_test(g1['time'], g0['time'], g1['event'], g0['event'])\n"),
        ("moderate", "r", "library(survival)\ncox <- coxph(Surv(time, death) ~ ever_treated + age + sex, data = df)\n"),
        ("severe", "python", "from lifelines import KaplanMeierFitter\nkmf = KaplanMeierFitter()\nkmf.fit(df[df['completed_cycles'] >= 4]['time'], df[df['completed_cycles'] >= 4]['event'])\n"),
        ("severe", "r", "library(survminer)\nfit <- survfit(Surv(time, status) ~ post_op_therapy, data = clinical_df)\nggsurvplot(fit)\n"),
    ]
    for i, (inten, lang, src) in enumerate(immortal_cases):
        _eval_code(f"M2_fault_immortal_{i:02d}", True, inten, "immortal_time_bias", lang, src)

    # 7 (f) Test-set checkpoint / hyperparameter selection (2 mild, 2 moderate, 3 severe)
    test_sel_cases = [
        ("mild", "python", "model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20)\n"),
        ("mild", "python", "keras_model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test))\n"),
        ("moderate", "python", "best_auc = 0.0\nfor c in [0.01, 0.1, 1.0, 10.0]:\n    clf = LogisticRegression(C=c).fit(X_train, y_train)\n    test_auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])\n    if test_auc > best_auc:\n        best_auc = test_auc\n        best_model = clf\n"),
        ("moderate", "python", "grid = GridSearchCV(rf, param_grid, cv=5)\ngrid.fit(X_test, y_test)\n"),
        ("severe", "python", "best_score = 0.0\nfor epoch in range(100):\n    train_epoch(net, train_loader)\n    test_auc = eval_loader(net, test_loader)\n    if test_auc > best_score:\n        best_score = test_auc\n        torch.save(net.state_dict(), 'best_ckpt.pt')\n"),
        ("severe", "python", "lgb.train(params, dtrain, valid_sets=[dtest], num_boost_round=500)\n"),
        ("severe", "python", "best_f1 = 0\nfor depth in range(2, 12):\n    m = XGBClassifier(max_depth=depth).fit(X_train, y_train)\n    test_f1 = f1_score(y_test, m.predict(X_test))\n    if test_f1 > best_f1:\n        best_f1 = test_f1\n        best_params = {'max_depth': depth}\n"),
    ]
    for i, (inten, lang, src) in enumerate(test_sel_cases):
        _eval_code(f"M2_fault_testsel_{i:02d}", True, inten, "test_set_selection", lang, src)

    return records


# =============================================================================
# MODULE 3 BENCHMARK: Statistical Inference & CI Sanity Auditor (50 + 50)
# =============================================================================
def _hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float:
    q1 = auc / (2.0 - auc)
    q2 = (2.0 * auc * auc) / (1.0 + auc)
    var = (auc * (1.0 - auc) + (n_pos - 1) * (q1 - auc * auc) + (n_neg - 1) * (q2 - auc * auc)) / max(n_pos * n_neg, 1)
    return math.sqrt(max(var, 1e-10))


def _make_table(name: str, headers: list[str], rows: list[list[Any]], raw_rows: list[list[str]] | None = None) -> TableData:
    if raw_rows is None:
        raw_rows = [["" if v is None else str(v) for v in r] for r in rows]
    return TableData(name=name, headers=headers, rows=rows, raw_rows=raw_rows, source_file=f"{name}.xlsx")


def run_module3_benchmark(rng: np.random.Generator) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def _eval_tables(
        case_id: str,
        is_faulty: bool,
        intensity: str,
        subtype: str,
        tables: list[TableData],
        ctx: dict[str, Any] | None = None,
    ) -> None:
        t0 = time.perf_counter()
        alerts, _ = audit_tables_suite(tables, methods_context=ctx)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "module": "statcheck",
                "case_id": case_id,
                "ground_truth_faulty": is_faulty,
                "intensity": intensity,
                "subtype": subtype,
                "predicted_faulty": len(alerts) > 0,
                "n_alerts": len(alerts),
                "rule_ids": sorted({a.rule_id for a in alerts}),
                "runtime_ms": round(rt_ms, 4),
            }
        )

    # --- 50 Clean Negative Controls ---
    # 12 genuine Hanley-McNeil / bootstrap tables (3-4 decimals, varied N)
    for i in range(12):
        n_rows = int(rng.integers(6, 14))
        rows, raws = [], []
        for r in range(n_rows):
            n_tot = int(rng.integers(180, 2500))
            auc = float(rng.uniform(0.68, 0.94))
            se = _hanley_mcneil_se(auc, int(n_tot * 0.3), int(n_tot * 0.7)) * float(rng.uniform(0.9, 1.15))
            lo = max(0.5, auc - 1.96 * se)
            hi = min(0.999, auc + 1.96 * se)
            cell = f"{auc:.3f} ({lo:.3f}-{hi:.3f})"
            rows.append([f"Model_{r}", n_tot, cell])
            raws.append([f"Model_{r}", str(n_tot), cell])
        _eval_tables(f"M3_clean_hm_{i:02d}", False, "clean", "genuine_hanley_mcneil", [_make_table("Table1", ["Model", "N", "AUROC"], rows, raws)])

    # 12 tricky clean 2-decimal rounded tables on same large cohort (natural legitimate rounding ties!)
    for i in range(12):
        n_rows = int(rng.integers(6, 10))
        n_tot = int(rng.integers(1500, 4000))
        base_auc = float(rng.uniform(0.80, 0.86))
        rows, raws = [], []
        for r in range(n_rows):
            auc = float(np.clip(base_auc + rng.uniform(-0.03, 0.03), 0.75, 0.91))
            se = _hanley_mcneil_se(auc, int(n_tot * 0.3), int(n_tot * 0.7))
            lo = round(auc - 1.96 * se, 2)
            hi = round(auc + 1.96 * se, 2)
            est = round(auc, 2)
            rows.append([f"Alg_{r}", n_tot, est, lo, hi])
            raws.append([f"Alg_{r}", str(n_tot), f"{est:.2f}", f"{lo:.2f}", f"{hi:.2f}"])
        _eval_tables(f"M3_clean_2dec_{i:02d}", False, "clean", "clean_2dec_rounding_ties", [_make_table("Table2", ["Model", "N", "AUC", "CI_low", "CI_high"], rows, raws)])

    # 10 large-cohort tables (N=15,000-80,000) with legitimately narrow CIs and legitimately tiny P-values!
    for i in range(10):
        rows, raws = [], []
        for r in range(6):
            n_tot = int(rng.integers(15000, 80000))
            auc1 = float(rng.uniform(0.82, 0.88))
            delta = float(rng.uniform(0.02, 0.05))
            auc2 = min(0.96, auc1 + delta)
            se1 = _hanley_mcneil_se(auc1, int(n_tot * 0.25), int(n_tot * 0.75))
            se2 = _hanley_mcneil_se(auc2, int(n_tot * 0.25), int(n_tot * 0.75))
            # True paired SE at rho=0.8
            se_d = math.sqrt(se1 * se1 + se2 * se2 - 1.6 * se1 * se2)
            z = delta / max(se_d, 1e-9)
            p_true = max(1e-25, math.erfc(z / math.sqrt(2.0)))
            c1 = f"{auc1:.4f} ({auc1 - 1.96*se1:.4f}-{auc1 + 1.96*se1:.4f})"
            c2 = f"{auc2:.4f} ({auc2 - 1.96*se2:.4f}-{auc2 + 1.96*se2:.4f})"
            rows.append([f"Cohort_{r}", n_tot, round(delta, 4), c1, c2, p_true])
            raws.append([f"Cohort_{r}", str(n_tot), f"{delta:.4f}", c1, c2, f"{p_true:.3e}"])
        _eval_tables(f"M3_clean_largeN_{i:02d}", False, "clean", "large_cohort_narrow_ci", [_make_table("LargeN", ["Cohort", "N", "delta", "AUC_Base", "AUC_New", "p_value"], rows, raws)])

    # 8 multi-sheet matched-task tables with legitimate modest CI ratio (1.1x - 2.8x)
    for i in range(8):
        rows_a, rows_b, raws_a, raws_b = [], [], [], []
        for r in range(6):
            auc = float(rng.uniform(0.75, 0.90))
            hw_a = float(rng.uniform(0.015, 0.035))
            hw_b = hw_a * float(rng.uniform(0.55, 2.2))
            ca = f"{auc:.3f} ({auc - hw_a:.3f}-{auc + hw_a:.3f})"
            cb = f"{auc:.3f} ({auc - hw_b:.3f}-{auc + hw_b:.3f})"
            rows_a.append([f"Task_{r}", ca])
            raws_a.append([f"Task_{r}", ca])
            rows_b.append([f"Task_{r}", cb])
            raws_b.append([f"Task_{r}", cb])
        t_a = _make_table("Fig2a", ["Task", "AUROC"], rows_a, raws_a)
        t_b = _make_table("SuppTable1", ["Task", "AUROC"], rows_b, raws_b)
        _eval_tables(f"M3_clean_multi_{i:02d}", False, "clean", "consistent_multi_sheet", [t_a, t_b])

    # 8 clean valid 5-fold Wilcoxon + single boundary 1.00(1.00-1.00) edge cases
    for i in range(8):
        rows, raws = [], []
        for r in range(5):
            auc = float(rng.uniform(0.76, 0.91))
            hw = float(rng.uniform(0.014, 0.042))
            p_valid = float(rng.choice([0.0625, 0.125, 0.1875, 0.3125, 0.5]))
            c = f"{auc:.3f} ({auc - hw:.3f}-{auc + hw:.3f})"
            rows.append([f"FoldTask_{r}", c, p_valid, "5-fold Wilcoxon"])
            raws.append([f"FoldTask_{r}", c, f"{p_valid:.4f}", "5-fold Wilcoxon"])
        if i % 2 == 0:
            # Add ONE legitimate degenerate boundary specificity row 1.00 (1.00-1.00)
            rows.append(["BoundarySpec", "1.000 (1.000-1.000)", 0.125, "5-fold Wilcoxon"])
            raws.append(["BoundarySpec", "1.000 (1.000-1.000)", "0.1250", "5-fold Wilcoxon"])
        _eval_tables(f"M3_clean_wilcox_{i:02d}", False, "clean", "valid_wilcoxon_and_boundary", [_make_table("CV5", ["Task", "AUC", "p_value", "method"], rows, raws)])

    # --- 50 Fault-Injected Positives (17 mild, 17 moderate, 16 severe) ---
    # 11 (a) Fixed-width placeholder CIs (4 mild, 4 moderate, 3 severe)
    fixed_specs = (
        [("mild", 6, 5)] * 4
        + [("moderate", 8, 8)] * 4
        + [("severe", 12, 12)] * 3
    )
    for i, (inten, n_rows, n_const) in enumerate(fixed_specs):
        hw_const = float(rng.choice([0.012, 0.015, 0.018, 0.020, 0.025]))
        rows, raws = [], []
        for r in range(n_rows):
            n_tot = int(rng.integers(120, 1800))
            auc = float(rng.uniform(0.65, 0.92))
            hw = hw_const if r < n_const else float(rng.uniform(0.008, 0.045))
            if i % 2 == 0:
                c = f"{auc:.3f} ({auc - hw:.3f}-{auc + hw:.3f})"
                rows.append([f"Cohort_{r}", n_tot, c])
                raws.append([f"Cohort_{r}", str(n_tot), c])
            else:
                c = f"{auc:.3f} ± {hw:.3f}"
                rows.append([f"Cohort_{r}", n_tot, c])
                raws.append([f"Cohort_{r}", str(n_tot), c])
        _eval_tables(f"M3_fault_fixedci_{i:02d}", True, inten, "fixed_width_placeholder_ci", [_make_table("FixedCI", ["Cohort", "N", "AUROC"], rows, raws)])

    # 10 (b) Zero-width CIs (lo == hi) on interior metrics (3 mild, 4 moderate, 3 severe)
    zero_specs = [("mild", 1)] * 3 + [("moderate", 3)] * 4 + [("severe", 7)] * 3
    for i, (inten, n_zero) in enumerate(zero_specs):
        rows, raws = [], []
        for r in range(8):
            auc = float(rng.uniform(0.70, 0.91))
            hw = 0.0 if r < n_zero else float(rng.uniform(0.015, 0.040))
            rows.append([f"M_{r}", round(auc, 3), round(auc - hw, 3), round(auc + hw, 3)])
            raws.append([f"M_{r}", f"{auc:.3f}", f"{auc - hw:.3f}", f"{auc + hw:.3f}"])
        _eval_tables(f"M3_fault_zeroci_{i:02d}", True, inten, "zero_width_ci", [_make_table("ZeroCI", ["Model", "AUC", "CI_low", "CI_high"], rows, raws)])

    # 11 (c) Cross-sheet CI width contradictions (>5x and sqrt(B) SE-of-bootstrap bug) (4 mild, 3 moderate, 4 severe)
    cross_specs = (
        [("mild", float(rng.uniform(5.6, 8.5)))] * 4
        + [("moderate", math.sqrt(200.0))] * 3
        + [("severe", math.sqrt(1000.0))] * 2
        + [("severe", math.sqrt(2000.0))] * 2
    )
    for i, (inten, ratio) in enumerate(cross_specs):
        rows_a, rows_b, raws_a, raws_b = [], [], [], []
        for r in range(6):
            auc = float(rng.uniform(0.75, 0.91))
            hw_true = float(rng.uniform(0.025, 0.055))
            hw_bug = hw_true / ratio if r < 4 else hw_true * float(rng.uniform(0.9, 1.1))
            ca = f"{auc:.4f} ({auc - hw_true:.4f}-{auc + hw_true:.4f})"
            cb = f"{auc:.4f} ({auc - hw_bug:.4f}-{auc + hw_bug:.4f})"
            rows_a.append([f"Endpoint_{r}", ca])
            raws_a.append([f"Endpoint_{r}", ca])
            rows_b.append([f"Endpoint_{r}", cb])
            raws_b.append([f"Endpoint_{r}", cb])
        t_a = _make_table("MainFig", ["Endpoint", "AUROC"], rows_a, raws_a)
        t_b = _make_table("SourceData_ED", ["Endpoint", "AUROC"], rows_b, raws_b)
        _eval_tables(f"M3_fault_cross_{i:02d}", True, inten, "cross_sheet_ci_contradiction", [t_a, t_b])

    # 12 (d) Impossible 5-fold Wilcoxon P-values (5) + bootstrap pseudoreplication P-values (7)
    # 5 impossible 5-fold Wilcoxon (2 mild, 2 moderate, 1 severe)
    wilcox_bad = [("mild", 0.021), ("mild", 0.012), ("moderate", 0.003), ("moderate", 0.0004), ("severe", 1.2e-7)]
    for i, (inten, p_bad) in enumerate(wilcox_bad):
        rows = [[f"Task_{r}", f"{0.80 + 0.02*r:.3f} ({0.77 + 0.02*r:.3f}-{0.83 + 0.02*r:.3f})", p_bad if r == 0 else 0.125, "5-fold Wilcoxon"] for r in range(5)]
        _eval_tables(f"M3_fault_wilcox_{i:02d}", True, inten, "impossible_5fold_wilcoxon", [_make_table("WilcoxBad", ["Task", "AUC", "p_value", "method"], rows)])

    # 7 bootstrap pseudoreplication microscopic P-values on modest N / tiny delta (2 mild, 2 moderate, 3 severe)
    pseudo_specs = [
        ("mild", 600, 0.016, 2.5e-11),
        ("mild", 800, 0.014, 1.0e-12),
        ("moderate", 400, 0.012, 3.0e-16),
        ("moderate", 350, 0.009, 5.0e-20),
        ("severe", 200, 0.008, 1.0e-35),
        ("severe", 150, 0.005, 1.0e-50),
        ("severe", None, 0.007, 4.2e-18),
    ]
    for i, (inten, n_val, d_val, p_bad) in enumerate(pseudo_specs):
        if n_val is not None:
            rows = [[f"Cmp_{r}", n_val, d_val if r == 0 else 0.03, p_bad if r == 0 else 0.042] for r in range(4)]
            raws = [[f"Cmp_{r}", str(n_val), f"{d_val if r == 0 else 0.03:.4f}", f"{p_bad:.2e}" if r == 0 else "0.042"] for r in range(4)]
            _eval_tables(f"M3_fault_pseudo_{i:02d}", True, inten, "bootstrap_pseudoreplication_p", [_make_table("PseudoP", ["Comparison", "N", "delta_auc", "p_value"], rows, raws)])
        else:
            rows = [[f"Cmp_{r}", d_val if r == 0 else 0.03, p_bad if r == 0 else 0.042] for r in range(4)]
            raws = [[f"Cmp_{r}", f"{d_val if r == 0 else 0.03:.4f}", f"{p_bad:.2e}" if r == 0 else "0.042"] for r in range(4)]
            _eval_tables(f"M3_fault_pseudo_{i:02d}", True, inten, "bootstrap_pseudoreplication_p", [_make_table("PseudoP", ["Comparison", "delta_auc", "p_value"], rows, raws)])

    # 6 (e) Paired-Z contradiction between overlapping CIs and microscopic reported P-value (2 mild, 2 moderate, 2 severe)
    pz_specs = [
        ("mild", 0.015, 0.030, 5e-6),
        ("mild", 0.012, 0.028, 1e-6),
        ("moderate", 0.010, 0.032, 1e-10),
        ("moderate", 0.008, 0.035, 1e-12),
        ("severe", 0.005, 0.040, 1e-20),
        ("severe", 0.004, 0.038, 1e-25),
    ]
    for i, (inten, delta, hw, p_bad) in enumerate(pz_specs):
        e1 = 0.820
        e2 = e1 + delta
        c1 = f"{e1:.3f} ({e1 - hw:.3f}-{e1 + hw:.3f})"
        c2 = f"{e2:.3f} ({e2 - hw:.3f}-{e2 + hw:.3f})"
        rows = [["MainTask", c1, c2, p_bad]]
        raws = [["MainTask", c1, c2, f"{p_bad:.2e}"]]
        _eval_tables(f"M3_fault_pairedz_{i:02d}", True, inten, "paired_z_ci_contradiction", [_make_table("PairedZ", ["Task", "AUC_Base", "AUC_Model", "p_value"], rows, raws)])

    return records


# =============================================================================
# MODULE 4 BENCHMARK: Clinical Calculator Axiom & Monotonicity Auditor (50 + 50)
# =============================================================================
def _make_calculator_grid(rng: np.random.Generator, n: int = 120) -> tuple[list[str], np.ndarray]:
    feat_names = ["age", "sex", "sbp", "egfr", "albumin", "creatinine", "uacr", "diabetes"]
    X = np.column_stack(
        [
            rng.uniform(40.0, 85.0, size=n),     # age (+1)
            rng.binomial(1, 0.5, size=n),        # sex {0, 1}
            rng.uniform(105.0, 175.0, size=n),   # sbp (+1)
            rng.uniform(20.0, 110.0, size=n),    # egfr (-1)
            rng.uniform(2.6, 4.8, size=n),       # albumin (-1)
            rng.uniform(0.6, 3.2, size=n),       # creatinine (+1)
            rng.uniform(5.0, 600.0, size=n),     # uacr (+1)
            rng.binomial(1, 0.3, size=n),        # diabetes (+1)
        ]
    ).astype(np.float64)
    return feat_names, X


def _standard_logit(X: np.ndarray, coefs: np.ndarray, intercept: float) -> np.ndarray:
    means = np.array([62.0, 0.5, 135.0, 65.0, 3.8, 1.3, 120.0, 0.3])
    stds = np.array([12.0, 0.5, 20.0, 25.0, 0.5, 0.6, 150.0, 0.45])
    Z = (X - means) / stds
    return intercept + Z @ coefs


def run_module4_benchmark(rng: np.random.Generator) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def _eval_calc(
        case_id: str,
        is_faulty: bool,
        intensity: str,
        subtype: str,
        models: Any,
        fnames: list[str],
        X: np.ndarray,
        hpairs: list[tuple[str, str]] | None = None,
    ) -> None:
        t0 = time.perf_counter()
        alerts, _ = audit_risk_calculator(models, fnames, X, hierarchy_pairs=hpairs)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "module": "deploycheck",
                "case_id": case_id,
                "ground_truth_faulty": is_faulty,
                "intensity": intensity,
                "subtype": subtype,
                "predicted_faulty": len(alerts) > 0,
                "n_alerts": len(alerts),
                "rule_ids": sorted({a.rule_id for a in alerts}),
                "runtime_ms": round(rt_ms, 4),
            }
        )

    def _eval_code(case_id: str, is_faulty: bool, intensity: str, subtype: str, lang: str, src: str) -> None:
        doc = CodeDocument(path=f"{case_id}.{'R' if lang == 'r' else 'py'}", language=lang, source=src)
        t0 = time.perf_counter()
        alerts = scan_deploy_code_document(doc)
        rt_ms = (time.perf_counter() - t0) * 1000.0
        records.append(
            {
                "module": "deploycheck",
                "case_id": case_id,
                "ground_truth_faulty": is_faulty,
                "intensity": intensity,
                "subtype": subtype,
                "predicted_faulty": len(alerts) > 0,
                "n_alerts": len(alerts),
                "rule_ids": sorted({a.rule_id for a in alerts}),
                "runtime_ms": round(rt_ms, 4),
            }
        )

    # --- 50 Clean Negative Controls ---
    # 18 clean calibrated multi-endpoint calculators (competing-risks nested P_cause <= P_all)
    for i in range(18):
        fnames, X = _make_calculator_grid(rng, n=120)
        b_all = np.array([
            rng.uniform(0.4, 0.8),    # age +
            rng.uniform(0.15, 0.35),  # sex +
            rng.uniform(0.15, 0.4),   # sbp +
            -rng.uniform(0.35, 0.7),  # egfr -
            -rng.uniform(0.3, 0.6),   # albumin -
            rng.uniform(0.2, 0.5),    # creatinine +
            rng.uniform(0.2, 0.45),   # uacr +
            rng.uniform(0.2, 0.5),    # diabetes +
        ])
        b0 = float(rng.uniform(-1.6, -0.6))
        share = float(rng.uniform(0.35, 0.65))

        def _make_fns(coef: np.ndarray, ic: float, sh: float) -> dict[str, Any]:
            def fn_all(arr: np.ndarray) -> np.ndarray:
                return _sigmoid(_standard_logit(arr, coef, ic))

            def fn_cause(arr: np.ndarray) -> np.ndarray:
                return fn_all(arr) * sh

            return {"all_cause": fn_all, "cv_mortality": fn_cause}

        _eval_calc(f"M4_clean_multi_{i:02d}", False, "clean", "clean_multi_endpoint", _make_fns(b_all, b0, share), fnames, X, [("cv_mortality", "all_cause")])

    # 12 tricky clean noisy piecewise/tree-like calculators (small local +/-0.002 step jitter, valid global trend)
    for i in range(12):
        fnames, X = _make_calculator_grid(rng, n=100)
        b_all = np.array([0.6, 0.25, 0.3, -0.55, -0.45, 0.35, 0.3, 0.35])
        b0 = -1.1
        phase = float(rng.uniform(0.0, 3.14))

        def _make_noisy_fn(coef: np.ndarray, ic: float, ph: float) -> Any:
            def fn(arr: np.ndarray) -> np.ndarray:
                base = _sigmoid(_standard_logit(arr, coef, ic))
                # Deterministic small tree-like piecewise wiggle (+/- 0.002)
                wiggle = 0.002 * np.sin(arr[:, 0] * 0.7 + arr[:, 3] * 0.4 + ph)
                return np.clip(base + wiggle, 0.001, 0.999)

            return fn

        _eval_calc(f"M4_clean_noisy_{i:02d}", False, "clean", "clean_noisy_tree_model", _make_noisy_fn(b_all, b0, phase), fnames, X)

    # 10 tricky clean calculators with U-shaped biomarkers (BMI, sodium, potassium) + monotone markers
    for i in range(10):
        fnames = ["age", "egfr", "albumin", "bmi", "sodium", "potassium"]
        n = 100
        X = np.column_stack([
            rng.uniform(40, 85, size=n),
            rng.uniform(25, 110, size=n),
            rng.uniform(2.8, 4.8, size=n),
            rng.uniform(17, 42, size=n),    # U-shaped optimal ~24
            rng.uniform(128, 152, size=n),  # U-shaped optimal ~140
            rng.uniform(3.0, 6.2, size=n),  # U-shaped optimal ~4.3
        ])

        def fn_ushape(arr: np.ndarray) -> np.ndarray:
            lp = (
                -1.2
                + 0.045 * (arr[:, 0] - 60.0)
                - 0.025 * (arr[:, 1] - 70.0)
                - 0.9 * (arr[:, 2] - 4.0)
                + 0.015 * ((arr[:, 3] - 24.0) ** 2) / 10.0
                + 0.02 * ((arr[:, 4] - 140.0) ** 2) / 10.0
                + 0.4 * ((arr[:, 5] - 4.3) ** 2)
            )
            return _sigmoid(lp)

        _eval_calc(f"M4_clean_ushape_{i:02d}", False, "clean", "clean_ushaped_biomarkers", fn_ushape, fnames, X)

    # 10 tricky clean Python & R scripts using abs(coef) strictly for feature-importance ranking/plotting
    clean_imp_scripts = [
        ("python", "import numpy as np\norder = np.argsort(np.abs(model.coef_[0]))[::-1]\ntop_k = [feature_names[i] for i in order[:10]]\n"),
        ("python", "import numpy as np\nfeature_importance = np.abs(lr.coef_[0])\nplt.barh(feature_names, feature_importance)\n"),
        ("python", "ranked = sorted(zip(names, coefs), key=lambda kv: abs(kv[1]), reverse=True)\n"),
        ("python", "l1_penalty = lam * np.sum(np.abs(weights))\nloss = bce_loss + l1_penalty\n"),
        ("r", "imp <- abs(coef(cox_fit))\ntop_feat <- names(sort(imp, decreasing = TRUE))[1:5]\n"),
    ]
    for i in range(10):
        lang, src = clean_imp_scripts[i % len(clean_imp_scripts)]
        _eval_code(f"M4_clean_imp_{i:02d}", False, "clean", "clean_abs_feature_importance", lang, src)

    # --- 50 Fault-Injected Positives (17 mild, 17 moderate, 16 severe) ---
    # 9 (a) Probability bounds violations (<0, >1, NaN) (3 mild, 3 moderate, 3 severe)
    bounds_specs = [("mild", 0.04)] * 3 + [("moderate", 0.25)] * 3 + [("severe", 0.70)] * 3
    for i, (inten, frac) in enumerate(bounds_specs):
        fnames, X = _make_calculator_grid(rng, n=100)
        b_all = np.array([0.6, 0.2, 0.25, -0.5, -0.4, 0.3, 0.25, 0.3])

        def _make_bad_bounds(coef: np.ndarray, idx: int, fr: float) -> Any:
            def fn(arr: np.ndarray) -> np.ndarray:
                p = _sigmoid(_standard_logit(arr, coef, -1.0)).copy()
                k = max(2, int(round(len(p) * fr)))
                if idx % 3 == 0:
                    p[:k] = 1.05 + 0.1 * (p[:k])
                elif idx % 3 == 1:
                    p[:k] = -0.05 - 0.1 * (p[:k])
                else:
                    p[:k] = np.nan
                return p

            return fn

        _eval_calc(f"M4_fault_bounds_{i:02d}", True, inten, "probability_bounds_violation", _make_bad_bounds(b_all, i, frac), fnames, X)

    # 13 (b) Strict hierarchy violation P_cause(x) > P_all_cause(x) (5 mild, 4 moderate, 4 severe)
    hier_specs = (
        [("mild", 0.04, 0.02)] * 5
        + [("moderate", 0.25, 0.08)] * 4
        + [("severe", 0.75, 0.22)] * 4
    )
    for i, (inten, shift_all, boost_cause) in enumerate(hier_specs):
        fnames, X = _make_calculator_grid(rng, n=120)
        b_all = np.array([0.55, 0.2, 0.25, -0.45, -0.4, 0.3, 0.25, 0.3])
        # Uncoupled cause-specific model with stronger weight on egfr/uacr so high-CKD patients have P_cause > P_all!
        b_cause = b_all.copy()
        b_cause[3] = -0.95
        b_cause[6] = 0.75

        def _make_hier_bug(ba: np.ndarray, bc: np.ndarray, s_all: float, b_c: float) -> dict[str, Any]:
            def fn_all(arr: np.ndarray) -> np.ndarray:
                return _sigmoid(_standard_logit(arr, ba, -1.2 - s_all))

            def fn_cause(arr: np.ndarray) -> np.ndarray:
                return np.clip(_sigmoid(_standard_logit(arr, bc, -1.35)) + b_c, 0.0, 0.999)

            return {"all_cause": fn_all, "kidney_mortality": fn_cause}

        _eval_calc(f"M4_fault_hier_{i:02d}", True, inten, "hierarchy_violation", _make_hier_bug(b_all, b_cause, shift_all, boost_cause), fnames, X, [("kidney_mortality", "all_cause")])

    # 9 (c) Categorical encoding mismatch ({0,1} vs {1,2} and polarity reversal + static) (3 mild, 3 moderate, 3 severe)
    # 4 black-box {0,1} vs {1,2} domain mismatch
    for i in range(4):
        fnames, X = _make_calculator_grid(rng, n=100)
        b_all = np.array([0.55, 0.45, 0.25, -0.45, -0.4, 0.3, 0.25, 0.3])
        inten = "mild" if i == 0 else ("moderate" if i < 3 else "severe")

        def _make_enc_shift(ba: np.ndarray) -> dict[str, Any]:
            def fn_all(arr: np.ndarray) -> np.ndarray:
                # Uses {0, 1}: male indicator (sex == 1)
                a = arr.copy()
                a[:, 1] = (arr[:, 1] == 1).astype(np.float64)
                return _sigmoid(_standard_logit(a, ba, -1.0))

            def fn_cause(arr: np.ndarray) -> np.ndarray:
                # BUG: expects {1, 2} where 2 = Male!
                a = arr.copy()
                a[:, 1] = (arr[:, 1] == 2).astype(np.float64)
                return _sigmoid(_standard_logit(a, ba, -1.8)) * 0.6

            return {"all_cause": fn_all, "cv_mortality": fn_cause}

        _eval_calc(f"M4_fault_encshift_{i:02d}", True, inten, "encoding_01_vs_12_mismatch", _make_enc_shift(b_all), fnames, X, [("cv_mortality", "all_cause")])

    # 3 black-box cross-endpoint sex polarity reversal
    for i in range(3):
        fnames, X = _make_calculator_grid(rng, n=100)
        b_all = np.array([0.55, 0.55, 0.25, -0.45, -0.4, 0.3, 0.25, 0.3])
        b_rev = b_all.copy()
        b_rev[1] = -0.55  # Reversed sex coding in cause-specific model!
        inten = "mild" if i == 0 else "moderate"

        def _make_enc_rev(ba: np.ndarray, br: np.ndarray) -> dict[str, Any]:
            def fn_all(arr: np.ndarray) -> np.ndarray:
                return _sigmoid(_standard_logit(arr, ba, -0.8))

            def fn_cause(arr: np.ndarray) -> np.ndarray:
                return _sigmoid(_standard_logit(arr, br, -2.1)) * 0.5

            return {"all_cause": fn_all, "cv_mortality": fn_cause}

        _eval_calc(f"M4_fault_encrev_{i:02d}", True, inten, "encoding_polarity_reversal", _make_enc_rev(b_all, b_rev), fnames, X, [("cv_mortality", "all_cause")])

    # 2 static code {0,1} vs {1,2} encoding contradiction
    static_enc_bugs = [
        ("mild", "python", "def all_cause(sex, age):\n    is_male = 1 if sex == 0 else 0\n    return sigmoid(0.05*age + 0.3*is_male)\ndef cv_risk(sex, age):\n    is_male = 1 if sex == 2 else 0\n    return sigmoid(0.04*age + 0.2*is_male)\n"),
        ("severe", "r", "sex_map_a <- c('female' = 0, 'male' = 1)\nsex_map_b <- c('female' = 2, 'male' = 1)\n"),
    ]
    for i, (inten, lang, src) in enumerate(static_enc_bugs):
        _eval_code(f"M4_fault_encstatic_{i:02d}", True, inten, "static_encoding_contradiction", lang, src)

    # 10 (d) Clinical monotonicity / directionality inversions (3 mild, 4 moderate, 3 severe)
    mono_specs = (
        [("mild", [3])] * 2          # single protective marker (egfr) flipped
        + [("mild", [4])] * 1        # single protective marker (albumin) flipped
        + [("moderate", [3, 4])] * 4 # both egfr & albumin flipped (protective sign loss)
        + [("severe", "abs_all")] * 3 # full abs(coef) applied!
    )
    for i, (inten, flip_spec) in enumerate(mono_specs):
        fnames, X = _make_calculator_grid(rng, n=120)
        b_true = np.array([0.6, 0.2, 0.3, -0.65, -0.55, 0.35, 0.3, 0.35])
        b_bug = np.abs(b_true) if flip_spec == "abs_all" else b_true.copy()
        if isinstance(flip_spec, list):
            for idx in flip_spec:
                b_bug[idx] = abs(b_bug[idx]) * (0.6 if inten == "mild" else 1.0)

        def _make_mono_bug(coef: np.ndarray) -> Any:
            return lambda arr: _sigmoid(_standard_logit(arr, coef, -1.0))

        _eval_calc(f"M4_fault_mono_{i:02d}", True, inten, "monotonicity_sign_inversion", _make_mono_bug(b_bug), fnames, X)

    # 9 (e) Static Python & R abs(coef) coefficient-sign destruction bugs (3 mild, 3 moderate, 3 severe)
    abs_code_bugs = [
        ("mild", "python", "import numpy as np\nweights = np.abs(cox_coef)\nscore = X @ weights\n"),
        ("mild", "python", "def calc_risk(x, beta):\n    return float(np.sum(np.abs(beta) * x))\n"),
        ("mild", "r", "w <- abs(coef(cox_model))\nlinear_pred <- as.numeric(X %*% w)\n"),
        ("moderate", "python", "import numpy as np\ncoefs = np.abs(model.coef_[0])\nlogit = intercept + np.dot(X, coefs)\nprob = 1.0 / (1.0 + np.exp(-logit))\n"),
        ("moderate", "python", "points = np.round(np.abs(log_hr) * 10)\nrisk_score = np.sum(points * patient_vec)\n"),
        ("moderate", "r", "betas <- abs(fit$coefficients)\nrisk_score <- sum(betas * patient_x)\n"),
        ("severe", "python", "def predict_mortality(x, coefficients, b0):\n    lp = b0 + sum(abs(c) * v for c, v in zip(coefficients, x))\n    return 1 / (1 + math.exp(-lp))\n"),
        ("severe", "python", "def score_patient(X, params):\n    return 1.0 / (1.0 + np.exp(-(X @ np.abs(params))))\n"),
        ("severe", "r", "lp <- X %*% abs(coefficients)\nprob <- 1 / (1 + exp(-lp))\n"),
    ]
    for i, (inten, lang, src) in enumerate(abs_code_bugs):
        _eval_code(f"M4_fault_abscoef_{i:02d}", True, inten, "static_abs_coef_destruction", lang, src)

    return records


def _summarize_module(mod_name: str, recs: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(1 for r in recs if r["ground_truth_faulty"] and r["predicted_faulty"])
    fp = sum(1 for r in recs if (not r["ground_truth_faulty"]) and r["predicted_faulty"])
    tn = sum(1 for r in recs if (not r["ground_truth_faulty"]) and (not r["predicted_faulty"]))
    fn = sum(1 for r in recs if r["ground_truth_faulty"] and (not r["predicted_faulty"]))

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2.0 * prec * sens / (prec + sens)) if (prec + sens) > 0 else 0.0

    rts = np.asarray([r["runtime_ms"] for r in recs], dtype=np.float64)
    by_intensity: dict[str, dict[str, Any]] = {}
    for inten in ("mild", "moderate", "severe"):
        sub = [r for r in recs if r["intensity"] == inten]
        detected = sum(1 for r in sub if r["predicted_faulty"])
        by_intensity[inten] = {
            "n": len(sub),
            "detected": detected,
            "sensitivity": round(detected / len(sub), 4) if sub else None,
        }

    return {
        "module": mod_name,
        "n_cases": len(recs),
        "n_clean": tn + fp,
        "n_faulty": tp + fn,
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "precision": round(prec, 4),
        "f1": round(f1, 4),
        "median_runtime_ms": round(float(np.median(rts)), 4),
        "p95_runtime_ms": round(float(np.percentile(rts, 95)), 4),
        "mean_runtime_ms": round(float(np.mean(rts)), 4),
        "by_intensity": by_intensity,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    t_all = time.perf_counter()

    m1 = run_module1_benchmark(rng)
    m2 = run_module2_benchmark(rng)
    m3 = run_module3_benchmark(rng)
    m4 = run_module4_benchmark(rng)

    all_recs = m1 + m2 + m3 + m4
    summaries = [
        _summarize_module("leakage", m1),
        _summarize_module("evalcheck", m2),
        _summarize_module("statcheck", m3),
        _summarize_module("deploycheck", m4),
    ]
    overall = _summarize_module("overall", all_recs)

    total_wall_ms = round((time.perf_counter() - t_all) * 1000.0, 2)
    out_dir = Path(__file__).resolve().parent
    json_path = out_dir / "synthetic_benchmark_results.json"
    csv_path = out_dir / "synthetic_benchmark_summary.csv"

    payload = {
        "benchmark": "clinaudit_synthetic_fault_injection_v1",
        "clinaudit_version": __version__,
        "seed": SEED,
        "total_cases": len(all_recs),
        "total_wall_ms": total_wall_ms,
        "overall": overall,
        "modules": {s["module"]: s for s in summaries},
        "errors": [
            r for r in all_recs if r["ground_truth_faulty"] != r["predicted_faulty"]
        ],
        "cases": all_recs,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_fields = [
        "module",
        "n_cases",
        "n_clean",
        "n_faulty",
        "TP",
        "FP",
        "TN",
        "FN",
        "sensitivity",
        "specificity",
        "precision",
        "f1",
        "median_runtime_ms",
        "p95_runtime_ms",
        "mean_runtime_ms",
        "mild_sensitivity",
        "moderate_sensitivity",
        "severe_sensitivity",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for s in summaries + [overall]:
            writer.writerow(
                {
                    "module": s["module"],
                    "n_cases": s["n_cases"],
                    "n_clean": s["n_clean"],
                    "n_faulty": s["n_faulty"],
                    "TP": s["TP"],
                    "FP": s["FP"],
                    "TN": s["TN"],
                    "FN": s["FN"],
                    "sensitivity": s["sensitivity"],
                    "specificity": s["specificity"],
                    "precision": s["precision"],
                    "f1": s["f1"],
                    "median_runtime_ms": s["median_runtime_ms"],
                    "p95_runtime_ms": s["p95_runtime_ms"],
                    "mean_runtime_ms": s["mean_runtime_ms"],
                    "mild_sensitivity": s["by_intensity"]["mild"]["sensitivity"],
                    "moderate_sensitivity": s["by_intensity"]["moderate"]["sensitivity"],
                    "severe_sensitivity": s["by_intensity"]["severe"]["sensitivity"],
                }
            )

    print(f"clinaudit v{__version__} | 400-Case Synthetic Benchmark (seed={SEED}) | wall={total_wall_ms:.1f} ms")
    print(f"{'Module':<12} {'N':>4} {'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3} {'Sens':>7} {'Spec':>7} {'Prec':>7} {'F1':>7} {'Med(ms)':>8} {'P95(ms)':>8}")
    print("-" * 86)
    for s in summaries + [overall]:
        print(
            f"{s['module']:<12} {s['n_cases']:>4} {s['TP']:>3} {s['FP']:>3} {s['TN']:>3} {s['FN']:>3} "
            f"{s['sensitivity']:>7.4f} {s['specificity']:>7.4f} {s['precision']:>7.4f} {s['f1']:>7.4f} "
            f"{s['median_runtime_ms']:>8.3f} {s['p95_runtime_ms']:>8.3f}"
        )
    if payload["errors"]:
        print(f"\nMisclassified cases ({len(payload['errors'])}):")
        for e in payload["errors"]:
            print(f"  - {e['case_id']} ({e['subtype']}, gt={e['ground_truth_faulty']}, pred={e['predicted_faulty']}, rules={e['rule_ids']})")
    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
