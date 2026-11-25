# `clinaudit`

**Lightweight, zero-heavy-dependency forensic auditing suite for clinical AI studies, Source Data workbooks, evaluation pipelines, and deployed clinical risk calculators.**

`clinaudit` requires **only the Python standard library and NumPy** (zero `pandas`, `scipy`, `scikit-learn`, or `openpyxl` dependencies), enabling deterministic integration into journal editorial screening, peer-review workflows, and CI/CD pipelines across Python 3.10–3.15.

---

## Core Modules

| Module | File | Target Artifacts | Key Forensic Checks |
|---|---|---|---|
| **Module 1 — Split Integrity & Mirror-Clone Detector** | `clinaudit/leakage.py` | `.xlsx`, `.csv`/`.tsv`, NumPy arrays | Exact & near-exact ($\ell_\infty \le 10^{-10}$) train–test row leakage; exact log-space binomial multiplicity-parity test for improbable $(k_{\text{train}}, k_{\text{test}})$ 1:1 mirror symmetry; conflicting-label identical profiles ($y=0$ & $y=1$); model-free confusion matrix upper/lower bounds on unseen specificity and sensitivity. |
| **Module 2 — Clinical Evaluation & Baseline Integrity Scanner** | `clinaudit/evalcheck.py` | `.py`, `.ipynb`, `.R`, DCA tables | AST/regex detection of binarized clinical risk scores (CHA₂DS₂-VASc, MELD, SOFA, Framingham, etc.) in ROC/Brier/DeLong tests; hardcoded single-threshold or tiled constant Decision Curve Analysis (DCA) + tabular `check_dca_curve()`; binary $\{0,1\}$ inputs to Brier/calibration & constant NRI; pre-split SMOTE/scaling/feature selection; immortal-time bias in static survival models; test-set checkpoint/hyperparameter selection. |
| **Module 3 — Statistical Inference & CI Sanity Auditor** | `clinaudit/statcheck.py` | Source Data `.xlsx`, `.csv`/`.tsv` | Precision-aware fixed-width placeholder CI detection; zero-width CIs ($lo = hi$) on finite-sample metrics; cross-sheet/cross-table CI width contradictions ($>5\times$ and $\sqrt{B}$ erroneous SE-of-bootstrap-mean division); impossible $K$-fold paired Wilcoxon ($P < 2^{-K}$) & bootstrap-pseudoreplication P-values; normal-approximation paired-$Z$ correlation sensitivity grid & exact McNemar/binomial helpers. |
| **Module 4 — Deployed Calculator Axiom & Monotonicity Auditor** | `clinaudit/deploycheck.py` | Black-box callables, coefficients, `.py`/`.ipynb`/`.R`/`.js` | Probability bounds $[0, 1]$ and strict hierarchy enforcement ($P_{\text{cause}}(x) \le P_{\text{all-cause}}(x)$); categorical encoding consistency probing ($\{0,1\}$ vs $\{1,2\}$ and cross-endpoint polarity reversals); ceteris-paribus clinical monotonicity verification across protective & risk markers (eGFR, albumin, age, creatinine, etc.); static `abs(coef)` coefficient-sign destruction detector (excluding feature-importance sorting/plotting). |

---

## Installation

```bash
pip install -e .
```

Or run directly from the repository root with zero build step (requires only Python $\ge 3.10$ and `numpy`).

---

## Quick Start

### Command-Line Interface

Scan a manuscript repository or supplementary package directory and export a structured JSON audit report:

```bash
python -m clinaudit scan path/to/study_artifacts --out audit_report.json
```

Use `--fail-on-critical` in automated editorial or CI pipelines to exit with status code `2` when any `CRITICAL` defect is detected.

### Programmatic Python API

```python
import numpy as np
from clinaudit import (
    audit_directory,
    audit_split_leakage,
    confusion_matrix_unseen_bounds,
    check_dca_curve,
    audit_stat_file,
    audit_risk_calculator,
)

# 1. Directory-wide audit across all 4 modules
report = audit_directory("path/to/study_artifacts")
print(report.summary)

# 2. Split leakage + model-free confusion matrix bounds
alerts, diag = audit_split_leakage(X_train, X_test, y_train=y_train, y_test=y_test, tol=1e-10)
bounds = confusion_matrix_unseen_bounds(
    tn=60, fp=40, fn=15, tp=85, n_leak_neg=50, n_leak_pos=40
)

# 3. Tabular Decision Curve Analysis sanity audit
dca_alerts, dca_diag = check_dca_curve(thresholds, net_benefits, prevalence=0.30, treat_all=treat_all)

# 4. Source Data statistical sanity audit (.xlsx / .csv)
stat_alerts, stat_diag = audit_stat_file("Source_Data.xlsx")

# 5. Deployed risk calculator axiom & monotonicity audit
calc_alerts, calc_diag = audit_risk_calculator(
    models={"all_cause": fn_all_cause, "cv_mortality": fn_cv},
    feature_names=["age", "sex", "sbp", "egfr", "albumin", "creatinine"],
    X_grid=X_cohort,
    hierarchy_pairs=[("cv_mortality", "all_cause")],
)
```

---

## 400-Case Synthetic Fault-Injection Benchmark

`benchmarks/run_synthetic_benchmark.py` executes a deterministic (`seed=2026`), self-contained 400-case evaluation (100 cases per module: 50 realistic clean negative controls including tricky edge cases + 50 fault-injected positives spanning mild, moderate, and severe intensities):

```bash
python benchmarks/run_synthetic_benchmark.py
```

| Module | Cases (Clean / Faulty) | TP | FP | TN | FN | Sensitivity | Specificity | Precision | F1 | Median (ms) | P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `leakage` | 100 (50 / 50) | 50 | 0 | 50 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.591 | 6.870 |
| `evalcheck` | 100 (50 / 50) | 50 | 0 | 50 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.258 | 1.073 |
| `statcheck` | 100 (50 / 50) | 48 | 0 | 50 | 2 | 0.9600 | 1.0000 | 1.0000 | 0.9796 | 0.268 | 0.620 |
| `deploycheck` | 100 (50 / 50) | 50 | 0 | 50 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 3.012 | 6.501 |
| **Overall** | **400 (200 / 200)** | **198** | **0** | **200** | **2** | **0.9900** | **1.0000** | **1.0000** | **0.9950** | **0.660** | **6.059** |

Stratified sensitivity across injected fault intensities: **97.1% mild** (66/68; the 2 conservative misses are boundary mild paired-$Z$ contrasts statistically indistinguishable from extreme inter-model correlation $\rho > 0.97$), **100.0% moderate** (68/68), and **100.0% severe** (64/64).

---

## Reproducing the 2026 & Late-2025 Cross-Journal Empirical Study

All unit tests, synthetic benchmarks, figure renders, and verification summaries reproduce cleanly in seconds from this repository (no third-party copyrighted PDFs or Source Data workbooks are redistributed; primary artifacts are downloaded directly from publisher DOIs with SHA-256 verification):

```bash
# 1. Run all 19 unit tests (< 1s)
python -m unittest discover -s tests -v

# 2. Run the 400-case deterministic synthetic benchmark (seed = 2026, < 1s)
python benchmarks/run_synthetic_benchmark.py

# 3. Inspect verified 2026 cohort (N0=3,380 -> N1=306; 168 parsed / 174 downloaded), 50 unflagged controls (0/50, FOR=0.0%),
#    and 4-month out-of-time late-2025 validation (Sep-Dec 2025, 45 parsed / 46 downloaded, 8.9% / 8.7% flag rate)
#    -> harvest/processed/funnel_n1_to_n3_final.json
#    -> harvest/processed/v010_adjudication_20_papers.json
#    -> harvest/external_2025/external_2025_4mo_adjudication.json
#    -> PRIMARY_SOURCE_VERIFICATION.md
```

---

## Author & License

- **Author:** Xingyu Zhang ([ORCID: 0009-0005-6181-7465](https://orcid.org/0009-0005-6181-7465))
- **License:** MIT License (see [LICENSE](LICENSE))
